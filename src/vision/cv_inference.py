"""CV inference — detect ingredients from a fridge photo via Gemini Vision.

A fridge image is sent to a Gemini Vision model, which returns a comma-separated
list of visible ingredients. These feed the Stage 2 reranker as the user's
pantry (an OPTIONAL input path — see README "CV Inference").

Design notes:
- The Gemini client is constructed LAZILY (inside the call), not at import
  time, so importing this module never requires GEMINI_API_KEY. This keeps
  `pytest` collection green for teammates/CI without a key.
- `GEMINI_API_KEY` is read from the environment (optionally via a local
  `.env`, which is gitignored). Get a free key at aistudio.google.com.
- Model selection is resilient: we try `model` first and, if that id isn't
  available on the key (e.g. a newer preview model that hasn't shipped),
  transparently fall back to `fallback_model`. The resolved model is cached
  for the process so we pay the fallback round-trip at most once.
"""

from __future__ import annotations

import functools
import os
import warnings


@functools.lru_cache(maxsize=1)
def _get_client():
    """Construct (once) and return the Gemini client.

    Raises a clear error if the API key isn't configured, rather than the
    SDK's lower-level message. Cached so repeated calls reuse one client.
    """
    from dotenv import load_dotenv

    load_dotenv()  # no-op if there's no .env; pulls GEMINI_API_KEY into env
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project "
            "root or export it in your shell. Get a free key at "
            "aistudio.google.com."
        )
    from google import genai

    return genai.Client(api_key=api_key)


# Cache the model id we actually got a response from, so a missing primary
# model only costs one failed round-trip per process, not one per detection.
_RESOLVED_MODEL: str | None = None


def _is_model_unavailable(exc: Exception) -> bool:
    """True if `exc` looks like 'this model id doesn't exist / isn't available'.

    We only fall back for availability errors — NOT for auth, quota, or
    network failures, which should surface to the caller unchanged.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (403, 404):
        return True
    msg = str(exc).lower()
    needles = ("not_found", "not found", "does not exist", "is not supported",
               "not supported", "permission_denied", "unavailable")
    return any(n in msg for n in needles)


def detect_ingredients_from_image(
    image_path: str,
    model: str = "gemini-3.1-flash-lite",
    mime_type: str = "image/jpeg",
    fallback_model: str = "gemini-2.5-flash",
) -> list[str]:
    """Send a fridge image to Gemini Vision; return detected ingredient names.

    Tries `model` first; if that id isn't available on the key, transparently
    retries once with `fallback_model` (and remembers the working id for the
    rest of the process).

    Parameters
    ----------
    image_path : str
        Path to the image file.
    model : str
        Preferred Gemini model id. Default "gemini-3.1-flash-lite".
    mime_type : str
        MIME type of the image. Default "image/jpeg"; pass "image/png" for PNGs.
    fallback_model : str
        Known-good model used if `model` is unavailable. Default
        "gemini-2.5-flash" (fast + cheap, vision-capable). Pass None/"" to
        disable fallback and let an unavailable `model` raise.

    Returns
    -------
    list[str]
        Lower-cased, stripped ingredient labels. Empty list if the model
        returns nothing parseable.

    Raises
    ------
    RuntimeError
        If GEMINI_API_KEY is not configured.
    FileNotFoundError
        If image_path does not exist.
    """
    global _RESOLVED_MODEL
    from google.genai import types

    with open(image_path, "rb") as f:
        image_data = f.read()

    client = _get_client()
    contents = [
        types.Part.from_bytes(data=image_data, mime_type=mime_type),
        "List every food ingredient you can see in this fridge image. "
        "Reply with only a comma-separated list of ingredient names, "
        "nothing else. Example: eggs, milk, carrots, cheese",
    ]

    # If we've already discovered the working model this session, use it.
    primary = _RESOLVED_MODEL or model
    try:
        response = client.models.generate_content(model=primary, contents=contents)
        _RESOLVED_MODEL = primary
    except Exception as exc:
        if fallback_model and fallback_model != primary and _is_model_unavailable(exc):
            warnings.warn(
                f"Gemini model {primary!r} is unavailable ({exc}); "
                f"falling back to {fallback_model!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
            response = client.models.generate_content(
                model=fallback_model, contents=contents
            )
            _RESOLVED_MODEL = fallback_model
        else:
            raise

    raw = response.text or ""
    return [item.strip().lower() for item in raw.split(",") if item.strip()]
