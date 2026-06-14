"""CV inference — detect ingredients from a fridge photo via Gemini Vision.

A fridge image is sent to Gemini 2.5 Flash, which returns a comma-separated
list of visible ingredients. These feed the Stage 2 reranker as the user's
pantry (an OPTIONAL input path — see README "CV Inference").

Design notes:
- The Gemini client is constructed LAZILY (inside the call), not at import
  time, so importing this module never requires GEMINI_API_KEY. This keeps
  `pytest` collection green for teammates/CI without a key.
- `GEMINI_API_KEY` is read from the environment (optionally via a local
  `.env`, which is gitignored). Get a free key at aistudio.google.com.
"""

from __future__ import annotations

import functools
import os


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


def detect_ingredients_from_image(
    image_path: str,
    model: str = "gemini-2.5-flash",
    mime_type: str = "image/jpeg",
) -> list[str]:
    """Send a fridge image to Gemini Vision; return detected ingredient names.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    model : str
        Gemini model id. Default "gemini-2.5-flash" (fast + cheap, vision-capable).
    mime_type : str
        MIME type of the image. Default "image/jpeg"; pass "image/png" for PNGs.

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
    from google.genai import types

    with open(image_path, "rb") as f:
        image_data = f.read()

    client = _get_client()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_data, mime_type=mime_type),
            "List every food ingredient you can see in this fridge image. "
            "Reply with only a comma-separated list of ingredient names, "
            "nothing else. Example: eggs, milk, carrots, cheese",
        ],
    )

    raw = response.text or ""
    return [item.strip().lower() for item in raw.split(",") if item.strip()]
