"""Tests for the CV inference pipeline (src/vision/).

Two tiers:
- Offline tests (always run): the ingredient normalizer + the import-safety
  contract. No network, no API key.
- Live test (opt-in): the real Gemini Vision call. Skipped unless
  GEMINI_API_KEY is set AND a sample image exists; also marked `slow` so it
  stays out of the default `uv run pytest tests/` run (it makes a billed
  network call).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.vision.ingredient_normalizer import normalize

TEST_IMAGE = Path("data/test_fridge.jpg")


# ============================================================
# Offline — ingredient normalizer
# ============================================================

class TestNormalize:
    def test_maps_known_labels(self):
        assert normalize(["bell_pepper"]) == ["green pepper"]
        assert normalize(["spring_onion"]) == ["green onions"]
        assert normalize(["egg"]) == ["eggs"]

    def test_passes_through_unknown_labels(self):
        assert normalize(["chicken", "rice"]) == ["chicken", "rice"]

    def test_mixed_known_and_unknown(self):
        assert normalize(["egg", "chicken", "bell_pepper"]) == [
            "eggs", "chicken", "green pepper",
        ]

    def test_empty_list(self):
        assert normalize([]) == []


# ============================================================
# Offline — import safety contract
# ============================================================

def test_importing_cv_inference_does_not_require_api_key(monkeypatch):
    """Importing the module must NOT construct the client (no key needed).

    Regression guard: a module-level client meant `pytest` collection broke
    for anyone without GEMINI_API_KEY. The client is now lazy.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import importlib

    import src.vision.cv_inference as cv

    importlib.reload(cv)  # re-exec module body with the key removed
    assert hasattr(cv, "detect_ingredients_from_image")


def test_detect_raises_clear_error_without_key(monkeypatch):
    """Calling the detector without a key should raise a clear RuntimeError,
    not an opaque SDK error — and only at call time, not import time."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from src.vision.cv_inference import _get_client

    _get_client.cache_clear()  # drop any cached client from a prior test
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        _get_client()


# ============================================================
# Live — real Gemini Vision call (opt-in)
# ============================================================

@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") or not TEST_IMAGE.exists(),
    reason="needs GEMINI_API_KEY and data/test_fridge.jpg (billed network call)",
)
def test_detect_ingredients_live():
    from src.vision.cv_inference import detect_ingredients_from_image

    result = detect_ingredients_from_image(str(TEST_IMAGE))
    print("Detected ingredients:", result)
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)
