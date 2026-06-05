"""Tests for src/models/sentence_bert.py.

Tiered:
- Unit tests use tiny in-memory data and a 1-recipe encoder smoke (fast).
- Integration tests use the real catalogue + a small sample of train interactions.
  They skip if data files aren't present.

The full warm/cold harness evaluation isn't run inside pytest — it's exercised
in a separate scratch script (or notebook) once the cache is warm.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.sentence_bert import (
    DEFAULT_MODEL,
    SentenceBERTRecommender,
    _build_recipe_text,
    _cache_path_for_model,
)


RECIPES_CSV = Path("data/raw/RAW_recipes.csv")
TRAIN_CSV = Path("data/raw/interactions_train.csv")


# ============================================================
# Recipe text builder
# ============================================================

class TestBuildRecipeText:
    def test_uses_parsed_lists_when_present(self):
        row = pd.Series({
            "name": "spicy tofu stir-fry",
            "ingredients_parsed": ["tofu", "soy sauce", "ginger"],
            "tags_parsed": ["vegan", "asian", "30-minutes-or-less"],
        })
        text = _build_recipe_text(row)
        assert text == "spicy tofu stir-fry | tofu, soy sauce, ginger | vegan, asian, 30-minutes-or-less"

    def test_falls_back_to_raw_strings(self):
        row = pd.Series({
            "name": "test recipe",
            "ingredients": "['chicken', 'salt']",
            "tags": "['easy']",
        })
        text = _build_recipe_text(row)
        # Brackets/quotes stripped, separators kept
        assert "test recipe" in text
        assert "chicken" in text and "salt" in text
        assert "easy" in text
        assert "[" not in text and "'" not in text

    def test_handles_missing_fields(self):
        row = pd.Series({"name": "minimal"})
        text = _build_recipe_text(row)
        # Should not crash; just gives a name with empty other sections
        assert "minimal" in text


# ============================================================
# Cache path helper
# ============================================================

class TestCachePathForModel:
    def test_replaces_slash(self):
        path = _cache_path_for_model("sentence-transformers/all-MiniLM-L6-v2", Path("/tmp"))
        assert "/" not in path.name
        assert path.name.endswith(".npy")
        assert "all-MiniLM-L6-v2" in path.name


# ============================================================
# fit + recommend (mocked encoder, no real model load)
# ============================================================

def _fake_encoder_class(dim: int = 8, seed: int = 0):
    """Returns a class-mock — calling it like `SentenceTransformer(name, device=...)`
    yields an instance whose `.encode(texts, ...)` returns a real numpy array."""
    rng = np.random.default_rng(seed)

    def fake_encode(texts, **kwargs):
        n = len(texts)
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs

    encoder_instance = MagicMock()
    encoder_instance.encode = fake_encode
    return MagicMock(return_value=encoder_instance)


class TestFitAndRecommendMocked:
    """Exercise the full fit/recommend flow with a fake encoder."""

    def _make_recipes(self, n: int = 20) -> pd.DataFrame:
        return pd.DataFrame({
            "id": list(range(1000, 1000 + n)),
            "name": [f"recipe {i}" for i in range(n)],
            "ingredients_parsed": [["ing_a", "ing_b"]] * n,
            "tags_parsed": [["tag1"]] * n,
        })

    def _make_train(self, n_users: int = 5, recipe_start: int = 1000) -> pd.DataFrame:
        """Each user rates 3 recipes, all with rating 5."""
        rows = []
        for u in range(n_users):
            for offset in range(3):
                rows.append({"user_id": u, "recipe_id": recipe_start + (u * 3) + offset, "rating": 5})
        return pd.DataFrame(rows)

    @patch("src.models.sentence_bert.load_recipes")
    def test_fit_builds_matrix_and_profiles(self, mock_load, tmp_path):
        recipes = self._make_recipes(20)
        mock_load.return_value = recipes

        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8)):
            model = SentenceBERTRecommender(cache_dir=tmp_path, force_rebuild=True)
            train = self._make_train(n_users=5)
            model.fit(train)

        assert model._recipe_matrix.shape == (20, 8)
        # All recipe vectors are L2-normalized
        norms = np.linalg.norm(model._recipe_matrix, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
        # Every user has a profile
        assert len(model._user_vectors) == 5
        # Each user vector is L2-normalized
        for vec in model._user_vectors.values():
            assert vec.shape == (8,)
            np.testing.assert_allclose(np.linalg.norm(vec), 1.0, rtol=1e-5)

    @patch("src.models.sentence_bert.load_recipes")
    def test_recommend_returns_k_ints(self, mock_load, tmp_path):
        recipes = self._make_recipes(20)
        mock_load.return_value = recipes
        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8)):
            model = SentenceBERTRecommender(cache_dir=tmp_path, force_rebuild=True)
            train = self._make_train(n_users=3)
            model.fit(train)

        # Sanity: encoder ran and produced a real matrix (not a MagicMock pass-through)
        assert model._recipe_matrix.shape == (20, 8)

        recs = model.recommend(user_id=0, k=5, exclude_seen=False)
        assert isinstance(recs, list)
        assert len(recs) == 5
        assert all(isinstance(r, int) for r in recs)
        assert set(recs).issubset(set(recipes["id"].astype(int)))

    @patch("src.models.sentence_bert.load_recipes")
    def test_exclude_seen_actually_excludes(self, mock_load, tmp_path):
        recipes = self._make_recipes(20)
        mock_load.return_value = recipes
        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8)):
            model = SentenceBERTRecommender(cache_dir=tmp_path, force_rebuild=True)
            train = self._make_train(n_users=3)
            model.fit(train)

        seen_by_user_0 = train.loc[train["user_id"] == 0, "recipe_id"].tolist()
        recs = model.recommend(user_id=0, k=15, exclude_seen=True)
        assert not (set(recs) & set(seen_by_user_0))

    @patch("src.models.sentence_bert.load_recipes")
    def test_cold_user_falls_back_to_popularity(self, mock_load, tmp_path):
        recipes = self._make_recipes(20)
        mock_load.return_value = recipes
        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8)):
            model = SentenceBERTRecommender(cache_dir=tmp_path, force_rebuild=True)
            train = self._make_train(n_users=3)
            model.fit(train)

        # user_id 99 has no train interactions
        recs = model.recommend(user_id=99, k=5, exclude_seen=True)
        assert len(recs) == 5
        # Top-1 should be the most-rated recipe in train
        top_in_train = train.groupby("recipe_id")["user_id"].nunique().idxmax()
        assert recs[0] == top_in_train

    @patch("src.models.sentence_bert.load_recipes")
    def test_cache_is_reused_on_second_fit(self, mock_load, tmp_path):
        recipes = self._make_recipes(20)
        mock_load.return_value = recipes

        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8, seed=1)) as st_mock:
            model1 = SentenceBERTRecommender(cache_dir=tmp_path, force_rebuild=False)
            model1.fit(self._make_train(n_users=2))
            assert st_mock.call_count == 1  # encoder ran on first fit

        # Second fit — cache hit means SentenceTransformer never gets called
        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8, seed=999)) as st_mock_2:
            model2 = SentenceBERTRecommender(cache_dir=tmp_path, force_rebuild=False)
            model2.fit(self._make_train(n_users=2))
            assert st_mock_2.call_count == 0

        # Matrices match (cache was the source of truth)
        np.testing.assert_allclose(model1._recipe_matrix, model2._recipe_matrix)

    def test_recommend_before_fit_raises(self, tmp_path):
        model = SentenceBERTRecommender(cache_dir=tmp_path)
        with pytest.raises(RuntimeError, match="Call .fit"):
            model.recommend(user_id=1, k=5)


# ============================================================
# Integration tests (real catalogue + real encoder)
# Skipped if data files are missing OR if running fast tests.
# ============================================================

@pytest.mark.skipif(
    not RECIPES_CSV.exists() or not TRAIN_CSV.exists(),
    reason="raw data files not present",
)
class TestIntegrationSmoke:
    """One small end-to-end run with the real encoder on a tiny sample."""

    @pytest.mark.slow
    def test_tiny_fit_with_real_encoder(self, tmp_path):
        """Encode 50 recipes, fit on a sample of train, get recs.

        Marked `slow` because it loads the real model (~5s first time).
        Run with: uv run pytest tests/test_sentence_bert.py -m slow
        """
        from src.data.loader import load_train_interactions, load_recipes

        # Sample a tiny subset of recipes that we'll mock load_recipes to return
        recipes_sample = load_recipes().sample(50, random_state=42).copy()
        train = load_train_interactions().head(500)
        # Only keep train rows for recipes in our sample
        train = train[train["recipe_id"].isin(recipes_sample["id"].astype(int))]
        if len(train) < 5:
            pytest.skip("not enough overlapping train rows in the tiny sample")

        with patch("src.models.sentence_bert.load_recipes", return_value=recipes_sample):
            model = SentenceBERTRecommender(
                cache_dir=tmp_path,
                batch_size=32,
                force_rebuild=True,
            )
            model.fit(train)

        # Sanity: matrix shape and a recommend call
        assert model._recipe_matrix.shape == (50, 384)  # MiniLM-L6-v2 dim
        sample_uid = int(train["user_id"].iloc[0])
        recs = model.recommend(sample_uid, k=10, exclude_seen=True)
        assert len(recs) == 10
        assert all(r in recipes_sample["id"].astype(int).values for r in recs)
