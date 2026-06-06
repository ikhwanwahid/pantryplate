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
    VALID_PROFILE_STRATEGIES,
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
# Profile strategy variants
# ============================================================

class TestProfileStrategies:
    """Verify each profile_strategy produces sane vectors and differs from baseline."""

    def _make_recipes(self, n: int = 20) -> pd.DataFrame:
        return pd.DataFrame({
            "id": list(range(1000, 1000 + n)),
            "name": [f"recipe {i}" for i in range(n)],
            "ingredients_parsed": [["ing_a", "ing_b"]] * n,
            "tags_parsed": [["tag1"]] * n,
        })

    def _make_train_with_dates(self, n_users: int = 3) -> pd.DataFrame:
        """Each user has 3 positives spanning ~1 year, with mixed 4/5 star ratings."""
        rows = []
        base_date = pd.Timestamp("2024-01-01")
        for u in range(n_users):
            for offset in range(3):
                rows.append({
                    "user_id": u,
                    "recipe_id": 1000 + (u * 3) + offset,
                    "rating": 4 + (offset % 2),  # alternates 4, 5, 4
                    "date": base_date + pd.Timedelta(days=offset * 180),
                })
        return pd.DataFrame(rows)

    def test_rejects_invalid_strategy(self, tmp_path):
        with pytest.raises(ValueError, match="profile_strategy must be one of"):
            SentenceBERTRecommender(cache_dir=tmp_path, profile_strategy="bogus")

    def test_all_strategies_listed_are_valid(self):
        assert set(VALID_PROFILE_STRATEGIES) == {"mean", "rating_weighted", "recency_weighted"}

    @patch("src.models.sentence_bert.load_recipes")
    def test_each_strategy_builds_normalized_profiles(self, mock_load, tmp_path):
        recipes = self._make_recipes(20)
        mock_load.return_value = recipes
        train = self._make_train_with_dates(n_users=3)

        for strategy in VALID_PROFILE_STRATEGIES:
            with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8, seed=7)):
                model = SentenceBERTRecommender(
                    cache_dir=tmp_path, force_rebuild=True, profile_strategy=strategy
                )
                model.fit(train)
            assert len(model._user_vectors) == 3, f"{strategy}: expected 3 profiles"
            for uid, vec in model._user_vectors.items():
                assert vec.shape == (8,), f"{strategy} user {uid}: shape mismatch"
                np.testing.assert_allclose(
                    np.linalg.norm(vec), 1.0, rtol=1e-5,
                    err_msg=f"{strategy} user {uid}: not L2-normalized"
                )

    @patch("src.models.sentence_bert.load_recipes")
    def test_strategies_produce_different_profiles(self, mock_load, tmp_path):
        """Verify the strategies actually compute differently (not silently identical)."""
        recipes = self._make_recipes(20)
        mock_load.return_value = recipes
        train = self._make_train_with_dates(n_users=3)

        profiles_by_strategy = {}
        for strategy in VALID_PROFILE_STRATEGIES:
            # Use SAME seed so any difference comes from the strategy logic, not encoder randomness
            with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8, seed=7)):
                model = SentenceBERTRecommender(
                    cache_dir=tmp_path, force_rebuild=True, profile_strategy=strategy
                )
                model.fit(train)
            profiles_by_strategy[strategy] = model._user_vectors[0]

        # Mean and rating_weighted should differ (rating_weighted upweights 5★ items)
        assert not np.allclose(
            profiles_by_strategy["mean"], profiles_by_strategy["rating_weighted"], atol=1e-4
        ), "rating_weighted should differ from mean when ratings vary"

        # Mean and recency_weighted should differ (recency_weighted upweights recent items)
        assert not np.allclose(
            profiles_by_strategy["mean"], profiles_by_strategy["recency_weighted"], atol=1e-4
        ), "recency_weighted should differ from mean when dates span time"


# ============================================================
# Popularity blending
# ============================================================

class TestPopularityBlend:
    """Verify content_weight actually mixes popularity into the score."""

    def _make_recipes(self, n: int = 20) -> pd.DataFrame:
        return pd.DataFrame({
            "id": list(range(1000, 1000 + n)),
            "name": [f"recipe {i}" for i in range(n)],
            "ingredients_parsed": [["ing_a", "ing_b"]] * n,
            "tags_parsed": [["tag1"]] * n,
        })

    def _make_train_skewed(self) -> pd.DataFrame:
        """Recipe 1019 (the LAST recipe) is hugely popular; recipe 1000 has 1 rater."""
        rows = []
        # 50 users all rate recipe 1019
        for u in range(50):
            rows.append({"user_id": u, "recipe_id": 1019, "rating": 5})
        # 1 user rates recipe 1000
        rows.append({"user_id": 100, "recipe_id": 1000, "rating": 5})
        return pd.DataFrame(rows)

    def test_rejects_invalid_alpha(self, tmp_path):
        for bad in (-0.1, 1.1, 2.0):
            with pytest.raises(ValueError, match="content_weight"):
                SentenceBERTRecommender(cache_dir=tmp_path, content_weight=bad)

    @patch("src.models.sentence_bert.load_recipes")
    def test_alpha_1_matches_pure_content_baseline(self, mock_load, tmp_path):
        """alpha=1.0 should be exactly the pre-blending behavior."""
        recipes = self._make_recipes(20)
        mock_load.return_value = recipes
        train = self._make_train_skewed()

        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8, seed=11)):
            baseline = SentenceBERTRecommender(cache_dir=tmp_path, force_rebuild=True, content_weight=1.0)
            baseline.fit(train)
        # Pick a user with a profile
        recs_alpha_1 = baseline.recommend(0, k=5, exclude_seen=False)
        # And manually compute what pure-cosine should give
        scores = baseline._recipe_matrix @ baseline._user_vectors[0]
        expected = [int(baseline.recipe_ids[i]) for i in np.argsort(-scores)[:5]]
        assert recs_alpha_1 == expected

    @patch("src.models.sentence_bert.load_recipes")
    def test_alpha_0_returns_popularity_ranking_top1(self, mock_load, tmp_path):
        """alpha=0.0 means score=(1-0)*pop, so top-1 should be the most-popular recipe."""
        recipes = self._make_recipes(20)
        mock_load.return_value = recipes
        train = self._make_train_skewed()

        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8, seed=11)):
            model = SentenceBERTRecommender(cache_dir=tmp_path, force_rebuild=True, content_weight=0.0)
            model.fit(train)
        # User 100 has a profile (they rated 1000). With alpha=0, popularity dominates.
        # Most popular recipe is 1019.
        recs = model.recommend(user_id=100, k=3, exclude_seen=False)
        assert recs[0] == 1019, f"alpha=0 should rank most-popular first; got {recs}"

    @patch("src.models.sentence_bert.load_recipes")
    def test_alpha_0_and_alpha_1_produce_different_rankings(self, mock_load, tmp_path):
        """Sanity: alpha=0 (pure pop) and alpha=1 (pure content) differ when both have signal."""
        recipes = self._make_recipes(20)
        mock_load.return_value = recipes
        train = self._make_train_skewed()

        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8, seed=11)):
            m_pop = SentenceBERTRecommender(cache_dir=tmp_path, force_rebuild=True, content_weight=0.0)
            m_pop.fit(train)
        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8, seed=11)):
            m_content = SentenceBERTRecommender(cache_dir=tmp_path, force_rebuild=True, content_weight=1.0)
            m_content.fit(train)

        recs_pop = m_pop.recommend(100, k=10, exclude_seen=False)
        recs_content = m_content.recommend(100, k=10, exclude_seen=False)
        assert recs_pop != recs_content, "alpha=0 and alpha=1 should produce different rankings"

    @patch("src.models.sentence_bert.load_recipes")
    def test_popularity_score_is_zero_for_cold_recipes(self, mock_load, tmp_path):
        """Recipes with no train ratings should have popularity_score = 0."""
        recipes = self._make_recipes(20)
        mock_load.return_value = recipes
        train = self._make_train_skewed()  # only recipes 1019 and 1000 are rated

        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8, seed=11)):
            model = SentenceBERTRecommender(cache_dir=tmp_path, force_rebuild=True, content_weight=0.5)
            model.fit(train)

        # Recipes 1001..1018 have no train ratings → pop score = 0
        for rid in range(1001, 1019):
            row = model._recipe_id_to_row[rid]
            assert model._popularity_score[row] == 0.0, f"recipe {rid} should have pop_score=0"
        # Recipes 1019 (most popular) should have score > 0
        row_1019 = model._recipe_id_to_row[1019]
        assert model._popularity_score[row_1019] > 0


# ============================================================
# Layer 4 — SBERT + Tag SVD concat
# ============================================================

class TestTagFeatureConcat:
    """Verify tag_feature_weight concatenates Tag SVD features onto SBERT."""

    def _make_recipes(self, n: int = 20) -> pd.DataFrame:
        return pd.DataFrame({
            "id": list(range(1000, 1000 + n)),
            "name": [f"recipe {i}" for i in range(n)],
            "ingredients_parsed": [["ing_a", "ing_b"]] * n,
            "tags_parsed": [["tag1"]] * n,
        })

    def _make_train(self, n_users: int = 3) -> pd.DataFrame:
        rows = []
        for u in range(n_users):
            for offset in range(3):
                rows.append({"user_id": u, "recipe_id": 1000 + (u * 3) + offset, "rating": 5})
        return pd.DataFrame(rows)

    def _fake_tag_features(self, recipe_ids, dim: int = 4, seed: int = 13):
        """Returns a DataFrame indexed by recipe_id, mimicking build_recipe_feature_matrix()."""
        rng = np.random.default_rng(seed)
        data = rng.standard_normal((len(recipe_ids), dim)).astype(np.float32)
        return pd.DataFrame(
            data,
            columns=[f"feat_{i}" for i in range(dim)],
            index=pd.Index(recipe_ids, name="recipe_id"),
        )

    def test_rejects_invalid_weight(self, tmp_path):
        for bad in (-0.1, 1.5, 2.0):
            with pytest.raises(ValueError, match="tag_feature_weight"):
                SentenceBERTRecommender(cache_dir=tmp_path, tag_feature_weight=bad)

    @patch("src.models.sentence_bert.load_recipes")
    def test_weight_0_does_not_change_matrix(self, mock_load, tmp_path):
        """tag_feature_weight=0 should leave the SBERT matrix untouched."""
        recipes = self._make_recipes(20)
        mock_load.return_value = recipes
        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8)):
            model = SentenceBERTRecommender(
                cache_dir=tmp_path, force_rebuild=True, tag_feature_weight=0.0
            )
            model.fit(self._make_train(3))
        assert model._recipe_matrix.shape == (20, 8), \
            "tag_feature_weight=0 should not concat tag features"

    @patch("src.data.features.build_recipe_feature_matrix")
    @patch("src.models.sentence_bert.load_recipes")
    def test_weight_above_0_concatenates(self, mock_load, mock_features, tmp_path):
        """tag_feature_weight>0 should widen the recipe matrix by the tag dim."""
        recipes = self._make_recipes(20)
        mock_load.return_value = recipes
        mock_features.return_value = self._fake_tag_features(list(range(1000, 1020)), dim=4)

        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8)):
            model = SentenceBERTRecommender(
                cache_dir=tmp_path, force_rebuild=True, tag_feature_weight=0.5
            )
            model.fit(self._make_train(3))

        # 8-dim SBERT + 4-dim tag = 12-dim combined
        assert model._recipe_matrix.shape == (20, 12)
        # All rows L2-normalized
        norms = np.linalg.norm(model._recipe_matrix, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    @patch("src.data.features.build_recipe_feature_matrix")
    @patch("src.models.sentence_bert.load_recipes")
    def test_recipe_id_alignment(self, mock_load, mock_features, tmp_path):
        """If Tag SVD comes in a different order, reindex should still align it correctly."""
        recipes = self._make_recipes(20)  # ids 1000..1019 in order
        mock_load.return_value = recipes
        # Shuffle the tag feature order — reindex must put them back in recipe_ids order
        shuffled_ids = list(range(1000, 1020))[::-1]  # reversed
        mock_features.return_value = self._fake_tag_features(shuffled_ids, dim=4)

        with patch("sentence_transformers.SentenceTransformer", _fake_encoder_class(dim=8)):
            model = SentenceBERTRecommender(
                cache_dir=tmp_path, force_rebuild=True, tag_feature_weight=0.5
            )
            model.fit(self._make_train(3))

        # Just verify it doesn't crash and produces the expected shape
        assert model._recipe_matrix.shape == (20, 12)


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
