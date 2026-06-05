"""Tests for src/models/tag_svd_content.py.

Mirrors the structure of test_sentence_bert.py since the two models share
the same .fit/.recommend contract and profile/blend logic. Uses a
feature_provider override to inject tiny matrices instead of loading the
real cached parquet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.tag_svd_content import (
    VALID_PROFILE_STRATEGIES,
    TagSVDRecommender,
)


RECIPES_CSV = Path("data/raw/RAW_recipes.csv")
TRAIN_CSV = Path("data/raw/interactions_train.csv")
FEATURE_CACHE = Path("data/processed/recipe_features.parquet")


# ============================================================
# Helpers
# ============================================================

def _fake_feature_matrix(n: int = 20, dim: int = 6, seed: int = 0) -> pd.DataFrame:
    """Return a (n, dim) DataFrame indexed by recipe_id, like build_recipe_feature_matrix()."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, dim)).astype(np.float32)
    cols = [f"feat_{i}" for i in range(dim)]
    df = pd.DataFrame(data, columns=cols, index=pd.Index(range(1000, 1000 + n), name="recipe_id"))
    return df


def _make_train(n_users: int = 5, recipe_start: int = 1000) -> pd.DataFrame:
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for u in range(n_users):
        for offset in range(3):
            rows.append({
                "user_id": u,
                "recipe_id": recipe_start + (u * 3) + offset,
                "rating": 4 + (offset % 2),  # alternates 4, 5
                "date": base_date + pd.Timedelta(days=offset * 90),
            })
    return pd.DataFrame(rows)


# ============================================================
# Basic API + fit
# ============================================================

class TestFitAndRecommend:
    def test_fit_returns_self(self):
        features = lambda: _fake_feature_matrix(20)
        model = TagSVDRecommender(feature_provider=features)
        result = model.fit(_make_train(3))
        assert result is model

    def test_fit_populates_state(self):
        features = lambda: _fake_feature_matrix(20)
        model = TagSVDRecommender(feature_provider=features)
        model.fit(_make_train(3))

        assert model._recipe_matrix.shape == (20, 6)
        assert model.recipe_ids.shape == (20,)
        # All recipe vectors L2-normalized
        norms = np.linalg.norm(model._recipe_matrix, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
        # User profiles built and L2-normed
        assert len(model._user_vectors) == 3
        for vec in model._user_vectors.values():
            np.testing.assert_allclose(np.linalg.norm(vec), 1.0, rtol=1e-5)

    def test_recommend_returns_k_ints(self):
        features = lambda: _fake_feature_matrix(20)
        model = TagSVDRecommender(feature_provider=features)
        model.fit(_make_train(3))

        recs = model.recommend(user_id=0, k=5, exclude_seen=False)
        assert isinstance(recs, list)
        assert len(recs) == 5
        assert all(isinstance(r, int) for r in recs)

    def test_recommend_before_fit_raises(self):
        model = TagSVDRecommender(feature_provider=lambda: _fake_feature_matrix(20))
        with pytest.raises(RuntimeError, match="Call .fit"):
            model.recommend(user_id=1, k=5)

    def test_exclude_seen_excludes(self):
        features = lambda: _fake_feature_matrix(20)
        model = TagSVDRecommender(feature_provider=features)
        train = _make_train(3)
        model.fit(train)

        seen = train.loc[train["user_id"] == 0, "recipe_id"].tolist()
        recs = model.recommend(0, k=15, exclude_seen=True)
        assert not (set(recs) & set(seen))

    def test_cold_user_falls_back_to_popularity(self):
        features = lambda: _fake_feature_matrix(20)
        model = TagSVDRecommender(feature_provider=features)
        train = _make_train(3)
        model.fit(train)

        # user_id 99 has no train interactions
        recs = model.recommend(user_id=99, k=5, exclude_seen=True)
        assert len(recs) == 5
        top_in_train = train.groupby("recipe_id")["user_id"].nunique().idxmax()
        assert recs[0] == top_in_train


# ============================================================
# Profile strategy variants
# ============================================================

class TestProfileStrategies:
    def test_rejects_invalid_strategy(self):
        with pytest.raises(ValueError, match="profile_strategy must be one of"):
            TagSVDRecommender(profile_strategy="bogus")

    def test_all_strategies_listed_are_valid(self):
        assert set(VALID_PROFILE_STRATEGIES) == {"mean", "rating_weighted", "recency_weighted"}

    def test_each_strategy_builds_normalized_profiles(self):
        features = lambda: _fake_feature_matrix(20, seed=7)
        train = _make_train(3)
        for strategy in VALID_PROFILE_STRATEGIES:
            model = TagSVDRecommender(profile_strategy=strategy, feature_provider=features)
            model.fit(train)
            assert len(model._user_vectors) == 3
            for vec in model._user_vectors.values():
                np.testing.assert_allclose(np.linalg.norm(vec), 1.0, rtol=1e-5,
                                           err_msg=f"{strategy}: not L2-normalized")

    def test_strategies_produce_different_profiles(self):
        features = lambda: _fake_feature_matrix(20, seed=7)
        train = _make_train(3)
        profiles_by_strategy = {}
        for strategy in VALID_PROFILE_STRATEGIES:
            model = TagSVDRecommender(profile_strategy=strategy, feature_provider=features)
            model.fit(train)
            profiles_by_strategy[strategy] = model._user_vectors[0]
        assert not np.allclose(profiles_by_strategy["mean"],
                               profiles_by_strategy["rating_weighted"], atol=1e-4)
        assert not np.allclose(profiles_by_strategy["mean"],
                               profiles_by_strategy["recency_weighted"], atol=1e-4)


# ============================================================
# Popularity blending
# ============================================================

class TestPopularityBlend:
    def _train_skewed(self) -> pd.DataFrame:
        """Recipe 1019 is hugely popular; recipe 1000 has 1 rater."""
        rows = []
        for u in range(50):
            rows.append({"user_id": u, "recipe_id": 1019, "rating": 5})
        rows.append({"user_id": 100, "recipe_id": 1000, "rating": 5})
        return pd.DataFrame(rows)

    def test_rejects_invalid_weight(self):
        for bad in (-0.1, 1.1, 2.0):
            with pytest.raises(ValueError, match="content_weight"):
                TagSVDRecommender(content_weight=bad)

    def test_alpha_0_returns_popularity_top1(self):
        features = lambda: _fake_feature_matrix(20, seed=11)
        train = self._train_skewed()
        model = TagSVDRecommender(content_weight=0.0, feature_provider=features)
        model.fit(train)
        recs = model.recommend(user_id=100, k=3, exclude_seen=False)
        assert recs[0] == 1019

    def test_alpha_0_and_alpha_1_differ(self):
        features = lambda: _fake_feature_matrix(20, seed=11)
        train = self._train_skewed()
        m_pop = TagSVDRecommender(content_weight=0.0, feature_provider=features)
        m_pop.fit(train)
        m_content = TagSVDRecommender(content_weight=1.0, feature_provider=features)
        m_content.fit(train)
        recs_pop = m_pop.recommend(100, k=10, exclude_seen=False)
        recs_content = m_content.recommend(100, k=10, exclude_seen=False)
        assert recs_pop != recs_content

    def test_pop_score_zero_for_cold_recipes(self):
        features = lambda: _fake_feature_matrix(20, seed=11)
        train = self._train_skewed()  # only 1019 and 1000 are rated
        model = TagSVDRecommender(content_weight=0.5, feature_provider=features)
        model.fit(train)
        for rid in range(1001, 1019):
            row = model._recipe_id_to_row[rid]
            assert model._popularity_score[row] == 0.0
        row_1019 = model._recipe_id_to_row[1019]
        assert model._popularity_score[row_1019] > 0


# ============================================================
# Integration test (real cached feature matrix)
# ============================================================

@pytest.mark.skipif(
    not RECIPES_CSV.exists() or not TRAIN_CSV.exists() or not FEATURE_CACHE.exists(),
    reason="raw data or feature cache not present",
)
class TestIntegrationOnRealData:
    def test_fit_on_real_train_sample(self):
        """Smoke test: fit on a slice of real train + the real feature matrix."""
        from src.data.loader import load_train_interactions

        train = load_train_interactions().head(2000)
        model = TagSVDRecommender()
        model.fit(train)

        # Real feature matrix is (231637, 107)
        assert model._recipe_matrix.shape == (231637, 107)
        # Some users should have a profile (the head(2000) has multiple positives)
        assert len(model._user_vectors) > 0
        # Sanity-check a recommend call
        sample_uid = int(train["user_id"].iloc[0])
        recs = model.recommend(sample_uid, k=10, exclude_seen=True)
        assert len(recs) == 10
