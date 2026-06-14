"""Tests for src/models/hybrid_linear.py.

Uses a tiny synthetic interaction frame + a fake feature provider so no
real data download is required. Mirrors the structure of test_cf_models.py
and test_tag_svd_content.py.

    uv run pytest tests/test_hybrid_linear.py -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.hybrid_linear import HybridLinearRecommender, _rank_normalize


# ============================================================
# Helpers
# ============================================================

def _fake_feature_matrix(recipe_ids, dim: int = 8, seed: int = 0) -> pd.DataFrame:
    """Return a (n_recipes, dim) DataFrame indexed by recipe_id."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((len(recipe_ids), dim)).astype(np.float32)
    cols = [f"feat_{i}" for i in range(dim)]
    return pd.DataFrame(
        data,
        columns=cols,
        index=pd.Index(list(recipe_ids), name="recipe_id"),
    )


@pytest.fixture
def tiny_train() -> pd.DataFrame:
    """30 users × 40 recipes of synthetic interactions."""
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2018-01-01")
    n_users, n_items = 30, 40
    rows = []
    for u in range(n_users):
        n = int(rng.integers(5, 13))
        for it in rng.choice(n_items, size=n, replace=False):
            rows.append({
                "user_id": 1000 + u,
                "recipe_id": 5000 + int(it),
                "date": base + pd.Timedelta(days=int(rng.integers(0, 400))),
                "rating": int(rng.integers(3, 6)),
            })
    df = pd.DataFrame(rows)
    df["u"] = df["user_id"].astype("category").cat.codes
    df["i"] = df["recipe_id"].astype("category").cat.codes
    return df


@pytest.fixture
def fake_provider(tiny_train):
    recipe_ids = sorted(tiny_train["recipe_id"].unique())
    matrix = _fake_feature_matrix(recipe_ids)
    return lambda: matrix


def _make_model(tiny_train, fake_provider, alpha: float = 0.5) -> HybridLinearRecommender:
    return HybridLinearRecommender(
        alpha=alpha,
        lambda_reg=10.0,
        min_item_ratings=1,
        feature_provider=fake_provider,
    ).fit(tiny_train)


# ============================================================
# _rank_normalize unit tests
# ============================================================

class TestRankNormalize:
    def test_best_score_maps_to_one(self):
        scores = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        norm = _rank_normalize(scores)
        assert norm[0] == pytest.approx(1.0)

    def test_worst_score_maps_to_zero(self):
        scores = np.array([3.0, 1.0, 2.0], dtype=np.float32)
        norm = _rank_normalize(scores)
        assert norm[1] == pytest.approx(0.0)

    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(7)
        scores = rng.standard_normal(100).astype(np.float32)
        norm = _rank_normalize(scores)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_single_element(self):
        norm = _rank_normalize(np.array([42.0], dtype=np.float32))
        assert norm[0] == pytest.approx(1.0)


# ============================================================
# Stage 1 contract tests
# ============================================================

class TestContract:
    def test_fit_returns_self(self, tiny_train, fake_provider):
        m = HybridLinearRecommender(
            alpha=0.5, lambda_reg=10.0, min_item_ratings=1,
            feature_provider=fake_provider,
        )
        assert m.fit(tiny_train) is m

    def test_recommend_returns_list_of_ints(self, tiny_train, fake_provider):
        m = _make_model(tiny_train, fake_provider)
        recs = m.recommend(1000, k=10)
        assert isinstance(recs, list)
        assert all(isinstance(r, int) for r in recs)

    def test_fills_to_k(self, tiny_train, fake_provider):
        m = _make_model(tiny_train, fake_provider)
        assert len(m.recommend(1000, k=10)) == 10

    def test_no_duplicate_recipes(self, tiny_train, fake_provider):
        m = _make_model(tiny_train, fake_provider)
        recs = m.recommend(1000, k=15)
        assert len(recs) == len(set(recs))

    def test_exclude_seen(self, tiny_train, fake_provider):
        m = _make_model(tiny_train, fake_provider)
        uid = 1000
        seen = set(tiny_train.loc[tiny_train["user_id"] == uid, "recipe_id"])
        recs = m.recommend(uid, k=15, exclude_seen=True)
        assert seen.isdisjoint(recs)

    def test_exclude_seen_false_may_include_seen(self, tiny_train, fake_provider):
        m = _make_model(tiny_train, fake_provider)
        uid = 1000
        seen = set(tiny_train.loc[tiny_train["user_id"] == uid, "recipe_id"])
        recs_with = set(m.recommend(uid, k=30, exclude_seen=False))
        # With exclude_seen=False the seen set is eligible; with 30 slots and
        # only 40 items it's very likely at least one seen item appears.
        assert len(recs_with) > 0  # basic sanity

    def test_only_known_recipes_returned(self, tiny_train, fake_provider):
        m = _make_model(tiny_train, fake_provider)
        catalog = set(tiny_train["recipe_id"])
        recs = set(m.recommend(1000, k=20))
        assert recs.issubset(catalog)

    def test_unknown_user_falls_back_to_popularity(self, tiny_train, fake_provider):
        m = _make_model(tiny_train, fake_provider)
        recs = m.recommend(999999, k=10)
        assert len(recs) == 10
        assert all(isinstance(r, int) for r in recs)

    def test_deterministic(self, tiny_train, fake_provider):
        a = _make_model(tiny_train, fake_provider).recommend(1000, k=10)
        b = _make_model(tiny_train, fake_provider).recommend(1000, k=10)
        assert a == b

    def test_raises_before_fit(self, fake_provider):
        m = HybridLinearRecommender(feature_provider=fake_provider)
        with pytest.raises(RuntimeError, match="fit"):
            m.recommend(1000, k=5)

    def test_invalid_alpha_raises(self, fake_provider):
        with pytest.raises(ValueError, match="alpha"):
            HybridLinearRecommender(alpha=1.5, feature_provider=fake_provider)


# ============================================================
# Alpha-behaviour tests
# ============================================================

class TestAlphaBehaviour:
    def test_alpha_zero_no_cf_noise(self, tiny_train, fake_provider):
        # alpha=0 → CF component is zeroed; model should still return valid recs
        m = _make_model(tiny_train, fake_provider, alpha=0.0)
        recs = m.recommend(1000, k=10)
        assert len(recs) == 10
        assert len(set(recs)) == 10

    def test_alpha_one_no_content_noise(self, tiny_train, fake_provider):
        # alpha=1 → content component is zeroed; model should still return valid recs
        m = _make_model(tiny_train, fake_provider, alpha=1.0)
        recs = m.recommend(1000, k=10)
        assert len(recs) == 10
        assert len(set(recs)) == 10

    def test_different_alphas_produce_different_rankings(self, tiny_train, fake_provider):
        # EASE and Tag SVD have different score distributions, so blending at
        # different weights should generally reorder results.
        m0 = _make_model(tiny_train, fake_provider, alpha=0.0)
        m1 = _make_model(tiny_train, fake_provider, alpha=1.0)
        uid = 1000
        # With orthogonal CF and content signals the full lists will differ;
        # top-1 might coincide by chance but top-10 as a set should not match.
        assert set(m0.recommend(uid, k=10)) != set(m1.recommend(uid, k=10))
