"""Tests for src/eval/harness.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.harness import (
    bootstrap_ci,
    clear_cache,
    compare_models,
    evaluate,
)
from src.models.popularity import PopularityRecommender


TRAIN_CSV = Path("data/raw/interactions_train.csv")


# ---------- bootstrap_ci ----------


class TestBootstrapCI:
    def test_returns_three_values(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = bootstrap_ci(scores, n_bootstrap=200)
        assert len(result) == 3
        mean, lo, hi = result
        assert lo <= mean <= hi

    def test_deterministic_given_seed(self):
        scores = np.random.RandomState(0).rand(100)
        r1 = bootstrap_ci(scores, n_bootstrap=200, seed=42)
        r2 = bootstrap_ci(scores, n_bootstrap=200, seed=42)
        assert r1 == r2

    def test_different_seeds_give_different_cis(self):
        scores = np.random.RandomState(0).rand(50)
        r1 = bootstrap_ci(scores, n_bootstrap=200, seed=1)
        r2 = bootstrap_ci(scores, n_bootstrap=200, seed=2)
        # Means should be very close (same scores), but bootstrap CIs differ
        assert r1[0] == pytest.approx(r2[0])  # same mean
        # CIs are close but should not be exactly identical
        assert (r1[1], r1[2]) != (r2[1], r2[2])

    def test_empty_scores(self):
        assert bootstrap_ci([]) == (0.0, 0.0, 0.0)

    def test_ci_width_shrinks_with_more_samples(self):
        small = bootstrap_ci(np.random.RandomState(0).rand(10), n_bootstrap=200)
        large = bootstrap_ci(np.random.RandomState(0).rand(1000), n_bootstrap=200)
        # CI width should be smaller for larger samples
        small_width = small[2] - small[1]
        large_width = large[2] - large[1]
        assert large_width < small_width

    def test_accepts_pandas_series(self):
        s = pd.Series([0.1, 0.2, 0.3])
        mean, lo, hi = bootstrap_ci(s, n_bootstrap=100)
        assert isinstance(mean, float)


# ---------- evaluate (integration tests, need real data) ----------


@pytest.fixture(autouse=True)
def _clear_cache_per_test():
    """Reset the test-set cache between tests for isolation."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(scope="module")
def fitted_popularity():
    """Train popularity once for the module, reuse across tests."""
    if not TRAIN_CSV.exists():
        pytest.skip("pre-split files not downloaded")
    from src.data.loader import load_train_interactions, time_based_split
    full_train = load_train_interactions()
    train, _ = time_based_split(full_train, holdout_per_user=1)
    return PopularityRecommender().fit(train)


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="pre-split files not downloaded")
class TestEvaluateWarm:
    def test_returns_expected_keys(self, fitted_popularity):
        result = evaluate(fitted_popularity, track="warm", n_users=200, seed=42)
        for k in [5, 10, 20]:
            assert f"recall@{k}" in result
            assert f"ndcg@{k}" in result
        assert "mrr" in result
        assert "n_users_evaluated" in result
        assert result["track"] == "warm"
        assert result["seed"] == 42

    def test_deterministic_given_seed(self, fitted_popularity):
        r1 = evaluate(fitted_popularity, track="warm", n_users=500, seed=42)
        r2 = evaluate(fitted_popularity, track="warm", n_users=500, seed=42)
        for key in ["recall@10", "ndcg@10", "mrr"]:
            assert r1[key] == r2[key]

    def test_different_seeds_sample_different_users(self, fitted_popularity):
        r1 = evaluate(fitted_popularity, track="warm", n_users=500, seed=1)
        r2 = evaluate(fitted_popularity, track="warm", n_users=500, seed=2)
        # Numbers should be close but not identical
        assert r1["recall@10"] != r2["recall@10"]

    def test_popularity_warm_recall_in_expected_range(self, fitted_popularity):
        """Popularity Recall@10 on warm should be ~1-4% — non-zero, meaningful."""
        result = evaluate(fitted_popularity, track="warm", n_users=2000, seed=42)
        assert 0.01 < result["recall@10"] < 0.10, (
            f"Expected popularity warm Recall@10 ~ 1-10%, got {result['recall@10']:.4f}. "
            "If outside this range, something is off in the harness or data."
        )

    def test_n_users_is_respected(self, fitted_popularity):
        result = evaluate(fitted_popularity, track="warm", n_users=300, seed=42)
        assert result["n_users_evaluated"] == 300

    def test_return_per_user(self, fitted_popularity):
        result = evaluate(
            fitted_popularity, track="warm", n_users=100, seed=42, return_per_user=True
        )
        assert "per_user" in result
        assert isinstance(result["per_user"], pd.DataFrame)
        assert len(result["per_user"]) == 100
        assert "recall@10" in result["per_user"].columns


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="pre-split files not downloaded")
class TestEvaluateCold:
    def test_popularity_cold_recall_is_zero(self, fitted_popularity):
        """Popularity CANNOT do cold-item recommendation by construction.

        Test items have 0 raters in train, so popularity has no signal.
        Recall must be exactly 0 — this is a feature of cold evaluation,
        not a bug.
        """
        result = evaluate(fitted_popularity, track="cold", n_users=500, seed=42)
        assert result["recall@10"] == 0.0
        assert result["recall@20"] == 0.0
        assert result["mrr"] == 0.0


class TestEvaluateValidation:
    def test_invalid_track_raises(self):
        with pytest.raises(ValueError, match="track"):
            evaluate(PopularityRecommender(), track="invalid")


# ---------- compare_models ----------


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="pre-split files not downloaded")
class TestCompareModels:
    def test_returns_dataframe(self, fitted_popularity):
        df = compare_models(
            {"pop": fitted_popularity},
            track="warm",
            n_users=100,
            seed=42,
        )
        assert isinstance(df, pd.DataFrame)
        assert "pop" in df.index
        assert "recall@10" in df.columns

    def test_multiple_models(self, fitted_popularity):
        # Two popularity variants — different ranking modes
        pop_unique = fitted_popularity  # default: unique users
        df = compare_models(
            {"popularity_unique": pop_unique, "popularity_alias": pop_unique},
            track="warm",
            n_users=100,
            seed=42,
        )
        assert len(df) == 2
        # Same model under different names → identical numbers
        for col in df.columns:
            if col != "track":
                assert df.loc["popularity_unique", col] == df.loc["popularity_alias", col]


# ---------- candidate_filter hook (Stage 2 future use) ----------


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="pre-split files not downloaded")
class TestCandidateFilter:
    def test_filter_can_reduce_recommendations(self, fitted_popularity):
        """A trivial filter that returns empty should produce 0 recall."""
        def reject_all(user_id, recs):
            return []

        result = evaluate(
            fitted_popularity,
            track="warm",
            n_users=200,
            seed=42,
            candidate_filter=reject_all,
        )
        assert result["recall@10"] == 0.0

    def test_filter_can_pass_through(self, fitted_popularity):
        """An identity filter should give the same result as no filter."""
        def passthrough(user_id, recs):
            return recs

        r_no_filter = evaluate(fitted_popularity, track="warm", n_users=200, seed=42)
        r_filtered = evaluate(
            fitted_popularity,
            track="warm",
            n_users=200,
            seed=42,
            candidate_filter=passthrough,
        )
        assert r_no_filter["recall@10"] == r_filtered["recall@10"]
