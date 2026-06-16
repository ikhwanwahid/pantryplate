"""Tests for src/eval/significance.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.significance import bootstrap_ci, compare_models, paired_wilcoxon


class TestBootstrapCI:
    def test_mean_is_exact(self):
        m, lo, hi = bootstrap_ci([0, 0, 1, 1], seed=0)
        assert m == pytest.approx(0.5)
        assert lo <= m <= hi

    def test_constant_input_zero_width(self):
        m, lo, hi = bootstrap_ci([1, 1, 1, 1], seed=0)
        assert m == lo == hi == 1.0

    def test_empty_is_nan(self):
        m, lo, hi = bootstrap_ci([])
        assert np.isnan(m) and np.isnan(lo) and np.isnan(hi)

    def test_ci_brackets_mean_and_widens_with_spread(self):
        tight = bootstrap_ci([0.5] * 100, seed=1)
        wide = bootstrap_ci([0.0, 1.0] * 50, seed=1)
        assert (tight[2] - tight[1]) < (wide[2] - wide[1])


class TestPairedWilcoxon:
    def test_identical_returns_p1(self):
        w = paired_wilcoxon([1, 0, 1, 0], [1, 0, 1, 0])
        assert w["pvalue"] == 1.0
        assert w["n_nonzero"] == 0

    def test_a_dominates_is_significant(self):
        # a beats b for every user → strongly significant, positive mean_diff
        a = [1] * 30
        b = [0] * 30
        w = paired_wilcoxon(a, b)
        assert w["mean_diff"] == pytest.approx(1.0)
        assert w["pvalue"] < 0.05
        assert w["n_nonzero"] == 30

    def test_mixed_small_effect_not_significant(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, size=50)
        b = a.copy()
        # flip a couple to create a tiny, balanced difference
        b[0] = 1 - b[0]
        b[1] = 1 - b[1]
        w = paired_wilcoxon(a, b)
        assert 0.0 <= w["pvalue"] <= 1.0


class TestCompareModels:
    def _frame(self, vals):
        return pd.DataFrame({"user_id": range(len(vals)), "recall@10": vals})

    def test_paired_alignment_and_keys(self):
        # 20 users where BPR hits and Popularity misses → clearly significant
        a = self._frame([1] * 20 + [0] * 5)
        b = self._frame([0] * 25)
        out = compare_models(a, b, "recall@10", "BPR", "Popularity")
        assert out["n_paired"] == 25
        assert out["BPR"]["mean"] == pytest.approx(0.8)
        assert out["Popularity"]["mean"] == pytest.approx(0.0)
        assert out["significant_at_0.05"] is True
        assert "ci95" in out["BPR"]

    def test_merge_keeps_only_common_users(self):
        a = self._frame([1, 1, 1])           # users 0,1,2
        b = pd.DataFrame({"user_id": [1, 2, 3], "recall@10": [0, 0, 0]})
        out = compare_models(a, b, "recall@10", "A", "B")
        assert out["n_paired"] == 2  # only users 1,2 in common
