"""Statistical-rigor helpers for model evaluation — bootstrap CIs + paired tests.

The leaderboard reports point estimates (means over a user sample). These tools
answer the two questions a reviewer will ask:

  1. "How precise is each number?"   → bootstrap_ci  (95% CI on a mean)
  2. "Is model A really better than B?" → paired_wilcoxon  (p-value on the
     per-user difference, the standard non-parametric test for recsys
     model-vs-model comparison: paired, no normality assumption)

Both consume the per-user score vectors that `harness.evaluate(...,
return_per_user=True)` produces. Comparisons are PAIRED — same users scored by
both models (use the same seed + n_users so the samples align).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_ci(
    values,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean of `values`.

    Resamples with replacement `n_boot` times, recomputes the mean each time,
    and takes the central `ci` percentile band.

    Returns (mean, lo, hi). Returns (nan, nan, nan) for an empty input.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    # (n_boot, n) resample → row means
    boot_means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return (float(v.mean()), lo, hi)


def paired_wilcoxon(a, b) -> dict:
    """Wilcoxon signed-rank test on paired samples a vs b (a, b aligned per user).

    Returns dict with:
      mean_diff  : mean(a - b)
      n_nonzero  : number of users where a != b (the test's effective sample)
      statistic  : Wilcoxon W
      pvalue     : two-sided p-value (prob the difference is chance)

    Edge cases: if a == b everywhere (no signal to test), returns pvalue=1.0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    n_nonzero = int(np.count_nonzero(diff))
    if n_nonzero == 0:
        return {"mean_diff": 0.0, "n_nonzero": 0, "statistic": float("nan"), "pvalue": 1.0}
    from scipy.stats import wilcoxon
    try:
        res = wilcoxon(a, b)  # paired; default drops zero-differences
        stat, p = float(res.statistic), float(res.pvalue)
    except ValueError:
        # e.g. all differences zero after dropping ties
        stat, p = float("nan"), 1.0
    return {"mean_diff": float(diff.mean()), "n_nonzero": n_nonzero,
            "statistic": stat, "pvalue": p}


def compare_models(
    per_user_a: pd.DataFrame,
    per_user_b: pd.DataFrame,
    metric: str,
    label_a: str = "A",
    label_b: str = "B",
    seed: int = 42,
) -> dict:
    """Paired comparison of two models on one metric.

    Aligns the two per-user frames on user_id (so the test is genuinely paired),
    then returns each model's mean + 95% CI and the Wilcoxon result.

    per_user_a / per_user_b : DataFrames from harness.evaluate(..., return_per_user=True)
                              — must contain 'user_id' and `metric` columns.
    """
    merged = per_user_a[["user_id", metric]].merge(
        per_user_b[["user_id", metric]], on="user_id", suffixes=("_a", "_b")
    )
    a = merged[f"{metric}_a"].to_numpy()
    b = merged[f"{metric}_b"].to_numpy()
    mean_a, lo_a, hi_a = bootstrap_ci(a, seed=seed)
    mean_b, lo_b, hi_b = bootstrap_ci(b, seed=seed)
    w = paired_wilcoxon(a, b)
    return {
        "metric": metric,
        "n_paired": len(merged),
        label_a: {"mean": mean_a, "ci95": (lo_a, hi_a)},
        label_b: {"mean": mean_b, "ci95": (lo_b, hi_b)},
        "wilcoxon": w,
        "significant_at_0.05": w["pvalue"] < 0.05,
    }


def ci_row(per_user: pd.DataFrame, metric: str, seed: int = 42) -> tuple[float, float, float]:
    """Convenience: (mean, lo, hi) for one model's metric, ×100 for percent display."""
    m, lo, hi = bootstrap_ci(per_user[metric].to_numpy(), seed=seed)
    return (m * 100, lo * 100, hi * 100)
