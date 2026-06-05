"""Dual-track evaluation harness for PantryPlate Stage 1 models.

This module is the load-bearing Week 2 deliverable. Every Stage 1 model
in the project goes through this harness so results are directly comparable.
Without it, four people would implement Recall@10 four different ways and
the comparison table would be noise.

Usage (from a model author's perspective):

    from src.eval.harness import evaluate
    from src.data.loader import load_train_interactions, time_based_split
    from your_model import YourRecommender

    train_full = load_train_interactions()
    train, _ = time_based_split(train_full, holdout_per_user=1)   # use full train for fit
    model = YourRecommender().fit(train)

    results = evaluate(model, track="warm")
    # results = {"recall@5": 0.02, "recall@10": 0.03, ..., "n_users": 24382}

    cold_results = evaluate(model, track="cold")
    # Same metrics, against the authors' pre-split cold test set.

The harness handles:
- Loading the appropriate test set per track
- Looping over test users (or a sampled subset for fast iteration)
- Computing Recall@K, NDCG@K, MRR consistently
- Per-user score collection for bootstrap CIs
- Deterministic sampling via seed control

What models must provide:
- `.recommend(user_id: int, k: int, exclude_seen: bool = True) -> list[int]`
- The model must have been fit on a training set that does NOT include any
  test holdouts (the harness does not police this — caller's responsibility).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from src.data.loader import (
    load_test_interactions,
    load_train_interactions,
    time_based_split,
)
from src.eval.metrics import recall_at_k, ndcg_at_k, mrr


# Cache loaded test sets so repeated calls within a session are fast
_TEST_SET_CACHE: dict[str, pd.DataFrame] = {}


class Recommender(Protocol):
    """Minimal interface every Stage 1 model must satisfy."""

    def recommend(
        self, user_id: int, k: int = 10, exclude_seen: bool = True
    ) -> list[int]: ...


def _load_test_set(track: str, data_path: str | Path) -> pd.DataFrame:
    """Get the test set for the given track. Caches to avoid re-loading."""
    cache_key = f"{track}:{data_path}"
    if cache_key in _TEST_SET_CACHE:
        return _TEST_SET_CACHE[cache_key]

    if track == "warm":
        # Track A: time-based LOO holdout from the authors' train file
        full_train = load_train_interactions(path=data_path)
        _, test_warm = time_based_split(full_train, holdout_per_user=1)
        _TEST_SET_CACHE[cache_key] = test_warm
        return test_warm

    elif track == "cold":
        # Track B: authors' pre-split test file (cold items)
        test_cold = load_test_interactions(path=data_path)
        _TEST_SET_CACHE[cache_key] = test_cold
        return test_cold

    else:
        raise ValueError(f"track must be 'warm' or 'cold', got {track!r}")


def evaluate(
    model: Recommender,
    track: str = "warm",
    k_values: tuple[int, ...] = (5, 10, 20),
    n_users: int | None = None,
    seed: int = 42,
    return_per_user: bool = False,
    candidate_filter: Callable[[int, list[int]], list[int]] | None = None,
    data_path: str | Path = "data/raw",
) -> dict:
    """Evaluate a Stage 1 model on one evaluation track.

    Parameters
    ----------
    model : Recommender
        Any object with a `.recommend(user_id, k, exclude_seen)` method.
        Must have been `.fit()` on training data that excludes any
        test-set holdouts for the chosen track.
    track : {"warm", "cold"}
        - "warm" : Track A — time-based LOO holdout from authors' train.
          Items have rater history. All Stage 1 models compete here.
        - "cold" : Track B — authors' pre-split test file.
          Items have ZERO raters in train. Only content-aware models
          can meaningfully compete; CF-only models will (correctly) score
          ~0 by construction.
    k_values : tuple[int, ...]
        K cutoffs for Recall@K and NDCG@K. Default (5, 10, 20).
    n_users : int or None
        If int: randomly sample this many test users (deterministic given seed).
        Use this for fast iteration during development (~2000 = ~5 sec).
        If None: evaluate against the full test set.
    seed : int
        Random seed for user sampling. Same seed → same users sampled.
    return_per_user : bool
        If True, the returned dict includes a "per_user" key with a DataFrame
        of every user's per-metric scores. Useful for bootstrap CIs and
        statistical comparisons across models.
    candidate_filter : callable, optional
        Stage 2 hook (used in Week 4+). Function `(user_id, recommended_ids)`
        → filtered list. Lets the reranker apply diet hard-filter and
        constraint-aware reranking before scoring.
    data_path : str or Path
        Where raw data lives. Default "data/raw".

    Returns
    -------
    dict with keys:
        recall@5, recall@10, recall@20, ndcg@5, ndcg@10, ndcg@20, mrr
            mean values across evaluated users
        n_users_evaluated : int
        track : str
        seed : int
        per_user : pd.DataFrame  (only if return_per_user=True)
    """
    if track not in ("warm", "cold"):
        raise ValueError(f"track must be 'warm' or 'cold', got {track!r}")

    test_df = _load_test_set(track, data_path)
    truth = test_df.groupby("user_id")["recipe_id"].agg(set).to_dict()

    # Sample users if requested
    rng = random.Random(seed)
    all_uids = list(truth.keys())
    if n_users is not None and n_users < len(all_uids):
        eval_uids = rng.sample(all_uids, n_users)
    else:
        eval_uids = all_uids

    k_max = max(k_values)
    per_user_records: list[dict] = []

    for uid in eval_uids:
        recs = model.recommend(int(uid), k=k_max, exclude_seen=True)

        if candidate_filter is not None:
            recs = candidate_filter(int(uid), recs)

        relevant = truth[uid]
        row: dict[str, float | int] = {"user_id": uid}
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(recs, relevant, k)
            row[f"ndcg@{k}"] = ndcg_at_k(recs, relevant, k)
        row["mrr"] = mrr(recs, relevant)
        per_user_records.append(row)

    per_user_df = pd.DataFrame(per_user_records)
    if per_user_df.empty:
        raise RuntimeError(
            f"No test users found for track={track!r}. "
            "Check the data path and that the train/test split files exist."
        )

    # Aggregate
    metric_cols = [c for c in per_user_df.columns if c != "user_id"]
    means = per_user_df[metric_cols].mean().to_dict()

    out: dict = {
        **means,
        "n_users_evaluated": len(per_user_df),
        "track": track,
        "seed": seed,
    }
    if return_per_user:
        out["per_user"] = per_user_df
    return out


def bootstrap_ci(
    per_user_scores: pd.Series | np.ndarray | list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a per-user metric.

    Resamples the per-user scores with replacement `n_bootstrap` times,
    computes the mean each time, and returns the (mean, lower_ci, upper_ci).
    Standard practice for recsys metric comparisons.

    Parameters
    ----------
    per_user_scores : array-like
        One score per user (e.g., recall@10 for every user evaluated).
    n_bootstrap : int
        Number of bootstrap resamples. Default 1000.
    ci : float
        Confidence level (between 0 and 1). Default 0.95.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (mean, lower_ci, upper_ci) : tuple[float, float, float]
    """
    scores = np.asarray(per_user_scores, dtype=float)
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(boot_means, alpha))
    upper = float(np.quantile(boot_means, 1.0 - alpha))
    mean = float(scores.mean())
    return mean, lower, upper


def compare_models(
    models: dict[str, Recommender],
    track: str = "warm",
    k_values: tuple[int, ...] = (5, 10, 20),
    n_users: int | None = None,
    seed: int = 42,
    data_path: str | Path = "data/raw",
) -> pd.DataFrame:
    """Evaluate multiple models against the same test set and return a comparison.

    Parameters
    ----------
    models : dict[name, Recommender]
        Mapping from model name (displayed in results) to fitted model.
    track : str
        "warm" or "cold". Same for all models.
    n_users : int or None
        Sample this many test users (same sample across all models, for fair comparison).
    seed : int
        Seed used both for user sampling and for any model-side randomness.

    Returns
    -------
    pd.DataFrame
        Rows = models. Columns = metrics. Numerical values are means across users.
    """
    rows: list[dict] = []
    for name, model in models.items():
        result = evaluate(
            model,
            track=track,
            k_values=k_values,
            n_users=n_users,
            seed=seed,
            data_path=data_path,
        )
        row = {"model": name, "track": track}
        for k, v in result.items():
            if k not in ("track", "seed"):
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def clear_cache() -> None:
    """Clear the test-set cache. Useful in tests or when data files change."""
    _TEST_SET_CACHE.clear()
