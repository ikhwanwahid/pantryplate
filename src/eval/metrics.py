"""Evaluation metrics for top-K recommendation.

Single-user metrics. Aggregation (mean, bootstrap CI, etc.) is the
caller's responsibility — typically done by the eval harness in Week 2.

Convention:
    recommended : List[item_id]    ordered, highest-ranked first
    relevant    : Set[item_id]     ground truth (e.g., held-out positives)
    k           : int              cutoff (top-K)
"""

from __future__ import annotations

import math
from typing import Iterable, Set


def _to_set(x) -> Set:
    return x if isinstance(x, set) else set(x)


def recall_at_k(recommended: Iterable, relevant: Iterable, k: int) -> float:
    """Fraction of relevant items captured in the top-K recommendations.

    With a single held-out positive per user (our protocol),
    recall@k is equivalent to hit@k: 1.0 if the held-out item is in
    the top-k, else 0.0. Generalizes correctly to multiple positives.
    """
    relevant = _to_set(relevant)
    if not relevant:
        return 0.0
    top_k = list(recommended)[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Iterable, relevant: Iterable, k: int) -> float:
    """Normalized Discounted Cumulative Gain at K, binary relevance.

    DCG = Σ 1 / log2(rank + 1)   for rank starting at 1.
    IDCG = the DCG of the ideal ranking (all relevant items at the top).
    """
    relevant = _to_set(relevant)
    if not relevant:
        return 0.0
    top_k = list(recommended)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(top_k, start=1)
        if item in relevant
    )
    # Ideal: min(|relevant|, k) hits at the top of the list
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(recommended: Iterable, relevant: Iterable) -> float:
    """Reciprocal rank of the first relevant item in the full ranked list.

    Despite the name, this is the *per-user* RR. The "mean" comes from
    averaging across users at the harness level.
    Returns 0.0 if no relevant item appears in the list.
    """
    relevant = _to_set(relevant)
    if not relevant:
        return 0.0
    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0
