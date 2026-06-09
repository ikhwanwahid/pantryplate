"""Stage 1 candidate filtering — hard constraints applied before Stage 2.

Diet is a hard, binary constraint: a recipe is either edible for the user
or not. Ranking impossible-to-eat recipes wastes Stage 2's budget. So we
filter the candidate pool at Stage 1's exit, BEFORE the reranker sees it.

The canonical Stage 1 → Stage 2 contract becomes:

    raw_candidates   = stage1_model.recommend(...)        # diet-blind
    eligible         = filter_by_diet(raw_candidates,     # hard filter
                                       persona["restrictions"],
                                       recipes_df,
                                       target_k=100)
    final_top_k      = stage2_reranker.rerank(persona,    # soft constraints
                                               eligible, ...)

Why this lives in `src/reranker/` and not `src/models/`: it's about
*how Stage 2 expects its inputs to look* (diet-compliant), so the
contract belongs alongside the reranker. Stage 1 models stay simple.

See docs/data_decisions.md §8 for the architectural rationale.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.reranker.scores import diet_compliant


def filter_by_diet(
    candidate_ids: Iterable[int],
    restrictions: Iterable[str],
    recipes_df: pd.DataFrame,
    target_k: int,
) -> list[int]:
    """Return up to `target_k` candidates that satisfy ALL restrictions.

    Preserves the input order (so Stage 1's content/CF ranking signal
    survives the filter). If `restrictions` is empty, returns the first
    `target_k` candidates unchanged.

    Implementation pattern: "expand then filter". The caller should pass
    a candidate list LARGER than `target_k` (typically 3-5x) so we have
    headroom to throw out non-compliant recipes and still hit the target.

    Parameters
    ----------
    candidate_ids : iterable of recipe_id
        Output of a Stage 1 model. Should already be ranked by Stage 1's
        scoring (we preserve order).
    restrictions : iterable of str
        Dietary restrictions the persona declared. See `TAG_FOR_RESTRICTION`
        in scores.py for the recognized set. Unknown restrictions are
        silently ignored (we don't fail-closed on labels we don't recognize).
    recipes_df : pd.DataFrame
        Must be indexed by recipe_id and contain `ingredients_parsed`
        and `tags_parsed` columns.
    target_k : int
        Maximum number of compliant candidates to return. Stops scanning
        once this many compliant candidates are found.

    Returns
    -------
    list[int]
        At most `target_k` recipe_ids, in original Stage 1 order, all
        satisfying ALL restrictions. May be shorter than `target_k` if
        the input list runs out before enough compliant recipes are found.
    """
    target_k = int(target_k)
    restrictions = list(restrictions)
    if not restrictions:
        return [int(c) for c in candidate_ids][:target_k]

    compliant: list[int] = []
    for rid in candidate_ids:
        rid = int(rid)
        if rid not in recipes_df.index:
            continue
        row = recipes_df.loc[rid]
        if diet_compliant(
            row.get("ingredients_parsed") or [],
            row.get("tags_parsed") or [],
            restrictions,
        ):
            compliant.append(rid)
            if len(compliant) >= target_k:
                break
    return compliant
