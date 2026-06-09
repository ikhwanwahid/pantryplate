"""Useful Recall@K — the project's signature end-to-end metric.

A recipe counts as a "useful hit" only if ALL FOUR conditions hold:
    1. It is the held-out positive (standard recall)
    2. It is pantry-feasible: missing_count(recipe, pantry) ≤ pantry_missing_threshold
    3. It is macro-near: each macro within ±macro_tolerance of the persona's target
    4. It is diet-compliant: diet_score(recipe, restrictions) == 1

Defaults match the proposal:
    pantry_missing_threshold = 3   (user would need to buy ≤3 more things)
    macro_tolerance          = 0.2 (±20% of each macro target)

The metric is computed ONLY when the user supplies a persona. Without a
persona, we have no pantry / macro / diet to check, so it degenerates to
standard Recall@K — in which case just use src/eval/metrics.recall_at_k.
"""

from __future__ import annotations

from typing import Iterable, Set

import pandas as pd

from src.reranker.scores import (
    diet_compliant,
    get_staples_for_persona,
    missing_count,
)


def _to_set(x) -> Set:
    return x if isinstance(x, set) else set(x)


def is_macro_near(
    recipe_nutrition: dict,
    macro_targets: dict,
    tolerance: float = 0.2,
) -> bool:
    """True if EVERY macro in `macro_targets` is within ±tolerance of its target.

    All-must-pass (vs. nutrition_score which averages). Matches the
    proposal's Useful Recall condition for the macro check.

    A missing nutrition field on the recipe is treated as a fail — we can't
    confirm it's near the target without data.
    """
    if not macro_targets:
        return True
    if not recipe_nutrition:
        return False
    for field, target in macro_targets.items():
        if target <= 0:
            continue
        actual = recipe_nutrition.get(field)
        if actual is None:
            return False
        if abs(actual - target) / target > tolerance:
            return False
    return True


def is_useful(
    recipe_id: int,
    persona: dict,
    recipes_df: pd.DataFrame,
    pantry_missing_threshold: int = 3,
    macro_tolerance: float = 0.2,
) -> bool:
    """True if a single recipe satisfies all three constraints for this persona.

    Diet → pantry-feasible → macro-near. Short-circuits on the cheapest check
    that fails for speed.
    """
    if recipe_id not in recipes_df.index:
        return False

    row = recipes_df.loc[recipe_id]
    ings = row.get("ingredients_parsed") or []
    tags = row.get("tags_parsed") or []
    nutrition = row.get("nutrition_parsed") or {}

    # Diet first — hard filter, often the cheapest to fail
    restrictions = persona.get("restrictions", [])
    if not diet_compliant(ings, tags, restrictions):
        return False

    # Pantry feasibility
    user_pantry = persona.get("pantry", [])
    staples = get_staples_for_persona(persona)
    if missing_count(ings, user_pantry, staples=staples) > pantry_missing_threshold:
        return False

    # Macro proximity
    macro_targets = persona.get("macro_targets", {})
    if not is_macro_near(nutrition, macro_targets, tolerance=macro_tolerance):
        return False

    return True


def useful_recall_at_k(
    recommended: Iterable[int],
    relevant: Iterable[int],
    persona: dict,
    recipes_df: pd.DataFrame,
    k: int,
    pantry_missing_threshold: int = 3,
    macro_tolerance: float = 0.2,
) -> float:
    """Useful Recall@K for one user.

    Fraction of `relevant` items that appear in the top-K of `recommended`
    AND satisfy all three constraints for the given persona.

    With a single held-out positive per user (our protocol), this is
    binary 0/1: 1 if the held-out item is in top-K AND is useful for the
    persona; 0 otherwise. Aggregation (mean across users) happens in the
    caller, same convention as the regular harness.

    Returns
    -------
    float in [0, 1].
    """
    relevant = _to_set(relevant)
    if not relevant:
        return 0.0
    top_k = list(recommended)[:k]
    hits = 0
    for r in top_k:
        if r not in relevant:
            continue
        if is_useful(
            r, persona, recipes_df,
            pantry_missing_threshold=pantry_missing_threshold,
            macro_tolerance=macro_tolerance,
        ):
            hits += 1
    return hits / len(relevant)
