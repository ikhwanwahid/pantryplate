"""Stage 2 constraint scorers — pantry, nutrition, diet.

Each scorer takes a single recipe + a persona and returns a value in [0, 1]
(or {0, 1} for the hard diet filter). The combiner in `combiner.py` wires
them into the final scoring formula.

Locked decisions referenced:
    §7  Pantry — non-staple overlap, continuous score
    §8  Diet  — hard filter, tag check + ingredient blocklist
    §9  Nutrition — clipped at (5000 kcal, 1000% PDV) in the loader
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

# Re-export pantry primitives so reranker callers have one import path
from src.utils.staples import (
    STAPLES,
    get_staples_for_persona,
    missing_count,
    pantry_score,
)


__all__ = [
    "STAPLES",
    "get_staples_for_persona",
    "missing_count",
    "pantry_score",
    "nutrition_score",
    "diet_score",
    "diet_compliant",
    "INGREDIENT_BLOCKLIST",
    "TAG_FOR_RESTRICTION",
]


# ============================================================
# Nutrition score — Gaussian proximity to macro targets
# ============================================================

def nutrition_score(
    recipe_nutrition: dict,
    macro_targets: dict,
    tolerance: float = 0.2,
) -> float:
    """Gaussian proximity between a recipe's macros and a persona's targets.

    For each field in `macro_targets`, compute the relative deviation
    `d = (recipe_value - target) / target` and a Gaussian similarity
    `exp(-d^2 / (2 * tolerance^2))`. Average across fields.

    At target → 1.0. At `tolerance` deviation → ~0.61. At 2x tolerance → ~0.14.
    A `tolerance` of 0.2 means "±20% of target counts as roughly on-target",
    matching the Useful Recall threshold in the proposal.

    Parameters
    ----------
    recipe_nutrition : dict
        Keys are nutrition fields (e.g., "calories", "protein_pdv"). Values
        are per-recipe numbers (already clipped by `parse_nutrition`).
    macro_targets : dict
        Persona's per-field targets. Only fields present here are scored,
        so personas can specify a subset (e.g., just calories + protein).
    tolerance : float
        Std-dev for the Gaussian. Smaller → stricter scoring.

    Returns
    -------
    float in [0, 1]. 0.0 if the recipe has no nutrition data or targets is empty.
    """
    if not macro_targets or not recipe_nutrition:
        return 0.0
    if tolerance <= 0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}")

    sims: list[float] = []
    for field, target in macro_targets.items():
        if target <= 0:
            continue
        actual = recipe_nutrition.get(field)
        if actual is None:
            continue
        rel_dev = (actual - target) / target
        sims.append(math.exp(-(rel_dev ** 2) / (2 * tolerance ** 2)))

    if not sims:
        return 0.0
    return sum(sims) / len(sims)


# ============================================================
# Diet hard filter — tag check + ingredient blocklist
# ============================================================

# Tag names in the dataset that signal a recipe satisfies a restriction.
# A recipe with this tag is treated as compliant for that restriction.
# `None` means "no reliable tag coverage; fall back to ingredient blocklist only".
TAG_FOR_RESTRICTION: dict[str, Optional[str]] = {
    "vegan":           "vegan",
    "vegetarian":      "vegetarian",
    "gluten-free":     "gluten-free",
    "dairy-free":      "dairy-free",
    "low-carb":        "low-carb",
    "low-fat":         "low-fat",
    "low-sodium":      "low-sodium",
    "low-cholesterol": "low-cholesterol",
    "diabetic":        "diabetic",
    "kosher":          "kosher",
    "nut-free":        None,
    "egg-free":        "egg-free",
}

# Ingredient substrings that violate each restriction. A recipe with ANY
# of these ingredients fails the corresponding restriction, regardless of
# whether the tag is present. (Tag + ingredient AND-check per decision §8.)
INGREDIENT_BLOCKLIST: dict[str, frozenset[str]] = {
    "vegan": frozenset({
        "chicken", "beef", "pork", "lamb", "turkey", "bacon", "ham", "sausage",
        "fish", "salmon", "tuna", "shrimp", "prawn", "crab", "lobster", "anchovy",
        "egg", "milk", "butter", "cheese", "yogurt", "cream", "ghee",
        "honey", "gelatin",
    }),
    "vegetarian": frozenset({
        "chicken", "beef", "pork", "lamb", "turkey", "bacon", "ham", "sausage",
        "fish", "salmon", "tuna", "shrimp", "prawn", "crab", "lobster", "anchovy",
        "gelatin",
    }),
    "gluten-free": frozenset({
        "wheat", "flour", "all-purpose flour", "flmy", "bread", "pasta",
        "spaghetti", "noodles", "barley", "rye", "couscous", "bulgur",
        "semolina", "panko", "breadcrumb",
    }),
    "dairy-free": frozenset({
        "milk", "butter", "cheese", "yogurt", "cream", "ghee", "buttermilk",
    }),
    "nut-free": frozenset({
        "almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut",
        "peanut", "macadamia", "brazil nut", "pine nut",
    }),
    "egg-free": frozenset({"egg"}),
}


def _ingredient_blocked(ingredients: Iterable[str], blocklist: Iterable[str]) -> bool:
    """True if any blocklist substring appears in any ingredient (case-insensitive)."""
    blocked = set(blocklist)
    for ing in ingredients:
        if not isinstance(ing, str):
            continue
        ing_low = ing.lower()
        for bad in blocked:
            if bad in ing_low:
                return True
    return False


def diet_compliant(
    recipe_ingredients: Iterable[str],
    recipe_tags: Iterable[str],
    restrictions: Iterable[str],
) -> bool:
    """True if the recipe satisfies ALL restrictions.

    A restriction is satisfied when:
      (a) the corresponding tag is present in `recipe_tags`, AND
      (b) no ingredient in `recipe_ingredients` matches the blocklist.

    Both must hold — tags are noisy (EDA found ~3% vegan-tagged recipes
    still contain animal products), so the ingredient blocklist is a
    defense-in-depth check.

    Restrictions without tag coverage (e.g., "nut-free") use only the
    blocklist check. Unknown restrictions are silently treated as
    "no constraint" — we don't fail-closed on unknown labels.
    """
    if not restrictions:
        return True

    tags_lower = {t.lower() for t in recipe_tags if isinstance(t, str)}

    for r in restrictions:
        r = r.lower().strip()
        required_tag = TAG_FOR_RESTRICTION.get(r)
        if required_tag is not None and required_tag not in tags_lower:
            return False
        blocklist = INGREDIENT_BLOCKLIST.get(r)
        if blocklist is not None and _ingredient_blocked(recipe_ingredients, blocklist):
            return False

    return True


def diet_score(
    recipe_ingredients: Iterable[str],
    recipe_tags: Iterable[str],
    restrictions: Iterable[str],
) -> int:
    """1 if compliant, 0 otherwise — the hard filter in the combiner formula.

    Thin alias for `diet_compliant` returning int instead of bool so the
    combiner formula `s_diet × (...)` reads naturally.
    """
    return int(diet_compliant(recipe_ingredients, recipe_tags, restrictions))
