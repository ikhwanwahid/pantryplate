"""Universal kitchen staples — items every household is assumed to have.

The Stage 2 reranker's `s_pantry` score is computed on **non-staple**
ingredients only. This avoids penalizing recipes for needing
salt/pepper/water/oil/etc. that every household has anyway.

Why: Week 1 EDA found that without assuming staples, median s_pantry
was 0.00 (a typical 6-9 ingredient recipe shares zero items with a
25-item user-specific pantry). Assuming the ~20 staples below raises
median s_pantry to 0.30-0.40 — discriminative without being trivial,
and matches how cooks actually think about "what do I need to buy?"

This is a project-wide constant. To override for a specific user:
  - vegan personas auto-exclude DAIRY_AND_EGGS
  - any persona can list `exclude_from_staples` in its JSON to override

See `docs/data_decisions.md` §7 for the empirical evidence and the
decision rationale.
"""

from __future__ import annotations

from typing import Iterable, Optional


# Items the average household always has on hand.
# Includes canonicalization variants because `ingr_map.pkl` is imperfect
# (e.g., "flmy" is its canonical form for "flour"; "garlic clove" is
# distinct from "garlic" in the map).
#
# Set was empirically validated against recipe-level ingredient frequencies
# in `notebooks/week1_eda.ipynb` §4b. 28 of the original 32 items rank
# in the top 50 most common ingredients. The 5 items added in the 2026-05-28
# refinement (cinnamon, ground cinnamon, cornstarch, chicken broth, honey,
# paprika) appear in 3-5% of recipes each and are defensible household pantry
# items.
#
# Known issue: `honey` and `chicken broth` are not vegan-friendly. The
# current `get_staples_for_persona()` only drops DAIRY_AND_EGGS for vegan
# personas. A follow-up fix would add ANIMAL_PRODUCTS_BEYOND_DAIRY
# = {"honey", "chicken broth"} and exclude it for vegans too.
STAPLES: frozenset[str] = frozenset({
    # Seasonings
    "salt", "pepper", "black pepper", "salt and pepper",
    # Liquids & cooking media
    "water", "olive oil", "vegetable oil", "oil", "cooking spray",
    # Sweeteners
    "sugar", "brown sugar", "honey",
    # Baking essentials
    "flour", "all-purpose flour", "flmy", "all-purpose flmy",
    "baking powder", "baking soda", "cornstarch",
    # Dairy & eggs (auto-excluded for vegan personas)
    "butter", "egg", "eggs", "milk",
    # Aromatics
    "garlic", "garlic clove", "garlic cloves", "garlic powder",
    "onion", "onion powder",
    # Acidity
    "vinegar", "white vinegar",
    "lemon juice",
    # Flavor & spices
    "vanilla", "vanilla extract",
    "cinnamon", "ground cinnamon",
    "paprika",
    # Stocks
    "chicken broth",
})

# Items that vegan users typically don't have — auto-removed from STAPLES
# when a persona has "vegan" in its restrictions list.
DAIRY_AND_EGGS: frozenset[str] = frozenset({
    "butter", "egg", "eggs", "milk",
})


def get_staples_for_persona(persona: Optional[dict] = None) -> frozenset[str]:
    """Return the staples set adjusted for a persona's restrictions/overrides.

    Two override mechanisms:
      1. If persona has "vegan" in its restrictions list, dairy and eggs
         are removed from the staples set (the user needs them in their
         explicit pantry to count).
      2. If persona has an "exclude_from_staples" list, those items are
         removed (e.g., gluten-free user excludes "flour").

    Passing `None` or `{}` returns the full default STAPLES set.
    """
    excluded: set[str] = set()
    if persona:
        excluded.update(persona.get("exclude_from_staples", []))
        if "vegan" in persona.get("restrictions", []):
            excluded |= DAIRY_AND_EGGS
    return STAPLES - excluded


def pantry_score(
    recipe_ingredients: Iterable[str],
    user_pantry: Iterable[str],
    staples: frozenset[str] = STAPLES,
) -> float:
    """Compute s_pantry on non-staple ingredients only.

    s_pantry = |non_staple_ings ∩ user_pantry| / max(1, |non_staple_ings|)

    Returns:
        0.0 if recipe has no ingredients
        1.0 if recipe contains only staples (nothing to match — always achievable)
        Otherwise: overlap fraction of non-staple ingredients
    """
    recipe = set(recipe_ingredients) if not isinstance(recipe_ingredients, set) else recipe_ingredients
    pantry = set(user_pantry) if not isinstance(user_pantry, set) else user_pantry
    if not recipe:
        return 0.0
    non_staple = recipe - staples
    if not non_staple:
        return 1.0
    return len(non_staple & pantry) / len(non_staple)


def pantry_score_with_expiry(
    recipe_ingredients: Iterable[str],
    user_pantry: Iterable[str],
    pantry_expiry: Optional[dict] = None,
    staples: frozenset[str] = STAPLES,
) -> float:
    """Like `pantry_score`, but boosts recipes that use soon-to-expire items.

        boosted = base_score * (1 + 0.5 * avg_urgency)   (clipped to 1.0)

    where `avg_urgency` averages `urgency_score()` over the matched non-staple
    ingredients that have an entry in `pantry_expiry` (an {item: expiry_date}
    map, e.g. from `assign_expiry_dates`). If `pantry_expiry` is empty/None this
    is byte-for-byte identical to `pantry_score`, so the offline eval / α-sweep
    (which never pass expiry) are unaffected — the boost is demo-only.
    """
    base = pantry_score(recipe_ingredients, user_pantry, staples)
    if not pantry_expiry:
        return base

    # Lazy import: keeps this core util free of a src.vision dependency at
    # module-load time (expiry_model itself is stdlib-only).
    from src.vision.expiry_model import urgency_score

    recipe = set(recipe_ingredients) if not isinstance(recipe_ingredients, set) else recipe_ingredients
    pantry = set(user_pantry) if not isinstance(user_pantry, set) else user_pantry
    matched = (recipe - staples) & pantry

    urgencies = [
        urgency_score(expiry)
        for ing in matched
        for item, expiry in pantry_expiry.items()
        if item.lower() in ing.lower() or ing.lower() in item.lower()
    ]
    if not urgencies:
        return base

    avg_urgency = sum(urgencies) / len(urgencies)
    return min(base * (1 + 0.5 * avg_urgency), 1.0)


def missing_count(
    recipe_ingredients: Iterable[str],
    user_pantry: Iterable[str],
    staples: frozenset[str] = STAPLES,
) -> int:
    """Count non-staple ingredients NOT in the user's pantry.

    Used for the 'flexible feasibility' alternative metric:
        feasible = missing_count(...) <= N    (typically N=3)

    Staples are never counted as missing (they're assumed available).
    """
    recipe = set(recipe_ingredients) if not isinstance(recipe_ingredients, set) else recipe_ingredients
    pantry = set(user_pantry) if not isinstance(user_pantry, set) else user_pantry
    if not recipe:
        return 0
    return len((recipe - staples) - pantry)
