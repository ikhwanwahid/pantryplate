"""Pantry derivation and persona loading.

A persona JSON file lives at `data/personas/{persona_id}.json` and looks like:

    {
      "id": "fitness_focused",
      "label": "Fitness-focused lifter",
      "description": "...",
      "macro_targets": {"calories": 600, "protein_pdv": 50, ...},
      "restrictions": ["vegetarian"],
      "pantry": ["chicken", "rice", "broccoli"],
      "taste_seeds": ["high-protein grilled chicken", ...]
    }

The pantry can also be derived from a user's owned/cooked recipes:
sum canonical ingredients across those recipes to get a frequency-weighted
inventory.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from .ingredients import normalize_ingredient


def derive_pantry_from_recipes(
    owned_recipe_ids: Iterable[int],
    recipes_df: pd.DataFrame,
    ingr_map_path: str | Path = "data/raw/ingr_map.pkl",
) -> dict[str, int]:
    """Aggregate canonical ingredients across a set of owned recipes.

    Returns {canonical_ingredient: count}. An ingredient that appears in
    three of the user's owned recipes gets count=3 — a rough proxy for
    "this user probably has this on hand."
    """
    owned_ids = set(owned_recipe_ids)
    owned = recipes_df[recipes_df["id"].isin(owned_ids)]
    if "ingredients_parsed" not in owned.columns:
        raise ValueError(
            "recipes_df must have 'ingredients_parsed' column "
            "(load via loader.load_recipes)"
        )
    counter: Counter[str] = Counter()
    for ings in owned["ingredients_parsed"]:
        for raw in ings:
            canonical = normalize_ingredient(raw, ingr_map_path)
            if canonical:
                counter[canonical] += 1
    return dict(counter)


def load_persona(
    persona_id: str,
    personas_dir: str | Path = "data/personas",
) -> dict:
    """Load a persona JSON. Raises FileNotFoundError if missing."""
    path = Path(personas_dir) / f"{persona_id}.json"
    with open(path) as f:
        return json.load(f)
