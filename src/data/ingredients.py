"""Ingredient normalization and nutrition parsing.

The Food.com dataset ships free-text ingredient names like "1 cup
shredded cheddar cheese" and a canonical map (`ingr_map.pkl`) that
collapses surface variants to ~8k canonical ingredient names.

Nutrition is a stringified list of 7 PDV (Percent Daily Value) numbers
in the order: [calories, total_fat, sugar, sodium, protein,
saturated_fat, carbs]. Note: `calories` is absolute kcal, the other six
are PDV percentages.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd


NUTRITION_FIELDS = (
    "calories",
    "fat_pdv",
    "sugar_pdv",
    "sodium_pdv",
    "protein_pdv",
    "satfat_pdv",
    "carbs_pdv",
)

# Outlier caps derived from sanity check (max calories was 434,360 — clearly
# misparsed portion sizes). 5000 kcal is "enormous single meal" upper bound;
# 1000% PDV is "implausibly nutrient-dense" cap for the percentage fields.
DEFAULT_CALORIE_CAP = 5000.0
DEFAULT_PDV_CAP = 1000.0


@lru_cache(maxsize=1)
def _load_ingr_map(path: str) -> dict[str, str]:
    """Build a {raw_ingr -> canonical} lookup from ingr_map.pkl."""
    df = pd.read_pickle(path)
    return dict(zip(df["raw_ingr"], df["replaced"]))


def normalize_ingredient(
    raw_text: str,
    ingr_map_path: str | Path = "data/raw/ingr_map.pkl",
) -> str:
    """Map a raw ingredient string to its canonical form.

    Falls back to a lowercased, stripped version of the input if no mapping
    exists — better to keep an un-canonicalized token than to drop it.
    """
    if not isinstance(raw_text, str):
        return ""
    cleaned = raw_text.strip().lower()
    lookup = _load_ingr_map(str(ingr_map_path))
    return lookup.get(cleaned, cleaned)


def parse_nutrition(
    nutrition_str: str,
    clip: bool = True,
    calorie_cap: float = DEFAULT_CALORIE_CAP,
    pdv_cap: float = DEFAULT_PDV_CAP,
) -> Optional[dict]:
    """Parse a nutrition string into a {field: value} dict.

    Returns None if the string is not a valid 7-element list. When `clip` is
    True, values exceeding the caps are clipped — this removes the long tail
    of misparsed portion sizes (the sanity check found 1,049 recipes >5000
    kcal including one at 434,360 kcal).
    """
    if not isinstance(nutrition_str, str):
        return None
    try:
        values = ast.literal_eval(nutrition_str)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(values, list) or len(values) != 7:
        return None
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if clip:
        values[0] = min(values[0], calorie_cap)
        values[1:] = [min(v, pdv_cap) for v in values[1:]]
    return dict(zip(NUTRITION_FIELDS, values))


def safe_parse_list(s: str) -> list:
    """Parse a stringified list of strings, returning [] on failure.

    Used for `tags`, `ingredients`, `steps` columns.
    """
    if not isinstance(s, str):
        return []
    try:
        result = ast.literal_eval(s)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []
