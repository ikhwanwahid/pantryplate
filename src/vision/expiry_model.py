"""Synthetic ingredient-expiry heuristic for the 'use it before it spoils' demo.

This is NOT a learned model — it's a small, transparent shelf-life lookup that
turns a list of detected fridge ingredients into estimated expiry dates, so the
Stage 2 reranker can gently favour recipes that use items about to go bad.

Flow: detected ingredient -> shelf-life category (keyword match) -> sampled
shelf life (days) -> expiry date -> urgency score in [0, 1]. Sampling is seeded
so the same photo gives the same estimates across runs.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

# Typical shelf life in days, by ingredient category (low, high) inclusive.
SHELF_LIFE_DAYS = {
    "leafy_greens": (3, 5),
    "berries": (3, 6),
    "dairy": (5, 10),
    "meat": (2, 5),
    "fish": (1, 3),
    "eggs": (14, 21),
    "root_vegetables": (14, 30),
    "fruit": (4, 10),
    "condiments": (60, 180),
    "grains_dry": (180, 365),
    "default": (5, 10),
}

KEYWORD_CATEGORIES = [
    (["chicken", "beef", "pork", "turkey", "bacon", "ham", "sausage"], "meat"),
    (["salmon", "tuna", "shrimp", "fish", "cod"], "fish"),
    (["milk", "cheese", "yogurt", "cream", "butter"], "dairy"),
    (["egg"], "eggs"),
    (["spinach", "kale", "lettuce", "broccoli", "asparagus"], "leafy_greens"),
    (["berry", "berries", "strawberr", "blueberr"], "berries"),
    (["carrot", "potato", "onion", "garlic", "ginger", "celery"], "root_vegetables"),
    (["apple", "banana", "lemon", "lime", "tomato", "avocado", "pepper", "cucumber", "zucchini"], "fruit"),
    (["sauce", "vinegar", "honey", "ketchup", "mustard", "mayo", "oil"], "condiments"),
    (["rice", "pasta", "bread", "oats", "flour", "noodle", "quinoa", "nuts", "almond", "walnut"], "grains_dry"),
]


def categorize(ingredient: str) -> str:
    """Map an ingredient name to a shelf-life category via keyword matching."""
    ing_low = ingredient.lower()
    for keywords, category in KEYWORD_CATEGORIES:
        if any(kw in ing_low for kw in keywords):
            return category
    return "default"


def assign_expiry_dates(
    ingredients: list[str],
    detected_on: date | None = None,
    seed: int = 42,
) -> dict[str, date]:
    """Assign a synthetic expiry date to each detected ingredient.

    Deterministic given `seed`: shelf life is sampled from each category's
    range with a seeded RNG, so the same fridge photo yields the same expiry
    estimates across runs. `detected_on` defaults to today.
    """
    detected_on = detected_on or date.today()
    rng = random.Random(seed)
    expiry_map: dict[str, date] = {}
    for ing in ingredients:
        category = categorize(ing)
        low, high = SHELF_LIFE_DAYS[category]
        shelf_life = rng.randint(low, high)
        expiry_map[ing] = detected_on + timedelta(days=shelf_life)
    return expiry_map


def urgency_score(expiry: date, today: date | None = None) -> float:
    """Return urgency in [0, 1]. 1 = expires today/overdue, 0 = expires far away (>14 days)."""
    today = today or date.today()
    days_left = (expiry - today).days
    if days_left <= 0:
        return 1.0
    if days_left >= 14:
        return 0.0
    return 1.0 - (days_left / 14.0)
