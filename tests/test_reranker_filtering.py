"""Tests for src/reranker/filtering.py — diet pre-filter for Stage 1 candidates."""

from __future__ import annotations

import pandas as pd
import pytest

from src.reranker.filtering import filter_by_diet


def _make_recipes() -> pd.DataFrame:
    """5 recipes mixed across diet compliance for vegan/vegetarian/gluten-free."""
    return pd.DataFrame({
        "ingredients_parsed": [
            ["tofu", "broccoli", "soy sauce"],         # 100: vegan
            ["chicken breast", "rice"],                 # 101: meat
            ["pasta", "tomato", "olive oil"],           # 102: vegetarian but has gluten
            ["quinoa", "kale", "olive oil"],            # 103: vegan + gluten-free
            ["eggs", "cheese", "spinach"],              # 104: vegetarian, not vegan
        ],
        "tags_parsed": [
            ["vegan", "asian"],
            ["main-dish"],
            ["vegetarian", "italian"],
            ["vegan", "gluten-free", "healthy"],
            ["vegetarian", "breakfast"],
        ],
    }, index=pd.Index([100, 101, 102, 103, 104], name="recipe_id"))


# ============================================================
# filter_by_diet
# ============================================================

class TestFilterByDiet:
    def test_empty_restrictions_passes_through(self):
        recipes = _make_recipes()
        out = filter_by_diet([100, 101, 102, 103, 104], [], recipes, target_k=3)
        assert out == [100, 101, 102]  # first 3 unchanged

    def test_vegan_keeps_only_vegan(self):
        recipes = _make_recipes()
        out = filter_by_diet([100, 101, 102, 103, 104], ["vegan"], recipes, target_k=10)
        assert out == [100, 103]

    def test_vegetarian_keeps_veg_and_vegan(self):
        recipes = _make_recipes()
        out = filter_by_diet([100, 101, 102, 103, 104], ["vegetarian"], recipes, target_k=10)
        # Recipe 100 is vegan-tagged (not vegetarian-tagged), so it should not be
        # included — our tag check is strict
        assert 101 not in out  # chicken
        assert 102 in out      # vegetarian italian
        assert 104 in out      # vegetarian breakfast

    def test_preserves_input_order(self):
        recipes = _make_recipes()
        # Pass in reverse order — output should also be in reverse (of compliant ones)
        out = filter_by_diet([104, 103, 102, 101, 100], ["vegan"], recipes, target_k=10)
        assert out == [103, 100]

    def test_stops_at_target_k(self):
        recipes = _make_recipes()
        out = filter_by_diet([100, 103, 100, 103], ["vegan"], recipes, target_k=1)
        assert len(out) == 1
        assert out[0] == 100

    def test_returns_fewer_than_target_if_pool_runs_out(self):
        """If the pool doesn't contain enough compliant items, return what we have."""
        recipes = _make_recipes()
        out = filter_by_diet([100, 101, 102, 103, 104], ["vegan"], recipes, target_k=100)
        # Only 2 vegan-compliant recipes in the pool
        assert len(out) == 2

    def test_unknown_recipe_ids_silently_skipped(self):
        recipes = _make_recipes()
        out = filter_by_diet([99999, 100, 88888, 103], ["vegan"], recipes, target_k=5)
        assert out == [100, 103]

    def test_multiple_restrictions_all_must_pass(self):
        recipes = _make_recipes()
        out = filter_by_diet(
            [100, 103, 102], ["vegan", "gluten-free"], recipes, target_k=10
        )
        # Recipe 100: vegan-tagged but no gluten-free tag → fails
        # Recipe 103: vegan-tagged AND gluten-free-tagged → passes
        # Recipe 102: vegetarian (not vegan) → fails
        assert out == [103]

    def test_empty_input_returns_empty(self):
        recipes = _make_recipes()
        assert filter_by_diet([], ["vegan"], recipes, target_k=10) == []
