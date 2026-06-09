"""Tests for src/eval/useful_recall.py."""

from __future__ import annotations

import pandas as pd
import pytest

from src.eval.useful_recall import (
    is_macro_near,
    is_useful,
    useful_recall_at_k,
)


def _make_recipes() -> pd.DataFrame:
    """Same 5-recipe dataset as the combiner tests, indexed by recipe_id."""
    return pd.DataFrame({
        "ingredients_parsed": [
            ["tofu", "broccoli", "soy sauce"],
            ["chicken breast", "rice"],
            ["tofu", "almonds", "broccoli"],
            ["tofu", "broccoli"],
            ["pasta", "tomato"],
        ],
        "tags_parsed": [
            ["vegan", "asian"],
            ["main-dish"],
            ["vegan"],
            ["vegan"],
            ["italian"],
        ],
        "nutrition_parsed": [
            {"calories": 500, "protein_pdv": 50},
            {"calories": 500, "protein_pdv": 50},
            {"calories": 500, "protein_pdv": 50},
            {"calories": 1200, "protein_pdv": 10},
            {"calories": 500, "protein_pdv": 50},
        ],
    }, index=pd.Index([100, 101, 102, 103, 104], name="recipe_id"))


def _vegan_persona():
    return {
        "pantry": ["tofu", "broccoli", "soy sauce"],
        "macro_targets": {"calories": 500, "protein_pdv": 50},
        "restrictions": ["vegan"],
        "exclude_from_staples": [],
    }


# ============================================================
# is_macro_near
# ============================================================

class TestIsMacroNear:
    targets = {"calories": 500, "protein_pdv": 50}

    def test_exact_match_passes(self):
        assert is_macro_near({"calories": 500, "protein_pdv": 50}, self.targets)

    def test_within_tolerance_passes(self):
        assert is_macro_near({"calories": 599, "protein_pdv": 41}, self.targets, tolerance=0.2)

    def test_just_outside_tolerance_fails(self):
        # 600/500 = 1.2 exactly at boundary; > tolerance check is strict.
        assert is_macro_near({"calories": 600, "protein_pdv": 50}, self.targets, tolerance=0.2)
        assert not is_macro_near({"calories": 601, "protein_pdv": 50}, self.targets, tolerance=0.2)

    def test_all_must_pass(self):
        """Hitting calories but missing protein → fail."""
        assert not is_macro_near({"calories": 500, "protein_pdv": 80}, self.targets, tolerance=0.2)

    def test_missing_field_fails(self):
        """Recipe missing a target field can't be verified → fail."""
        assert not is_macro_near({"calories": 500}, self.targets, tolerance=0.2)

    def test_empty_targets_passes_trivially(self):
        assert is_macro_near({"calories": 500}, {})


# ============================================================
# is_useful
# ============================================================

class TestIsUseful:
    def test_recipe_100_useful_for_vegan(self):
        """100 = tofu+broccoli+soy, vegan-tagged, perfect macros, pantry-matching."""
        assert is_useful(100, _vegan_persona(), _make_recipes())

    def test_recipe_101_chicken_fails_diet(self):
        assert not is_useful(101, _vegan_persona(), _make_recipes())

    def test_recipe_103_fails_macros(self):
        """103 is vegan but 1200 kcal way off the 500 target."""
        assert not is_useful(103, _vegan_persona(), _make_recipes())

    def test_recipe_not_in_index_returns_false(self):
        assert not is_useful(99999, _vegan_persona(), _make_recipes())

    def test_pantry_threshold_respected(self):
        """Recipe 100 has 3 ingredients all in pantry → 0 missing. Passes."""
        assert is_useful(100, _vegan_persona(), _make_recipes(), pantry_missing_threshold=0)
        # If we remove an item from the pantry, recipe 100 has 1 missing — still passes with default 3
        persona_minus = dict(_vegan_persona())
        persona_minus["pantry"] = ["tofu"]  # only one matches
        assert is_useful(100, persona_minus, _make_recipes(), pantry_missing_threshold=3)
        # With threshold=0, this fails
        assert not is_useful(100, persona_minus, _make_recipes(), pantry_missing_threshold=0)


# ============================================================
# useful_recall_at_k
# ============================================================

class TestUsefulRecallAtK:
    def test_hit_at_top_with_useful_recipe(self):
        recipes = _make_recipes()
        recs = [100, 101, 102]
        score = useful_recall_at_k(recs, relevant={100}, persona=_vegan_persona(),
                                   recipes_df=recipes, k=3)
        assert score == 1.0

    def test_hit_at_top_but_not_useful_scores_zero(self):
        """Even if the held-out item is in top-K, if it fails constraints it's not a useful hit."""
        recipes = _make_recipes()
        # Recipe 101 (chicken) is in top-1 but fails vegan diet
        score = useful_recall_at_k([101, 100], relevant={101},
                                   persona=_vegan_persona(),
                                   recipes_df=recipes, k=2)
        assert score == 0.0

    def test_miss_returns_zero(self):
        recipes = _make_recipes()
        score = useful_recall_at_k([102, 103], relevant={100},
                                   persona=_vegan_persona(),
                                   recipes_df=recipes, k=2)
        assert score == 0.0

    def test_k_cutoff_respected(self):
        """Held-out item is in top-3 but not top-2."""
        recipes = _make_recipes()
        recs = [102, 103, 100]  # 100 at rank 3
        score_k2 = useful_recall_at_k(recs, {100}, _vegan_persona(), recipes, k=2)
        score_k3 = useful_recall_at_k(recs, {100}, _vegan_persona(), recipes, k=3)
        assert score_k2 == 0.0
        assert score_k3 == 1.0

    def test_empty_relevant_returns_zero(self):
        recipes = _make_recipes()
        score = useful_recall_at_k([100, 101], relevant=set(),
                                   persona=_vegan_persona(),
                                   recipes_df=recipes, k=10)
        assert score == 0.0
