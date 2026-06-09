"""Tests for src/reranker/scores.py — nutrition_score and diet_score.

(pantry_score lives in src/utils/staples.py and is covered by test_staples.py.)
"""

from __future__ import annotations

import math

import pytest

from src.reranker.scores import (
    INGREDIENT_BLOCKLIST,
    TAG_FOR_RESTRICTION,
    diet_compliant,
    diet_score,
    nutrition_score,
)


# ============================================================
# nutrition_score
# ============================================================

class TestNutritionScore:
    targets = {"calories": 500, "protein_pdv": 50}

    def test_perfect_match_scores_1(self):
        recipe = {"calories": 500, "protein_pdv": 50}
        assert nutrition_score(recipe, self.targets) == pytest.approx(1.0)

    def test_at_tolerance_scores_approx_0_61(self):
        """At exactly tolerance deviation, Gaussian → exp(-1/2) ≈ 0.607."""
        recipe = {"calories": 600, "protein_pdv": 60}  # +20% on each
        score = nutrition_score(recipe, self.targets, tolerance=0.2)
        assert score == pytest.approx(math.exp(-0.5), abs=1e-6)

    def test_double_tolerance_scores_low(self):
        recipe = {"calories": 700, "protein_pdv": 70}  # +40% on each (2σ)
        score = nutrition_score(recipe, self.targets, tolerance=0.2)
        assert score == pytest.approx(math.exp(-2.0), abs=1e-6)

    def test_partial_field_coverage(self):
        """Recipe with only some target fields: only those are scored."""
        recipe = {"calories": 500}  # missing protein
        score = nutrition_score(recipe, self.targets)
        assert score == pytest.approx(1.0)

    def test_empty_targets_returns_zero(self):
        assert nutrition_score({"calories": 500}, {}) == 0.0

    def test_empty_recipe_returns_zero(self):
        assert nutrition_score({}, self.targets) == 0.0

    def test_invalid_tolerance_raises(self):
        with pytest.raises(ValueError, match="tolerance"):
            nutrition_score({"calories": 500}, {"calories": 500}, tolerance=0)

    def test_zero_target_field_is_skipped(self):
        """Avoids divide-by-zero on a degenerate persona."""
        recipe = {"calories": 500, "protein_pdv": 50}
        score = nutrition_score(recipe, {"calories": 0, "protein_pdv": 50})
        assert score == pytest.approx(1.0)

    def test_symmetric_around_target(self):
        """Going under or over by the same fraction should score identically."""
        under = nutrition_score({"calories": 400}, {"calories": 500})
        over  = nutrition_score({"calories": 600}, {"calories": 500})
        assert under == pytest.approx(over)


# ============================================================
# diet_compliant / diet_score
# ============================================================

class TestDietCompliant:
    def test_empty_restrictions_always_passes(self):
        assert diet_compliant(["chicken", "rice"], [], [])

    def test_vegan_recipe_with_tag_and_clean_ingredients(self):
        assert diet_compliant(
            ["tofu", "broccoli", "soy sauce"],
            ["vegan", "asian"],
            ["vegan"],
        )

    def test_vegan_recipe_missing_tag_is_rejected(self):
        """Tag check is required even if ingredients are clean."""
        assert not diet_compliant(
            ["tofu", "broccoli"],
            ["asian"],  # no vegan tag
            ["vegan"],
        )

    def test_vegan_tagged_but_contains_chicken_is_rejected(self):
        """Defense-in-depth: tag is noisy, ingredient blocklist catches violations."""
        assert not diet_compliant(
            ["chicken breast", "rice"],
            ["vegan"],
            ["vegan"],
        )

    def test_substring_match_catches_compound_ingredients(self):
        """'chicken broth' should fail vegan check via 'chicken' substring."""
        assert not diet_compliant(
            ["chicken broth", "rice"],
            ["vegan"],
            ["vegan"],
        )

    def test_unknown_restriction_does_not_fail_closed(self):
        """Unknown restrictions don't block recipes (we don't fail-closed)."""
        assert diet_compliant(["chicken"], ["main-dish"], ["made-up-restriction"])

    def test_multiple_restrictions_must_all_pass(self):
        assert diet_compliant(
            ["tofu", "rice"],
            ["vegan", "gluten-free"],
            ["vegan", "gluten-free"],
        )
        # If one fails (gluten-free violated by wheat noodles), recipe fails
        assert not diet_compliant(
            ["tofu", "wheat noodles"],
            ["vegan", "gluten-free"],
            ["vegan", "gluten-free"],
        )

    def test_nut_free_uses_blocklist_only(self):
        """nut-free has no reliable tag, so blocklist alone gates it."""
        # No tags required, no nuts → passes
        assert diet_compliant(["chicken", "rice"], ["main-dish"], ["nut-free"])
        # Almonds present → fails
        assert not diet_compliant(["chicken", "rice", "almonds"], ["main-dish"], ["nut-free"])

    def test_diet_score_returns_int(self):
        assert diet_score(["tofu"], ["vegan"], ["vegan"]) == 1
        assert diet_score(["chicken"], ["main-dish"], ["vegan"]) == 0


# ============================================================
# Sanity on the constant tables
# ============================================================

class TestConstants:
    def test_blocklist_keys_align_with_tags(self):
        """Every restriction with a tag should also have a blocklist (and vice versa,
        except restrictions like 'low-carb' / 'kosher' that use macro-derived rules)."""
        # nut-free is the deliberate "blocklist-only" entry
        assert TAG_FOR_RESTRICTION["nut-free"] is None
        assert "nut-free" in INGREDIENT_BLOCKLIST

    def test_vegan_blocklist_includes_dairy_and_eggs(self):
        for animal_product in ("milk", "butter", "cheese", "egg"):
            assert animal_product in INGREDIENT_BLOCKLIST["vegan"]

    def test_vegetarian_blocklist_does_not_include_dairy(self):
        """Vegetarian allows dairy & eggs — only meat is blocked."""
        for ok_for_veggie in ("milk", "butter", "egg", "cheese"):
            assert ok_for_veggie not in INGREDIENT_BLOCKLIST["vegetarian"]
