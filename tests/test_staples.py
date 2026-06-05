"""Unit tests for src/utils/staples.py."""

from __future__ import annotations

import pytest

from src.utils.staples import (
    STAPLES,
    DAIRY_AND_EGGS,
    get_staples_for_persona,
    pantry_score,
    missing_count,
)


class TestStaplesSet:
    def test_includes_obvious_basics(self):
        for item in ["salt", "water", "olive oil", "flour", "egg", "garlic"]:
            assert item in STAPLES, f"{item!r} should be in STAPLES"

    def test_dairy_eggs_is_subset(self):
        assert DAIRY_AND_EGGS.issubset(STAPLES)

    def test_includes_canonicalization_variants(self):
        # ingr_map.pkl canonicalizes "flour" to "flmy"; both should match
        assert "flour" in STAPLES
        assert "flmy" in STAPLES


class TestGetStaplesForPersona:
    def test_no_persona_returns_full(self):
        assert get_staples_for_persona(None) == STAPLES

    def test_empty_persona_returns_full(self):
        assert get_staples_for_persona({}) == STAPLES

    def test_vegan_drops_dairy_eggs(self):
        out = get_staples_for_persona({"restrictions": ["vegan"]})
        for item in DAIRY_AND_EGGS:
            assert item not in out
        # Other staples preserved
        assert "salt" in out and "olive oil" in out

    def test_custom_exclusions_respected(self):
        out = get_staples_for_persona({"exclude_from_staples": ["salt", "sugar"]})
        assert "salt" not in out
        assert "sugar" not in out
        assert "water" in out  # not excluded

    def test_vegan_and_custom_combine(self):
        out = get_staples_for_persona({
            "restrictions": ["vegan"],
            "exclude_from_staples": ["flour"],
        })
        assert "flour" not in out
        for item in DAIRY_AND_EGGS:
            assert item not in out


class TestPantryScore:
    def test_only_staples_returns_one(self):
        # Recipe contains only staples → nothing to "match", always achievable
        assert pantry_score({"salt", "water", "olive oil"}, set()) == 1.0

    def test_no_overlap_returns_zero(self):
        # 2 non-staple ingredients, 0 in pantry
        assert pantry_score({"chicken", "rice"}, set()) == 0.0

    def test_half_overlap_after_removing_staples(self):
        # 2 non-staples (chicken, rice); 1 in pantry (chicken)
        assert pantry_score({"salt", "chicken", "rice"}, {"chicken"}) == 0.5

    def test_full_overlap_after_staples(self):
        # Only non-staple is chicken; user has it
        assert pantry_score({"salt", "chicken"}, {"chicken"}) == 1.0

    def test_empty_recipe(self):
        assert pantry_score(set(), {"chicken"}) == 0.0

    def test_accepts_lists_too(self):
        # Should work with lists, not just sets
        assert pantry_score(["salt", "chicken"], ["chicken"]) == 1.0

    def test_vegan_treats_eggs_as_non_staple(self):
        # With vegan staples, eggs become non-staple
        vegan_staples = get_staples_for_persona({"restrictions": ["vegan"]})
        # Recipe: egg + chicken. Both are non-staple now.
        # User pantry is empty → 0 / 2 = 0
        assert pantry_score({"egg", "chicken"}, set(), vegan_staples) == 0.0
        # Standard staples: only chicken is non-staple → 0 / 1 = 0 (same)
        assert pantry_score({"egg", "chicken"}, set()) == 0.0
        # But: recipe with just eggs → standard staples returns 1.0,
        # vegan returns 0.0
        assert pantry_score({"egg"}, set()) == 1.0
        assert pantry_score({"egg"}, set(), vegan_staples) == 0.0


class TestMissingCount:
    def test_zero_missing(self):
        assert missing_count({"salt", "chicken"}, {"chicken"}) == 0

    def test_one_missing(self):
        assert missing_count({"chicken", "rice"}, {"chicken"}) == 1

    def test_staples_never_counted_missing(self):
        # All staples, empty pantry → 0 missing (staples assumed)
        assert missing_count({"salt", "pepper", "water"}, set()) == 0

    def test_empty_recipe(self):
        assert missing_count(set(), {"chicken"}) == 0

    def test_accepts_lists(self):
        assert missing_count(["chicken", "rice", "salt"], ["chicken"]) == 1


class TestPersonaIntegration:
    """End-to-end: persona with restrictions → adjusted staples → score."""

    def test_vegan_recipe_with_eggs(self):
        persona = {"restrictions": ["vegan"], "pantry": ["chicken"]}
        staples = get_staples_for_persona(persona)
        # Recipe: salt + egg + tofu. For a vegan:
        #   non-staples = {egg, tofu} (egg removed from staples for vegan)
        #   pantry = {chicken}
        #   overlap = 0
        assert pantry_score({"salt", "egg", "tofu"}, persona["pantry"], staples) == 0.0
        assert missing_count({"salt", "egg", "tofu"}, persona["pantry"], staples) == 2

    def test_gluten_free_recipe_with_flour(self):
        persona = {
            "restrictions": ["gluten-free"],
            "exclude_from_staples": ["flour", "all-purpose flour", "flmy"],
            "pantry": ["rice flour"],
        }
        staples = get_staples_for_persona(persona)
        # Recipe with regular flour:
        #   non-staples = {flour, chicken} (flour now NOT a staple)
        #   pantry = {rice flour}
        #   overlap = 0
        assert pantry_score({"salt", "flour", "chicken"}, persona["pantry"], staples) == 0.0
        assert missing_count({"salt", "flour", "chicken"}, persona["pantry"], staples) == 2
