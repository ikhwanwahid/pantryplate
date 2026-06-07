"""Tests for src/reranker/combiner.py — Stage2Reranker."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reranker.combiner import Stage2Reranker, _minmax_norm


# ============================================================
# Helpers — tiny recipes_df + persona for testing
# ============================================================

def _make_recipes() -> pd.DataFrame:
    """5 recipes spanning constraint variations.

    id | ingredients               | tags         | nutrition  | notes
    100| tofu, broccoli, soy sauce | vegan, asian | 500/50     | vegan, perfect macros
    101| chicken, rice             | main-dish    | 500/50     | meat, perfect macros
    102| tofu, almonds, broccoli   | vegan        | 500/50     | vegan but has nuts
    103| tofu, broccoli            | vegan        | 1200/10    | vegan, way over kcal
    104| pasta, tomato             | italian      | 500/50     | none, perfect macros (no restrictions)
    """
    return pd.DataFrame({
        "id": [100, 101, 102, 103, 104],
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
    })


def _vegan_persona():
    return {
        "id": "test_vegan",
        "pantry": ["tofu", "broccoli"],
        "macro_targets": {"calories": 500, "protein_pdv": 50},
        "restrictions": ["vegan"],
        "exclude_from_staples": [],
    }


def _no_restriction_persona():
    return {
        "id": "test_any",
        "pantry": ["chicken breast", "rice"],
        "macro_targets": {"calories": 500},
        "restrictions": [],
    }


# ============================================================
# Initialization
# ============================================================

class TestInit:
    def test_default_alphas_sum_to_one(self):
        r = Stage2Reranker()
        assert r.alpha_taste + r.alpha_pantry + r.alpha_nutrition == pytest.approx(1.0)

    def test_alphas_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="simplex"):
            Stage2Reranker(alpha_taste=0.5, alpha_pantry=0.5, alpha_nutrition=0.5)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Stage2Reranker(alpha_taste=-0.1, alpha_pantry=0.6, alpha_nutrition=0.5)

    def test_zero_tolerance_raises(self):
        with pytest.raises(ValueError, match="nutrition_tolerance"):
            Stage2Reranker(nutrition_tolerance=0)


# ============================================================
# score_candidates
# ============================================================

class TestScoreCandidates:
    def test_score_columns_present_and_in_range(self):
        recipes = _make_recipes()
        persona = _vegan_persona()
        r = Stage2Reranker()
        taste = {100: 1.0, 101: 0.9, 102: 0.5, 103: 0.3, 104: 0.1}
        scored = r.score_candidates(persona, [100, 101, 102, 103, 104], taste, recipes)
        assert list(scored.columns) == ["recipe_id", "s_taste", "s_pantry",
                                        "s_nutrition", "s_diet", "final"]
        assert ((scored["s_taste"] >= 0) & (scored["s_taste"] <= 1)).all()
        assert ((scored["s_pantry"] >= 0) & (scored["s_pantry"] <= 1)).all()
        assert ((scored["s_nutrition"] >= 0) & (scored["s_nutrition"] <= 1)).all()
        assert scored["s_diet"].isin([0, 1]).all()

    def test_diet_filter_zeros_out_non_vegan(self):
        """Chicken recipe (101) should have s_diet=0 → final=0 for a vegan persona."""
        recipes = _make_recipes()
        r = Stage2Reranker()
        taste = {100: 1.0, 101: 1.0, 102: 1.0, 103: 1.0, 104: 1.0}
        scored = r.score_candidates(_vegan_persona(), [100, 101, 102, 103, 104], taste, recipes)
        # Recipes 100, 102, 103 are all vegan-tagged with no animal-product ingredients.
        # 101 (chicken) and 104 (pasta — no vegan tag) are not vegan-compliant.
        diet_by_id = dict(zip(scored["recipe_id"], scored["s_diet"]))
        assert diet_by_id[100] == 1
        assert diet_by_id[101] == 0
        assert diet_by_id[102] == 1  # vegan tag + tofu/almonds/broccoli are all vegan
        assert diet_by_id[103] == 1
        assert diet_by_id[104] == 0  # pasta — no vegan tag

    def test_nutrition_score_penalizes_outliers(self):
        """Recipe 103 (1200 kcal, way off target 500) scores lower than recipe 100."""
        recipes = _make_recipes()
        r = Stage2Reranker()
        taste = {100: 1.0, 103: 1.0}
        scored = r.score_candidates(_vegan_persona(), [100, 103], taste, recipes)
        nut = dict(zip(scored["recipe_id"], scored["s_nutrition"]))
        assert nut[100] > nut[103]

    def test_pantry_score_rewards_matching_ingredients(self):
        """Vegan persona has tofu+broccoli; recipe 100 (tofu+broccoli+soy) > recipe 104 (pasta+tomato)."""
        recipes = _make_recipes()
        r = Stage2Reranker()
        taste = {100: 1.0, 104: 1.0}
        scored = r.score_candidates(_vegan_persona(), [100, 104], taste, recipes)
        pan = dict(zip(scored["recipe_id"], scored["s_pantry"]))
        assert pan[100] > pan[104]

    def test_taste_scores_are_minmax_normalized(self):
        recipes = _make_recipes()
        r = Stage2Reranker()
        raw_taste = {100: 100.0, 101: 50.0, 102: 0.0}
        scored = r.score_candidates(_no_restriction_persona(), [100, 101, 102], raw_taste, recipes)
        taste_by_id = dict(zip(scored["recipe_id"], scored["s_taste"]))
        # Max-taste should be 1.0, min should be 0.0
        assert taste_by_id[100] == pytest.approx(1.0)
        assert taste_by_id[102] == pytest.approx(0.0)
        assert taste_by_id[101] == pytest.approx(0.5)

    def test_empty_candidates_returns_empty_frame(self):
        r = Stage2Reranker()
        scored = r.score_candidates(_vegan_persona(), [], {}, _make_recipes())
        assert scored.empty
        assert list(scored.columns) == ["recipe_id", "s_taste", "s_pantry",
                                        "s_nutrition", "s_diet", "final"]


# ============================================================
# rerank — top-K ordering
# ============================================================

class TestRerank:
    def test_alpha_taste_only_reproduces_stage1_order(self):
        """alpha_taste=1.0: rerank should just sort by Stage 1 score (modulo diet filter)."""
        recipes = _make_recipes()
        r = Stage2Reranker(alpha_taste=1.0, alpha_pantry=0.0, alpha_nutrition=0.0)
        # No restrictions persona so diet doesn't filter
        taste = {100: 1.0, 101: 0.9, 102: 0.5, 103: 0.3, 104: 0.7}
        top = r.rerank(_no_restriction_persona(), [100, 101, 102, 103, 104], taste, recipes, k=5)
        # Expected order by raw taste: 100, 101, 104, 102, 103
        assert top == [100, 101, 104, 102, 103]

    def test_diet_filter_pushes_to_bottom(self):
        """Vegan persona: chicken recipe (101) should never rank above vegan recipes
        (because s_diet=0 → final=0)."""
        recipes = _make_recipes()
        r = Stage2Reranker(alpha_taste=1.0, alpha_pantry=0.0, alpha_nutrition=0.0)
        # 101 (chicken) has highest raw taste; but diet should zero its final.
        # Give 100 and 102 distinct taste values so we can verify the diet filter
        # without running into ties.
        taste = {100: 0.5, 101: 1.0, 102: 0.7}
        top = r.rerank(_vegan_persona(), [100, 101, 102], taste, recipes, k=3)
        # 100 and 102 are vegan-compliant; 101 is not. After diet × s_taste, 101's
        # final is 0 and the two vegan recipes have positive scores.
        assert top.index(101) > top.index(100)
        assert top.index(101) > top.index(102)

    def test_return_scores_includes_full_dataframe(self):
        recipes = _make_recipes()
        r = Stage2Reranker()
        taste = {100: 1.0, 102: 0.5}
        top = r.rerank(_vegan_persona(), [100, 102], taste, recipes, k=2, return_scores=True)
        assert isinstance(top, pd.DataFrame)
        assert set(top.columns) == {"recipe_id", "s_taste", "s_pantry",
                                     "s_nutrition", "s_diet", "final"}
        # Already sorted by final desc
        assert (top["final"].diff().dropna() <= 0).all()

    def test_k_larger_than_candidates_returns_all(self):
        recipes = _make_recipes()
        r = Stage2Reranker()
        top = r.rerank(_no_restriction_persona(), [100, 101], {100: 0.5, 101: 0.5}, recipes, k=10)
        assert len(top) == 2


# ============================================================
# _minmax_norm
# ============================================================

class TestMinMaxNorm:
    def test_basic_normalization(self):
        x = np.array([1.0, 3.0, 5.0])
        out = _minmax_norm(x)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])

    def test_constant_input_returns_zeros(self):
        x = np.array([2.0, 2.0, 2.0])
        np.testing.assert_array_equal(_minmax_norm(x), [0.0, 0.0, 0.0])

    def test_empty_returns_empty(self):
        out = _minmax_norm(np.array([]))
        assert out.size == 0
