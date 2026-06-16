"""Tests for src/eval/alpha_sweep.py.

All synthetic — no real models or data. Uses a stub Stage 1 model and a tiny
recipes frame to exercise constraint derivation, precompute, and the sweep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.alpha_sweep import (
    MACRO_FIELDS,
    UserCase,
    best_point,
    derive_user_constraints,
    per_user_metrics,
    precompute_user_cases,
    simplex_grid,
    sweep,
)


# ============================================================
# simplex_grid
# ============================================================

class TestSimplexGrid:
    def test_points_sum_to_one(self):
        for at, ap, an in simplex_grid(step=0.1):
            assert at + ap + an == pytest.approx(1.0)

    def test_count_for_step(self):
        # n=10 → (n+1)(n+2)/2 = 66 points
        assert len(simplex_grid(step=0.1)) == 66
        # n=20 → 231
        assert len(simplex_grid(step=0.05)) == 231

    def test_corners_present(self):
        pts = simplex_grid(step=0.25)
        assert (1.0, 0.0, 0.0) in pts
        assert (0.0, 1.0, 0.0) in pts
        assert (0.0, 0.0, 1.0) in pts


# ============================================================
# derive_user_constraints
# ============================================================

def _recipes() -> pd.DataFrame:
    return pd.DataFrame({
        "ingredients_parsed": [
            ["chicken", "rice", "salt"],          # 10
            ["chicken", "broccoli", "garlic"],    # 11
            ["tofu", "soy sauce"],                # 12
            ["beef", "potato"],                   # 13
        ],
        "nutrition_parsed": [
            {"calories": 500, "protein_pdv": 50, "carbs_pdv": 30, "fat_pdv": 20, "sodium_pdv": 25},
            {"calories": 600, "protein_pdv": 60, "carbs_pdv": 20, "fat_pdv": 25, "sodium_pdv": 30},
            {"calories": 400, "protein_pdv": 40, "carbs_pdv": 35, "fat_pdv": 15, "sodium_pdv": 20},
            {"calories": 800, "protein_pdv": 30, "carbs_pdv": 50, "fat_pdv": 40, "sodium_pdv": 60},
        ],
    }, index=pd.Index([10, 11, 12, 13], name="recipe_id"))


def _train() -> pd.DataFrame:
    # user 1 likes recipes 10 + 11 (both 5★); recipe 13 is a 2★ (not liked)
    return pd.DataFrame({
        "user_id":   [1, 1, 1, 2],
        "recipe_id": [10, 11, 13, 12],
        "rating":    [5, 5, 2, 5],
    })


class TestDeriveConstraints:
    def test_pantry_excludes_staples_and_disliked(self):
        c = derive_user_constraints(_train(), _recipes(), user_ids=[1])
        pantry = c[1]["pantry"]
        # from recipes 10 + 11 (liked), minus staples (salt, garlic)
        assert "chicken" in pantry
        assert "rice" in pantry
        assert "broccoli" in pantry
        assert "salt" not in pantry      # staple
        assert "garlic" not in pantry    # staple
        assert "potato" not in pantry    # from disliked recipe 13

    def test_macro_targets_are_means(self):
        c = derive_user_constraints(_train(), _recipes(), user_ids=[1])
        mt = c[1]["macro_targets"]
        # mean of recipes 10 + 11
        assert mt["calories"] == pytest.approx(550.0)
        assert mt["protein_pdv"] == pytest.approx(55.0)
        assert set(mt.keys()) == set(MACRO_FIELDS)

    def test_user_with_no_positives_gets_empty(self):
        c = derive_user_constraints(_train(), _recipes(), user_ids=[999])
        assert c[999]["pantry"] == set()
        assert c[999]["macro_targets"] == {}


# ============================================================
# precompute + sweep with a stub model
# ============================================================

class _StubModel:
    """Returns a fixed candidate list per user (ignores k beyond slicing)."""
    def __init__(self, recs_by_user):
        self._recs = recs_by_user

    def recommend(self, user_id, k=10, exclude_seen=True):
        return self._recs.get(int(user_id), [])[:k]


def test_precompute_and_sweep_end_to_end():
    recipes = _recipes()
    constraints = derive_user_constraints(_train(), recipes, user_ids=[1])
    # user 1's pool: held-out is recipe 11 (in pool); 10,12,13 also present
    model = _StubModel({1: [11, 10, 12, 13]})
    holdout = {1: 11}

    cases = precompute_user_cases(model, holdout, recipes, constraints, k_pool=4)
    assert len(cases) == 1
    case = cases[0]
    assert case.holdout_in_pool is True
    assert case.s_taste.shape == (4,)
    # taste is min-max normalized → max 1.0 (rank 0), min 0.0 (last)
    assert case.s_taste.max() == pytest.approx(1.0)
    assert case.s_taste.min() == pytest.approx(0.0)

    df = sweep(cases, k=1, step=0.5)
    # at k=1, recall is 1 only when held-out (11) is the single top item
    expected_cols = {"alpha_taste", "alpha_pantry", "alpha_nutrition",
                     "recall@1", "useful_recall@1", "useful_rate@1",
                     "feasible_rate@1", "near_rate@1"}
    assert expected_cols <= set(df.columns)
    assert ((df["recall@1"] >= 0) & (df["recall@1"] <= 1)).all()
    # useful_recall <= recall everywhere (subset condition)
    assert (df["useful_recall@1"] <= df["recall@1"] + 1e-9).all()
    # dense rates are valid fractions; useful_rate ⊆ feasible_rate and ⊆ near_rate
    for col in ("useful_rate@1", "feasible_rate@1", "near_rate@1"):
        assert ((df[col] >= 0) & (df[col] <= 1)).all()
    assert (df["useful_rate@1"] <= df["feasible_rate@1"] + 1e-9).all()
    assert (df["useful_rate@1"] <= df["near_rate@1"] + 1e-9).all()


def test_holdout_not_in_pool_never_recalled():
    recipes = _recipes()
    constraints = derive_user_constraints(_train(), recipes, user_ids=[1])
    # held-out 99 not in pool → recall 0 at every alpha
    model = _StubModel({1: [10, 12, 13]})
    cases = precompute_user_cases(model, {1: 99}, recipes, constraints, k_pool=4)
    assert cases[0].holdout_in_pool is False
    df = sweep(cases, k=3, step=0.5)
    assert (df["recall@3"] == 0.0).all()


def test_taste_corner_recovers_stage1_top1():
    """At α=(1,0,0), top-1 should be the Stage 1 rank-0 item."""
    recipes = _recipes()
    constraints = derive_user_constraints(_train(), recipes, user_ids=[1])
    # Stage 1 puts recipe 10 first; held-out = 10 → recall@1 = 1 at taste corner
    model = _StubModel({1: [10, 11, 12, 13]})
    cases = precompute_user_cases(model, {1: 10}, recipes, constraints, k_pool=4)
    df = sweep(cases, k=1, step=1.0)  # only the 3 corners
    taste_corner = df[(df["alpha_taste"] == 1.0)].iloc[0]
    assert taste_corner["recall@1"] == 1.0


def test_per_user_metrics_shape_and_pairing():
    """per_user_metrics returns one aligned row per case — the input to a paired test."""
    recipes = _recipes()
    constraints = derive_user_constraints(_train(), recipes, user_ids=[1, 2])
    model = _StubModel({1: [11, 10, 12, 13], 2: [12, 10, 11, 13]})
    cases = precompute_user_cases(model, {1: 11, 2: 12}, recipes, constraints, k_pool=4)
    a = per_user_metrics(cases, (1.0, 0.0, 0.0), k=2)
    b = per_user_metrics(cases, (0.0, 1.0, 0.0), k=2)
    # one row per case, same user order (paired)
    assert len(a) == len(b) == len(cases)
    assert list(a["user_id"]) == list(b["user_id"]) == [c.user_id for c in cases]
    for col in ("recall@2", "cookable_rate@2", "useful_rate@2"):
        assert col in a.columns
        assert ((a[col] >= 0) & (a[col] <= 1)).all()


def test_best_point_returns_max():
    df = pd.DataFrame({
        "alpha_taste": [1.0, 0.0, 0.0],
        "alpha_pantry": [0.0, 1.0, 0.0],
        "alpha_nutrition": [0.0, 0.0, 1.0],
        "useful_recall@10": [0.1, 0.3, 0.2],
    })
    best = best_point(df, "useful_recall@10")
    assert best["alpha_pantry"] == 1.0
    assert best["useful_recall@10"] == 0.3
