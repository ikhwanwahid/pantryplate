"""Stage 2 reranker — soft-constraint ranking of Stage-1-eligible candidates.

Formula:

    final(u, r) = αₜ·s_taste(u,r) + αₚ·s_pantry(u,r) + αₙ·s_nutrition(u,r)

    αₜ + αₚ + αₙ = 1   (the simplex constraint — the deck's X-factor)

Where:
    s_taste     — predicted relevance from Stage 1, normalized to [0, 1]
    s_pantry    — non-staple ingredient overlap with user's pantry, [0, 1]
    s_nutrition — Gaussian proximity to user's macro targets, [0, 1]

The (αₜ, αₚ, αₙ) triplet is what gets swept in the headline experiment
to characterize the trade-off between taste, pantry feasibility, and
nutritional match.

**Note on diet** (architectural decision §8): diet is a HARD constraint
applied at Stage 1's exit via `filter_by_diet`, NOT a multiplicative term
in the Stage 2 formula. Stage 2 assumes its input candidates are already
diet-compliant. We still COMPUTE s_diet per recipe for visibility (a "✓ diet OK"
badge in the demo), but it does not affect the final score.

If a non-compliant recipe somehow reaches Stage 2 (e.g., caller skipped
the filter), it will still be scored and ranked normally — Stage 2 trusts
its input. Defense-in-depth lives at the caller layer, not here.

Convention:
    .rerank(persona, candidates, taste_scores, recipes_df, k=10)
        → list[recipe_id]  (top-K by final score, ties broken by stable sort)
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.reranker.scores import (
    diet_score,
    get_staples_for_persona,
    missing_count,
    nutrition_score,
    pantry_score,
    pantry_score_with_expiry,
)


class Stage2Reranker:
    """Multi-constraint reranker.

    Parameters
    ----------
    alpha_taste, alpha_pantry, alpha_nutrition : float
        Constraint weights. Must satisfy αₜ + αₚ + αₙ = 1 (validated at init
        with tolerance 1e-6). Default (0.5, 0.3, 0.2) is a reasonable demo
        baseline; the headline experiment sweeps these.
    nutrition_tolerance : float
        σ for the Gaussian nutrition score (see scores.nutrition_score).
        Default 0.2 = "±20% counts as on-target", matching the deck's
        Useful Recall threshold.
    """

    def __init__(
        self,
        alpha_taste: float = 0.5,
        alpha_pantry: float = 0.3,
        alpha_nutrition: float = 0.2,
        nutrition_tolerance: float = 0.2,
    ):
        if not all(0.0 <= a <= 1.0 for a in (alpha_taste, alpha_pantry, alpha_nutrition)):
            raise ValueError(
                f"alphas must be in [0, 1]; got "
                f"taste={alpha_taste}, pantry={alpha_pantry}, nutrition={alpha_nutrition}"
            )
        total = alpha_taste + alpha_pantry + alpha_nutrition
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"alphas must sum to 1.0 (the simplex constraint), got {total}"
            )
        if nutrition_tolerance <= 0:
            raise ValueError(f"nutrition_tolerance must be > 0, got {nutrition_tolerance}")

        self.alpha_taste = alpha_taste
        self.alpha_pantry = alpha_pantry
        self.alpha_nutrition = alpha_nutrition
        self.nutrition_tolerance = nutrition_tolerance

    # ---------------------------------------------------------------------
    # Public scoring API
    # ---------------------------------------------------------------------
    def score_candidates(
        self,
        persona: dict,
        candidates: Iterable[int],
        taste_scores: dict[int, float],
        recipes_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute the four constraint scores + the final blended score per candidate.

        Returns a DataFrame with one row per candidate, columns:
            recipe_id, s_taste, s_pantry, s_nutrition, s_diet, final

        Useful for the demo notebook so we can see how each score
        contributes per candidate.

        Parameters
        ----------
        persona : dict
            Persona JSON. Must have "pantry", "macro_targets", "restrictions".
            "exclude_from_staples" is optional.
        candidates : iterable of recipe_id
            Stage 1 candidate pool (typically top-100).
        taste_scores : dict[recipe_id -> float]
            Stage 1 score per candidate. Will be min-max normalized to [0, 1]
            within this candidate set before blending.
        recipes_df : pd.DataFrame
            Must be indexed by recipe_id (or have an "id" column) and contain
            "ingredients_parsed", "tags_parsed", "nutrition_parsed".
        """
        cand_list = [int(c) for c in candidates]
        if not cand_list:
            return pd.DataFrame(columns=["recipe_id", "s_taste", "s_pantry",
                                         "s_nutrition", "s_diet", "final"])

        recipes = self._ensure_indexed(recipes_df).reindex(cand_list)

        user_pantry = set(persona.get("pantry", []))
        macro_targets = persona.get("macro_targets", {})
        restrictions = persona.get("restrictions", [])
        staples = get_staples_for_persona(persona)
        pantry_expiry = persona.get("pantry_expiry", {})

        # Pull Stage 1 taste scores into a numpy array aligned to cand_list
        raw_taste = np.array(
            [float(taste_scores.get(int(c), 0.0)) for c in cand_list],
            dtype=np.float64,
        )
        s_taste_arr = _minmax_norm(raw_taste)

        s_pantry_arr: list[float] = []
        s_nutrition_arr: list[float] = []
        s_diet_arr: list[int] = []
        for rid, row in zip(cand_list, recipes.itertuples(index=False)):
            ings = getattr(row, "ingredients_parsed", None) or []
            tags = getattr(row, "tags_parsed", None) or []
            nutrition = getattr(row, "nutrition_parsed", None) or {}
            s_pantry_arr.append(pantry_score_with_expiry(ings, user_pantry, pantry_expiry=pantry_expiry, staples=staples))
            s_nutrition_arr.append(
                nutrition_score(nutrition, macro_targets, tolerance=self.nutrition_tolerance)
            )
            s_diet_arr.append(diet_score(ings, tags, restrictions))

        s_pantry = np.array(s_pantry_arr, dtype=np.float64)
        s_nutrition = np.array(s_nutrition_arr, dtype=np.float64)
        s_diet = np.array(s_diet_arr, dtype=np.int64)

        # final = αₜ·s_taste + αₚ·s_pantry + αₙ·s_nutrition
        # Diet is NOT multiplicative here — it's a Stage 1 filter (see module
        # docstring + docs/data_decisions.md §8). s_diet is still reported per
        # recipe for visibility in the demo (the "✓ diet OK" badge) but does
        # not affect the score.
        final = (
            self.alpha_taste * s_taste_arr
            + self.alpha_pantry * s_pantry
            + self.alpha_nutrition * s_nutrition
        )

        return pd.DataFrame({
            "recipe_id":   cand_list,
            "s_taste":     s_taste_arr,
            "s_pantry":    s_pantry,
            "s_nutrition": s_nutrition,
            "s_diet":      s_diet,
            "final":       final,
        })

    def rerank(
        self,
        persona: dict,
        candidates: Iterable[int],
        taste_scores: dict[int, float],
        recipes_df: pd.DataFrame,
        k: int = 10,
        return_scores: bool = False,
    ) -> list[int] | pd.DataFrame:
        """Return the top-K candidates after constraint reranking.

        Parameters
        ----------
        k : int
            Number of items to return.
        return_scores : bool
            If True, return the full scored DataFrame (sorted desc by `final`)
            instead of just the recipe-id list. Useful for the demo notebook
            so we can show why the top-K won.

        Notes on ties: when multiple candidates score 0.0 (e.g., diet-filtered
        out), they keep their Stage 1 order via stable sort, which is what
        you want for "show the user something even if nothing perfectly fits".
        """
        scored = self.score_candidates(persona, candidates, taste_scores, recipes_df)
        if scored.empty:
            return scored if return_scores else []
        # Stable sort: ties → preserve original (Stage 1) order
        scored = scored.sort_values("final", ascending=False, kind="stable")
        top_k = scored.head(k)
        if return_scores:
            return top_k.reset_index(drop=True)
        return top_k["recipe_id"].astype(int).tolist()

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    @staticmethod
    def _ensure_indexed(recipes_df: pd.DataFrame) -> pd.DataFrame:
        """Return recipes_df indexed by recipe_id (int64). Idempotent."""
        if recipes_df.index.name == "recipe_id":
            return recipes_df
        if "id" in recipes_df.columns:
            return recipes_df.set_index(recipes_df["id"].astype(np.int64).rename("recipe_id"))
        if "recipe_id" in recipes_df.columns:
            return recipes_df.set_index("recipe_id")
        raise ValueError(
            "recipes_df must be indexed by recipe_id or have an 'id' / 'recipe_id' column"
        )


def _minmax_norm(x: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns zeros if the array is constant."""
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
