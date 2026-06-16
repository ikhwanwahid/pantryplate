"""Stage 2 α-sweep — characterize the (αₜ, αₚ, αₙ) constraint simplex.

The project's headline experiment. Vary the Stage 2 weights over the simplex
(αₜ + αₚ + αₙ = 1) and measure how recommendation quality changes:

    final = αₜ·s_taste + αₚ·s_pantry + αₙ·s_nutrition     (diet filtered at Stage 1)

Two design choices that make this both rigorous and cheap:

1. **Derived constraints (real ground truth).** Each eval user's pantry and
   macro targets are derived from their OWN training history (pantry = the
   non-staple ingredients of their 4★+ recipes; macros = the mean macros of
   those recipes). So every user has a held-out item (ground truth for
   relevance) AND a self-consistent constraint profile (for Useful Recall).

2. **Precompute once, sweep cheap.** The Stage 1 candidate pool and each
   candidate's (s_taste, s_pantry, s_nutrition) are FIXED per user — only the
   α weights change. So we precompute per-user sub-scores once, then each
   simplex point is a re-weight + re-sort of ~100 numbers. A dense grid over
   the simplex costs seconds.

Metrics per simplex point (single held-out item per user → binary per user):
    Recall@K         = held-out item in top-K
    Useful Recall@K  = held-out item in top-K AND it satisfies the user's
                       derived constraints (missing ≤ threshold, macros ±tol)

Model-agnostic: `precompute_user_cases` takes any fitted Stage 1 model with
`.recommend(user_id, k, exclude_seen)`. Start with BPR (best warm pool);
swap in any other generator without changing the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.reranker.combiner import _minmax_norm
from src.reranker.scores import nutrition_score, pantry_score
from src.utils.staples import STAPLES, missing_count
from src.eval.useful_recall import is_macro_near


# Macro fields we derive targets for (keys must match recipe nutrition dicts)
MACRO_FIELDS = ("calories", "protein_pdv", "carbs_pdv", "fat_pdv", "sodium_pdv")


# ============================================================
# 1. Derive per-user constraints from training history
# ============================================================

def derive_user_constraints(
    train_df: pd.DataFrame,
    recipes_df: pd.DataFrame,
    user_ids,
    positive_threshold: int = 4,
) -> dict[int, dict]:
    """Per user, build {pantry, macro_targets} from their 4★+ training recipes.

    pantry        = set of non-staple ingredients across their liked recipes
                    ("what they cook with")
    macro_targets = mean of MACRO_FIELDS over their liked recipes
                    ("what they tend to eat")

    Parameters
    ----------
    train_df : interactions with user_id / recipe_id / rating
    recipes_df : indexed by recipe_id, with ingredients_parsed + nutrition_parsed
    user_ids : iterable of users to compute for (typically the eval sample)
    positive_threshold : rating ≥ this counts as a liked recipe

    Returns dict[user_id -> {"pantry": set[str], "macro_targets": dict,
                             "restrictions": []}]. Users with no positives get
    an empty pantry + empty macro_targets (they'll score 0 on those terms).
    """
    user_ids = set(int(u) for u in user_ids)
    pos = train_df[
        (train_df["rating"] >= positive_threshold)
        & (train_df["user_id"].isin(user_ids))
    ]

    out: dict[int, dict] = {int(u): {"pantry": set(), "macro_targets": {}, "restrictions": []}
                            for u in user_ids}

    for uid, grp in pos.groupby("user_id"):
        rids = [int(r) for r in grp["recipe_id"].to_numpy() if int(r) in recipes_df.index]
        if not rids:
            continue
        rows = recipes_df.loc[rids]

        # pantry: union of non-staple ingredients
        pantry: set[str] = set()
        for ings in rows["ingredients_parsed"]:
            if isinstance(ings, list):
                pantry.update(i for i in ings if isinstance(i, str) and i not in STAPLES)

        # macro targets: mean of each field across liked recipes
        sums = {f: 0.0 for f in MACRO_FIELDS}
        n = 0
        for nut in rows["nutrition_parsed"]:
            if isinstance(nut, dict):
                n += 1
                for f in MACRO_FIELDS:
                    v = nut.get(f)
                    if v is not None:
                        sums[f] += float(v)
        macro_targets = {f: sums[f] / n for f in MACRO_FIELDS} if n else {}

        out[int(uid)] = {"pantry": pantry, "macro_targets": macro_targets, "restrictions": []}

    return out


# ============================================================
# 2. Precompute per-user candidate sub-scores (the one-time cost)
# ============================================================

@dataclass
class UserCase:
    """Cached per-user state — everything the sweep needs, α-independent."""
    user_id: int
    candidate_ids: np.ndarray   # (n_pool,) recipe ids, Stage-1 order
    s_taste: np.ndarray         # (n_pool,) min-max normalized taste score
    s_pantry: np.ndarray        # (n_pool,) pantry overlap in [0,1]
    s_nutrition: np.ndarray     # (n_pool,) macro proximity in [0,1]
    holdout_id: int
    holdout_in_pool: bool
    holdout_useful: bool        # held-out item satisfies the user's constraints


def precompute_user_cases(
    model,
    holdout_map: dict[int, int],
    recipes_df: pd.DataFrame,
    constraints: dict[int, dict],
    k_pool: int = 100,
    macro_tolerance: float = 0.2,
    pantry_missing_threshold: int = 3,
) -> list[UserCase]:
    """Run Stage 1 + score each candidate's three sub-scores, once per user.

    Parameters
    ----------
    model : fitted Stage 1 model with .recommend(user_id, k, exclude_seen)
    holdout_map : {user_id -> held-out recipe_id} (the LOO ground truth)
    recipes_df : indexed by recipe_id, with ingredients_parsed + nutrition_parsed
    constraints : output of derive_user_constraints
    k_pool : Stage 1 candidate pool size (top-N handed to Stage 2)
    macro_tolerance, pantry_missing_threshold : Useful Recall thresholds

    Returns a list of UserCase (skips users with no derived constraints).
    """
    cases: list[UserCase] = []

    for uid, holdout_id in holdout_map.items():
        uid = int(uid)
        prof = constraints.get(uid)
        if prof is None:
            continue
        pantry = prof["pantry"]
        macro_targets = prof["macro_targets"]

        cand = [int(c) for c in model.recommend(uid, k=k_pool, exclude_seen=True)]
        if not cand:
            continue
        cand = [c for c in cand if c in recipes_df.index]
        if not cand:
            continue

        rows = recipes_df.loc[cand]
        taste_raw = np.array([1.0 / (r + 1) for r in range(len(cand))], dtype=np.float64)
        s_taste = _minmax_norm(taste_raw)

        s_pantry = np.empty(len(cand), dtype=np.float64)
        s_nutrition = np.empty(len(cand), dtype=np.float64)
        for i, (_, row) in enumerate(rows.iterrows()):
            ings = row.get("ingredients_parsed") or []
            nut = row.get("nutrition_parsed") or {}
            s_pantry[i] = pantry_score(ings, pantry)
            s_nutrition[i] = nutrition_score(nut, macro_targets, tolerance=macro_tolerance)

        holdout_id = int(holdout_id)
        in_pool = holdout_id in set(cand)

        # held-out item's usefulness (α-independent)
        useful = False
        if holdout_id in recipes_df.index:
            hrow = recipes_df.loc[holdout_id]
            h_ings = hrow.get("ingredients_parsed") or []
            h_nut = hrow.get("nutrition_parsed") or {}
            feasible = missing_count(h_ings, pantry) <= pantry_missing_threshold
            near = is_macro_near(h_nut, macro_targets, tolerance=macro_tolerance) if macro_targets else True
            useful = bool(feasible and near)

        cases.append(UserCase(
            user_id=uid,
            candidate_ids=np.array(cand, dtype=np.int64),
            s_taste=s_taste.astype(np.float64),
            s_pantry=s_pantry,
            s_nutrition=s_nutrition,
            holdout_id=holdout_id,
            holdout_in_pool=in_pool,
            holdout_useful=useful,
        ))

    return cases


# ============================================================
# 3. The sweep
# ============================================================

def simplex_grid(step: float = 0.05) -> list[tuple[float, float, float]]:
    """All (αₜ, αₚ, αₙ) on the simplex with the given step (each a multiple of step).

    step=0.05 → n=20 → 231 points. step=0.1 → 66 points.
    """
    n = round(1.0 / step)
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((i / n, j / n, k / n))
    return pts


def _hit_at_k(case: UserCase, weights: tuple[float, float, float], k: int) -> bool:
    """Is the held-out item in the top-K of this user's pool at these weights?"""
    if not case.holdout_in_pool:
        return False
    at, ap, an = weights
    final = at * case.s_taste + ap * case.s_pantry + an * case.s_nutrition
    k_eff = min(k, final.size)
    top_idx = np.argpartition(-final, k_eff - 1)[:k_eff]
    top_ids = case.candidate_ids[top_idx]
    return case.holdout_id in top_ids


def sweep(
    cases: list[UserCase],
    k: int = 10,
    step: float = 0.05,
) -> pd.DataFrame:
    """Run the α-sweep; return a DataFrame with one row per simplex point.

    Columns: alpha_taste, alpha_pantry, alpha_nutrition, recall@k, useful_recall@k.
    Both metrics are means over the user cases.
    """
    grid = simplex_grid(step)
    n_users = len(cases)
    records = []
    for (at, ap, an) in grid:
        recall_hits = 0
        useful_hits = 0
        for case in cases:
            hit = _hit_at_k(case, (at, ap, an), k)
            if hit:
                recall_hits += 1
                if case.holdout_useful:
                    useful_hits += 1
        records.append({
            "alpha_taste": at,
            "alpha_pantry": ap,
            "alpha_nutrition": an,
            f"recall@{k}": recall_hits / n_users if n_users else 0.0,
            f"useful_recall@{k}": useful_hits / n_users if n_users else 0.0,
        })
    return pd.DataFrame(records)


def best_point(sweep_df: pd.DataFrame, metric: str) -> pd.Series:
    """Return the simplex point (row) maximizing the given metric column."""
    return sweep_df.loc[sweep_df[metric].idxmax()]
