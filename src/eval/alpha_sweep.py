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
    cand_feasible: np.ndarray   # (n_pool,) bool — candidate pantry-feasible (missing ≤ thresh)
    cand_near: np.ndarray       # (n_pool,) bool — candidate macros within ±tol
    cand_useful: np.ndarray     # (n_pool,) bool — feasible AND near
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
        cand_feasible = np.empty(len(cand), dtype=bool)
        cand_near = np.empty(len(cand), dtype=bool)
        has_macros = bool(macro_targets)
        for i, (_, row) in enumerate(rows.iterrows()):
            ings = row.get("ingredients_parsed") or []
            nut = row.get("nutrition_parsed") or {}
            s_pantry[i] = pantry_score(ings, pantry)
            s_nutrition[i] = nutrition_score(nut, macro_targets, tolerance=macro_tolerance)
            # per-candidate hard constraint flags (for the dense top-K useful-rate)
            cand_feasible[i] = missing_count(ings, pantry) <= pantry_missing_threshold
            cand_near[i] = is_macro_near(nut, macro_targets, tolerance=macro_tolerance) if has_macros else True
        cand_useful = cand_feasible & cand_near

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
            cand_feasible=cand_feasible,
            cand_near=cand_near,
            cand_useful=cand_useful,
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


def _topk_idx(case: UserCase, weights: tuple[float, float, float], k: int) -> np.ndarray:
    """Indices of the top-K candidates for this user at these weights."""
    at, ap, an = weights
    final = at * case.s_taste + ap * case.s_pantry + an * case.s_nutrition
    k_eff = min(k, final.size)
    top = np.argpartition(-final, k_eff - 1)[:k_eff]
    return top


def per_user_metrics(cases: list[UserCase], weights: tuple[float, float, float], k: int = 10) -> pd.DataFrame:
    """Per-user top-K metrics at one simplex point — the inputs to a paired test.

    Returns a DataFrame (one row per user) with columns:
      user_id, recall@k (0/1 hit), cookable_rate@k, useful_rate@k

    Pair two of these (same `cases`, two weightings) with
    `significance.compare_models` / `paired_wilcoxon` to test whether moving
    along the simplex changes a metric significantly. Because both corners use
    the same users + pools (only α differs), the comparison is naturally paired.
    """
    rows = []
    for c in cases:
        top = _topk_idx(c, weights, k)
        hit = 1.0 if (c.holdout_in_pool and c.holdout_id in c.candidate_ids[top]) else 0.0
        rows.append({
            "user_id": c.user_id,
            f"recall@{k}": hit,
            f"cookable_rate@{k}": float(c.cand_feasible[top].mean()),
            f"useful_rate@{k}": float(c.cand_useful[top].mean()),
        })
    return pd.DataFrame(rows)


def sweep(
    cases: list[UserCase],
    k: int = 10,
    step: float = 0.05,
) -> pd.DataFrame:
    """Run the α-sweep; return a DataFrame with one row per simplex point.

    Columns per simplex point (all means over the user cases):
      - recall@k           : held-out item in top-K  (RELEVANCE; needs ground truth)
      - useful_recall@k    : held-out item in top-K AND it's useful (sparse here —
                             only meaningful with looser thresholds; see notes)
      - useful_rate@k      : fraction of the top-K that are useful (DENSE constraint-
                             satisfaction signal; no ground truth needed)
      - feasible_rate@k    : fraction of the top-K that are pantry-feasible
      - near_rate@k        : fraction of the top-K that are macro-near

    The headline trade-off is recall@k (relevance) vs useful_rate@k (constraint
    satisfaction) across the simplex.
    """
    grid = simplex_grid(step)
    n_users = len(cases)
    records = []
    for (at, ap, an) in grid:
        recall_hits = 0
        useful_hits = 0
        useful_sum = 0.0
        feasible_sum = 0.0
        near_sum = 0.0
        for case in cases:
            top = _topk_idx(case, (at, ap, an), k)
            # dense top-K constraint-satisfaction rates
            useful_sum += float(case.cand_useful[top].mean())
            feasible_sum += float(case.cand_feasible[top].mean())
            near_sum += float(case.cand_near[top].mean())
            # held-out relevance (needs the item to be in the pool)
            if case.holdout_in_pool and case.holdout_id in case.candidate_ids[top]:
                recall_hits += 1
                if case.holdout_useful:
                    useful_hits += 1
        denom = n_users if n_users else 1
        records.append({
            "alpha_taste": at,
            "alpha_pantry": ap,
            "alpha_nutrition": an,
            f"recall@{k}": recall_hits / denom,
            f"useful_recall@{k}": useful_hits / denom,
            f"useful_rate@{k}": useful_sum / denom,
            f"feasible_rate@{k}": feasible_sum / denom,
            f"near_rate@{k}": near_sum / denom,
        })
    return pd.DataFrame(records)


def best_point(sweep_df: pd.DataFrame, metric: str) -> pd.Series:
    """Return the simplex point (row) maximizing the given metric column."""
    return sweep_df.loc[sweep_df[metric].idxmax()]
