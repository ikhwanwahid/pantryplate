"""Hybrid linear Stage 1 recommender — blends EASE (CF) with Tag SVD (content).

score(user, item) = α · cf_score(user, item) + (1 − α) · content_score(user, item)

Both score components are rank-normalised to [0, 1] before blending so the
α weight is interpretable across users and models regardless of the raw score
distributions.

Why EASE + Tag SVD:
  - EASE is deterministic, closed-form, no SGD, and empirically strong on warm.
  - Tag SVD is cheap (no external deps, fits from cached parquet in <1 s) and
    produces non-zero cold-track scores that EASE cannot.
  - For users with no EASE history, α is auto-zeroed → pure content, so the
    model degrades gracefully to content-only rather than injecting noise.

Stage 1 contract (locked decision §11):
    .fit(train_df) -> self
    .recommend(user_id, k, exclude_seen=True) -> list[int]
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.data.loader import POSITIVE_THRESHOLD
from src.models.ease import EASERecommender
from src.models.tag_svd_content import TagSVDRecommender


def _rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Map a score array to [0, 1] via rank: best score → 1.0, worst → 0.0."""
    n = len(scores)
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    order = np.argsort(-scores)
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    return (1.0 - ranks / (n - 1)).astype(np.float32)


class HybridLinearRecommender:
    """Linear blend of EASE (CF) and Tag SVD (content) scores.

    Parameters
    ----------
    alpha : float in [0, 1]
        Weight on the CF (EASE) component. 1.0 = pure EASE, 0.0 = pure Tag SVD.
    lambda_reg : float
        EASE regularisation λ.
    min_item_ratings : int
        EASE item-frequency filter. Items with fewer ratings are excluded from
        the CF matrix (rare items fall back to content-only scoring).
    positive_threshold : int
        Minimum rating treated as a positive when building Tag SVD user profiles.
    profile_strategy : str
        Tag SVD profile aggregation: "mean", "rating_weighted", "recency_weighted".
    seed : int
        Passed through to EASE for contract parity (EASE is deterministic).
    feature_provider : callable or None
        Override the Tag SVD feature loader. Mainly for injecting tiny synthetic
        matrices in tests without loading the real 107-dim parquet.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        lambda_reg: float = 250.0,
        min_item_ratings: int = 10,
        positive_threshold: int = POSITIVE_THRESHOLD,
        profile_strategy: str = "mean",
        seed: int = 42,
        feature_provider=None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.min_item_ratings = min_item_ratings
        self.positive_threshold = positive_threshold
        self.profile_strategy = profile_strategy
        self.seed = seed
        self._feature_provider = feature_provider

        # Sub-models — populated by fit()
        self._ease: EASERecommender | None = None
        self._content: TagSVDRecommender | None = None

        # Vectorised bridge: maps EASE column indices ↔ content row indices
        # for items that appear in both catalogs (built once in fit).
        self._ease_cols: np.ndarray = np.empty(0, dtype=np.int64)
        self._content_rows_for_ease: np.ndarray = np.empty(0, dtype=np.int64)
        self._n_all: int = 0

        # Fallback
        self._pop: list[int] = []
        self._user_seen: dict[int, set[int]] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "HybridLinearRecommender":
        if not {"user_id", "recipe_id", "rating"}.issubset(train_df.columns):
            raise ValueError("train_df must have 'user_id', 'recipe_id', 'rating' columns")

        # 1. Fit CF sub-model
        self._ease = EASERecommender(
            lambda_reg=self.lambda_reg,
            min_item_ratings=self.min_item_ratings,
            seed=self.seed,
        ).fit(train_df)

        # 2. Fit content sub-model (content_weight=1.0 = pure content;
        #    blending across CF and content happens here at the hybrid level)
        content_kwargs: dict = dict(
            positive_threshold=self.positive_threshold,
            profile_strategy=self.profile_strategy,
            content_weight=1.0,
        )
        if self._feature_provider is not None:
            content_kwargs["feature_provider"] = self._feature_provider
        self._content = TagSVDRecommender(**content_kwargs).fit(train_df)

        # 3. Build the EASE-column → content-row bridge for vectorised score
        #    expansion. Items that exist only in EASE (not the feature parquet)
        #    are silently dropped; items only in the feature parquet get CF score 0.
        self._n_all = len(self._content.recipe_ids)
        ease_cols, content_rows = [], []
        for recipe_id, ease_col in self._ease._i2idx.items():
            content_row = self._content._recipe_id_to_row.get(recipe_id)
            if content_row is not None:
                ease_cols.append(ease_col)
                content_rows.append(content_row)
        self._ease_cols = np.array(ease_cols, dtype=np.int64)
        self._content_rows_for_ease = np.array(content_rows, dtype=np.int64)

        # 4. Global popularity order + seen sets (used for fallback)
        self._pop = train_df["recipe_id"].value_counts().index.tolist()
        self._user_seen = (
            train_df.groupby("user_id")["recipe_id"].agg(set).to_dict()
        )
        return self

    # ------------------------------------------------------------------
    # recommend
    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[int]:
        if self._ease is None or self._content is None:
            raise RuntimeError("Call .fit(train_df) before .recommend(...)")

        uid = int(user_id)
        in_ease = uid in self._ease._u2idx
        user_vec = self._content._user_vectors.get(uid)
        has_content = user_vec is not None

        # No CF signal and no content profile → popularity fallback
        if not in_ease and not has_content:
            return self._popularity_fallback(uid, k, exclude_seen)

        # Auto-zero CF weight for users absent from EASE's catalog so we don't
        # pollute content scores with all-zero CF noise after rank-normalisation.
        effective_alpha = self.alpha if in_ease else 0.0

        # --- CF scores: rank-normalise ONLY within the EASE item set ---
        # Items outside EASE (rare items, cold items) get cf_norm = 0 so the
        # CF component doesn't penalise them vs items that genuinely have no
        # CF signal. Without this, cold items tied at raw-score 0 would be
        # rank-normalised to ~0.5, suppressing their content scores.
        cf_norm = np.zeros(self._n_all, dtype=np.float32)
        if effective_alpha > 0.0:
            u_idx = self._ease._u2idx[uid]
            ease_raw = (
                self._ease._X[u_idx].toarray().ravel() @ self._ease._B
            ).astype(np.float32)
            if self._content_rows_for_ease.size > 0:
                cf_norm[self._content_rows_for_ease] = _rank_normalize(
                    ease_raw[self._ease_cols]
                )

        # --- Content scores (full catalogue via cosine sim) ---
        content_scores = np.zeros(self._n_all, dtype=np.float32)
        if has_content:
            content_scores = (
                self._content._recipe_matrix @ user_vec
            ).astype(np.float32)

        content_norm = (
            _rank_normalize(content_scores)
            if has_content
            else np.zeros(self._n_all, dtype=np.float32)
        )
        scores = effective_alpha * cf_norm + (1.0 - effective_alpha) * content_norm

        # --- Mask seen items ---
        if exclude_seen:
            seen = self._user_seen.get(uid, set())
            if seen:
                seen_rows = [
                    self._content._recipe_id_to_row[r]
                    for r in seen
                    if r in self._content._recipe_id_to_row
                ]
                if seen_rows:
                    scores = scores.copy()
                    scores[seen_rows] = -np.inf

        # --- Top-k via argpartition (O(n) vs O(n log n) for full sort) ---
        k_eff = min(k, self._n_all)
        top_unsorted = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top_sorted = top_unsorted[np.argsort(-scores[top_unsorted])]
        return [int(self._content.recipe_ids[i]) for i in top_sorted]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _popularity_fallback(self, user_id: int, k: int, exclude_seen: bool) -> List[int]:
        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()
        out: list[int] = []
        for rid in self._pop:
            if rid not in seen:
                out.append(int(rid))
                if len(out) == k:
                    break
        return out
