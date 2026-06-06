"""Tag SVD content recommender — Stage 1 model #5.

Uses the centralized 107-dim recipe feature matrix from `src/data/features.py`
(100-dim tag SVD + 7-dim normalized nutrition) as the item representation.
User profiles are built as the L2-normalized mean of positive-recipe vectors;
cold users fall back to popularity. Ranking is by cosine similarity.

Why this model: it's the cold-track content reference that asks "do explicit
tag co-occurrence + nutrition macros predict novel-recipe preferences?". Pairs
with SBERT (free-form text) for an ablation: do structured features and dense
text embeddings carry the same information or different information?

Convention (locked decision 11):
    .fit(train_df)              -> returns self
    .recommend(user_id, k, exclude_seen=True) -> list[int]

Mirrors `SentenceBERTRecommender`'s API (profile_strategy, content_weight)
so model-comparison code can swap them in without conditional logic.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from src.data.features import build_recipe_feature_matrix
from src.data.loader import POSITIVE_THRESHOLD


VALID_PROFILE_STRATEGIES = ("mean", "rating_weighted", "recency_weighted")


class TagSVDRecommender:
    """Cosine-similarity recommender over the 107-dim tag-SVD + nutrition matrix.

    Parameters
    ----------
    positive_threshold : int
        Rating ≥ this counts as a positive for user-profile construction.
        Defaults to the project-wide POSITIVE_THRESHOLD (= 4).
    profile_strategy : {"mean", "rating_weighted", "recency_weighted"}
        How to aggregate a user's positive recipes into a single profile vector.
        See SentenceBERTRecommender for full details; semantics are identical.
    recency_half_life_days : int
        Half-life for the recency weight. Only used by recency_weighted.
    content_weight : float in [0, 1]
        Stage-1-internal blend between content (cosine) and popularity:
            score = content_weight * cosine + (1 - content_weight) * popularity_score
        See SentenceBERTRecommender for the rationale. content_weight=1.0 (default)
        is pure content; content_weight=0.0 is equivalent to PopularityRecommender.

        NOTE: this α is INTERNAL to Stage 1. Distinct from the deck's Stage 2
        (αₜ, αₚ, αₙ) constraint-weight simplex.
    feature_provider : callable or None
        Optional override for the feature loader. Default: `build_recipe_feature_matrix()`.
        Passing a custom callable is mainly for tests (so we can inject a tiny matrix).
    """

    def __init__(
        self,
        positive_threshold: int = POSITIVE_THRESHOLD,
        profile_strategy: str = "mean",
        recency_half_life_days: int = 180,
        content_weight: float = 1.0,
        feature_provider=None,
    ):
        if profile_strategy not in VALID_PROFILE_STRATEGIES:
            raise ValueError(
                f"profile_strategy must be one of {VALID_PROFILE_STRATEGIES}, "
                f"got {profile_strategy!r}"
            )
        if not 0.0 <= content_weight <= 1.0:
            raise ValueError(
                f"content_weight must be in [0, 1], got {content_weight}"
            )
        self.positive_threshold = positive_threshold
        self.profile_strategy = profile_strategy
        self.recency_half_life_days = recency_half_life_days
        self.content_weight = content_weight
        self._feature_provider = feature_provider or build_recipe_feature_matrix

        # Populated by fit()
        self.recipe_ids: np.ndarray = np.empty(0, dtype=np.int64)
        self._recipe_id_to_row: dict[int, int] = {}
        self._recipe_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)  # L2-normalized
        self._user_vectors: dict[int, np.ndarray] = {}
        self._popularity_rank: np.ndarray = np.empty(0, dtype=np.int64)
        self._popularity_score: np.ndarray = np.empty(0, dtype=np.float32)
        self._user_seen: dict[int, set[int]] = {}

    # ---------------------------------------------------------------------
    # fit
    # ---------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> "TagSVDRecommender":
        if not {"user_id", "recipe_id", "rating"}.issubset(train_df.columns):
            raise ValueError("train_df must have 'user_id', 'recipe_id', 'rating' columns")

        # 1. Load the cached 107-dim feature matrix and L2-normalize each row
        features_df = self._feature_provider()
        self.recipe_ids = features_df.index.to_numpy(dtype=np.int64)
        self._recipe_id_to_row = {int(rid): i for i, rid in enumerate(self.recipe_ids)}

        raw = features_df.to_numpy(dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        # Recipes with all-zero features (shouldn't happen for the real matrix, but
        # guard anyway so the divide is safe).
        norms[norms == 0] = 1.0
        self._recipe_matrix = (raw / norms).astype(np.float32, copy=False)

        # 2. Per-user profile vectors
        positives = train_df[train_df["rating"] >= self.positive_threshold]
        self._user_vectors = self._build_user_profiles(positives)

        # 3. Popularity rank + per-recipe popularity score (for cold-user fallback
        #    and for the content_weight blend).
        popularity = train_df.groupby("recipe_id")["user_id"].nunique()
        popularity = popularity[popularity.index.isin(self._recipe_id_to_row)]
        self._popularity_rank = popularity.sort_values(ascending=False).index.to_numpy()

        pop_score = np.zeros(len(self.recipe_ids), dtype=np.float32)
        n_with_signal = len(self._popularity_rank)
        if n_with_signal > 0:
            for rank, rid in enumerate(self._popularity_rank):
                row = self._recipe_id_to_row[int(rid)]
                pop_score[row] = 1.0 - (rank / n_with_signal)
        self._popularity_score = pop_score

        # 4. Seen-set per user
        self._user_seen = (
            train_df.groupby("user_id")["recipe_id"]
            .agg(set)
            .to_dict()
        )
        return self

    # ---------------------------------------------------------------------
    # recommend
    # ---------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[int]:
        if self._recipe_matrix.size == 0:
            raise RuntimeError("Call .fit(train_df) before .recommend(...)")

        user_vec = self._user_vectors.get(int(user_id))
        if user_vec is None:
            return self._popularity_fallback(user_id, k, exclude_seen)

        scores = self._recipe_matrix @ user_vec  # (n_recipes,) cosine

        w = self.content_weight
        if w < 1.0:
            scores = w * scores + (1.0 - w) * self._popularity_score

        if exclude_seen:
            seen = self._user_seen.get(int(user_id), set())
            if seen:
                seen_rows = [self._recipe_id_to_row[r] for r in seen if r in self._recipe_id_to_row]
                if seen_rows:
                    scores = scores.copy()
                    scores[seen_rows] = -np.inf

        k_eff = min(k, scores.size)
        top_unsorted = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top_sorted = top_unsorted[np.argsort(-scores[top_unsorted])]
        return [int(self.recipe_ids[i]) for i in top_sorted]

    def recommend_many(
        self,
        user_ids: Iterable[int],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> dict[int, List[int]]:
        return {int(uid): self.recommend(int(uid), k=k, exclude_seen=exclude_seen) for uid in user_ids}

    # ---------------------------------------------------------------------
    # internals
    # ---------------------------------------------------------------------
    def _build_user_profiles(self, positives: pd.DataFrame) -> dict[int, np.ndarray]:
        """Aggregate per-user positive recipe vectors into a single L2-normalized profile.

        Strategy is selected by self.profile_strategy. Logic mirrors
        SentenceBERTRecommender so the two are interchangeable.
        """
        profiles: dict[int, np.ndarray] = {}
        strategy = self.profile_strategy

        for user_id, group in positives.groupby("user_id"):
            recipe_ids = group["recipe_id"].to_numpy()
            mask = np.array([int(rid) in self._recipe_id_to_row for rid in recipe_ids])
            if not mask.any():
                continue

            rows = np.array(
                [self._recipe_id_to_row[int(rid)] for rid in recipe_ids[mask]],
                dtype=np.int64,
            )
            vectors = self._recipe_matrix[rows]

            if strategy == "mean":
                weights = None
            elif strategy == "rating_weighted":
                ratings = group["rating"].to_numpy()[mask].astype(np.float32)
                weights = np.clip(ratings - 3.0, 0.0, None)
            elif strategy == "recency_weighted":
                dates = pd.to_datetime(group["date"].to_numpy()[mask], errors="coerce")
                if dates.isna().all():
                    weights = None
                else:
                    ref = dates.max()
                    days_old = np.asarray((ref - dates).total_seconds()) / 86400.0
                    weights = np.exp(-np.log(2.0) * days_old / self.recency_half_life_days).astype(np.float32)
            else:
                raise RuntimeError(f"unknown profile_strategy: {strategy}")

            if weights is not None and weights.sum() > 0:
                profile_vec = (vectors * weights[:, None]).sum(axis=0) / weights.sum()
            else:
                profile_vec = vectors.mean(axis=0)

            norm = np.linalg.norm(profile_vec)
            if norm == 0:
                continue
            profiles[int(user_id)] = (profile_vec / norm).astype(np.float32, copy=False)

        return profiles

    def _popularity_fallback(self, user_id: int, k: int, exclude_seen: bool) -> List[int]:
        if exclude_seen:
            seen = self._user_seen.get(int(user_id), set())
            out: list[int] = []
            for item in self._popularity_rank:
                if int(item) not in seen:
                    out.append(int(item))
                    if len(out) >= k:
                        break
            return out
        return [int(i) for i in self._popularity_rank[:k]]
