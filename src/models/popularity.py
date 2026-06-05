"""Popularity-based recommender — Week 1 baseline.

The simplest non-trivial recommender: rank items by how many distinct
users rated them in training, recommend the top-k that the user hasn't
already seen.

This serves two purposes:
1. A lower-bound reference. Any "learned" model should beat popularity
   on Recall@K / NDCG@K. If it doesn't, the model is broken.
2. A cold-start fallback. When other Stage 1 models can't score an item
   (e.g., EASE excludes recipes with <10 ratings), popularity fills the gap.

Convention for all recommenders in this project:
    .fit(train_df)              -> returns self
    .recommend(user_id, k, exclude_seen=True) -> List[recipe_id]

Where train_df has columns ['user_id', 'recipe_id', 'rating'].
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd


class PopularityRecommender:
    """Recommends the most-rated recipes in the training set.

    Parameters
    ----------
    count_unique_users : bool
        If True (default), popularity = # distinct users who rated the item.
        If False, popularity = # total ratings. Distinct-user count is more
        robust to bot-like users with many ratings.
    """

    def __init__(self, count_unique_users: bool = True):
        self.count_unique_users = count_unique_users
        self._ranked_items: np.ndarray = np.empty(0, dtype=int)
        self._user_seen: dict[int, set[int]] = {}

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        if not {"user_id", "recipe_id"}.issubset(train_df.columns):
            raise ValueError("train_df must have 'user_id' and 'recipe_id' columns")

        if self.count_unique_users:
            popularity = train_df.groupby("recipe_id")["user_id"].nunique()
        else:
            popularity = train_df.groupby("recipe_id").size()

        # Descending order — most popular first. Stable sort so ties resolve by recipe_id.
        self._ranked_items = popularity.sort_values(ascending=False).index.to_numpy()

        # Record what each user already saw, so we can exclude on recommend()
        self._user_seen = (
            train_df.groupby("user_id")["recipe_id"]
            .agg(set)
            .to_dict()
        )
        return self

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[int]:
        """Return the top-k recipe IDs for this user, ordered by popularity."""
        if self._ranked_items.size == 0:
            raise RuntimeError("Call .fit(train_df) before .recommend(...)")

        if exclude_seen:
            seen = self._user_seen.get(user_id, set())
            # Walk down the popularity list, skip seen items, stop at k
            out: list[int] = []
            for item in self._ranked_items:
                if int(item) not in seen:
                    out.append(int(item))
                    if len(out) >= k:
                        break
            return out
        else:
            return [int(i) for i in self._ranked_items[:k]]

    def recommend_many(
        self,
        user_ids: Iterable[int],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> dict[int, List[int]]:
        """Batched convenience wrapper."""
        return {uid: self.recommend(uid, k=k, exclude_seen=exclude_seen) for uid in user_ids}
