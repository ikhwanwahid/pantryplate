"""BPR (Bayesian Personalized Ranking) Stage 1 recommender.

Wraps Cornac's implicit-feedback BPR in the project's Stage 1 contract:
    model.fit(train_df) -> self
    model.recommend(user_id, k, exclude_seen) -> list[int]

Evaluate via src.eval.harness.evaluate(model, track="warm" | "cold").
CF-only: cold-track Recall is ~0 by construction (expected, not a bug).

Goes in: src/models/bpr.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import cornac
from cornac.data import Dataset


class BPRRecommender:
    def __init__(self, k: int = 100, max_iter: int = 500,
                learning_rate: float = 0.01, lambda_reg: float = 0.01,
                min_item_ratings: int = 10, seed: int = 42):
        self.k = k
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.min_item_ratings = min_item_ratings
        self.seed = seed

    def fit(self, train_df: pd.DataFrame) -> "BPRRecommender":
        # memory filter (convention 3 / decision 6): keep items with enough raters
        counts = train_df["recipe_id"].value_counts()
        keep = set(counts[counts >= self.min_item_ratings].index)
        df = train_df[train_df["recipe_id"].isin(keep)]

        uir = list(zip(df["user_id"].astype(int),
                       df["recipe_id"].astype(int),
                       df["rating"].astype(float)))
        self._dataset = Dataset.from_uir(uir, seed=self.seed)

        self._model = cornac.models.BPR(
            k=self.k, max_iter=self.max_iter,
            learning_rate=self.learning_rate, lambda_reg=self.lambda_reg,
            seed=self.seed, verbose=False,
        ).fit(self._dataset)

        # raw id <-> Cornac internal index
        self._uid_map = self._dataset.uid_map
        self._idx2item = {v: k for k, v in self._dataset.iid_map.items()}

        # full seen-set + popularity fallback over the FULL train (not filtered)
        self._user_seen = train_df.groupby("user_id")["recipe_id"].agg(set).to_dict()
        self._pop = train_df["recipe_id"].value_counts().index.tolist()
        return self

    def recommend(self, user_id: int, k: int = 10,
                  exclude_seen: bool = True) -> list[int]:
        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()
        ranked: list[int] = []
        if user_id in self._uid_map:
            scores = self._model.score(self._uid_map[user_id])
            ranked = [self._idx2item[i] for i in np.argsort(-scores)]

        out: list[int] = []
        taken: set[int] = set()
        for rid in ranked + self._pop:   # BPR first, popularity fills the tail
            if rid in taken or rid in seen:
                continue
            out.append(int(rid))
            taken.add(rid)
            if len(out) == k:
                break
        return out
