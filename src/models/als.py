"""ALS (implicit-feedback Alternating Least Squares) Stage 1 recommender.

Wraps the `implicit` library's AlternatingLeastSquares (Hu, Koren & Volinsky
2008) in the project's Stage 1 contract:
    model.fit(train_df) -> self
    model.recommend(user_id, k, exclude_seen) -> list[int]

Uses the CURRENT implicit API: fit() takes a (users, items) CSR matrix and
recommend() takes the user's own row. CF-only: cold-track Recall is ~0 by
construction (expected). Unlike explicit rating-prediction MF, this ignores
the rating value and ranks by confidence, so it is competitive on top-K.

Goes in: src/models/als.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


class ALSRecommender:
    def __init__(self, factors: int = 64, regularization: float = 0.05,
                 alpha: float = 40.0, iterations: int = 20,
                 min_item_ratings: int = 10, num_threads: int = 0,
                 seed: int = 42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha            # Hu et al. confidence scaling
        self.iterations = iterations
        self.min_item_ratings = min_item_ratings
        self.num_threads = num_threads  # set to 1 for bit-exact reproducibility
        self.seed = seed

    def _build_model(self):
        common = dict(
            factors=self.factors, regularization=self.regularization,
            iterations=self.iterations, num_threads=self.num_threads,
            random_state=self.seed,
        )
        try:
            # implicit >= 0.6: confidence scaling is a model param
            return AlternatingLeastSquares(alpha=self.alpha, **common), False
        except TypeError:
            # older implicit: scale the matrix instead of passing alpha
            return AlternatingLeastSquares(**common), True

    def fit(self, train_df: pd.DataFrame) -> "ALSRecommender":
        # memory filter (convention 3 / decision 6)
        counts = train_df["recipe_id"].value_counts()
        keep = set(counts[counts >= self.min_item_ratings].index)
        df = train_df[train_df["recipe_id"].isin(keep)]

        users = df["user_id"].unique()
        items = df["recipe_id"].unique()
        self._u2idx = {u: r for r, u in enumerate(users)}
        self._i2idx = {i: c for c, i in enumerate(items)}
        self._idx2item = {c: i for i, c in self._i2idx.items()}

        rows = df["user_id"].map(self._u2idx).to_numpy()
        cols = df["recipe_id"].map(self._i2idx).to_numpy()
        vals = np.ones(len(df), dtype=np.float32)   # binary implicit feedback

        self._model, prescale = self._build_model()
        if prescale:
            vals *= np.float32(self.alpha)
        self._ui = csr_matrix((vals, (rows, cols)),
                              shape=(len(users), len(items)))
        self._model.fit(self._ui)     # (users, items) — current implicit API

        # full seen-set + popularity fallback over the FULL train
        self._user_seen = train_df.groupby("user_id")["recipe_id"].agg(set).to_dict()
        self._pop = train_df["recipe_id"].value_counts().index.tolist()
        return self

    def recommend(self, user_id: int, k: int = 10,
                  exclude_seen: bool = True) -> list[int]:
        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()
        ranked: list[int] = []
        if user_id in self._u2idx:
            u = self._u2idx[user_id]
            n = min(k + len(seen) + 50, self._ui.shape[1])
            ids, _ = self._model.recommend(
                u, self._ui[u], N=n,
                filter_already_liked_items=exclude_seen,
            )
            ranked = [self._idx2item[int(c)] for c in ids]

        out: list[int] = []
        taken: set[int] = set()
        for rid in ranked + self._pop:   # ALS first, popularity fills the tail
            if rid in taken or rid in seen:
                continue
            out.append(int(rid))
            taken.add(rid)
            if len(out) == k:
                break
        return out
