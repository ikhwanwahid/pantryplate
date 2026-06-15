"""EASE (Embarrassingly Shallow Autoencoder) Stage 1 recommender.

Closed-form item-item model (Steck, WWW 2019). One hyperparameter (lambda_reg),
no SGD: a single regularized Gram-matrix inverse. Known to resist the
popularity bias that pairwise samplers like BPR drift into.

Wraps the project's Stage 1 contract:
    model.fit(train_df) -> self
    model.recommend(user_id, k, exclude_seen) -> list[int]

MEMORY NOTE: builds a dense (n_items x n_items) matrix and inverts it, i.e.
O(n_items^2) memory. Keep min_item_ratings >= 10 so the item count stays small.
Lowering the filter toward 1 balloons n_items and can exhaust RAM.

Goes in: src/models/ease.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class EASERecommender:
    def __init__(self, lambda_reg: float = 250.0,
                 min_item_ratings: int = 10, seed: int = 42):
        self.lambda_reg = lambda_reg
        self.min_item_ratings = min_item_ratings
        self.seed = seed  # unused: EASE is deterministic. Kept for contract parity.

    def fit(self, train_df: pd.DataFrame) -> "EASERecommender":
        # memory filter (convention 3 / decision 6) — also caps the Gram matrix size
        counts = train_df["recipe_id"].value_counts()
        keep = set(counts[counts >= self.min_item_ratings].index)
        df = train_df[train_df["recipe_id"].isin(keep)]

        users = df["user_id"].unique()
        items = df["recipe_id"].unique()
        self._u2idx = {u: r for r, u in enumerate(users)}
        self._i2idx = {it: c for c, it in enumerate(items)}
        self._idx2item = {c: it for it, c in self._i2idx.items()}

        rows = df["user_id"].map(self._u2idx).to_numpy()
        cols = df["recipe_id"].map(self._i2idx).to_numpy()
        vals = np.ones(len(df), dtype=np.float32)
        X = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
        self._X = X.tocsr()

        # closed form: B = -(P / diag(P)) with diag(B)=0,  P = (X^T X + lambda*I)^-1
        G = (X.T @ X).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.lambda_reg
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0.0)
        self._B = B

        # full seen-set + popularity fallback over the FULL train (not filtered)
        self._user_seen = train_df.groupby("user_id")["recipe_id"].agg(set).to_dict()
        self._pop = train_df["recipe_id"].value_counts().index.tolist()
        return self

    def recommend(self, user_id: int, k: int = 10,
                  exclude_seen: bool = True) -> list[int]:
        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()
        ranked: list[int] = []
        if user_id in self._u2idx:
            u = self._u2idx[user_id]
            scores = self._X[u].toarray().ravel() @ self._B   # (n_items,)
            ranked = [self._idx2item[i] for i in np.argsort(-scores)]

        out: list[int] = []
        taken: set[int] = set()
        for rid in ranked + self._pop:   # EASE first, popularity fills the tail
            if rid in taken or rid in seen:
                continue
            out.append(int(rid))
            taken.add(rid)
            if len(out) == k:
                break
        return out
