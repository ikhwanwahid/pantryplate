"""LightGCN -- personal exploration only.

NOT part of the official Stage 1 model menu (see docs/data_decisions.md
locked decision #5). Not wired into src/models/ or the leaderboard -- a
side experiment to compare against the existing CF baselines.

He et al., SIGIR 2020. Strips feature transforms and nonlinearities out of
a GCN: embeddings are propagated over the symmetric-normalized user-item
bipartite adjacency, and the final embedding is the layer-wise mean.
Trained with the same BPR pairwise ranking loss as bpr.py.

Keeps the project's Stage 1 contract for drop-in compatibility with
src.eval.harness.evaluate:
    model.fit(train_df) -> self
    model.recommend(user_id, k, exclude_seen) -> list[int]
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix


class LightGCNRecommender:
    def __init__(self, embedding_dim: int = 32, n_layers: int = 2,
                 epochs: int = 8, batch_size: int = 8192, lr: float = 1e-3,
                 l2_reg: float = 1e-5, min_item_ratings: int = 10,
                 seed: int = 42, verbose: bool = True,
                 eval_every: int = 0,
                 eval_callback: Callable[[int, "LightGCNRecommender"], None] | None = None):
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.l2_reg = l2_reg
        self.min_item_ratings = min_item_ratings
        self.seed = seed
        self.verbose = verbose
        self.eval_every = eval_every          # checkpoint cadence; 0 disables
        self.eval_callback = eval_callback    # called as eval_callback(epoch, self)

    @staticmethod
    def _build_norm_adj(u_arr: np.ndarray, i_arr: np.ndarray,
                         n_users: int, n_items: int) -> torch.Tensor:
        n = n_users + n_items
        rows = np.concatenate([u_arr, i_arr + n_users])
        cols = np.concatenate([i_arr + n_users, u_arr])
        vals = np.ones(len(rows), dtype=np.float64)
        A = csr_matrix((vals, (rows, cols)), shape=(n, n))

        deg = np.asarray(A.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        d_inv_sqrt = 1.0 / np.sqrt(deg)
        A = A.tocoo()
        norm_vals = d_inv_sqrt[A.row] * A.data * d_inv_sqrt[A.col]

        idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
        val = torch.tensor(norm_vals, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()

    def _propagate(self, emb0: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        e = emb0
        layers = [e]
        for _ in range(self.n_layers):
            e = torch.sparse.mm(adj, e)
            layers.append(e)
        return torch.stack(layers, dim=0).mean(dim=0)

    def fit(self, train_df: pd.DataFrame) -> "LightGCNRecommender":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # memory/sparsity filter, same convention as bpr.py / als.py / ease.py
        counts = train_df["recipe_id"].value_counts()
        keep = set(counts[counts >= self.min_item_ratings].index)
        df = train_df[train_df["recipe_id"].isin(keep)]

        users = df["user_id"].unique()
        items = df["recipe_id"].unique()
        self._u2idx = {u: r for r, u in enumerate(users)}
        self._i2idx = {it: c for c, it in enumerate(items)}
        self._idx2item = {c: it for it, c in self._i2idx.items()}
        n_users, n_items = len(users), len(items)

        u_arr = df["user_id"].map(self._u2idx).to_numpy()
        i_arr = df["recipe_id"].map(self._i2idx).to_numpy()

        pos_by_user: dict[int, set[int]] = {}
        for uu, ii in zip(u_arr, i_arr):
            pos_by_user.setdefault(uu, set()).add(ii)

        adj = self._build_norm_adj(u_arr, i_arr, n_users, n_items)

        emb0 = nn.Parameter(torch.empty(n_users + n_items, self.embedding_dim))
        nn.init.normal_(emb0, std=0.1)
        opt = torch.optim.Adam([emb0], lr=self.lr)

        # set early so eval_callback can call self.recommend() at checkpoints
        self._user_seen = train_df.groupby("user_id")["recipe_id"].agg(set).to_dict()
        self._pop = train_df["recipe_id"].value_counts().index.tolist()

        n_pos = len(u_arr)
        for epoch in range(self.epochs):
            order = rng.permutation(n_pos)
            total_loss = 0.0
            for start in range(0, n_pos, self.batch_size):
                batch_idx = order[start:start + self.batch_size]
                bu, bi = u_arr[batch_idx], i_arr[batch_idx]

                bneg = rng.integers(0, n_items, size=len(bu))
                for j in range(len(bu)):
                    while bneg[j] in pos_by_user[bu[j]]:
                        bneg[j] = rng.integers(0, n_items)

                final = self._propagate(emb0, adj)
                u_emb, i_emb = final[:n_users], final[n_users:]

                u_t = torch.as_tensor(bu, dtype=torch.long)
                pos_t = torch.as_tensor(bi, dtype=torch.long)
                neg_t = torch.as_tensor(bneg, dtype=torch.long)

                pos_score = (u_emb[u_t] * i_emb[pos_t]).sum(-1)
                neg_score = (u_emb[u_t] * i_emb[neg_t]).sum(-1)
                bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()
                reg = self.l2_reg * (u_emb[u_t].pow(2).sum() + i_emb[pos_t].pow(2).sum()
                                      + i_emb[neg_t].pow(2).sum()) / len(bu)
                loss = bpr_loss + reg

                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(bu)

            if self.verbose:
                print(f"  LightGCN epoch {epoch + 1}/{self.epochs}  "
                      f"loss={total_loss / n_pos:.4f}")

            is_checkpoint = self.eval_every and (epoch + 1) % self.eval_every == 0
            if is_checkpoint or epoch == self.epochs - 1:
                with torch.no_grad():
                    final = self._propagate(emb0, adj)
                    # .tolist() instead of .numpy(): this torch build's numpy
                    # bridge is broken under numpy 2.x (ABI mismatch).
                    self._u_emb = np.array(final[:n_users].tolist())
                    self._i_emb = np.array(final[n_users:].tolist())
                if is_checkpoint and self.eval_callback is not None:
                    self.eval_callback(epoch + 1, self)

        return self

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()
        ranked: list[int] = []
        if user_id in self._u2idx:
            u = self._u2idx[user_id]
            scores = self._i_emb @ self._u_emb[u]
            ranked = [self._idx2item[i] for i in np.argsort(-scores)]

        out: list[int] = []
        taken: set[int] = set()
        for rid in ranked + self._pop:   # LightGCN first, popularity fills the tail
            if rid in taken or rid in seen:
                continue
            out.append(int(rid))
            taken.add(rid)
            if len(out) == k:
                break
        return out
