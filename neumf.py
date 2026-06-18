"""NeuMF (Neural Matrix Factorization) -- personal exploration only.

NOT part of the official Stage 1 model menu (docs/data_decisions.md locked
decision #5 already replaced NCF with two-tower neural). Not wired into
src/models/ or the leaderboard -- a side experiment to compare against the
existing CF baselines.

He et al., WWW 2017. Combines GMF (element-wise product of user/item
embeddings) with an MLP (concat + dense layers); a final linear+sigmoid head
fuses both. Trained with binary cross-entropy + uniform negative sampling on
implicit feedback (positive = a row that survives the item-rating filter).

Keeps the project's Stage 1 contract for drop-in compatibility with
src.eval.harness.evaluate:
    model.fit(train_df) -> self
    model.recommend(user_id, k, exclude_seen) -> list[int]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _NeuMFNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, gmf_dim: int, mlp_dim: int,
                 mlp_layers: tuple[int, ...]):
        super().__init__()
        self.u_gmf = nn.Embedding(n_users, gmf_dim)
        self.i_gmf = nn.Embedding(n_items, gmf_dim)
        self.u_mlp = nn.Embedding(n_users, mlp_dim)
        self.i_mlp = nn.Embedding(n_items, mlp_dim)

        layers = []
        in_dim = mlp_dim * 2
        for h in mlp_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(gmf_dim + mlp_layers[-1], 1)

        for emb in (self.u_gmf, self.i_gmf, self.u_mlp, self.i_mlp):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        gmf = self.u_gmf(u) * self.i_gmf(i)
        mlp_out = self.mlp(torch.cat([self.u_mlp(u), self.i_mlp(i)], dim=-1))
        return self.out(torch.cat([gmf, mlp_out], dim=-1)).squeeze(-1)


class NeuMFRecommender:
    def __init__(self, gmf_dim: int = 16, mlp_dim: int = 16,
                 mlp_layers: tuple[int, ...] = (64, 32, 16),
                 n_negatives: int = 4, epochs: int = 10,
                 batch_size: int = 4096, lr: float = 1e-3,
                 min_item_ratings: int = 10, seed: int = 42, verbose: bool = True):
        self.gmf_dim = gmf_dim
        self.mlp_dim = mlp_dim
        self.mlp_layers = mlp_layers
        self.n_negatives = n_negatives
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.min_item_ratings = min_item_ratings
        self.seed = seed
        self.verbose = verbose

    def fit(self, train_df: pd.DataFrame) -> "NeuMFRecommender":
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

        net = _NeuMFNet(n_users, n_items, self.gmf_dim, self.mlp_dim, self.mlp_layers)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        n_pos = len(u_arr)
        block = 1 + self.n_negatives
        for epoch in range(self.epochs):
            order = rng.permutation(n_pos)
            total_loss = 0.0
            for start in range(0, n_pos, self.batch_size):
                batch_idx = order[start:start + self.batch_size]
                bu, bi = u_arr[batch_idx], i_arr[batch_idx]

                neg_i = rng.integers(0, n_items, size=(len(bu), self.n_negatives))
                all_u = np.repeat(bu, block)
                all_i = np.concatenate([bi[:, None], neg_i], axis=1).ravel()
                labels = np.zeros(len(all_u), dtype=np.float32)
                labels[0::block] = 1.0

                u_t = torch.as_tensor(all_u, dtype=torch.long)
                i_t = torch.as_tensor(all_i, dtype=torch.long)
                y_t = torch.as_tensor(labels, dtype=torch.float32)

                opt.zero_grad()
                loss = loss_fn(net(u_t, i_t), y_t)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(all_u)

            if self.verbose:
                print(f"  NeuMF epoch {epoch + 1}/{self.epochs}  "
                      f"loss={total_loss / (n_pos * block):.4f}")

        net.eval()
        self._net = net
        self._n_items = n_items

        # full seen-set + popularity fallback over the FULL train (not filtered)
        self._user_seen = train_df.groupby("user_id")["recipe_id"].agg(set).to_dict()
        self._pop = train_df["recipe_id"].value_counts().index.tolist()
        return self

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()
        ranked: list[int] = []
        if user_id in self._u2idx:
            u = self._u2idx[user_id]
            with torch.no_grad():
                u_t = torch.full((self._n_items,), u, dtype=torch.long)
                i_t = torch.arange(self._n_items, dtype=torch.long)
                # .tolist() instead of .numpy(): this torch build's numpy
                # bridge is broken under numpy 2.x (ABI mismatch).
                scores = np.array(self._net(u_t, i_t).tolist())
            ranked = [self._idx2item[i] for i in np.argsort(-scores)]

        out: list[int] = []
        taken: set[int] = set()
        for rid in ranked + self._pop:   # NeuMF first, popularity fills the tail
            if rid in taken or rid in seen:
                continue
            out.append(int(rid))
            taken.add(rid)
            if len(out) == k:
                break
        return out
