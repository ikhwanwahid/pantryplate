"""LightGCN: effect of n_layers (embedding_dim fixed at 32) -- exploration.

NOT part of the official Stage 1 model menu. Companion to
lightgcn_dim_sweep.py -- isolates the propagation-depth knob. GCN
literature generally expects diminishing or negative returns past ~3-4
layers (oversmoothing), so this checks whether that pattern holds here.

Run from the project root:

    python lightgcn_layers_sweep.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import load_train_interactions, time_based_split
from src.eval.harness import evaluate
from lightgcn import LightGCNRecommender

N_USERS = 2000
SEED = 42
EPOCHS = 15
LAYERS = (1, 2, 3, 4)

full = load_train_interactions()
train, _ = time_based_split(full, holdout_per_user=1)

rows = []
for n_layers in LAYERS:
    model = LightGCNRecommender(embedding_dim=32, n_layers=n_layers, epochs=EPOCHS,
                                 batch_size=8192, seed=SEED, verbose=False).fit(train)
    r = evaluate(model, track="warm", k_values=(10, 100), n_users=N_USERS, seed=SEED)
    rows.append({"n_layers": n_layers, "recall@10": r["recall@10"], "recall@100": r["recall@100"]})
    print(f"LightGCN layers={n_layers}  R@10={r['recall@10']:.4f}  R@100={r['recall@100']:.4f}")

table = pd.DataFrame(rows)
print("\n", table.round(4))
table.to_csv("lightgcn_layers_sweep.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for ax, metric in zip(axes, ("recall@10", "recall@100")):
    ax.plot(table["n_layers"], table[metric], marker="o")
    ax.set_xlabel("n_layers")
    ax.set_ylabel(metric)
fig.suptitle("LightGCN warm-track Recall vs. n_layers (embedding_dim=32)")
fig.tight_layout()
fig.savefig("lightgcn_layers_sweep.png", dpi=150)
print("\nSaved plot to lightgcn_layers_sweep.png")
