"""LightGCN: effect of embedding_dim (n_layers fixed at 2) -- exploration.

NOT part of the official Stage 1 model menu. Follow-up to
lightgcn_epoch_curve.py, which showed LightGCN saturates by ~epoch 10-20
and more training doesn't move recall. This checks the architecture knob
instead: does embedding width help? Uses EPOCHS=12 (above the ~10-epoch
convergence point measured earlier) to keep runtime under ~10 min for the
five-point sweep.

Run from the project root:

    python experiments/deep_cf/lightgcn_dim_sweep.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# Allow running directly from the project root after the move under experiments/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import load_train_interactions, time_based_split
from src.eval.harness import evaluate
from experiments.deep_cf.lightgcn import LightGCNRecommender

N_USERS = 2000
SEED = 42
EPOCHS = 12
DIMS = (8, 16, 32, 64, 128)

full = load_train_interactions()
train, _ = time_based_split(full, holdout_per_user=1)

rows = []
for dim in DIMS:
    model = LightGCNRecommender(embedding_dim=dim, n_layers=2, epochs=EPOCHS,
                                 batch_size=8192, seed=SEED, verbose=False).fit(train)
    r = evaluate(model, track="warm", k_values=(10, 100), n_users=N_USERS, seed=SEED)
    rows.append({"dim": dim, "recall@10": r["recall@10"], "recall@100": r["recall@100"]})
    print(f"LightGCN dim={dim:4d}  R@10={r['recall@10']:.4f}  R@100={r['recall@100']:.4f}")

table = pd.DataFrame(rows)
print("\n", table.round(4))
table.to_csv("lightgcn_dim_sweep.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for ax, metric in zip(axes, ("recall@10", "recall@100")):
    ax.plot(table["dim"], table[metric], marker="o")
    ax.set_xlabel("embedding_dim")
    ax.set_ylabel(metric)
    ax.set_xscale("log", base=2)
fig.suptitle("LightGCN warm-track Recall vs. embedding_dim (n_layers=2)")
fig.tight_layout()
fig.savefig("lightgcn_dim_sweep.png", dpi=150)
print("\nSaved plot to lightgcn_dim_sweep.png")
