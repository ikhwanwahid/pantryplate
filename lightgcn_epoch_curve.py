"""LightGCN: does warm Recall keep improving with more training? -- exploration.

NOT part of the official Stage 1 model menu. Personal follow-up to
deep_cf_compare.py: that run used 30 epochs and landed within noise of
Popularity/EASE/BPR. This trains one LightGCN run out to EPOCHS, checking
in every CHECKPOINT epochs via the eval_every/eval_callback hooks added to
LightGCNRecommender, to see whether recall is still climbing or has
plateaued.

Run from the project root:

    python lightgcn_epoch_curve.py
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
EPOCHS = 80
CHECKPOINT = 10

full = load_train_interactions()
train, _ = time_based_split(full, holdout_per_user=1)

curve = []


def checkpoint_eval(epoch: int, model: LightGCNRecommender) -> None:
    r = evaluate(model, track="warm", k_values=(10, 100), n_users=N_USERS, seed=SEED)
    curve.append({"epoch": epoch, "recall@10": r["recall@10"], "recall@100": r["recall@100"]})
    print(f"  [checkpoint] epoch={epoch:4d}  R@10={r['recall@10']:.4f}  R@100={r['recall@100']:.4f}")


model = LightGCNRecommender(embedding_dim=32, n_layers=2, epochs=EPOCHS, batch_size=8192,
                             seed=SEED, verbose=False,
                             eval_every=CHECKPOINT, eval_callback=checkpoint_eval)
model.fit(train)

table = pd.DataFrame(curve)
print("\n", table.round(4))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for ax, metric in zip(axes, ("recall@10", "recall@100")):
    ax.plot(table["epoch"], table[metric], marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
fig.suptitle("LightGCN warm-track Recall vs. training epoch")
fig.tight_layout()
fig.savefig("lightgcn_epoch_curve.png", dpi=150)
print("\nSaved plot to lightgcn_epoch_curve.png")
