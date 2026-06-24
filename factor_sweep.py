"""Latent-factor dimensionality sweep for BPR and ALS (warm track).

Fixes every other hyperparameter and varies only the latent dimension
(BPR's `k`, ALS's `factors`) to produce the classic recsys-paper plot of
Recall@K vs. model capacity. EASE has no latent-dimension knob (it's an
item-item closed-form model, not a factor model), so it's excluded here.

Run from the project root:

    python factor_sweep.py
"""
import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import load_train_interactions, time_based_split
from src.eval.harness import evaluate
from src.models.als import ALSRecommender
from src.models.bpr import BPRRecommender

DIMS = (8, 16, 32, 64, 128, 256)
N_USERS = 2000
SEED = 42

full = load_train_interactions()
train, _ = time_based_split(full, holdout_per_user=1)

rows = []
for dim in DIMS:
    bpr = BPRRecommender(k=dim, max_iter=300, learning_rate=0.01, seed=SEED).fit(train)
    r_bpr = evaluate(bpr, track="warm", k_values=(10, 100), n_users=N_USERS, seed=SEED)
    rows.append({"model": "BPR", "dim": dim,
                 "recall@10": r_bpr["recall@10"], "recall@100": r_bpr["recall@100"]})
    print(f"BPR  dim={dim:4d}  R@10={r_bpr['recall@10']:.4f}  R@100={r_bpr['recall@100']:.4f}")

    als = ALSRecommender(factors=dim, seed=SEED).fit(train)
    r_als = evaluate(als, track="warm", k_values=(10, 100), n_users=N_USERS, seed=SEED)
    rows.append({"model": "ALS", "dim": dim,
                 "recall@10": r_als["recall@10"], "recall@100": r_als["recall@100"]})
    print(f"ALS  dim={dim:4d}  R@10={r_als['recall@10']:.4f}  R@100={r_als['recall@100']:.4f}")

table = pd.DataFrame(rows)
print("\n", table.pivot(index="dim", columns="model", values=["recall@10", "recall@100"]).round(4))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for ax, metric in zip(axes, ("recall@10", "recall@100")):
    for model_name, grp in table.groupby("model"):
        ax.plot(grp["dim"], grp[metric], marker="o", label=model_name)
    ax.set_xlabel("latent dimension (k / factors)")
    ax.set_ylabel(metric)
    ax.set_xscale("log", base=2)
    ax.legend()
fig.suptitle("Warm-track Recall vs. latent-factor dimension")
fig.tight_layout()
fig.savefig("factor_sweep.png", dpi=150)
print("\nSaved plot to factor_sweep.png")

# --- ALS factors x regularization grid -----------------------------------
# ALS recall@10 fell monotonically as factors grew in the sweep above, with
# regularization held fixed at its default (0.05). That's the classic
# capacity-without-regularization overfitting signature. This grid checks
# whether scaling regularization up alongside factors recovers performance
# at high dims, or whether ALS just doesn't benefit from more capacity here.
ALS_REGS = (0.01, 0.05, 0.1, 0.5, 1.0)

grid_rows = []
for reg in ALS_REGS:
    for dim in DIMS:
        als = ALSRecommender(factors=dim, regularization=reg, seed=SEED).fit(train)
        r = evaluate(als, track="warm", k_values=(10, 100), n_users=N_USERS, seed=SEED)
        grid_rows.append({"reg": reg, "dim": dim,
                           "recall@10": r["recall@10"], "recall@100": r["recall@100"]})
        print(f"ALS  reg={reg:<5} dim={dim:4d}  R@10={r['recall@10']:.4f}  R@100={r['recall@100']:.4f}")

grid = pd.DataFrame(grid_rows)
print("\nrecall@10 by (dim, reg):\n",
      grid.pivot(index="dim", columns="reg", values="recall@10").round(4))
print("\nrecall@100 by (dim, reg):\n",
      grid.pivot(index="dim", columns="reg", values="recall@100").round(4))

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for ax, metric in zip(axes2, ("recall@10", "recall@100")):
    for reg, grp in grid.groupby("reg"):
        ax.plot(grp["dim"], grp[metric], marker="o", label=f"reg={reg}")
    ax.set_xlabel("latent dimension (factors)")
    ax.set_ylabel(metric)
    ax.set_xscale("log", base=2)
    ax.legend()
fig2.suptitle("ALS warm-track Recall vs. factors, by regularization")
fig2.tight_layout()
fig2.savefig("als_reg_dim_sweep.png", dpi=150)
print("\nSaved plot to als_reg_dim_sweep.png")

# --- ALS iterations x factors grid ----------------------------------------
# The regularization grid above ruled out under-regularization as the cause
# of ALS's decline at high factors (curves were flat across a 100x reg
# range). The other candidate: `iterations=20` is fixed regardless of
# factors, so a 256-factor model has far more parameters to fit in the same
# optimization budget as an 8-factor one. This checks whether more
# iterations recovers recall at high dims (undertrained) or not (factors
# just don't help here, full stop).
ITERS = (20, 50, 100, 200)
ITER_DIMS = (16, 64, 256)

iter_rows = []
for n_iter in ITERS:
    for dim in ITER_DIMS:
        als = ALSRecommender(factors=dim, iterations=n_iter, seed=SEED).fit(train)
        r = evaluate(als, track="warm", k_values=(10, 100), n_users=N_USERS, seed=SEED)
        iter_rows.append({"iterations": n_iter, "dim": dim,
                           "recall@10": r["recall@10"], "recall@100": r["recall@100"]})
        print(f"ALS  iters={n_iter:<4} dim={dim:4d}  R@10={r['recall@10']:.4f}  R@100={r['recall@100']:.4f}")

iter_grid = pd.DataFrame(iter_rows)
print("\nrecall@10 by (dim, iterations):\n",
      iter_grid.pivot(index="dim", columns="iterations", values="recall@10").round(4))
print("\nrecall@100 by (dim, iterations):\n",
      iter_grid.pivot(index="dim", columns="iterations", values="recall@100").round(4))

fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
for ax, metric in zip(axes3, ("recall@10", "recall@100")):
    for dim, grp in iter_grid.groupby("dim"):
        ax.plot(grp["iterations"], grp[metric], marker="o", label=f"factors={dim}")
    ax.set_xlabel("ALS iterations")
    ax.set_ylabel(metric)
    ax.legend()
fig3.suptitle("ALS warm-track Recall vs. iterations, by factor dimension")
fig3.tight_layout()
fig3.savefig("als_iter_dim_sweep.png", dpi=150)
print("\nSaved plot to als_iter_dim_sweep.png")
