"""NeuMF / LightGCN vs. the existing CF baselines -- personal exploration.

NOT part of the official Stage 1 model menu (docs/data_decisions.md locked
decision #5). Not wired into src/models/ or the leaderboard. Uses the same
warm-track eval convention as the rest of the repo's CF scripts
(n_users=2000, seed=42, k_values=(10, 100)) so the numbers are directly
comparable to docs/stage1_leaderboard.md.

Run from the project root:

    python deep_cf_compare.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from src.data.loader import load_train_interactions, time_based_split
from src.eval.harness import evaluate, bootstrap_ci
from src.models.bpr import BPRRecommender
from src.models.ease import EASERecommender
from src.models.popularity import PopularityRecommender
from neumf import NeuMFRecommender
from lightgcn import LightGCNRecommender

N_USERS = 2000
SEED = 42

full = load_train_interactions()
train, _ = time_based_split(full, holdout_per_user=1)

per_user = {}


def report(name, model):
    model.fit(train)
    r = evaluate(model, track="warm", k_values=(10, 100), n_users=N_USERS, seed=SEED,
                 return_per_user=True)
    m, lo, hi = bootstrap_ci(r["per_user"]["recall@10"])
    print(f"{name:24s} warm R@10={r['recall@10']:.4f}  [{lo:.4f}, {hi:.4f}]  "
          f"R@100={r['recall@100']:.4f}")
    per_user[name] = r["per_user"][["user_id", "recall@10"]]
    return r


print("--- existing CF baselines (re-run for an apples-to-apples comparison) ---")
report("Popularity", PopularityRecommender())
report("EASE", EASERecommender(lambda_reg=250.0, seed=SEED))
report("BPR (k=100)", BPRRecommender(k=100, max_iter=300, learning_rate=0.01, seed=SEED))

print("\n--- deep CF candidates ---")
report("NeuMF", NeuMFRecommender(gmf_dim=16, mlp_dim=16, mlp_layers=(64, 32, 16),
                                  n_negatives=4, epochs=30, batch_size=4096,
                                  seed=SEED, verbose=False))
report("LightGCN", LightGCNRecommender(embedding_dim=32, n_layers=2, epochs=30,
                                        batch_size=8192, seed=SEED, verbose=False))

print("\n--- paired bootstrap: is LightGCN's edge over the baselines real? ---")
base = per_user["LightGCN"]
for rival in ("Popularity", "EASE", "BPR (k=100)"):
    merged = base.merge(per_user[rival], on="user_id", suffixes=("_lgcn", "_rival"))
    diff = merged["recall@10_lgcn"] - merged["recall@10_rival"]
    m, lo, hi = bootstrap_ci(diff)
    sig = "significant (CI excludes 0)" if lo > 0 or hi < 0 else "not significant (CI spans 0)"
    print(f"  LightGCN - {rival:14s} mean diff={m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {sig}")
