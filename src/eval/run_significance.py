"""Reproduce the statistical-significance + CF-tuning results in the leaderboard.

Run from the project root:
    uv run python -m src.eval.run_significance

Deterministic (seed=42). Prints three tables:
  1. 95% bootstrap CIs on Recall@10 for the key models
  2. Paired Wilcoxon tests for the key model-vs-model comparisons
  3. A CF hyperparameter sweep (EASE λ; BPR factors×iters×lr) with, for each
     config, its Recall@10/@100 and a paired Wilcoxon p-value vs Popularity

Takes ~25 min (EASE Gram-matrix inversions + dense warm evals dominate).
The numbers it prints are the ones quoted in docs/stage1_leaderboard.md
§ "Statistical significance & CF tuning".
"""

from __future__ import annotations

import warnings

import pandas as pd

from src.data.loader import load_train_interactions, time_based_split
from src.eval.harness import evaluate
from src.eval.significance import ci_row, compare_models
from src.models.popularity import PopularityRecommender
from src.models.ease import EASERecommender
from src.models.bpr import BPRRecommender
from src.models.sentence_bert import SentenceBERTRecommender
from src.models.tag_svd_content import TagSVDRecommender


def _per_user(model, track, n_users=2000):
    kw = dict(track=track, k_values=(10, 100), seed=42, return_per_user=True)
    if track == "warm":
        kw["n_users"] = n_users
    return evaluate(model, **kw)["per_user"]


def main() -> None:
    warnings.filterwarnings("ignore")
    full = load_train_interactions()
    train_part, _ = time_based_split(full, holdout_per_user=1)

    # --- fit ---
    pop = PopularityRecommender().fit(train_part)
    bpr = BPRRecommender(seed=42).fit(train_part)
    ease = EASERecommender(seed=42).fit(train_part)
    sbert = SentenceBERTRecommender(batch_size=256).fit(full)
    tagsvd = TagSVDRecommender().fit(full)

    pu_pop = _per_user(pop, "warm")
    pu_bpr = _per_user(bpr, "warm")
    pu_ease = _per_user(ease, "warm")
    pu_sb = _per_user(sbert, "cold")
    pu_ts = _per_user(tagsvd, "cold")

    print("\n## 1. 95% bootstrap CIs — Recall@10")
    for name, p in [("Popularity (warm)", pu_pop), ("EASE (warm)", pu_ease),
                    ("BPR (warm)", pu_bpr), ("SBERT (cold)", pu_sb), ("Tag SVD (cold)", pu_ts)]:
        m, lo, hi = ci_row(p, "recall@10")
        print(f"  {name:20}: {m:6.3f}%  95% CI [{lo:.3f}, {hi:.3f}]")

    print("\n## 2. Paired Wilcoxon — Recall@10")
    pairs = [("BPR", "Popularity", pu_bpr, pu_pop), ("EASE", "Popularity", pu_ease, pu_pop),
             ("BPR", "EASE", pu_bpr, pu_ease), ("SBERT", "Tag SVD", pu_sb, pu_ts)]
    for la, lb, pa, pb in pairs:
        r = compare_models(pa, pb, "recall@10", la, lb)
        w = r["wilcoxon"]
        print(f"  {la:8} vs {lb:12}: diff={w['mean_diff']*100:+.3f}pp  "
              f"p={w['pvalue']:.4f}  sig={r['significant_at_0.05']}")

    print("\n## 3. CF tuning vs Popularity (warm) — does any config win?")
    for lam in (50, 100, 250, 500, 1000):
        m = EASERecommender(lambda_reg=lam, seed=42).fit(train_part)
        p = _per_user(m, "warm")
        r = compare_models(p, pu_pop, "recall@10", "m", "pop")
        m10, _, _ = ci_row(p, "recall@10")
        m100, _, _ = ci_row(p, "recall@100")
        print(f"  EASE λ={lam:<5} warm@10={m10:.3f}% warm@100={m100:.2f}% "
              f"p_vs_pop={r['wilcoxon']['pvalue']:.3f}")
    for k in (64, 128, 200):
        for it in (500, 1000):
            for lr in (0.01, 0.05):
                m = BPRRecommender(k=k, max_iter=it, learning_rate=lr, seed=42).fit(train_part)
                p = _per_user(m, "warm")
                r = compare_models(p, pu_pop, "recall@10", "m", "pop")
                m10, _, _ = ci_row(p, "recall@10")
                m100, _, _ = ci_row(p, "recall@100")
                print(f"  BPR k={k},it={it},lr={lr}  warm@10={m10:.3f}% "
                      f"warm@100={m100:.2f}% p_vs_pop={r['wilcoxon']['pvalue']:.3f}")


if __name__ == "__main__":
    main()
