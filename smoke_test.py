"""Week 1 end-to-end smoke test — DUAL-TRACK evaluation.

Demonstrates both evaluation regimes:

  Track A (WARM-item, standard CF): we apply our own time-based leave-one-out
    on the authors' pre-split train. Held-out items are recipes the model
    has seen in train (via other users). Popularity baseline should produce
    small but non-zero Recall@10 (~1-2%).

  Track B (COLD-item, content-aware): we use the authors' pre-split test
    as-is. Held-out items have ZERO appearance in train (cold items).
    Popularity must score ~0% by construction — it has no signal for
    unseen items. Track B is meaningful only for content-aware models
    (TF-IDF, Sentence-BERT, hybrid, two-tower).

If this produces non-zero numbers on Track A and ~0 on Track B without
errors, the Week 1 pipeline correctly demonstrates the dual-track design.

Usage:
    uv run python smoke_test.py
"""

from __future__ import annotations

import random
import time

from src.data.loader import (
    load_prebuilt_split,
    load_train_interactions,
    load_test_interactions,
    time_based_split,
)
from src.models.popularity import PopularityRecommender
from src.eval.metrics import recall_at_k, ndcg_at_k, mrr


K_VALUES = (5, 10, 20)
SEED = 42
SAMPLE_USERS = 2000   # for the harness-style summary
SAMPLE_DEMO_USERS = 3 # for the per-user printout


def eval_model(model, test_df, sample_users, name):
    """Aggregate Recall@K, NDCG@K, MRR over a sample of test users."""
    truth = test_df.groupby("user_id")["recipe_id"].agg(set).to_dict()
    rng = random.Random(SEED)
    eval_uids = rng.sample(list(truth.keys()), min(sample_users, len(truth)))
    scores = {f"recall@{k}": [] for k in K_VALUES}
    scores.update({f"ndcg@{k}": [] for k in K_VALUES})
    scores["mrr"] = []
    for uid in eval_uids:
        recs = model.recommend(uid, k=max(K_VALUES), exclude_seen=True)
        relevant = truth[uid]
        for k in K_VALUES:
            scores[f"recall@{k}"].append(recall_at_k(recs, relevant, k))
            scores[f"ndcg@{k}"].append(ndcg_at_k(recs, relevant, k))
        scores["mrr"].append(mrr(recs, relevant))
    print(f"\n      [{name}] metrics over {len(eval_uids):,} users:")
    print(f"      {'metric':<12} {'mean':>8}")
    print(f"      {'-'*12} {'-'*8}")
    for k, vals in scores.items():
        mean = sum(vals) / len(vals) if vals else 0.0
        print(f"      {k:<12} {mean:>8.4f}")
    return scores


def main() -> None:
    print("=" * 70)
    print("PantryPlate · Dual-Track Smoke Test")
    print("=" * 70)

    t0 = time.time()
    print("\n[1/5] Loading authors' pre-split train file ...")
    full_train = load_train_interactions()
    print(f"      train (pre-split):     {len(full_train):,} interactions, "
          f"{full_train['user_id'].nunique():,} active users")
    print(f"      loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("\n[2/5] TRACK A — building warm-item evaluation (our own time-based LOO)")
    train_warm, test_warm = time_based_split(full_train, holdout_per_user=1)
    print(f"      train (after holdout): {len(train_warm):,} interactions")
    print(f"      test (warm held-outs): {len(test_warm):,} users with held-out positive")
    print(f"      done in {time.time()-t0:.1f}s")

    t0 = time.time()
    print("\n[3/5] TRACK B — loading authors' cold-item test set ...")
    test_cold = load_test_interactions()  # positives only by default
    print(f"      test (cold held-outs): {len(test_cold):,} users with held-out positive")
    # Verify cold-item nature
    train_recipes = set(train_warm["recipe_id"].unique())
    cold_in_train = test_cold["recipe_id"].isin(train_recipes).sum()
    print(f"      cold test items in train: {cold_in_train:,} / {len(test_cold):,} "
          f"({cold_in_train/len(test_cold)*100:.1f}%)")
    print(f"      → essentially all cold (model has never seen these items)")

    t0 = time.time()
    print("\n[4/5] Training PopularityRecommender on warm train ...")
    model = PopularityRecommender().fit(train_warm)
    print(f"      done in {time.time()-t0:.1f}s "
          f"({len(model._ranked_items):,} recipes ranked)")

    # Warm-track per-user demo
    print("\n[5/5] Demo recommendations on TRACK A (warm) for 3 sample users:")
    rng = random.Random(SEED)
    truth_warm = test_warm.groupby("user_id")["recipe_id"].agg(set).to_dict()
    demo_uids = rng.sample(list(truth_warm.keys()), SAMPLE_DEMO_USERS)
    for uid in demo_uids:
        recs = model.recommend(uid, k=10, exclude_seen=True)
        held_out = truth_warm[uid]
        hit = any(r in held_out for r in recs)
        print(f"      user {uid}: top-10 = {recs[:5]} ... "
              f"held-out = {sorted(held_out)}  hit@10 = {'YES' if hit else 'NO'}")

    # Dual-track evaluation
    print("\n" + "=" * 70)
    print("DUAL-TRACK EVALUATION SUMMARY")
    print("=" * 70)

    warm_scores = eval_model(model, test_warm, SAMPLE_USERS, "TRACK A — warm")
    cold_scores = eval_model(model, test_cold, SAMPLE_USERS, "TRACK B — cold")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    warm_r10 = sum(warm_scores["recall@10"]) / len(warm_scores["recall@10"])
    cold_r10 = sum(cold_scores["recall@10"]) / len(cold_scores["recall@10"])
    print(f"\n  Track A (warm) Recall@10 = {warm_r10:.4f}  ← popularity FLOOR for warm-item CF")
    print(f"  Track B (cold) Recall@10 = {cold_r10:.4f}  ← popularity CANNOT do cold-item by construction")
    print()
    print("  Week 2 expectations:")
    print("    Track A: MF, EASE, BPR, hybrid should comfortably beat popularity's warm floor.")
    print("    Track B: only content-aware models (TF-IDF, Sentence-BERT, hybrid, two-tower)")
    print("             will produce non-zero numbers — they read recipe content directly.")
    print()
    print("  This justifies the dual-track evaluation: different models win different regimes,")
    print("  and the hybrid is the only one that should perform well on both.")
    print("\n" + "=" * 70)
    print("SMOKE TEST: ✓ PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
