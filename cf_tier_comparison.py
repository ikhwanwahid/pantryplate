"""Warm-track Recall@10 by user activity tier: popularity vs BPR vs EASE.

This is the table that goes in the deck. It shows where personalization helps
(positive vs popularity) and where it reproduces popularity bias (negative on
high-activity users). Run from the project root:

    python cf_tier_comparison.py
"""
import pandas as pd

from src.data.loader import load_train_interactions, time_based_split
from src.data.features import classify_user_activity
from src.eval.harness import evaluate
from src.models.popularity import PopularityRecommender
from src.models.bpr import BPRRecommender
from src.models.ease import EASERecommender

full = load_train_interactions()
train, _ = time_based_split(full, holdout_per_user=1)
tiers = classify_user_activity(train)[["user_id", "activity_tier"]]


def per_tier(label, model):
    w = evaluate(model.fit(train), track="warm", return_per_user=True)
    pu = w["per_user"].merge(tiers, on="user_id", how="left")
    by_tier = pu.groupby("activity_tier")["recall@10"].mean()
    print(f"  {label:12s} done  (overall R@10 = {w['recall@10']:.4f})")
    return by_tier, w["recall@10"]


print("Fitting and evaluating 3 models (EASE is fast; BPR is the slow one)...")
models = {
    "popularity": PopularityRecommender(),
    "bpr": BPRRecommender(k=100, max_iter=300, learning_rate=0.01, seed=42),
    "ease": EASERecommender(lambda_reg=250.0, seed=42),
}

tier_cols, overall = {}, {}
for label, m in models.items():
    tier_cols[label], overall[label] = per_tier(label, m)

table = pd.DataFrame(tier_cols)
table["bpr_minus_pop"] = table["bpr"] - table["popularity"]
table["ease_minus_pop"] = table["ease"] - table["popularity"]

print("\nWarm Recall@10 by activity tier")
print(table.round(4))
print("\noverall:")
for label in models:
    print(f"  {label:12s} {overall[label]:.4f}")
print("\nRead: positive *_minus_pop means the model beats popularity on that tier.")
print("Watch the 'high' row — that is where popularity bias shows up.")
