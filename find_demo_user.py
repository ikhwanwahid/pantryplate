"""Find a good real user_id + pantry for a 'returning user (BPR)' Streamlit demo mode.

Picks an active user from train, fits BPR, derives a pantry from that user's
own liked-recipe history (same method as derive_user_constraints in
src/eval/alpha_sweep.py), then runs the full Stage 1 -> Stage 2 pipeline to
sanity-check the result actually looks good before recommending it for the demo.

Run from the project root:

    python find_demo_user.py
"""
import numpy as np
import pandas as pd

from collections import Counter

from src.data.loader import load_train_interactions, load_recipes
from src.models.bpr import BPRRecommender
from src.eval.alpha_sweep import derive_user_constraints, MACRO_FIELDS
from src.reranker.combiner import Stage2Reranker
from src.utils.staples import STAPLES

PANTRY_SIZE = 30


def top_frequency_pantry(train_df, recipes_df, user_id, positive_threshold=4, n=PANTRY_SIZE):
    """Most frequently recurring non-staple ingredients across a user's liked recipes.

    Unlike the full union (which balloons to hundreds of items for an active
    user), this approximates "what they probably actually keep on hand" --
    things that show up again and again, not everything they've ever cooked
    with once. Matches the project's 25-35 item persona-pantry convention.
    """
    pos = train_df[(train_df["user_id"] == user_id) & (train_df["rating"] >= positive_threshold)]
    rids = [int(r) for r in pos["recipe_id"].to_numpy() if int(r) in recipes_df.index]
    counter = Counter()
    for ings in recipes_df.loc[rids, "ingredients_parsed"]:
        if isinstance(ings, list):
            counter.update(i for i in ings if isinstance(i, str) and i not in STAPLES)
    return [item for item, _ in counter.most_common(n)]

train = load_train_interactions()
recipes = load_recipes()
recipes["id"] = recipes["id"].astype(np.int64)
recipes = recipes.set_index("id")
recipes.index.name = "recipe_id"

# Activity tier: pick a user with a healthy-but-plausible rating count
# (not a 1000+ outlier, not a 1-rating edge case) so BPR has real signal
# and the pantry derivation isn't dominated by a single genre of recipe.
counts = train.groupby("user_id").size()
candidates = counts[(counts >= 60) & (counts <= 120)].index.tolist()
print(f"{len(candidates)} users with 60-120 ratings in train")

import random
rng = random.Random(7)
shortlist = rng.sample(candidates, min(15, len(candidates)))

print("Fitting BPR on full train (this is the slow one)...")
bpr = BPRRecommender(seed=42).fit(train)

results = {}
for uid in shortlist:
    uid = int(uid)
    cons = derive_user_constraints(train, recipes, [uid])
    macro_targets = cons[uid]["macro_targets"]
    pantry = top_frequency_pantry(train, recipes, uid)
    if not pantry or not macro_targets:
        continue

    cand = [int(c) for c in bpr.recommend(uid, k=100, exclude_seen=True)]
    cand = [c for c in cand if c in recipes.index]
    if len(cand) < 20:
        continue

    taste_scores = {rid: 1.0 / (r + 1) for r, rid in enumerate(cand)}
    rr = Stage2Reranker(alpha_taste=0.4, alpha_pantry=0.3, alpha_nutrition=0.3)
    persona_like = {"pantry": pantry, "macro_targets": macro_targets, "restrictions": []}
    scored = rr.rerank(persona_like, cand, taste_scores, recipes, k=10, return_scores=True)
    if scored.empty:
        continue

    mean_pantry = scored["s_pantry"].mean()
    mean_taste = scored["s_taste"].mean()
    n_ratings = int(counts[uid])
    # Balance both: a high-pantry, near-zero-taste result looks like noise
    # in a demo. Want decent cookability AND recognizably relevant picks.
    balance = mean_pantry * 0.6 + mean_taste * 0.4 if mean_pantry < 0.95 else 0.0
    print(f"user {uid}: {n_ratings} ratings, mean s_pantry@10={mean_pantry:.2f}, "
          f"mean s_taste@10={mean_taste:.2f}, balance={balance:.2f}")

    results[uid] = (pantry, macro_targets, scored, n_ratings, balance)

ranked = sorted(results.items(), key=lambda kv: kv[1][4], reverse=True)
for uid, (pantry, macro_targets, scored, n_ratings, balance) in ranked[:3]:
    print(f"\n=== user_id={uid} ({n_ratings} ratings in train, balance={balance:.2f}) ===")
    print(f"Suggested pantry ({len(pantry)} most frequently recurring non-staple ingredients):")
    print(pantry)
    print(f"Derived macro targets: {macro_targets}")
    print("Top-10 Stage 2 output (alpha_taste=0.4, alpha_pantry=0.3, alpha_nutrition=0.3):")
    scored["name"] = scored["recipe_id"].map(lambda r: recipes.loc[r, "name"] if r in recipes.index else "?")
    print(scored[["recipe_id", "name", "s_taste", "s_pantry", "s_nutrition", "final"]].round(3).to_string(index=False))
