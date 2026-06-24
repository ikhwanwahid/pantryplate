"""Per-persona α-sweep — tests proposal Hypothesis 2 (optimal α differs by user type).

The global sweep (`alpha_sweep_bpr_warm.csv`) derives each user's pantry/macro
constraints from their OWN history, then averages over 2000 users — it can't
tell us whether the optimal (αₜ, αₚ, αₙ) shifts with the *kind* of constraint
profile a user has. This script answers that: same 2000 users, same held-out
items (so relevance ground truth + the Stage-1 pool are unchanged), but every
user's pantry/macro_targets/restrictions are swapped for one persona's fixed
profile at a time. Three runs (fitness_focused, vegan_busy, family_friendly)
isolate the effect of the constraint profile from the effect of who the user is.

Run from the project root:

    python persona_alpha_sweep.py
"""
import json
import random

import numpy as np
import pandas as pd

from src.data.loader import load_train_interactions, load_recipes, time_based_split
from src.models.bpr import BPRRecommender
from src.eval.alpha_sweep import precompute_user_cases, sweep, best_point

PERSONA_IDS = ["fitness_focused", "vegan_busy", "family_friendly"]

full = load_train_interactions()
train_part, test_part = time_based_split(full, holdout_per_user=1)
recipes = load_recipes()
recipes["id"] = recipes["id"].astype(np.int64)
recipes = recipes.set_index("id")
recipes.index.name = "recipe_id"

holdout = test_part.groupby("user_id")["recipe_id"].first().to_dict()
users = random.Random(42).sample(list(holdout), 2000)
hmap = {int(u): int(holdout[u]) for u in users}

print("Fitting BPR on the post-LOO train split (shared across all persona runs)...")
bpr = BPRRecommender(seed=42).fit(train_part)

summary_rows = []
for pid in PERSONA_IDS:
    persona = json.load(open(f"data/personas/{pid}.json"))
    profile = {
        "pantry": set(persona["pantry"]),
        "macro_targets": persona["macro_targets"],
        "restrictions": persona["restrictions"],
    }
    constraints = {uid: profile for uid in users}

    print(f"\n[{pid}] precomputing candidate sub-scores for 2000 users...")
    cases = precompute_user_cases(bpr, hmap, recipes, constraints, k_pool=100)

    print(f"[{pid}] sweeping the simplex...")
    df = sweep(cases, k=10, step=0.05)
    out_path = f"data/processed/alpha_sweep_persona_{pid}.csv"
    df.to_csv(out_path, index=False)
    print(f"[{pid}] saved {out_path}")

    for metric in ("recall@10", "feasible_rate@10", "useful_rate@10"):
        b = best_point(df, metric)
        summary_rows.append({
            "persona": pid,
            "metric": metric,
            "best_alpha_taste": b["alpha_taste"],
            "best_alpha_pantry": b["alpha_pantry"],
            "best_alpha_nutrition": b["alpha_nutrition"],
            "best_value": b[metric],
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv("data/processed/alpha_sweep_persona_summary.csv", index=False)
print("\nBest (αₜ, αₚ, αₙ) per persona per metric:")
print(summary.round(3).to_string(index=False))
