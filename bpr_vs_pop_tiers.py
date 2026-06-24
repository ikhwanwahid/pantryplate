import pandas as pd
from src.data.loader import load_train_interactions, time_based_split
from src.data.features import classify_user_activity
from src.eval.harness import evaluate
from src.models.bpr import BPRRecommender
from src.models.popularity import PopularityRecommender

full = load_train_interactions()
train, _ = time_based_split(full, holdout_per_user=1)
tiers = classify_user_activity(train)[["user_id", "activity_tier"]]

def per_tier(model):
    w = evaluate(model.fit(train), track="warm", return_per_user=True)
    pu = w["per_user"].merge(tiers, on="user_id", how="left")
    return pu.groupby("activity_tier")["recall@10"].mean(), w["recall@10"]

pop_t, pop_all = per_tier(PopularityRecommender())
bpr_t, bpr_all = per_tier(BPRRecommender(k=100, max_iter=300, learning_rate=0.01, seed=42))

out = pd.DataFrame({"popularity": pop_t, "bpr": bpr_t})
out["bpr_minus_pop"] = out["bpr"] - out["popularity"]
print(out.round(4))
print(f"\noverall  popularity {pop_all:.4f}   bpr {bpr_all:.4f}")
