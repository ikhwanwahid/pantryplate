from src.data.loader import load_train_interactions, time_based_split
from src.data.features import classify_user_activity
from src.eval.harness import evaluate
from src.models.bpr import BPRRecommender

full = load_train_interactions()
train, _ = time_based_split(full, holdout_per_user=1)
model = BPRRecommender(k=100, max_iter=300, learning_rate=0.01, seed=42).fit(train)

w = evaluate(model, track="warm", return_per_user=True)
pu = w["per_user"]                      # DataFrame, one row per evaluated user
tiers = classify_user_activity(train)   # user_id -> activity_tier

print("per_user columns:", list(pu.columns))
merged = pu.merge(tiers[["user_id", "activity_tier"]], on="user_id", how="left")
print(merged.groupby("activity_tier")["recall@10"].agg(["mean", "count"]))
