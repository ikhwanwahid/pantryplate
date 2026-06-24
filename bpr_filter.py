from src.data.loader import load_train_interactions, time_based_split
from src.eval.harness import evaluate, bootstrap_ci
from src.models.bpr import BPRRecommender

full = load_train_interactions()
train, _ = time_based_split(full, holdout_per_user=1)

print("floor 0.0304")
for mir in (10, 3, 1):
    m = BPRRecommender(k=100, max_iter=300, learning_rate=0.01,
                       min_item_ratings=mir, seed=42).fit(train)
    w = evaluate(m, track="warm", return_per_user=True)
    mean, lo, hi = bootstrap_ci(w["per_user"]["recall@10"])
    print(f"min_item_ratings={mir:<3} R@10 = {w['recall@10']:.4f}  [{lo:.4f}, {hi:.4f}]")

# diagnostic for the next step: how are per-user results shaped?
print("per_user type:", type(w["per_user"]))
print("warm keys:", list(w.keys()))
