from src.data.loader import load_train_interactions, time_based_split
from src.eval.harness import evaluate, bootstrap_ci
from src.models.bpr import BPRRecommender

full = load_train_interactions()
train, _ = time_based_split(full, holdout_per_user=1)   # all 1-5 ratings kept

model = BPRRecommender(seed=42).fit(train)              # now lr=0.01, 500 iters, k=100
w = evaluate(model, track="warm", return_per_user=True)
m, lo, hi = bootstrap_ci(w["per_user"]["recall@10"])
print(f"BPR (all ratings)  warm R@10 = {w['recall@10']:.4f}  [{lo:.4f}, {hi:.4f}]   floor 0.0304")
