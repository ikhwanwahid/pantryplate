from src.data.loader import load_train_interactions, time_based_split
from src.eval.harness import evaluate, bootstrap_ci
from src.models.bpr import BPRRecommender
from src.models.als import ALSRecommender

full = load_train_interactions()
train, _ = time_based_split(full, holdout_per_user=1)
train_pos = train[train["rating"] >= 4]   # drop 1-3 star noise from the positive signal

def report(name, model, tr):
    model.fit(tr)
    w = evaluate(model, track="warm", return_per_user=True)
    m, lo, hi = bootstrap_ci(w["per_user"]["recall@10"])
    print(f"{name:22s} warm R@10 = {w['recall@10']:.4f}  [{lo:.4f}, {hi:.4f}]")

print("floor: popularity = 0.0304")
report("BPR default (control)", BPRRecommender(seed=42), train)
report("BPR 4+ , 500 iter",     BPRRecommender(max_iter=500, seed=42), train_pos)
for a in (1, 5, 15, 40):
    report(f"ALS 4+ alpha={a}", ALSRecommender(alpha=float(a), seed=42), train_pos)
