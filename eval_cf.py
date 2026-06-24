from src.data.loader import load_train_interactions, time_based_split
from src.eval.harness import evaluate, bootstrap_ci
from src.models.bpr import BPRRecommender
from src.models.als import ALSRecommender

full = load_train_interactions()
train, _ = time_based_split(full, holdout_per_user=1)

for name, model in [("BPR", BPRRecommender(seed=42)),
                    ("ALS", ALSRecommender(seed=42))]:
    model.fit(train)
    warm = evaluate(model, track="warm", return_per_user=True)
    cold = evaluate(model, track="cold")
    m, lo, hi = bootstrap_ci(warm["per_user"]["recall@10"])
    print(f"{name}  warm R@10 = {warm['recall@10']:.4f}  [{lo:.4f}, {hi:.4f}]   "
          f"cold R@10 = {cold['recall@10']:.4f}")
