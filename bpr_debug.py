from src.data.loader import load_train_interactions, time_based_split
from src.models.bpr import BPRRecommender

full = load_train_interactions()
train, test = time_based_split(full, holdout_per_user=1)

model = BPRRecommender(k=100, max_iter=300, learning_rate=0.01, seed=42).fit(train)

# pick an active user with a known held-out item
uid = train.groupby("user_id").size().idxmax()
recs = model.recommend(uid, k=10)
held = test[test["user_id"] == uid]["recipe_id"].tolist()

print("rec ids:   ", recs[:5])
print("rec dtype: ", [type(r).__name__ for r in recs[:3]])
print("held item: ", held, [type(h).__name__ for h in held])
print("recs in catalog? ", [r in set(train['recipe_id']) for r in recs[:5]])

# is BPR even producing a non-popularity order?
import numpy as np
scores = model._model.score(model._uid_map[uid])
print("score spread (should NOT be ~equal):", float(scores.min()), float(scores.max()), float(np.std(scores)))
