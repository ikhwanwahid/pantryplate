# Eval Harness Usage Guide

**Quick reference for `src/eval/harness.py`.** Five-minute read.

If you haven't yet, read the **"Data files" section in `week2_onboarding.md`** first. The most common harness mistakes come from confusing the train / validation / test files. Don't skip it.

---

## TL;DR

```python
from src.eval.harness import evaluate, bootstrap_ci, compare_models

# Single model, one track
result = evaluate(my_model, track="warm")
# → {"recall@10": 0.045, "ndcg@10": 0.021, ..., "n_users_evaluated": 24382}

# Sample 2000 users for fast iteration
result = evaluate(my_model, track="warm", n_users=2000, seed=42)

# Multiple models side-by-side
df = compare_models({"ease": ease, "bpr": bpr, "popularity": pop}, track="warm")

# Bootstrap 95% CI on a metric
mean, lo, hi = bootstrap_ci(per_user_scores, n_bootstrap=1000)
```

That's it. Three functions, one mental model: *"the harness loads the right test set for the track you specify and runs your model against it."*

---

## ⭐ Which test set does each `track` use?

This is the bit teammates trip on most. Memorize this table:

| `track` | Test set source | What it tests | Your model trains on |
|---|---|---|---|
| `"warm"` | Held-out positives from `interactions_train.csv` (via deterministic time-based LOO) | Standard CF benchmarking — items have rater history | The other ~658K interactions from `interactions_train.csv` |
| `"cold"` | `interactions_test.csv` (authors' pre-split, 10,393 cold positives) | Cold-item generalization — items have ZERO raters in train | Same warm training data as above |

**You do not read `interactions_test.csv` or `interactions_validation.csv` from your model code.** The harness reads them. Your model only trains on the warm training data (held-out portion removed by `time_based_split`).

### Visual

```
       ┌────────────────── interactions_train.csv (698K) ──────────────────┐
       │                                                                  │
       │   time_based_split(holdout_per_user=1)                           │
       │                                                                  │
       │   ┌──────────────────────┐         ┌─────────────────────┐       │
       │   │  ~658K rows          │         │  ~24K positives     │       │
       │   │  YOUR MODEL TRAINS   │         │  Harness uses for   │       │
       │   │  ON THIS             │         │  track="warm"       │       │
       │   └──────────────────────┘         └─────────────────────┘       │
       └──────────────────────────────────────────────────────────────────┘

       ┌────────────────── interactions_test.csv (10K) ────────────────────┐
       │   Harness uses for track="cold". Items have ZERO raters in train │
       │   → CF models score 0 here by construction (this is correct).   │
       └──────────────────────────────────────────────────────────────────┘

       ┌────────────────── interactions_validation.csv (7K) ───────────────┐
       │   NOT USED BY THE HARNESS.                                        │
       │   Available via load_validation_interactions() if your model has  │
       │   hyperparameters to tune during development.                     │
       └──────────────────────────────────────────────────────────────────┘
```

---

## `evaluate(...)` — the main function

```python
def evaluate(
    model,                      # Any object with .recommend(user_id, k, exclude_seen)
    track: str = "warm",        # "warm" or "cold"
    k_values: tuple = (5, 10, 20),
    n_users: int | None = None, # None = full test set; int = random sample
    seed: int = 42,
    return_per_user: bool = False,
    candidate_filter: callable = None,  # Stage 2 hook (Week 4+; ignore for now)
    data_path: str = "data/raw",
) -> dict:
    ...
```

### Returns

```python
{
    "recall@5": 0.019,
    "recall@10": 0.030,
    "recall@20": 0.050,
    "ndcg@5": 0.011,
    "ndcg@10": 0.015,
    "ndcg@20": 0.017,
    "mrr": 0.011,
    "n_users_evaluated": 24382,
    "track": "warm",
    "seed": 42,
    # Plus "per_user": pd.DataFrame  (only if return_per_user=True)
}
```

### Speed (so you know what to expect)

| Call | Time |
|---|---|
| Full warm eval (24K users) | ~0.8 sec |
| Full cold eval (10K users) | ~0.1 sec |
| Sampled 2000 users | <1 sec |

The first call loads the test set; subsequent calls reuse the cached set. So 30 evaluations in an α-sweep don't re-load the CSV 30 times.

### Common patterns

**Quick smoke test during development:**
```python
quick = evaluate(my_model, track="warm", n_users=500, seed=42)
print(f"Recall@10 = {quick['recall@10']:.4f}")
```

**Final reporting (full test set + CIs):**
```python
final = evaluate(my_model, track="warm", return_per_user=True)
mean, lo, hi = bootstrap_ci(final["per_user"]["recall@10"])
print(f"Recall@10 = {mean:.4f} [95% CI: {lo:.4f}, {hi:.4f}]")
print(f"NDCG@10  = {final['ndcg@10']:.4f}")
```

**Evaluate on both tracks at once:**
```python
for track in ("warm", "cold"):
    r = evaluate(my_model, track=track)
    print(f"{track:5} Recall@10 = {r['recall@10']:.4f}  ({r['n_users_evaluated']:,} users)")
```

---

## `compare_models(...)` — side-by-side comparison

```python
df = compare_models(
    {"popularity": pop_model, "ease": ease_model, "bpr": bpr_model, "sbert": sbert_model},
    track="warm",
    n_users=2000,   # same sample applied to all models for fair comparison
    seed=42,
)
print(df)
#               recall@5  recall@10  recall@20  ndcg@10     mrr  n_users_evaluated
# model
# popularity      0.019      0.030      0.050    0.015   0.011               2000
# ease            0.041      0.062      0.092    0.030   0.022               2000
# bpr             0.038      0.058      0.085    0.028   0.020               2000
# sbert           0.022      0.034      0.052    0.016   0.012               2000
```

All models evaluated on the **same sampled users** (same seed), so the comparison is fair. The output is a pandas DataFrame — easy to write to CSV or display in a notebook.

---

## `bootstrap_ci(...)` — confidence intervals

For statistical comparisons across models, use bootstrap CIs:

```python
result = evaluate(my_model, track="warm", return_per_user=True)
per_user_recalls = result["per_user"]["recall@10"]

mean, lo, hi = bootstrap_ci(per_user_recalls, n_bootstrap=1000, ci=0.95)
print(f"Recall@10: {mean:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")
# → Recall@10: 0.0304  [95% CI: 0.0284, 0.0325]
```

1000 bootstrap iterations is standard. If two models' CIs don't overlap, the difference is statistically significant.

---

## Things to know

### Determinism

Same `seed` → same sampled users → same numbers. You can re-run an evaluation 5 times and get bit-identical results. Required for reproducible comparisons.

### `exclude_seen` is enforced by your model

The harness calls `model.recommend(user_id, k=max(k_values), exclude_seen=True)`. Your model is responsible for actually excluding items the user has seen in train. The reference `PopularityRecommender` does this in 4 lines — copy that pattern.

### Cold-track Recall@K = 0 is correct for CF models

If your BPR / EASE / MF model returns 0 on `track="cold"`, that's not a bug. The cold test items have zero rater history in train, so CF models have no signal. **Only content-aware models (TF-IDF, Sentence-BERT, hybrid, two-tower) can produce non-zero Recall on cold.**

If you're evaluating a CF model and expect non-zero cold numbers — you're misunderstanding the setup. Re-read the data flow section.

### The `candidate_filter` parameter is a Week 4 hook

Ignore for now. It's the integration point for the Stage 2 reranker (constraint-aware reranking). When you build a Stage 1 model in Week 2, just don't pass it.

### Test-set caching

The harness caches loaded test sets. So calling `evaluate(model, track="warm")` 50 times in an α-sweep loads the test set ONCE, not 50 times. If you need to clear the cache for some reason (e.g., you changed the data files):

```python
from src.eval.harness import clear_cache
clear_cache()
```

---

## Common mistakes

**1. Training your model on the test set.**
```python
# WRONG — leaks the test set into training
full_train = load_train_interactions()
my_model = MyRecommender().fit(full_train)   # ❌
```
```python
# RIGHT — hold out the warm test set first
full_train = load_train_interactions()
train, _ = time_based_split(full_train, holdout_per_user=1)
my_model = MyRecommender().fit(train)   # ✓
```

**2. Reading `interactions_test.csv` directly in your model.**
Don't. Your model never reads the test files. The harness does that. If you find yourself importing `load_test_interactions` in your model file, you're doing something wrong.

**3. Skipping `exclude_seen`.**
The harness passes `exclude_seen=True` by default. If your model ignores that parameter and recommends items the user has already rated, your Recall@K will look artificially high (you're "predicting" the user's history). Always honor `exclude_seen`.

**4. Returning fewer than `k` items.**
The harness asks for `max(k_values)` items (default 20). If you return only 10, Recall@20 will be artificially low. Aim to return at least `k` items unless your model fundamentally can't (e.g., very thin catalog).

**5. Non-deterministic models without seeds.**
If your model uses random init / negative sampling / dropout and you don't seed it, you get different numbers every run. Seed everything via `self.seed` in `__init__`.

---

## When something doesn't work

1. **Run the harness on `PopularityRecommender` first.** If popularity warm Recall@10 ≠ ~3%, your environment is off. Re-clone or reinstall deps.
2. **Try `n_users=100`** for faster iteration during debugging.
3. **Set `return_per_user=True`** and inspect the per-user DataFrame — find users your model is getting wrong and look at their data.
4. **Compare your model's `recommend()` output to popularity's directly.** Different shape? Different types? Different exclusion behavior?

Floor numbers from `PopularityRecommender` you can sanity-check against:

| Track | Recall@10 (full test) |
|---|---|
| warm | 0.0304 (95% CI [0.0284, 0.0325]) |
| cold | 0.0000 (by construction) |

---

## You're set

If you can run the quickstart in `week2_onboarding.md` §9 against your model and get a non-degenerate number, you're done with the eval setup. Build the model, run the harness, ship it.
