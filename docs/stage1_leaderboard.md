# PantryPlate — Stage 1 leaderboard

**Living document.** Update this when you add a new Stage 1 model or re-run the eval harness with different settings.

- **Canonical data**: [`data/processed/stage1_leaderboard.csv`](../data/processed/stage1_leaderboard.csv) (machine-readable; one row per model)
- **Reference notebooks**: [`notebooks/sentence_bert_smoke.ipynb`](../notebooks/sentence_bert_smoke.ipynb), [`notebooks/tag_svd_smoke.ipynb`](../notebooks/tag_svd_smoke.ipynb)
- **Last updated**: 2026-06-16 (added statistical-significance + CF-tuning section)

---

## Current numbers

All values are **Recall@K × 100** (i.e. percentage). **Bold** marks the per-column winner.

| Model | Val @10 | Val @100 | Warm @10 | Warm @100 | Cold @10 | Cold @100 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Popularity | 0.000 | 0.000 | **2.950** | 11.550 | 0.000 | 0.000 |
| BPR | 0.000 | 0.000 | 2.700 | **12.250** | 0.000 | 0.000 |
| EASE | 0.000 | 0.000 | 2.900 | 8.100 | 0.000 | 0.000 |
| ALS (`implicit`) | 0.000 | 0.000 | 2.050 | 7.950 | 0.000 | 0.000 |
| Hybrid linear (α=0.7, EASE+TagSVD) | 0.000 | — | 1.200 | — | 0.000 | — |
| Hybrid linear (α=0.5, EASE+TagSVD) | 0.000 | — | 1.200 | — | 0.000 | — |
| Hybrid linear (α=0.3, EASE+TagSVD) | 0.000 | — | 0.900 | — | 0.000 | — |
| Hybrid linear (α=0.7, EASE+SBERT) | — | — | 1.750 | 4.450 | 0.019 | 0.087 |
| Hybrid linear (α=0.5, EASE+SBERT) | — | — | 1.600 | 4.200 | 0.019 | 0.087 |
| Hybrid linear (α=0.3, EASE+SBERT) | — | — | 1.400 | 3.950 | 0.019 | 0.087 |
| Tag SVD content | 0.017 | 0.288 | 0.000 | 0.500 | 0.010 | 0.164 |
| **SBERT content** | **0.169** | 0.373 | 0.150 | 0.700 | **0.087** | **0.452** |
| SBERT + Tag SVD (w=0.25) | 0.102 | **0.458** | 0.100 | 0.800 | 0.087 | 0.366 |

`—` = not yet computed. CF @100 for BPR/EASE/ALS independently verified by reviewer; hybrid @100 pending.

Missing entries (open workstreams):

| Model | Owner | Status | Notes |
|---|---|---|---|
| Two-tower neural | TBD | ⬜ | Week 4-5 deep + multimodal |
| SASRec / GRU4Rec (stretch) | TBD | ⬜ | Week 8 if green |

---

## How to read this

### The three tracks (locked decision §1b)

- **Validation** — authors' pre-split `interactions_validation.csv` (5,900 users, all 4+ stars). Use to tune hyperparameters; reserve warm/cold for final reporting so we don't tune on test. *Validation items are cold by construction (zero raters in train).*
- **Warm** — our own time-based LOO holdout on the authors' train file (Track A). ~24,382 users. Held-out items have rater history. All Stage 1 models compete here.
- **Cold** — authors' pre-split `interactions_test.csv` (10,393 users). Held-out items have **zero** raters in train. Only content-aware models can meaningfully compete; CF-only models score ~0 by construction.

### The two K values

Different K's answer different questions in our two-stage pipeline:

```
Stage 1 (candidate generator) → top-100 → Stage 2 (constraint reranker) → top-5/10 → user
```

- **Recall@10** = "did the held-out item appear in the model's top-10?" Conventional recsys metric. Useful for isolated model comparison (every paper reports this).
- **Recall@100** = "did the held-out item appear in the model's top-100?" The candidate-pool coverage metric. Pipeline-relevant — Stage 2 will re-rank the pool, so what matters is whether the right item is *somewhere* in it. Stage 2 can lift a rank-50 item into the top-10, but it cannot conjure a miss.

A model can win @100 and lose @10 (and vice versa). SBERT+Tag SVD is a clear example: loses to pure SBERT at @10 because Tag SVD pulls warm-popular candidates up, but wins on val/warm @100 because Tag SVD adds variety that broadens the pool.

For deck framing: report both. @10 is the standard; @100 is the pipeline argument.

---

## Findings so far

1. **No CF model significantly beats Popularity on warm.** Popularity 2.95%, EASE 2.90%, BPR 2.70% @10 — the 95% CIs overlap almost entirely and paired Wilcoxon gives p > 0.4 (see § Statistical significance). This holds **even after tuning** (17 EASE/BPR configs, all p > 0.07). On sparse Food.com LOO, collaborative filtering offers no significant gain over a popularity prior — a rigorous null result, not under-tuning.
2. **At @100, BPR has the widest pool** (12.25% vs Popularity 11.55%), but the gap is **not statistically significant** (p = 0.16). We still use BPR as the α-sweep warm generator because it's the numerically-best candidate pool, while noting Popularity is statistically equivalent. EASE collapses at @100 (8.10%); ALS is weakest (2.05 / 7.95).
3. **SBERT is the cold winner.** 0.087% @10, 0.452% @100 (~10× random chance). The content stream produces meaningful candidate coverage on novel recipes.
4. **Tag SVD alone is weak** (0.017% val @10, 0.010% cold @10), but its *features* become useful when concatenated with SBERT — they add complementary candidates.
5. **Layer 4 (SBERT + Tag SVD concat) wins val/warm @100, loses cold @100.** Mixed verdict that supports model-routing: different upstream content model per query context.
6. **Pure-CF cold = 0 by construction.** Not a bug. Popularity, BPR, EASE all score 0 on Track B (and on validation, which is also cold-by-construction) — those items have no rater history.
7. **Linear hybrids lose to routing — both content sides.** EASE+TagSVD: best 1.2% warm @10 (TagSVD has zero warm signal → dilutes EASE). EASE+SBERT (now measured): better than the TagSVD blend (1.75% warm @10 / 4.45% @100 at α=0.7) AND gets *non-zero cold* (0.019% @10, 0.087% @100) — but still **loses to BPR on warm** (2.70/12.25) and **to SBERT on cold** (0.087/0.452). It beats neither parent on its home track. **Conclusion: model routing (BPR for warm, SBERT for cold) beats a single linear hybrid** — the architectural takeaway. (Cold is flat across α for the hybrid since cold items get content-only scoring.)

See the notebooks linked at the top for the full iteration log (SBERT layers 2/3/4 sweeps with executed results).

---

## Statistical significance & CF tuning

Point estimates alone don't say whether differences are real. We report **bootstrap 95% CIs**
on every mean and **paired Wilcoxon signed-rank** tests for model-vs-model comparisons
(non-parametric, paired on the same users — the standard recsys choice). Tooling:
`src/eval/significance.py`. Reproduce: `uv run python -m src.eval.run_significance`
(deterministic, seed=42; warm = 2,000-user sample, cold = full 10,393).

### 95% bootstrap CIs — Recall@10

| Model | Mean | 95% CI |
|---|---|---|
| Popularity (warm) | 2.95% | [2.25, 3.70] |
| EASE (warm) | 2.90% | [2.15, 3.65] |
| BPR (warm) | 2.70% | [2.00, 3.40] |
| SBERT (cold) | 0.087% | [0.038, 0.144] |
| Tag SVD (cold) | 0.010% | [0.000, 0.029] |

The three warm CIs overlap almost completely.

### Paired Wilcoxon (Recall@10)

| Comparison | Δ (pp) | p-value | Significant @0.05 |
|---|---|---|---|
| BPR vs Popularity (warm) | −0.25 | 0.398 | ❌ |
| EASE vs Popularity (warm) | −0.05 | 0.901 | ❌ |
| BPR vs EASE (warm) | −0.20 | 0.612 | ❌ |
| BPR vs Popularity (warm @100) | +0.70 | 0.157 | ❌ |
| **SBERT vs Tag SVD (cold)** | **+0.077** | **0.011** | ✅ |

**Only the cold-track content comparison is significant** — SBERT genuinely beats the
structured-feature baseline on novel recipes. Every warm CF comparison is a statistical tie.

### CF tuning — does any config beat Popularity? (No.)

Swept EASE λ ∈ {50, 100, 250, 500, 1000} and BPR over factors × iters × lr (12 configs).
Warm Recall@10 and the paired Wilcoxon p-value vs Popularity for each:

| Model family | Best config | warm@10 | warm@100 | p vs Popularity |
|---|---|---|---|---|
| Popularity (baseline) | — | 2.95% | 11.55% | — |
| EASE (λ sweep) | λ=500 | 2.95% | 8.50% | 1.00 |
| BPR (grid) | k=200, it=1000, lr=.01 | 2.80% | 12.50% | 0.65 |
| BPR (best @100) | k=200, it=500, lr=.01 | 2.70% | 12.75% | 0.35 |

**All 17 configs: p > 0.07 — none significantly beats Popularity.** The warm CF ≈ Popularity
result is a property of the sparse data, not under-tuning.

### What this means

- **CF ≈ Popularity on warm** (tuned, significance-tested) → the architectural leverage is
  *not* Stage-1 CF. It's Stage-2 constraint handling and cold-start content.
- **SBERT significantly helps cold-start** (p=0.011) → the one statistically real model win,
  and it backs the routing decision (content for novel recipes).
- Absolute numbers are small because we rank against the **full catalogue** (~20K–231K items;
  random Recall@10 ≈ 0.05%), not sampled negatives. Report the lift-over-random framing.

---

## How to add a new row

If you're claiming a model from [`docs/week2_onboarding.md`](week2_onboarding.md) §4b:

### 1. Implement the model

Follow the contract from locked decision §11:

```python
class YourRecommender:
    def __init__(self, ...): ...
    def fit(self, train_df: pd.DataFrame) -> "YourRecommender": ...
    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]: ...
```

Reference impls: [`src/models/popularity.py`](../src/models/popularity.py), [`src/models/sentence_bert.py`](../src/models/sentence_bert.py).

### 2. Compute the same numbers as everyone else

```python
from src.data.loader import load_train_interactions, time_based_split
from src.eval.harness import evaluate

train = load_train_interactions()
train_part, _ = time_based_split(train, holdout_per_user=1)

# Warm eval: fit on the post-LOO train, NOT the full train
# (full train contains the held-out item → exclude_seen=True silently drops it → 0% recall).
model_warm = YourRecommender(**config).fit(train_part)
r_warm = evaluate(model_warm, track='warm', k_values=(10, 100), n_users=2000, seed=42)

# Validation + cold: fit on full train (those items aren't in train anyway)
model_full = YourRecommender(**config).fit(train)
r_val  = evaluate(model_full, track='validation', k_values=(10, 100), seed=42)
r_cold = evaluate(model_full, track='cold',       k_values=(10, 100), seed=42)
```

The harness applies `exclude_seen=True` internally. Use `seed=42` and `n_users=2000` for warm so results are comparable across teammates.

### 3. Update this doc and the CSV

- Add a row to the table above with your model's numbers (in percent — multiply by 100).
- Append the same row to `data/processed/stage1_leaderboard.csv` so downstream code keeps working.
- Open a PR; the table is the source of truth on `main`.

### Optional but useful

- Write a small pipeline-validation notebook in `notebooks/` (mirror `sentence_bert_smoke.ipynb` or `tag_svd_smoke.ipynb`). Future teammates can open it side-by-side with the others for direct comparison.
- Briefly note in the **Findings so far** section above if you discovered anything non-obvious (negative results count too — see the SBERT layer iterations).

---

## What's not measured here

- **NDCG@K, MRR**: also in the CSV and notebooks; omitted from this summary table for readability. The deck headline plot will use Recall@K + NDCG@K; MRR is a per-user sanity statistic.
- **Useful Recall@K**: end-to-end metric from the proposal (recall × pantry-feasible × macro-near × diet-compliant). This is a *Stage 2* metric. Lives in `src/eval/useful_recall.py` (Week 4 work). The leaderboard above is Stage 1-only.
- **Per-user latency, memory footprint**: not yet measured. Will matter once the demo widget is wired up.
- **Cold-track LOO on warm cohort**: not currently considered. The "cold" track means *item-cold* (Track B from the authors), not *user-cold*.

---

## Pointers

- Locked data and modeling decisions: [`docs/data_decisions.md`](data_decisions.md)
- Eval harness usage: [`docs/eval_harness_usage.md`](eval_harness_usage.md)
- Team onboarding + model coordination table: [`docs/week2_onboarding.md`](week2_onboarding.md)
- Feature pipeline: [`src/data/features.py`](../src/data/features.py)
- Personas: [`data/personas/`](../data/personas/)
