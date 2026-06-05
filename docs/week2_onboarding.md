# Week 2 Onboarding — PantryPlate

**For teammates joining at the start of Week 2 (model-building phase).**

This doc gets you from zero to *"I can build a Stage 1 model that plugs cleanly into our evaluation harness"* in about 30 minutes of reading.

---

## 0. The project in one paragraph

PantryPlate is a multi-constraint recipe recommender. Stage 1 generates candidate recipes from the user's rating history (collaborative filtering or content-based). Stage 2 reranks those candidates using four scores — taste, pantry overlap, nutrition fit, dietary restrictions — combined via α-weighting. The **research object** is studying how recommendations shift across the (αt, αp, αn) simplex. **Stage 1 is interchangeable; Stage 2 is constant.** Everything you build in Week 2 is a Stage 1 model.

---

## 1. Read these in order (~30 min total)

| What | Where | Why |
|---|---|---|
| Proposal deck (final) | `PantryPlate_Proposal_v2.pdf` | The pitch — read the whole thing |
| EDA + feasibility notebook | `notebooks/week1_eda.ipynb` | What we learned about the data; verdicts per pillar |
| Walkthrough notebook | `notebooks/week1_walkthrough.ipynb` | Runnable code tour of the data loaders, metrics, popularity baseline |
| Locked decisions register | `docs/data_decisions.md` | 11 decisions with evidence — read at least the summary table at the bottom |
| **Harness usage guide** | `docs/eval_harness_usage.md` | How to plug your model into evaluation (5-min read) |

If you only have 10 minutes: read **this doc** + the **summary table in data_decisions.md** + skim the **Useful Recall@K slide** in the proposal.

---

## 2. ⭐ Data files — what each one is and what touches what

**This is the section to read twice.** Misunderstanding here causes the most bugs in Week 2.

The Food.com Kaggle dataset ships with **3 pre-split files** from the authors (Majumder et al. 2019). We use all three differently:

```
data/raw/
├── interactions_train.csv       (698,901 rows)   ← MODEL TRAINING DATA
├── interactions_validation.csv  (7,023 rows)     ← optional hyperparameter tuning
├── interactions_test.csv        (10,393 rows)    ← Track B (cold-item) evaluation
└── RAW_interactions.csv         (1.13M rows)     ← raw; we use the pre-split instead
```

### How they map to our dual-track evaluation

```
                    interactions_train.csv  (698,901 rows after 0-star drop)
                                │
                                │   apply time_based_split(holdout_per_user=1)
                                │   (this is deterministic — same input → same split)
                                ▼
                ┌───────────────────────────────────┐
                │                                   │
        ┌───────▼──────────┐              ┌─────────▼──────────┐
        │  warm TRAIN      │              │  warm HOLDOUT      │
        │  ~658K rows      │              │  ~24K positives    │
        │                  │              │                    │
        │  YOUR MODEL      │   FITS ON    │  TRACK A test set  │
        │  TRAINS HERE     │              │  (warm-item)       │
        └──────────────────┘              └────────────────────┘


        interactions_test.csv (authors' file, 10,393 cold positives)
                                │
                                ▼
                ┌────────────────────────────────────┐
                │  TRACK B test set (cold-item)      │
                │  Items have ZERO raters in train   │
                │  Only content-aware models compete │
                └────────────────────────────────────┘


        interactions_validation.csv (7,023 positives)
                                │
                                ▼
        ┌─────────────────────────────────────────────────────────┐
        │  NOT USED BY THE HARNESS                                │
        │  Available via load_validation_interactions() if your   │
        │  model has hyperparameters to tune during development.  │
        └─────────────────────────────────────────────────────────┘
```

### One-liner for each file

| File | What it is | Who touches it |
|---|---|---|
| `interactions_train.csv` | The training cohort — 25K active users with their rating histories | Your `model.fit()` (after holdout) + the warm-track eval uses the holdout |
| `interactions_test.csv` | 10K cold-item held-out positives | The harness, for `track="cold"`. Your model code never reads this. |
| `interactions_validation.csv` | 7K validation positives | Optional — use during model development if tuning hyperparameters |
| `RAW_interactions.csv` | The unsplit raw 1.13M | Generally not used — kept for exploration only |

### What you (the modeler) should literally do

```python
from src.data.loader import load_train_interactions, time_based_split

# Step 1: get the training data
full_train = load_train_interactions()  # 681,944 rows after 0-star drop
train, _ = time_based_split(full_train, holdout_per_user=1)
# train has ~658K interactions; the held-out portion becomes the warm test set
# (don't worry about the holdout — the harness handles it deterministically)

# Step 2: fit your model on `train`
my_model = MyRecommender().fit(train)

# Step 3: evaluate
from src.eval.harness import evaluate
warm = evaluate(my_model, track="warm")  # warm-track Recall@K, NDCG@K, MRR
cold = evaluate(my_model, track="cold")  # cold-track (only meaningful for content-aware models)
```

You **never** read `interactions_test.csv` or `interactions_validation.csv` from your model code. The harness does that for you, behind the scenes, against the same fixed holdout your model didn't see.

### "Wait, but won't my model and the harness use different holdouts?"

No. `time_based_split` is **deterministic** — it splits based on the most-recent positive rating per user, which is a fixed property of the data, not a random sample. Calling `time_based_split(full_train, holdout_per_user=1)` twice gives identical (train, holdout) pairs. The harness uses the same call internally, so its `holdout` matches what `fit()` didn't see.

---

## 3. The Stage 1 model contract

Every Stage 1 model in the project must satisfy this minimal interface:

```python
class YourRecommender:
    def fit(self, train_df: pd.DataFrame) -> "YourRecommender":
        """Learn from training data. Return self for chaining."""
        ...
        return self

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[int]:
        """Return top-K recipe IDs for this user, highest-ranked first."""
        ...
```

That's it. Two methods. `train_df` will have columns `['user_id', 'recipe_id', 'date', 'rating', 'u', 'i']` — use whichever you need:

- `user_id`, `recipe_id`: original Food.com IDs (use these for joining with recipe metadata in `RAW_recipes.csv`)
- `u`, `i`: authors' remapped 0-indexed integer IDs (useful for matrix-factorization implementations)
- `date`: datetime — useful for time-aware models
- `rating`: 1-5 integer — most implicit-feedback models will ignore this

The `PopularityRecommender` in `src/models/popularity.py` is the reference implementation. Copy that pattern.

---

## 4. The 11 locked decisions you must respect

If you ever wonder *"can I change X?"*, the answer is *"check `docs/data_decisions.md` — if it's in the summary table, no, don't change it."*

The 11 in one-liner form:

1. **Training cohort = authors' pre-split train** (25K active users / 681K interactions after 0-star drop). Use `load_train_interactions()`.
2. **Dual-track evaluation**: warm-item LOO + cold-item from authors' test. The harness handles both via `track="warm"` / `track="cold"`.
3. **Drop 0-star ratings** — they're "review without rating" entries. `load_train_interactions()` drops them by default.
4. **Positive rating threshold = 4 stars** — for LOO holdout logic.
5. **Stage 1 model menu** is set: Popularity, MF, EASE, BPR, Content TF-IDF, Sentence-BERT, Hybrid linear, Two-tower neural (+ SASRec as stretch). Don't add models unless coordinated.
6. **CF-only models can be restricted to recipes with ≥10 ratings** for memory reasons (your call per model — see decision 6 in the register for the trade-off).
7. **Pantry reranker score** is non-staple overlap (continuous); **Useful Recall pantry condition** is `missing_count ≤ 3`. Both live in `src/utils/staples.py`. You don't touch these in Stage 1.
8. **Diet enforcement** is hard-filter (Week 4 work, not yours).
9. **Nutrition clipping** at (5000 kcal, 1000% PDV) — already done in the loader.
10. **Persona pantry size** = 25-35 user-specific items. Staples are project-wide. Persona JSONs live in `data/personas/`.
11. **Eval harness** is the only sanctioned evaluation path. ✓ Built; see `src/eval/harness.py`.

---

## 4b. Pick a model to build — coordination table

**How to claim a model**: edit the **Owner** column in this table, set **Status** to 🟡, push. First push wins. If a conflict happens, sort it out at the next standup. The table is the single source of truth for who's building what.

### Stage 1 models — coordination

| Status | Priority | Model | Family | Effort | Difficulty | Source / hint | Why we want it | Owner |
|---|---|---|---|---|---|---|---|---|
| ✅ | ref | Popularity | CF baseline | done | — | `src/models/popularity.py` (reference impl) | Baseline floor (~3% Recall@10). Every other model should comfortably beat this. | — |
| 🟡 | 1 | Sentence-BERT content | Content-aware | 4-6 hr | Low-Med | `sentence-transformers` library; cache embeddings to `data/processed/` | Cold-track contender. Semantic embeddings; works on both tracks. PoC for end-to-end pipeline validation. | Ikhwan |
| ⬜ | 2 | **BPR** (Cornac) | CF implicit | 4-6 hr | Low | `cornac.models.BPR`; defaults are good | Course-syllabus model. Expected to dominate warm-track CF. Pair with EASE for paradigm comparison. | TBD |
| ⬜ | 3 | **EASE** | CF implicit | 3-4 hr | Low | `cornac.models.EASE` if available, else ~15 lines of numpy. One hyperparam (λ). | Closed-form, no SGD. Often beats fancier models on classic recsys benchmarks. Very fast to train. | TBD |
| ⬜ | 4 | Content TF-IDF | Content-aware | 3-4 hr | Low | sklearn `TfidfVectorizer` on (ingredients + tags + name); cosine sim | Cold-track reference (sparse representation). Completes the content paradigm comparison vs SBERT. | TBD |
| ⬜ | 5 | **Hybrid linear** | Combination | 2-3 hr | Low | `α · cf_score + (1-α) · content_score`, normalized | **Expected overall winner.** Combines best CF (priority 2 or 3) with best content (priority 1 or 4). Depends on at least one CF + one content model being done first. | TBD |
| ⬜ | 6 | MF / ALS (Cornac) | CF rating-prediction | 3-4 hr | Low | `cornac.models.MF` or `cornac.models.WMF` | Reference baseline — included for paradigm comparison. Expected to lose on Recall@K (rating distribution is heavy 4-5★). | TBD |
| ⬜ | 7 | Two-tower neural | Deep DL | 8-12 hr | High | PyTorch from scratch; user tower + item tower with content features; co-trained with BPR-style loss | Course week W4-W5 coverage. Only if a PyTorch-comfortable teammate has a clear ~12-hour window. **Drop if time tight.** | TBD |
| ⬜ | stretch | SASRec / GRU4Rec | Sequential | 8-12 hr | High | PyTorch from scratch | Course week W8 coverage. Only if everything above lands by mid-Week 3. | TBD |

### Notes on priority

The priority numbers reflect **what will most strengthen the project's story**, not what's easiest:

1. **Sentence-BERT** is the cold-track headliner. Without a content model, Track B Recall@10 will be 0% across the board — and the whole "dual-track" narrative falls apart.
2. **BPR + EASE** are the warm-track headliners. Implicit-feedback CF is expected to dominate warm; we want both for paradigm comparison.
3. **Content TF-IDF** is the warm-side content reference — completes the content paradigm comparison (sparse vs dense embeddings).
4. **Hybrid linear** is expected to be the overall winner on Useful Recall@K — it combines CF + content signals. **Don't start this before at least one CF + one content model is in.**
5. **MF / ALS** completes the paradigm comparison story ("we tried rating-prediction; here's why it doesn't win"). Lower priority since it's expected to lose.
6. **Two-tower** is the deep-learning representative. High effort; only commit if you have the PyTorch chops and the time.
7. **SASRec** is genuinely stretch — only if the rest of the team lands their models early.

### Realistic Week 2 commitment

For the team to ship a defensible results table by end of Week 3, we need at minimum:

**Minimum viable** (4 models): Popularity ✅ + Sentence-BERT + BPR + Hybrid linear

**Defensible** (6 models): + EASE + TF-IDF content

**Ambitious** (8 models): + MF/ALS + Two-tower neural

Pick a model that matches your bandwidth this week. **Better to ship 4 well-tested models than promise 8 and ship 4.**

### Other workstreams (not Stage 1 models)

These also need owners — coordinate at standup:

| Workstream | Effort | Owner |
|---|---|---|
| Stage 2 reranker (4 score functions + combiner) — Week 4 main | 8-10 hr | Ikhwan (with Sentence-BERT as the PoC) |
| α-sweep experiments + per-persona analysis | 4-6 hr | TBD |
| Results aggregation + comparison plots | 3 hr | TBD |
| Demo widget (Streamlit) | 6-8 hr | TBD |
| Slide deck updates with real Week 2-3 numbers | 2-3 hr | TBD |
| Physical prop (cook a recommended recipe) | 1-2 hr + cook time | TBD |
| Week 2 progress note (end of week) | 1 hr | TBD |

---

## 5. Numbers you should beat (warm-track floors)

Run the harness on `PopularityRecommender` once locally to anchor your expectations:

| Metric | Popularity baseline | Your model should… |
|---|---|---|
| Warm Recall@10 | **3.04%** (95% CI [2.84%, 3.25%]) | Beat this comfortably (CF: 5-10%; hybrid: 6-12%) |
| Warm NDCG@10 | 1.47% | Beat by similar margin |
| Warm MRR | 1.13% | Beat |
| Cold Recall@10 | **0.00%** | (CF models: also 0% — by construction. Content/hybrid: produce any non-zero number → win.) |

If you train EASE or BPR and your warm Recall@10 is 1% — something's wrong. Debug before tuning.

If you train a content model and your cold Recall@10 is 0% — something's wrong. Content models should produce a non-zero number on cold by design.

---

## 6. Conventions to follow

### File structure

```
src/models/
├── popularity.py        ← already there, reference impl
├── your_model.py        ← put your model class here
└── __init__.py
```

One file per model. Class name = TitleCase of the model. Stick the `.fit()` and `.recommend()` methods on the class.

### Naming

- Model class: `EASERecommender`, `BPRRecommender`, `SentenceBERTContentRecommender`
- File: `ease.py`, `bpr.py`, `sentence_bert_content.py`

### Determinism

Seed any random number generators inside your model. The harness will run with `seed=42`. Your model needs to honor seeded fits for reproducibility.

```python
class MyRecommender:
    def __init__(self, seed: int = 42):
        self.seed = seed
        # use self.seed in any sampling, negative sampling, weight init
```

### Where to commit code

- Model code: `src/models/your_model.py`
- Tests: `tests/test_your_model.py` (basic sanity tests — fit doesn't crash, recommend returns a list of ints, etc.)
- Don't commit model checkpoints or intermediate files. Add to `.gitignore` if needed.

### Run commands from the project root

All commands (tests, harness, scripts) should be run from the project root directory — the directory containing `pyproject.toml`. If you're in `src/models/` and run `python my_model.py`, imports like `from src.data.loader import ...` will fail.

```bash
# WRONG
cd src/models/
uv run python my_model.py    # ModuleNotFoundError: src

# RIGHT
cd /path/to/CS608Project2/
uv run python -m src.models.my_model
# or use the harness which already does the path setup
```

### Git workflow

We're a small team (5 people) for 3 weeks — no need for elaborate branching. **Everyone works on `main`.**

**Before pushing**:
1. `git pull --rebase` — gets latest, rebases your commits cleanly on top
2. Re-run tests (`uv run pytest tests/ -q`) to make sure nothing broke
3. `git push`

**Commit message conventions** (lightweight):

```
<verb> <what> — <why if non-obvious>

Examples:
  Add BPR model — Alekhya, ready for harness
  Claim EASE in onboarding table
  Fix cold-track flag handling in harness
  Refactor staples — drop chicken broth for vegan personas
```

**To claim a model from §4b's table**:

```bash
git pull
# edit docs/week2_onboarding.md — change Status to 🟡 and add your name to Owner
git add docs/week2_onboarding.md
git commit -m "Claim <model name> — <your name>"
git push
```

If two people race to claim the same model: the second push will fail on merge conflict. Pull, pick a different model, push again. No bad blood — just standup-resolve over coffee.

**Conflict on shared files**: if you and a teammate both edited the same file:
1. `git pull --rebase` shows the conflict
2. Open the file, look for `<<<<<<<` / `=======` / `>>>>>>>` markers
3. Edit to resolve, save
4. `git add <file>` then `git rebase --continue`
5. `git push`

If you're unsure how to resolve, ping the team chat before force-anything.

---

## 7. Stage 2 (reranker) is not your problem

If you see references to `s_pantry`, `s_nutrition`, `s_diet`, α-weights, or the reranker — that's Week 4 work. Your Stage 1 model produces top-K recommendations. The reranker happens downstream, in `src/reranker/`, on the candidates your model produces.

Concretely: your model's output is the `s_taste` ingredient that goes into the reranker. You don't need to know anything else about Stage 2.

---

## 8. Common questions / gotchas

**Q: Should I fit on the validation set or test set?**
No. Fit only on the warm training data. The validation set is optional for tuning your model's hyperparameters during development. The test sets are used by the harness only.

**Q: My BPR / EASE model gets ~0 on cold-track Recall. Bug?**
No, that's correct. CF-only models have zero signal for cold items (the items have zero raters in train). This is what cold-track is designed to demonstrate. Content models will do better here.

**Q: My content model gets lower warm Recall than BPR. Should I worry?**
No, that's also expected. Content models lose on warm to CF models because they can't capture user-specific personalization. The hybrid is the architecture that handles both.

**Q: How long should training take?**
Reasonable budgets per model: Popularity (1 sec), EASE (1-5 min depending on item count), BPR (5-20 min), Sentence-BERT (10-30 min for embedding all recipes once, then fast inference), Two-tower (30-60 min). If yours takes hours, debug or reduce scope.

**Q: I want to compare my model against another teammate's. How?**
```python
from src.eval.harness import compare_models
df = compare_models(
    {"ease": ease_model, "sentence_bert": sbert_model, "popularity": pop_model},
    track="warm",
)
print(df)
```
That's the official way. Don't reinvent.

**Q: How do I get bootstrap CIs on my metric?**
```python
warm = evaluate(my_model, track="warm", return_per_user=True)
mean, lo, hi = bootstrap_ci(warm["per_user"]["recall@10"], n_bootstrap=1000)
print(f"Recall@10: {mean:.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")
```

**Q: I get `ModuleNotFoundError: No module named 'src'` when running my model file directly.**
You're running from the wrong directory. All commands should be run from the project root (the directory with `pyproject.toml`). Use `uv run python -m src.models.my_model` from the root, or use the harness which sets the path up for you.

**Q: Cornac install fails with a build error.**
Cornac sometimes needs Cython at build time. Try `uv add cython && uv add cornac` or check the [Cornac install docs](https://github.com/PreferredAI/cornac#installation). On macOS, you may need `brew install libomp` first.

**Q: First import of `sentence-transformers` takes 30+ seconds — is that normal?**
Yes, that's the model loading from disk. Subsequent imports are fast (just a few ms). To speed up the first load even further, you can pre-download the model: `uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"` once. Cache embeddings to `data/processed/` so you don't re-encode all 230K recipes every time you re-fit.

**Q: Tests are failing after I pull latest.**
Run `uv sync` to refresh dependencies (someone may have added a new package). Then `uv run pytest tests/ -q` again. If still failing, check git diff for unexpected changes to test files; if you didn't touch tests, ping the team.

**Q: The ingredient "flmy" is showing up in my top ingredients list. Typo?**
No — that's the dataset's canonical form for "flour". The Food.com authors' ingredient canonicalization stripped "ou" → "y" via consonant clustering. Internally consistent; don't try to "fix" it. The `STAPLES` set in `src/utils/staples.py` already includes both `"flour"` and `"flmy"` to handle this.

**Q: How do I evaluate my content model on cold-track specifically?**
Just call `evaluate(my_model, track="cold")`. The harness routes to `interactions_test.csv` automatically. Your model just needs to be able to score arbitrary recipe IDs (including ones it hasn't seen via rater history) — i.e., it needs to use recipe metadata from `RAW_recipes.csv` rather than just rating co-occurrence.

**Q: My model is sometimes returning the same recipe multiple times in top-K. Bug?**
Yes — your scoring logic isn't deduplicating. Make sure you return unique recipe IDs in your top-K list.

**Q: Where do I ask questions?**
Daily standup or team chat. If you're stuck for more than 30 min, ping the team — likely someone has hit the same wall.

---

## 9. Quickstart — copy this to start

```python
# my_model.py — a stub to extend
import numpy as np
import pandas as pd


class MyRecommender:
    def __init__(self, seed: int = 42):
        self.seed = seed
        # initialize anything you need

    def fit(self, train_df: pd.DataFrame) -> "MyRecommender":
        # Step 1: extract whatever you need from train_df
        # Step 2: train / pre-compute / index
        # Step 3: store internal state
        return self

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> list[int]:
        # Step 1: compute scores for all candidate recipes
        # Step 2: filter out items the user has already seen if exclude_seen
        # Step 3: return top-k as list of int recipe_ids
        return []
```

```python
# eval_my_model.py — copy this to evaluate
from src.data.loader import load_train_interactions, time_based_split
from src.eval.harness import evaluate, bootstrap_ci
from src.models.my_model import MyRecommender

# 1. Build training data
full_train = load_train_interactions()
train, _ = time_based_split(full_train, holdout_per_user=1)

# 2. Fit
model = MyRecommender(seed=42).fit(train)

# 3. Evaluate (full test set — fast)
warm = evaluate(model, track="warm", return_per_user=True)
cold = evaluate(model, track="cold")

# 4. Print
print(f"WARM  Recall@10 = {warm['recall@10']:.4f}  ({warm['n_users_evaluated']:,} users)")
print(f"COLD  Recall@10 = {cold['recall@10']:.4f}  ({cold['n_users_evaluated']:,} users)")

# 5. Bootstrap CI on warm Recall@10
mean, lo, hi = bootstrap_ci(warm["per_user"]["recall@10"])
print(f"WARM  Recall@10 95% CI: [{lo:.4f}, {hi:.4f}]")
```

That's the full loop. Two files. Build your model, evaluate, iterate.

---

## 10. When you're done

1. Run the full test suite (`uv run pytest tests/ -q`). Should be 98+ passing including yours.
2. Add 2-3 short unit tests for your model in `tests/test_your_model.py` (fit doesn't crash, recommend returns list of ints, exclude_seen works).
3. Run the evaluation and check the warm Recall@10 is above the popularity floor (3.04%). If not, debug.
4. Push to the shared branch. Update the team chat with your model's warm + cold numbers.
5. Take a break. You're done with Stage 1 for this lane.

Welcome to the project.

---

## 11. Glossary

Recsys terminology in case it's unfamiliar:

### Recsys fundamentals

- **CF (Collaborative Filtering)** — recommender approach that learns from user-item interaction patterns (who rated what), not from item content. Examples: MF, EASE, BPR.
- **Content-based recommender** — recommends items based on item features (ingredients, tags, text) rather than user behavior. Examples: TF-IDF cosine sim, Sentence-BERT.
- **Hybrid** — combines CF and content signals in a single model.
- **Implicit feedback** — using the *occurrence* of an interaction (e.g., "user rated this") as a binary positive signal, ignoring the rating value.
- **LOO (Leave-one-out)** — evaluation protocol where one interaction per user is held out for testing.

### Project-specific terms

- **Track A / warm-item** — evaluation where held-out items have rater history in train. Standard CF benchmark setup. Uses time-based LOO on `interactions_train.csv`.
- **Track B / cold-item** — evaluation where held-out items have ZERO raters in train. Tests cold-start generalization. Uses authors' `interactions_test.csv`.
- **`s_taste`, `s_pantry`, `s_nutrition`, `s_diet`** — the four constraint scores computed by Stage 2 reranker. See `docs/data_decisions.md` §7.
- **α-sweep** — systematic experiment that varies the (αt, αp, αn) weights across the simplex to study the trade-off curve.
- **Simplex** — the geometric surface where (αt, αp, αn) sum to 1. Different points on this surface represent different priorities.
- **Useful Recall@K** — the project's signature metric. A recommendation counts as "useful" if it's in top-K AND satisfies all constraints.
- **Staples-aware pantry score** — `s_pantry` is computed only on non-staple ingredients. Universal staples (salt, oil, flour, eggs, etc.) are assumed available; see `src/utils/staples.py`.

### Models

- **MF (Matrix Factorization)** — classic latent-factor CF. Predicts ratings as user_factor · item_factor.
- **ALS (Alternating Least Squares)** — optimization algorithm commonly used to fit MF.
- **BPR (Bayesian Personalized Ranking)** — implicit-feedback CF that learns to rank rated items above unrated items via pairwise loss. Doesn't predict rating values.
- **EASE (Embarrassingly Shallow Autoencoder)** — closed-form implicit-feedback model. Item-item linear, single matrix inversion, one hyperparam (λ).
- **TF-IDF** — sparse vector representation of text based on term frequency × inverse document frequency. Used for content-based recsys via cosine similarity.
- **Sentence-BERT (SBERT)** — pretrained transformer that maps text to dense semantic embeddings. Used for content-based recsys via embedding similarity.
- **Two-tower neural** — deep architecture with separate user and item towers. Item tower can incorporate content features for cold-item handling.
- **SASRec** — self-attentive sequential recommender. Uses transformer attention over the user's rating history.

### Tools

- **Cornac** — recsys library by Preferred AI with implementations of MF, BPR, EASE, etc. Salah et al. JMLR 2020.
- **`sentence-transformers`** — Python library wrapping Sentence-BERT models. Cached locally on first use.
- **uv** — Python package manager. Faster than pip, manages `.venv` automatically.

### Metrics

- **Recall@K** — of the items that are actually relevant, what fraction did the model rank in top-K? For LOO with 1 positive: Recall@K ∈ {0, 1} (binary).
- **NDCG@K (Normalized Discounted Cumulative Gain)** — rewards relevance with rank position; relevant items higher up = higher NDCG. Normalized to [0, 1].
- **MRR (Mean Reciprocal Rank)** — average of 1/rank-of-first-relevant-item. 1.0 if the model always puts the relevant item at rank 1.
- **PDV (Percent Daily Value)** — how nutrition values are stored in the dataset. E.g., `fat_pdv = 25` means "25% of the recommended daily fat intake," not "25 grams of fat."

### Data anatomy

- **Train cohort** — the 24,961 active users in `interactions_train.csv` (after dropping 0-star reviews).
- **0-star ratings** — entries where the user wrote a review but didn't give a rating. We drop these by default; see decision §2.
- **Staples** — universal kitchen items assumed available for every persona (salt, pepper, oil, flour, eggs, milk, butter, garlic, onion, etc.). Defined in `src/utils/staples.py`.
- **Persona** — a JSON file describing a user context (pantry, macros, restrictions). Used for evaluation and the demo. See `data/personas/README.md`.

That's the jargon. If you encounter a term that's not here, ping the team — it might be worth adding.
