# PantryPlate

> Multi-constraint recipe recommendation system — balancing taste, pantry, nutrition, and dietary restrictions.

A graduate Recommender Systems course project (CS608, Project 2). Studies how recipe recommendations evolve as users tune the relative importance of competing constraints.

**Status**: proposal submitted (3 Jun 2026); final presentation 24 Jun 2026. **Modeling + evaluation complete** — Stage 1 models, Stage 2 reranker, α-sweep, and significance tests all done. See `docs/stage1_leaderboard.md` for results and `notebooks/pantryplate_e2e.ipynb` for the full walkthrough.

> **AI assistants**: read `CLAUDE.md` first. It captures conventions, locked decisions, and pending workstreams. Then read `docs/week2_onboarding.md` for the comprehensive engineering onboarding.

---

## What this project does

PantryPlate is a two-stage recommender:

```
                      USER CONTEXT
       (rating history · pantry · macros · diet · α-weights)
                            │
                   ┌────────┴────────┐
                   │  OPTIONAL INPUT │
              📷 Fridge photo        │
              → Gemini Vision        │
              → detected ingredients │
                   └────────┬────────┘
                            │
                            ▼
          STAGE 1 · CANDIDATE GENERATION (real recsys)
    Popularity · MF/ALS · EASE · BPR · Tag SVD · Sentence-BERT · hybrid
            → returns top-100 candidates by predicted taste
                            │
                            ▼
        STAGE 2 · MULTI-CONSTRAINT RERANKING (X-factor)
   scores each candidate on (taste × pantry × nutrition × diet)
            combined via α-weighting → returns top-10
```

**Stage 1** is interchangeable (we evaluate 7 models). **Stage 2** is the project's research contribution — studying the (αt, αp, αn) simplex via empirical α-sweep across two evaluation tracks (warm-item CF + cold-item content-aware).

For the full pitch, read [`PantryPlate_Proposal.pdf`](PantryPlate_Proposal.pdf) (~3 min).

---

## Repository structure

```
.
├── PantryPlate_Proposal.pdf          # Final submitted proposal (Jun 3)
├── PantryPlate_Proposal.pptx         # Editable source for further iteration
│
├── data/
│   ├── raw/                          # Food.com CSVs (gitignored, ~1.5 GB)
│   ├── processed/                    # Cached intermediates (gitignored)
│   └── personas/                     # Persona JSONs for evaluation + demo
│
├── docs/
│   ├── week2_onboarding.md           # ⭐ START HERE if you're joining
│   ├── eval_harness_usage.md         # How to use src/eval/harness.py
│   ├── data_decisions.md             # 12 locked decisions with evidence
│   ├── project_brief.md              # Original project pitch (pre-implementation)
│   ├── week1_progress.md             # Week 1 wrap-up summary
│   └── proposal_deck_rebuild_brief.md
│
├── notebooks/
│   ├── week1_eda.ipynb               # Feasibility analysis + dual-track justification
│   └── week1_walkthrough.ipynb       # Runnable code tour
│
├── src/
│   ├── vision/                       # CV inference pipeline
│   │   ├── cv_inference.py           # Gemini Vision ingredient detection from fridge images
│   │   └── ingredient_normalizer.py  # Maps CV labels → Food.com ingredient vocabulary
│   ├── data/                         # Loaders, ingredient/nutrition parsing
│   │   ├── loader.py                 # Pre-split train/val/test loaders
│   │   ├── ingredients.py            # Nutrition + ingredient normalization
│   │   └── pantry.py                 # Pantry derivation + persona loading
│   ├── models/                       # Stage 1 models (popularity, ease, bpr,
│   │                                 #   als, tag_svd_content, sentence_bert,
│   │                                 #   hybrid_linear)
│   ├── eval/                         # metrics, harness, significance, alpha_sweep
│   ├── reranker/                     # Stage 2: scores, combiner, diet filter
│   ├── utils/
│   │   └── staples.py                # Universal kitchen staples + pantry_score
│   └── vision/                       # CV inference (already listed above)
│
├── streamlit_app.py                  # Interactive demo (persona / walk-in / fridge photo)
├── tests/                            # 290+ tests covering all modules
├── results/                          # Sanity/smoke test outputs
├── smoke_test.py                     # Quick end-to-end pipeline check
├── pyproject.toml                    # uv project manifest
└── uv.lock                           # Pinned dependencies
```

---

## Getting started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- ~2 GB of disk space (after downloading Food.com data)

### 1. Clone and install dependencies

```bash
git clone <repo-url> pantryplate
cd pantryplate
uv sync
```

This sets up a `.venv/` and installs everything from `uv.lock`. ~5 minutes including PyTorch download.

### 2. Set up environment variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

Get a free Gemini API key at **aistudio.google.com**

### 3. Download the Food.com dataset

This dataset is too large to ship in git (~1.5 GB). Download it from [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions):

```bash
# Either via the Kaggle CLI:
kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions
unzip food-com-recipes-and-user-interactions.zip -d data/raw/

# Or download the ZIP manually from the URL above and extract to data/raw/.
```

After extraction, `data/raw/` should contain:

```
data/raw/
├── RAW_recipes.csv              (~295 MB)
├── RAW_interactions.csv         (~349 MB)
├── interactions_train.csv       (~28 MB)    ← used for model training
├── interactions_validation.csv  (~290 KB)   ← optional hyperparameter tuning
├── interactions_test.csv        (~520 KB)   ← cold-item evaluation
├── PP_recipes.csv               (~205 MB)
├── PP_users.csv                 (~13 MB)
└── ingr_map.pkl                 (~900 KB)
```

### 4. Verify the setup

Run the test suite:

```bash
uv run pytest tests/ -q
# Expected: 100+ tests pass in ~10 seconds
```

Run the smoke test (dual-track end-to-end pipeline check):

```bash
uv run python smoke_test.py
# Expected:
#   Track A (warm) Recall@10 ≈ 3.0%  ← popularity baseline floor
#   Track B (cold) Recall@10 = 0.0%  ← correct (popularity can't do cold)
```

If both pass, your environment is set up correctly.

---

## CV Inference (fridge image → ingredients)

PantryPlate supports ingredient detection from fridge photos via Google Gemini Vision. A photo of your fridge is sent to Gemini 2.5 Flash, which returns a list of detected ingredients that feed directly into the Stage 2 reranker as the user's pantry.

```
📷 Fridge photo → Gemini 2.5 Flash → ["eggs", "milk", "carrots"] → Stage 2 reranker
```

To test it, place a fridge image at `data/test_fridge.jpg` and run:

```bash
uv run pytest tests/test_cv.py -s
# Expected: Detected ingredients: ['eggs', 'milk', 'carrots', ...]
```

The detected ingredient labels are normalized to match Food.com vocabulary via `src/vision/ingredient_normalizer.py` before being passed to the reranker.

---

## For contributors joining the project

**Start here**: [`docs/week2_onboarding.md`](docs/week2_onboarding.md).

That doc covers:

- Project recap (1 paragraph)
- Pointers to deeper reading (proposal deck, EDA notebook, walkthrough)
- ⭐ **The data routing** — which CSV file feeds which evaluation track. This is the single most-misunderstood thing in the project; read it twice.
- The Stage 1 model interface contract
- The 12 locked decisions you must respect
- Numbers your model should beat (popularity floor: Recall@10 ≈ 3.0%)
- Conventions (naming, file structure, determinism, where to commit)
- Quickstart template for new models

After onboarding:
- Plug your Stage 1 model into the harness via [`docs/eval_harness_usage.md`](docs/eval_harness_usage.md).
- Drop your model's numbers into the **[Stage 1 leaderboard](docs/stage1_leaderboard.md)** — living doc that's the source of truth on model performance.

---

## Key concepts (quick reference)

### Dual-track evaluation

Two complementary evaluation regimes, same architecture:

| Track | What it tests | Test set | Floor (popularity baseline) |
|---|---|---|---|
| **A (warm)** | Standard CF recommendation on items with rating history | Time-based LOO holdout from `interactions_train.csv` | Recall@10 ≈ 3.0% |
| **B (cold)** | Generalization to brand-new recipes nobody has rated | `interactions_test.csv` (cold by design) | Recall@10 = 0% (CF can't do cold) |

Only content-aware models (Tag SVD, Sentence-BERT, hybrid) produce non-zero numbers on Track B. CF-only models score 0 by construction — this is **correct**, not a bug.

### Four constraint scores (Stage 2)

```
final(user, recipe) = s_diet × (αt · s_taste + αp · s_pantry + αn · s_nutrition)
```

- `s_taste` ∈ [0, 1] — predicted relevance from Stage 1
- `s_pantry` ∈ [0, 1] — overlap fraction of non-staple ingredients with user's pantry
- `s_nutrition` ∈ [0, 1] — Gaussian proximity to user's macro targets
- `s_diet` ∈ {0, 1} — hard filter; 1 if all dietary restrictions met, 0 otherwise

The (αt, αp, αn) triplet is the project's research object — we sweep it across the simplex to study the trade-off.

### Useful Recall@K (signature metric)

A recommendation counts as "useful" if it's both in the top-K and satisfies the user's constraints. Specifically:

- Held-out recipe appears in top-K AND
- ≤ 3 non-staple ingredients missing from the user's pantry AND
- Within ±20% of user's macro targets AND
- All dietary restrictions respected

The gap between standard Recall@K and Useful Recall@K is the project's empirical contribution.

---

## Running tests

```bash
# Full test suite (~10 seconds)
uv run pytest tests/ -q

# Specific module
uv run pytest tests/test_harness.py -v

# CV inference test
uv run pytest tests/test_cv.py -s

# Quick subset (skipping integration tests that need real data)
uv run pytest tests/ -q -k "not integration"
```

---

## Running the EDA notebooks

```bash
# Start Jupyter (uv-managed)
uv run jupyter lab

# Then open:
# - notebooks/week1_eda.ipynb       (deep feasibility analysis)
# - notebooks/week1_walkthrough.ipynb (code tour for new contributors)
```

Both notebooks are pre-executed with outputs included — you can read them without re-running. Re-execution requires the Food.com data in `data/raw/`.

---

## Timeline

**Original plan** (Week 1–3 went into data, EDA, and proposal deck — Stage 1 model implementation was deferred):

| Week | Dates | Focus | Status |
|---|---|---|---|
| 1 | May 13–19 | Data pipeline, sanity checks, EDA, feasibility analysis | ✅ Done |
| 2 | May 20–26 | (planned: Stage 1 models) → focused on EDA refinement + proposal deck v1-v5 | Replanned |
| 3 | May 27–Jun 2 | (planned: hybrid + results) → proposal deck finalization | Replanned |
| — | **Jun 3** | **Proposal submitted** | ✅ Done |

**Remaining 3-week execution sprint** (where we are now):

| Week | Dates | Focus | Status |
|---|---|---|---|
| 4 | **Jun 3–9** | Eval harness ✅ + onboarding docs ✅ + 3 personas ✅ + Stage 1 models start | 🟡 Active |
| 5 | Jun 10–16 | Stage 2 reranker + α-sweep + per-persona analysis + CV inference ✅ | 🟡 Active |
| 6 | Jun 17–23 | Demo widget + slide polish + physical prop + dress rehearsal | Pending |
| — | **Jun 24** | **Final presentation** | — |

The compressed Week-4-6 execution requires sharp scoping. See `docs/week2_onboarding.md` §4b for the Stage 1 model claim table and `CLAUDE.md` for the current state summary.

---

## Dataset

[Food.com Kaggle dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) — Majumder et al., *Generating Personalized Recipes from Historical User Preferences*, EMNLP 2019.

- 231,637 recipes (with 7-axis nutrition vectors, tags, ingredients, descriptions)
- 1.13M user-recipe interactions (2000–2018)
- Authors' pre-split into train (698K interactions) / validation (7K positives) / test (10K cold positives)
- 8,023 canonical ingredients via `ingr_map.pkl`

The training cohort is 24,961 active users (≥5 ratings each).


---

## Team

- Muhammad Ikhwan Bin Wahid
- Koh We Kiat
- Anastasia Frances Frederica
- Alekhya Kodavatiganti
- Abinav Shajil

---

## Citations

- **Dataset**: Majumder et al., EMNLP 2019
- **Cornac library** (BPR): Salah et al., JMLR 2020
- **implicit library** (ALS): Frederickson; Hu et al. (implicit ALS), ICDM 2008
- **EASE**: Steck, *Embarrassingly Shallow Autoencoders*, WWW 2019
- **BPR**: Rendle et al., *Bayesian Personalized Ranking*, UAI 2009
- **Sentence-BERT**: Reimers & Gurevych, EMNLP 2019
