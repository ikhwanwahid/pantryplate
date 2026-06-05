# PantryPlate — Project Brief

**Project**: Multi-constraint recipe recommendation system
**Course**: Recommender Systems · Project 2
**Proposal due**: June 3, 2026
**Presentation**: June 24, 2026
**Working directory**: `/home/claude/pantryplate` (or wherever your local repo lives)

---

## The one-paragraph pitch

> Existing recipe recommenders optimize for a single objective: taste prediction, calorie filtering, or available-ingredient matching. Real cooks operate under multiple simultaneous constraints — what's in the fridge, what fits the day's nutritional goals, what they actually enjoy eating, what their dietary restrictions allow. We build a multi-constraint recommender that explicitly models this trade-off and study how recommendations evolve as users tune the relative importance of competing constraints.

---

## What the system does

A two-stage pipeline:

1. **Stage 1 — Candidate generation (real recsys).** Learns user taste from Food.com rating history using matrix factorization, BPR, hybrid CF + content, or similar. Returns top-200 candidate recipes per user.

2. **Stage 2 — Multi-constraint reranking (the X-factor).** Scores each candidate on four dimensions:
   - `s_taste` — predicted relevance from Stage 1
   - `s_pantry` — ingredient overlap with user's pantry
   - `s_nutrition` — proximity to macro/calorie targets
   - `s_diet` — hard filter on dietary restrictions

   Combines them as: `final = s_diet × (αt·s_taste + αp·s_pantry + αn·s_nutrition)`

   The (αt, αp, αn) triplet is the project's research object. Different weightings = different user contexts.

3. **Output**: Top-10 reranked recommendations with explanations.

---

## Dataset

**Primary source**: Food.com via Kaggle — `shuyangli94/food-com-recipes-and-user-interactions`
- URL: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
- Size: ~1.5 GB compressed
- Access: Free, requires Kaggle account, direct download

**Key files**:
| File | Contents | Approx rows |
|---|---|---|
| `RAW_recipes.csv` | Recipe metadata | ~230K |
| `RAW_interactions.csv` | User-recipe ratings + reviews | ~1.1M |
| `PP_recipes.csv` | Pre-processed recipes | ~230K |
| `PP_users.csv` | Pre-processed user data | ~226K active users |
| `interactions_train.csv`, `_validation.csv`, `_test.csv` | Pre-split | — |
| `ingr_map.pkl` | Canonical ingredient mappings | ~8K canonical ingredients |

**Nutrition column format**: stringified list of 7 PDV (Percent Daily Value) values: `[calories, total_fat, sugar, sodium, protein, saturated_fat, carbs]`

**No scraping required.** This is the project's structural advantage — real user ratings, structured nutrition, parsed ingredients, all in one download.

---

## Reach goal — vision-based pantry input

**If recsys is on track by Week 4**, add a fridge-photo input mode:
- User uploads photo → vision-LLM (Claude/GPT-4V/Gemini) extracts ingredient list with confidence scores
- Confidence-weighted pantry vector feeds into Stage 2 reranker
- Adds two recsys ablations: ground-truth vs CV-derived pantry; hard vs soft pantry

**Critically**: zero custom CV training. Use a pretrained vision-LLM. The recsys core is unchanged; CV is an input modality only.

**Decision gate**: end of Week 4 progress check. If recsys is behind schedule, skip CV. Always-on fallback: manual ingredient typing.

---

## Course-week coverage

| Week | Topic | Project component |
|---|---|---|
| W1 | Matrix factorization, Cornac | Stage 1 baselines: popularity, MF/ALS |
| W2 | Learning algorithms, evaluation | Eval protocol, leave-one-out |
| W3 | Implicit feedback, ranking | BPR; soft-pantry as implicit signal |
| W4 | Multimodal | Two-tower model with image + text + parts; vision-LLM input (reach goal) |
| W5 | Deep learning | NCF; hybrid CF + content; two-tower neural |
| W6 | Contextual awareness | α-weights as user context; pantry/macros as context |
| W8 | Sequential | SASRec / GRU4Rec on rating timestamps |
| W9 | LLM / linguistic | Sentence-BERT for content; LLM ingredient extraction |

Substantively covers 7 of 9 course weeks.

---

## Six-week timeline

| Week | Dates | Goal |
|---|---|---|
| W1 | May 13–19 | Data pipeline · sanity checks · popularity + item-CF baselines |
| W2 | May 20–26 | MF / BPR via Cornac · content-based model |
| W3 | May 27–Jun 2 | Hybrid + two-tower · **Proposal due Jun 3** |
| W4 | Jun 3–9 | Stage 2 reranker · personas · CV decision gate |
| W5 | Jun 10–16 | α-sweep experiments · ablations · CV integration (if go) |
| W6 | Jun 17–23 | Analysis · slides · demo polish · physical prop · **Present Jun 24** |

---

## Evaluation plan

**Three metric categories**:

1. **Predictive (standard recsys)** — Recall@5, Recall@10, NDCG@10, MRR
   - Ground truth: the recipe the user actually rated next, held out from training
   - Protocol: leave-one-out by date

2. **Mechanical (constraint satisfaction)** — Pantry Hit Rate@K, Nutrition Adherence@K, Diet Compliance@K
   - No ground truth needed; computed deterministically from data

3. **Joint (signature)** — Useful Recall@K, Multi-Constraint Score
   - Counts a recommendation as useful only if (a) it's in the user's held-out positives AND (b) it satisfies all constraints
   - This is where the project's contribution becomes visible

**Trade-off analysis**: α-sweep across the simplex. Headline plot is Useful Recall@K vs α-position, showing the inflection where balanced weights outperform single-objective.

**Statistical rigor**: Bootstrap 95% CIs on means, Wilcoxon signed-rank for model-vs-model comparisons.

**Subjects**: 5-8 user personas (manually defined) + 4-6 team-member profiles (self-curated). Definition of "positive rating": 4 stars or higher.

---

## Architecture

```
USER CONTEXT
  ├── rating history (from Food.com)
  ├── pantry inventory (typed or photo→LLM)
  ├── macro targets (declared)
  └── dietary restrictions (declared)
       ↓
STAGE 1 · CANDIDATE GENERATION (real recsys)
  Models to compare:
   - Popularity (baseline)
   - Item-item CF
   - MF / ALS (Cornac)
   - BPR (Cornac)
   - Content-based (TF-IDF on ingredients + tags + nutrition)
   - Hybrid CF + content
   - Neural CF
   - Sequential (SASRec) — stretch
   - LLM-augmented content — stretch
  → top-200 candidates per user
       ↓
STAGE 2 · MULTI-CONSTRAINT RERANKING (X-factor)
  For each candidate:
   - s_taste from Stage 1
   - s_pantry = |ingredients ∩ pantry| / |ingredients|
   - s_nutrition = exp(-Σ |recipe_macro - target| / target)
   - s_diet = 1 if all restrictions met else 0
  final = s_diet × (αt·s_taste + αp·s_pantry + αn·s_nutrition)
       ↓
TOP-10 RECOMMENDATIONS (with explanations)
```

---

## Code structure (suggested)

```
pantryplate/
├── data/
│   ├── raw/           # Downloaded Food.com CSVs (gitignored, too large)
│   ├── processed/     # SQLite DB, cleaned CSVs
│   └── personas/      # Persona JSON files
├── src/
│   ├── data/          # Data loading, parsing, normalization
│   ├── models/        # Stage 1 models (MF, BPR, content, hybrid, NCF, sequential)
│   ├── reranker/      # Stage 2 multi-constraint reranking
│   ├── eval/          # Evaluation harness, metrics, α-sweep
│   ├── cv/            # Vision-LLM ingredient detection (reach goal)
│   └── utils/         # Helpers
├── notebooks/         # Exploratory analysis, sanity checks
├── demo/              # Web widget for presentation
├── results/           # Experiment outputs, plots, tables
├── docs/              # This README + proposal + slides
└── tests/             # Unit tests
```

---

## Key design decisions already made

These don't need re-debating; they're locked in:

1. **Domain is recipes, not LEGO or music** — chose because Food.com gives us real user-rating data at scale; LEGO and music had data accessibility issues we documented and rejected.

2. **User-rating data comes from Food.com, not scraping** — verified that the dataset is fully downloadable, no scraping required. Catalog and ratings both included.

3. **CV is a reach goal, not core** — pretrained vision-LLM only, no custom training. Decision gate at end of Week 4. Always have manual-input fallback.

4. **Evaluation uses leave-one-out by date** — most recent positive rating per user is held out; model trained on the rest. Standard recsys protocol.

5. **Positive rating threshold = 4 stars** — pick one and stick with it for all metrics.

6. **Architecture is two-stage** — Stage 1 (real recsys) does candidate generation; Stage 2 (X-factor) does multi-constraint reranking. Keeps the project on the right side of "recsys, not filtering."

7. **Headline metric is Useful Recall@K** — composite of "predicted future positive AND satisfies all constraints." This is the project's signature.

---

## Why we picked this project (the structural argument)

A common failure mode in coursework recsys projects: no real user-behavior data, so the project becomes "ask the user their preferences, filter the catalog" — which is search, not recommendation.

PantryPlate avoids this because Food.com has ~1.1M real user-recipe interactions. The taste signal in Stage 1 is *learned from observed behavior*, not declared. The constraints (pantry, macros, diet) are layered on top as context, not as substitutes for behavioral learning.

This is why the two-stage design matters. If you ever find yourself collapsing the stages and using constraints as the primary lens, you've drifted toward the filtering trap. Stay disciplined: Stage 1 learns, Stage 2 constrains.

---

## What's risky and what to watch for

1. **Density of active users**. If many Food.com users have only 1-2 ratings, CF will be weak. Filter to users with ≥5 ratings for primary evaluation; report cold-start performance separately.

2. **Nutrition data outliers**. PDV values can be wildly inflated for tiny portion recipes or wildly low for misparsed data. Clip nutrition values to reasonable ranges and document.

3. **Ingredient parsing quality**. Free-text ingredients ("1 cup flour, sifted") need normalization. Use `ingr_map.pkl` from the dataset.

4. **Scope creep on Stage 1 models**. Eight Stage 1 models are listed; commit to building 4-5 well rather than 8 poorly. Choose in Week 1.

5. **CV scope creep (if go)**. Hard-freeze CV by end of Week 5. The temptation to "improve accuracy" can absorb arbitrary time.

6. **Demo dependency**. Always have manual-input fallback. If WiFi dies or the CV API is down, the demo must still work.

---

## Team work split (suggested)

Adjust to fit your team's strengths:

| Person | Owns |
|---|---|
| A | Data engineer — pipeline, sanity checks, parts-inventory derivation, persona definitions |
| B | Core models — Stage 1 model implementations and training |
| C | X-factor — Stage 2 reranker, α-sweep, evaluation harness, results analysis |
| D | Demo + integration — web widget, CV integration (if go), slides, presentation polish |

---

## Critical first-week tasks

1. **Pull the Food.com dataset** — single Kaggle download, ~1.5 GB
2. **Run the 5 sanity checks** (see "Week 1 sanity checks" task below)
3. **Each team member exports their own taste profile** — pick ~20-30 Food.com recipes they'd realistically rate highly, build a starter profile from those
4. **Confirm Cornac + PyTorch install** — both will be needed
5. **Verify a basic MF baseline runs end-to-end** — load data, train ALS, output top-10 for a sample user

After this week, you'll know whether to commit to the project as scoped or adjust.

---

## Reference documents

- Proposal slide deck: `PantryPlate_Proposal.pptx` (separately delivered)
- Discussion history: see `chat-history-summary.md` if attached
- Course context: 11-week course, Project 2 proposal at Week 7, presentation at Week 10
