# PantryPlate — Locked Data & Modelling Decisions

This document captures the decisions made during Week 1 (data sanity checks
+ EDA + feasibility analysis) that should not be re-debated during
implementation. Each decision lists the evidence behind it, where it
applies in the code, and what happens if you find a reason to revisit it.

Last updated: 2026-06-01 (Path A refactor — dual-track evaluation).

---

## 1. Training cohort = authors' pre-split train (25K active users)

**Decision** (refined 2026-06-01): Use the authors' published `interactions_train.csv`
from Majumder et al. 2019 as the training cohort. This is pre-filtered to
**24,961 active users with 698,901 interactions** (681,944 after dropping
0-star "review without rating" entries).

We previously used the raw interactions with our own `filter_active_users(min_ratings=5)`
which produced 22,018 active users. The new approach is functionally similar
but uses the published filter for consistency with the original paper.

**Where it applies**:
- `src.data.loader.load_train_interactions()` — preferred
- `src.data.loader.load_interactions()` + `filter_active_users()` — kept for
  raw-data exploration, no longer the default model-training path

**If you want to revisit**: the authors' filter is "users with sufficient
activity for the published evaluation"; the precise threshold isn't
documented in the paper. The numbers (24,961 users / 698,901 interactions)
are stable so we treat the filter as a fixed property of the dataset.

---

## 1b. Dual-track evaluation (warm-item + cold-item)

**Decision** (new 2026-06-01): the project evaluates Stage 1 models on
**two complementary test sets** that ask different questions:

### Track A — warm-item LOO (standard CF benchmark)

Apply `time_based_split(authors_train, holdout_per_user=1)` to the authors'
train file. This holds out each user's most-recent positive rating; held-out
items typically have many other raters in train (median = 20 raters).

- **Test set**: ~24,382 held-out positives, one per user
- **Question**: "given a user's history, predict the next item they'll engage with"
- **Models that compete**: ALL 8 Stage 1 models
- **Floor**: popularity baseline → Recall@10 ≈ 3.00%
- **Expected winners**: BPR, EASE, Hybrid

### Track B — cold-item (authors' pre-split test)

Use `interactions_test.csv` as-is (load via `load_test_interactions()`).
The authors hold out items that have **0 raters in train** — pure cold-item
evaluation requiring content features.

- **Test set**: 10,393 held-out positives, one per user, all on cold items
- **Question**: "given a user's history, predict a recipe NO ONE has rated yet"
- **Models that meaningfully compete**: TF-IDF, Sentence-BERT, hybrid, two-tower (content-aware only)
- **Floor**: popularity baseline → Recall@10 = 0.00% (by construction)
- **Models that fail-by-design**: Popularity, MF, EASE, BPR (need item co-occurrence)
- **Expected winners**: Sentence-BERT, two-tower, hybrid

### Why both

- Track A demonstrates the project's CF spine works (standard recsys benchmark).
- Track B demonstrates content-aware generalization to novel recipes (cold-start, a real-world recipe-app scenario).
- Hybrid models should perform well on both — this is the architectural payoff.
- The Stage 2 reranker (multi-constraint α-sweep) is applied to BOTH tracks; the X-factor study runs twice.

**Where it applies**:
- `src.data.loader.load_prebuilt_split()` returns (train, val, test) in one call
- Eval harness (Week 2 Day-1 deliverable) needs a `track` parameter to switch between warm/cold test sets
- Reranker (Week 4) is track-agnostic

**If you want to revisit**: drop Track B if proposal feedback says the cold-item story is too ambitious. Track A alone is still a complete project.

---

## 2. Drop 0-star interactions

**Decision**: Filter all ratings with `rating == 0` out of the training
and evaluation data.

**Evidence**: Sanity check found 60,847 zero-star interactions (5.4% of
the total). These are almost certainly "review left, no rating" entries
rather than literal zero scores. Keeping them would distort both
implicit-feedback signals (false positives in the rating matrix) and
rating-magnitude models (they pull the mean down toward zero).

**Where it applies**: `src.data.loader.load_interactions(drop_zero_stars=True)` — the default.

**If you want to revisit**: keep them only if a reviewer specifically
asks. The dataset's own documentation hints that zero-star is the missing
value, so dropping is the right behavior.

---

## 3. Positive rating threshold = 4 stars

**Decision**: A rating of 4 or 5 stars counts as a "positive" interaction
for purposes of train/test split and Recall@K / NDCG@K evaluation.

**Evidence**: This is locked in the project README. EDA shows 88.6% of
non-zero ratings are 4+ (72.1% are 5-star alone), so the threshold is
generous. We chose 4 over 5 to retain enough positives per user for the
time-based split to produce non-trivial test sets.

**Where it applies**:
- `src.data.loader.POSITIVE_THRESHOLD = 4`
- `time_based_split` holds out the most recent rating with `rating >= 4`

**If you want to revisit**: don't. This is a locked design decision in
the README. Changing it would invalidate the headline metric comparisons.

---

## 4. Time-based, per-user, leave-one-out split (for Track A — warm)

**Decision**: For each user in the authors' filtered train cohort, hold out
their most recent positive (4+ star) rating as the test point. All earlier
interactions go to train. This is applied **on top of the authors' train
file** to construct Track A's warm-item evaluation set.

**Evidence**: Standard recsys protocol. Produces ~24,382 held-out positives
across the authors' 24,961 active users (a few users have all their positives
held out by virtue of having only one positive — see edge cases below).

**Where it applies**: `time_based_split(load_train_interactions(), holdout_per_user=1)`

**Edge cases handled**:
- Users with no positives at all: kept entirely in train (no holdout)
- Users with only one interaction: kept entirely in train (otherwise train would be empty for them)
- Users where their only positive *is* their only interaction: same as above

**Note**: this is the Track A split. Track B uses the authors' published
`interactions_test.csv` directly (a cold-item evaluation; see decision 1b).

---

## 5. Stage 1 model menu = 8 models (+ SASRec stretch)

**Decision**: The Stage 1 candidate-generator portfolio is:

| Order | Model | Course Week | Role |
|---|---|---|---|
| 1 | Popularity | W1 | Lower-bound baseline ✓ Week 1 |
| 2 | MF / ALS (Cornac) | W1 | Rating-prediction reference baseline |
| 3 | **EASE** | W1-W3 | Closed-form implicit-feedback shallow |
| 4 | BPR (Cornac) | W3 | SGD pairwise implicit ranker |
| 5 | Content TF-IDF (ingredients + tags + nutrition) | W4 | Content-only reference |
| 6 | **Two-tower neural** (replaces NCF) | W4-W5 | Deep + multimodal-ready |
| 7 | Hybrid linear (α·CF + (1-α)·content) | W5 | Combination |
| 8 | Sentence-BERT content | W9 | Modern semantic content |
| stretch | SASRec / GRU4Rec | W8 | Only if W4 progress check is green |

**Models explicitly dropped** from the README's longer list:
- Item-item CF (cosine): subsumed by EASE, which is its regularized cousin
- NCF (Neural CF): replaced by two-tower neural, which serves the same
  "depth helps?" question while also covering the W4 multimodal angle
- WMF / ALS-implicit (Hu et al.): redundant with BPR for the
  implicit-feedback story

**Evidence**: Each retained model answers a distinct project question.
See the model rationale in `docs/proposal_deck_rebuild_brief.md` §2d.

**Where it applies**: `src/models/` will contain one module per model.
Each follows the same interface as `PopularityRecommender`: `.fit(train_df)`
and `.recommend(user_id, k, exclude_seen=True)`.

**If you want to revisit**: adding a model is fine if it answers a new
question. Removing one needs a story about why its question is no longer
worth answering.

---

## 6. Item filter for memory-bound CF models (EASE, BPR, two-tower)

**Decision**: EASE, BPR, and two-tower neural train on **only the 20,060
recipes with ≥10 ratings**. Recipes outside this pool are predicted via
popularity fallback at inference time.

**Evidence**:
- 226,590 distinct rated recipes total
- EASE's item-item matrix would be ~226K × 226K × 4 bytes = ~200 GB at
  full scale — unfeasible
- At ≥10 ratings (20,060 recipes), the matrix is ~1.6 GB — fits in RAM
- Recipes with <10 ratings have weak signal anyway; popularity fallback
  is the principled handling

**Where it applies**: each model's `.fit()` will filter `train_df` to
this pool before training. The eval harness will route requests for
out-of-pool recipes through popularity automatically.

**If you want to revisit**: ≥5 ratings would give 50,978 recipes
(matrix ~10 GB — tight on 16 GB machines). Document any machine-specific
choice that differs from this default.

---

## 7. Pantry — soft ranking signal with staples assumed

**Decision** (refined after staples-aware analysis): the pantry constraint
is operationalized as **two related but distinct metrics**, both built on
the principle that universal staples (salt, pepper, water, oil, flour,
eggs, milk, butter, garlic, onion, baking powder/soda, vinegar, vanilla,
lemon juice, cinnamon, cornstarch, paprika, honey, chicken broth +
canonicalization variants — 38 items total) are assumed available in
every kitchen and excluded from the calculation.

**Empirical justification** (see `notebooks/week1_eda.ipynb` §4b):
- 30 of 38 STAPLES items (79%) appear in the top-50 most common
  ingredients across 231K recipes — strong frequency evidence
- 34 of 38 (89%) appear in the top-200 — at least defensible
- 4 are "dead variants" (alternate canonical forms like `flour` vs `flmy`
  that the dataset doesn't use but we keep defensively)
- 89.4% of recipes contain ≥1 staple; mean = ~3 staples per recipe
- Staples explain ~35% of the average recipe's ingredient list —
  high enough to matter, low enough that the score isn't trivialized

**Known issue (deferred)**: `honey` and `chicken broth` are not vegan.
Current `get_staples_for_persona({"restrictions": ["vegan"]})` only drops
DAIRY_AND_EGGS. A follow-up fix would add `ANIMAL_PRODUCTS_BEYOND_DAIRY
= {"honey", "chicken broth"}` and exclude it for vegans as well.

### Metric A — `pantry_score` (reranker uses this)

`pantry_score(recipe_ings, user_pantry) = |non_staple_ings ∩ user_pantry| / |non_staple_ings|`

A continuous score in [0, 1]. Used as the `s_pantry` term in the Stage 2
reranker formula:

`final(u, r) = s_diet × (αt · s_taste + αp · s_pantry + αn · s_nutrition)`

There is **no minimum pantry threshold** below which recipes are excluded.
Only `s_diet` is a hard filter. Low pantry-score recipes simply rank
lower, they don't disappear.

### Metric B — `missing_count ≤ 3` (Useful Recall uses this)

`missing_count(recipe_ings, user_pantry) = |(recipe_ings - staples) - user_pantry|`

A count of non-staple ingredients NOT in the user's pantry. A recipe is
"pantry-feasible" for the Useful Recall metric if `missing_count ≤ 3`
("the user would only need to buy 3 more things"). More interpretable
than an overlap-fraction threshold; matches realistic cooking behavior
(people will run to the store for 1-2 missing items).

### Evidence — why two metrics

EDA simulated both formulations against a 25-item user-specific pantry
across 30,000 sampled recipes, using the staples-assumed approach:

| Pantry definition | Median s_pantry | s_pantry ≥ 0.5 | missing ≤ 3 |
|---|---|---|---|
| Strict (no staples) | 0.00 | 0.4% | — |
| **Staples assumed (project default)** | **0.00** | **2.6%** | **24.7%** |

Key insight: even with staples assumed, **median `pantry_score` is near
zero** because a 25-item user pantry rarely contains 5+ of a recipe's
non-staple ingredients. This is **fine for the reranker** — within a
single user, recipes that match a few user items still rank above those
that match none. But it makes `s_pantry ≥ 0.5` a near-empty threshold
for Useful Recall (only 2.6% of recipes). The count-based `missing ≤ 3`
gives a workable 24.7% match rate AND is more intuitive to explain in
the proposal ("user only needs to buy 3 more things").

### Persona implications

- Persona JSONs need only enumerate **25-35 user-specific items** —
  the things that make this user different from the next.
- Staples are a **project-wide constant** in `src/utils/staples.py`,
  not part of any individual persona.
- Vegan personas auto-drop dairy/eggs from staples
  (`get_staples_for_persona(persona)` handles this).
- Personas with other restrictions can list `exclude_from_staples` in
  their JSON to override further (e.g., gluten-free excludes "flour").

### Demo framing implications

- Demo language: *"recipes that best match what you have on hand"* OR
  *"recipes you could cook with only a few items from the store"*.
- **Avoid**: *"recipes you can cook tonight from your fridge"* —
  implies hard-filter semantics we explicitly rejected.
- Demo input: user types/clicks 15-25 ingredients beyond staples.
  Much faster than enumerating salt/pepper/oil/etc.

### Where it applies

- `src/utils/staples.py` — `STAPLES`, `DAIRY_AND_EGGS`,
  `get_staples_for_persona()`, `pantry_score()`, `missing_count()`
- `src/reranker/` (Week 4 work) — calls `pantry_score()` for s_pantry
- `src/eval/` Useful Recall metric (Week 4 work) — checks `missing_count ≤ 3`

---

## 8. Diet enforcement = (tag OR derived rule) AND ingredient blocklist

**Decision**: `s_diet` is a hard filter (0/1) computed as follows:
1. If the dietary restriction has well-covered tags in the dataset
   (vegan, vegetarian, gluten-free, dairy-free, low-carb, low-fat,
   low-sodium, low-cholesterol, diabetic, kosher, nut-free, egg-free),
   check the recipe's tag list.
2. If the restriction lacks tag coverage (**keto, paleo, whole30,
   lactose-free, halal, low-sugar**), derive it from macros + ingredient
   blocklist. Specific derivations TBD per restriction in Week 4.
3. In **both** cases, AND with an ingredient blocklist (e.g., vegan AND
   no animal-product ingredients).

**Evidence**:
- 6 of the 17 desired dietary tags have **zero** recipes in the dataset
- For the 11 that do have coverage, EDA's consistency check showed
  vegan-tagged recipes contain animal-product ingredients **3.1%** of
  the time. Tags alone are unreliable for hard filtering.

**Example derivation rules** (Week 4 will finalize):
- `keto`: `carbs_pdv < 10` AND no high-carb ingredients (rice, pasta, bread)
- `lactose-free`: no `milk, cheese, butter, cream, yogurt, ice cream, sour cream`
- `paleo`: no `grain, legume, sugar, dairy` ingredient families
- `whole30`: paleo + no `alcohol, MSG, sulfites, carrageenan`
- `low-sugar`: `sugar_pdv < 5` AND no `sugar, honey, syrup` in top ingredients
- `halal`: no `pork, alcohol, lard, gelatin (from non-halal sources)`

**Where it applies**: `src/reranker/diet.py` (Week 4 work). Derivation
rules will be in a single `DIET_RULES` dict so they're auditable.

---

## 9. Nutrition clipping at (5000 kcal, 1000% PDV)

**Decision**: `parse_nutrition` clips by default:
- `calories` capped at 5000 kcal
- All PDV percentages capped at 1000%

**Evidence**:
- Sanity check found 1 recipe at 434,360 kcal — clearly misparsed
- 1,049 recipes (0.45%) exceed 5,000 kcal; many are misparsed portion sizes
- PDV cap of 1000% binds on <2% of recipes (max on sugar_pdv: 1.23%)
- Without clipping, distance-based nutrition scoring (`s_nutrition`)
  blows up for outlier recipes

**Where it applies**: `src.data.ingredients.parse_nutrition(clip=True, calorie_cap=5000, pdv_cap=1000)` — these are the defaults.

**If you want to revisit**: tighten the calorie cap to 3000 if a reviewer
finds the 5000 still includes obvious miscounts. Loosening doesn't help.

---

## 10. Persona pantries should be 25-35 user-specific items

**Decision** (refined alongside §7's staples-aware approach): each
persona's `pantry` list should contain **25-35 canonical ingredients
that are NOT staples**. Staples (salt, pepper, oil, flour, eggs, etc.)
are project-wide via `src/utils/staples.py` and don't need to be in
individual persona JSONs.

**Evidence**: Under the staples-assumed metric, `pantry_score` median
is near zero regardless of user-pantry size because user pantries can't
reasonably cover most non-staple ingredients in arbitrary recipes. The
within-user *ranking* signal works fine at 25-35 items. Beyond ~50
items the personas start to feel artificial (real cooks don't track 60
non-staple ingredients).

**Persona JSON schema** (refined):
```json
{
  "id": "fitness_focused",
  "label": "Fitness-focused lifter",
  "description": "Tracks macros carefully...",
  "macro_targets": {"calories": 600, "protein_pdv": 50, ...},
  "restrictions": ["high-protein"],
  "pantry": ["chicken", "rice", "broccoli", "eggs", ...],  // 25-35 items, no staples
  "exclude_from_staples": [],  // optional, for users who don't have specific staples
  "taste_seeds": [recipe_id, ...]  // 20-30 Food.com recipe IDs
}
```

**Where it applies**: Persona JSON schema in `data/personas/*.json`.
See `src/data/pantry.py` (loader) and `src/utils/staples.py` (staples
list + helpers).

**If you want to revisit**: if a persona's pantry produces zero matches
for most recipes (because all items are obscure), expand toward 35. If
the persona feels generic (matches too many recipes), reduce toward 25.

---

## 11. Eval harness is Week 2 Day-1 deliverable

**Decision**: Before any Week 2 model is built, one team member produces
`src/eval/harness.py` exposing:

```python
def evaluate(
    model,                      # any object with .recommend(user_id, k)
    train_df, test_df,
    k_values=(5, 10, 20),
    candidate_filter=None,      # Stage 2 hook
) -> dict:
    """Returns {'recall@5': ..., 'ndcg@10': ..., 'mrr': ..., 'n_users': ...}"""
```

Plus a bootstrap CI helper, deterministic seed control, and a reference
run on `PopularityRecommender` that everyone can sanity-check against.

**Evidence**: Without one shared harness, four people will implement
"Recall@10" with subtly different tie-breaking, candidate-set treatment,
or empty-list handling. The model comparison table would be noise. The
metrics module (`src/eval/metrics.py`) ✓ already exists; the harness
glues it to the data.

**Where it applies**: `src/eval/harness.py` (does not yet exist).

---

## Summary table

| # | Decision | Default value | Code reference |
|---|---|---|---|
| 1 | Training cohort | authors' pre-split (24,961 active users / 681K interactions) | `load_train_interactions()` |
| 1b | Evaluation | dual-track: warm-item LOO + cold-item (authors' test) | `time_based_split`, `load_test_interactions` |
| 2 | Drop 0-star ratings | True | `load_train_interactions(drop_zero_stars=True)` |
| 3 | Positive threshold | 4 stars | `POSITIVE_THRESHOLD = 4` |
| 4 | Track A split protocol | leave-one-out by date, most recent positive | `time_based_split` |
| 5 | Stage 1 model menu | 8 models + SASRec stretch | `src/models/` |
| 6 | CF item filter | recipes with ≥10 ratings (EASE, BPR, two-tower) | per-model fit() |
| 7 | Pantry — reranker score | `pantry_score()` (non-staple overlap, continuous) | `src/utils/staples.py` |
| 7 | Pantry — Useful Recall condition | `missing_count() ≤ 3` non-staple items | `src/utils/staples.py` |
| 8 | Diet enforcement | tag OR derived rule, AND ingredient blocklist | `src/reranker/diet.py` (W4) |
| 9 | Nutrition clipping | calories ≤ 5000 kcal; PDV ≤ 1000% | `parse_nutrition` |
| 10 | Persona pantry size | 25-35 user-specific items (staples are project-wide) | `data/personas/*.json` |
| 11 | Eval harness | Week 2 Day-1, supports both tracks via `track=` param | `src/eval/harness.py` (W2) |
