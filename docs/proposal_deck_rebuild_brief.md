# PantryPlate — Proposal Deck Rebuild Brief

**Hand this document to Claude.ai (or any AI assistant) and ask it to rebuild `PantryPlate_Proposal.pptx`.**

The original 17-slide deck was authored *before* the Week 1 EDA. After running data sanity checks and a feasibility analysis, we identified three load-bearing changes (one of them a logical error in the signature metric) and several smaller content updates. This brief contains everything needed to produce an updated deck without having to re-derive decisions.

---

## 0. Use instructions for Claude

You are being asked to rebuild a 17-slide PowerPoint proposal deck for a graduate Recommender Systems course project (Project 2) called **PantryPlate**. The deck exists in v1 form (text quoted below for each slide). Your job is to produce an updated v2 deck.

**Suggested output**: a Python script using `python-pptx` (`pip install python-pptx`) that generates `PantryPlate_Proposal_v2.pptx` when run. The script should match the v1 visual identity described in Section 3 as closely as possible — clean academic style, slide numbering "N / 17", big section labels at top, footer with category label, pull-out boxes for emphasis.

**If you prefer**: produce a structured markdown spec with explicit text-per-slide content and visual layout descriptions, and the user can hand it to a designer or to a slide-generation tool.

**Do not**: re-debate the project's design decisions. They are locked in Section 1. If something seems incongruous to you, note it in a comment at the end of your script rather than changing the spec unilaterally.

---

## 1. Project at a glance (locked context)

**What PantryPlate is**: a multi-constraint recipe recommender. Stage 1 learns user taste from Food.com rating history (matrix factorization, BPR, EASE, etc.). Stage 2 reranks Stage 1's top-200 candidates using four scores — taste, pantry, nutrition, diet — combined via `final = s_diet × (αt·s_taste + αp·s_pantry + αn·s_nutrition)`. The (αt, αp, αn) triplet is the project's research object: how do recommendations evolve as a user tunes the relative importance of competing constraints?

**Team**: 5 people. **Course**: Recommender Systems, 11-week semester, Project 2.

**Timeline**:
- Week 1 (May 13-19): data pipeline, sanity checks, popularity baseline, EDA — **complete**
- Week 2 (May 20-26): Stage 1 models — eval harness, MF, EASE, BPR, content
- Week 3 (May 27-Jun 2): hybrid + two-tower, **proposal due Jun 3**
- Week 4 (Jun 3-9): Stage 2 reranker, personas, CV decision gate
- Week 5 (Jun 10-16): α-sweep experiments, ablations
- Week 6 (Jun 17-23): analysis, slides, demo polish, **present Jun 24**

**Dataset**: Food.com via Kaggle (`shuyangli94/food-com-recipes-and-user-interactions`). ~1.5 GB direct download, no scraping required. Contains:
- `RAW_recipes.csv` — 231,637 recipes with parsed ingredients, 7-axis nutrition vector (calories + 6 PDV percentages), tags
- `RAW_interactions.csv` — 1,132,367 user-recipe ratings, 2000-2018 time range
- `ingr_map.pkl` — 8,023 canonical ingredient mappings

**Locked design decisions** (do not re-debate):
1. Domain is recipes (Food.com). LEGO, music, fitness, chess were considered and rejected — documented elsewhere.
2. Two-stage architecture: Stage 1 = real recsys (learned from behavior); Stage 2 = constraint reranking. This separation prevents the project from collapsing into "search by filter."
3. Evaluation = leave-one-out by date, positive rating threshold = 4 stars.
4. Active user threshold = ≥5 ratings (cold-start users evaluated separately).
5. CV (fridge-photo input) is a Week 4 reach goal using pretrained vision-LLM (Claude/GPT-4V/Gemini); no custom training. Always-on fallback = manual ingredient entry.
6. Headline metric = **Useful Recall@K** (joint of predicted-positive + constraints-satisfied).

---

## 2. What changed since v1 of the deck

### 2a. Real data numbers (replace placeholders)

| Quantity | v1 deck said | EDA found |
|---|---|---|
| Recipes | ~230K | **231,637** |
| User reviews | ~1.1M | **1,132,367** raw; **1,071,520** after dropping 0-star ratings (sanity check found these are "review without rating") |
| Active users | "~226K" — **wrong, that's total users** | **22,018** users with ≥5 ratings (carry 78.9% of rating mass) |
| Total users | implicit | 226,570 |
| Distinct rated recipes | implicit | 226,590 |
| Time range | not stated | 2000-01-25 to 2018-12-20; peak year 2008, 2018 has only 10% of peak activity |
| Nutrition coverage | claimed but unverified | **100%** of recipes have valid 7-element nutrition vector |
| Dietary tags | claimed | 11 of 17 desired tags have ≥100 recipes; **6 tags have ZERO coverage** (keto, paleo, whole30, lactose-free, halal, low-sugar) — need to be derived from macros/ingredients |
| Rating distribution | "~88% are 4+ stars" | confirmed: 88.6% are 4+ stars, 72.1% are 5 stars |

### 2b. Pantry overlap reality (affects framing + a numeric threshold)

EDA simulated `s_pantry = |ingredients ∩ pantry| / |ingredients|` with a realistic 25-item starter pantry across 30,000 sampled recipes:

| Threshold | Recipes matching | Implication |
|---|---|---|
| s_pantry ≥ 0.3 | 35.0% | Loose match — most |
| s_pantry ≥ 0.5 | 11.4% | Solid match — minority |
| s_pantry ≥ 0.75 | 0.7% | Strong match — rare |
| s_pantry ≥ 0.9 | **0.003% (7 recipes)** | Essentially never |
| Median s_pantry | **0.23** | A typical recipe shares only 2/9 ingredients with the pantry |

**Implication for the deck** (refined after a follow-up staples-aware analysis): the original `pantry ≥ 90%` threshold for Useful Recall would be essentially always zero. We use a two-metric approach:

1. **Reranker `s_pantry`** = `|non_staple_ings ∩ user_pantry| / |non_staple_ings|` — a continuous score in [0,1] computed on non-staple ingredients only. Universal staples (salt, pepper, water, oil, flour, eggs, milk, butter, garlic, onion, baking soda/powder, vinegar, vanilla, cinnamon, cornstarch, paprika, honey, chicken broth — 38 items including canonicalization variants) are assumed available in every kitchen. Empirically validated: 79% of these rank in the top-50 most common ingredients in the dataset; staples explain ~35% of the average recipe. The reranker formula stays unchanged; the only difference is `s_pantry` excludes staples.

2. **Useful Recall pantry condition** = `missing_count ≤ 3` non-staple items NOT in user pantry. More interpretable than the overlap fraction ("user only needs to buy 3 more things"), more usable threshold (~25% of recipes match vs ~3% for `s_pantry ≥ 0.5` because the overlap metric is right-skewed near zero).

**Empirical evidence (staples-assumed, on 30K sampled recipes, 25-item user pantry)**:
- Median `pantry_score`: 0.00 (most recipes have no non-staple items in a small user pantry, but the score still discriminates *within-user*)
- `s_pantry ≥ 0.5`: 2.6% of recipes (too tight for a binary threshold)
- `missing ≤ 3`: 24.7% of recipes (workable threshold)

**Persona implications**: each persona's `pantry` field lists **25-35 user-specific items** (no staples — those are project-wide). Vegan personas auto-drop dairy/eggs from staples via the `get_staples_for_persona()` helper; any persona can list `exclude_from_staples` in its JSON to further override.

**Demo language**: *"recipes that best match what you have on hand"* OR *"recipes you could cook with only a few items from the store"*. Avoid *"recipes you can cook tonight from your fridge"* — implies the hard-filter semantics we explicitly rejected.

### 2c. Rating signal is noisier than expected (affects Stage 1 model choice)

EDA on per-user rating behavior (among 22,018 active users):
- 21.8% **always** give the same rating
- 57.7% use only 1-2 distinct rating values
- Mean of per-user means = 4.65 (heavily compressed to top)

**Implication**: rating-magnitude prediction (classic MF/ALS) has very little gradient to learn from. **Implicit-feedback models (EASE, BPR) are likely to outperform** rating-prediction models on the project's actual metrics (Recall@K, NDCG@K — not RMSE). MF stays in the menu as a "reference baseline" for paradigm comparison, not as a primary contender.

### 2d. Stage 1 model menu updated (8 models, not 8 different ones)

The v1 deck listed: Popularity, MF/ALS, BPR, Content-based, Hybrid CF+content, Neural CF, Sequential (SASRec/GRU4Rec), LLM-augmented Sentence-BERT. The locked v2 menu is below. Net changes: **EASE added**; **NCF removed**; **Two-tower neural replaces NCF** (also picks up the Week 4 multimodal angle); item-item CF removed (it's the un-regularized cousin of EASE — no extra story). SASRec stays as stretch.

| Order | Model | Course week | Role |
|---|---|---|---|
| 1 | Popularity | W1 | Lower-bound baseline |
| 2 | MF / ALS via Cornac | W1 | Rating-prediction reference baseline |
| 3 | **EASE** | W1-W3 | Closed-form implicit-feedback shallow model |
| 4 | BPR via Cornac | W3 | SGD-trained pairwise implicit ranker |
| 5 | Content TF-IDF (ingredients + tags + nutrition) | W4 | Content-only reference |
| 6 | **Two-tower neural** (replaces NCF) | W4-W5 | Deep + multimodal-ready |
| 7 | Hybrid linear (α·CF + (1-α)·content) | W5 | Combination model |
| 8 | Sentence-BERT content | W9 | Modern semantic content |
| stretch | SASRec / GRU4Rec | W8 | Sequential — only if Week 4 progress check is green |

Note for slide footnote: *"Rating distribution is heavily 4-5 star (88.6% are 4+); implicit-feedback models (EASE, BPR) are expected to outperform rating-prediction MF on Recall@K. MF is included for paradigm comparison."*

### 2e. Diet enforcement strategy (new detail)

For dietary tags with zero coverage in the dataset (keto, paleo, whole30, lactose-free, halal, low-sugar), we **derive** them from nutrition + ingredient blocklists rather than relying on tags alone. Example: keto = `carbs_pdv < 10`; lactose-free = "no milk, cheese, butter, cream, yogurt" ingredient blocklist. For tags with adequate coverage (vegan, vegetarian, gluten-free, dairy-free, low-carb, low-fat, low-sodium, low-cholesterol, diabetic, kosher, nut-free, egg-free), we AND the tag check with an ingredient blocklist — EDA found vegan-tag-to-ingredient consistency is 96.9%, so tags alone are unreliable.

This is implementation detail; deck should mention it as one bullet on the Stage 2 reranker slide.

### 2f. Useful Recall@K — corrected definition

**v1 deck (Slide 12) said**:
> "The held-out recipe must be in the top-K AND the user must be able to build it (pantry ≥ 90%), it must fit their macro targets (within ±15%), and it must respect their dietary restrictions."

**v2 corrected**:
> "The held-out recipe must be in the top-K of the model's recommendations AND must satisfy the user's constraints — specifically: pantry overlap ≥ 50% (rather than the impractical ≥ 90%, which essentially no recipe meets), macro targets within ±20%, and all dietary restrictions respected. This makes Useful Recall the joint test of 'did we predict correctly AND did we predict something usable.'"

### 2g. Dual-track evaluation (added 2026-06-01)

**Key methodology decision**: the Food.com dataset comes with the authors' pre-split train/validation/test CSVs. Investigation showed the **authors' test set is a cold-item evaluation** (held-out items have 0 raters in train — median 1 rater across the entire raw dataset). This is fundamentally different from a standard CF benchmark where held-out items have rich co-occurrence signal.

Rather than choose one regime, we **use both** as the project's evaluation design:

**Track A — warm-item LOO (standard CF)**
- Test set: ~24K held-out positives produced by `time_based_split` on authors' train (most-recent positive per user)
- Held-out items have median 20 raters in train → CF models have signal
- Popularity baseline gets Recall@10 ≈ 3.00% (a meaningful floor)
- All 8 Stage 1 models compete
- Question: "given a user's history, predict the next item they'll engage with"

**Track B — cold-item (authors' pre-split test)**
- Test set: 10,393 held-out positives from `interactions_test.csv`
- Held-out items have 0 raters in train (cold by construction)
- Popularity baseline gets Recall@10 = 0.00% (correct — popularity can't recommend unseen items)
- Only content-aware models compete: TF-IDF, Sentence-BERT, hybrid, two-tower
- Pure CF models (popularity, MF, EASE, BPR) score ~0 by design
- Question: "given a user's history, predict a recipe NO ONE has rated yet"

**Why both**: this is the project's architectural payoff. Different models win different regimes (BPR/EASE win warm; Sentence-BERT/two-tower win cold; hybrid is the only one that handles both gracefully). It also adds a real cold-start story without scope creep — both tracks use the same model implementations, just evaluated against different test sets. The Stage 2 multi-constraint reranker is applied to both tracks; the α-sweep study runs on both.

**Layperson framing for the deck**: "Recipe sites have two challenges. They need to recommend recipes from the known library (the things people are already cooking) AND recommend brand-new recipes that just got added (the things nobody has tried yet). PantryPlate evaluates on both — same models, same constraints, two evaluation regimes that map to those two scenarios."

---

## 3. Visual identity to preserve

The v1 deck has a consistent visual structure. Match it:

- **Slide numbering**: "N / 17" in a corner, clearly visible
- **Section label at top**: caps, large, e.g., "THE PROBLEM", "OUR CONTRIBUTION", "DATASET", "ARCHITECTURE", "STAGE 1 MODELS", "STAGE 2 RERANKER", "X-FACTOR", "EVALUATION · [SUBTOPIC]", "REACH GOAL", "TIMELINE", "DEMO & CLOSE"
- **Headline + subhead**: each slide has a short bold headline below the section label, then a one-line italicized or smaller subhead explaining the slide's role
- **Pull-out emphasis boxes**: caps labels like "GAP", "HEADLINE HYPOTHESIS", "RESEARCH OBJECT", "RISK", "ALWAYS-ON FALLBACK", "POSITIONING", "HEADLINE PLOT" — these stand out as highlighted callouts
- **Multi-column layouts**: Slide 2 uses a four-quadrant layout for the four constraints; Slide 9 uses a three-column layout for metric categories; Slide 13 uses a 6-column results table
- **Footer**: a small italicized label on each slide indicating its section role (e.g., "Problem and significance", "Algorithms — Stage 1", "Evaluation — metrics overview")
- **Typography**: clean sans-serif, hierarchy via size/weight not color; uses em-dashes consistently
- **Color scheme**: neutral academic — likely white/cream background with dark text, one accent color for emphasis. Avoid clutter.

If python-pptx, use `Pt(40)` for section labels, `Pt(24)` for headlines, `Pt(16)` for body, `Pt(11)` for footers as approximate sizes. Adjust to taste.

---

## 4. Slide-by-slide specification (all 17 slides)

For each slide: STATUS = [UNCHANGED | MINOR EDIT | MAJOR REWRITE | NEW]. The "old text" is the actual content from the v1 deck. The "new text" is what should appear in v2.

---

### Slide 1 — Title    [STATUS: UNCHANGED]

```
[Header label]  PROJECT 2 PROPOSAL
[Title]         PantryPlate
[Subtitle]      Multi-constraint recipe recommendation balancing
                taste, pantry, nutrition, and dietary restrictions

[Tagline]       A study of how recommendations evolve as users tune the relative
                importance of competing objectives.

[Footer]        Recommender Systems  ·  Project 2 Proposal  ·  3 June 2026
```

---

### Slide 2 — The Problem    [STATUS: MINOR EDIT]

**Section label**: THE PROBLEM (2 / 17)

**Headline**: "What should I cook tonight?"
**Subhead**: Harder than existing recommenders treat it as.

**Four-quadrant constraints layout**:
- **Pantry** — *change subtitle from "What's in the fridge right now" to* **"What's on hand and available"** (acknowledges that pantry is a ranking signal, not a hard filter)
- **Nutrition** — "Calorie and macro alignment" (unchanged)
- **Diet** — "Vegetarian, gluten-free, allergens" (unchanged)
- **Taste** — "What they'd actually enjoy" (unchanged)

**Pull-out box "GAP"**: "Existing recommenders (Spoonacular, MyFitnessPal, Eat This Much) optimize for one constraint at a time. None model the trade-off explicitly."

**Footer**: Problem and significance

---

### Slide 3 — Our Contribution    [STATUS: UNCHANGED]

**Section label**: OUR CONTRIBUTION (3 / 17)

**Headline**: The trade-off as our research object
**Subhead**: We make multi-constraint balancing the unit of study, not a side-effect.

**Body**: "We build a recommender that explicitly models the joint trade-off between four objectives, and study how recommendations evolve as users tune the relative weights."

**Pull-out box "HEADLINE HYPOTHESIS"**: "Balanced multi-objective ranking produces qualitatively better recommendations than single-objective optimization on a composite usefulness metric."

**Closing line**: "The slider between objectives isn't a UI feature — it's the empirical finding."

**Footer**: Significance and X-factor framing

---

### Slide 4 — Dataset    [STATUS: MAJOR REWRITE — load-bearing fix]

**Section label**: DATASET (4 / 17)

**Headline**: Food.com via Kaggle
**Subhead**: Real user-recipe interactions at scale. No scraping required.

**Stats panel (replace v1 numbers)** — present as 4 large stat blocks:
- **Recipes**: 231,637
- **User reviews**: 1.07M  *(after dropping 5.4% zero-star "review without rating" entries; raw 1.13M)*
- **Active users (≥5 ratings)**: **22,018**  *(carry 79% of rating mass — this is the primary modeling cohort)*
- **All users**: 226,570  *(cold-start evaluated as separate slice)*

**Secondary panel (new — supporting facts)**:
- Time range: 2000-2018 (peak activity in 2008; 2018 has 10% of peak)
- Rating distribution: 88.6% are 4+ stars, 72.1% are 5 stars *(motivates implicit-feedback models)*
- Nutrition coverage: 100% of recipes have valid 7-axis nutrition vector
- 8,023 canonical ingredients via `ingr_map.pkl`

**Body line**: "Per-recipe data includes structured ingredients, 7-axis nutrition vector (calories, fat, sugar, sodium, protein, sat fat, carbs), dietary tags, descriptions, and time-stamped reviews."

**Pull-out box "WEEK 1 SANITY CHECKS"**: Nutrition coverage rate ✓ STRONG · Interaction density ⚠ ACCEPTABLE · Tag taxonomy ✓ STRONG · Time range ✓ · Ingredient parsing ✓

**Footer**: Dataset, methodology, real Week 1 numbers · kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

---

### Slide 5 — Architecture    [STATUS: MINOR EDIT]

**Section label**: ARCHITECTURE (5 / 17)

**Headline**: Two-stage pipeline
**Subhead**: Recsys learning in Stage 1; constraint reasoning in Stage 2.

**Diagram (vertical flow)**:
```
USER CONTEXT
Rating history · pantry · macro targets · dietary restrictions · α-weights
    ↓
STAGE 1 · CANDIDATE GENERATION (REAL RECSYS)
MF, EASE, BPR, content TF-IDF, Sentence-BERT, hybrid, two-tower neural, sequential
(EDIT: replaced "MF, BPR, content, hybrid, NCF, sequential, LLM-augmented" — see Slide 6 for full menu)
— learns taste from observed ratings. Returns top-200 candidates.
    ↓
STAGE 2 · MULTI-CONSTRAINT RERANKING (X-FACTOR)
Scores each candidate on taste × pantry × nutrition × diet.
Combines via α-weighting. Returns top-10.
    ↓
Top-10 recommendations
```

**Closing line**: "Recsys is the project. Constraints are how we extend it."

**Footer**: Architecture overview

---

### Slide 6 — Stage 1 Models    [STATUS: MAJOR REWRITE — load-bearing fix + dual-track annotation]

**Section label**: STAGE 1 MODELS (6 / 17)

**Headline**: Candidate generators by course week
**Subhead**: Ablation comparison across the full recsys stack.

**Model list (replace v1's 8-model list)**:

| Model | Course Week | Role |
|---|---|---|
| Popularity baseline | W1 | Lower-bound reference |
| Matrix factorisation / ALS via Cornac | W1 | Rating-prediction reference baseline |
| **EASE (new)** | W1-W3 | Closed-form implicit-feedback shallow model |
| Bayesian Personalised Ranking (BPR) | W3 | SGD pairwise implicit ranker |
| Content-based — ingredients TF-IDF, tags, nutrition | W4 | Content-only reference |
| **Two-tower neural** (replaces Neural CF) | W4-W5 | Deep + multimodal-ready |
| Hybrid linear (α·CF + (1-α)·content) | W5 | Combination |
| Sentence-BERT content | W9 | Modern semantic content |
| Sequential — SASRec or GRU4Rec | W8 | Stretch goal |

**Body line**: "Each candidate set passes through the same Stage 2 reranker, isolating the contribution of the Stage 1 architecture."

**New annotation (add this — dual-track related)**: tag each model row as either "CF only" (popularity, MF, EASE, BPR — work only on warm-item evaluation) or "content-aware" (TF-IDF, Sentence-BERT, hybrid, two-tower — work on both warm and cold). Could be a small column or a visual badge.

**New footnote (add this — important)**: "Rating distribution is heavily 4-5 star (88.6% are 4+ stars). Implicit-feedback models (EASE, BPR) are expected to outperform rating-prediction MF on Recall@K. MF is included for paradigm comparison, not as a primary contender. CF-only models score ~0 on cold-item evaluation by construction — content-aware models are needed for the cold-item track."

**Footer**: Algorithms — Stage 1

---

### Slide 7 — Stage 2 Reranker    [STATUS: MINOR EDIT]

**Section label**: STAGE 2 RERANKER (7 / 17)

**Headline**: The four constraint scores
**Subhead**: Where the X-factor lives.

**Score definitions** (edit to clarify hard vs soft):
- **s_taste** — predicted relevance from Stage 1 *(continuous, [0,1])*
- **s_pantry** — ingredient overlap with user's pantry, *as a soft ranking signal* *(continuous, [0,1])*
- **s_nutrition** — Gaussian proximity to macro targets *(continuous, [0,1])*
- **s_diet** — *hard filter:* 1 if all restrictions met, 0 otherwise

**Formula** (centered, prominent):
`final(u, r) = s_diet × (αt · s_taste + αp · s_pantry + αn · s_nutrition)`

**New small note (add this — important)**: "Only `s_diet` is a hard filter. The other three are continuous ranking signals — weighted, never excluding. For dietary restrictions without tag coverage in the dataset (keto, paleo, whole30, lactose-free, halal, low-sugar), `s_diet` is derived from macros plus an ingredient blocklist."

**Pull-out box "RESEARCH OBJECT"**: "The (αt, αp, αn) simplex. Where on this surface do recommendations land, and what does the optimal balance look like?"

**Footer**: Algorithms — Stage 2 reranker

---

### Slide 8 — X-Factor    [STATUS: MINOR EDIT — note dual-track applies to α-sweep]

**Section label**: X-FACTOR (8 / 17)

**Headline**: The α-sweep study
**Subhead**: The trade-off curve is the headline empirical result.

**Body line**: "We sweep α across the simplex and measure how each metric shifts."

**Three hypotheses (numbered)**:
- HYPOTHESIS 1: single-objective optimization produces worse Useful Recall@K than balanced ranking
- HYPOTHESIS 2: optimal α-weighting differs by user type — fitness-focused users weight nutrition higher; budget cooks weight pantry higher
- HYPOTHESIS 3: as constraints tighten, learned taste signal remains material — distinguishing this from pure filtering

**Pull-out box "HEADLINE PLOT"**: "Useful Recall@K vs α-position. Inflection point visible. Per-persona overlays."

**New small note (add this — dual-track related)**: "The α-sweep runs on both evaluation tracks (warm-item LOO and cold-item) so we can study how the trade-off behaves under different recommendation scenarios."

**Footer**: X-factor — what we'll learn that's new

---

### Slide 9 — Evaluation · What We Measure    [STATUS: UNCHANGED]

**Section label**: EVALUATION · WHAT WE MEASURE (9 / 17)

**Headline**: Three categories of metric
**Subhead**: Each plays a distinct role. Together they evaluate the full system.

**Three-column layout**:
- **Predictive** ("did the model guess right?"): Recall@5, @10 · NDCG@10 · MRR
- **Mechanical** ("does output have desired properties?"): Pantry Hit Rate@K · Nutrition Adherence@K · Diet Compliance@K
- **Joint (signature)** ("both at once"): **Useful Recall@K** · Multi-Constraint Score · α-sweep curves

**Closing line**: "Bootstrap 95% CIs on all means. Wilcoxon signed-rank for model-vs-model comparisons."

**Footer**: Evaluation — metrics overview

---

### Slide 10 — Evaluation · Ground Truth    [STATUS: UNCHANGED]

**Section label**: EVALUATION · GROUND TRUTH (10 / 17)

**Headline**: What "ground truth" actually means here
**Subhead**: There is no canonical "correct top-10." So what do we compare against?

**Body**: "We use prediction reframing: the model is given a user's past ratings and asked to predict their future ones."

**Two-column**:
- **PREDICTIVE METRICS**: Ground truth = the recipe a user actually rated next in held-out data. We grade the model on whether that recipe appears in its top-K recommendations from the rest of the catalog.
- **MECHANICAL METRICS**: No ground-truth labels needed. Computed deterministically: does this recipe fit the user's pantry? Does it hit the macro targets? Yes / no, by simple comparison.

**Closing line**: "The 'user's real next behavior' is the only ground truth we can defend. The constraint side doesn't need ground truth — it's checked mechanically."

**Footer**: Evaluation — what counts as truth

---

### Slide 11 — Evaluation · Protocol    [STATUS: MINOR EDIT]

**Section label**: EVALUATION · PROTOCOL (11 / 17)

**Headline**: The leave-one-out protocol, step by step
**Subhead**: For each user with sufficient interactions.

**Numbered steps**:
1. Sort the user's ratings by date. Most recent first.
2. Hold out the user's most recent positive rating. Definition of "positive": 4 stars or higher.
3. Train the model on all earlier ratings. The model never sees the held-out item.
4. Ask the model for a top-10 recommendation list. From the full catalog, excluding already-rated items.
5. Check whether the held-out recipe is in that top-10. If yes, it's a hit for that user.
6. Aggregate across all users. Mean across users gives Recall@10, NDCG@10, MRR.

**New closing line (add this — concretes the numbers)**: "Headline evaluation cohort: 22,018 active users (≥5 ratings) with 823,334 training interactions and 22,012 held-out positives. Cold-start users (fewer than 5 ratings) are evaluated separately, not in the headline numbers."

**Footer**: Evaluation — protocol

---

### Slide 11b — Evaluation · Dual-Track Design    [STATUS: NEW SLIDE — insert between 11 and 12]

**Section label**: EVALUATION · DUAL-TRACK DESIGN (11b / 18)

*Note: this adds 1 slide; total deck becomes 18 slides. Renumber 12-17 to 13-18, or alternatively insert as "11b" and keep parent numbering. Up to layout preference.*

**Headline**: Two evaluation regimes, same architecture
**Subhead**: Real recipe apps face both scenarios — we evaluate on both.

**Body** (two-column layout, side-by-side):

| **Track A — Warm-item (standard CF)** | **Track B — Cold-item (content-aware)** |
|---|---|
| Test set: ~24K held-out positives produced by our time-based leave-one-out on the authors' train cohort | Test set: 10,393 held-out positives from the authors' `interactions_test.csv` |
| Held-out items have **median 20 raters** in train — rich co-occurrence signal | Held-out items have **0 raters** in train — pure cold-item generalization |
| Question: *"given the user's history, what's the next item they'll engage with from the known library?"* | Question: *"given the user's history, what novel recipe should we recommend before anyone has rated it?"* |
| All 8 Stage 1 models compete | Only content-aware models compete (TF-IDF, Sentence-BERT, hybrid, two-tower) |
| Popularity floor: **Recall@10 ≈ 3.0%** | Popularity floor: **0.0% by construction** — popularity has no signal for unseen items |
| Real-world analogy: *"trending recipes among users like me"* | Real-world analogy: *"surface new recipes that match my taste, before they go viral"* |

**Pull-out box "WHY BOTH"**: "Different models win different regimes. CF-only models (BPR, EASE) dominate warm. Content models (Sentence-BERT, two-tower) dominate cold. Only the hybrid handles both gracefully — that's the architectural payoff. The Stage 2 multi-constraint reranker is applied to both tracks; the α-sweep study runs on both."

**Closing line**: "Recipe apps need both. PantryPlate evaluates on both."

**Footer**: Evaluation — dual-track design

---

### Slide 12 — Evaluation · Useful Recall@K    [STATUS: MAJOR REWRITE — load-bearing fix]

**Section label**: EVALUATION · USEFUL RECALL@K (12 / 17)

**Headline**: The signature metric in plain language
**Subhead**: What makes Useful Recall@K specific to PantryPlate.

**Body**:

"Standard Recall@K asks one question:
*Did the model rank a known future positive interaction in the top-K?*

**Useful Recall@K asks two**:
1. Did the model rank a known future positive interaction in the top-K?
2. AND does that recommendation satisfy all of the user's constraints?

For a recommendation to count as 'useful', the held-out recipe must appear in the model's top-K AND satisfy:
- **Pantry**: user is missing ≤ 3 non-staple ingredients. *(We assume universal staples — salt, pepper, oil, flour, eggs, milk, butter, garlic, onion, etc. — are in every kitchen. This avoids penalizing recipes for needing salt. EDA showed even with staples assumed, an overlap-fraction threshold would be too tight; 'missing ≤ 3' gives ~25% feasibility — discriminative without being trivial.)*
- **Nutrition**: within ±20% of the user's macro targets
- **Diet**: all dietary restrictions respected

This is where model performance and constraint awareness combine into a single number. We report Recall@K and Useful Recall@K side by side — **the gap between them is the project's contribution.**"

**Footer**: Evaluation — signature metric

---

### Slide 13 — Evaluation · Expected Results    [STATUS: MAJOR REWRITE — add EASE row + dual-track columns]

**Section label**: EVALUATION · EXPECTED RESULTS (13 / 17)

**Headline**: What the final results table looks like
**Subhead**: Illustrative numbers — actual values come from experiments.

**Results table — DUAL TRACK** (now reports Recall@10 / Useful Recall@10 on both warm and cold):

| Model | Warm Recall@10 | Warm UR@10 | Cold Recall@10 | Cold UR@10 |
|---|---|---|---|---|
| Popularity | 0.030 | 0.005 | 0.000 | 0.000 |
| MF / ALS | 0.045 | 0.012 | 0.000 | 0.000 |
| **EASE** | 0.060 | 0.018 | 0.000 | 0.000 |
| BPR | 0.055 | 0.016 | 0.000 | 0.000 |
| Content TF-IDF | 0.025 | 0.008 | 0.020 | 0.006 |
| Sentence-BERT | 0.030 | 0.010 | 0.028 | 0.009 |
| Hybrid + reranker | 0.058 | **0.025** | 0.025 | **0.011** |
| Two-tower | 0.052 | 0.020 | 0.030 | 0.010 |

*(All numbers are illustrative — actuals come from Week 2 experiments. Warm numbers track the smoke-test floor (popularity ≈ 3.0%) scaled by expected model lift. Cold numbers are ~0 for pure-CF by construction and small but non-zero for content-aware models.)*

**Closing line**: "Hybrid + reranker wins on Useful Recall@10 in BOTH regimes — that's the architectural payoff. Pure CF models (BPR, EASE) dominate warm-item Recall but cannot recommend cold items. Pure content models do meaningfully on cold but lose to CF on warm. The hybrid is the only model that handles both — and the multi-constraint reranker improves usefulness across the board."

**Below table**: "Plus: α-sweep plot (per track) · per-persona heatmap · modality ablation · qualitative case studies · honest failure case"

**Below table**: "Plus: α-sweep plot · per-persona heatmap · modality ablation · qualitative case studies · honest failure case"

**Footer**: Evaluation — expected output

---

### Slide 14 — Reach Goal    [STATUS: UNCHANGED]

**Section label**: REACH GOAL (14 / 17)

**Headline**: Vision-based pantry input
**Subhead**: If the recsys is on track by Week 4, we add a fridge-photo input mode.

**Body**: "The user takes a fridge photo → pretrained vision-language model returns ingredient list with confidence → the recsys's pantry constraint becomes a soft, probabilistic input."

**Bullets**:
- Uses Claude / GPT-4V / Gemini — zero custom CV training
- The recsys core is unchanged; CV is an input modality only
- Adds two genuine recsys ablations: ground-truth vs CV-derived pantry; hard vs soft confidence-weighted pantry

**Pull-out box "POSITIONING"**: "The recommender is the project. The camera is a sensor for one constraint. We explicitly avoid the CV-with-thin-recsys failure mode."

**Footer**: Reach goal — what gets added if we have time

---

### Slide 15 — Reach Goal · Considerations    [STATUS: UNCHANGED]

**Section label**: REACH GOAL · CONSIDERATIONS (15 / 17)

**Headline**: If we do it — what to think about
**Subhead**: Decision happens at the Week 4 progress gate, not earlier.

**Two-column**:
- **What it adds**: Visual demo moment · Multimodal angle (Week 4) · Two real ablations · Audience interaction in demo
- **What it costs**: ~$5 vision-LLM API budget · ~3 person-days of integration · Internet dependency live · ~25 annotated fridge photos

**Pull-out box "RISK"**: "Scope creep. CV 'improvements' can absorb arbitrary time. We hard-freeze CV by end of Week 5."

**Pull-out box "ALWAYS-ON FALLBACK"**: "Manual ingredient entry. Demo never depends on CV working live."

**Footer**: Reach goal — risk register and decision gate

---

### Slide 16 — Timeline    [STATUS: MINOR EDIT]

**Section label**: TIMELINE (16 / 17)

**Headline**: Six-week schedule
**Subhead**: From May 13 to presentation on June 24.

**Weekly rows (edited to reflect actual W1 progress + updated W2 model menu)**:
- **W1** · May 13-19 · Data pipeline · sanity checks · EDA · feasibility analysis · popularity baseline ✓
- **W2** · May 20-26 · Eval harness · MF · EASE · BPR · content TF-IDF · Sentence-BERT
- **W3** · May 27-Jun 2 · Hybrid + two-tower neural · results table · **Proposal due Jun 3**
- **W4** · Jun 3-9 · Stage 2 reranker · personas (40-60-item pantries) · CV decision gate
- **W5** · Jun 10-16 · α-sweep experiments · ablations · CV integration (if go)
- **W6** · Jun 17-23 · Analysis · slides · demo polish · physical prop · **Present Jun 24**

**Footer**: Milestones

---

### Slide 17 — Demo & Close    [STATUS: UNCHANGED]

**Section label**: DEMO & CLOSE (17 / 17)

**Headline**: What the room sees
**Subhead**: Web widget plus a physical reveal.

**Bullets**:
- Input panel — pick a persona, type pantry, or upload fridge photo (if reach goal completes)
- Three α-sliders — taste, pantry, nutrition. Move them live; the recommendation list re-ranks visibly
- Recommendation cards — each with "why this?" explanation showing constraint scores
- A/B compare — full model vs single-objective baseline, same user, side-by-side
- Audience moment — invite someone to call out ingredients or pick a persona
- Physical prop — one of the top recommendations, cooked beforehand, on the table

**Closing**: Questions?

**Footer**: Thank you · PantryPlate · Project 2 Proposal

---

## 5. Suggested generation approach (Python + python-pptx)

```python
# pip install python-pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

ACCENT = RGBColor(0x2E, 0x4A, 0x66)  # muted navy — pick what looks academic
BODY   = RGBColor(0x22, 0x22, 0x22)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]

def add_section_label(slide, text):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(8), Inches(0.6))
    tf = box.text_frame; p = tf.paragraphs[0]
    p.text = text.upper(); p.font.size = Pt(14); p.font.bold = True
    p.font.color.rgb = ACCENT

def add_slide_number(slide, n, total=17):
    box = slide.shapes.add_textbox(Inches(11.5), Inches(0.3), Inches(1.5), Inches(0.4))
    p = box.text_frame.paragraphs[0]
    p.text = f"{n} / {total}"; p.font.size = Pt(11)
    p.alignment = PP_ALIGN.RIGHT

def add_headline(slide, text, subhead=None):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(12), Inches(1.2))
    tf = box.text_frame
    p = tf.paragraphs[0]; p.text = text; p.font.size = Pt(36); p.font.bold = True
    if subhead:
        sp = tf.add_paragraph(); sp.text = subhead; sp.font.size = Pt(18); sp.font.italic = True

# ... build each slide per the spec in Section 4
prs.save("PantryPlate_Proposal_v2.pptx")
```

Layouts to use for visual variety:
- Slides 1, 3, 8, 12: centered text with prominent pull-out box
- Slides 2, 9: multi-column (4-quadrant for Slide 2, 3-column for Slide 9)
- Slides 4, 6, 13: stat blocks or tables
- Slides 5, 11: vertical flow (arrows for Slide 5, numbered steps for Slide 11)
- Slides 14, 15, 17: two-column with pull-out boxes
- Slide 16: timeline as horizontal week bars

Match v1's restraint — lots of whitespace, no clip-art, no gradients.

---

## 6. Sanity checks for the output

Before declaring done, verify:
- [ ] All 17 slides present, numbered 1/17 through 17/17
- [ ] Slide 4 says **22,018 active users** (not 226K)
- [ ] Slide 6 includes **EASE** and **Two-tower neural** (and does not list NCF as a primary model)
- [ ] Slide 6 has the rating-distribution footnote about implicit-feedback models
- [ ] Slide 7 has the soft-vs-hard note about which scores are filters
- [ ] Slide 12 uses **missing ≤ 3 non-staple ingredients** (not ≥90% overlap, not ≥50% overlap) and macros ±20% (not ±15%); mentions staples-assumed approach
- [ ] New Slide 11b inserted: dual-track evaluation design (warm vs cold)
- [ ] Slide 13 results table has BOTH warm and cold columns; popularity/MF/EASE/BPR show 0 on cold (correct by construction)
- [ ] Slide 6 annotates models as CF-only vs content-aware
- [ ] Slide 8 mentions α-sweep runs on both tracks
- [ ] Slide 13 includes an **EASE row** in the results table
- [ ] Slide 16 timeline mentions EDA in W1 and EASE/Sentence-BERT in W2
- [ ] No claim that you "can cook tonight from your fridge" anywhere — pantry framing is "what you have on hand" / "best match"

If any of these fail, the edit didn't land.

---

## End of brief
