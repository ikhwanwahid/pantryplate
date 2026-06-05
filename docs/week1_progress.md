# PantryPlate — Week 1 Progress Note

**Period**: May 13-19, 2026 (Week 1; +26 May follow-up: staples-aware pantry refinement)
**Status**: Week 1 substantially complete. Pipeline runs end-to-end on real data; popularity baseline produces sensible numbers; EDA + feasibility analysis done; proposal deck updated to reflect findings; staples-aware pantry metric added based on follow-up analysis. Personas + git remote setup outstanding (team).

---

## What was built

### Project infrastructure
- Full directory tree (`data/{raw,processed,personas}`, `src/{data,models,reranker,eval,cv,utils}`, `notebooks/`, `results/`, `tests/`, `demo/`)
- uv-native dependency management via `pyproject.toml` + `uv.lock` (replaces requirements.txt)
- `.venv/` with pandas 3.0, numpy 2.4, scikit-learn 1.8, cornac 2.3, torch 2.12, sentence-transformers 5.5, jupyter, pytest, kaggle
- `.gitignore` excluding `data/raw/`, `.venv/`, model checkpoints, `kaggle.json`
- `git init` (no commits yet — pending team consensus on remote)

### Data layer (`src/data/`)
- `loader.py` — `load_recipes`, `load_interactions` (drops 0-star by default), `filter_active_users`, `time_based_split` (leave-one-out by date on most recent positive)
- `ingredients.py` — `parse_nutrition` (with clipping), `normalize_ingredient` (via `ingr_map.pkl`), `safe_parse_list`, `NUTRITION_FIELDS` constant
- `pantry.py` — `derive_pantry_from_recipes`, `load_persona`

### Models (`src/models/`)
- `popularity.py` — `PopularityRecommender` (the W1 baseline). Standard `.fit(train_df)` and `.recommend(user_id, k, exclude_seen=True)` interface that all Week 2 models will follow.

### Evaluation (`src/eval/`)
- `metrics.py` — `recall_at_k`, `ndcg_at_k`, `mrr` (per-user). Aggregation deferred to the Week 2 harness.

### Utils (`src/utils/`)
- `staples.py` — Universal kitchen staples (~30 canonical items including variants). Exposes `STAPLES`, `DAIRY_AND_EGGS`, `get_staples_for_persona()`, `pantry_score()`, `missing_count()`. Added 2026-05-26 after follow-up pantry analysis (see below).

### Tests (`tests/`)
- `test_data.py` — 24 tests covering parsing, splitting, filtering, pantry derivation
- `test_metrics.py` — 19 tests covering recall/NDCG/MRR including edge cases
- `test_models.py` — 7 tests covering popularity ranking, seen-item exclusion, error paths
- `test_staples.py` — 22 tests covering staples set, per-persona overrides, pantry_score, missing_count
- **Total: 72 tests, all passing in ~2s**

### Scripts at project root
- `sanity_checks.py` — produced `results/week1_sanity_report.txt`
- `smoke_test.py` — produced `results/week1_smoke_test.txt`

### Notebooks
- `notebooks/week1_eda.ipynb` — 17 cells, fully executed with plots inline (~576 KB). Covers Stage 1 feasibility, rating-signal quality, pantry-matching feasibility, nutrition feasibility, diet tag feasibility, evaluation feasibility. Ends with per-pillar verdict table.

### Documents (`docs/`)
- `data_decisions.md` — 11 locked decisions with evidence and code references
- `proposal_deck_rebuild_brief.md` — 591-line self-contained brief that lets any LLM rebuild the proposal deck without re-debate. Used to generate `PantryPlate_Proposal_v2.pptx`.
- `week1_progress.md` — this file

---

## What the sanity checks revealed

All 5 checks passed:

| Check | Result | Verdict |
|---|---|---|
| Nutrition coverage | 100% of 231,637 recipes have valid 7-vector | ✓ STRONG |
| Active users (≥5 ratings) | 22,018 / 226,570 | ⚠ ACCEPTABLE |
| Dietary tags well-covered | 11 of 17 | ✓ STRONG |
| Time range | 2000-01 → 2018-12, valid on all 1.13M interactions | ✓ |
| Ingredient parsing | 100% parseable; 8,023 canonical ingredients | ✓ |

Headline verdict from script: *"Dataset is suitable for the project as scoped."*

---

## What the EDA added on top of sanity checks

The EDA went past pass/fail and asked: *does the data support what each of PantryPlate's four pillars is trying to do?* Findings, in order of consequence:

1. **Stage 1 CF — ⚠ Caution.** 22K active users carry 78.9% of rating mass. Matrix density 2.4×10⁻⁵. Median user has 1 rating. CF will effectively be trained on the active subset only.

2. **Rating signal quality — ⚠ Caution.** Among active users: 21.8% always give the same rating; 57.7% use only 1-2 distinct rating values; mean per-user mean = 4.65. **Implicit-feedback models (BPR, EASE) are expected to outperform rating-prediction MF on Recall@K.** This locks in our Week 2 model priority.

3. **Pantry matching — ⚠ Adjust.** Ingredient frequency is long-tailed (top 73 ingredients cover 50% of mentions). But a 25-item starter pantry produces median `s_pantry` = 0.23; only 11.4% of recipes hit ≥0.5 overlap; only 0.003% hit ≥0.9. **Pantry is a soft ranking signal, not a hard "you can cook tonight" filter.** Personas need 40-60-item pantries to make the signal discriminative.

4. **Nutrition targeting — ✓ Go.** Macro distributions wide; even a tight "fitness lifter" target (400-800 kcal, ≥40% protein PDV, ≤40% carbs PDV) matches 33,981 recipes (14.7%). Clip cap binds on <0.5% of recipes.

5. **Diet filtering — ⚠ Mitigate.** 6 desired dietary tags have **zero** recipes (keto, paleo, whole30, lactose-free, halal, low-sugar). For these we'll derive constraints from macros + ingredient blocklists. Vegan-tag-to-ingredient consistency is 96.9% — tags alone are unreliable; we AND tag checks with ingredient blocklists.

6. **Useful Recall — ✓ Go (with framing).** Held-out positives satisfy realistic constraints at meaningful but non-trivial rates (15.2% vegetarian, 80.2% <600 kcal, 13.5% both). The gap between Recall and Useful Recall is the project's signature.

7. **Dataset is historical, not active.** Peak year 2008 (161K interactions); 2018 has only 10% of peak. Time-based split is fine; recency bias is limited.

8. **Data quirk — `flmy`.** The canonical form for "flour" in `ingr_map.pkl` is `flmy` (and "all-purpose flmy"). Internally consistent; cosmetic only.

---

## Pivots and scope changes from the original plan

| Original plan | Updated plan | Reason |
|---|---|---|
| Pantry constraint implicitly hard ("cook tonight") | Pantry is a soft ranking signal with universal staples assumed | EDA: 25-item pantry → median 0.23 (with mixed staples); with strict staples separation, ≥0.9 essentially empty |
| `requirements.txt` | `pyproject.toml` + `uv.lock` (uv-native) | User preference (uv workflow) |
| Stage 1 menu: 8+ models including NCF, item-CF, WMF | Tighter menu: Popularity, MF, EASE, BPR, TF-IDF, Sentence-BERT, Hybrid, Two-tower (+ SASRec stretch) | NCF redundant with two-tower; item-CF subsumed by EASE; WMF redundant with BPR. EASE added (literature shows strong results; closed-form, low cost). |
| Week 2 model priority: MF first, BPR later | Implicit-feedback (EASE, BPR) prioritized; MF as reference | Rating distribution is heavily 4-5 star; rating-magnitude prediction has little gradient |
| Useful Recall pantry threshold ≥ 0.9 → 0.5 → `missing ≤ 3` | Final: `missing_count ≤ 3` non-staple items | ≥ 0.9 produces zero matches; ≥ 0.5 produces 2.6%; `missing ≤ 3` produces 24.7% — workable and intuitive |
| MF/ALS via Cornac as a primary contender | MF as paradigm-comparison baseline only | Same reason as model-priority swap |
| Personas with ~20-25 item pantries → 40-60 → 25-35 | Final: 25-35 user-specific items; staples are project-wide constant | Staples-aware refinement decoupled staples from per-persona pantries; smaller user-pantries are easier to author and the staples constant covers what the bigger pantries were trying to capture |
| Proposal deck v1 as authored → v2 staples-aware | Proposal deck v2 rebuilt to reflect findings; Slide 12 wording updated 2026-05-26 for staples + missing-count threshold | 3 load-bearing fixes (Slides 4, 6, 12) and several smaller updates. Brief in `docs/proposal_deck_rebuild_brief.md`. |

---

## End-to-end smoke test results

The Week 1 pipeline runs cleanly on real data:

```
[1/5] Loading interactions ........... 1,071,520 in 11.6s
[2/5] Filtering active users (>=5) ... 22,018 users / 845,346 ratings
[3/5] Time-based split ............... train: 823,334 / test: 22,012
[4/5] Training popularity ............ 205,618 recipes ranked in 0.6s
[5/5] Eval over 2,000 sampled users:

      metric           mean
      recall@5       0.0070
      recall@10      0.0115
      recall@20      0.0250
      ndcg@5         0.0040
      ndcg@10        0.0055
      ndcg@20        0.0088
      mrr            0.0045
```

Recall@10 = 1.15% is squarely in the expected 1-3% range for a popularity baseline at this dataset scale. **This is the floor that every Week 2 model needs to beat.**

---

## Follow-up: staples-aware pantry refinement (2026-05-26)

After Week 1 wrap-up, a follow-up question — *does the original pantry analysis penalize recipes for needing condiments everyone has?* — prompted a more careful staples-aware analysis. Key findings:

- A 25-item "user pantry" plus 38 universal staples (salt, pepper, water, oil, flour, eggs, milk, butter, garlic, onion, baking soda/powder, vinegar, vanilla, cinnamon, cornstarch, paprika, honey, chicken broth + canonicalization variants) gives a much more honest feasibility picture. The staples set was empirically validated against ingredient frequencies in EDA §4b: 79% rank in the top-50 most common ingredients; staples explain ~35% of the average recipe.
- Under the staples-aware `pantry_score(recipe, user_pantry)` formula (overlap on non-staple ingredients only), **median s_pantry is near 0**, because a 25-item user pantry rarely contains all the specific non-staple items in a typical recipe. This is fine for the reranker — it still ranks recipes from "best match" to "worst match" *within a single user*.
- For the **Useful Recall metric**, however, an overlap-fraction threshold of ≥ 0.5 only matches 2.6% of recipes — too tight to be discriminative. Switched to `missing_count(recipe, user_pantry) ≤ 3` which matches ~25% of recipes and is more intuitive ("user only needs to buy 3 more things").
- Personas can now be 25-35 user-specific items instead of 40-60 (staples are project-wide).
- Vegan personas auto-drop dairy/eggs from staples via `get_staples_for_persona()`.

**Code added**: `src/utils/staples.py` (90 lines, 22 tests). **Updated**: EDA notebook Section 4 + verdict + decisions; `data_decisions.md` §7 + §10 + summary table; `proposal_deck_rebuild_brief.md` §2b + Slide 12 spec + sanity checklist; walkthrough notebook decisions cheat sheet.

The reranker formula in §7 of `data_decisions.md` and Slide 7 of the proposal deck is unchanged — only the definition of `s_pantry` evolved. The architecture remains:
`final(u, r) = s_diet × (αt · s_taste + αp · s_pantry + αn · s_nutrition)`

---

## What's next (Week 2 — May 20-26)

**Critical path**: one team member owns the eval harness (`src/eval/harness.py`) on Day 1. Every other model evaluation depends on it. While the harness is being built, others can pre-work content models (TF-IDF, Sentence-BERT) which have their own scoring paths.

| Day | Suggested focus |
|---|---|
| Mon | Eval harness (B1) · TF-IDF scaffold (B5) · SBERT embedding pass (B6) |
| Tue-Wed | EASE (B3) · BPR (B4) · MF (B2) · content models complete |
| Thu | Hybrid (B7) · two-tower (B8) · results table (B9) starts |
| Fri | BPR tuning · results aggregation · `week2_progress.md` |

**Week 2 deliverable**: results table with ≥5 models compared on Recall@10, NDCG@10, MRR — ready to inform the proposal deck's Slide 13.

---

## Outstanding Week 1 work for the team

These weren't done by Claude Code because they require human judgment or team consensus:

1. **5 generic persona JSONs** (`data/personas/`) — fitness_focused, vegan_busy, family_friendly, low_sodium, quick_dinner. Each with id, label, description, macro_targets, restrictions, **40-60 item pantry**, 20-30 taste-seed recipe IDs.

2. **5 team self-profile JSONs** — one per teammate. Pick 20-30 favorite recipes from Food.com (cross-reference IDs in `data/raw/RAW_recipes.csv`), define own pantry, macros, restrictions.

3. **Git remote setup + first push** — decide where the repo lives (private GitHub recommended), add as remote, push everything except `data/raw/`.

4. **Visual review of `PantryPlate_Proposal_v2.pptx`** — open in PowerPoint/Keynote, verify Slides 6 and 12 (the two slides with the most added content density) don't have layout issues. Tune accent color and font weights to team taste.

5. **Pick team lanes for Week 2.** Suggested affinities: Cornac/CF specialist · content/NLP specialist · deep learning (PyTorch) specialist · eval/analysis specialist · persona/demo/slides specialist.

6. **Schedule a Week 2 kick-off sync.** Confirm everyone read this doc and `docs/data_decisions.md`. Pick lanes. Confirm Monday's blocker (eval harness) has an owner.

---

## Files modified or created this week

```
docs/
  data_decisions.md             [new]
  proposal_deck_rebuild_brief.md [new]
  week1_progress.md             [new]
notebooks/
  week1_eda.ipynb               [new, fully executed]
results/
  week1_sanity_report.txt       [new]
  week1_smoke_test.txt          [new]
src/data/
  __init__.py                   [new]
  ingredients.py                [new]
  loader.py                     [new]
  pantry.py                     [new]
src/eval/
  __init__.py                   [new]
  metrics.py                    [new]
src/models/
  __init__.py                   [new]
  popularity.py                 [new]
src/{reranker,utils,cv}/
  __init__.py                   [new — empty, scaffolding for later weeks]
tests/
  __init__.py                   [new]
  test_data.py                  [new]
  test_metrics.py               [new]
  test_models.py                [new]
.gitignore                      [new]
pyproject.toml                  [new]
uv.lock                         [new]
smoke_test.py                   [new]
PantryPlate_Proposal_v2.pptx    [new, generated from docs/proposal_deck_rebuild_brief.md]
```

The Week 1 sanity script (`sanity_checks.py`) and the Week 1 task list (`docs/week1_tasks.md`) and the original deck (`PantryPlate_Proposal.pptx`) were provided at session start.
