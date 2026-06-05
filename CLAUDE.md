# Claude Code instructions for PantryPlate

**Read this first** if you're an AI assistant picking up work on this project. This page is the entry point for Claude Code / Cursor / Claude.ai / etc. — it tells you the current state, what NOT to re-debate, and how to make decisions consistent with prior choices.

---

## Project status at a glance

- **What this is**: graduate Recommender Systems course project. Multi-constraint recipe recommender.
- **Final presentation**: 24 Jun 2026
- **Proposal**: submitted 3 Jun 2026 → `PantryPlate_Proposal.pdf`
- **Current phase**: Week 4 build sprint. Eval harness ✓ done. Stage 1 models pending (see `docs/week2_onboarding.md` §4b). Stage 2 reranker not yet started.

For a complete project overview, read **`README.md`** first, then **`docs/week2_onboarding.md`** (532 lines, comprehensive onboarding).

---

## Critical reading list (in this order)

If you're going to make any non-trivial change, read these first:

| Priority | File | Why |
|---|---|---|
| 1 | `README.md` | Project structure, setup, key concepts |
| 2 | `docs/week2_onboarding.md` | ⭐ Comprehensive onboarding (~30 min read). Data file routing, model contract, locked decisions, conventions, troubleshooting, glossary. |
| 3 | `docs/data_decisions.md` | The 11 locked decisions with empirical evidence. If your instinct conflicts with one of these, the decision wins. |
| 4 | `docs/eval_harness_usage.md` | How to evaluate any Stage 1 model. The harness is the single sanctioned eval path. |
| 5 | `PantryPlate_Proposal.pdf` | The submitted proposal. Captures the project's pitch and architecture. |

---

## Things you must NOT re-debate

These have been decided with evidence and documented in `docs/data_decisions.md`. They are **load-bearing** — changing them invalidates work that depends on them. If you have a strong reason to revisit, surface it to the user explicitly; don't silently override.

1. **Training cohort** = authors' pre-split `interactions_train.csv` (24,961 active users). Use `load_train_interactions()`.
2. **Dual-track evaluation**: warm-item LOO (Track A) + cold-item from authors' test (Track B). Use the harness.
3. **0-star ratings are dropped** during loading (they're "review without rating" entries).
4. **Positive rating threshold = 4 stars** for LOO holdout logic.
5. **Stage 1 model menu is set**: Popularity, MF, EASE, BPR, TF-IDF, Sentence-BERT, hybrid, two-tower (+ SASRec stretch). Don't propose adding/swapping models unless coordinated.
6. **Pantry score uses non-staple overlap**; Useful Recall uses `missing_count ≤ 3`. Both live in `src/utils/staples.py`.
7. **Diet is a hard filter** in Stage 2 (others are continuous ranking).
8. **Nutrition clipping** at (5000 kcal, 1000% PDV) — already in the loader.
9. **Personas**: 25-35 user-specific items per pantry; staples are project-wide via `src/utils/staples.py`. 3 personas already exist in `data/personas/`.
10. **Eval harness is the only sanctioned evaluation path**. Don't reimplement Recall@K or run your own loops over users.
11. **Stage 1 model interface**: `.fit(train_df) -> self` + `.recommend(user_id, k, exclude_seen=True) -> list[int]`. Reference impl: `src/models/popularity.py`.

---

## How to work in this repo

### Always run from project root

```bash
cd /path/to/CS608Project2/
uv run pytest tests/ -q    # tests
uv run python smoke_test.py # quick pipeline check
```

Running from subdirectories breaks imports.

### Tests must pass before claiming work done

```bash
uv run pytest tests/ -q
# Expected: 98+ tests pass in ~10 seconds
```

If your changes break tests, fix them before pushing. Don't disable tests.

### Use uv, not pip

The project uses `uv` for dependency management. To add a package:

```bash
uv add <package>           # adds to pyproject.toml + uv.lock
```

Don't run `pip install` — it won't update the lock file.

### Match the existing style

- Type hints throughout (`def fn(x: int) -> str:`)
- Short docstrings on public functions
- Snake_case for files and functions; TitleCase for classes
- One model per file in `src/models/`
- Deterministic by default (seed control on all randomness)
- No emoji in code unless the user explicitly asked

### Don't create files unnecessarily

Match the existing organization. Don't create a new directory or doc unless asked. If a piece of information fits in an existing doc, add it there.

---

## Conventions specific to this project

### Stage 1 model contract

Every model in `src/models/` must satisfy:

```python
class YourRecommender:
    def __init__(self, seed: int = 42): ...
    def fit(self, train_df: pd.DataFrame) -> "YourRecommender": ...
    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]: ...
```

`train_df` columns: `user_id`, `recipe_id`, `date`, `rating`, `u`, `i`.

### Evaluation

```python
from src.eval.harness import evaluate
result = evaluate(model, track="warm")   # or track="cold"
# returns {"recall@10": ..., "ndcg@10": ..., "mrr": ..., "n_users_evaluated": ...}
```

For comparing multiple models or computing bootstrap CIs, see `docs/eval_harness_usage.md`.

### Cold-track expectations

- Pure-CF models (Popularity, MF, EASE, BPR) score **exactly 0** on Track B by construction (cold items have no rater history). This is **correct, not a bug**.
- Content-aware models (TF-IDF, Sentence-BERT, hybrid, two-tower) should produce **non-zero** numbers on cold.

### Staples and personas

- `src/utils/staples.py` defines `STAPLES`, `pantry_score()`, `missing_count()`, `get_staples_for_persona()`. Use these — don't reimplement.
- Personas live in `data/personas/`. Three exist: `fitness_focused`, `vegan_busy`, `family_friendly`. Schema documented in `data/personas/README.md`.
- For Stage 2 work, use `pantry_score()` (continuous, [0,1]) for the reranker; use `missing_count() ≤ 3` for the Useful Recall threshold.

### Git workflow

- **`main` is protected** — direct pushes blocked except for the repo admin (ikhwanwahid).
- All other contributors use **feature branch + PR + 1 approval**.
- Branch naming: `model/<name>-<author>`, `feature/<short-desc>`, `fix/<short-desc>`, `docs/<short-desc>`.
- Commit messages: `<verb> <what> — <why if non-obvious>`.
- To claim a model from `docs/week2_onboarding.md` §4b: branch from main, edit Status to 🟡 + add Owner name, push branch, open PR, get 1 approval, merge. First merged PR wins.
- Always run `uv run pytest tests/ -q` locally before requesting review.

Full workflow detail in `docs/week2_onboarding.md` §6 "Git workflow".

---

## What's pending (as of 2026-06-04)

If you're picking up work, these are the open workstreams in priority order:

| # | Workstream | State | Notes |
|---|---|---|---|
| 1 | Content model PoC (Sentence-BERT) | Pending — assigned to Ikhwan | Pipeline validation; first content-aware model |
| 2 | Stage 2 reranker scaffold | Pending — assigned to Ikhwan | 4 score functions + combiner; validates end-to-end with Sentence-BERT |
| 3 | BPR (Cornac) | Open | Warm-track CF |
| 4 | EASE | Open | Warm-track CF, closed-form |
| 5 | TF-IDF content | Open | Cold-track content reference |
| 6 | Hybrid linear | Open (depends on #1 + #3 or #4) | Expected overall winner |
| 7 | α-sweep experiments | Pending (after Stage 2) | Generates the headline plot |
| 8 | Demo widget (Streamlit) | Pending | Week 5-6 work |
| 9 | Slide deck updates | Pending | Replace placeholder Stage 1 results with real numbers |
| 10 | Physical prop | Pending | Cook one of the recommended recipes |

See `docs/week2_onboarding.md` §4b for the live coordination table with effort estimates and owners.

---

## When making code changes

1. **Read the relevant doc first**. Don't infer from filenames.
2. **Run tests** before and after your change.
3. **Match existing patterns**. The reference Stage 1 implementation is `src/models/popularity.py`. The reference utility module is `src/utils/staples.py`.
4. **Update docs if you change conventions**. If you change a locked decision, update `docs/data_decisions.md` and the affected docs.
5. **Use TaskCreate/TaskUpdate** to track multi-step work in your session. This is a Claude Code convention.

---

## What this project is NOT

To prevent scope drift:

- Not a meal-planning system (no multi-day planning)
- Not a calorie tracker (we use targets; we don't track intake)
- Not a recipe generator (we retrieve from Food.com catalog)
- Not a CV-driven system (CV reach goal dropped — see proposal)
- Not a real-time deployed service (this is a course project; final deliverable is presentation + cooked recipe)

If a feature request would push the project into one of these areas, push back.

---

## Common first-session tasks

If a new Claude Code session is asked to do generic work, here's what they should do:

| User says... | Right move |
|---|---|
| "Build a model" | Check `docs/week2_onboarding.md` §4b for unclaimed model. Then read the model contract in §3. Reference `src/models/popularity.py`. |
| "Run the tests" | `uv run pytest tests/ -q` from project root. |
| "Evaluate a model" | Use `evaluate()` from `src/eval/harness.py`. Reference `docs/eval_harness_usage.md`. |
| "Add a persona" | Schema in `data/personas/README.md`. Save as `<id>.json`. 25-35 user-specific pantry items. |
| "Build the demo" | Streamlit. Reference the Slide 17 demo plan in the proposal. Three α-sliders + persona switcher. |
| "Update the proposal slides" | Latest editable source is `PantryPlate_Proposal.pptx`. Final submitted version is `PantryPlate_Proposal.pdf` (don't edit). |

---

## Project-specific gotchas

- The dataset's canonical form for "flour" is **"flmy"** (artifact of original processing). Not a bug. `STAPLES` includes both.
- `interactions_validation.csv` and `interactions_test.csv` are **never read by Stage 1 model code**. The harness reads them. Common cause of training-on-test bugs.
- Cornac may need `libomp` on macOS or Cython at build time. See troubleshooting in `docs/week2_onboarding.md` §8.
- Sentence-BERT first import takes 30+ seconds (model loading). Subsequent imports are fast. Cache recipe embeddings to `data/processed/` to avoid re-encoding 230K recipes on every fit.

---

## Final note

This is a graduate-level course project, not production code. The bar is **methodological rigor + a working end-to-end demonstration**, not enterprise-grade architecture. When in doubt, prefer the simpler thing that gives a defensible empirical result over the more elaborate thing that's harder to ship in 3 weeks.
