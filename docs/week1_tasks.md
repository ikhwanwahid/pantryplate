# First Claude Code Session — Task List

This is what to ask Claude Code to do in your first session. Hand it the
README.md and this file, and work through the tasks in order. Stop after
each numbered block to verify before moving on.

---

## Setup (15 minutes)

1. **Create the project directory and structure**:
   ```
   pantryplate/
   ├── data/raw/
   ├── data/processed/
   ├── data/personas/
   ├── src/data/
   ├── src/models/
   ├── src/reranker/
   ├── src/eval/
   ├── notebooks/
   ├── demo/
   ├── results/
   ├── docs/
   └── tests/
   ```

2. **Initialize git, add `.gitignore`** that excludes:
   - `data/raw/` (CSVs too big for git)
   - `data/processed/*.db` (SQLite DBs)
   - `__pycache__/`, `*.pyc`
   - `.env`, `*.key`
   - `results/checkpoints/` (model files)

3. **Create `requirements.txt`** with:
   - pandas, numpy, scikit-learn (data)
   - cornac (baseline recsys models)
   - pytorch (deep learning)
   - sentence-transformers (text embeddings)
   - matplotlib, seaborn (plots)
   - pytest (tests)
   - jupyter (notebooks)

4. **Create a virtual environment, install deps**:
   ```
   python -m venv venv
   source venv/bin/activate  # or .\venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

---

## Data acquisition (30 minutes)

5. **Download Food.com dataset from Kaggle**:
   ```
   pip install kaggle  # if not already installed
   # Set up ~/.kaggle/kaggle.json with API token
   kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions
   unzip food-com-recipes-and-user-interactions.zip -d data/raw/
   ```
   Or download manually from https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions and extract to `data/raw/`.

6. **Verify expected files exist**:
   - `data/raw/RAW_recipes.csv` (~300 MB)
   - `data/raw/RAW_interactions.csv` (~600 MB)
   - `data/raw/PP_recipes.csv`
   - `data/raw/PP_users.csv`
   - `data/raw/interactions_train.csv` etc.
   - `data/raw/ingr_map.pkl`

---

## Sanity checks (1-2 hours)

7. **Copy `sanity_checks.py` into the project root**, run it:
   ```
   python sanity_checks.py > results/week1_sanity_report.txt
   cat results/week1_sanity_report.txt
   ```

8. **Read the report and discuss with team**. Specifically:
   - Is nutrition coverage ≥ 70%?
   - Are there ≥ 10K users with ≥ 5 ratings?
   - Are at least 4 dietary tags well-covered (≥ 100 recipes each)?
   - Are ingredients parsing cleanly?

9. **If any check fails, decide remediation**:
   - Filter aggressively (active users only, well-tagged recipes only)
   - Pivot scope if structural problems found
   - Document decisions in `docs/data_decisions.md`

---

## Data pipeline foundation (2-3 hours)

10. **Build `src/data/loader.py`** that exposes:
    ```python
    def load_recipes(path="data/raw") -> pd.DataFrame:
        # Returns recipes with parsed ingredients, nutrition, tags
        pass

    def load_interactions(path="data/raw") -> pd.DataFrame:
        # Returns user-recipe ratings with parsed dates
        pass

    def filter_active_users(interactions, min_ratings=5) -> pd.DataFrame:
        pass

    def time_based_split(interactions, holdout_per_user=1):
        # Returns (train, test) — most recent positive rating held out per user
        pass
    ```

11. **Build `src/data/pantry.py`** that exposes:
    ```python
    def derive_pantry_from_recipes(owned_recipe_ids, recipes_df) -> dict:
        # Returns {canonical_ingredient: count} aggregated across owned recipes
        pass

    def load_persona(persona_id) -> dict:
        # Returns {pantry, macro_targets, restrictions, taste_seeds}
        pass
    ```

12. **Build `src/data/ingredients.py`** that exposes:
    ```python
    def normalize_ingredient(raw_text) -> str:
        # Uses ingr_map.pkl to canonicalize
        pass

    def parse_nutrition(nutrition_str) -> dict:
        # Returns {calories, fat_pdv, sugar_pdv, sodium_pdv,
        #          protein_pdv, satfat_pdv, carbs_pdv}
        pass
    ```

13. **Write unit tests** in `tests/test_data.py` for each of the above.

---

## Sanity-check baseline (1-2 hours)

14. **Build `src/models/popularity.py`** — popularity baseline. Recommend
    the most-rated recipes overall, optionally filtered by constraints.

15. **Build `src/eval/metrics.py`** with:
    ```python
    def recall_at_k(recommended, relevant, k):
        pass

    def ndcg_at_k(recommended, relevant, k):
        pass

    def mrr(recommended, relevant):
        pass
    ```

16. **Run end-to-end on a sample user**: load data → train popularity model
    → predict for one user → compute Recall@10 against held-out item.

    If this works, the foundation is in place.

---

## Document personas (30 minutes)

17. **Define 5-8 personas as JSON files** in `data/personas/`. Each like:
    ```json
    {
      "id": "fitness_focused",
      "label": "Fitness-focused lifter",
      "description": "Tracks macros carefully, prioritizes protein and low carbs",
      "macro_targets": {
        "calories": 600,
        "protein_pdv": 50,
        "carbs_pdv": 30,
        "fat_pdv": 20
      },
      "restrictions": [],
      "pantry": ["chicken", "rice", "broccoli", "eggs", "olive oil", "..."],
      "taste_seeds": [
        "high-protein grilled chicken",
        "salmon with vegetables",
        "..."
      ]
    }
    ```

18. **Team members each export their own taste profile** as a similar JSON
    file — pick 20-30 Food.com recipes they'd realistically rate highly,
    define their own pantry, macros, restrictions.

---

## Stop and review (15 minutes)

19. **Push to git, write a brief progress note** in `docs/week1_progress.md`:
    - What was built
    - What sanity checks revealed
    - Any pivots or scope changes
    - What's next for Week 2

20. **Team sync meeting** — confirm everyone understands the data shape
    and the project's structural argument (Stage 1 learns, Stage 2 constrains).

---

## What NOT to do this week

- Don't start training MF/BPR/NCF yet. That's Week 2.
- Don't worry about the reranker yet. That's Week 3-4.
- Don't touch CV. That's Week 5 at earliest.
- Don't optimize anything. Get the pipeline working first.

---

## How to brief Claude Code at the start of the session

Paste this at the top of your first Claude Code conversation:

> I'm starting Project 2 for my recommender systems class. We've decided
> on a multi-constraint recipe recommender called PantryPlate, using
> the Food.com Kaggle dataset. I'm attaching the project README and
> a Week 1 task list. Please read the README first, then work through
> the tasks in order, pausing after each numbered block so I can verify.
> Start by reading `pantryplate/README.md` and `pantryplate/docs/week1_tasks.md`.

Then attach (or paste) the contents of:
- `README.md` (the project brief)
- `week1_tasks.md` (this file)
- `sanity_checks.py` (so it can be dropped straight in)
