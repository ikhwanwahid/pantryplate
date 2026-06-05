"""Recipe and user feature engineering for content-aware models.

Builds compact, normalized feature representations of recipes (and users)
suitable for use as item features in Stage 1 content / hybrid / two-tower
models.

Pipelines:

  - Tag features: meta-tag filter → frequency filter → variance filter
    → MultiLabelBinarizer → L2 normalize → TruncatedSVD (100-dim)
  - Nutrition features: 99th-percentile clip → RobustScaler (7-dim)
  - Combined recipe feature matrix: 107-dim per recipe
  - User activity tier classification: low / medium / high

Outputs are cached to data/processed/ to avoid recomputation across
model lanes. Fitted models (SVD, scaler, MLB) are saved as .pkl alongside.

Originally developed by Anastasia Frances Frederica in
notebooks/anastasia_kaggle.ipynb; ported here for reuse across Stage 1
content/hybrid/two-tower model implementations.

Usage:

    from src.data.features import build_recipe_feature_matrix
    features = build_recipe_feature_matrix()
    # features.shape == (231637, 107) — 100 tag-SVD + 7 nutrition

    # Or build individual pieces:
    from src.data.features import build_tag_features, build_nutrition_features
    tag_emb, tag_models = build_tag_features(recipes_df)
    nut_emb, nut_scaler = build_nutrition_features(recipes_df)
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer, RobustScaler, normalize


# ============================================================
# Constants — tunable but stable across the project
# ============================================================

# Tag pipeline
TAG_MIN_FREQUENCY = 100          # drop tags appearing in fewer recipes than this
TAG_VARIANCE_THRESHOLD = 0.5     # discriminative tags: variance >= this across user tiers
TAG_CONTENT_MIN_PCT = 1.0        # content tags: in >= this % of recipes
TAG_SVD_COMPONENTS = 100         # final tag embedding dimension

# Meta tags (category headers; not useful as content signal)
META_TAGS: frozenset[str] = frozenset({
    "preparation", "time-to-make", "course", "main-ingredient",
    "dietary", "occasion", "cuisine", "equipment", "taste-mood",
    "number-of-servings", "low-in-something", "meat",
})

# Nutrition pipeline
NUTRITION_COLS: tuple[str, ...] = (
    "calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
    "protein_pdv", "sat_fat_pdv", "carbs_pdv",
)
NUTRITION_CLIP_PERCENTILE = 0.99

# User activity tier thresholds (rating counts)
USER_TIER_LOW_MAX = 5      # < this = "low-activity" (excluded from our active cohort)
USER_TIER_HIGH_MIN = 20    # >= this = "high-activity"; in between = "medium-activity"

# Cache paths
CACHE_DIR = Path("data/processed")
FEATURES_CACHE = CACHE_DIR / "recipe_features.parquet"
TAG_SVD_CACHE = CACHE_DIR / "tag_svd_model.pkl"
NUTRITION_SCALER_CACHE = CACHE_DIR / "nutrition_scaler.pkl"
TAG_MLB_CACHE = CACHE_DIR / "tag_mlb.pkl"


# ============================================================
# Helpers
# ============================================================

def _parse_list_string(s) -> list:
    """Safely parse a stringified list (used for tags column)."""
    if not isinstance(s, str):
        return []
    try:
        result = ast.literal_eval(s)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


# ============================================================
# Tag features
# ============================================================

def select_useful_tags(
    recipes_df: pd.DataFrame,
    interactions_df: Optional[pd.DataFrame] = None,
    min_frequency: int = TAG_MIN_FREQUENCY,
    variance_threshold: float = TAG_VARIANCE_THRESHOLD,
    content_min_pct: float = TAG_CONTENT_MIN_PCT,
    meta_tags: Iterable[str] = META_TAGS,
) -> list[str]:
    """Select tags worth using as features, applying a three-tier filter.

    1. Drop meta tags (category headers like 'preparation', 'cuisine')
    2. Keep tags appearing >= min_frequency times across recipes
    3. Either: (a) variance >= variance_threshold across user activity tiers
              OR (b) appears in >= content_min_pct% of recipes

    If `interactions_df` is None, only filters 1 and 2 are applied (no
    variance-based selection).

    Returns: list of tag names to use as features.
    """
    meta_tags = set(meta_tags)

    # Get parsed tags (compute if not present)
    if "tags_parsed" in recipes_df.columns:
        tags_parsed = recipes_df["tags_parsed"]
    else:
        tags_parsed = recipes_df["tags"].apply(_parse_list_string)

    # Step 1+2: meta filter + frequency filter
    all_tags = [t for tag_list in tags_parsed for t in tag_list]
    tag_counts = Counter(all_tags)
    meaningful = {
        tag: count for tag, count in tag_counts.items()
        if tag not in meta_tags and count >= min_frequency
    }

    if not meaningful:
        return []

    # If no interactions, return all meaningful tags
    if interactions_df is None:
        return list(meaningful.keys())

    # Step 3a: variance across user activity tiers
    user_stats = classify_user_activity(interactions_df)
    user_tier_map = user_stats.set_index("user_id")["activity_tier"]

    inter_with_tiers = interactions_df.copy()
    inter_with_tiers["user_tier"] = inter_with_tiers["user_id"].map(user_tier_map)

    # Join with tags
    recipes_for_join = recipes_df[["id"]].copy()
    recipes_for_join["tags_parsed"] = tags_parsed.values

    inter_with_tags = inter_with_tiers.merge(
        recipes_for_join, left_on="recipe_id", right_on="id", how="left"
    ).explode("tags_parsed")
    inter_with_tags = inter_with_tags[
        inter_with_tags["tags_parsed"].isin(meaningful)
    ]

    # Compute tag rate per tier
    tier_rates = {}
    for tier_name in ("low", "medium", "high"):
        subset = inter_with_tags[inter_with_tags["user_tier"] == tier_name]
        n_interactions = (inter_with_tiers["user_tier"] == tier_name).sum()
        if n_interactions == 0:
            tier_rates[tier_name] = pd.Series(0.0, index=list(meaningful.keys()))
            continue
        tier_rates[tier_name] = (
            subset["tags_parsed"].value_counts() / n_interactions * 100
        ).reindex(meaningful.keys()).fillna(0.0)

    tier_rates_df = pd.DataFrame(tier_rates)
    tier_rates_df["variance"] = tier_rates_df.var(axis=1)
    discriminative = tier_rates_df[
        tier_rates_df["variance"] >= variance_threshold
    ].index.tolist()

    # Step 3b: content tags (high recipe frequency)
    tag_recipe_pct = (
        pd.Series(meaningful) / len(recipes_df) * 100
    )
    content = tag_recipe_pct[tag_recipe_pct >= content_min_pct].index.tolist()

    return list(set(discriminative + content))


def build_tag_features(
    recipes_df: pd.DataFrame,
    interactions_df: Optional[pd.DataFrame] = None,
    n_components: int = TAG_SVD_COMPONENTS,
    selected_tags: Optional[list[str]] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Build tag SVD embeddings for recipes.

    Returns:
        (tag_embedding_df, fitted_models)
        - tag_embedding_df: DataFrame indexed by recipe_id, columns are
          tag_svd_0 ... tag_svd_{n_components-1}
        - fitted_models: dict with 'mlb' (MultiLabelBinarizer), 'svd'
          (TruncatedSVD), 'selected_tags' (list[str]) for reuse on new recipes
    """
    if "tags_parsed" in recipes_df.columns:
        tags_parsed = recipes_df["tags_parsed"]
    else:
        tags_parsed = recipes_df["tags"].apply(_parse_list_string)

    if selected_tags is None:
        selected_tags = select_useful_tags(recipes_df, interactions_df)

    selected_set = set(selected_tags)
    mlb = MultiLabelBinarizer(classes=selected_tags)
    tag_matrix = mlb.fit_transform(
        tags_parsed.apply(lambda tags: [t for t in tags if t in selected_set])
    )

    tag_matrix_norm = normalize(tag_matrix, norm="l2")
    # Cap n_components at the matrix rank
    n_components_actual = min(n_components, tag_matrix_norm.shape[1])
    svd = TruncatedSVD(n_components=n_components_actual, random_state=seed)
    embeddings = svd.fit_transform(tag_matrix_norm)

    embedding_df = pd.DataFrame(
        embeddings,
        index=pd.Index(recipes_df["id"].values, name="recipe_id"),
        columns=[f"tag_svd_{i}" for i in range(n_components_actual)],
    )

    return embedding_df, {
        "mlb": mlb,
        "svd": svd,
        "selected_tags": selected_tags,
    }


# ============================================================
# Nutrition features
# ============================================================

def _parse_nutrition_to_columns(recipes_df: pd.DataFrame) -> pd.DataFrame:
    """Parse the 'nutrition' string column into 7 numeric columns."""
    def _parse(s):
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) and len(v) == 7 else [np.nan] * 7
        except (ValueError, SyntaxError, TypeError):
            return [np.nan] * 7

    parsed = recipes_df["nutrition"].apply(_parse)
    return pd.DataFrame(
        parsed.tolist(),
        columns=list(NUTRITION_COLS),
        index=recipes_df.index,
    )


def build_nutrition_features(
    recipes_df: pd.DataFrame,
    clip_percentile: float = NUTRITION_CLIP_PERCENTILE,
) -> tuple[pd.DataFrame, RobustScaler]:
    """Build normalized nutrition feature vectors for recipes.

    Clips each nutrition column at the given percentile (default 99th),
    then applies RobustScaler. This is for ML model TRAINING — for the
    Stage 2 reranker's `s_nutrition`, use raw PDV values via
    src.data.ingredients.parse_nutrition() instead.

    Returns:
        (nutrition_embedding_df, fitted_scaler)
    """
    if all(c in recipes_df.columns for c in NUTRITION_COLS):
        nut_cols = recipes_df[list(NUTRITION_COLS)].copy()
    else:
        nut_cols = _parse_nutrition_to_columns(recipes_df)

    upper = nut_cols.quantile(clip_percentile)
    nut_clipped = nut_cols.clip(upper=upper, axis=1)

    scaler = RobustScaler()
    nut_scaled = scaler.fit_transform(nut_clipped.fillna(0))

    embedding_df = pd.DataFrame(
        nut_scaled,
        index=pd.Index(recipes_df["id"].values, name="recipe_id"),
        columns=[f"nutrition_{c}" for c in NUTRITION_COLS],
    )

    return embedding_df, scaler


# ============================================================
# Combined recipe feature matrix
# ============================================================

def build_recipe_feature_matrix(
    recipes_df: Optional[pd.DataFrame] = None,
    interactions_df: Optional[pd.DataFrame] = None,
    cache_path: str | Path = FEATURES_CACHE,
    force_rebuild: bool = False,
    save_models: bool = True,
) -> pd.DataFrame:
    """Build (or load cached) recipe feature matrix.

    Combines tag SVD (100-dim) + normalized nutrition (7-dim) = 107-dim
    per recipe. Cached to data/processed/recipe_features.parquet.

    First call builds the matrix (a few minutes). Subsequent calls load
    from cache instantly. To force rebuild after changing constants:
    pass `force_rebuild=True`.

    Returns:
        DataFrame indexed by recipe_id (Food.com recipe id) with 107 columns:
        tag_svd_0 ... tag_svd_99, nutrition_calories ... nutrition_carbs_pdv
    """
    cache_path = Path(cache_path)

    if cache_path.exists() and not force_rebuild:
        return pd.read_parquet(cache_path)

    if recipes_df is None:
        from src.data.loader import load_recipes
        recipes_df = load_recipes()

    if interactions_df is None:
        from src.data.loader import load_train_interactions
        interactions_df = load_train_interactions()

    tag_features, tag_models = build_tag_features(recipes_df, interactions_df)
    nutrition_features, nutrition_scaler = build_nutrition_features(recipes_df)

    combined = tag_features.join(nutrition_features, how="inner")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path)

    if save_models:
        joblib.dump(tag_models["svd"], TAG_SVD_CACHE)
        joblib.dump(tag_models["mlb"], TAG_MLB_CACHE)
        joblib.dump(nutrition_scaler, NUTRITION_SCALER_CACHE)

    return combined


# ============================================================
# User activity tier classification
# ============================================================

def classify_user_activity(
    interactions_df: pd.DataFrame,
    low_max: int = USER_TIER_LOW_MAX,
    high_min: int = USER_TIER_HIGH_MIN,
) -> pd.DataFrame:
    """Classify users into activity tiers by rating count.

    Tier thresholds (configurable):
        low    : rating_count < low_max         (default < 5)
        medium : low_max <= rating_count < high_min  (default 5-19)
        high   : rating_count >= high_min       (default >= 20)

    Note: contrary to a common misconception, the authors' train file is
    NOT pre-filtered to >=5 ratings per user. It actually contains a wide
    activity spread. Empirically (as of the current train file):
      - low    (<5 ratings):  ~10,300 users (~41%)
      - medium (5-19):        ~9,500 users  (~38%)
      - high   (>=20):        ~5,100 users  (~21%)
    All three tiers are populated within our cohort and worth stratifying on.

    Returns:
        DataFrame with columns: user_id, rating_count, mean_rating,
        std_rating, activity_tier
    """
    stats = (
        interactions_df.groupby("user_id")["rating"]
        .agg(rating_count="count", mean_rating="mean", std_rating="std")
        .reset_index()
    )

    def _tier(c: int) -> str:
        if c < low_max:
            return "low"
        elif c < high_min:
            return "medium"
        return "high"

    stats["activity_tier"] = stats["rating_count"].apply(_tier)
    return stats


# ============================================================
# CLI entry point — build and cache
# ============================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Building recipe feature matrix")
    print("=" * 60)
    t0 = time.time()

    features = build_recipe_feature_matrix(force_rebuild=True)

    print(f"\n✓ Built feature matrix in {time.time()-t0:.1f}s")
    print(f"  Shape: {features.shape}")
    print(f"  Cached to: {FEATURES_CACHE}")
    print(f"  Models cached to: {CACHE_DIR}/{{tag_svd,nutrition_scaler,tag_mlb}}.pkl")
    print(f"\n  First 5 columns of first 3 rows:")
    print(features.iloc[:3, :5].round(3))
