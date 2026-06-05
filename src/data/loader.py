"""Recipe and interaction loaders for the Food.com Kaggle dataset.

Design choices baked in (from Week 1 sanity check findings + 2026-06-01 refactor):
- Nutrition values are clipped to remove misparsed portion outliers.
- 0-star interactions are dropped by default (likely "review without rating").
- Positive rating threshold is 4 stars or higher (locked decision in README).
- **Headline cohort uses the authors' published train/validation/test split**
  from Majumder et al. (2019). The pre-split is filtered to 25,076 active users
  with 1 held-out rating per user in val and test.
- The raw + custom-split path (`load_interactions` + `time_based_split`) is
  kept for raw-data exploration and ad-hoc analysis, but no longer the default
  for model training/evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .ingredients import parse_nutrition, safe_parse_list


POSITIVE_THRESHOLD = 4  # 4-star or higher counts as a positive interaction


def load_recipes(
    path: str | Path = "data/raw",
    clip_nutrition: bool = True,
) -> pd.DataFrame:
    """Load RAW_recipes.csv with parsed ingredients, nutrition, tags.

    Returns a DataFrame with the original columns plus:
        ingredients_parsed : list[str]
        tags_parsed        : list[str]
        steps_parsed       : list[str]
        nutrition_parsed   : dict or None (keys from NUTRITION_FIELDS)
        submitted          : datetime
    """
    path = Path(path)
    recipes = pd.read_csv(path / "RAW_recipes.csv")
    recipes["ingredients_parsed"] = recipes["ingredients"].apply(safe_parse_list)
    recipes["tags_parsed"] = recipes["tags"].apply(safe_parse_list)
    recipes["steps_parsed"] = recipes["steps"].apply(safe_parse_list)
    recipes["nutrition_parsed"] = recipes["nutrition"].apply(
        lambda s: parse_nutrition(s, clip=clip_nutrition)
    )
    recipes["submitted"] = pd.to_datetime(recipes["submitted"], errors="coerce")
    return recipes


def load_interactions(
    path: str | Path = "data/raw",
    drop_zero_stars: bool = True,
) -> pd.DataFrame:
    """Load RAW_interactions.csv with parsed dates.

    Sanity check found 60,847 zero-star interactions (5.4%) — these almost
    certainly represent "review left, no rating" rather than literal zero
    scores, and they break a 4-star positive threshold. Drop them by default.
    """
    path = Path(path)
    interactions = pd.read_csv(path / "RAW_interactions.csv")
    interactions["date"] = pd.to_datetime(interactions["date"], errors="coerce")
    if drop_zero_stars:
        interactions = interactions[interactions["rating"] > 0].reset_index(drop=True)
    return interactions


def filter_active_users(
    interactions: pd.DataFrame,
    min_ratings: int = 5,
) -> pd.DataFrame:
    """Keep only users with at least `min_ratings` total interactions."""
    counts = interactions.groupby("user_id").size()
    active = counts[counts >= min_ratings].index
    return interactions[interactions["user_id"].isin(active)].reset_index(drop=True)


def load_train_interactions(
    path: str | Path = "data/raw",
    drop_zero_stars: bool = True,
) -> pd.DataFrame:
    """Load the authors' pre-split training interactions.

    This is the headline-evaluation training set: pre-filtered to active
    users (~25K) with held-out positives in `interactions_validation.csv`
    and `interactions_test.csv`.

    Columns: user_id, recipe_id, date, rating, u, i
        - user_id/recipe_id are original Food.com IDs (use these for joining
          with recipes_df).
        - u/i are the authors' remapped 0-indexed integer IDs (useful for
          matrix-factorization model implementations).

    Drops 0-star ratings by default (these are "review without rating" entries).
    """
    path = Path(path)
    df = pd.read_csv(path / "interactions_train.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if drop_zero_stars:
        df = df[df["rating"] > 0].reset_index(drop=True)
    return df


def load_validation_interactions(
    path: str | Path = "data/raw",
    positives_only: bool = True,
    positive_threshold: int = POSITIVE_THRESHOLD,
) -> pd.DataFrame:
    """Load the authors' pre-split validation set (1 held-out per user).

    The pre-split val set contains ratings of all values (0-5). By default we
    filter to positives (>= POSITIVE_THRESHOLD) so it works as held-out
    positives for Recall@K-style evaluation. Pass `positives_only=False` to
    get the raw val set.
    """
    path = Path(path)
    df = pd.read_csv(path / "interactions_validation.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if positives_only:
        df = df[df["rating"] >= positive_threshold].reset_index(drop=True)
    return df


def load_test_interactions(
    path: str | Path = "data/raw",
    positives_only: bool = True,
    positive_threshold: int = POSITIVE_THRESHOLD,
) -> pd.DataFrame:
    """Load the authors' pre-split test set (1 held-out per user).

    Same conventions as `load_validation_interactions`. Use this for headline
    evaluation; reserve validation for hyperparameter tuning.
    """
    path = Path(path)
    df = pd.read_csv(path / "interactions_test.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if positives_only:
        df = df[df["rating"] >= positive_threshold].reset_index(drop=True)
    return df


def load_prebuilt_split(
    path: str | Path = "data/raw",
    drop_zero_stars: bool = True,
    positives_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience: load (train, validation, test) in one call.

    `drop_zero_stars` applies to train; `positives_only` applies to val/test.
    """
    return (
        load_train_interactions(path, drop_zero_stars=drop_zero_stars),
        load_validation_interactions(path, positives_only=positives_only),
        load_test_interactions(path, positives_only=positives_only),
    )


def time_based_split(
    interactions: pd.DataFrame,
    holdout_per_user: int = 1,
    positive_threshold: int = POSITIVE_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user leave-one-out on the most recent positive interaction.

    For each user with at least one positive (rating >= threshold) AND at
    least one earlier interaction, hold out the `holdout_per_user` most
    recent positives. All other interactions go to train.

    Users with no positives, or whose only positive is also their only
    interaction, are kept entirely in train (no holdout).
    """
    df = interactions.sort_values(["user_id", "date"]).reset_index(drop=True)
    is_positive = df["rating"] >= positive_threshold

    # Rank positives per user in reverse chronological order (1 = most recent)
    pos_rank = (
        df[is_positive]
        .groupby("user_id")["date"]
        .rank(method="first", ascending=False)
    )
    holdout_mask = pd.Series(False, index=df.index)
    holdout_mask.loc[pos_rank.index] = pos_rank <= holdout_per_user

    # Only hold out users who'd still have training data after the holdout
    user_total = df.groupby("user_id").size()
    eligible_users = user_total[user_total > holdout_per_user].index
    holdout_mask &= df["user_id"].isin(eligible_users)

    test = df[holdout_mask].reset_index(drop=True)
    train = df[~holdout_mask].reset_index(drop=True)
    return train, test
