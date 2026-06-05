"""Unit tests for src/data/ modules."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.ingredients import (
    NUTRITION_FIELDS,
    parse_nutrition,
    safe_parse_list,
    normalize_ingredient,
)
from src.data.loader import (
    filter_active_users,
    time_based_split,
    load_train_interactions,
    load_validation_interactions,
    load_test_interactions,
    load_prebuilt_split,
    POSITIVE_THRESHOLD,
)
from src.data.pantry import derive_pantry_from_recipes, load_persona


INGR_MAP = Path("data/raw/ingr_map.pkl")
TRAIN_CSV = Path("data/raw/interactions_train.csv")


# ---------- ingredients.py ----------


class TestParseNutrition:
    def test_valid_seven_element_list(self):
        result = parse_nutrition("[300.0, 15.0, 5.0, 20.0, 25.0, 8.0, 12.0]")
        assert result is not None
        assert set(result.keys()) == set(NUTRITION_FIELDS)
        assert result["calories"] == 300.0
        assert result["protein_pdv"] == 25.0

    def test_clipping_caps_outlier_calories(self):
        # 50,000 kcal should clip to default cap (5000)
        result = parse_nutrition("[50000, 10, 10, 10, 10, 10, 10]", clip=True)
        assert result["calories"] == 5000.0

    def test_clipping_caps_outlier_pdv(self):
        result = parse_nutrition("[300, 9999, 10, 10, 10, 10, 10]", clip=True)
        assert result["fat_pdv"] == 1000.0

    def test_no_clip_preserves_outliers(self):
        result = parse_nutrition("[50000, 10, 10, 10, 10, 10, 10]", clip=False)
        assert result["calories"] == 50000.0

    def test_wrong_length_returns_none(self):
        assert parse_nutrition("[1, 2, 3]") is None

    def test_malformed_returns_none(self):
        assert parse_nutrition("not a list") is None
        assert parse_nutrition(None) is None
        assert parse_nutrition(float("nan")) is None


class TestSafeParseList:
    def test_valid_list(self):
        assert safe_parse_list("['a', 'b', 'c']") == ["a", "b", "c"]

    def test_empty_list(self):
        assert safe_parse_list("[]") == []

    def test_malformed_returns_empty(self):
        assert safe_parse_list("not a list") == []
        assert safe_parse_list(None) == []


@pytest.mark.skipif(not INGR_MAP.exists(), reason="ingr_map.pkl not downloaded")
class TestNormalizeIngredient:
    def test_unknown_falls_back_to_lowercase_strip(self):
        result = normalize_ingredient("  Made-up Ingredient XYZ  ")
        assert result == "made-up ingredient xyz"

    def test_empty_string(self):
        assert normalize_ingredient("") == ""

    def test_non_string(self):
        assert normalize_ingredient(None) == ""


# ---------- loader.py ----------


@pytest.fixture
def sample_interactions():
    """A small DataFrame covering edge cases for split/filter logic."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4],
            "recipe_id": [10, 11, 12, 13, 20, 21, 30, 31, 32, 33, 34, 40],
            "rating":   [5, 4, 3, 5, 4, 5, 5, 5, 5, 2, 4, 5],
            "date": pd.to_datetime([
                "2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01",  # user 1
                "2020-05-01", "2020-06-01",                              # user 2
                "2020-01-01", "2020-02-01", "2020-03-01",
                "2020-04-01", "2020-05-01",                              # user 3
                "2020-07-01",                                            # user 4
            ]),
        }
    )


class TestFilterActiveUsers:
    def test_default_threshold_drops_low_count_users(self, sample_interactions):
        result = filter_active_users(sample_interactions, min_ratings=5)
        # users 1=4, 2=2, 3=5, 4=1 → only user 3 keeps
        assert set(result["user_id"].unique()) == {3}

    def test_lower_threshold_keeps_more(self, sample_interactions):
        result = filter_active_users(sample_interactions, min_ratings=3)
        assert set(result["user_id"].unique()) == {1, 3}


class TestTimeBasedSplit:
    def test_holds_out_most_recent_positive_per_user(self, sample_interactions):
        train, test = time_based_split(sample_interactions, holdout_per_user=1)
        # User 1's most recent positive (4+) is recipe 13 on 2020-04-01
        user1_test = test[test["user_id"] == 1]
        assert len(user1_test) == 1
        assert user1_test["recipe_id"].iloc[0] == 13

    def test_skips_users_with_no_positives(self, sample_interactions):
        # Construct a user whose only ratings are below threshold
        df = pd.DataFrame({
            "user_id": [99, 99],
            "recipe_id": [1, 2],
            "rating": [1, 2],
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        })
        train, test = time_based_split(df)
        assert len(test) == 0
        assert len(train) == 2

    def test_skips_single_interaction_users(self, sample_interactions):
        # User 4 has only one rating → no train left if we hold it out
        train, test = time_based_split(sample_interactions, holdout_per_user=1)
        assert 4 not in test["user_id"].values
        assert 4 in train["user_id"].values

    def test_train_and_test_partition_input(self, sample_interactions):
        train, test = time_based_split(sample_interactions, holdout_per_user=1)
        assert len(train) + len(test) == len(sample_interactions)

    def test_user3_holdout_picks_most_recent_positive_not_negative(
        self, sample_interactions
    ):
        # User 3 has positives on 2020-01..03 and 05, negative (2) on 04
        # Most recent positive is 2020-05-01 (recipe 34)
        train, test = time_based_split(sample_interactions, holdout_per_user=1)
        user3_test = test[test["user_id"] == 3]
        assert len(user3_test) == 1
        assert user3_test["recipe_id"].iloc[0] == 34


# ---------- pantry.py ----------


@pytest.fixture
def sample_recipes():
    return pd.DataFrame(
        {
            "id": [100, 101, 102],
            "name": ["a", "b", "c"],
            "ingredients_parsed": [
                ["chicken", "rice", "salt"],
                ["chicken", "broccoli", "salt"],
                ["beef", "rice", "salt"],
            ],
        }
    )


@pytest.mark.skipif(not INGR_MAP.exists(), reason="ingr_map.pkl not downloaded")
class TestDerivePantry:
    def test_aggregates_across_owned_recipes(self, sample_recipes):
        # User owns recipes 100 and 101
        pantry = derive_pantry_from_recipes([100, 101], sample_recipes)
        # chicken in both → 2; salt in both → 2; rice and broccoli → 1 each
        assert pantry["chicken"] == 2
        assert pantry["salt"] == 2
        assert pantry["rice"] == 1
        assert pantry["broccoli"] == 1
        assert "beef" not in pantry

    def test_unknown_recipe_id_is_ignored(self, sample_recipes):
        pantry = derive_pantry_from_recipes([999], sample_recipes)
        assert pantry == {}

    def test_missing_ingredients_column_raises(self):
        df = pd.DataFrame({"id": [1], "name": ["x"]})
        with pytest.raises(ValueError, match="ingredients_parsed"):
            derive_pantry_from_recipes([1], df)


@pytest.mark.skipif(not TRAIN_CSV.exists(), reason="pre-split files not downloaded")
class TestPreSplitLoaders:
    """Integration tests for the authors' pre-split loaders.

    These read the actual CSV files so they're skipped if data not present.
    """

    def test_train_has_expected_columns(self):
        df = load_train_interactions()
        assert {"user_id", "recipe_id", "date", "rating", "u", "i"}.issubset(df.columns)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_train_drops_zero_stars_by_default(self):
        df = load_train_interactions()
        assert (df["rating"] == 0).sum() == 0
        assert (df["rating"] > 0).all()

    def test_train_can_keep_zero_stars(self):
        df = load_train_interactions(drop_zero_stars=False)
        # The raw pre-split train has 0-star ratings (16,957 of them per EDA)
        assert (df["rating"] == 0).sum() > 0

    def test_validation_positives_only_default(self):
        df = load_validation_interactions()
        assert (df["rating"] >= POSITIVE_THRESHOLD).all()

    def test_validation_can_keep_negatives(self):
        df = load_validation_interactions(positives_only=False)
        # Raw val has 0-3 star ratings (1,123 of them per data inspection)
        assert (df["rating"] < POSITIVE_THRESHOLD).sum() > 0

    def test_test_positives_only_default(self):
        df = load_test_interactions()
        assert (df["rating"] >= POSITIVE_THRESHOLD).all()

    def test_one_row_per_user_in_test(self):
        df = load_test_interactions()
        # The authors' split holds out 1 per user; positives_only may filter some out
        # but never duplicate
        assert df["user_id"].nunique() == len(df)

    def test_prebuilt_split_returns_three_frames(self):
        train, val, test = load_prebuilt_split()
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        # Train should be much bigger than val + test
        assert len(train) > len(val) + len(test)


class TestLoadPersona:
    def test_loads_json(self, tmp_path):
        persona = {
            "id": "test_persona",
            "label": "Test",
            "macro_targets": {"calories": 500},
            "restrictions": [],
            "pantry": ["egg"],
            "taste_seeds": [],
        }
        (tmp_path / "test_persona.json").write_text(json.dumps(persona))
        result = load_persona("test_persona", personas_dir=tmp_path)
        assert result["id"] == "test_persona"
        assert result["macro_targets"]["calories"] == 500

    def test_missing_persona_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_persona("nonexistent", personas_dir=tmp_path)
