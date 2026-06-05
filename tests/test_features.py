"""Tests for src/data/features.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    META_TAGS,
    NUTRITION_COLS,
    TAG_SVD_COMPONENTS,
    build_nutrition_features,
    build_recipe_feature_matrix,
    build_tag_features,
    classify_user_activity,
    select_useful_tags,
)


RECIPES_CSV = Path("data/raw/RAW_recipes.csv")
TRAIN_CSV = Path("data/raw/interactions_train.csv")


# ============================================================
# select_useful_tags
# ============================================================

class TestSelectUsefulTags:
    def test_drops_meta_tags(self):
        """Meta tags (category headers) should not appear in the output."""
        recipes = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "tags": [
                "['preparation', 'vegetarian']",
                "['course', 'vegan']",
                "['cuisine', 'gluten-free']",
                "['equipment', 'vegetarian']",
                "['main-ingredient', 'vegan']",
            ],
        })
        # Use frequency=1 to keep things in, just test meta filter
        result = select_useful_tags(recipes, min_frequency=1)
        for meta in META_TAGS:
            assert meta not in result, f"meta tag {meta!r} leaked"

    def test_frequency_filter_drops_rare_tags(self):
        recipes = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "tags": [
                "['vegetarian', 'one-off-tag-A']",
                "['vegetarian', 'one-off-tag-B']",
                "['vegetarian']",
                "['vegetarian']",
            ],
        })
        result = select_useful_tags(recipes, min_frequency=2)
        assert "vegetarian" in result
        assert "one-off-tag-A" not in result
        assert "one-off-tag-B" not in result

    def test_empty_recipes_returns_empty(self):
        recipes = pd.DataFrame({"id": [], "tags": []})
        result = select_useful_tags(recipes, min_frequency=1)
        assert result == []


# ============================================================
# build_tag_features
# ============================================================

class TestBuildTagFeatures:
    def test_returns_correct_shape(self):
        recipes = pd.DataFrame({
            "id": list(range(10)),
            "tags": ["['vegetarian', 'vegan']"] * 10,
        })
        features, models = build_tag_features(
            recipes, selected_tags=["vegetarian", "vegan"], n_components=2
        )
        assert features.shape == (10, 2)
        assert features.index.name == "recipe_id"
        assert list(features.columns) == ["tag_svd_0", "tag_svd_1"]

    def test_returns_fitted_models(self):
        recipes = pd.DataFrame({
            "id": list(range(10)),
            "tags": ["['vegetarian', 'vegan']"] * 10,
        })
        _, models = build_tag_features(
            recipes, selected_tags=["vegetarian", "vegan"], n_components=2
        )
        assert {"mlb", "svd", "selected_tags"}.issubset(models.keys())

    def test_n_components_capped_at_n_tags(self):
        """If n_components > number of selected tags, it caps to n_tags."""
        recipes = pd.DataFrame({
            "id": list(range(20)),
            "tags": ["['a', 'b', 'c']"] * 20,
        })
        features, _ = build_tag_features(
            recipes, selected_tags=["a", "b", "c"], n_components=100
        )
        # Should be capped at 3 (the number of tags)
        assert features.shape[1] == 3


# ============================================================
# build_nutrition_features
# ============================================================

class TestBuildNutritionFeatures:
    def test_returns_correct_shape(self):
        recipes = pd.DataFrame({
            "id": list(range(5)),
            "nutrition": ["[300.0, 15.0, 5.0, 20.0, 25.0, 8.0, 12.0]"] * 5,
        })
        features, scaler = build_nutrition_features(recipes)
        assert features.shape == (5, 7)
        assert features.index.name == "recipe_id"
        for col in NUTRITION_COLS:
            assert f"nutrition_{col}" in features.columns

    def test_handles_outlier_clipping(self):
        """Values above 99th percentile should be clipped before scaling."""
        nutrition_values = [
            "[300.0, 15.0, 5.0, 20.0, 25.0, 8.0, 12.0]"
        ] * 99 + ["[1000000.0, 99999.0, 99999.0, 99999.0, 99999.0, 99999.0, 99999.0]"]
        recipes = pd.DataFrame({
            "id": list(range(100)),
            "nutrition": nutrition_values,
        })
        features, _ = build_nutrition_features(recipes)
        # After clipping at 99th percentile + RobustScaler, no value should be wildly extreme
        assert features.abs().max().max() < 1e5

    def test_handles_malformed_nutrition(self):
        recipes = pd.DataFrame({
            "id": [1, 2, 3],
            "nutrition": ["[300, 15, 5, 20, 25, 8, 12]", "not a list", None],
        })
        # Should not crash
        features, _ = build_nutrition_features(recipes)
        assert features.shape == (3, 7)


# ============================================================
# classify_user_activity
# ============================================================

class TestClassifyUserActivity:
    def test_three_tiers(self):
        # user 1: low (<5), user 2: medium (5-19), user 3: high (>=20)
        interactions = pd.DataFrame({
            "user_id": [1] * 3 + [2] * 10 + [3] * 30,
            "rating": [5.0] * 43,
            "recipe_id": list(range(43)),
        })
        stats = classify_user_activity(interactions)
        tier_by_user = stats.set_index("user_id")["activity_tier"].to_dict()
        assert tier_by_user[1] == "low"
        assert tier_by_user[2] == "medium"
        assert tier_by_user[3] == "high"

    def test_returns_rating_stats(self):
        interactions = pd.DataFrame({
            "user_id": [1] * 5,
            "rating": [3.0, 4.0, 5.0, 4.0, 5.0],
            "recipe_id": list(range(5)),
        })
        stats = classify_user_activity(interactions)
        row = stats.iloc[0]
        assert row["rating_count"] == 5
        assert row["mean_rating"] == pytest.approx(4.2)
        assert "std_rating" in stats.columns

    def test_custom_thresholds(self):
        interactions = pd.DataFrame({
            "user_id": [1] * 8,
            "rating": [5.0] * 8,
            "recipe_id": list(range(8)),
        })
        stats = classify_user_activity(interactions, low_max=10, high_min=20)
        # 8 ratings, low_max=10 → tier is "low"
        assert stats.iloc[0]["activity_tier"] == "low"


# ============================================================
# Integration tests (skip if data not downloaded)
# ============================================================

@pytest.mark.skipif(
    not RECIPES_CSV.exists() or not TRAIN_CSV.exists(),
    reason="raw data files not present",
)
class TestIntegrationOnRealData:
    def test_build_recipe_feature_matrix_runs_on_sample(self):
        """End-to-end: tiny sample through the full pipeline."""
        from src.data.loader import load_recipes, load_train_interactions

        recipes = load_recipes().sample(500, random_state=42)
        # Use a small subset of interactions matching these recipes
        interactions = load_train_interactions().head(2000)

        # Force a small SVD so we can run with tiny tag set
        tag_features, _ = build_tag_features(
            recipes, interactions, n_components=10
        )
        nut_features, _ = build_nutrition_features(recipes)

        assert tag_features.shape[0] == 500
        assert nut_features.shape[0] == 500
        assert nut_features.shape[1] == 7

    def test_classify_user_activity_on_real_train(self):
        """Verify the three-tier classification on real train data."""
        from src.data.loader import load_train_interactions

        interactions = load_train_interactions()
        stats = classify_user_activity(interactions)

        # Sanity: tiers should sum to total user count
        tier_counts = stats["activity_tier"].value_counts()
        assert tier_counts.sum() == stats["user_id"].nunique()

        # All three tiers should be populated in the authors' train file
        # (the train file is NOT pre-filtered to >=5 ratings as previously
        # assumed — it has a wide activity spread).
        assert tier_counts.get("low", 0) > 0
        assert tier_counts["medium"] > 1000
        assert tier_counts["high"] > 1000
