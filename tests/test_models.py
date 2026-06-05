"""Unit tests for src/models/."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.popularity import PopularityRecommender


@pytest.fixture
def small_train():
    # Recipe 10 rated by 3 users, 20 by 2, 30 by 1
    return pd.DataFrame({
        "user_id":   [1, 2, 3, 1, 2, 1],
        "recipe_id": [10, 10, 10, 20, 20, 30],
        "rating":    [5, 4, 5, 4, 5, 3],
    })


class TestPopularity:
    def test_ranks_by_unique_users(self, small_train):
        model = PopularityRecommender().fit(small_train)
        # Recipe 10 (3 users) > 20 (2 users) > 30 (1 user)
        recs = model.recommend(user_id=999, k=10, exclude_seen=False)
        assert recs[:3] == [10, 20, 30]

    def test_count_total_ratings_mode(self):
        df = pd.DataFrame({
            "user_id":   [1, 1, 1, 2],
            "recipe_id": [10, 10, 10, 20],
            "rating":    [5, 4, 5, 5],
        })
        # By unique users: 10=1, 20=1 (tied)
        # By total ratings: 10=3, 20=1
        m_unique = PopularityRecommender(count_unique_users=True).fit(df)
        m_total = PopularityRecommender(count_unique_users=False).fit(df)
        assert m_total.recommend(999, k=2, exclude_seen=False)[0] == 10
        # In the tied case, unique-user mode still puts 10 first due to stable sort
        recs_unique = m_unique.recommend(999, k=2, exclude_seen=False)
        assert set(recs_unique) == {10, 20}

    def test_excludes_seen_items(self, small_train):
        model = PopularityRecommender().fit(small_train)
        # User 1 has rated 10, 20, 30 — all three. So no recommendations possible.
        recs = model.recommend(user_id=1, k=10, exclude_seen=True)
        assert recs == []

    def test_unseen_user_gets_full_top_k(self, small_train):
        model = PopularityRecommender().fit(small_train)
        recs = model.recommend(user_id=99999, k=2, exclude_seen=True)
        assert recs == [10, 20]

    def test_recommend_many(self, small_train):
        model = PopularityRecommender().fit(small_train)
        out = model.recommend_many([1, 2, 999], k=3)
        assert set(out.keys()) == {1, 2, 999}
        assert out[999] == [10, 20, 30]
        # User 2 has rated 10 and 20 → top-2 unseen are 30 then nothing
        assert out[2] == [30]

    def test_recommend_before_fit_errors(self):
        with pytest.raises(RuntimeError, match="fit"):
            PopularityRecommender().recommend(1, k=5)

    def test_fit_missing_columns_errors(self):
        df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
        with pytest.raises(ValueError, match="user_id"):
            PopularityRecommender().fit(df)
