"""Unit tests for src/vision/expiry_model.py (synthetic shelf-life heuristic)."""

from __future__ import annotations

from datetime import date, timedelta

from src.vision.expiry_model import (
    SHELF_LIFE_DAYS,
    assign_expiry_dates,
    categorize,
    urgency_score,
)


class TestCategorize:
    def test_known_keywords(self):
        assert categorize("chicken breast") == "meat"
        assert categorize("fresh salmon") == "fish"
        assert categorize("whole milk") == "dairy"
        assert categorize("eggs") == "eggs"
        assert categorize("baby spinach") == "leafy_greens"
        assert categorize("strawberries") == "berries"
        assert categorize("carrots") == "root_vegetables"
        assert categorize("granny smith apple") == "fruit"
        assert categorize("soy sauce") == "condiments"
        assert categorize("brown rice") == "grains_dry"

    def test_unknown_falls_back_to_default(self):
        assert categorize("dragonfruit widget") == "default"

    def test_case_insensitive(self):
        assert categorize("CHICKEN") == "meat"


class TestAssignExpiryDates:
    def test_deterministic_for_same_seed(self):
        a = assign_expiry_dates(["chicken", "milk", "apple"], detected_on=date(2026, 1, 1))
        b = assign_expiry_dates(["chicken", "milk", "apple"], detected_on=date(2026, 1, 1))
        assert a == b

    def test_dates_lie_within_category_shelf_life(self):
        on = date(2026, 1, 1)
        out = assign_expiry_dates(["chicken", "rice"], detected_on=on)
        lo, hi = SHELF_LIFE_DAYS["meat"]
        assert on + timedelta(days=lo) <= out["chicken"] <= on + timedelta(days=hi)
        lo, hi = SHELF_LIFE_DAYS["grains_dry"]
        assert on + timedelta(days=lo) <= out["rice"] <= on + timedelta(days=hi)

    def test_empty_list(self):
        assert assign_expiry_dates([], detected_on=date(2026, 1, 1)) == {}

    def test_seed_changes_sampling(self):
        on = date(2026, 1, 1)
        a = assign_expiry_dates(["chicken"], detected_on=on, seed=1)
        b = assign_expiry_dates(["chicken"], detected_on=on, seed=999)
        # Not guaranteed different, but the meat range (2..5) is wide enough that
        # at least one of several items should differ across seeds.
        many_a = assign_expiry_dates(["chicken", "beef", "pork", "fish"], detected_on=on, seed=1)
        many_b = assign_expiry_dates(["chicken", "beef", "pork", "fish"], detected_on=on, seed=999)
        assert many_a != many_b


class TestUrgencyScore:
    def test_overdue_is_max(self):
        assert urgency_score(date.today() - timedelta(days=3)) == 1.0

    def test_today_is_max(self):
        assert urgency_score(date.today()) == 1.0

    def test_far_future_is_zero(self):
        assert urgency_score(date.today() + timedelta(days=14)) == 0.0
        assert urgency_score(date.today() + timedelta(days=30)) == 0.0

    def test_monotonic_decay(self):
        d1 = urgency_score(date.today() + timedelta(days=2))
        d7 = urgency_score(date.today() + timedelta(days=7))
        assert 0.0 < d7 < d1 < 1.0

    def test_explicit_today_arg(self):
        today = date(2026, 1, 1)
        assert urgency_score(date(2026, 1, 8), today=today) == 0.5
