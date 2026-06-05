"""Unit tests for src/eval/metrics.py."""

from __future__ import annotations

import math

import pytest

from src.eval.metrics import recall_at_k, ndcg_at_k, mrr


# ---------- recall_at_k ----------


class TestRecallAtK:
    def test_hit_at_position_1(self):
        assert recall_at_k([10, 20, 30], {10}, k=10) == 1.0

    def test_miss(self):
        assert recall_at_k([10, 20, 30], {99}, k=10) == 0.0

    def test_outside_top_k(self):
        # Relevant item is at position 4 but k=3 — should miss
        assert recall_at_k([10, 20, 30, 40], {40}, k=3) == 0.0

    def test_multiple_relevant_partial_hit(self):
        # 2 of 4 relevant are in top-3
        assert recall_at_k([10, 20, 30, 99], {10, 20, 100, 101}, k=3) == pytest.approx(0.5)

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k([10, 20], set(), k=5) == 0.0

    def test_empty_recommended(self):
        assert recall_at_k([], {1}, k=5) == 0.0

    def test_k_larger_than_recommended(self):
        # Should not error — just uses whole list
        assert recall_at_k([10, 20], {20}, k=100) == 1.0


# ---------- ndcg_at_k ----------


class TestNdcgAtK:
    def test_perfect_ranking(self):
        # Single relevant item at position 1 → NDCG = 1.0
        assert ndcg_at_k([10, 20, 30], {10}, k=10) == pytest.approx(1.0)

    def test_at_position_2(self):
        # DCG = 1/log2(3); IDCG = 1/log2(2) = 1
        expected = (1 / math.log2(3)) / 1.0
        assert ndcg_at_k([99, 10, 20], {10}, k=10) == pytest.approx(expected)

    def test_missing_returns_zero(self):
        assert ndcg_at_k([1, 2, 3], {99}, k=10) == 0.0

    def test_two_relevant_top_two(self):
        # DCG = 1/log2(2) + 1/log2(3); IDCG = same → NDCG = 1.0
        assert ndcg_at_k([10, 20, 99], {10, 20}, k=10) == pytest.approx(1.0)

    def test_two_relevant_split_positions(self):
        # Relevant at positions 1 and 3
        # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309
        dcg = 1.0 + 1.0 / math.log2(4)
        idcg = 1.0 + 1.0 / math.log2(3)
        assert ndcg_at_k([10, 99, 20], {10, 20}, k=10) == pytest.approx(dcg / idcg)

    def test_cutoff_excludes_relevant(self):
        # Relevant at position 5, k=3 → no hit, NDCG = 0
        assert ndcg_at_k([1, 2, 3, 4, 10], {10}, k=3) == 0.0

    def test_empty_relevant(self):
        assert ndcg_at_k([10, 20], set(), k=5) == 0.0


# ---------- mrr ----------


class TestMrr:
    def test_first_position(self):
        assert mrr([10, 20, 30], {10}) == pytest.approx(1.0)

    def test_third_position(self):
        assert mrr([99, 88, 10], {10}) == pytest.approx(1 / 3)

    def test_missing(self):
        assert mrr([1, 2, 3], {99}) == 0.0

    def test_takes_first_match(self):
        # Both 10 and 20 are relevant; 10 comes first → RR = 1.0
        assert mrr([10, 20, 30], {10, 20}) == pytest.approx(1.0)

    def test_empty_relevant(self):
        assert mrr([1, 2], set()) == 0.0
