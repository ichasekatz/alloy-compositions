"""Tests for comp_gen and jitterbug modules."""

from __future__ import annotations

from unittest import TestCase

import numpy as np
import pytest

from alloy_compositions import generate_compositions, jitter


class TestGenerateCompositions(TestCase):
    """Tests for generate_compositions."""

    def test_binary_count(self):
        """Binary system at n_comps=4 produces exactly 5 points including endpoints."""
        df = generate_compositions(["A", "B"], n_comps=4, max_order=2)
        assert len(df) == 5  # [0,1],[0.25,0.75],[0.5,0.5],[0.75,0.25],[1,0]

    def test_rows_sum_to_one(self):
        """All rows sum to exactly 1.0 within floating-point tolerance."""
        df = generate_compositions(["A", "B", "C"], n_comps=10, max_order=3)
        sums = df.sum(axis=1)
        assert np.allclose(sums, 1.0)

    def test_columns_match_elements(self):
        """Output columns match sorted input elements."""
        elements = ["Ni", "Al", "Co"]
        df = generate_compositions(elements, n_comps=5, max_order=2)
        assert set(df.columns) == set(elements)

    def test_no_negative_fractions(self):
        """All composition values are non-negative."""
        df = generate_compositions(["A", "B", "C"], n_comps=10, max_order=3)
        assert (df.values >= 0).all()

    def test_no_duplicates(self):
        """Returned DataFrame has no duplicate rows."""
        df = generate_compositions(["A", "B", "C"], n_comps=10, max_order=3)
        assert not df.duplicated().any()

    def test_max_order_two_only_binary(self):
        """max_order=2 on a 4-element pool returns at most 2 non-zero elements per row."""
        df = generate_compositions(["A", "B", "C", "D"], n_comps=5, max_order=2)
        # Rows include endmembers (1 non-zero) and interior (2 non-zero)
        non_zero = (df > 0).sum(axis=1)
        assert (non_zero <= 2).all()

    def test_invalid_max_order_raises(self):
        """max_order exceeding element count raises ValueError."""
        with pytest.raises(ValueError, match="max_order"):
            generate_compositions(["A", "B"], n_comps=5, max_order=5)

    def test_single_element_pool_raises(self):
        """max_order < 2 raises ValueError."""
        with pytest.raises(ValueError, match="max_order"):
            generate_compositions(["A", "B"], n_comps=5, max_order=1)


class TestJitter(TestCase):
    """Tests for jitter."""

    def setUp(self):
        """Small composition table for all jitter tests."""
        self.df = generate_compositions(["A", "B", "C"], n_comps=5, max_order=3)
        self.cols = ["A", "B", "C"]

    def test_output_rows_sum_to_one(self):
        """Jittered rows still sum to 1.0."""
        out = jitter(self.df, self.cols, sigma=0.01)
        assert np.allclose(out[self.cols].sum(axis=1), 1.0)

    def test_no_negative_fractions(self):
        """Jitter with clip ensures all values >= 0."""
        out = jitter(self.df, self.cols, sigma=0.05)
        assert (out[self.cols].values >= 0).all()

    def test_reproducible_with_seed(self):
        """Same seed produces identical output."""
        out1 = jitter(self.df, self.cols, sigma=0.005, seed=0)
        out2 = jitter(self.df, self.cols, sigma=0.005, seed=0)
        assert np.allclose(out1[self.cols].values, out2[self.cols].values)

    def test_different_seeds_differ(self):
        """Different seeds produce different perturbations."""
        out1 = jitter(self.df, self.cols, sigma=0.005, seed=0)
        out2 = jitter(self.df, self.cols, sigma=0.005, seed=99)
        assert not np.allclose(out1[self.cols].values, out2[self.cols].values)

    def test_missing_column_raises(self):
        """Requesting a column not in the DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            jitter(self.df, ["X"], sigma=0.005)

    def test_input_not_mutated(self):
        """Original DataFrame is unchanged after jitter."""
        original = self.df.copy()
        jitter(self.df, self.cols, sigma=0.05)
        assert np.allclose(self.df[self.cols].values, original[self.cols].values)
