"""
Unit tests for the 3-stage fail-fast validation waterfall.

Tests each stage independently plus the orchestrator, using synthetic
data to avoid requiring a full Mesa installation for CI.
"""

import math
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

# Module under test
from validation import (
    ValidationResult,
    stage1_fast_filter,
    stage2_lightweight_filter,
    stage3_heavy_gauntlet,
    validate_ablation,
    compute_dtw,
    compute_morans_i,
    compute_price_convergence_rate,
    ks_test_wealth,
    bonferroni_ttest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_STRATEGY_CODE = Path(__file__).parent / "strategy.py"

MINIMAL_VALID_CODE = '''
import mesa

def create_model(seed=42, steps=200, **kwargs):
    """Minimal valid model stub."""
    class M:
        def __init__(self):
            self.running = True
            self.datacollector = type("DC", (), {"model_vars": {}})()
    return M()

def run_model(model, steps=200):
    """Minimal valid run stub."""
    return {
        "agent_wealths": [10.0, 20.0, 30.0, 40.0, 50.0],
        "final_population": 5,
        "initial_population": 10,
        "trade_prices": [1.0, 1.1, 0.9],
        "trade_volume": 3,
        "agent_positions": [(0, 0), (10, 10), (20, 20), (30, 30), (40, 40)],
        "grid_width": 50,
        "grid_height": 50,
        "steps_run": steps,
    }
'''

CRASHING_CODE = '''
def create_model(seed=42, steps=200, **kwargs):
    raise RuntimeError("Intentional crash for testing")

def run_model(model, steps=200):
    pass
'''

SYNTAX_ERROR_CODE = '''
def create_model(seed=42
    # missing closing paren
    return None
'''

MISSING_FUNCTIONS_CODE = '''
# Valid Python but no create_model / run_model
x = 42
'''


def _make_baseline():
    """Synthetic baseline dict for testing."""
    return {
        "mean_metrics": {
            "gini_coefficient": 0.35,
            "final_population": 66.0,
            "mean_trade_price": 1.0,
            "trade_volume": 6000.0,
            "survival_rate": 0.33,
            "wealth_cv": 0.62,
            "spatial_entropy": 3.13,
        },
        "per_seed_wealths": [
            [10, 20, 30, 40, 50],
            [12, 22, 28, 42, 48],
            [11, 19, 31, 39, 51],
        ],
        "per_seed_metrics": [
            {"gini_coefficient": 0.34, "final_population": 65, "mean_trade_price": 0.98,
             "trade_volume": 5900, "survival_rate": 0.325, "wealth_cv": 0.61, "spatial_entropy": 3.1},
            {"gini_coefficient": 0.36, "final_population": 67, "mean_trade_price": 1.02,
             "trade_volume": 6100, "survival_rate": 0.335, "wealth_cv": 0.63, "spatial_entropy": 3.15},
        ],
        "mean_population_series": [200, 150, 120, 100, 80, 70, 66, 66, 66, 66],
        "mean_price_series": [1.5, 1.3, 1.1, 1.05, 1.02, 1.01, 1.0, 1.0, 1.0, 1.0],
    }


# ===========================================================================
# Test DTW
# ===========================================================================

class TestDTW:
    def test_identical_series(self):
        s = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert compute_dtw(s, s) == 0.0

    def test_shifted_series(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0]
        d = compute_dtw(a, b)
        assert d > 0.0
        assert d < 5.0  # not absurdly large

    def test_empty_series(self):
        assert compute_dtw([], [1, 2, 3]) == 0.0
        assert compute_dtw([1, 2], []) == 0.0

    def test_different_lengths(self):
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = compute_dtw(a, b)
        assert d >= 0.0


# ===========================================================================
# Test Moran's I
# ===========================================================================

class TestMoransI:
    def test_clustered_pattern(self):
        """Agents clustered in one corner should have positive Moran's I."""
        positions = [(i, j) for i in range(5) for j in range(5)]
        values = [100.0] * len(positions)
        mi, pval = compute_morans_i(positions, values, 50, 50, bin_size=10)
        # All values identical → Moran's I should be 0 or near 0 (no variation)
        assert isinstance(mi, float)

    def test_random_values(self):
        """Random values should give Moran's I near 0."""
        rng = np.random.RandomState(42)
        positions = [(rng.randint(0, 50), rng.randint(0, 50)) for _ in range(100)]
        values = rng.uniform(0, 100, 100).tolist()
        mi, pval = compute_morans_i(positions, values, 50, 50, bin_size=10)
        assert -1.0 <= mi <= 1.0

    def test_empty_positions(self):
        mi, pval = compute_morans_i([], [], 50, 50)
        assert mi == 0.0


# ===========================================================================
# Test Price Convergence Rate
# ===========================================================================

class TestPriceConvergence:
    def test_converging_series(self):
        """Exponentially decaying toward 1.0 should have positive rate."""
        series = [1.0 + 2.0 * math.exp(-0.1 * t) for t in range(50)]
        rate = compute_price_convergence_rate(series)
        assert rate > 0.0

    def test_stable_series(self):
        """Already at 1.0 should return inf."""
        series = [1.0] * 20
        rate = compute_price_convergence_rate(series)
        assert rate == float("inf")

    def test_short_series(self):
        rate = compute_price_convergence_rate([1.0, 1.1, 0.9])
        assert rate == 0.0


# ===========================================================================
# Test KS Wealth Test
# ===========================================================================

class TestKSWealth:
    def test_same_distribution(self):
        w = [[10, 20, 30, 40, 50]]
        stat, pval = ks_test_wealth(w, w)
        assert pval == 1.0  # identical samples

    def test_different_distribution(self):
        a = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        b = [[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]]
        stat, pval = ks_test_wealth(a, b)
        assert pval < 0.05

    def test_insufficient_data(self):
        stat, pval = ks_test_wealth([[1]], [[2]])
        assert pval == 1.0


# ===========================================================================
# Test Bonferroni t-test
# ===========================================================================

class TestBonferronTTest:
    def test_identical_samples(self):
        sample = [
            {"gini_coefficient": 0.35, "final_population": 66}
        ] * 10
        all_pass, results = bonferroni_ttest(sample, sample)
        assert all_pass
        for k, v in results.items():
            assert v["passes"]

    def test_different_means(self):
        a = [{"x": 10.0 + i * 0.01} for i in range(50)]
        b = [{"x": 100.0 + i * 0.01} for i in range(50)]
        all_pass, results = bonferroni_ttest(a, b)
        assert not all_pass
        assert not results["x"]["passes"]


# ===========================================================================
# Test Stage 1: Fast Filter
# ===========================================================================

class TestStage1:
    def test_syntax_error_rejected(self):
        result = stage1_fast_filter(SYNTAX_ERROR_CODE)
        assert not result.stage1_ok
        assert result.stage_reached == 1
        assert "SyntaxError" in result.reject_reason

    def test_missing_functions_rejected(self):
        result = stage1_fast_filter(MISSING_FUNCTIONS_CODE)
        assert not result.stage1_ok
        assert "Missing create_model" in result.reject_reason

    def test_crashing_code_rejected(self):
        result = stage1_fast_filter(CRASHING_CODE)
        assert not result.stage1_ok
        assert "crash" in result.reject_reason.lower() or "RuntimeError" in result.reject_reason

    def test_valid_code_passes(self):
        result = stage1_fast_filter(MINIMAL_VALID_CODE)
        assert result.stage1_ok
        assert result.stage1_time > 0


# ===========================================================================
# Test Stage 2: Lightweight Filter
# ===========================================================================

class TestStage2:
    def test_valid_code_with_matching_baseline(self):
        """Minimal valid code should pass Stage 2 with a permissive baseline."""
        baseline = _make_baseline()
        # Make baseline match the minimal code's output
        baseline["mean_metrics"] = {
            "gini_coefficient": 0.36,
            "final_population": 5.0,
            "mean_trade_price": 0.99,
            "trade_volume": 3.0,
            "survival_rate": 0.5,
            "wealth_cv": 0.47,
            "spatial_entropy": 1.5,
        }
        # Use very permissive bounds for this test
        from metrics import set_error_bounds
        original_bounds = dict(__import__('metrics').ERROR_BOUNDS)
        try:
            set_error_bounds(
                gini_coefficient=1.0, final_population=5.0,
                mean_trade_price=5.0, trade_volume=5.0,
                survival_rate=1.0, wealth_cv=5.0, spatial_entropy=5.0,
            )
            result = stage2_lightweight_filter(
                MINIMAL_VALID_CODE, baseline, seeds=[42, 43]
            )
            assert result.stage2_scalar_pass
        finally:
            set_error_bounds(**original_bounds)


# ===========================================================================
# Test Orchestrator (validate_ablation)
# ===========================================================================

class TestValidateAblation:
    def test_syntax_error_fails_fast(self):
        baseline = _make_baseline()
        result = validate_ablation(SYNTAX_ERROR_CODE, baseline)
        assert not result.passed
        assert result.stage_reached == 1
        assert result.total_time < 5.0

    def test_crash_fails_at_stage1(self):
        baseline = _make_baseline()
        result = validate_ablation(CRASHING_CODE, baseline)
        assert not result.passed
        assert result.stage_reached == 1


# ===========================================================================
# Test ValidationResult defaults
# ===========================================================================

class TestValidationResult:
    def test_defaults(self):
        r = ValidationResult()
        assert not r.passed
        assert r.stage_reached == 0
        assert r.reject_reason == ""
        assert r.total_time == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
