"""
Sugarscape Metrics — Fixed Evaluation Harness (DO NOT MODIFY)

Inspired by Epstein & Axtell's "Growing Artificial Societies" (1996).
Computes all key emergent metrics from the Sugarscape model and returns
them as a standardized dictionary suitable for comparison.

These metrics capture the original paper's primary findings:
  1. Wealth distribution → Gini coefficient
  2. Population dynamics → carrying capacity convergence
  3. Trade dynamics → volume, geometric mean price
  4. Agent survival → fraction alive after N steps
  5. Wealth variance → coefficient of variation
  6. Spatial patterns → entropy of agent distribution

This file is the GROUND TRUTH. Never modify it during ablation.
"""

import math
import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# Error bounds per metric (from Epstein & Axtell, calibrated empirically)
# If a variant's metric deviates by more than epsilon from baseline,
# the variant FAILS and is reverted.
# ---------------------------------------------------------------------------
ERROR_BOUNDS = {
    "gini_coefficient":     0.02,   # absolute tolerance - matches Gini 2 sig figs
    "final_population":     0.05,   # relative tolerance - matches carrying capacity SEM
    "mean_trade_price":     0.15,   # relative tolerance - matches price convergence SEM (~18% std)
    "trade_volume":         0.10,   # relative tolerance - matches trade activity SEM (~11% std)
    "survival_rate":        0.02,   # absolute tolerance - matches survival rate 2 sig figs
    "wealth_cv":            0.05,   # relative tolerance - matches inequality variance SEM
    "spatial_entropy":      0.10,   # relative tolerance - matches migration entropy SEM (~5.6% std)
}

# Which metrics use absolute vs relative comparison
ABSOLUTE_METRICS = {"gini_coefficient", "survival_rate"}


def gini_coefficient(values: list[float]) -> float:
    """
    Compute the Gini coefficient for a list of values.
    Returns 0 (perfect equality) to 1 (perfect inequality).
    
    This is the primary metric from Epstein & Axtell showing that
    skewed wealth distribution emerges from simple foraging rules.
    """
    if not values or len(values) < 2:
        return 0.0
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    
    if total == 0:
        return 0.0
    
    cumulative = 0.0
    weighted_sum = 0.0
    for i, val in enumerate(sorted_vals):
        cumulative += val
        weighted_sum += (2 * (i + 1) - n - 1) * val
    
    return weighted_sum / (n * total)


def geometric_mean(values: list[float]) -> float:
    """
    Geometric mean of a list of positive values.
    Used for trade price computation per the Sugarscape trading rules.
    """
    if not values:
        return -1.0
    
    positive = [v for v in values if v > 0]
    if not positive:
        return -1.0
    
    return float(np.exp(np.mean(np.log(positive))))


def coefficient_of_variation(values: list[float]) -> float:
    """CV = std / mean. Measures relative variability of wealth."""
    if not values or len(values) < 2:
        return 0.0
    
    mean = np.mean(values)
    if mean == 0:
        return 0.0
    
    return float(np.std(values) / mean)


def spatial_entropy(positions: list[tuple[int, int]], grid_width: int, grid_height: int, 
                    bin_size: int = 10) -> float:
    """
    Compute Shannon entropy of agent spatial distribution.
    Divides the grid into bins and measures how uniformly agents are spread.
    Higher entropy = more uniform distribution.
    
    This captures migration patterns described in the original paper.
    """
    if not positions:
        return 0.0
    
    n_bins_x = max(1, grid_width // bin_size)
    n_bins_y = max(1, grid_height // bin_size)
    
    grid = np.zeros((n_bins_x, n_bins_y))
    for x, y in positions:
        bx = min(x // bin_size, n_bins_x - 1)
        by = min(y // bin_size, n_bins_y - 1)
        grid[bx, by] += 1
    
    total = grid.sum()
    if total == 0:
        return 0.0
    
    probs = grid.flatten() / total
    probs = probs[probs > 0]
    
    return float(-np.sum(probs * np.log2(probs)))


def compute_all_metrics(model_data: dict[str, Any]) -> dict[str, float]:
    """
    Compute all Sugarscape metrics from a model run's collected data.
    
    Expected model_data keys:
        - "agent_wealths": list of (sugar + spice) for each living agent
        - "final_population": int, number of surviving agents
        - "initial_population": int, starting population
        - "trade_prices": list of all trade prices in the run
        - "trade_volume": int, total trades executed
        - "agent_positions": list of (x, y) tuples for living agents
        - "grid_width": int
        - "grid_height": int
        - "steps_run": int
    
    Returns dict of metric_name -> float value.
    """
    agent_wealths = model_data.get("agent_wealths", [])
    final_pop = model_data.get("final_population", 0)
    initial_pop = model_data.get("initial_population", 200)
    trade_prices = model_data.get("trade_prices", [])
    trade_vol = model_data.get("trade_volume", 0)
    positions = model_data.get("agent_positions", [])
    grid_w = model_data.get("grid_width", 50)
    grid_h = model_data.get("grid_height", 50)
    
    return {
        "gini_coefficient": gini_coefficient(agent_wealths),
        "final_population": float(final_pop),
        "mean_trade_price": geometric_mean(trade_prices),
        "trade_volume": float(trade_vol),
        "survival_rate": float(final_pop) / max(initial_pop, 1),
        "wealth_cv": coefficient_of_variation(agent_wealths),
        "spatial_entropy": spatial_entropy(positions, grid_w, grid_h),
    }


def compute_similarity_matrix(baseline_metrics: dict[str, float],
                              variant_metrics: dict[str, float]) -> dict[str, float]:
    """
    Compute per-metric similarity between baseline and variant.
    
    Returns a dict of metric_name -> similarity_score where:
      - 1.0 = perfect match
      - 0.0 = maximally different
    
    Uses absolute tolerance for Gini and survival_rate,
    relative tolerance for all other metrics.
    """
    similarity = {}
    
    for metric_name in baseline_metrics:
        base_val = baseline_metrics[metric_name]
        var_val = variant_metrics.get(metric_name, 0.0)
        
        if metric_name in ABSOLUTE_METRICS:
            # Absolute similarity: 1 - |diff|
            diff = abs(base_val - var_val)
            similarity[metric_name] = max(0.0, 1.0 - diff)
        else:
            # Relative similarity: 1 - |diff/base|
            if abs(base_val) < 1e-10:
                # If baseline is ~0, check if variant is also ~0
                similarity[metric_name] = 1.0 if abs(var_val) < 1e-10 else 0.0
            else:
                rel_diff = abs(base_val - var_val) / abs(base_val)
                similarity[metric_name] = max(0.0, 1.0 - rel_diff)
    
    return similarity


def check_within_bounds(baseline_metrics: dict[str, float],
                        variant_metrics: dict[str, float]) -> tuple[bool, dict[str, dict]]:
    """
    Check whether ALL variant metrics are within the allowed error bounds 
    of the baseline metrics.
    
    Returns:
        (passes: bool, details: dict)
        
    details maps metric_name -> {
        "baseline": float, 
        "variant": float,
        "error": float,     # actual error (abs or relative)
        "bound": float,     # allowed error
        "passes": bool
    }
    """
    details = {}
    all_pass = True
    
    for metric_name, bound in ERROR_BOUNDS.items():
        base_val = baseline_metrics.get(metric_name, 0.0)
        var_val = variant_metrics.get(metric_name, 0.0)
        
        if metric_name in ABSOLUTE_METRICS:
            error = abs(base_val - var_val)
        else:
            if abs(base_val) < 1e-10:
                error = 0.0 if abs(var_val) < 1e-10 else float('inf')
            else:
                error = abs(base_val - var_val) / abs(base_val)
        
        passes = error <= bound
        if not passes:
            all_pass = False
        
        details[metric_name] = {
            "baseline": base_val,
            "variant": var_val,
            "error": error,
            "bound": bound,
            "passes": passes,
        }
    
    return all_pass, details


def format_metrics_report(metrics: dict[str, float]) -> str:
    """Format metrics as a human-readable report string."""
    lines = ["--- Sugarscape Metrics ---"]
    for name, val in metrics.items():
        lines.append(f"  {name:24s}: {val:.6f}")
    return "\n".join(lines)


def format_comparison_report(baseline: dict[str, float], variant: dict[str, float],
                              similarity: dict[str, float], 
                              bounds_details: dict[str, dict]) -> str:
    """Format a full comparison report between baseline and variant."""
    lines = ["═══ Sugarscape Metrics Comparison ═══"]
    lines.append(f"{'Metric':24s} {'Baseline':>12s} {'Variant':>12s} {'Sim':>8s} {'Error':>8s} {'Bound':>8s} {'Pass':>6s}")
    lines.append("─" * 86)
    
    for name in ERROR_BOUNDS:
        b = baseline.get(name, 0.0)
        v = variant.get(name, 0.0)
        s = similarity.get(name, 0.0)
        d = bounds_details.get(name, {})
        err = d.get("error", 0.0)
        bound = d.get("bound", 0.0)
        passes = d.get("passes", False)
        
        emoji = "✅" if passes else "❌"
        lines.append(f"  {name:22s} {b:12.4f} {v:12.4f} {s:8.4f} {err:8.4f} {bound:8.4f} {emoji}")
    
    return "\n".join(lines)
