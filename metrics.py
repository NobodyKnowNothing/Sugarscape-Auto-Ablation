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

import json
import math
from pathlib import Path
from typing import Any
import numpy as np


# ---------------------------------------------------------------------------
# Error bounds per metric (from Epstein & Axtell, calibrated empirically)
# If a variant's metric deviates by more than epsilon from baseline,
# the variant FAILS and is reverted.
#
# These are exposed as module-level global variables so they can be easily
# inspected, modified, or scaled in a Jupyter notebook or interactive script.
# ---------------------------------------------------------------------------

# Which metrics use absolute vs relative comparison
ABSOLUTE_METRICS = {"gini_coefficient", "survival_rate"}

# Individual metric error bounds (global variables)
BOUND_GINI_COEFFICIENT: float = 0.02   # absolute tolerance (~0.79 sigma, ~2.48 SEM)
BOUND_FINAL_POPULATION: float = 0.05   # relative tolerance (~0.81 sigma, ~2.57 SEM)
BOUND_MEAN_TRADE_PRICE: float = 0.15   # relative tolerance (~0.55 sigma, ~1.73 SEM)
BOUND_TRADE_VOLUME: float = 0.10       # relative tolerance (~1.62 sigma, ~5.13 SEM)
BOUND_SURVIVAL_RATE: float = 0.02      # absolute tolerance (~0.98 sigma, ~3.11 SEM)
BOUND_WEALTH_CV: float = 0.05          # relative tolerance (~0.63 sigma, ~1.99 SEM)
BOUND_SPATIAL_ENTROPY: float = 0.10    # relative tolerance (~2.04 sigma, ~6.45 SEM)

# Master dictionary of error bounds (metric_name -> bound)
ERROR_BOUNDS: dict[str, float] = {
    "gini_coefficient":     BOUND_GINI_COEFFICIENT,
    "final_population":     BOUND_FINAL_POPULATION,
    "mean_trade_price":     BOUND_MEAN_TRADE_PRICE,
    "trade_volume":         BOUND_TRADE_VOLUME,
    "survival_rate":        BOUND_SURVIVAL_RATE,
    "wealth_cv":            BOUND_WEALTH_CV,
    "spatial_entropy":      BOUND_SPATIAL_ENTROPY,
}

# Calibrated sigma equivalents for the Mesa canonical baseline (N=10 runs)
DEFAULT_SIGMA_EQUIVALENTS: dict[str, float] = {
    "gini_coefficient":     0.79,   # bound 0.02 is ~0.79 sigma (2.48 SEM)
    "final_population":     0.81,   # bound 5% is ~0.81 sigma (2.57 SEM)
    "mean_trade_price":     0.55,   # bound 15% is ~0.55 sigma (1.73 SEM)
    "trade_volume":         1.62,   # bound 10% is ~1.62 sigma (5.13 SEM)
    "survival_rate":        0.98,   # bound 0.02 is ~0.98 sigma (3.11 SEM)
    "wealth_cv":            0.63,   # bound 5% is ~0.63 sigma (1.99 SEM)
    "spatial_entropy":      2.04,   # bound 10% is ~2.04 sigma (6.45 SEM)
}

BASELINE_METRICS_PATH = Path(__file__).parent / "baseline_metrics.json"


_LAST_KNOWN_BOUNDS: dict[str, float] = dict(ERROR_BOUNDS)


def sync_error_bounds() -> dict[str, float]:
    """
    Synchronize global BOUND_* variables and the ERROR_BOUNDS dictionary.
    Bi-directionally propagates changes whether you modified individual BOUND_*
    variables or the ERROR_BOUNDS dictionary directly.
    """
    global BOUND_GINI_COEFFICIENT, BOUND_FINAL_POPULATION, BOUND_MEAN_TRADE_PRICE
    global BOUND_TRADE_VOLUME, BOUND_SURVIVAL_RATE, BOUND_WEALTH_CV, BOUND_SPATIAL_ENTROPY
    global _LAST_KNOWN_BOUNDS

    var_map = {
        "gini_coefficient": "BOUND_GINI_COEFFICIENT",
        "final_population": "BOUND_FINAL_POPULATION",
        "mean_trade_price": "BOUND_MEAN_TRADE_PRICE",
        "trade_volume": "BOUND_TRADE_VOLUME",
        "survival_rate": "BOUND_SURVIVAL_RATE",
        "wealth_cv": "BOUND_WEALTH_CV",
        "spatial_entropy": "BOUND_SPATIAL_ENTROPY",
    }

    for k, var_name in var_map.items():
        var_val = globals()[var_name]
        dict_val = ERROR_BOUNDS.get(k, var_val)
        last_val = _LAST_KNOWN_BOUNDS.get(k, var_val)

        if var_val != last_val:
            # Individual variable was modified; propagate to dict
            ERROR_BOUNDS[k] = var_val
            _LAST_KNOWN_BOUNDS[k] = var_val
        elif dict_val != last_val:
            # Dict was modified; propagate to individual variable
            globals()[var_name] = dict_val
            _LAST_KNOWN_BOUNDS[k] = dict_val

    return ERROR_BOUNDS


def set_error_bounds(**kwargs) -> dict[str, float]:
    """
    Conveniently update one or more error bounds from a notebook.
    Updates both the global variables and ERROR_BOUNDS.
    
    Example:
        import metrics
        metrics.set_error_bounds(gini_coefficient=0.04, survival_rate=0.03)
    """
    global BOUND_GINI_COEFFICIENT, BOUND_FINAL_POPULATION, BOUND_MEAN_TRADE_PRICE
    global BOUND_TRADE_VOLUME, BOUND_SURVIVAL_RATE, BOUND_WEALTH_CV, BOUND_SPATIAL_ENTROPY
    
    var_map = {
        "gini_coefficient": "BOUND_GINI_COEFFICIENT",
        "final_population": "BOUND_FINAL_POPULATION",
        "mean_trade_price": "BOUND_MEAN_TRADE_PRICE",
        "trade_volume": "BOUND_TRADE_VOLUME",
        "survival_rate": "BOUND_SURVIVAL_RATE",
        "wealth_cv": "BOUND_WEALTH_CV",
        "spatial_entropy": "BOUND_SPATIAL_ENTROPY",
    }
    
    for k, v in kwargs.items():
        if k in ERROR_BOUNDS:
            ERROR_BOUNDS[k] = float(v)
            if k in var_map:
                globals()[var_map[k]] = float(v)
        else:
            raise KeyError(f"Unknown metric '{k}'. Valid metrics: {list(ERROR_BOUNDS.keys())}")
            
    return ERROR_BOUNDS


def set_sigma_bounds(n_sigma: float = 2.0, baseline_file: str | Path | None = None) -> dict[str, float]:
    """
    Set error bounds dynamically as multiples of standard deviations (sigma).
    Uses baseline_metrics.json to obtain each metric's standard deviation (std)
    and mean.
    
    For absolute metrics (gini_coefficient, survival_rate):
        bound = n_sigma * std
    For relative metrics:
        bound = (n_sigma * std) / |mean|
        
    Example:
        import metrics
        metrics.set_sigma_bounds(n_sigma=2.0)  # set 2-sigma bounds for all metrics
    """
    path = Path(baseline_file) if baseline_file else BASELINE_METRICS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Baseline metrics file not found: {path}")
        
    with open(path, "r") as f:
        data = json.load(f)
        
    means = data.get("mean_metrics", {})
    new_bounds = {}
    
    for metric_name in ERROR_BOUNDS:
        std_key = f"{metric_name}_std"
        std_val = data.get(std_key, 0.0)
        mean_val = means.get(metric_name, 1.0)
        
        if metric_name in ABSOLUTE_METRICS:
            new_bounds[metric_name] = float(n_sigma * std_val)
        else:
            denom = abs(mean_val) if abs(mean_val) > 1e-10 else 1.0
            new_bounds[metric_name] = float((n_sigma * std_val) / denom)
            
    return set_error_bounds(**new_bounds)


def set_sem_bounds(n_sem: float = 2.0, baseline_file: str | Path | None = None) -> dict[str, float]:
    """
    Set error bounds dynamically as multiples of Standard Error of the Mean (SEM = std / sqrt(n_runs)).
    
    Example:
        import metrics
        metrics.set_sem_bounds(n_sem=3.0)  # set 3-SEM bounds for all metrics
    """
    path = Path(baseline_file) if baseline_file else BASELINE_METRICS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Baseline metrics file not found: {path}")
        
    with open(path, "r") as f:
        data = json.load(f)
        
    n_runs = data.get("n_runs", 10)
    sqrt_n = math.sqrt(n_runs) if n_runs > 0 else 1.0
    return set_sigma_bounds(n_sigma=n_sem / sqrt_n, baseline_file=path)


def reset_default_bounds() -> dict[str, float]:
    """Reset all error bounds back to their canonical defaults."""
    return set_error_bounds(
        gini_coefficient=0.02,
        final_population=0.05,
        mean_trade_price=0.15,
        trade_volume=0.10,
        survival_rate=0.02,
        wealth_cv=0.05,
        spatial_entropy=0.10,
    )


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


def compute_all_metrics_with_timeseries(model_data: dict[str, Any]) -> dict[str, Any]:
    """
    Compute all scalar metrics AND return raw time-series data for DTW analysis.
    
    Extends compute_all_metrics() with additional keys:
        - "population_series": list of population counts per step
        - "price_series": list of mean trade prices per step
        - "gini_series": list of Gini coefficients per step
        - "agent_wealths": raw wealth list (pass-through for KS tests)
        - "agent_positions": raw position list (pass-through for Moran's I)
    
    Used by the Stage 3 Heavy Gauntlet in the validation waterfall.
    """
    result = compute_all_metrics(model_data)
    
    # Pass through raw data needed by downstream validation
    result["agent_wealths"] = model_data.get("agent_wealths", [])
    result["agent_positions"] = model_data.get("agent_positions", [])
    result["population_series"] = model_data.get("population_series", [])
    result["price_series"] = model_data.get("price_series", [])
    result["gini_series"] = model_data.get("gini_series", [])
    
    return result


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
                        variant_metrics: dict[str, float],
                        error_bounds: dict[str, float] | None = None) -> tuple[bool, dict[str, dict]]:
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
    sync_error_bounds()
    bounds = error_bounds if error_bounds is not None else ERROR_BOUNDS
    details = {}
    all_pass = True
    
    for metric_name, bound in bounds.items():
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
                              bounds_details: dict[str, dict],
                              error_bounds: dict[str, float] | None = None) -> str:
    """Format a full comparison report between baseline and variant."""
    lines = ["═══ Sugarscape Metrics Comparison ═══"]
    lines.append(f"{'Metric':24s} {'Baseline':>12s} {'Variant':>12s} {'Sim':>8s} {'Error':>8s} {'Bound':>8s} {'Pass':>6s}")
    lines.append("─" * 86)
    
    active_bounds = error_bounds if error_bounds is not None else bounds_details
    for name in active_bounds:
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
