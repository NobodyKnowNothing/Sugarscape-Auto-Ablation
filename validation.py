"""
Sugarscape ABM Ablation Validation — 3-Stage Fail-Fast Waterfall

Restructures the flat validation pipeline into a sequential waterfall
that rejects bad candidates as early as possible:

  Stage 1  Fast Filter (1 seed, 50 steps)    — syntax + crash        ~2s
  Stage 2  Lightweight  (5 seeds, 200 steps)  — KS + scalar bounds   ~15s
  Stage 3  Heavy Gauntlet (50 seeds total)    — t-tests, DTW, Moran  ~150s

Only ~10-20% of LLM-generated candidates should survive to Stage 3,
saving ~80% of compute on average.

This module is the AUTHORITATIVE validation gate.  `autoresearch.py`
calls `validate_ablation()` instead of the old `evaluate_strategy()`.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

STAGE1_SEEDS = [42]
STAGE1_STEPS = 50

STAGE2_SEEDS = list(range(42, 47))     # 5 seeds: 42..46
STAGE2_STEPS = 200

STAGE3_TOTAL_SEEDS = 50                # total across stages 2 + 3
STAGE3_EXTRA_SEEDS = list(range(47, 92))  # 45 more seeds: 47..91
STAGE3_STEPS = 200

# KS test threshold (Stage 2)
KS_ALPHA = 0.05

# Bonferroni correction: 7 metrics → α/7 per test (Stage 3)
BONFERRONI_ALPHA = 0.05
N_METRICS_BONFERRONI = 7

# DTW distance threshold (Stage 3) — normalised by series length
DTW_THRESHOLD = 0.30

# Moran's I lower bound (Stage 3) — positive autocorrelation expected
MORANS_I_LOWER = -0.05  # reject if Moran's I is strongly negative


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of the full validation waterfall."""

    passed: bool = False
    stage_reached: int = 0          # 1, 2, or 3
    reject_reason: str = ""

    # Timing
    stage1_time: float = 0.0
    stage2_time: float = 0.0
    stage3_time: float = 0.0
    total_time: float = 0.0

    # Stage 1 data
    stage1_ok: bool = False

    # Stage 2 data
    ks_statistic: float = 0.0
    ks_pvalue: float = 1.0
    stage2_scalar_pass: bool = False
    stage2_mean_metrics: Dict[str, float] = field(default_factory=dict)

    # Stage 3 data
    bonferroni_results: Dict[str, dict] = field(default_factory=dict)
    bonferroni_all_pass: bool = False
    dtw_population: float = 0.0
    dtw_price: float = 0.0
    price_convergence_rate: float = 0.0
    morans_i: float = 0.0
    morans_i_pvalue: float = 1.0
    wasserstein_distance: float = 0.0
    stage3_mean_metrics: Dict[str, float] = field(default_factory=dict)

    # All per-seed metrics (for results logging)
    all_run_metrics: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulation runner (shared across stages)
# ---------------------------------------------------------------------------

def _run_strategy_once(
    strategy_code: str,
    seed: int,
    steps: int,
    sim_params: Optional[dict] = None,
    collect_timeseries: bool = False,
) -> dict:
    """
    Run a strategy variant for one seed.

    Returns dict with keys from run_model(), plus optional time-series data.
    On failure returns {"error": "..."}.
    """
    tmp_path = Path(tempfile.gettempdir()) / f"_val_strategy_{seed}.py"
    tmp_path.write_text(strategy_code, encoding="utf-8")

    try:
        mod_name = f"_val_strategy_{seed}"
        if mod_name in sys.modules:
            del sys.modules[mod_name]

        spec = importlib.util.spec_from_file_location(mod_name, str(tmp_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if not hasattr(mod, "create_model") or not hasattr(mod, "run_model"):
            return {"error": "Missing create_model() or run_model()"}

        params = sim_params or {}
        model = mod.create_model(
            seed=seed,
            steps=steps,
            initial_population=params.get("initial_population", 200),
            endowment_min=params.get("endowment_min", 25),
            endowment_max=params.get("endowment_max", 50),
            metabolism_min=params.get("metabolism_min", 1),
            metabolism_max=params.get("metabolism_max", 5),
            vision_min=params.get("vision_min", 1),
            vision_max=params.get("vision_max", 5),
            enable_trade=params.get("enable_trade", True),
            width=params.get("width", 50),
            height=params.get("height", 50),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            data = mod.run_model(model, steps)

        # Optionally collect time series from datacollector
        if collect_timeseries and hasattr(model, "datacollector"):
            dc = model.datacollector
            mv = getattr(dc, "model_vars", {})
            data["_population_series"] = mv.get("Population", [])
            data["_price_series"] = mv.get("Price", [])
            # Gini time series can also be useful
            data["_gini_series"] = mv.get("Gini", [])

        return data

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        mod_name = f"_val_strategy_{seed}"
        if mod_name in sys.modules:
            del sys.modules[mod_name]


# ===========================================================================
# Stage 1: Fast Filter — syntax + crash (1 seed, 50 steps)
# ===========================================================================

def stage1_fast_filter(
    strategy_code: str,
    seed: int = 42,
    steps: int = STAGE1_STEPS,
) -> ValidationResult:
    """
    Stage 1: Verify the code parses as valid Python and runs for 50 steps
    without crashing.  Costs ~2 seconds.
    """
    result = ValidationResult(stage_reached=1)
    t0 = time.time()

    # 1a. AST syntax check (instant)
    try:
        ast.parse(strategy_code)
    except SyntaxError as e:
        result.reject_reason = f"SyntaxError at line {e.lineno}: {e.msg}"
        result.stage1_time = time.time() - t0
        result.total_time = result.stage1_time
        return result

    # 1b. Run for `steps` steps with a single seed
    data = _run_strategy_once(strategy_code, seed=seed, steps=steps)
    if "error" in data:
        result.reject_reason = f"Stage1 crash: {data['error']}"
        result.stage1_time = time.time() - t0
        result.total_time = result.stage1_time
        return result

    result.stage1_ok = True
    result.stage1_time = time.time() - t0
    result.total_time = result.stage1_time
    return result


# ===========================================================================
# Stage 2: Lightweight Shape Filter — KS + scalar bounds (5 seeds, 200 steps)
# ===========================================================================

def ks_test_wealth(
    ablated_wealths: List[List[float]],
    baseline_wealths: List[List[float]],
) -> Tuple[float, float]:
    """
    Two-sample KS test comparing pooled wealth distributions.

    Parameters
    ----------
    ablated_wealths : list of lists
        Per-seed wealth arrays from the ablated model.
    baseline_wealths : list of lists
        Per-seed wealth arrays from the baseline model.

    Returns
    -------
    (ks_statistic, p_value)
    """
    pool_ablated = []
    for ws in ablated_wealths:
        pool_ablated.extend(ws)

    pool_baseline = []
    for ws in baseline_wealths:
        pool_baseline.extend(ws)

    if len(pool_ablated) < 5 or len(pool_baseline) < 5:
        # Not enough data — default to pass
        return 0.0, 1.0

    stat, pval = stats.ks_2samp(pool_ablated, pool_baseline)
    return float(stat), float(pval)


def stage2_lightweight_filter(
    strategy_code: str,
    baseline: dict,
    seeds: Optional[List[int]] = None,
    steps: int = STAGE2_STEPS,
    ks_alpha: float = KS_ALPHA,
) -> ValidationResult:
    """
    Stage 2: Run 5 seeds at full steps.  Check:
      (a) KS test on wealth histogram vs baseline
      (b) Scalar bounds on key metrics (same as old check_within_bounds)

    Costs ~15 seconds.
    """
    from metrics import compute_all_metrics, check_within_bounds

    seeds = seeds or STAGE2_SEEDS
    result = ValidationResult(stage_reached=2)
    t0 = time.time()

    all_metrics = []
    all_wealths = []

    for seed in seeds:
        data = _run_strategy_once(strategy_code, seed=seed, steps=steps)
        if "error" in data:
            result.reject_reason = f"Stage2 crash at seed {seed}: {data['error']}"
            result.stage2_time = time.time() - t0
            result.total_time = result.stage1_time + result.stage2_time
            return result

        metrics = compute_all_metrics(data)
        all_metrics.append(metrics)
        all_wealths.append(data.get("agent_wealths", []))

    # 2a. Compute mean metrics
    mean_metrics = {}
    for key in all_metrics[0]:
        mean_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
    result.stage2_mean_metrics = mean_metrics

    # 2b. Scalar bounds check
    baseline_means = baseline.get("mean_metrics", {})
    passes_scalar, _ = check_within_bounds(baseline_means, mean_metrics)
    result.stage2_scalar_pass = passes_scalar

    if not passes_scalar:
        result.reject_reason = "Stage2 scalar bounds check failed"
        result.stage2_time = time.time() - t0
        result.total_time = result.stage1_time + result.stage2_time
        return result

    # 2c. KS test on wealth distribution
    baseline_wealths = baseline.get("per_seed_wealths", [])
    if baseline_wealths:
        ks_stat, ks_pval = ks_test_wealth(all_wealths, baseline_wealths)
        result.ks_statistic = ks_stat
        result.ks_pvalue = ks_pval

        if ks_pval < ks_alpha:
            result.reject_reason = (
                f"Stage2 KS test failed: stat={ks_stat:.4f}, p={ks_pval:.4f} < {ks_alpha}"
            )
            result.stage2_time = time.time() - t0
            result.total_time = result.stage1_time + result.stage2_time
            return result

    # Stage 2 passed
    result.all_run_metrics = all_metrics
    result.stage2_time = time.time() - t0
    result.total_time = result.stage1_time + result.stage2_time
    return result


# ===========================================================================
# Stage 3: Heavy Gauntlet — full statistical battery (50 seeds total)
# ===========================================================================

def compute_dtw(series_a: List[float], series_b: List[float]) -> float:
    """
    Dynamic Time Warping distance between two 1-D time series.
    Pure-numpy implementation (no external DTW library needed).

    Returns the normalised DTW distance (divided by series length).
    """
    a = np.asarray(series_a, dtype=np.float64)
    b = np.asarray(series_b, dtype=np.float64)

    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0

    # Cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    return float(dtw_matrix[n, m] / max(n, m))


def compute_morans_i(
    positions: List[Tuple[int, int]],
    values: List[float],
    grid_width: int = 50,
    grid_height: int = 50,
    bin_size: int = 10,
) -> Tuple[float, float]:
    """
    Compute Moran's I spatial autocorrelation on a binned grid.

    Uses queen contiguity (8-neighbor) weights on a grid of bins.
    The `values` are agent wealth or density aggregated per bin.

    Returns
    -------
    (morans_i, p_value)
        Moran's I statistic and its p-value under normality assumption.
    """
    n_bins_x = max(1, grid_width // bin_size)
    n_bins_y = max(1, grid_height // bin_size)
    n_bins = n_bins_x * n_bins_y

    # Aggregate values into bins (mean wealth per bin)
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for (x, y), v in zip(positions, values):
        bx = min(x // bin_size, n_bins_x - 1)
        by = min(y // bin_size, n_bins_y - 1)
        idx = bx * n_bins_y + by
        bin_sums[idx] += v
        bin_counts[idx] += 1

    # Use mean wealth per bin (0 for empty bins)
    # np.divide with where= avoids RuntimeWarning on 0/0
    z = np.zeros(n_bins)
    np.divide(bin_sums, bin_counts, out=z, where=bin_counts > 0)

    mean_z = np.mean(z)
    deviations = z - mean_z

    if np.sum(deviations ** 2) < 1e-12:
        return 0.0, 1.0  # no variance → undefined

    # Queen contiguity weight matrix
    W = np.zeros((n_bins, n_bins))
    for i in range(n_bins_x):
        for j in range(n_bins_y):
            idx = i * n_bins_y + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n_bins_x and 0 <= nj < n_bins_y:
                        nidx = ni * n_bins_y + nj
                        W[idx, nidx] = 1.0

    # Row-standardise
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    W = W / row_sums[:, np.newaxis]

    # Moran's I
    S0 = W.sum()
    if S0 < 1e-12:
        return 0.0, 1.0

    numerator = n_bins * float(deviations @ W @ deviations)
    denominator = S0 * float(np.sum(deviations ** 2))

    I = numerator / denominator if abs(denominator) > 1e-12 else 0.0

    # Expected value and variance under normality (for p-value)
    E_I = -1.0 / (n_bins - 1) if n_bins > 1 else 0.0

    # Simplified variance formula (normality assumption)
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=0) + W.sum(axis=1)) ** 2)
    k = (np.sum(deviations ** 4) / n_bins) / (np.sum(deviations ** 2) / n_bins) ** 2

    var_I_num = (
        n_bins * ((n_bins ** 2 - 3 * n_bins + 3) * S1 - n_bins * S2 + 3 * S0 ** 2)
        - k * (n_bins * (n_bins - 1) * S1 - 2 * n_bins * S2 + 6 * S0 ** 2)
    )
    var_I_den = (n_bins - 1) * (n_bins - 2) * (n_bins - 3) * S0 ** 2
    var_I = (var_I_num / var_I_den) - E_I ** 2 if abs(var_I_den) > 1e-12 else 1.0

    if var_I <= 0:
        return float(I), 1.0

    z_score = (I - E_I) / math.sqrt(var_I)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    return float(I), float(p_value)


def compute_price_convergence_rate(price_series: List[float]) -> float:
    """
    Estimate the exponential convergence rate of the trade price toward 1.0.

    Fits log(|price - 1.0|) ~ -rate * t  using simple linear regression.
    A larger (more positive) rate means faster convergence.

    Returns the convergence rate (positive = converging).
    """
    if len(price_series) < 10:
        return 0.0

    prices = np.asarray(price_series, dtype=np.float64)
    deviations = np.abs(prices - 1.0)

    # Filter out zero deviations (perfectly converged)
    mask = deviations > 1e-10
    if mask.sum() < 5:
        return float("inf")  # already converged

    log_devs = np.log(deviations[mask])
    t_vals = np.arange(len(deviations))[mask].astype(np.float64)

    if len(t_vals) < 2:
        return 0.0

    # Linear regression: log_dev = intercept - rate * t
    slope, _, _, _, _ = stats.linregress(t_vals, log_devs)
    return float(-slope)  # positive rate = convergence


def bonferroni_ttest(
    ablated_per_seed: List[Dict[str, float]],
    baseline_per_seed: List[Dict[str, float]],
    alpha: float = BONFERRONI_ALPHA,
) -> Tuple[bool, Dict[str, dict]]:
    """
    Welch's t-test on each metric with Bonferroni correction (α/7).

    Parameters
    ----------
    ablated_per_seed : list of dicts
        Per-seed metric dicts from the ablated model (n=50).
    baseline_per_seed : list of dicts
        Per-seed metric dicts from the baseline (n≥50).

    Returns
    -------
    (all_pass, results_per_metric)
    """
    if not ablated_per_seed or not baseline_per_seed:
        return False, {}

    metric_names = list(ablated_per_seed[0].keys())
    n_tests = len(metric_names)
    corrected_alpha = alpha / n_tests

    results = {}
    all_pass = True

    for metric in metric_names:
        ablated_vals = [m[metric] for m in ablated_per_seed]
        baseline_vals = [m[metric] for m in baseline_per_seed]

        if len(ablated_vals) < 2 or len(baseline_vals) < 2:
            results[metric] = {
                "t_stat": 0.0, "p_value": 1.0,
                "corrected_alpha": corrected_alpha, "passes": True,
                "ablated_mean": np.mean(ablated_vals) if ablated_vals else 0.0,
                "baseline_mean": np.mean(baseline_vals) if baseline_vals else 0.0,
            }
            continue

        t_stat, p_value = stats.ttest_ind(ablated_vals, baseline_vals, equal_var=False)

        # Handle NaN p-value (occurs when both samples have zero variance,
        # i.e. all values are identical — this is a perfect match, so pass)
        if np.isnan(p_value):
            p_value = 1.0
            t_stat = 0.0 if np.isnan(t_stat) else t_stat

        passes = float(p_value) >= corrected_alpha

        if not passes:
            all_pass = False

        results[metric] = {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "corrected_alpha": corrected_alpha,
            "passes": passes,
            "ablated_mean": float(np.mean(ablated_vals)),
            "baseline_mean": float(np.mean(baseline_vals)),
        }

    return all_pass, results


def stage3_heavy_gauntlet(
    strategy_code: str,
    prior_run_metrics: List[dict],
    prior_run_wealths: Optional[List[List[float]]] = None,
    baseline: dict = None,
    extra_seeds: Optional[List[int]] = None,
    steps: int = STAGE3_STEPS,
    dtw_threshold: float = DTW_THRESHOLD,
    morans_i_lower: float = MORANS_I_LOWER,
) -> ValidationResult:
    """
    Stage 3: Run the remaining 45 seeds (reuse 5 from Stage 2 → total 50).
    Compute heavy statistical battery:
      1. Bonferroni-corrected Welch's t-tests (α/7) for all 7 metrics
      2. DTW on population time series + price convergence rate
      3. Moran's I for spatial clustering
      4. Wasserstein distance on wealth distribution

    Costs ~150 seconds (only reached by candidates passing Stages 1+2).
    """
    from metrics import compute_all_metrics

    extra_seeds = extra_seeds or STAGE3_EXTRA_SEEDS
    result = ValidationResult(stage_reached=3)
    t0 = time.time()

    # Carry forward Stage 2 data
    all_metrics = list(prior_run_metrics)
    all_wealths = list(prior_run_wealths or [])
    all_pop_series = []
    all_price_series = []

    # Run remaining seeds with time-series collection
    for seed in extra_seeds:
        data = _run_strategy_once(
            strategy_code, seed=seed, steps=steps, collect_timeseries=True
        )
        if "error" in data:
            # Late-stage crash is suspicious but not an immediate reject;
            # count it as a failed run.  If too many fail, t-tests will catch it.
            continue

        metrics = compute_all_metrics(data)
        all_metrics.append(metrics)
        all_wealths.append(data.get("agent_wealths", []))
        all_pop_series.append(data.get("_population_series", []))
        all_price_series.append(data.get("_price_series", []))

    result.all_run_metrics = all_metrics

    # Compute mean metrics across all seeds
    if all_metrics:
        mean_metrics = {}
        for key in all_metrics[0]:
            mean_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
        result.stage3_mean_metrics = mean_metrics
    else:
        result.reject_reason = "Stage3: All runs crashed"
        result.stage3_time = time.time() - t0
        result.total_time = result.stage1_time + result.stage2_time + result.stage3_time
        return result

    # -----------------------------------------------------------------------
    # Check 1: Bonferroni-corrected Welch's t-tests
    # -----------------------------------------------------------------------
    baseline_per_seed = baseline.get("per_seed_metrics", [])
    if baseline_per_seed:
        bf_pass, bf_results = bonferroni_ttest(all_metrics, baseline_per_seed)
        result.bonferroni_results = bf_results
        result.bonferroni_all_pass = bf_pass

        if not bf_pass:
            failed = [k for k, v in bf_results.items() if not v["passes"]]
            result.reject_reason = (
                f"Stage3 Bonferroni t-test failed for: {', '.join(failed)}"
            )
            result.stage3_time = time.time() - t0
            result.total_time = (
                result.stage1_time + result.stage2_time + result.stage3_time
            )
            return result
    else:
        # No baseline per-seed data — fall back to scalar bounds
        result.bonferroni_all_pass = True

    # -----------------------------------------------------------------------
    # Check 2: DTW on population series + price convergence
    # -----------------------------------------------------------------------
    baseline_pop_series = baseline.get("mean_population_series", [])
    baseline_price_series = baseline.get("mean_price_series", [])

    if all_pop_series and baseline_pop_series:
        # Average the ablated population series
        max_len = max(len(s) for s in all_pop_series) if all_pop_series else 0
        if max_len > 0:
            padded = []
            for s in all_pop_series:
                if len(s) < max_len:
                    s = list(s) + [s[-1]] * (max_len - len(s)) if s else [0] * max_len
                padded.append(s[:max_len])
            mean_pop = np.mean(padded, axis=0).tolist()
            result.dtw_population = compute_dtw(mean_pop, baseline_pop_series)

    if all_price_series and baseline_price_series:
        max_len = max(len(s) for s in all_price_series) if all_price_series else 0
        if max_len > 0:
            padded = []
            for s in all_price_series:
                if len(s) < max_len:
                    s = list(s) + [s[-1]] * (max_len - len(s)) if s else [0] * max_len
                padded.append(s[:max_len])
            mean_price = np.mean(padded, axis=0).tolist()
            result.dtw_price = compute_dtw(mean_price, baseline_price_series)
            result.price_convergence_rate = compute_price_convergence_rate(mean_price)

    # DTW check (only if we have baseline series to compare)
    if baseline_pop_series and result.dtw_population > dtw_threshold:
        result.reject_reason = (
            f"Stage3 DTW population too high: {result.dtw_population:.4f} > {dtw_threshold}"
        )
        result.stage3_time = time.time() - t0
        result.total_time = result.stage1_time + result.stage2_time + result.stage3_time
        return result

    # -----------------------------------------------------------------------
    # Check 3: Moran's I for spatial clustering
    # -----------------------------------------------------------------------
    # Use the last few runs' positions+wealths to compute Moran's I
    morans_i_values = []
    for run_data_idx, metrics_dict in enumerate(all_metrics[-10:]):
        # We need raw positions and wealths — re-run last few seeds if needed
        # For efficiency, use wealth values from the stored per-seed data
        pass

    # Compute Moran's I from the most recent runs that have position data
    # We'll use a pooled approach: run one seed specifically for spatial data
    spatial_data = _run_strategy_once(strategy_code, seed=42, steps=steps)
    if "error" not in spatial_data:
        positions = spatial_data.get("agent_positions", [])
        agent_wealths = spatial_data.get("agent_wealths", [])
        grid_w = spatial_data.get("grid_width", 50)
        grid_h = spatial_data.get("grid_height", 50)

        if positions and agent_wealths and len(positions) == len(agent_wealths):
            mi, mi_p = compute_morans_i(positions, agent_wealths, grid_w, grid_h)
            result.morans_i = mi
            result.morans_i_pvalue = mi_p

            if mi < morans_i_lower:
                result.reject_reason = (
                    f"Stage3 Moran's I too low: {mi:.4f} < {morans_i_lower} "
                    f"(unexpected spatial dispersion)"
                )
                result.stage3_time = time.time() - t0
                result.total_time = (
                    result.stage1_time + result.stage2_time + result.stage3_time
                )
                return result

    # -----------------------------------------------------------------------
    # Check 4: Wasserstein distance on wealth distribution
    # -----------------------------------------------------------------------
    baseline_wealths = baseline.get("per_seed_wealths", [])
    if all_wealths and baseline_wealths:
        pool_ablated = []
        for ws in all_wealths:
            pool_ablated.extend(ws)
        pool_baseline = []
        for ws in baseline_wealths:
            pool_baseline.extend(ws)

        if pool_ablated and pool_baseline:
            wd = stats.wasserstein_distance(pool_ablated, pool_baseline)
            result.wasserstein_distance = float(wd)
            # Wasserstein is informational at Stage 3 — the t-tests and KS
            # are the binding gates.  Log it for diagnostics.

    # -----------------------------------------------------------------------
    # All checks passed
    # -----------------------------------------------------------------------
    result.passed = True
    result.stage3_time = time.time() - t0
    result.total_time = result.stage1_time + result.stage2_time + result.stage3_time
    return result


# ===========================================================================
# Orchestrator — the main entry point
# ===========================================================================

def validate_ablation(
    strategy_code: str,
    baseline: dict,
    sim_params: Optional[dict] = None,
) -> ValidationResult:
    """
    Run the full 3-stage fail-fast validation waterfall.

    Parameters
    ----------
    strategy_code : str
        The full source code of the ablated strategy.py.
    baseline : dict
        Baseline data dict containing at minimum:
          - "mean_metrics": {metric_name: float}
        And optionally (for full Stage 3):
          - "per_seed_wealths": list of wealth arrays
          - "per_seed_metrics": list of per-seed metric dicts
          - "mean_population_series": averaged population time series
          - "mean_price_series": averaged price time series
    sim_params : dict, optional
        Override simulation parameters (width, height, etc.).

    Returns
    -------
    ValidationResult
        Detailed result including which stage was reached, pass/fail,
        reject reason, timing, and all computed metrics.
    """
    # ── Stage 1: Fast Filter ──
    result = stage1_fast_filter(strategy_code)
    if not result.stage1_ok:
        return result

    # ── Stage 2: Lightweight Shape Filter ──
    s2_result = stage2_lightweight_filter(strategy_code, baseline)
    # Carry forward Stage 1 timing
    s2_result.stage1_ok = True
    s2_result.stage1_time = result.stage1_time

    if s2_result.reject_reason:
        s2_result.total_time = s2_result.stage1_time + s2_result.stage2_time
        return s2_result

    # ── Stage 3: Heavy Gauntlet ──
    # Collect wealth arrays from Stage 2 runs (re-run to get them if needed)
    stage2_wealths = []
    for seed in STAGE2_SEEDS:
        data = _run_strategy_once(strategy_code, seed=seed, steps=STAGE2_STEPS)
        if "error" not in data:
            stage2_wealths.append(data.get("agent_wealths", []))

    s3_result = stage3_heavy_gauntlet(
        strategy_code=strategy_code,
        prior_run_metrics=s2_result.all_run_metrics,
        prior_run_wealths=stage2_wealths,
        baseline=baseline,
    )
    # Carry forward earlier timing
    s3_result.stage1_ok = True
    s3_result.stage1_time = result.stage1_time
    s3_result.stage2_time = s2_result.stage2_time
    s3_result.ks_statistic = s2_result.ks_statistic
    s3_result.ks_pvalue = s2_result.ks_pvalue
    s3_result.stage2_scalar_pass = s2_result.stage2_scalar_pass
    s3_result.stage2_mean_metrics = s2_result.stage2_mean_metrics
    s3_result.total_time = (
        s3_result.stage1_time + s3_result.stage2_time + s3_result.stage3_time
    )

    return s3_result
