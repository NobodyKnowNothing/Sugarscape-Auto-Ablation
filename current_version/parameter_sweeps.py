"""
Sugarscape Parameter Sweep Validation — Epstein & Axtell (1996)

Implements the four canonical parameter sweeps from "Growing Artificial
Societies" to validate that the model faithfully reproduces the original
paper's key findings.

Statistical analysis follows the book's methodology:
  - Multiple independent runs per condition (default 10)
  - Mean ± SEM (standard error of the mean) for each metric
  - Two-sample t-tests or Mann-Whitney U tests for group comparisons
  - Effect size (Cohen's d) for practical significance
  - Time-series analysis for population/trade dynamics

Sweep A: Trade vs. No Trade (baseline comparison)
Sweep B: Vision Range Sensitivity  
Sweep C: Carrying Capacity & Population Density
Sweep D: Resource Regrowth Scarcity
"""

import json
import math
import sys
import time
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SWEEP_RUNS = 10           # runs per condition (matches book's methodology)
SIM_STEPS = 200
STRATEGY_FILE = Path(__file__).parent / "strategy.py"


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        print(f"[{ts}] {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"[{ts}] {msg.encode('ascii', 'replace').decode()}", flush=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_strategy_with_params(params: dict, seed: int) -> dict:
    """Run strategy.py with given parameters and seed."""
    spec = importlib.util.spec_from_file_location("_sweep_strategy", str(STRATEGY_FILE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    model = mod.create_model(seed=seed, **params)
    data = mod.run_model(model, params.get("steps", SIM_STEPS))

    # Clean up
    if "_sweep_strategy" in sys.modules:
        del sys.modules["_sweep_strategy"]

    return data


def run_condition(params: dict, n_runs: int = N_SWEEP_RUNS, label: str = "") -> dict:
    """
    Run a single experimental condition n_runs times.
    Returns aggregated metrics with mean, std, SEM, and all individual runs.
    """
    from metrics import compute_all_metrics

    all_metrics = []
    all_data = []

    for i in range(n_runs):
        seed = 42 + i
        try:
            data = run_strategy_with_params(params, seed)
            metrics = compute_all_metrics(data)
            all_metrics.append(metrics)
            all_data.append(data)
        except Exception as e:
            log(f"  Run {i+1}/{n_runs} failed: {e}")
            continue

    if not all_metrics:
        return {"error": "All runs failed", "label": label}

    metric_names = list(all_metrics[0].keys())
    result = {"label": label, "params": params, "n_runs": len(all_metrics), "runs": all_metrics}

    for name in metric_names:
        values = [m[name] for m in all_metrics]
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        result[f"{name}_sem"] = result[f"{name}_std"] / math.sqrt(len(values))
        result[f"{name}_values"] = values

    result["mean_metrics"] = {name: result[f"{name}_mean"] for name in metric_names}
    
    # Check against Mesa Baseline for this specific condition
    from prepare import evaluate_mesa_baseline
    from metrics import check_within_bounds, compute_similarity_matrix, format_comparison_report
    import hashlib, io, sys
    
    cache_key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    scratch_dir = Path(__file__).parent / "scratch"
    scratch_dir.mkdir(exist_ok=True)
    cache_path = scratch_dir / f"mesa_sweep_{cache_key}.json"
    
    if cache_path.exists():
        with open(cache_path) as f:
            mesa_means = json.load(f)
    else:
        log(f"    (Caching Mesa baseline for {label}...)")
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            mesa_res = evaluate_mesa_baseline(n_runs=n_runs, params=params)
        finally:
            sys.stdout = old_stdout
        mesa_means = mesa_res["mean_metrics"]
        with open(cache_path, "w") as f:
            json.dump(mesa_means, f)
            
    passes, details = check_within_bounds(mesa_means, result["mean_metrics"])
    result["matches_mesa"] = passes
    if not passes:
        log(f"  ❌ {label} diverges from Mesa baseline!")
        sim = compute_similarity_matrix(mesa_means, result["mean_metrics"])
        rep = format_comparison_report(mesa_means, result["mean_metrics"], sim, details)
        for line in rep.split('\n'):
            log("    " + line)
    else:
        log(f"  ✅ {label} matches Mesa baseline!")

    return result


# ---------------------------------------------------------------------------
# Statistical Tests (following Epstein & Axtell methodology)
# ---------------------------------------------------------------------------

def cohens_d(group1: list, group2: list) -> float:
    """Cohen's d effect size for two independent groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (m1 - m2) / pooled_std


def compare_conditions(cond_a: dict, cond_b: dict, metric: str) -> dict:
    """
    Statistical comparison between two conditions for a given metric.
    Uses Welch's t-test (unequal variances) and Mann-Whitney U.
    Returns test statistics, p-values, and effect size.
    """
    vals_a = cond_a.get(f"{metric}_values", [])
    vals_b = cond_b.get(f"{metric}_values", [])

    if len(vals_a) < 2 or len(vals_b) < 2:
        return {"error": "Insufficient data"}

    # Welch's t-test (parametric)
    t_stat, t_pval = stats.ttest_ind(vals_a, vals_b, equal_var=False)

    # Mann-Whitney U test (non-parametric, as used in the book for
    # non-normal distributions like wealth)
    try:
        u_stat, u_pval = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
    except ValueError:
        u_stat, u_pval = 0.0, 1.0

    # Effect size
    d = cohens_d(vals_a, vals_b)

    return {
        "metric": metric,
        "group_a": {"label": cond_a["label"], "mean": np.mean(vals_a), "std": np.std(vals_a, ddof=1), "n": len(vals_a)},
        "group_b": {"label": cond_b["label"], "mean": np.mean(vals_b), "std": np.std(vals_b, ddof=1), "n": len(vals_b)},
        "welch_t": {"t_statistic": float(t_stat), "p_value": float(t_pval)},
        "mann_whitney_u": {"u_statistic": float(u_stat), "p_value": float(u_pval)},
        "cohens_d": float(d),
        "significant_005": t_pval < 0.05,
        "significant_001": t_pval < 0.01,
    }


def format_comparison(comp: dict) -> str:
    """Format a statistical comparison as a readable string."""
    if "error" in comp:
        return f"  {comp.get('metric', '?')}: ERROR — {comp['error']}"

    ga = comp["group_a"]
    gb = comp["group_b"]
    sig = "***" if comp["significant_001"] else ("**" if comp["significant_005"] else "n.s.")

    return (
        f"  {comp['metric']:24s}  "
        f"{ga['label']}: {ga['mean']:.4f}±{ga['std']:.4f}  vs  "
        f"{gb['label']}: {gb['mean']:.4f}±{gb['std']:.4f}  "
        f"| t={comp['welch_t']['t_statistic']:+.3f} p={comp['welch_t']['p_value']:.4f} "
        f"d={comp['cohens_d']:+.3f} {sig}"
    )


# ---------------------------------------------------------------------------
# Default Parameters (matches prepare.py)
# ---------------------------------------------------------------------------

BASE_PARAMS = {
    "width": 50,
    "height": 50,
    "initial_population": 200,
    "endowment_min": 25,
    "endowment_max": 50,
    "metabolism_min": 1,
    "metabolism_max": 5,
    "vision_min": 1,
    "vision_max": 5,
    "enable_trade": True,
    "steps": SIM_STEPS,
}


# ---------------------------------------------------------------------------
# Sweep A: Trade vs. No Trade
# ---------------------------------------------------------------------------

def sweep_a_trade(n_runs: int = N_SWEEP_RUNS) -> dict:
    """
    Sweep A: Trade vs. No Trade (The Baseline Comparison)

    Paper prediction:
      - With trade: higher carrying capacity, higher Gini
      - Without trade: lower Gini, faster/lower population collapse
    """
    log("=========================================================")
    log("|  Sweep A: Trade vs. No Trade                        |")
    log("=========================================================")

    params_trade = {**BASE_PARAMS, "enable_trade": True}
    params_no_trade = {**BASE_PARAMS, "enable_trade": False}

    log("Running TRADE condition...")
    cond_trade = run_condition(params_trade, n_runs, label="Trade")

    log("Running NO-TRADE condition...")
    cond_no_trade = run_condition(params_no_trade, n_runs, label="No Trade")

    # Statistical comparisons for key metrics
    comparisons = {}
    for metric in ["gini_coefficient", "final_population", "survival_rate",
                   "trade_volume", "wealth_cv"]:
        comparisons[metric] = compare_conditions(cond_trade, cond_no_trade, metric)

    # Validate paper predictions
    validations = {}

    # 1. Trade should yield higher carrying capacity
    pop_comp = comparisons["final_population"]
    if "error" not in pop_comp:
        validations["higher_capacity_with_trade"] = (
            pop_comp["group_a"]["mean"] > pop_comp["group_b"]["mean"]
        )

    # 2. No-trade should yield lower Gini
    gini_comp = comparisons["gini_coefficient"]
    if "error" not in gini_comp:
        validations["lower_gini_without_trade"] = (
            gini_comp["group_b"]["mean"] < gini_comp["group_a"]["mean"]
        )

    # 3. Trade volume should be ~0 without trade
    tv_comp = comparisons["trade_volume"]
    if "error" not in tv_comp:
        validations["zero_trade_without_trade"] = (
            tv_comp["group_b"]["mean"] < 1.0
        )
        
    validations["matches_mesa"] = cond_trade.get("matches_mesa", False) and cond_no_trade.get("matches_mesa", False)

    result = {
        "sweep": "A",
        "conditions": {"trade": cond_trade, "no_trade": cond_no_trade},
        "comparisons": comparisons,
        "validations": validations,
        "all_valid": all(validations.values()) if validations else False,
    }

    # Report
    log("\n--- Sweep A Results ---")
    for metric, comp in comparisons.items():
        log(format_comparison(comp))
    log(f"\nValidations: {validations}")
    log(f"Sweep A {'PASSES ✅' if result['all_valid'] else 'FAILS ❌'}")

    return result


# ---------------------------------------------------------------------------
# Sweep B: Vision Range Sensitivity
# ---------------------------------------------------------------------------

def sweep_b_vision(n_runs: int = N_SWEEP_RUNS) -> dict:
    """
    Sweep B: Vision Range Sensitivity Sweep

    Paper prediction:
      - Higher vision → higher resource extraction efficiency
      - Higher vision → higher carrying capacity
      - Higher vision → higher Gini (wealth inequality)
    """
    log("=========================================================")
    log("|  Sweep B: Vision Range Sensitivity                  |")
    log("=========================================================")

    vision_configs = [
        {"label": "Vision [1,5]",  "vision_min": 1, "vision_max": 5},
        {"label": "Vision [1,10]", "vision_min": 1, "vision_max": 10},
        {"label": "Vision [1,15]", "vision_min": 1, "vision_max": 15},
    ]

    conditions = []
    for vc in vision_configs:
        params = {**BASE_PARAMS, "vision_min": vc["vision_min"], "vision_max": vc["vision_max"]}
        log(f"Running {vc['label']}...")
        cond = run_condition(params, n_runs, label=vc["label"])
        conditions.append(cond)

    # Pairwise comparisons: [1,5] vs [1,10] and [1,5] vs [1,15]
    comparisons = {}
    for i in range(1, len(conditions)):
        key = f"{conditions[0]['label']}_vs_{conditions[i]['label']}"
        comparisons[key] = {}
        for metric in ["gini_coefficient", "final_population", "survival_rate", "wealth_cv"]:
            comparisons[key][metric] = compare_conditions(conditions[0], conditions[i], metric)

    # Monotonicity check: population and Gini should increase with vision
    pop_means = [c["final_population_mean"] for c in conditions]
    gini_means = [c["gini_coefficient_mean"] for c in conditions]

    validations = {
        "population_increases": all(pop_means[i] <= pop_means[i+1] for i in range(len(pop_means)-1)),
        "gini_increases": all(gini_means[i] <= gini_means[i+1] for i in range(len(gini_means)-1)),
    }

    # Trend test: Spearman correlation between vision_max and metrics
    vision_maxes = [5, 10, 15]
    for metric_name, means in [("population", pop_means), ("gini", gini_means)]:
        if len(means) >= 3:
            rho, p = stats.spearmanr(vision_maxes, means)
            validations[f"{metric_name}_trend_rho"] = float(rho)
            validations[f"{metric_name}_trend_p"] = float(p)

    validations["matches_mesa"] = all(c.get("matches_mesa", False) for c in conditions)

    result = {
        "sweep": "B",
        "conditions": conditions,
        "comparisons": comparisons,
        "validations": validations,
        "all_valid": validations.get("population_increases", False) and validations.get("gini_increases", False) and validations.get("matches_mesa", False),
    }

    log("\n--- Sweep B Results ---")
    for vc, cond in zip(vision_configs, conditions):
        log(f"  {vc['label']:20s}  Pop={cond['final_population_mean']:.1f}±{cond['final_population_sem']:.1f}  "
            f"Gini={cond['gini_coefficient_mean']:.4f}±{cond['gini_coefficient_sem']:.4f}")
    log(f"\nValidations: {validations}")
    log(f"Sweep B {'PASSES ✅' if result['all_valid'] else 'FAILS ❌'}")

    return result


# ---------------------------------------------------------------------------
# Sweep C: Carrying Capacity & Population Density
# ---------------------------------------------------------------------------

def sweep_c_density(n_runs: int = N_SWEEP_RUNS) -> dict:
    """
    Sweep C: Carrying Capacity & Population Density Sweep

    Paper prediction:
      - Low density (100): low trade volume, high survival
      - Medium density (200): baseline behavior
      - High density (300): high early trade, sharp collapse to carrying capacity (~60-70)
    """
    log("=========================================================")
    log("|  Sweep C: Population Density                        |")
    log("=========================================================")

    density_configs = [
        {"label": "Pop=100 (low)",    "initial_population": 100},
        {"label": "Pop=200 (medium)", "initial_population": 200},
        {"label": "Pop=300 (high)",   "initial_population": 300},
    ]

    conditions = []
    for dc in density_configs:
        params = {**BASE_PARAMS, "initial_population": dc["initial_population"]}
        log(f"Running {dc['label']}...")
        cond = run_condition(params, n_runs, label=dc["label"])
        conditions.append(cond)

    # Pairwise comparisons
    comparisons = {}
    for i in range(len(conditions)):
        for j in range(i+1, len(conditions)):
            key = f"{conditions[i]['label']}_vs_{conditions[j]['label']}"
            comparisons[key] = {}
            for metric in ["final_population", "trade_volume", "survival_rate", "gini_coefficient"]:
                comparisons[key][metric] = compare_conditions(conditions[i], conditions[j], metric)

    pop_means = [c["final_population_mean"] for c in conditions]
    surv_means = [c["survival_rate_mean"] for c in conditions]
    trade_means = [c["trade_volume_mean"] for c in conditions]

    validations = {
        # Survival should decrease with density (competition)
        "survival_decreases_with_density": surv_means[0] > surv_means[2],
        # Final population should converge (not scale linearly with initial)
        "carrying_capacity_convergence": (
            pop_means[2] < 300 * 0.5  # high density collapses well below start
        ),
        # Trade volume should be higher with more agents (more partners available)
        "trade_scales_with_density": trade_means[2] > trade_means[0],
    }

    validations["matches_mesa"] = all(c.get("matches_mesa", False) for c in conditions)

    result = {
        "sweep": "C",
        "conditions": conditions,
        "comparisons": comparisons,
        "validations": validations,
        "all_valid": all(validations.values()),
    }

    log("\n--- Sweep C Results ---")
    for dc, cond in zip(density_configs, conditions):
        log(f"  {dc['label']:22s}  Pop={cond['final_population_mean']:.1f}±{cond['final_population_sem']:.1f}  "
            f"Surv={cond['survival_rate_mean']:.4f}  Trade={cond['trade_volume_mean']:.0f}")
    log(f"\nValidations: {validations}")
    log(f"Sweep C {'PASSES ✅' if result['all_valid'] else 'FAILS ❌'}")

    return result


# ---------------------------------------------------------------------------
# Sweep D: Resource Regrowth Scarcity
# ---------------------------------------------------------------------------

def sweep_d_scarcity(n_runs: int = N_SWEEP_RUNS) -> dict:
    """
    Sweep D: Resource Regrowth Scarcity Sweep

    Paper prediction:
      - High metabolism / slow regrowth → faster starvation
      - Survival drops dramatically
      - Gini decreases (no one accumulates wealth)

    Since our strategy.py doesn't expose a regrowth_rate parameter directly,
    we test scarcity by raising metabolisms to [3,7] (effectively the same
    as halving regrowth — agents consume faster relative to supply).
    """
    log("=========================================================")
    log("|  Sweep D: Resource Scarcity (High Metabolism)       |")
    log("=========================================================")

    scarcity_configs = [
        {"label": "Metabolism [1,5] (baseline)", "metabolism_min": 1, "metabolism_max": 5},
        {"label": "Metabolism [3,7] (scarce)",   "metabolism_min": 3, "metabolism_max": 7},
    ]

    conditions = []
    for sc in scarcity_configs:
        params = {**BASE_PARAMS, "metabolism_min": sc["metabolism_min"], "metabolism_max": sc["metabolism_max"]}
        log(f"Running {sc['label']}...")
        cond = run_condition(params, n_runs, label=sc["label"])
        conditions.append(cond)

    comparisons = {}
    for metric in ["gini_coefficient", "final_population", "survival_rate",
                   "trade_volume", "wealth_cv"]:
        comparisons[metric] = compare_conditions(conditions[0], conditions[1], metric)

    surv_baseline = conditions[0]["survival_rate_mean"]
    surv_scarce = conditions[1]["survival_rate_mean"]
    gini_baseline = conditions[0]["gini_coefficient_mean"]
    gini_scarce = conditions[1]["gini_coefficient_mean"]

    validations = {
        # Survival drops dramatically under scarcity
        "survival_drops": surv_scarce < surv_baseline,
        # Gini decreases (no one accumulates)
        "gini_decreases": gini_scarce <= gini_baseline,
        # Population drops
        "population_drops": (
            conditions[1]["final_population_mean"] < conditions[0]["final_population_mean"]
        ),
    }

    validations["matches_mesa"] = all(c.get("matches_mesa", False) for c in conditions)

    result = {
        "sweep": "D",
        "conditions": conditions,
        "comparisons": comparisons,
        "validations": validations,
        "all_valid": all(validations.values()),
    }

    log("\n--- Sweep D Results ---")
    for comp_name, comp in comparisons.items():
        log(format_comparison(comp))
    log(f"\nValidations: {validations}")
    log(f"Sweep D {'PASSES ✅' if result['all_valid'] else 'FAILS ❌'}")

    return result


# ---------------------------------------------------------------------------
# Full Sweep Suite
# ---------------------------------------------------------------------------

def run_all_sweeps(n_runs: int = N_SWEEP_RUNS) -> dict:
    """Run all four parameter sweeps and produce a consolidated report."""
    log("=============================================================")
    log("|  Sugarscape Parameter Sweep Validation Suite            |")
    log("|  Epstein & Axtell 'Growing Artificial Societies' (1996) |")
    log("=============================================================")

    t0 = time.time()

    results = {}
    results["A"] = sweep_a_trade(n_runs)
    results["B"] = sweep_b_vision(n_runs)
    results["C"] = sweep_c_density(n_runs)
    results["D"] = sweep_d_scarcity(n_runs)

    dt = time.time() - t0

    all_valid = all(r["all_valid"] for r in results.values())

    log("\n" + "=" * 60)
    log("CONSOLIDATED SWEEP RESULTS")
    log("=" * 60)
    for key, r in results.items():
        status = "✅ PASS" if r["all_valid"] else "❌ FAIL"
        log(f"  Sweep {key}: {status}")
    log(f"\nOverall: {'✅ ALL SWEEPS PASS' if all_valid else '❌ SOME SWEEPS FAIL'}")
    log(f"Total time: {dt:.1f}s")

    # Save results
    output_path = Path(__file__).parent / "sweep_results.json"
    serializable = _make_serializable(results)
    serializable["timestamp"] = datetime.now().isoformat()
    serializable["total_time_seconds"] = dt
    serializable["all_valid"] = all_valid

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    log(f"Results saved to {output_path}")

    return results


def _make_serializable(obj):
    """Convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sugarscape Parameter Sweep Validation")
    parser.add_argument("--sweep", type=str, default="all",
                        choices=["all", "A", "B", "C", "D"],
                        help="Which sweep to run (default: all)")
    parser.add_argument("--runs", type=int, default=N_SWEEP_RUNS,
                        help=f"Runs per condition (default: {N_SWEEP_RUNS})")
    args = parser.parse_args()

    if args.sweep == "all":
        run_all_sweeps(args.runs)
    elif args.sweep == "A":
        sweep_a_trade(args.runs)
    elif args.sweep == "B":
        sweep_b_vision(args.runs)
    elif args.sweep == "C":
        sweep_c_density(args.runs)
    elif args.sweep == "D":
        sweep_d_scarcity(args.runs)
