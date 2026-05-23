"""
Sugarscape Baseline Runner â€” Fixed Evaluation (DO NOT MODIFY)

This is the Karpathy-pattern 'prepare.py': it runs the CANONICAL Mesa 
Sugarscape G1MT model and collects ground-truth metrics. The agent
(LLM) cannot modify this file.

GROUND TRUTH = Mesa's built-in Sugarscape G1MT implementation.
This is the accepted faithful reproduction of Epstein & Axtell (1996).

The ablation compares variant strategy.py against MESA's output, not
against its own previous output. This ensures we are matching the
original paper's implementation, not a self-referential copy.

Usage:
    # Run baseline calibration against Mesa canonical:
    python prepare.py
    
    # Compare a strategy variant against Mesa baseline:
    python prepare.py --compare strategy.py
"""

import json
import math
import os
import sys
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
import time
import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

GRID_WIDTH = 50
GRID_HEIGHT = 50
DEFAULT_STEPS = 200
N_RUNS = 10                 # statistical runs per evaluation
INITIAL_POPULATION = 200
SEED_BASE = 42

DEFAULT_PARAMS = {
    "width": GRID_WIDTH,
    "height": GRID_HEIGHT,
    "initial_population": INITIAL_POPULATION,
    "endowment_min": 25,
    "endowment_max": 50,
    "metabolism_min": 1,
    "metabolism_max": 5,
    "vision_min": 1,
    "vision_max": 5,
    "enable_trade": True,
    "steps": DEFAULT_STEPS,
}

SUGAR_MAP_FILE = Path(__file__).parent / "sugar-map.txt"
BASELINE_METRICS_FILE = Path(__file__).parent / "baseline_metrics.json"


# ---------------------------------------------------------------------------
# Mesa Canonical Ground Truth
# ---------------------------------------------------------------------------

def run_mesa_canonical(params: dict, seed: int) -> dict:
    """
    Run Mesa's built-in Sugarscape G1MT â€” THE canonical implementation.
    This is the ground truth against which all ablations are compared.
    
    Returns the standardized metrics_data dict.
    """
    from mesa.examples.advanced.sugarscape_g1mt.model import SugarscapeG1mt
    
    model = SugarscapeG1mt(
        width=params["width"],
        height=params["height"],
        initial_population=params["initial_population"],
        endowment_min=params["endowment_min"],
        endowment_max=params["endowment_max"],
        metabolism_min=params["metabolism_min"],
        metabolism_max=params["metabolism_max"],
        vision_min=params["vision_min"],
        vision_max=params["vision_max"],
        enable_trade=params["enable_trade"],
    )
    
    steps = params.get("steps", DEFAULT_STEPS)
    
    try:
        model.run_model(step_count=steps)
    except (ValueError, IndexError) as e:
        # Mesa has occasional edge-case bugs (e.g., agent surrounded by occupied cells)
        # If it crashes, return an error so the caller can retry with a different seed
        return {"error": f"Mesa crash: {e}"}
    
    # Collect metrics from Mesa's datacollector (cumulative across all steps)
    df = model.datacollector.get_model_vars_dataframe()
    cumul_trade_volume = int(df['Trade Volume'].sum())
    
    # Agent-level data from final state
    agents = list(model.agents)
    agent_wealths = [float(a.sugar + a.spice) for a in agents]
    
    # Final-step trade prices (agents reset prices each step, so these are current-step only)
    # For cumulative price data we'd need to track across steps â€” use final step prices
    # as representative of the converged market price
    trade_prices = []
    for a in agents:
        if hasattr(a, 'prices'):
            trade_prices.extend(a.prices)
    
    agent_positions = []
    for a in agents:
        if hasattr(a, 'cell') and a.cell is not None:
            agent_positions.append(a.cell.coordinate)
    
    return {
        "agent_wealths": agent_wealths,
        "final_population": len(agents),
        "initial_population": params["initial_population"],
        "trade_prices": trade_prices,
        "trade_volume": cumul_trade_volume,
        "agent_positions": agent_positions,
        "grid_width": params["width"],
        "grid_height": params["height"],
        "steps_run": steps,
    }


def run_strategy_variant(strategy_path: str | Path, params: dict, seed: int) -> dict:
    """
    Run a strategy variant from a Python file.
    The file must export:
        create_model(seed, **params) -> model object
        run_model(model, steps) -> metrics_dict
    """
    spec = importlib.util.spec_from_file_location("strategy", str(strategy_path))
    strategy = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(strategy)
    except Exception as e:
        return {"error": f"Strategy import failed: {e}"}
    
    if not hasattr(strategy, 'create_model') or not hasattr(strategy, 'run_model'):
        return {"error": "Strategy must export create_model() and run_model()"}
    
    try:
        model = strategy.create_model(seed=seed, **params)
        result = strategy.run_model(model, params.get("steps", DEFAULT_STEPS))
        return result
    except Exception as e:
        return {"error": f"Strategy execution failed: {e}"}


# ---------------------------------------------------------------------------
# Multi-run evaluation
# ---------------------------------------------------------------------------

def evaluate_mesa_baseline(n_runs: int = N_RUNS, params: dict | None = None) -> dict:
    """
    Run the MESA canonical model n_runs times and compute averaged metrics.
    Uses crash-resilient seeding â€” skips failed seeds and finds enough good ones.
    """
    from metrics import compute_all_metrics
    
    params = params or DEFAULT_PARAMS
    all_metrics = []
    seed = SEED_BASE
    attempts = 0
    max_attempts = n_runs * 3  # allow retries for Mesa edge-case crashes
    
    while len(all_metrics) < n_runs and attempts < max_attempts:
        attempts += 1
        print(f"  Mesa canonical run {len(all_metrics)+1}/{n_runs} (seed={seed})...", end=" ", flush=True)
        t0 = time.time()
        
        data = run_mesa_canonical(params, seed)
        seed += 1
        
        if "error" in data:
            print(f"SKIP ({data['error']})")
            continue
        
        metrics = compute_all_metrics(data)
        all_metrics.append(metrics)
        dt = time.time() - t0
        print(f"done ({dt:.1f}s) pop={data['final_population']}")
    
    if not all_metrics:
        print("FATAL: No Mesa runs completed successfully")
        sys.exit(1)
    
    metric_names = list(all_metrics[0].keys())
    result = {"runs": all_metrics}
    
    for name in metric_names:
        values = [m[name] for m in all_metrics]
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values))
    
    result["mean_metrics"] = {name: result[f"{name}_mean"] for name in metric_names}
    return result


def evaluate_variant(strategy_path: str | Path, n_runs: int = N_RUNS, 
                     params: dict | None = None) -> dict:
    """Run a strategy variant n_runs times and compute averaged metrics."""
    from metrics import compute_all_metrics
    
    params = params or DEFAULT_PARAMS
    all_metrics = []
    errors = []
    
    for i in range(n_runs):
        seed = SEED_BASE + i
        print(f"  Variant run {i+1}/{n_runs} (seed={seed})...", end=" ", flush=True)
        t0 = time.time()
        
        data = run_strategy_variant(strategy_path, params, seed)
        
        if "error" in data:
            errors.append(data["error"])
            print(f"FAILED: {data['error']}")
            continue
        
        metrics = compute_all_metrics(data)
        all_metrics.append(metrics)
        dt = time.time() - t0
        print(f"done ({dt:.1f}s)")
    
    if not all_metrics:
        return {"error": "; ".join(errors), "runs": []}
    
    metric_names = list(all_metrics[0].keys())
    result = {"runs": all_metrics, "errors": errors}
    
    for name in metric_names:
        values = [m[name] for m in all_metrics]
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values))
    
    result["mean_metrics"] = {name: result[f"{name}_mean"] for name in metric_names}
    return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_variant_to_baseline(baseline_result: dict, variant_result: dict) -> dict:
    """Compare a variant's metrics against the Mesa canonical baseline."""
    from metrics import (compute_similarity_matrix, check_within_bounds, 
                         format_comparison_report)
    
    baseline_means = baseline_result["mean_metrics"]
    variant_means = variant_result["mean_metrics"]
    
    similarity = compute_similarity_matrix(baseline_means, variant_means)
    passes, details = check_within_bounds(baseline_means, variant_means)
    report = format_comparison_report(baseline_means, variant_means, similarity, details)
    
    return {
        "passes": passes,
        "similarity": similarity,
        "details": details,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sugarscape baseline evaluation")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to strategy.py variant to compare against Mesa baseline")
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help="Number of statistical runs per evaluation")
    parser.add_argument("--force-recalibrate", action="store_true",
                        help="Force re-run of Mesa baseline even if cached")
    args = parser.parse_args()
    
    print("â•â•â• Sugarscape Baseline â€” Mesa Canonical Ground Truth â•â•â•")
    print(f"Source: mesa.examples.advanced.sugarscape_g1mt")
    print(f"Parameters: {json.dumps(DEFAULT_PARAMS, indent=2)}")
    print()
    
    # Compute or load baseline from Mesa canonical
    if BASELINE_METRICS_FILE.exists() and not args.force_recalibrate:
        print(f"Loading cached Mesa baseline from {BASELINE_METRICS_FILE}")
        with open(BASELINE_METRICS_FILE) as f:
            baseline = json.load(f)
    else:
        print("Running Mesa canonical Sugarscape G1MT for baseline calibration...")
        baseline = evaluate_mesa_baseline(n_runs=args.runs)
        
        # Cache
        cache = {
            "source": "mesa.examples.advanced.sugarscape_g1mt",
            "n_runs": args.runs,
            "mean_metrics": baseline["mean_metrics"],
        }
        for name in baseline["mean_metrics"]:
            cache[f"{name}_std"] = baseline.get(f"{name}_std", 0.0)
        
        with open(BASELINE_METRICS_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"Saved Mesa baseline to {BASELINE_METRICS_FILE}")
    
    from metrics import format_metrics_report
    print()
    print(format_metrics_report(baseline["mean_metrics"]))
    
    # Compare if requested
    if args.compare:
        print(f"\nâ•â•â• Evaluating Variant: {args.compare} â•â•â•")
        variant = evaluate_variant(args.compare, n_runs=args.runs)
        
        if "error" in variant and not variant.get("runs"):
            print(f"FATAL: {variant['error']}")
            sys.exit(1)
        
        comparison = compare_variant_to_baseline(baseline, variant)
        print()
        print(comparison["report"])
        print()
        
        if comparison["passes"]:
            print("âœ… VARIANT PASSES â€” within error bounds of Mesa canonical implementation")
        else:
            print("âŒ VARIANT FAILS â€” diverges from Mesa canonical implementation")
        
        sys.exit(0 if comparison["passes"] else 1)

