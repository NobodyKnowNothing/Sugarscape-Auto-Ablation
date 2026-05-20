#!/usr/bin/env python3
"""
Sugarscape Structural Ablation — Autoresearch Pipeline

Adapts Karpathy's autoresearch ratchet loop to perform structural ablation
on the Sugarscape agent-based model. The LLM (Gemma) proposes simplifications
to strategy.py; if the emergent metrics from Epstein & Axtell (1996) remain
within tight error bounds, the simplification is committed. Otherwise reverted.

Architecture (3-file Karpathy pattern):
  - prepare.py  (FIXED) — baseline runner, evaluation harness
  - strategy.py (MUTABLE) — the code the LLM simplifies
  - program.md  (HUMAN)  — research agenda for the LLM

Usage:
    export GOOGLE_API_KEY="your_key_here"
    python sugarscape_autoresearch.py

The loop runs indefinitely — kill with Ctrl+C.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import traceback
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "gemma-4-26b-a4b-it"
VARIANTS_PER_GENERATION = 1          # mutations per round (fewer = more careful)
STRATEGY_FILE = Path(__file__).parent / "strategy.py"
AGENT_REPO = Path(__file__).parent / "agent_repo"
AGENT_STRATEGY = AGENT_REPO / "strategy.py"
RESULTS_FILE = Path(__file__).parent / "results.tsv"
PROGRAM_FILE = Path(__file__).parent / "program.md"
BASELINE_METRICS_FILE = Path(__file__).parent / "baseline_metrics.json"

# Simulation config
SIM_STEPS = 200
N_EVAL_RUNS = 3              # statistical runs per variant evaluation

API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        print(f"[{ts}] {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"[{ts}] {msg.encode('ascii', 'replace').decode()}", flush=True)


# ---------------------------------------------------------------------------
# Git helpers (operate on agent_repo)
# ---------------------------------------------------------------------------
def git(cmd: str, cwd: str | None = None) -> str:
    result = subprocess.run(
        f"git {cmd}", shell=True, capture_output=True, text=True,
        cwd=cwd or str(AGENT_REPO),
    )
    return result.stdout.strip()


def git_commit(message: str):
    git("add strategy.py")
    git(f'commit -m "{message}"')


def git_revert_to(commit_hash: str):
    git(f"reset --hard {commit_hash}")


def git_current_hash() -> str:
    return git("rev-parse --short HEAD")


def git_create_branch(name: str):
    existing = git("branch --list " + name)
    if existing.strip():
        git(f"checkout {name}")
    else:
        git(f"checkout -b {name}")


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------
def init_results():
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "generation\tvariant\tcommit\tstatus\tlines\tclasses\tmethods\t"
            "ast_nodes\tcyclomatic\tcombined_score\t"
            "gini\tpopulation\ttrade_price\ttrade_volume\tsurvival\twealth_cv\tentropy\t"
            "passes\tdescription\n"
        )


def append_result(gen, var, commit, status, complexity, metrics, passes, desc):
    with open(RESULTS_FILE, "a") as f:
        f.write(
            f"{gen}\t{var}\t{commit}\t{status}\t"
            f"{complexity.get('lines', 0)}\t{complexity.get('classes', 0)}\t{complexity.get('methods', 0)}\t"
            f"{complexity.get('ast_nodes', 0)}\t{complexity.get('cyclomatic_total', 0)}\t{complexity.get('combined_score', 0):.2f}\t"
            f"{metrics.get('gini_coefficient', 0):.4f}\t"
            f"{metrics.get('final_population', 0):.0f}\t"
            f"{metrics.get('mean_trade_price', 0):.4f}\t"
            f"{metrics.get('trade_volume', 0):.0f}\t"
            f"{metrics.get('survival_rate', 0):.4f}\t"
            f"{metrics.get('wealth_cv', 0):.4f}\t"
            f"{metrics.get('spatial_entropy', 0):.4f}\t"
            f"{'PASS' if passes else 'FAIL'}\t{desc}\n"
        )


def read_results_history(max_lines: int = 50) -> str:
    if RESULTS_FILE.exists():
        lines = RESULTS_FILE.read_text().strip().split("\n")
        if len(lines) <= max_lines + 1:
            return "\n".join(lines)
        header = lines[0]
        recent = lines[-max_lines:]
        return f"{header}\n... ({len(lines) - max_lines - 1} older rows truncated) ...\n" + "\n".join(recent)
    return "(no results yet)"


# ---------------------------------------------------------------------------
# Code complexity metrics (AST + Cyclomatic combined score)
# ---------------------------------------------------------------------------
from complexity import combined_complexity_score, format_complexity_report


def count_complexity(code: str) -> dict:
    """Compute combined AST node + cyclomatic complexity score.
    
    Returns dict with traditional metrics (lines, classes, methods) plus:
      - ast_nodes: total AST node count
      - cyclomatic_total: total cyclomatic complexity
      - combined_score: weighted sum (0.4*AST + 0.6*cyclomatic)
    """
    return combined_complexity_score(code)


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------
def evaluate_strategy(strategy_code: str) -> dict:
    """
    Run a strategy variant and compute its metrics.
    Returns dict with "mean_metrics", "passes", "details", "report", "error".
    """
    import importlib.util
    import tempfile
    
    # Write strategy to temp file
    tmp_path = Path(__file__).parent / "_tmp_strategy.py"
    tmp_path.write_text(strategy_code)
    
    try:
        # Import the strategy
        spec = importlib.util.spec_from_file_location("_tmp_strategy", str(tmp_path))
        strategy_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_mod)
        
        if not hasattr(strategy_mod, 'create_model') or not hasattr(strategy_mod, 'run_model'):
            return {"error": "Missing create_model() or run_model()"}
        
        from metrics import compute_all_metrics
        
        all_metrics = []
        import warnings
        for i in range(N_EVAL_RUNS):
            seed = 42 + i
            model = strategy_mod.create_model(seed=seed, steps=SIM_STEPS,
                                               initial_population=200,
                                               endowment_min=25, endowment_max=50,
                                               metabolism_min=1, metabolism_max=5,
                                               vision_min=1, vision_max=5,
                                               enable_trade=True,
                                               width=50, height=50)
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                data = strategy_mod.run_model(model, SIM_STEPS)
            metrics = compute_all_metrics(data)
            all_metrics.append(metrics)
        
        import numpy as np
        mean_metrics = {}
        for key in all_metrics[0]:
            mean_metrics[key] = float(np.mean([m[key] for m in all_metrics]))
        
        # Compare against baseline
        if BASELINE_METRICS_FILE.exists():
            with open(BASELINE_METRICS_FILE) as f:
                baseline = json.load(f)
            baseline_means = baseline["mean_metrics"]
        else:
            # Use variant's own metrics as baseline (first run)
            baseline_means = mean_metrics
        
        from metrics import compute_similarity_matrix, check_within_bounds, format_comparison_report
        
        similarity = compute_similarity_matrix(baseline_means, mean_metrics)
        passes, details = check_within_bounds(baseline_means, mean_metrics)
        report = format_comparison_report(baseline_means, mean_metrics, similarity, details)
        
        return {
            "mean_metrics": mean_metrics,
            "passes": passes,
            "details": details,
            "similarity": similarity,
            "report": report,
        }
    
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        # Clean up cached module
        if "_tmp_strategy" in sys.modules:
            del sys.modules["_tmp_strategy"]


# ---------------------------------------------------------------------------
# LLM Agent — Gemma via google-genai
# ---------------------------------------------------------------------------
def create_client() -> genai.Client:
    return genai.Client(api_key=API_KEY)


def generate_ablation_variants(
    client: genai.Client,
    current_strategy: str,
    results_history: str,
    baseline_metrics: dict,
    complexity: dict,
    n_variants: int = VARIANTS_PER_GENERATION,
    temperature: float = 0.7,
) -> list[tuple[str, str]]:
    """
    Ask the LLM to generate N structural simplifications of strategy.py.
    Returns list of (code, description) tuples.
    """
    program = PROGRAM_FILE.read_text() if PROGRAM_FILE.exists() else ""
    
    prompt = textwrap.dedent(f"""\
    {program}

    ---

    ## Current strategy.py (the code to simplify):
    ```python
    {current_strategy}
    ```

    ## Current Complexity (Combined AST + Cyclomatic Score):
    - Lines of code: {complexity['lines']}
    - Classes: {complexity['classes']}
    - Methods/Functions: {complexity['methods']}
    - AST Nodes: {complexity['ast_nodes']}
    - Cyclomatic Complexity: {complexity['cyclomatic_total']}
    - **Combined Score: {complexity['combined_score']:.1f}** (LOWER IS BETTER)

    ## Baseline Metrics (target to preserve):
    ```json
    {json.dumps(baseline_metrics, indent=2)}
    ```

    ## Results History (previous ablation attempts):
    ```
    {results_history}
    ```

    ---

    Generate {n_variants} different structural simplifications of strategy.py.
    Each variant should try a DIFFERENT simplification strategy.
    Learn from the results history — if a simplification caused a metric to fail,
    avoid similar changes. If a simplification passed, try pushing further.

    PRIORITY: Reduce the COMBINED COMPLEXITY SCORE (AST nodes + cyclomatic
    complexity) while keeping ALL metrics within error bounds. The combined
    score weights AST structural size (40%) and cyclomatic branching (60%).
    Focus on reducing decision points, nested conditionals, and structural
    depth — not just line count.

    For EACH variant, output:
    1. A brief one-line description of what was simplified
    2. The complete strategy.py code in a ```python code block

    Number them like: **Variant 1:**, **Variant 2:**, etc.
    """)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=16384,
                ),
            )
            
            if not response or not response.text:
                log("LLM returned empty response")
                return []
            
            return parse_variants(response.text, n_variants)
        
        except Exception as e:
            if "500" in str(e) or "ServerError" in type(e).__name__:
                log(f"LLM API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
            log(f"LLM API error: {e}")
            traceback.print_exc()
            return []
    
    return []


def parse_variants(text: str, expected: int) -> list[tuple[str, str]]:
    """Parse LLM output into (code, description) pairs."""
    variants = []
    
    parts = re.split(r'\*\*Variant\s+\d+[:\s]*\*\*', text)
    if len(parts) > 1:
        for part in parts[1:]:
            lines = part.strip().split("\n")
            desc = ""
            for line in lines:
                line = line.strip().strip("*").strip("-").strip()
                if line and not line.startswith("```"):
                    desc = line.replace("\t", " ")[:300]
                    break
            
            code_match = re.search(r'```python\s*\n(.*?)```', part, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                if "def create_model" in code and "def run_model" in code:
                    variants.append((code, desc or "unnamed variant"))
    
    # Fallback
    if not variants:
        blocks = re.split(r'```python', text)
        for i in range(1, len(blocks)):
            pre_text = blocks[i-1]
            code_part = blocks[i]
            
            desc = f"variant {len(variants) + 1}"
            pre_lines = pre_text.strip().split("\n")
            for line in reversed(pre_lines):
                line = line.strip().strip("*").strip("-").strip()
                if line and len(line) > 5:
                    desc = line.replace("\t", " ")[:300]
                    break
            
            code_match = re.search(r'^(.*?)```', code_part, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                if "def create_model" in code and "def run_model" in code:
                    variants.append((code, desc))
    
    return variants[:expected]


# ---------------------------------------------------------------------------
# Baseline calibration
# ---------------------------------------------------------------------------
def calibrate_baseline():
    """
    Calibrate ground truth from Mesa's canonical Sugarscape G1MT.
    
    This ensures all ablation comparisons are against the ORIGINAL
    implementation (Mesa's faithful reproduction of Epstein & Axtell 1996),
    not a self-referential copy of our standalone strategy.py.
    """
    log("╔══════════════════════════════════════════════════╗")
    log("║  Calibrating Against Mesa Canonical Sugarscape  ║")
    log("╚══════════════════════════════════════════════════╝")
    
    if BASELINE_METRICS_FILE.exists():
        with open(BASELINE_METRICS_FILE) as f:
            baseline = json.load(f)
        source = baseline.get("source", "unknown")
        log(f"Loaded cached baseline (source: {source})")
        from metrics import format_metrics_report
        log(format_metrics_report(baseline["mean_metrics"]))
        return baseline
    
    # No cached baseline — run Mesa canonical to establish ground truth
    log("No cached baseline found. Running Mesa canonical Sugarscape G1MT...")
    log("(This runs the ACTUAL canonical implementation, not our strategy.py)")
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "prepare.py", "--force-recalibrate", f"--runs={N_EVAL_RUNS}"],
        cwd=str(Path(__file__).parent),
        capture_output=True, text=True,
    )
    
    if result.returncode != 0:
        log(f"FATAL: Mesa baseline calibration failed:\n{result.stderr}")
        sys.exit(1)
    
    log(result.stdout)
    
    with open(BASELINE_METRICS_FILE) as f:
        baseline = json.load(f)
    
    # Also store initial strategy.py complexity for tracking
    strategy_code = STRATEGY_FILE.read_text()
    baseline["complexity"] = count_complexity(strategy_code)
    with open(BASELINE_METRICS_FILE, "w") as f:
        json.dump(baseline, f, indent=2)
    
    from metrics import format_metrics_report
    log("Mesa canonical baseline established:")
    log(format_metrics_report(baseline["mean_metrics"]))
    
    return baseline


def update_paper_draft(generation: int, commit_hash: str, complexity: dict, result: dict, desc: str):
    """
    Dynamically update the academic paper draft with the latest evolutionary
    ratchet outcomes (complexity tables, metrics comparison, and minimal code).
    """
    paper_path = Path("/home/owenwalker/.gemini/antigravity/brain/fe6f754c-3000-49c5-be59-3d2ca039388f/artifacts/sugarscape_ablation_paper.md")
    if not paper_path.exists():
        return
    
    content = paper_path.read_text()
    
    # 1. Update the Code Complexity Progression Table
    # Find the placeholder <!-- GENERATION_PLACEHOLDER --> and insert the new row before it
    initial_lines = 280
    reduction = 100.0 * (initial_lines - complexity["lines"]) / initial_lines
    
    new_row = (
        f"| **{generation} (Ablated)** | `{commit_hash}` | {complexity['lines']} | "
        f"{complexity['classes']} | {complexity['methods']} | {reduction:.1f}% | {desc} |\n"
        f"<!-- GENERATION_PLACEHOLDER -->"
    )
    content = content.replace("<!-- GENERATION_PLACEHOLDER -->", new_row)
    
    # 2. Update the Comparative Metric Performance Table
    # We read baseline metrics and format comparison rows
    if BASELINE_METRICS_FILE.exists():
        with open(BASELINE_METRICS_FILE) as f:
            baseline = json.load(f)
        b_means = baseline["mean_metrics"]
        v_means = result["mean_metrics"]
        
        # Build the comparative markdown table
        comp_table_lines = [
            f"| **Gini Coefficient** | {b_means.get('gini_coefficient', 0):.4f} | {v_means.get('gini_coefficient', 0):.4f} | {abs(b_means.get('gini_coefficient', 0)-v_means.get('gini_coefficient', 0)):.4f} (abs) | ✅ Passed |",
            f"| **Final Population** | {b_means.get('final_population', 0):.2f} | {v_means.get('final_population', 0):.2f} | {abs(b_means.get('final_population', 0)-v_means.get('final_population', 0))/b_means.get('final_population', 1)*100:.1f}% | ✅ Passed |",
            f"| **Mean Trade Price** | {b_means.get('mean_trade_price', 0):.4f} | {v_means.get('mean_trade_price', 0):.4f} | {abs(b_means.get('mean_trade_price', 0)-v_means.get('mean_trade_price', 0))/b_means.get('mean_trade_price', 1)*100:.1f}% | ✅ Passed |",
            f"| **Trade Volume** | {b_means.get('trade_volume', 0):.2f} | {v_means.get('trade_volume', 0):.2f} | {abs(b_means.get('trade_volume', 0)-v_means.get('trade_volume', 0))/b_means.get('trade_volume', 1)*100:.1f}% | ✅ Passed |",
            f"| **Survival Rate** | {b_means.get('survival_rate', 0):.4f} | {v_means.get('survival_rate', 0):.4f} | {abs(b_means.get('survival_rate', 0)-v_means.get('survival_rate', 0)):.4f} (abs) | ✅ Passed |",
            f"| **Wealth CV** | {b_means.get('wealth_cv', 0):.4f} | {v_means.get('wealth_cv', 0):.4f} | {abs(b_means.get('wealth_cv', 0)-v_means.get('wealth_cv', 0))/b_means.get('wealth_cv', 1)*100:.1f}% | ✅ Passed |",
            f"| **Spatial Entropy** | {b_means.get('spatial_entropy', 0):.4f} | {v_means.get('spatial_entropy', 0):.4f} | {abs(b_means.get('spatial_entropy', 0)-v_means.get('spatial_entropy', 0))/b_means.get('spatial_entropy', 1)*100:.1f}% | ✅ Passed |",
        ]
        
        # Replace the TBD table block with our updated comp table
        tbd_pattern = r"## 4.2. Comparative Metric Performance.*?\n\n"
        table_content = (
            "## 4.2. Comparative Metric Performance\n"
            "Comparison between the **Canonical Mesa G1MT baseline** and the **Final Ablated Model**:\n\n"
            "| Metric | Mesa Canonical Baseline | Final Ablated Model | Relative Error | Status |\n"
            "|---|---|---|---|---|\n" + "\n".join(comp_table_lines) + "\n\n"
        )
        # Find where ## 4.2 starts up to ## 5.
        idx_start = content.find("## 4.2. Comparative Metric Performance")
        idx_end = content.find("## 5. Discussion")
        if idx_start != -1 and idx_end != -1:
            content = content[:idx_start] + table_content + content[idx_end:]
            
    # 3. Update the Appendix with the final ablated code
    strategy_code = STRATEGY_FILE.read_text()
    idx_code = content.find("## Appendix: Minimal Viable Code")
    if idx_code != -1:
        content = content[:idx_code] + (
            "## Appendix: Minimal Viable Code\n"
            "```python\n" + strategy_code + "\n```\n"
        )
        
    paper_path.write_text(content)
    log(f"📝 Updated academic manuscript in {paper_path.name}")


# ---------------------------------------------------------------------------
# Main ablation loop
# ---------------------------------------------------------------------------
def run_ablation_round(generation: int, client: genai.Client, baseline: dict):
    """Execute one round of structural ablation."""
    log(f"\n═══ Generation {generation} ═══")
    
    # Read current strategy
    current_strategy = STRATEGY_FILE.read_text()
    current_complexity = count_complexity(current_strategy)
    current_hash = git_current_hash()
    
    log(format_complexity_report(current_complexity))
    
    # Generate variants
    results_history = read_results_history()
    variants = generate_ablation_variants(
        client, current_strategy, results_history,
        baseline["mean_metrics"], current_complexity,
        n_variants=VARIANTS_PER_GENERATION,
    )
    
    if not variants:
        log("No valid variants generated. Retrying this generation.")
        return False
    
    log(f"Generated {len(variants)} variants")
    
    # Evaluate each variant
    best_variant = None
    best_simplification = 0  # lines reduced
    
    for i, (code, desc) in enumerate(variants):
        var_num = i + 1
        log(f"\n--- Variant {var_num}: {desc[:80]} ---")
        
        var_complexity = count_complexity(code)
        score_delta = current_complexity["combined_score"] - var_complexity["combined_score"]
        log(f"  Combined Score: {var_complexity['combined_score']:.1f} ({score_delta:+.1f}), "
            f"AST={var_complexity['ast_nodes']}, CC={var_complexity['cyclomatic_total']}, "
            f"{var_complexity['lines']} lines")
        
        # Run evaluation
        t0 = time.time()
        result = evaluate_strategy(code)
        dt = time.time() - t0
        
        if "error" in result:
            log(f"  ❌ CRASH: {result['error']}")
            append_result(generation, var_num, current_hash, "crash",
                         var_complexity, {}, False, desc)
            continue
        
        log(f"  Evaluation took {dt:.1f}s")
        log(result["report"])
        
        passes = result["passes"]
        status = "pass" if passes else "fail"
        
        append_result(generation, var_num, current_hash, status,
                     var_complexity, result["mean_metrics"],
                     passes, desc)
        
        if passes:
            log(f"  ✅ PASSES all metric bounds!")
            # Track best passing variant (largest combined score reduction)
            if score_delta > best_simplification:
                best_variant = (code, desc, var_complexity, result)
                best_simplification = score_delta
        else:
            log(f"  ❌ FAILS metric bounds")
    
    # Commit best variant if it's simpler
    if best_variant:
        code, desc, complexity, result = best_variant
        score_saved = current_complexity["combined_score"] - complexity["combined_score"]
        
        if score_saved > 0:
            log(f"\n🎉 RATCHET FORWARD: {desc}")
            log(f"   Score: {current_complexity['combined_score']:.1f} → {complexity['combined_score']:.1f} ({score_saved:+.1f})")
            log(f"   Lines: {current_complexity['lines']} → {complexity['lines']}")
            
            # Write to strategy.py and agent_repo
            STRATEGY_FILE.write_text(code)
            AGENT_STRATEGY.write_text(code)
            git_commit(f"Gen {generation}: {desc[:50]} | score -{score_saved:.1f}")
            
            chash = git_current_hash()
            log(f"   Committed as {chash}")
            
            # Dynamically update the academic paper draft!
            try:
                update_paper_draft(generation, chash, complexity, result, desc)
            except Exception as pe:
                log(f"   ⚠️ Could not update paper draft: {pe}")
            return True
        else:
            log(f"\n⏸️  Variant passes but not simpler — skipping commit")
            return False
    else:
        log(f"\n❌ No passing variants this generation — reverting")
        git_revert_to(current_hash)
        return False


def main():
    log("╔══════════════════════════════════════════════════╗")
    log("║  Sugarscape Structural Ablation Pipeline        ║")
    log(f"║  Model: {MODEL:40s} ║")
    log("║  Pattern: Karpathy Autoresearch Ratchet Loop    ║")
    log("╚══════════════════════════════════════════════════╝")
    
    if not API_KEY:
        log("ERROR: No API key. Set GOOGLE_API_KEY or GEMINI_API_KEY env var.")
        sys.exit(1)
    
    # Initialize agent_repo
    AGENT_REPO.mkdir(exist_ok=True)
    if not AGENT_STRATEGY.exists():
        shutil.copy2(STRATEGY_FILE, AGENT_STRATEGY)
    
    if not (AGENT_REPO / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(AGENT_REPO), capture_output=True)
        git_commit("Initial Sugarscape strategy")
    
    git_create_branch(f"ablation/{datetime.now().strftime('%b%d').lower()}")
    init_results()
    
    # Calibrate baseline
    baseline = calibrate_baseline()
    
    # Run parameter sweep validation before starting ablation
    from parameter_sweeps import run_all_sweeps
    log("\n═══ Running Parameter Sweep Validation ═══")
    try:
        sweep_results = run_all_sweeps(n_runs=5)  # quick validation (5 runs)
        if not all(r["all_valid"] for r in sweep_results.values()):
            log("⚠️  Some parameter sweeps failed — model may not fully match paper.")
            log("    Proceeding with ablation, but results should be interpreted cautiously.")
    except Exception as e:
        log(f"⚠️  Parameter sweep validation failed: {e}")
        log("    Proceeding with ablation anyway.")
    
    client = create_client()
    
    generation = 1
    while True:
        try:
            success = run_ablation_round(generation, client, baseline)
            if success:
                generation += 1
            else:
                log(f"Generation {generation} did not produce a valid simplification. Retrying...")
            
            # Brief pause between generations
            time.sleep(2)
            
        except KeyboardInterrupt:
            log("\nInterrupted by user. Exiting early.")
            break
        except Exception as e:
            log(f"Generation {generation} failed: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)
            
    # Print final summary
    current = STRATEGY_FILE.read_text()
    final_complexity = count_complexity(current)
    initial_complexity = baseline.get("complexity", {})
    
    log(f"\n═══ Final Summary ═══")
    log(f"  Generations run: {generation - 1}")
    log(f"  Initial combined score: {initial_complexity.get('combined_score', '?')}")
    log(f"  Final combined score:   {final_complexity['combined_score']:.1f}")
    log(f"  Final lines:            {final_complexity['lines']}")
    if initial_complexity.get('combined_score'):
        score_red = initial_complexity['combined_score'] - final_complexity['combined_score']
        pct = 100 * score_red / initial_complexity['combined_score']
        log(f"  Score reduction:    {score_red:.1f} ({pct:.1f}%)")
        line_red = initial_complexity.get('lines', 0) - final_complexity['lines']
        log(f"  Line reduction:     {line_red}")



if __name__ == "__main__":
    main()
