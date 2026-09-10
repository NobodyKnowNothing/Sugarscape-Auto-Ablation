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
    python autoresearch.py

The loop runs indefinitely — kill with Ctrl+C.
"""

import json
import os
import re
import shutil
import subprocess
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import textwrap
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = os.environ.get("ABLATION_MODEL", "gemma-4-31b-it")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "gemma-4-26b-a4b-it")
VARIANTS_PER_GENERATION = 1          # mutations per round (fewer = more careful)
STRATEGY_FILE = Path(__file__).parent / "strategy.py"
AGENT_REPO = Path(__file__).parent / "agent_repo"
AGENT_STRATEGY = AGENT_REPO / "strategy.py"
RESULTS_FILE = Path(__file__).parent / "results.tsv"
AGENT_RESULTS = AGENT_REPO / "results.tsv"
SWEEPS_FILE = Path(__file__).parent / "sweep_results.json"
AGENT_SWEEPS = AGENT_REPO / "sweep_results.json"
PROGRAM_FILE = Path(__file__).parent / "program.md"
BASELINE_METRICS_FILE = Path(__file__).parent / "baseline_metrics.json"

# Edit harness configuration ('batch' for fast single-turn surgical edits, 'agent' for interactive multi-turn)
HARNESS_MODE = os.environ.get("HARNESS_MODE", "batch")


API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        print(f"[{ts}] {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"[{ts}] {msg.encode('ascii', 'replace').decode()}", flush=True)


# ---------------------------------------------------------------------------
# Git & Remote push helpers (operate on agent_repo)
# ---------------------------------------------------------------------------
def get_github_token() -> str:
    """Retrieve GitHub token from environment (.env), Colab Secrets, or Windows Credential Manager."""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_PAT", "")
    if token:
        return token.strip().strip("'\"")
    
    # Check Google Colab Secrets (google.colab.userdata)
    try:
        from google.colab import userdata
        colab_token = userdata.get("GITHUB_TOKEN") or userdata.get("GH_TOKEN")
        if colab_token:
            return colab_token.strip().strip("'\"")
    except Exception:
        pass

    # Fallback: Query Windows Credential Manager via git-credential-wincred
    wincred_candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Git/mingw64/libexec/git-core/git-credential-wincred.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Git/mingw64/libexec/git-core/git-credential-wincred.exe",
    ]
    for wincred in wincred_candidates:
        if wincred.exists():
            try:
                proc = subprocess.Popen(
                    [str(wincred), "get"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                out, _ = proc.communicate(input="protocol=https\nhost=github.com\n\n", timeout=3)
                for line in out.splitlines():
                    if line.startswith("password="):
                        return line.split("=", 1)[1].strip()
            except Exception:
                pass
    return ""


def get_remote_repo_url(token: str | None = None) -> str:
    """Determine the remote GitHub URL, optionally embedding authentication token."""
    remote = os.environ.get("GITHUB_REMOTE_URL") or os.environ.get("AUTORESEARCH_REMOTE_URL", "")
    if not remote:
        try:
            res = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                capture_output=True, text=True, cwd=str(Path(__file__).parent),
                stdin=subprocess.DEVNULL
            )
            remote = res.stdout.strip()
        except Exception:
            remote = ""
    
    if not remote:
        remote = "https://github.com/NobodyKnowNothing/Sugarscape-Auto-Ablation.git"
    
    clean_remote = re.sub(r'https://[^@]+@', 'https://', remote)
    if not clean_remote.endswith(".git"):
        clean_remote += ".git"

    if token:
        return clean_remote.replace("https://", f"https://{token}@")
    return clean_remote


def git(cmd: str, cwd: str | None = None, timeout: int = 30) -> str:
    """Run a git command in agent_repo (or specified cwd) in non-interactive mode."""
    try:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        result = subprocess.run(
            f"git {cmd}", shell=True, capture_output=True, text=True,
            cwd=cwd or str(AGENT_REPO), stdin=subprocess.DEVNULL,
            env=env, timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        log(f"⚠️  Git command 'git {cmd}' timed out after {timeout}s")
        return ""
    except Exception as e:
        log(f"⚠️  Git command 'git {cmd}' failed: {e}")
        return ""


def sync_logs_to_agent_repo():
    """Ensure results.tsv, sweep_results.json, and .gitignore are mirrored into agent_repo."""
    if not AGENT_REPO.exists():
        return
    try:
        if RESULTS_FILE.exists():
            shutil.copy2(RESULTS_FILE, AGENT_RESULTS)
        if SWEEPS_FILE.exists():
            shutil.copy2(SWEEPS_FILE, AGENT_SWEEPS)
        parent_gitignore = Path(__file__).parent / ".gitignore"
        if parent_gitignore.exists():
            shutil.copy2(parent_gitignore, AGENT_REPO / ".gitignore")
    except Exception as e:
        log(f"⚠️  Failed to sync logs to agent_repo: {e}")


def git_commit(message: str):
    sync_logs_to_agent_repo()
    git("add strategy.py")
    if AGENT_RESULTS.exists():
        git("add results.tsv")
    if AGENT_SWEEPS.exists():
        git("add sweep_results.json")
    git(f'commit -m "{message}"')


def git_revert_to(commit_hash: str):
    # Revert strategy.py to commit_hash without discarding test logs in results.tsv
    git(f"checkout {commit_hash} -- strategy.py")
    if AGENT_STRATEGY.exists():
        shutil.copy2(AGENT_STRATEGY, STRATEGY_FILE)


def git_current_hash() -> str:
    return git("rev-parse --short HEAD")


def git_current_branch() -> str:
    branch = git("rev-parse --abbrev-ref HEAD")
    return branch if branch and "fatal:" not in branch else f"ablation/{datetime.now(timezone.utc).strftime('%b%d').lower()}"


def git_create_branch(name: str):
    existing = git("branch --list " + name)
    if existing.strip():
        git(f"checkout {name}")
    else:
        git(f"checkout -b {name}")


def git_push(branch: str | None = None, remote: str = "origin") -> bool:
    """Autonomously push changes from agent_repo to the remote GitHub repository."""
    token = get_github_token()
    if not token:
        log("   ⚠️  Git push skipped: No GITHUB_TOKEN configured.")
        log("       To enable push in Colab, add 'GITHUB_TOKEN' to Secrets (🔑 icon)")
        log("       or run: os.environ['GITHUB_TOKEN'] = 'your_token_here'")
        return False

    target_branch = branch or git_current_branch()
    log(f"   Pushing {target_branch} to GitHub...")
    try:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        result = subprocess.run(
            f"git push -u {remote} {target_branch}",
            shell=True, capture_output=True, text=True,
            cwd=str(AGENT_REPO), stdin=subprocess.DEVNULL,
            env=env, timeout=45
        )
        if result.returncode == 0:
            log(f"   🚀 Successfully pushed {target_branch} to GitHub")
            return True
        else:
            err = result.stderr.strip() or result.stdout.strip()
            cleaned_err = re.sub(r'https://[^@]+@', 'https://***@', err)
            log(f"   ⚠️  Git push returned code {result.returncode}: {cleaned_err}")
            return False
    except subprocess.TimeoutExpired:
        log("   ⚠️  Git push timed out after 45s (network delay)")
        return False
    except Exception as e:
        log(f"   ⚠️  Git push error: {e}")
        return False


def init_agent_repo():
    """Initialize agent_repo with remote origin and target ablation branch."""
    token = get_github_token()
    remote_url = get_remote_repo_url(token)
    safe_remote_url = get_remote_repo_url(token=None)
    branch = os.environ.get("ABLATION_BRANCH") or f"ablation/{datetime.now(timezone.utc).strftime('%b%d').lower()}"
    parent_repo = str(Path(__file__).parent)

    AGENT_REPO.mkdir(exist_ok=True)

    if not token:
        log("⚠️  NOTICE: No GITHUB_TOKEN detected!")
        log("   Subrepo changes will be committed locally, but pushing to GitHub is disabled.")
        log("   In Google Colab, add 'GITHUB_TOKEN' to Secrets (🔑 icon on left sidebar)")
        log("   or set: os.environ['GITHUB_TOKEN'] = 'gho_...' before running.")

    is_new = not (AGENT_REPO / ".git").exists()
    if is_new:
        log(f"Initializing autonomous agent subrepo in {AGENT_REPO.name}...")
        subprocess.run(["git", "init"], cwd=str(AGENT_REPO), capture_output=True, stdin=subprocess.DEVNULL)
        
        # Configure non-interactive git environment & identity (critical for Colab / headless)
        git('config credential.helper ""')
        if not git("config user.name"):
            git('config user.name "Autoresearch Agent"')
        if not git("config user.email"):
            git('config user.email "agent@autoresearch.local"')
        
        git(f'remote add origin "{remote_url}"')
        
        # Check if target branch already exists on remote origin
        remote_synced = False
        if token:
            log(f"Checking if remote branch '{branch}' exists on GitHub...")
            fetch_remote = subprocess.run(
                ["git", "fetch", "origin", f"{branch}:{branch}"],
                cwd=str(AGENT_REPO), capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30
            )
            if fetch_remote.returncode == 0:
                log(f"Synced with existing remote branch '{branch}' from GitHub.")
                subprocess.run(
                    ["git", "checkout", branch],
                    cwd=str(AGENT_REPO), capture_output=True, text=True, stdin=subprocess.DEVNULL
                )
                remote_synced = True
                if not AGENT_STRATEGY.exists():
                    shutil.copy2(STRATEGY_FILE, AGENT_STRATEGY)
                if AGENT_RESULTS.exists() and not RESULTS_FILE.exists():
                    shutil.copy2(AGENT_RESULTS, RESULTS_FILE)
                sync_logs_to_agent_repo()

        if not remote_synced:
            # Align commit history with parent repository main branch (instant local fetch)
            log("Aligning commit history with repository main branch...")
            fetch_res = subprocess.run(
                ["git", "fetch", parent_repo, "main"],
                cwd=str(AGENT_REPO), capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30
            )
            if fetch_res.returncode == 0:
                subprocess.run(
                    ["git", "checkout", "-B", branch, "FETCH_HEAD"],
                    cwd=str(AGENT_REPO), capture_output=True, text=True, stdin=subprocess.DEVNULL
                )
                shutil.copy2(STRATEGY_FILE, AGENT_STRATEGY)
                sync_logs_to_agent_repo()
                status = git("status --porcelain")
                if status:
                    git_commit("Baseline Sugarscape strategy and logs for ablation")
            else:
                git_create_branch(branch)
                shutil.copy2(STRATEGY_FILE, AGENT_STRATEGY)
                sync_logs_to_agent_repo()
                git_commit("Initial Sugarscape strategy and logs")
    else:
        # Existing repo: update remote URL and disable interactive prompts
        git('config credential.helper ""')
        if not git("config user.name"):
            git('config user.name "Autoresearch Agent"')
        if not git("config user.email"):
            git('config user.email "agent@autoresearch.local"')
        
        existing_remotes = git("remote")
        if "origin" in existing_remotes:
            git(f'remote set-url origin "{remote_url}"')
        else:
            git(f'remote add origin "{remote_url}"')
        if token:
            subprocess.run(
                ["git", "fetch", "origin", f"{branch}:{branch}"],
                cwd=str(AGENT_REPO), capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30
            )
        git_create_branch(branch)
        if not AGENT_STRATEGY.exists():
            shutil.copy2(STRATEGY_FILE, AGENT_STRATEGY)
        # If remote branch already had results.tsv but parent doesn't, recover it
        if AGENT_RESULTS.exists() and not RESULTS_FILE.exists():
            shutil.copy2(AGENT_RESULTS, RESULTS_FILE)
        sync_logs_to_agent_repo()

    log(f"Subrepo ready on branch '{branch}' -> {safe_remote_url}")
    if token:
        git_push(branch)


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------
def init_results():
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "generation\tvariant\tcommit\tstatus\tlines\tclasses\tmethods\t"
            "ast_nodes\tcyclomatic\tcombined_score\t"
            "gini\tpopulation\ttrade_price\ttrade_volume\tsurvival\twealth_cv\tentropy\t"
            "passes\tstage_reached\tks_pvalue\tdtw_distance\tmorans_i\tdescription\n"
        )
    sync_logs_to_agent_repo()


def append_result(gen, var, commit, status, complexity, metrics, passes, desc,
                  stage_reached=0, ks_pvalue=None, dtw_distance=None, morans_i=None):
    ks_str = f"{ks_pvalue:.4f}" if ks_pvalue is not None else ""
    dtw_str = f"{dtw_distance:.4f}" if dtw_distance is not None else ""
    mi_str = f"{morans_i:.4f}" if morans_i is not None else ""
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
            f"{'PASS' if passes else 'FAIL'}\t"
            f"{stage_reached}\t"
            f"{ks_str}\t"
            f"{dtw_str}\t"
            f"{mi_str}\t"
            f"{desc}\n"
        )
    sync_logs_to_agent_repo()


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
def evaluate_strategy(strategy_code: str, baseline: dict = None) -> dict:
    """
    Validate a strategy variant through the 3-stage fail-fast waterfall.

    Returns dict with:
      - "mean_metrics": averaged metrics (from deepest stage reached)
      - "passes": bool
      - "validation_result": the full ValidationResult object
      - "error": str (if crashed before producing metrics)
    """
    from validation import validate_ablation

    # Load baseline if not provided
    if baseline is None:
        if BASELINE_METRICS_FILE.exists():
            with open(BASELINE_METRICS_FILE) as f:
                baseline = json.load(f)
        else:
            return {"error": "No baseline_metrics.json found"}

    vr = validate_ablation(strategy_code, baseline)

    # Build a backward-compatible result dict
    mean_metrics = vr.stage3_mean_metrics or vr.stage2_mean_metrics or {}
    result = {
        "mean_metrics": mean_metrics,
        "passes": vr.passed,
        "validation_result": vr,
    }

    if vr.reject_reason and not mean_metrics:
        result["error"] = vr.reject_reason

    return result


# ---------------------------------------------------------------------------
# LLM Agent — Gemma via google-genai
# ---------------------------------------------------------------------------
def create_client() -> genai.Client:
    return genai.Client(api_key=API_KEY)


from edit_harness import generate_ablation_variants_harness


def generate_ablation_variants(
    client: genai.Client,
    current_strategy: str,
    results_history: str,
    baseline_metrics: dict,
    complexity: dict,
    n_variants: int = VARIANTS_PER_GENERATION,
    temperature: float = 0.7,
    model_name: str = MODEL,
    fallback_model: str = FALLBACK_MODEL,
) -> list[tuple[str, str]]:
    """
    Generate N structural simplifications of strategy.py using the surgical edit harness.
    Applies pinpoint single-line or block replacements instead of full-file generation.
    Defaults to Gemma 4 31B with automatic failover to Gemma 4 26B if requests are exhausted.
    """
    program = PROGRAM_FILE.read_text() if PROGRAM_FILE.exists() else ""
    log(f"Prompting {model_name} (fallback: {fallback_model}) via mini-swe-agent edit harness (mode={HARNESS_MODE})...")
    return generate_ablation_variants_harness(
        client=client,
        current_strategy=current_strategy,
        results_history=results_history,
        baseline_metrics=baseline_metrics,
        complexity=complexity,
        program_text=program,
        n_variants=n_variants,
        mode=HARNESS_MODE,
        model_name=model_name,
        fallback_model=fallback_model,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Calibration against Mesa Canonical
# ---------------------------------------------------------------------------
def calibrate_baseline():
    """
    Establish ground-truth metrics against Mesa's canonical Sugarscape G1MT.
    Uses cached baseline if available, otherwise runs Mesa canonical.
    
    CRITICAL: The baseline MUST come from the CANONICAL Mesa 
    implementation (Mesa's faithful reproduction of Epstein & Axtell 1996),
    not a self-referential copy of our standalone strategy.py.
    
    For the 3-stage validation waterfall, we also store:
      - per_seed_wealths: wealth distributions per seed (for KS test)
      - per_seed_metrics: individual seed metric dicts (for Bonferroni t-tests)
      - mean_population_series: averaged population time series (for DTW)
      - mean_price_series: averaged price time series (for DTW)
    """
    log("╔═══════════════════════════════════════════════════════════╗")
    log("║  Calibrating Against Mesa Canonical Sugarscape            ║")
    log("╚═══════════════════════════════════════════════════════════╝")
    
    if BASELINE_METRICS_FILE.exists():
        with open(BASELINE_METRICS_FILE) as f:
            baseline = json.load(f)
        
        # Check if baseline has the per-seed data needed for waterfall validation
        has_waterfall_data = (
            "per_seed_wealths" in baseline
            and "per_seed_metrics" in baseline
            and len(baseline.get("per_seed_metrics", [])) >= 10
        )
        
        if has_waterfall_data:
            log(f"Loaded cached Mesa baseline ({baseline.get('source', 'unknown')})")
            log(f"  Runs: {baseline.get('n_runs', '?')} (with per-seed data for waterfall)")
            from metrics import format_metrics_report
            log(format_metrics_report(baseline["mean_metrics"]))
            return baseline
        else:
            log("Cached baseline lacks per-seed data for waterfall validation.")
            log("Will augment with per-seed wealth/metrics data...")
    
    # No cached baseline or needs augmentation — run Mesa canonical
    log("Running Mesa canonical Sugarscape G1MT for baseline calibration...")
    log("(This runs the ACTUAL canonical implementation, not our strategy.py)")
    
    from prepare import evaluate_mesa_baseline
    from metrics import compute_all_metrics
    
    n_baseline_runs = 50  # enough seeds for Bonferroni t-tests at n=50
    mesa_result = evaluate_mesa_baseline(n_runs=n_baseline_runs)
    
    # Also score initial strategy complexity
    initial_code = STRATEGY_FILE.read_text()
    initial_complexity = count_complexity(initial_code)
    
    # Extract per-seed data from mesa_result for waterfall validation
    per_seed_metrics = mesa_result.get("runs", [])
    per_seed_wealths = mesa_result.get("wealths", [])
    
    baseline = {
        "source": "mesa.examples.advanced.sugarscape_g1mt (Mesa 3.5.1 canonical)",
        "calibrated_at": datetime.now().isoformat(),
        "n_runs": n_baseline_runs,
        "mean_metrics": mesa_result["mean_metrics"],
        "complexity": initial_complexity,
        # Per-seed data for 3-stage validation waterfall
        "per_seed_metrics": per_seed_metrics,
        "per_seed_wealths": per_seed_wealths,
        "mean_population_series": mesa_result.get("mean_population_series", []),
        "mean_price_series": mesa_result.get("mean_price_series", []),
    }
    
    for name in mesa_result["mean_metrics"]:
        baseline[f"{name}_std"] = mesa_result.get(f"{name}_std", 0.0)
    
    with open(BASELINE_METRICS_FILE, "w") as f:
        json.dump(baseline, f, indent=2)
    
    log(f"Saved canonical baseline to {BASELINE_METRICS_FILE}")
    log(f"  Includes per-seed data for {n_baseline_runs} seeds (waterfall validation)")
    log(format_complexity_report(initial_complexity))
    return baseline


# ---------------------------------------------------------------------------
# Ratchet Loop
# ---------------------------------------------------------------------------
def run_ablation_round(generation: int, client: genai.Client, baseline: dict):
    """Execute one round of structural ablation with 3-stage fail-fast validation."""
    log(f"\n═══ Generation {generation} ═══")
    
    # Read current strategy
    current_strategy = STRATEGY_FILE.read_text()
    current_complexity = count_complexity(current_strategy)
    current_hash = git_current_hash()
    
    gz_info = ""
    if "gzip_compression_ratio" in current_complexity:
        gz_info = f", gzip={current_complexity['gzip_compression_ratio']:.1%} ({current_complexity['gzip_compressed_bytes']}B)"
    log(f"Current complexity: score={current_complexity['combined_score']:.1f} "
        f"(AST={current_complexity['ast_nodes']}, cyclo={current_complexity['cyclomatic_total']}"
        f"{gz_info}, lines={current_complexity['lines']})")
    
    # Read results history for context
    history = read_results_history()
    
    # Generate ablation variants from LLM
    log(f"Prompting {MODEL} for structural simplifications...")
    variants = generate_ablation_variants(
        client, current_strategy, history,
        baseline["mean_metrics"], current_complexity,
        n_variants=VARIANTS_PER_GENERATION,
    )
    
    if not variants:
        log("No valid variants generated this round.")
        return False
    
    log(f"Generated {len(variants)} variant(s). Evaluating via 3-stage waterfall...")
    
    best_variant = None
    best_simplification = 0.0  # combined score reduction
    
    for v_idx, (code, desc) in enumerate(variants):
        var_num = v_idx + 1
        log(f"\n--- Variant {var_num}/{len(variants)}: {desc} ---")
        
        # Check complexity
        var_complexity = count_complexity(code)
        score_delta = current_complexity["combined_score"] - var_complexity["combined_score"]
        var_gz_info = ""
        if "gzip_compression_ratio" in var_complexity:
            var_gz_info = f" | Gzip: {var_complexity['gzip_compression_ratio']:.1%} ({var_complexity['gzip_compressed_bytes']}B)"
        log(f"  Score: {var_complexity['combined_score']:.1f} ({score_delta:+.1f}) | "
            f"AST: {var_complexity['ast_nodes']} | Cyclo: {var_complexity['cyclomatic_total']}"
            f"{var_gz_info} | Lines: {var_complexity['lines']}")
        
        # Evaluate through 3-stage waterfall
        t0 = time.time()
        result = evaluate_strategy(code, baseline=baseline)
        dt = time.time() - t0
        
        vr = result.get("validation_result")
        stage_reached = vr.stage_reached if vr else 0
        ks_pval = vr.ks_pvalue if vr else None
        dtw_dist = vr.dtw_population if vr else None
        mi_val = vr.morans_i if vr else None
        
        if "error" in result:
            stage_label = f"Stage{stage_reached}" if stage_reached else "pre-validation"
            log(f"  ❌ REJECTED at {stage_label} ({dt:.1f}s): {result['error']}")
            append_result(generation, var_num, current_hash, "crash",
                         var_complexity, {}, False, desc,
                         stage_reached=stage_reached, ks_pvalue=ks_pval,
                         dtw_distance=dtw_dist, morans_i=mi_val)
            continue
        
        passes = result["passes"]
        reject_reason = vr.reject_reason if vr and not passes else ""
        timing_detail = ""
        if vr:
            timing_detail = (
                f" [S1={vr.stage1_time:.1f}s"
                f", S2={vr.stage2_time:.1f}s"
                f", S3={vr.stage3_time:.1f}s]"
            )
        log(f"  Validation ({dt:.1f}s{timing_detail}): "
            f"stage_reached={stage_reached}, passes={passes}")
        if reject_reason:
            log(f"  Reject reason: {reject_reason}")
        
        # Append to results log
        append_result(generation, var_num, current_hash,
                     "pass" if passes else "fail",
                     var_complexity, result["mean_metrics"],
                     passes, desc,
                     stage_reached=stage_reached, ks_pvalue=ks_pval,
                     dtw_distance=dtw_dist, morans_i=mi_val)
        
        if passes:
            log(f"  ✅ PASSES all 3 stages of validation!")
            if vr:
                log(f"     KS p={vr.ks_pvalue:.4f} | DTW pop={vr.dtw_population:.4f} "
                    f"| Moran's I={vr.morans_i:.4f} | Wasserstein={vr.wasserstein_distance:.4f}")
            # Track best passing variant (largest combined score reduction)
            if score_delta > best_simplification:
                best_variant = (code, desc, var_complexity, result)
                best_simplification = score_delta
        else:
            log(f"  ❌ FAILS at Stage {stage_reached}")
    
    # Commit best variant if it's simpler
    if best_variant:
        code, desc, complexity, result = best_variant
        score_saved = current_complexity["combined_score"] - complexity["combined_score"]
        
        if score_saved > 0:
            log(f"\n🎉 RATCHET FORWARD: {desc}")
            log(f"   Score: {current_complexity['combined_score']:.1f} → {complexity['combined_score']:.1f} ({score_saved:+.1f})")
            gz_line = ""
            if "gzip_compressed_bytes" in current_complexity and "gzip_compressed_bytes" in complexity:
                gz_line = f" | Gzip: {current_complexity['gzip_compressed_bytes']}B → {complexity['gzip_compressed_bytes']}B"
            log(f"   Lines: {current_complexity['lines']} → {complexity['lines']}{gz_line}")
            
            # Write to strategy.py and agent_repo
            STRATEGY_FILE.write_text(code)
            AGENT_STRATEGY.write_text(code)
            sync_logs_to_agent_repo()
            git_commit(f"Gen {generation}: {desc[:50]} | score -{score_saved:.1f}")
            
            chash = git_current_hash()
            log(f"   Committed as {chash}")
            
            # Autonomously push new commit to remote GitHub repository
            git_push()
            
            return True
        else:
            log(f"\n⏸️  Variant passes but not simpler — skipping commit")
            return False
    else:
        log(f"\n❌ No passing variants this generation — reverting")
        git_revert_to(current_hash)
        return False


def main():
    global MODEL, FALLBACK_MODEL
    import argparse
    from complexity import AVAILABLE_SCORERS, select_scoring_function, get_active_scoring_function

    parser = argparse.ArgumentParser(description="Sugarscape Structural Ablation Pipeline")
    parser.add_argument(
        "--scorer", "-s",
        choices=list(AVAILABLE_SCORERS.keys()),
        default=os.environ.get("COMPLEXITY_SCORER", None),
        help="Complexity metric / scorer preset to optimize (default: weighted_ast_cyclomatic)",
    )
    parser.add_argument(
        "--model", "-m",
        default=MODEL,
        help=f"Primary model to use for ablation (default: {MODEL})",
    )
    parser.add_argument(
        "--fallback-model",
        default=FALLBACK_MODEL,
        help=f"Fallback model when quota is exhausted (default: {FALLBACK_MODEL})",
    )
    args, _ = parser.parse_known_args()
    if args.scorer:
        select_scoring_function(args.scorer)

    MODEL = args.model
    FALLBACK_MODEL = args.fallback_model

    _, active_scorer_name = get_active_scoring_function()

    log("╔═══════════════════════════════════════════════════════════╗")
    log("║  Sugarscape Structural Ablation Pipeline                  ║")
    log(f"║  Model:     {MODEL:40s}      ║")
    log(f"║  Fallback:  {FALLBACK_MODEL:40s}      ║")
    log(f"║  Scorer:    {active_scorer_name:40s}      ║")
    log("║  Pattern:   Karpathy Autoresearch Ratchet Loop            ║")
    log("╚═══════════════════════════════════════════════════════════╝")

    if not API_KEY:
        log("ERROR: No API key. Set GOOGLE_API_KEY or GEMINI_API_KEY env var.")
        sys.exit(1)
    
    # Initialize results log & agent_repo tracking branch
    init_results()
    init_agent_repo()
    
    # Calibrate baseline
    baseline = calibrate_baseline()
    
    # Run parameter sweep validation before starting ablation
    from parameter_sweeps import run_all_sweeps
    log("\n═══ Running Parameter Sweep Validation ═══")
    try:
        sweep_results = run_all_sweeps(n_runs=5)  # quick validation (5 runs)
        sync_logs_to_agent_repo()
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
            log("\nInterrupted by user. Pushing final state to GitHub before exit...")
            sync_logs_to_agent_repo()
            status = git("status --porcelain")
            if status:
                git_commit("Update experiment logs before exit")
            git_push()
            log("Exiting early.")
            break
        except Exception as e:
            log(f"Generation {generation} failed: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)
            
    # Final push of completed research
    sync_logs_to_agent_repo()
    status = git("status --porcelain")
    if status:
        git_commit("Final experiment logs")
    git_push()
            
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
