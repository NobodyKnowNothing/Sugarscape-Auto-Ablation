# ABM Auto-Ablation (Sugarscape)

This repository hosts a **work in progress** automated research program where an AI agent performs **structural ablation** on a flattened version of the mesa implementation of Epstein & Axtell's classic *Sugarscape* agent-based model.

## Objective

The goal of this experiment is to systematically strip away, merge, and simplify agent-based model logic to discover the **Absolute Minimal Viable Model (MVM)** capable of faithfully reproducing the macroscopic emergent phenomena documented in Epstein & Axtell's classic *Growing Artificial Societies* (1996). 

In an automated ratchet loop, an LLM agent applies surgical code simplifications to `strategy.py`. Any function, attribute, or algorithmic step that cannot be simplified without causing macroscopic dynamics to diverge identifies an **irreducible component** of the emergent behavior. The agent optimizes for structural and computational simplicity while subjecting each candidate to a strict **3-stage fail-fast validation waterfall**:

- **Wealth Inequality**: Gini coefficient and wealth distribution shape matching the canonical Pareto-like distribution.
- **Carrying Capacity Convergence**: Population dynamic trajectories reaching sustainable environmental equilibrium.
- **Market Price Convergence**: Bilateral trading price dynamics converging to the geometric mean of Marginal Rates of Substitution (MRS).
- **Spatial Clustering & Entropy**: Coordinated spatial foraging patterns maintaining positive spatial autocorrelation (Moran's I) and spatial entropy.

---

## Repository Architecture

This codebase implements an autonomous research ratchet loop with multi-objective complexity scoring, a surgical mini-SWE-agent edit harness, and a 3-stage validation waterfall:

```
Sugarscape-Auto-Ablation/
├── strategy.py             # MUTABLE TARGET: Current ablated Sugarscape model
├── validation.py           # VALIDATION GATE: Authoritative 3-stage fail-fast waterfall
├── autoresearch.py         # ORCHESTRATOR: Autonomous LLM mutation & ratchet loop
├── edit_harness.py         # SURGICAL HARNESS: Mini-SWE-agent edit engine with Gemma failover
├── complexity.py           # SCORING ENGINE: Multi-objective complexity & compression metrics
├── prepare.py              # BASELINE: Mesa canonical calibration & ground-truth extractor
├── metrics.py              # GROUND TRUTH: Fixed emergent behavior metric definitions (Immutable)
├── parameter_sweeps.py     # SWEEP SUITE: Epstein & Axtell canonical sweep suite (Sweeps A–D)
├── baseline_metrics.json   # CALIBRATION DATA: Ground-truth distributions and time series from Mesa
├── sugar-map.txt           # TERRAIN: Canonical 50x50 resource landscape (Epstein & Axtell 1996)
├── test_validation.py      # TEST SUITE: Pytest suite for the 3-stage waterfall
├── test_failover_system.py # TEST SUITE: Unit tests for LLM quota failover
├── requirements.txt        # DEPENDENCIES: Required Python packages
└── .gitignore              # GIT RULES: Exclusions for caches, logs, subrepos, and venvs
```

### Core Components

- **[`strategy.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/strategy.py)**: The active model under ablation. Implements `create_model(seed, **params)` and `run_model(model, steps)` returning observable agent and macroeconomic data.
- **[`validation.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/validation.py)**: The authoritative gatekeeper. Implements the 3-stage fail-fast waterfall that screens out invalid or diverging models with minimal compute overhead.
- **[`autoresearch.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/autoresearch.py)**: Orchestrates the Karpathy-style ratchet loop: prompts the LLM, validates variants through the waterfall, commits complexity-reducing passing candidates, logs results to `results.tsv`, and synchronizes with GitHub.
- **[`edit_harness.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/edit_harness.py)**: Inspired by mini-SWE-agent. Prompts Gemma for surgical diffs/block replacements (reducing output tokens from ~600 to ~10) and handles automated failover from Gemma 4 31B to Gemma 4 26B upon API quota exhaustion.
- **[`complexity.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/complexity.py)**: Pluggable complexity engine supporting weighted AST/cyclomatic complexity, pure AST count, cyclomatic complexity, and gzip compression ratio/bytes.
- **[`prepare.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/prepare.py)**: Runs the canonical Mesa implementation (`mesa.examples.advanced.sugarscape_g1mt`) across 50 seeds to establish baseline distribution and time-series ground truth.
- **[`metrics.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/metrics.py)**: Fixed ground-truth calculations for Gini, population, trade volume, mean price, survival rate, wealth CV, and spatial entropy. **Never modified during ablation.**
- **[`parameter_sweeps.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/parameter_sweeps.py)**: Standalone verification of the four canonical Epstein & Axtell parameter sweeps (A: Trade vs No Trade, B: Vision Sensitivity, C: Carrying Capacity, D: Resource Scarcity).

---

## The 3-Stage Fail-Fast Validation Waterfall

To maximize throughput and prevent wasting computation on invalid or divergent strategies, candidate code must survive three sequential gates:

```
Candidate Variant Code
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 1: Fast Filter (Syntax & Execution)               │
│ • Valid Python AST parse                                │  Cost: ~2s
│ • 1 seed (seed=42), 50 steps execution                  │  Pass rate: ~80%
│ • Zero crashes / RuntimeWarnings                        │
└──────────────────────────┬──────────────────────────────┘
                           │ (Pass)
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 2: Lightweight Shape Filter                       │
│ • 5 seeds (seeds 42..46), full 200 steps                │  Cost: ~15s
│ • Two-Sample KS Test on wealth distribution (p >= 0.05) │  Pass rate: ~30%
│ • Scalar error bounds on 7 emergent metrics             │
└──────────────────────────┬──────────────────────────────┘
                           │ (Pass)
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Stage 3: The Heavy Gauntlet                             │
│ • 50 seeds total (reuses Stage 2 + 45 additional seeds) │  Cost: ~150s
│ • Bonferroni-corrected Welch's t-tests (α/7 = 0.00714)  │  Pass rate: ~10%
│ • Dynamic Time Warping (DTW) on population trajectories │
│ • Moran's I spatial clustering (queen contiguity)       │
│ • Wasserstein distance on pooled wealth distribution    │
└──────────────────────────┬──────────────────────────────┘
                           │ (Pass & Simpler)
                           ▼
                 Ratchet Commit & Git Push
```

1. **Stage 1 — Fast Filter** (~2s):
   Catches syntax errors, import failures, and immediate crashes in a single 50-step seed before executing longer simulations.
2. **Stage 2 — Lightweight Shape Filter** (~15s):
   Simulates 5 independent seeds for 200 steps. Evaluates empirical scalar error bounds and performs a **two-sample Kolmogorov-Smirnov (KS) test** on pooled agent wealth against the canonical Mesa baseline. Distorted wealth distributions are rejected here without running 50 seeds.
3. **Stage 3 — The Heavy Gauntlet** (~150s):
   Only candidates that survive Stages 1 and 2 face the full statistical gauntlet across 50 seeds:
   - **Bonferroni-Corrected Welch's t-tests**: Stringent hypothesis test ($\alpha/7 = 0.00714$) across all 7 macroeconomic observables.
   - **Dynamic Time Warping (DTW)**: Measures temporal distance on carrying capacity convergence and price discovery series against baseline trajectories.
   - **Moran's I Spatial Autocorrelation**: Assesses spatial clustering of agent wealth on a 5×5 queen-contiguity grid to ensure emergent foraging aggregation ($I \ge -0.05$).
   - **Wasserstein Distance**: Measures Earth Mover's Distance between candidate and baseline wealth distributions.

---

## Multi-Objective Complexity Engine

The ablation objective is pluggable via `complexity.py` to prevent "code golf" (superficial line compression) and target true algorithmic simplification:

| Scorer Preset | CLI Flag | Metric Evaluated | Description |
| :--- | :--- | :--- | :--- |
| `weighted_ast_cyclomatic` | `-s weighted` | $0.4 \times \text{AST} + 0.6 \times \text{Cyclo}$ | **Default**: Balanced structural AST and branching complexity |
| `ast_nodes` | `-s ast` | Total AST Node Count | Raw syntax tree simplification |
| `cyclomatic` | `-s cyclo` | McCabe Cyclomatic Total | Minimizes control-flow branches, loops, and conditions |
| `gzip_ratio` | `-s gzip_ratio` | Gzip Compression Ratio (%) | Measures algorithmic entropy via DEFLATE compressibility |
| `gzip_size` | `-s gzip_size` | Compressed Bytes | Minimizes compressed representation length |

Evaluate complexity on any file:
```bash
python complexity.py strategy.py --scorer gzip_ratio
```

---

## Surgical Edit Harness & LLM Failover

To maximize code stability and minimize token usage, `edit_harness.py` provides surgical mutation operators:
- `replace_line`: Single-line modification.
- `replace_block`: Range-based block replacement or deletion.
- `str_replace`: Exact snippet search-and-replace.
- `eval`: Runs Stage 1 fast simulation check with canonical 50×50 parameters.
- `check`: Validates Python syntax and reports complexity deltas.

### Automatic Model Failover
The pipeline defaults to **Gemma 4 31B** (`gemma-4-31b-it`). If Google GenAI API rate limits or quota requests are exhausted (`RESOURCE_EXHAUSTED` / HTTP 429), `edit_harness.py` automatically fails over to **Gemma 4 26B** (`gemma-4-26b-a4b-it`) without interrupting the research loop.

---

## Getting Started

### 1. Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Validation Suite

Verify that all unit tests and statistical helpers pass:

```bash
pytest test_validation.py -v
```

### 3. Run Strategy Sanity Check

Verify that `strategy.py` compiles and runs cleanly through the Stage 1 filter:

```bash
python edit_harness.py eval strategy.py
```

### 4. Baseline Calibration

To recalibrate the ground truth from Mesa's canonical Sugarscape model (50 seeds with wealth and time-series extraction):

```bash
python prepare.py
```

To compare `strategy.py` directly against the canonical baseline:

```bash
python prepare.py --compare strategy.py
```

### 5. Canonical Parameter Sweeps

Run Epstein & Axtell's four canonical validation sweeps:

```bash
python parameter_sweeps.py --sweep all
```

### 6. Launch the Autonomous Ratchet Loop

Start the autonomous ablation loop:

```bash
# On Windows PowerShell:
$env:GOOGLE_API_KEY="your-api-key-here"
python autoresearch.py

# With custom objective and model:
python autoresearch.py --scorer gzip_ratio --model gemma-4-31b-it --fallback-model gemma-4-26b-a4b-it
```

### 7. Autonomous Git & Remote Synchronization

`autoresearch.py` automatically maintains an isolated subrepo (`agent_repo`) that tracks ablation progress on an experiment branch (e.g. `ablation/sep09`). 

Whenever a variant passes all 3 validation stages and achieves a lower complexity score, the harness:
1. Updates `strategy.py` in the workspace and `agent_repo`.
2. Commits the simplification with score and line deltas.
3. Automatically pushes the commit to your remote GitHub repository.
4. Appends detailed telemetry (`stage_reached`, `ks_pvalue`, `dtw_distance`, `morans_i`) to `results.tsv`.

#### Key Configuration Options (via `.env` or CLI):
- `GOOGLE_API_KEY` / `GEMINI_API_KEY`: API key for Google GenAI models.
- `COMPLEXITY_SCORER`: Default scorer (`weighted`, `ast`, `cyclo`, `gzip_ratio`, `gzip_size`).
- `HARNESS_MODE`: Surgical edit mode (`batch` for fast single-turn edits, `agent` for interactive SWE-agent).
- `ABLATION_MODEL`: Primary LLM (default: `gemma-4-31b-it`).
- `FALLBACK_MODEL`: Fallback LLM (default: `gemma-4-26b-a4b-it`).
- `GITHUB_TOKEN`: Personal access token for push permissions (auto-detected if omitted).
- `ABLATION_BRANCH`: Target branch for autonomous commits (default: `ablation/<month><day>`).
