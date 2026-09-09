# ABM Auto-Ablation (Sugarscape)

This repository hosts a **work in progress** automated research program where an AI agent performs **structural ablation** on a flattened version of the mesa implementation of Epstein & Axtell's classic *Sugarscape* agent-based model.

## Objective

The goal of this experiment is to systematically strip away, merge, and simplify code to approach the **absolute minimal viable model** capable of reproducing the emergent macroscopic behaviors documented in the original paper. Stubborn functions that refuse to simplify are a sign of underlying logic required to produce the macrophenomena, revealing true key components of the emergent behavior. The AI is tasked with reducing structural and computational complexity while strictly maintaining emergent metrics within narrow error bounds:

- **Wealth Inequality** (Gini Coefficient ≈ 0.3–0.6)
- **Population Dynamics** (converging to environmental carrying capacity)
- **Trade Dynamics** (bilateral trade converging to the geometric mean of Marginal Rates of Substitution)
- **Survival Rate & Spatial Entropy** (foraging efficiency and clustering patterns)

---

## Repository Architecture

This codebase adapts the **3-file Karpathy autoresearch ratchet pattern** for agent-based model ablation:

```
Sugarscape-Auto-Ablation/
├── strategy.py             # MUTABLE: Current ablated Sugarscape model implementation
├── prepare.py              # FIXED: Baseline runner & evaluation harness (Mesa canonical)
├── program.md              # RESEARCH AGENDA: LLM constraints, objectives, and bounds
├── autoresearch.py         # ORCHESTRATOR: Autonomous LLM mutation & ratchet loop
├── complexity.py           # SCORING: AST node count + Cyclomatic complexity engine
├── metrics.py              # METRICS: Fixed emergent behavior definitions & statistical comparison
├── parameter_sweeps.py     # VALIDATION: Epstein & Axtell canonical sweep suite (Sweeps A–D)
├── baseline_metrics.json   # GROUND TRUTH: Canonical metrics from Mesa Sugarscape G1MT
├── sugar-map.txt           # TERRAIN: Canonical 50x50 resource landscape
├── requirements.txt        # DEPENDENCIES: Required Python packages
└── .gitignore              # GIT RULES: Standard exclusions for caches, logs, and venvs
```

### Component Details

- **[`strategy.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/strategy.py)**: The code under ablation. Implements `create_model(seed, **params)` and `run_model(model, steps)` returning standardized simulation observables.
- **[`prepare.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/prepare.py)**: Fixed evaluation harness that calibrates against Mesa's built-in `SugarscapeG1mt` canonical ground truth and evaluates variant strategies against it.
- **[`complexity.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/complexity.py)**: Calculates combined structural complexity score ($0.4 \times \text{AST Nodes} + 0.6 \times \text{Cyclomatic Complexity}$) to avoid "code golf" line minimization in favor of true computational simplicity.
- **[`metrics.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/metrics.py)**: Computes Gini coefficient, carrying capacity, mean trade price, trade volume, agent survival rate, wealth coefficient of variation, and spatial Shannon entropy with tight error bounds.
- **[`parameter_sweeps.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/parameter_sweeps.py)**: Validates that the model faithfully reproduces the four canonical parameter sweeps from *Growing Artificial Societies*:
  - **Sweep A**: Trade vs. No-Trade
  - **Sweep B**: Vision Range Sensitivity
  - **Sweep C**: Carrying Capacity & Population Density
  - **Sweep D**: Resource Scarcity & Metabolism
- **[`autoresearch.py`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/autoresearch.py)**: Runs the ratchet loop: prompts the LLM for structural simplifications, evaluates candidate variants against the ground-truth baseline, commits passing variants that reduce complexity, or reverts failing ones.
- **[`program.md`](file:///c:/Users/TEMP/Sugarscape-Auto-Ablation/program.md)**: Human-provided prompt context defining what the AI agent can modify, read-only constraints, and success criteria.

---

## Getting Started

### 1. Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Verify Current Strategy Against Baseline

Compare the current `strategy.py` against Mesa canonical ground truth:

```bash
python prepare.py --compare strategy.py
```

### 3. Run Parameter Sweep Validations

Validate that the model matches the four canonical sweeps from Epstein & Axtell (1996):

```bash
python parameter_sweeps.py --sweep all
```

Or run individual sweeps (`a`, `b`, `c`, or `d`):

```bash
python parameter_sweeps.py --sweep a
```

### 4. Run the Autoresearch Ratchet Loop

To launch autonomous structural ablation:

```bash
export GOOGLE_API_KEY="your-api-key-here"  # On Windows PowerShell: $env:GOOGLE_API_KEY="your-api-key-here"
python autoresearch.py
```

The pipeline defaults to **Gemma 4 31B** (`gemma-4-31b-it`). If Google GenAI API rate limits or quota requests are exhausted (HTTP 429 / `RESOURCE_EXHAUSTED`), the pipeline automatically routes requests to **Gemma 4 26B** (`gemma-4-26b-a4b-it`) so experimentation continues without interruption. Non-quota errors (such as formatting or syntax issues) do not trigger failover.

Results of each generation will be logged to `results.tsv`.

### 5. Autonomous GitHub Synchronization

`autoresearch.py` automatically maintains an isolated subrepo (`agent_repo`) that tracks ablation progress on an experiment branch (default: `ablation/<date>`).

Whenever a candidate variant passes all metric bounds and reduces complexity, the agent commits the change and pushes it autonomously to your remote GitHub repository.

#### Configuration Options (in `.env` or environment):
- `ABLATION_MODEL`: Primary LLM for ablation mutations (default: `gemma-4-31b-it`).
- `FALLBACK_MODEL`: Fallback model when request quota is exhausted (default: `gemma-4-26b-a4b-it`).
- `GITHUB_TOKEN`: GitHub Personal Access Token or OAuth token (auto-detected from Windows Credential Manager if omitted).
- `ABLATION_BRANCH`: Custom remote branch name (defaults to `ablation/<month><day>`, e.g., `ablation/sep08`).
- `GITHUB_REMOTE_URL`: Custom remote repository URL (auto-detected from the parent repo's `origin` remote).


