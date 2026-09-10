# Sugarscape Auto-Ablation: Searching for the Minimal Rules of a Sugarscape Society

An LLM research loop proposes creative changes to an agent-based model. Each candidate is scored by **compression-based complexity** and accepted only if its **emergent features** are **statistically equivalent** to Epstein & Axtell's Sugarscape with trade (Mesa's `sugarscape_g1mt`). Equivalence is tested over a pre-registered design of seeds and input parameters.

The goal is the minimal *mechanism* that produces the same society, not a shorter copy of the same simulation. Candidates may use different rules, representations and random draws. What must match are the society-level patterns, within margins set by how much Sugarscape itself varies under changes that leave the society unchanged.

The design rationale is in [PROPOSAL_MVM_REDESIGN.md](PROPOSAL_MVM_REDESIGN.md). The audit of the previous judge is in [EMERGENCE_JUDGE_REPORT.md](EMERGENCE_JUDGE_REPORT.md).

## Layout

```
judge/                     the judge (the research loop never edits this)
  contract.py              fixed world + model contract, judge-owned run loop, fail-closed snapshot validation
  features.py              emergent-feature statistics (E1-E4) and presence checks, computed from observed state
  stats.py                 equivalence tests (TOST, sup-distance bootstrap, difference-of-differences), Holm
  complexity.py            canonical compressed size of strategy.py + traced executed library (Mesa) code
  guard.py                 static guard (imports, dynamic code, file/OS access, global RNG, contract presence)
  design.json / design.py  design points (sweeps A-D + 24-point Latin hypercube), seed pools, sample sizes, profiles
  controls/                innocuous perturbations, an independent numpy re-implementation, knockouts, null societies
  pipeline.py              staged acceptance pipeline, audits, reference/margins storage
  reference/<profile>/     calibration output: reference ensemble, margins, control outcomes
calibrate_judge.py         builds the reference ensemble and margins and calibrates the judge against controls
autoresearch.py            the LLM research loop (Pareto archive, move portfolio, ledger, audits, git sub-repo)
edit_harness.py            Gemma prompting/parsing (hypothesis-driven proposals, full rewrites, multi-edit), failover
certify.py                 knockout certificate: is every surviving rule necessary for some emergent feature?
mechanisms.json            mechanism ledger: rule statuses and evidence
canonical_strategy.py      canonical Mesa G1MT ported to the judge contract (the reference model)
strategy.py                current best model (the loop's working copy)
program.md                 research agenda given to the LLM
test_judge.py              judge test suite
```

## The model contract

`strategy.py` defines the dynamics only. The judge owns the loop, the RNG and all measurement.

```python
def init(env, rng): ...      # env: inputs + env["landscape"] (2 x 50 x 50 capacities); rng: numpy Generator
def step(state, rng): ...
def observe(state):          # -> {"ids", "positions", "holdings" [N x G], "exchanges" [(i, j, dq)], "resources"}
```

A missing or malformed field, a non-finite value, non-determinism under a fixed seed, or a crash on any seed is a **hard reject**. Nothing fails open.

## Emergent features

All five are computed by the judge from observed snapshots, time-resolved and dimensionless where possible.

| | Feature | Presence check | Equivalence statistics |
|---|---|---|---|
| E1 | Wealth inequality | Gini grows beyond endowments; right skew | Gini at T/4 and T, top-10% share, CV, Hill tail index, Lorenz curve |
| E2 | Carrying capacity | Population plateaus | Plateau N/N₀, relaxation time, fluctuation, N(t)/N₀ trajectory |
| E3 | Price discovery | Log-price dispersion decreases | Late dispersion, decay, equilibrium price level, exchange rate, dispersion trajectory |
| E4 | Spatial self-organization | Density–resource correlation beats a uniform-placement null | Density–resource rank correlation over time, Moran's I of density |
| E5 | Environmental response | — | Candidate's response to each input change versus Sugarscape's (difference of differences) |

## Acceptance pipeline

| Stage | What it checks |
|---|---|
| 0 | The static guard, plus traced canonical complexity. C must drop by at least max(8 B, 0.1%·C). |
| 1 | Smoke run and determinism. If trajectories are bit-identical to the parent, the change is labelled `refactor` (a label only, never a reason to accept). |
| 2 | Presence of E1–E4 at the default point, on fresh seeds. |
| 3 | Sequential TOST of every statistic at every screening design point. Seeds come from a pool disjoint from all reference seeds. |
| 4 | E5 response-surface equivalence. |
| Audit | Every K accepted mechanism changes: the full design (screening + Latin hypercube) on holdout seeds. On failure, the unaudited changes are rolled back. |

How the tests are combined:
- **Acceptance.** Every test must pass. This is an intersection-union test, so the overall error rate is α without a multiplicity correction.
- **Certificates.** These need only one feature to break, so they use Holm correction.
- **Baseline.** Comparisons are always against canonical Sugarscape, never against the parent.

## Margins and calibration

Margins are measured in reference between-seed SDs. For each statistic and design point:

> margin = max(floor, largest upper confidence bound of the shift produced by an innocuous perturbation)

The innocuous perturbations are ±5% regrowth, growback after agents act, and ±1 endowment range.

`calibrate_judge.py` then checks the judge against controls:
- **Positive controls** must pass at least 95% of replications. These are an independent plain-numpy re-implementation (no Mesa), and the same rules with different random draws.
- **Knockouts** must fail the features they target at least 95% of the time: no trade, no growback, random movement, no MRS-crossing stop, uniform vision.
- **Null societies** must also fail their targeted features: random walkers with multiplicative random wealth, static agents, random prices.
- **Separation check.** A statistic that no knockout or null society moves beyond its margin is dropped as non-discriminative.

The loop refuses to start if calibration is missing, stale (the judge or design changed) or failing.

## Complexity

C = min over gzip/bz2/lzma of the compressed `canon(strategy.py + executed dependency code)`, minus the empty-template size.

- `canon()` strips docstrings, comments, annotations and asserts, then α-renames every identifier. Renaming or reformatting therefore gives ΔC = 0.
- A short traced run, in a fresh process, records every Python function executed outside the reference machine (CPython, stdlib, numpy). Those functions are extracted from source and compressed together with the model.
- For the canonical model, C = 5890 B. Of that, 1578 B is `strategy.py` and the rest is 76 executed Mesa functions.

## Usage

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt pytest

pytest test_judge.py -v                          # judge tests
python calibrate_judge.py --profile full         # required before a real run (estimated ~1 h on 12 cores)
python autoresearch.py --profile full            # research loop (needs GOOGLE_API_KEY; pushes to agent_repo branch)
python autoresearch.py --profile full --no-push  # commit locally only
python certify.py --profile full --knockouts knockouts/ --generate   # certificate for the final model
```

The `smoke` profile (100 steps, small samples, wide margin floors) exists to exercise every code path quickly:

```bash
python calibrate_judge.py --profile smoke
python autoresearch.py --profile smoke --generations 1 --no-git --allow-failed-calibration
```

Smoke results are **not** scientifically valid. Its calibration fails the 95% criteria by design.

## Pre-registered defaults (owner decisions)

These are set in `judge/design.json` and can be revised before a full calibration:

- **Target features:** E1–E5, including price discovery. A model without exchange can never pass.
- **Inputs and ranges:** sweeps A–D plus a Latin hypercube over N₀ [50, 400], vision_max [2, 8], metabolism_max [3, 7] and regrowth [0.5, 2.0].
- **Margin floors (full profile):** 1.0 SD for scalars, 1.25 SD for vectors, 1.4 SD for effects. This is the smallest margin that keeps all-tests-must-pass usable with 60/60 seeds.
- **Reference machine:** CPython + stdlib + numpy.
- **Compute (full profile):** 60 reference seeds per point, 30 per perturbation, candidates sequentially at 20 → 40 → 60 seeds per screening point, 40 holdout seeds per audit point, and an audit every 5 accepted mechanism changes.
