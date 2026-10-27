# Sugarscape Auto-Ablation: Searching for the Minimal Rules of a Sugarscape Society

An LLM research loop proposes creative changes to an agent-based model. Each candidate is scored by **compression-based complexity** and accepted only if its **emergent features**, and the way every feature **responds to every input**, are **statistically equivalent** to Epstein & Axtell's Sugarscape with trade (Mesa's `sugarscape_g1mt`). Equivalence is tested over a fresh random sample of a wide, pre-registered input domain, with trade both on and off.

The goal is the minimal *mechanism* that produces the same society, not a shorter copy of the same simulation. Candidates may use different rules, representations and random draws. What must match are the society-level patterns, within margins set by how much Sugarscape itself varies under changes that leave the society unchanged.

The design rationale is in [PROPOSAL_MVM_REDESIGN.md](PROPOSAL_MVM_REDESIGN.md). The decisions on trait responses, input-space sampling and trade, with the research they rest on, are in [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md). The audit of the previous judge is in [EMERGENCE_JUDGE_REPORT.md](EMERGENCE_JUDGE_REPORT.md).

## Layout

```
judge/                     the judge (the research loop never edits this)
  contract.py              fixed world + model contract, judge-owned run loop, fail-closed snapshot validation
  features.py              emergent-feature statistics (E1-E4), presence indicators and checks, computed from observed state
  domain.py                the input domain, its random samples (scrambled Sobol' / i.i.d.) and the response regressors
  surface.py               response-surface equivalence over a domain sample (l2, slopes, trade contrast, spread, named)
  stats.py                 per-point TOST (stage-2 presence gate, audit re-tests), Holm
  complexity.py            canonical compressed size of strategy.py + traced executed library (Mesa) code
  guard.py                 static guard (imports, dynamic code, file/OS access, global RNG, contract presence)
  design.json / design.py  input domain, trade worlds, named literature responses, margin floors, seed pools, profiles
  controls/                innocuous perturbations, a numpy re-implementation, knockouts (incl. "ignore an input"), nulls, probes
  pipeline.py              staged acceptance pipeline, audits, reference/margins storage
  reference/<profile>/     calibration output: reference banks, margins, control outcomes, CALIBRATION.md report
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

## Emergent features and trait responses

All are computed by the judge from observed snapshots, time-resolved and dimensionless where possible.

| | Feature | Presence | Equivalence statistics |
|---|---|---|---|
| E1 | Wealth inequality | Gini grows beyond endowments; right skew | Gini at T/4 and T, top-10% share, CV, log Hill tail index, Lorenz curve |
| E2 | Carrying capacity | Population plateaus | Plateau N/N₀, relaxation time, fluctuation, N(t)/N₀ trajectory |
| E3 | Price discovery | Log-price dispersion decreases; no exchange when trade is off | Late dispersion, decay, equilibrium price level, exchange rate, dispersion trajectory |
| E4 | Spatial self-organization | Density–resource correlation beats a uniform-placement null | Density–resource rank correlation over time, Moran's I of density |
| E5 | Trait responses | — | The response of every E1–E4 statistic to every input over its whole range (including no response where Sugarscape has none), the trade on/off contrast, and 8 responses from the book by name (Figs. IV-3, IV-6, IV-13; ch. II) |

The presence checks are also recorded per run as 0/1 indicators. Their probabilities over the domain must match too, so the collapse and no-death regions have to coincide.

## The input domain

Every input of Mesa's model is sampled jointly at random. Each point is run with trade on **and** off.

| input | sampled as | range |
|---|---|---|
| initial population | log-uniform | 50 – 800 |
| regrowth rate | log-uniform | 0.25 – 4 |
| vision | min U{1..5} + span U{0..7} | 1 – 12 |
| metabolism | min U{1..3} + span U{0..4} | 1 – 7 |
| endowment | min U{5..50} + span U{0..50} | 5 – 100 |

- Canonical runs are stored in two banks:
  - a **screening bank**: 384 scrambled-Sobol' points × 2 worlds × 4 seeds;
  - a **holdout bank**: 192 i.i.d. points, used only by audits.
- Each candidate is tested at its own random subset of the bank, on fresh seeds, so it never sees fixed design points it could overfit.
- [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) gives the precedent for each range.

## Acceptance pipeline

| Stage | What it checks |
|---|---|
| 0 | The static guard, plus traced canonical complexity. C must drop by at least max(8 B, 0.1%·C). |
| 1 | Smoke run and determinism. If trajectories are bit-identical to the parent, the change is labelled `refactor` (a label only, never a reason to accept). |
| 2 | Presence of E1–E4 at the default inputs, on fresh seeds. This is a cheap gate. |
| 3 | E1–E4 over a fresh random domain sample, in sequential looks of 32 → 96 → 192 points per world (2 seeds each). Each statistic gets three tests: the noise-corrected RMS discrepancy over the domain (`l2`), the ratio of seed-to-seed variance (`spread`), and where the statistic exists at all (`defined`). |
| 4 | E5 on the same sample. For every statistic × input, the regression of the discrepancy on the inputs (`slope`), the trade contrast, and the named literature responses. |
| Audit | Every K accepted mechanism changes: fresh i.i.d. holdout points on holdout seeds. The most discrepant points are re-tested with more seeds, and a Markov coverage bound is reported. On failure, the unaudited changes are rolled back. |

How the tests are combined:
- **Acceptance.** Every test must pass, with its confidence interval inside its margin. This is an intersection-union test, so the overall error rate is α without a multiplicity correction.
- **Certificates.** These need only one feature to break, so they use Holm correction.
- **Baseline.** Comparisons are always against canonical Sugarscape, never against the parent.

## Margins and calibration

Margins are measured in canonical's typical seed-to-seed SD of each statistic over the domain, or in probability units for indicators. Each margin is:

> margin = max(floor, relative floor × canonical's own response, largest upper confidence bound produced by an innocuous perturbation)

- **Floors:**
  - RMS discrepancy: 1 SD.
  - Indicators: 0.25.
  - Slopes: 2 SD across an input's whole range, or 20% of canonical's response.
  - Trade contrast: 0.5 SD, or 20%.
  - Variance ratio: 2.
- **Innocuous perturbations:** ±5% regrowth, growback after agents act, and ±1 endowment range.

`calibrate_judge.py` checks the judge against controls:
- **Positive controls** must pass at least 95% of replications. These are an independent plain-numpy re-implementation (no Mesa), and the same rules with different random draws.
- **Negative controls** must be rejected, with their targeted features failing:
  - knockouts: no trade, no growback, random movement, uniform vision, trading although trade is disabled;
  - "ignore an input" knockouts: population, regrowth, vision, metabolism and endowment. These count only for inputs canonical actually responds to. Otherwise they are reported as "input not required".
- **Null societies** must also fail their targeted features: random walkers with multiplicative random wealth, static agents, and noise exchange at random prices with no welfare check.
- **Probes** are reported but never part of the pass criterion. They are knockouts whose effect is an open question: removing the MRS-crossing stop, and zero-intelligence random-price trading.
- **Separation check.** A discrepancy statistic that no knockout or null society moves beyond its margin is dropped as non-discriminative.

The loop refuses to start if calibration is missing, stale (the judge or design changed) or failing. `judge/reference/<profile>/CALIBRATION.md` reports:
- the canonical trait-response map;
- which literature responses hold in canonical G1MT;
- every control's outcome.

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

pytest test_judge.py -v                                   # judge tests
JUDGE_WORKERS=4 python calibrate_judge.py --profile full  # required before a real run (~3-4 h on 4 cores)
python autoresearch.py --profile full                     # research loop (needs GOOGLE_API_KEY; pushes to agent_repo branch)
python autoresearch.py --profile full --no-push           # commit locally only
python certify.py --profile full --knockouts knockouts/ --generate   # certificate for the final model
```

Costs, measured on 4 cores (mean canonical run 3.3 s):
- A candidate that fails clearly is rejected after the first look: 128 runs, about 2 min.
- An accepted candidate needs the full 192-point look: 768 runs, about 11 min.

The `smoke` profile (100 steps, small samples, wide margin floors) exists to exercise every code path quickly:

```bash
python calibrate_judge.py --profile smoke
python autoresearch.py --profile smoke --generations 1 --no-git --allow-failed-calibration
```

Smoke results are **not** scientifically valid.

## Pre-registered choices

These are set in `judge/design.json` and justified in [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md):

- **Target:** E1–E4 plus E5, which requires all trait responses. Price discovery is required when trade is on, so a model without exchange can never pass. The bargaining rule itself is not required.
- **Inputs and ranges:** the domain table above, with every point run with trade on and off.
- **Reference machine:** CPython + stdlib + numpy.
- **Compute (full profile):**
  - Reference banks: 384 + 192 points × 2 worlds × 4 seeds.
  - Candidates: 2 seeds per point at 32 → 96 → 192 points per world.
  - Audits: 128 holdout points, plus 4 re-tested points with 8 + 8 seeds, every 5 accepted mechanism changes.
