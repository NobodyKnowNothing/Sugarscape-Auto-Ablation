# Proposal: Finding the Minimal Rules That Produce Sugarscape's Emergent Society

**Builds on:** [EMERGENCE_JUDGE_REPORT.md](EMERGENCE_JUDGE_REPORT.md)

**Thesis (unchanged):** An LLM research loop makes creative changes to an agent-based model. Each change is scored by compression-based complexity and accepted only if rigorous statistical tests show the **same emergent features** as Sugarscape.

**What "same" means here:**
- **Not** the same simulation: not the same rules, trajectories, random draws, or Mesa's exact numbers.
- **Not** "loosely similar" either, where comparing the two would be scientifically indefensible.
- **The target:** the emergent features must be *statistically equivalent* within pre-registered margins. Those margins are no wider than the differences Sugarscape itself shows under changes that nobody would claim alter the society. In the metaphor: a small Lyapunov distance between the two systems' long-run regimes, not identical trajectories.

---

## 1. Where the repo stands

| Area | Current state | Why it blocks the thesis |
|---|---|---|
| **What "comparable" means** | Fixed bands around Mesa's means. The t-tests treat "no difference detected" as a pass. Candidates share the baseline's seeds (42–91). | Rewards reproducing the *same simulation*: refactors that preserve the random draws pass trivially. Real mechanism changes that keep the emergent features face bands narrower than the 5-seed noise (about 4 in 10 false rejections). Some changes that remove features still pass, such as sep11 Gen 25, which removed the MRS-crossing stop. |
| **Who measures** | `run_model()` and the `datacollector` inside the editable `strategy.py`. | The model reports its own results and can delete its own measurements (sep11 Gen 8 silently turned DTW off). |
| **What is measured** | Snapshots at step 200 only. Price comes from the final step only. Moran's I accepts random scatter. `survival_rate` duplicates `final_population`. | Emergence is about how patterns form over time, and that is mostly unmeasured. |
| **Paper experiments** | Run once at startup on the original model. They only warn, and the Sweep C checks were weakened after failing. | Simplified models are never checked against the causal claims. |
| **Complexity** | The default is AST nodes plus cyclomatic complexity. Gzip runs on raw text, and `gzip_ratio` rewards adding repetition. | sep11's raw gzip size fell 4499 → 2358 B, but with comments stripped it fell only 2823 → 1795 B. **About half the "simplification" was deleted prose.** |
| **Hidden mechanism** | Neighbourhoods, the grid, property layers, agent bookkeeping and scheduling all come from Mesa. | Mechanisms implemented inside the library cost 0 bytes, so they can't be minimized. |
| **Search** | 1 greedy variant per generation. The prompt asks to "inline helpers". Each day restarts from canonical. Complexity is compared only *after* about 150 s of validation. | This is code golf, not mechanism search. sep11 (159 lines) and sep12 (251 lines) never built on each other. |
| **Hygiene** | `program.md` gives the LLM the wrong bounds. The committed baseline has 10 runs and no per-seed data. `results.tsv` records the parent commit. | The LLM is misinformed and runs can't be reproduced from the repo. |

---

## 2. Principles

1. **Same world, same measurements, free rules.** The landscape, initial population, run length and observation interface are fixed. Everything about agent behaviour is open to change.
2. **The judge owns measurement.** The editable file defines dynamics only. The judge runs the model, observes it and computes every statistic.
3. **Compare emergent features, never trajectories.** All statistics are about the population as a whole, dimensionless where possible, resolved over time, and computable for *any* model that fits the interface.
4. **Equivalence must be demonstrated.** A feature passes only when its confidence interval lies inside a margin, using TOST (two one-sided tests). "Failing to find a difference" is not a pass.
5. **Margins come from Sugarscape's own robustness.** The margin is the largest shift Sugarscape shows under changes that experts agree leave the society unchanged, written down before any run. It is not Mesa's seed noise (too tight) and not a judgement call (too loose).
6. **Calibrate the judge before trusting it.** Implementations that are faithful but differently coded must pass. Non-societies and knockouts must fail.
7. **Complexity counts all executed model logic**, including library code, measured after canonicalization.
8. **The output is a certificate.** For every surviving mechanism, there is evidence of which emergent feature it is necessary for.

---

## 3. Design

### 3.1 Fixed world, generic observer
**Environment contract (fixed by the judge):**
- the `sugar-map.txt` landscape, with its spice counterpart if two goods remain
- initial population N₀ = 200
- 200 steps
- environment parameters: resource regrowth rate, landscape scale, N₀

**Model contract (all that `strategy.py` must provide):**
```
init(env, rng) -> state      step(state, rng) -> None      observe(state) -> Snapshot
```

**Generic `Snapshot`.** It holds only what any society-like model has:
- `positions: int[N,2]`
- `holdings: float[N,G]`, with G ≥ 1 goods
- `exchanges: list[(i, j, qty_given, qty_received)]` for this step
- `resources: float[G,W,H]`

Notes on the snapshot:
- There are **no** fields for vision, metabolism or spice. If the loop removes them and the emergent features survive, that is a legitimate finding.
- Wealth means total holdings, and price means the exchange ratio. The judge derives both.

**Enforcement:**
- The judge calls `step` and `observe` every tick and keeps the full time series itself.
- A missing or malformed field, or a crash on any seed, means **reject**. Nothing fails open.
- The judge lives in `judge/`, is checksummed, and its hash is logged with every result.

### 3.2 Complexity: canonical compressed size of *all executed model logic*
```
C(model) = min_{gzip-9, bz2, lzma}( |compress( canon( strategy.py ⊕ traced_dependency_code ) )| )
           − C(empty contract template)
```

**Traced dependencies (instead of a flat Mesa charge).**
- During the smoke run (and the union over 2–3 seeds), a `sys.settrace` / coverage pass records every function executed outside the "reference machine".
- The reference machine is CPython, the standard library and numpy. Like the reference machine in Kolmogorov complexity, it is fixed once and documented.
- For each executed function in Mesa (or any other third-party package), the judge pulls the source of the function, plus the class-level statements of its class, and appends it to the bundle.
- The bundle is canonicalized and compressed *jointly* with `strategy.py`, so shared structure is not double-counted.
- Results are cached by (Mesa version, set of traced functions).

**Expected effect.** Mesa is general-purpose, so the starting C will rise sharply. The loop is then rewarded for replacing general machinery with small purpose-built rules, which is exactly the "mechanistic minimum" incentive. The current flattened model is the honest starting point.

**`canon()`:**
1. Parse to AST.
2. Strip docstrings, comments, type hints and `assert`s.
3. **α-rename** every identifier to canonical short names in first-use order.
4. Normalize with `ast.unparse`.

**Guards and thresholds:**
- The static guard bans `exec`, `eval`, `compile`, `importlib`, pickled or encoded code, and file reads other than the landscape, so logic can't hide outside the measured bundle.
- The landscape data is part of the environment and is not counted.

**Rules:**
- Remove the `gzip_ratio` scorer.
- Require a minimum gain **δ** (about 1%) to beat compressor noise.
- Log the NCD (normalized compression distance) between parent and child as a measure of how large a creative leap each change was.
- Compute complexity in Stage 0, before any simulation.

### 3.3 Emergent-feature suite
Every statistic is computed by the judge from `Snapshot` time series. Each feature has two layers:
- **Presence:** a qualitative property that must hold.
- **Equivalence:** dimensionless statistics that must match Sugarscape within the margin.

| ID | Emergent feature | Presence (must hold) | Equivalence statistics (dimensionless, time-resolved) |
|---|---|---|---|
| E1 | **Wealth inequality** | Right-skewed wealth and Gini above that of the initial endowments | Lorenz curve at t = 50, 100 and 200 (sup-distance), Gini, top-10% share, tail index, and wealth CV |
| E2 | **Carrying capacity** | Population reaches a plateau (trend over the last 50 steps ≈ 0) | Plateau fraction N∞/N₀, relaxation time τ (in steps), post-transient fluctuation (CV), and the normalized trajectory N(t)/N₀ |
| E3 | **Price discovery** | Dispersion of log exchange ratios *decreases* over time | Dispersion-decay rate, final dispersion, the equilibrium price level (geometric mean over the last 50 steps), and the exchange rate per agent |
| E4 | **Spatial self-organization** | Agent density correlates with resources more strongly than in a random-walk null model (permutation test) | Density–resource rank correlation over time, and pooled Moran's I of density (permutation-tested across seeds) |
| E5 | **Environmental response** | Outcomes move in the correct direction when the *environment* changes | Elasticities of E1–E4 with respect to regrowth rate, landscape scale and N₀ (the paper's scarcity and density sweeps, restricted to parameters every model has) |

Two features need to be decided by the owner (§6): **trade raises carrying capacity** and **trade raises inequality**. These are E&A's trade-on/off results. Including them means exchange can never be removed.

Further notes:
- E3 and E4 are the features that *require interaction*. They are what separates a society from a mere wealth-generating process.
- `survival_rate` is dropped.
- All statistics are computed per seed and compared as distributions over seeds.

### 3.4 Equivalence testing with robustness-derived margins
**The reference ensemble.** Sugarscape is run (canonical flattened model) under a pre-registered set of **innocuous perturbations**. These are changes that the E&A literature, or an expert reviewer, would agree do *not* change the society:
- random-sequential versus shuffled activation order
- synchronous versus asynchronous resource growback
- alternative movement tie-break rules (random, nearest, first)
- ±5% regrowth rate
- an independent, from-scratch faithful re-implementation (pure numpy) of G1MT

**The margin.** For each statistic s:
- **Margin** Δₛ = the largest |mean shift| of s across the innocuous perturbations, relative to canonical. Statistically, this is the indifference zone: differences smaller than this are not scientifically distinguishable from "Sugarscape".
- **Separation check.** Δₛ must be smaller than the shift produced by the weakest negative control (§3.5) for that statistic. If it isn't, s cannot tell the society apart from a non-society. It is dropped or redesigned before any run.
- **Pre-registration.** Margins, statistics and perturbation sets are frozen in `judge/margins.json` and hashed before the loop starts.

**The test:**
- **Per statistic:** TOST on per-seed values, candidate versus canonical, at α = 0.05. Holm correction applies across statistics within a feature. A feature passes if all of its statistics pass and its presence check holds.
- **Distributional statistics** (the Lorenz curve) use bootstrap confidence intervals of the sup-distance against Δ.
- **Trajectories** (N(t)/N₀ and price dispersion over time) use a bootstrap confidence band of the maximum deviation between mean curves, against Δ.
- **Seeds** are fresh for each candidate, drawn from a pool disjoint from both the baseline seeds and the holdout seeds.
- **Sample size is sequential**: 10, then 30, then 60 seeds. Reject early once a confidence interval is clearly outside the margin. Accept only when every interval is inside it.

### 3.4b Coverage of the input space (seeds × upstream parameters)
Equivalence at the default configuration alone is not enough. A model can match Sugarscape at N₀ = 200 and diverge everywhere else. So equivalence is tested over a **pre-registered design over the input space**, and every design point gets its own seeds and margins.

**Input variables.** Every candidate must accept the full original input set.
- **Environment / population:** N₀, regrowth rate, landscape scale or scarcity, and run length.
- **Agent-trait inputs from the original model:** vision range, metabolism range, endowment range, and trade on/off.

If the loop removes the rule that uses an input (for example, vision), the model must still accept that input. It must also reproduce Sugarscape's *response* to it, or the input is recorded in the ledger as "response not reproduced". That makes the model fail unless the owner has excluded that response from the target features (§6).

**Design.** The four existing sweeps in `parameter_sweeps.py` (A: trade, B: vision, C: density, D: scarcity) become fixed design points:
- Sweep C's original `carrying_capacity_convergence` and `low_pop_high_survival` checks are restored.
- A space-filling sample (for example, a 20–30 point Latin hypercube) is added over the joint ranges.

**At each design point:**
- **Reference ensemble and margins.** Sugarscape's reference ensemble is run there, and margins Δₛ(x) are derived per point with the same robustness rule as §3.4.
- **Equivalence.** The candidate's E1–E4 statistics are TOST-tested against canonical *at that point*.
- **Response surface.** E5 becomes equivalence of the **response surface**. The candidate's elasticities and directional effects across the design must match Sugarscape's within margin, not just point the same way.
- **Seeds.** Every point uses fresh seeds from a disjoint pool, with seed lists logged.

**Multiplicity and cost:**
- Holm correction applies across statistics × design points.
- Each candidate is tested sequentially on a **screening subset** (the default point plus the sweep extremes, about 8 points).
- Every audit runs the **full design**. The final model and certificate always use the full design with holdout seeds.

**Reproducibility.** The design, the seed pools, the margin table per point and the Mesa/numpy versions are frozen in `judge/design.json` and hashed. Any result can be regenerated from (commit, judge hash, design hash).

### 3.5 Judge calibration (`calibrate_judge.py`, run in CI)
| Control set | Examples | Requirement |
|---|---|---|
| **Positive** (same society, different simulation) | Held-out innocuous perturbations *not* used to set the margins; the independent numpy re-implementation | Pass rate ≥ 95% |
| **Negative: null societies** | Random walkers with random multiplicative wealth (inequality without a society); agents that never move; exchange at random prices | Each fails the features it lacks, ≥ 95% of the time |
| **Negative: knockouts** | No trade, no growback, random movement, no MRS-crossing stop, uniform vision | Each fails at least the feature it should break, ≥ 95% of the time, *or* the result is recorded as "mechanism not required for the chosen features" and reviewed by a human |

The calibration output (the power table, margins and separation checks) is committed. The loop refuses to start if calibration is stale or failing.

### 3.6 Acceptance pipeline
```
Stage 0  Static guard + canonical complexity  (must decrease by ≥ δ)       ~0 s
Stage 1  Smoke: 1 seed × 50 steps, valid Snapshots                         ~2 s
Stage 2  Presence checks E1–E4 on 10 fresh seeds at the default point  (cheap qualitative gate)
Stage 3  Sequential TOST on E1–E4 (10 → 30 → 60 fresh seeds) at each screening design point
Stage 4  E5 response-surface equivalence across the screening design
Audit    Every K accepted changes: full design (sweeps A–D + LHS), holdout seeds, full margins; roll back on failure
```
- **Every candidate goes through Stages 2–4.** A change whose output is bit-identical to an accepted parent is *labelled* `refactor`, for reporting only, so the writeup can split code-level from mechanism-level reductions. Identity is never itself a reason to accept.
- **Drift.** Comparisons are always against canonical Sugarscape, never against the parent. Audits stop small, individually acceptable shifts from compounding.

### 3.7 Mechanism ledger and certificate
- **Ledger.** `mechanisms.json` enumerates the G1MT rules: growback, welfare-maximizing movement, the tie-break, Cobb-Douglas welfare, metabolism, death, vision, bilateral trade, the MRS-crossing stop, the price rule, endowment heterogeneity, and the second good.
- **Tagging.** Every proposal tags which rule it removes, merges or substitutes. The ledger stores the status and the evidence (commit, judge hash, test results).
- **Certificate** (`certify.py`). For the final model, knock out each surviving rule. Each must fail at least one emergent feature, with a confidence interval outside the margin. Each removed rule links to its passing evidence.

### 3.8 LLM research loop
- **Hypothesis-driven prompts.** Each proposal names the rule, the change, which emergent features should survive and why, and the expected ΔC. Full rewrites of a subsystem are allowed.
- **Move portfolio, 4–8 candidates per generation:** remove a rule, substitute a simpler rule, change the representation, replace library machinery with local code (now rewarded, because Mesa is counted), collapse parameters, or refactor.
- **Pareto archive** over (C, equivalence distance), MAP-Elites style. The proposer picks parents, and the archive resumes across sessions instead of restarting from canonical.
- **Feedback without Goodhart.** Return per-feature pass/fail with the direction of failure, but not margin values or baseline numbers. Holdout seeds and audit statistics are never shown.
- **Anti-gaming.** Environment parameters come from the judge, and E5 catches models that ignore them. Hard-coded constants count toward C.

---

## 4. Implementation plan

| Phase | Scope | Done when |
|---|---|---|
| **0. Stop the bleeding** | Fail closed on missing data. Count crashes. Compute complexity before validation. Demote Moran's I to diagnostic. Commit the full baseline. Fix `program.md`. Log the candidate's commit hash and the judge hash. | Replaying sep11 Gen 8 is rejected. Tests cover the fail-closed paths. |
| **1. Judge owns the run** | Environment and model contract, generic `Snapshot`, judge-side recorder, immutable `judge/` package. Adapt `canonical_strategy.py`. | The canonical model runs through the judge and reproduces today's metrics. |
| **2. Complexity v2** | `canon()`, dependency tracing and bundling, multi-compressor minimum, template offset, δ, and the static guard. Drop `gzip_ratio`. | Renaming or comment-only edits give ΔC = 0. Moving a Mesa call into equivalent local code changes C only by the real difference. |
| **3. Reference ensemble and calibration** | Innocuous perturbation set, an independent numpy re-implementation, E1–E5 statistics, margins and separation checks, controls and the power table. | ≥ 95% pass on positive controls and ≥ 95% fail on negative controls, committed and run in CI. |
| **4. Acceptance pipeline** | Stages 0–4, sequential TOST, audits with rollback, refactor labelling. | sep11 Gen 25 is either rejected or explained by calibration. A faithful re-implementation is accepted. |
| **5. Research loop** | Hypothesis prompts, move portfolio, Pareto archive with resume, mechanism ledger, `certify.py`. | One run yields a minimal model, a certificate, and separate code-level and mechanism-level ΔC. |

Phase 3 is the scientific core and the critical path. Phases 0–2 can proceed in parallel with it.

---

## 5. Success criteria
1. **Judge validity.** Faithful-but-different implementations pass at least 95% of the time. Null societies and knockouts fail at least 95% of the time. Every margin passes its separation check.
2. **Objective validity.** Edits to names, comments or format give ΔC = 0. Library logic is counted.
3. **Scientific output.** A minimal rule set whose emergent features are TOST-equivalent to Sugarscape within pre-registered, robustness-derived margins. It comes with a knockout certificate for every surviving rule.
4. **Reproducibility.** Every result can be replayed from its commit, judge hash, seed list and margin file.

---

## 6. Decisions needed from the project owner

> **Resolved (Oct 2026):** points 1 and 2 are settled in [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md).
> - **Trait responses.** All of them are required, quantitatively, over a wide random-sampled input domain.
> - **Trade.** It is a required switch, tested as a paired on/off contrast with price discovery required, but the bargaining rule itself is not required.
> - **Fixed design points.** §3.4b's fixed design points are replaced by fresh random samples of the domain.
1. **The feature list is the definition of "Sugarscape society".** Are E1–E5 the right set? Should the trade-on/off results (trade raises carrying capacity and inequality) be included? Including them means exchange is never removable.
2. **Input-space scope.** What are the ranges for each input in the design (§3.4b)? Which agent-trait responses, such as the effect of vision on population, are required emergent features, and which may be dropped along with their rule?
3. **The innocuous perturbation set.** Which changes count as "the same society"? This sets every margin, so it should be reviewed before Phase 3 and then frozen.
4. **The reference machine boundary.** Is it CPython, the standard library and numpy? Should numpy be counted too?
5. **Compute budget per candidate.** This sets the sequential-testing ceiling and the audit interval K.

---

## 7. Risks
- **Margins are only as defensible as the perturbation set.** Mitigations: pre-registration, held-out positive controls and separation checks. The perturbation set should be reviewed by someone other than whoever runs the loop.
- **Some knockouts may pass.** That means a rule is not needed for the chosen features, which is a genuine finding. Record it; never tune it away.
- **Tracing captures only executed code paths.** Union the trace over several seeds and over the sweep conditions. Re-trace whenever the import set changes.
- **Compressed size is a proxy for description length**, not a proof. The multi-compressor minimum, canonicalization and δ keep it stable enough to rank changes.
