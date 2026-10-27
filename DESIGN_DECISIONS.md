# Design decisions: trait responses, sweeping the input space, and trade

**Builds on:** [PROPOSAL_MVM_REDESIGN.md](PROPOSAL_MVM_REDESIGN.md) §6, which left three questions to the owner. The owner's direction:
- "All trait responses should be required in some way. I want to know how much you can still throw away and get empirically the same thing as Sugarscape (the features again, not an identical copy)."
- Sweep the inputs "as wide as possible without being impossible to compute", in the spirit of reservoir-computing (ESN) hyperparameter search, which estimates from a random sample instead of sweeping every value.
- For trade, follow the precedent in ABM, ALife and related research, given what this experiment is trying to do.

Each section gives the precedent, what follows from it, the decision, and what was implemented. Calibration results are in [judge/reference/full/CALIBRATION.md](judge/reference/full/CALIBRATION.md) once the full calibration has run.

## Summary

| Question | Decision |
|---|---|
| Which trait responses are required? | **All of them, quantitatively, over the whole input domain.** Each feature statistic must respond to every input (population, regrowth, the mean and spread of vision, metabolism and endowment) and to trade on/off as Sugarscape does, within pre-registered margins. That includes *not* responding where Sugarscape doesn't. Eight responses reported in Epstein & Axtell's book are additionally checked by name and must show the right sign. |
| How are the inputs swept? | **A fresh random sample of a wide domain for every evaluation, rather than a grid or fixed points.** The ESN connection holds for the *sampling*. Random sampling of wide, log-scaled ranges with a few seeds per configuration needs a number of points that does not grow with the number of inputs. The connection does not hold for the *goal*: ESN search optimizes, this judge verifies. So the estimators and tests come from computer experiments, ABM sensitivity analysis and statistical model checking. Adaptive search is kept for falsification only. |
| Is trade required? | **Yes, as a switch, not as a mechanism.** Every point is run with trade on and off. Both societies, and the difference between them, must be reproduced. Price discovery is required when trade is on, and no exchange may happen when it is off. The bargaining rule itself (MRS-crossing stop, geometric-mean price, repeated bargaining) is not required. Whether simpler trading reproduces the same signatures is the empirical question the loop answers. |

---

## 1. What "the same thing as Sugarscape" means: trait responses

### Precedent

- **Docking (Axtell, Axelrod, Epstein & Cohen 1996).** This paper aligned Sugarscape with Axelrod's culture model. It distinguished three standards:
  - *Numerical identity*: the same numbers.
  - *Distributional equivalence*: statistically indistinguishable output distributions.
  - *Relational equivalence*: the same qualitative relationships between inputs and outputs.

  The CAR replication project later reported relational equivalence for all eight models it replicated. The owner's target, "the features again, not an identical copy", is distributional equivalence of the features plus relational equivalence of the responses. It is not numerical identity.
- **Replication standards (Wilensky & Rand 2007; follow-up JASSS 13(4)).** Judging whether a replication succeeded needs pre-agreed standards. The power of statistical replication tests is set by the available data.
- **Burden of proof (Robinson & Froese 2004; Robinson, Duursma & Marshall 2005).** Classical tests default to "no difference", so low power makes validation easier. Equivalence tests reverse the null. The regression-based version also checks agreement *across conditions*, through slope and intercept, not just on average. This is the template for the trait-response test used here.
- **Pattern-oriented modelling (Grimm et al. 2005; Grimm & Railsback 2012).**
  - A single pattern at one scale does not constrain model structure.
  - Several weak patterns at different scales and levels jointly filter out wrong structures.
  - The problem POM addresses is **equifinality**: many model structures reproduce one pattern.
- **Minimal models reproduce single patterns cheaply.** Kinetic exchange models, which are random pairwise exchange with no space, vision or metabolism, already produce Boltzmann/Gamma wealth distributions. With heterogeneous saving they produce Pareto tails (see the Patriarca–Chakraborti and kinetic-exchange reviews). So a matching wealth distribution at one setting is weak evidence of the same mechanism. Matching how inequality *responds* to vision, trade and metabolism is much stronger evidence.
- **Robustness analysis (Grimm & Berger 2016; Thiele & Grimm 2015).** Robustness analysis is "the systematic deconstruction of a model by forcefully changing its parameters, structure and representation of processes", which is what the research loop does. "Breaking" a model means finding the conditions under which its mechanisms no longer explain the phenomena. That requires the conditions, the inputs, to be part of the test.
- **Minimal model explanations (Batterman & Rice 2014).** A minimal model explains why a whole class of systems shares large-scale behaviour by showing that the details distinguishing them are irrelevant. Here the "large-scale behaviour" is the feature set *and its dependence on the inputs*. The loop searches for the smallest member of that class.

### Decision

**All trait responses are required, quantitatively, over the whole domain.** For every feature statistic (E1–E4), in each trade world:

1. **Distribution over the domain (E1–E4, `l2` test).** The root-mean-square discrepancy between the candidate's and canonical's mean statistic, taken over the input domain, must be below the margin. This is the "small Lyapunov-like distance" between the two feature maps. It is measured in units of canonical's seed-to-seed SD, and noise-corrected so seed noise does not count as discrepancy.
2. **Every response (E5, `slope` test).** The per-point discrepancy is regressed on the eight input regressors: log population, log regrowth, and the mean and spread of vision, metabolism and endowment. The regression is weighted by each point's noise level, estimated from canonical's seeds and shrunk toward the domain-wide level, with HC3 robust errors. Each coefficient must be equivalent to zero. That means the candidate's response to each input across its whole range matches canonical's. The margin is max(2 SD, 20% of canonical's own response, the largest shift an innocuous perturbation produces).
   - A response canonical *lacks* is also required to be absent. A spurious response fails.
   - Responses smaller than the per-point noise are invisible to point-by-point comparison but detectable when pooled over the domain. That is why the responses are tested explicitly instead of being left to the `l2` test.
3. **Trade contrast (E5, `contrast` test).** The trade on/off effect, paired by point, must match.
4. **Variability and existence.** Seed-to-seed variability (`spread`) must be within a factor of 2. It is measured as the mean absolute difference between seeds at the same point: a robust scale, unbiased for any seed count. This catches deterministic or twice-as-noisy candidates. *Where* each statistic exists at all (`defined`) must also match, which covers survival and trade persistence: collapse and no-death regions must coincide.
5. **Named responses (E5, `named` test).** Eight responses from the book are additionally checked as signed effects in the candidate itself. They feed the qualitative feedback the LLM sees ("does not reproduce 'carrying capacity rises with mean vision' (IV-6a)"). A response the canonical model does not show over this domain is reported and not required.

| id | claim | source |
|---|---|---|
| IV-6a/b | carrying capacity rises with mean vision (with / without trade) | Epstein & Axtell 1996, Fig. IV-6; Mesa's G1MT `tests.py` |
| IV-6c | carrying capacity is higher with trade | Fig. IV-6; Mesa's G1MT `tests.py` |
| IV-13a | inequality falls with mean vision | Fig. IV-13 (a listed target in Mesa's `tests.py`) |
| IV-13b | inequality is higher with trade | Fig. IV-13 (a listed target in Mesa's `tests.py`) |
| IV-3 | price dispersion falls over time | Fig. IV-3, p. 109; Mesa's G1MT `tests.py` |
| II-met | carrying capacity falls with mean metabolism | ch. II |
| II-growback | carrying capacity rises with the regrowth rate | ch. II (rule G) |

### What the pilot showed

The pilot ran canonical at 64 points × 2 worlds × 2 seeds over the domain below. Response magnitudes, in units of the statistic's seed-to-seed SD, across each input's full range:

| | log N₀ | log regrowth | vision mean | vision spread | metab. mean | metab. spread | endow. mean | endow. spread |
|---|---|---|---|---|---|---|---|---|
| plateau N∞/N₀ (trade) | −5.0 | +6.4 | **+6.8** | −2.5 | **−49** | +10.5 | +5.3 | −3.8 |
| plateau N∞/N₀ (no trade) | −4.7 | +6.2 | **+11.3** | −5.0 | **−53** | +10.0 | +4.6 | −4.0 |
| Gini at T (trade) | +1.4 | −2.1 | **−1.4** | +1.7 | −1.2 | +3.5 | −0.1 | +0.2 |
| density–resource corr. at T (trade) | +6.3 | +2.8 | +1.3 | +0.6 | −22 | +7.7 | +4.1 | −0.1 |
| late price dispersion (trade) | +5.3 | +1.7 | −10.4 | +0.6 | +2.4 | −3.6 | +0.3 | +1.0 |

These were estimated from 64 points, so they are indicative. The calibration report recomputes them from 384 points. Averaged over the domain:
- Trade raises the plateau by **+1.08 SD** (SE 0.20).
- Trade raises the Gini by **+0.83 SD** (SE 0.21).

The 384-point screening bank of the full calibration confirms 7 of the 8 named responses across the whole domain, not just at the defaults:

| id | canonical effect over the domain | 95% one-sided bound | holds |
|---|---|---|---|
| IV-6a carrying capacity rises with vision (trade) | +7.6 SD | +5.1 | yes |
| IV-6b same, without trade | +8.1 SD | +5.6 | yes |
| IV-6c carrying capacity higher with trade | +0.90 SD | +0.79 | yes |
| IV-13a inequality falls with vision | −0.51 SD (wrong direction) | −1.37 | **no** |
| IV-13b inequality higher with trade | +0.64 SD | +0.56 | yes |
| IV-3 price dispersion falls over time | log ratio −1.34 | −1.24 | yes |
| II-met carrying capacity falls with metabolism | +48 SD | +46 | yes |
| II-growback carrying capacity rises with regrowth | +4.6 SD | +3.0 | yes |

Fig. IV-13a does not hold robustly in Mesa's G1MT over this domain. Mesa's own test suite never implemented it either; it is listed as a TODO. So it is reported, not required by name. "All trait responses required" still applies to it: the slope test requires a candidate to match canonical's *actual* response of the Gini to vision, whatever its sign. The reference is the model, not the book's claim about it.

Responses span three orders of magnitude, from about 1 SD to about 50 SD. That is why slope margins combine an absolute floor with a relative one.

### What "required" means when an input does not matter

For each input there is an "ignore this input" knockout: the model behaves as at Mesa's default whatever the input says. This covers `ko_ignore_population`, `_regrowth`, `_vision`, `_metabolism`, `_endowment` and `ko_trade_always`.
- If canonical responds to an input beyond margin, its knockout must be rejected. This is the calibration check that "all trait responses are required" actually has teeth.
- If canonical does not respond to an input, the knockout is reported as *input not required*. That is itself a direct answer to "how much can you throw away".

---

## 2. Sweeping the input space: the reservoir-computing question

### What ESN practice actually does

- **Bergstra & Bengio (2012).** Random search beats grid search for hyperparameters. Only a few hyperparameters matter for any given task (low effective dimensionality), and *which* ones varies. A grid spends most trials repeating values of the unimportant ones. A random sample gives every trial a distinct value of every hyperparameter.
- **Hinaut & Trouvain (2021, "Which hype for my new task?")**, building on Lukoševičius's practical ESN guide (2012).
  - Practitioners commonly use grids over more than two hyperparameters and keep the spectral radius below 1. Both are sub-optimal.
  - Their remedy is random search, read by looking at how performance depends on each hyperparameter to understand its *influence* and find robust regions. ReservoirPy ships this method, with hyperopt as the search engine; its documentation calls random search "the most robust way" to set the spectral radius.
  - The tooling in this family samples scale-like hyperparameters (spectral radius, input scaling, ridge) log-uniformly, and averages each configuration over a few reservoir instances (seeds). The R package reservoirnet, for example, uses log-uniform draws and the mean over 2 runs per configuration.

### Does the connection make sense?

**For sampling, yes.** Sugarscape has eight continuous or integer inputs plus a switch.
- A grid with 5 levels per input needs 5⁸ ≈ 390,000 points.
- The random-sampling argument applies unchanged. An unbiased estimate of a domain average, or of a response coefficient, has error that shrinks with the number of sampled points regardless of how many inputs there are.
- "Plot the outcome against each input to see its influence" is exactly the trait-response map this judge computes. CALIBRATION.md tabulates it for canonical.

**For the goal, no.**
- ESN search is **optimization**. Adaptive searchers (TPE, Bayesian optimization) deliberately concentrate samples where performance is good.
- This judge does **verification**: it must show equivalence *everywhere* in the domain. Sampling concentrated in good regions would hide exactly the regions where a candidate fails.
- So the estimators and tests come from other fields:
  - **Computer experiments and ABM sensitivity analysis.** Space-filling Latin-hypercube or Sobol' designs, regression- and variance-based sensitivity analysis, and the ~10·d points rule for a first emulator (Thiele, Kurth & Grimm 2014; ten Broeke, van Voorn & Ligtenberg 2016; Loeppky, Sacks & Welch 2009).
  - **Statistical model checking.** Monte-Carlo sampling with a priori error bounds (Okamoto/Chernoff–Hoeffding, Clopper–Pearson).

**Adaptive search does have a role: falsification.**
- Chérel, Cottineau & Reuillon (2015) argue that validation must also look for *unexpected* patterns, and use novelty search to explore a model's pattern space.
- Scenario discovery (Bryant & Lempert, PRIM) finds the regions of input space where a model's output fails.
- The audit therefore re-tests the most discrepant sampled points with more seeds and reports confirmed failures with their input values.

### Decisions

**The domain.** It contains every input of Mesa's `sugarscape_g1mt`. Ranges were taken from precedent and then widened until the pilot map showed both phase boundaries inside the domain.

| input | sampled as | range | precedent |
|---|---|---|---|
| initial population N₀ | log-uniform | 50 – 800 | book 400; Mesa 200; NetLogo models up to ~1000 |
| regrowth rate | log-uniform | 0.25 – 4 | book α = 1; α ≥ 4 is instant growback (cell capacity ≤ 4) |
| vision | min U{1..5}, span U{0..7} | 1 – 12 | book 1–6; Mesa 1–5; Mesa's tests sweep vision_max 1–9 |
| metabolism | min U{1..3}, span U{0..4} | 1 – 7 | book 1–4; Mesa 1–5 |
| endowment | min U{5..50}, span U{0..50} | 5 – 100 | book 5–25; Mesa 25–50 |
| trade | both, paired | on / off | Mesa's `enable_trade`; book ch. IV |

- Traits are sampled as (min, span), so homogeneous populations (span 0) are part of the domain.
- In the pilot, about 10% of the domain collapses (high metabolism with slow regrowth) and about 11% never loses an agent (metabolism fixed at 1). The rest spans every regime in between.
- Mean run time was 3.3 s (median 1.7 s, max 62 s), so this width is affordable.

**Sampling.**
- *Screening bank.* Canonical runs at 384 scrambled-Sobol' points × 2 worlds × 4 seeds.
- *Holdout bank.* 192 i.i.d. uniform points × 2 worlds × 4 seeds, used by audits only.
- *Fresh subsets.* Every candidate evaluation draws its own random subset, keyed by the candidate's hash, and runs it on fresh seeds. No candidate is ever tested on fixed design points it could overfit.
- *Many points, few seeds.* The candidate gets 2 seeds per point. The variance of a domain-level estimate scales as (1 + n_cand/n_ref)/budget, so spreading runs across points beats repeating seeds. Two seeds are kept so seed noise can be removed from the discrepancy estimate.

**Sequential looks.**
- The candidate is screened at 32, then 96, then 256 points per world.
- α is spent mostly on the last look (0.0025, 0.0075, 0.04), because early looks exist to reject, not to accept. Clear failures are rejected at the first look, after 128 runs (about 2 minutes on 4 cores).
- Acceptance needs every one of ~300 tests to pass. In practice that takes the full 256 points (1,024 runs, about 15 minutes on 4 cores).

**Power check (design stage).** Before trusting this, canonical Sugarscape was compared against *itself* on disjoint seeds of the screening bank (2 vs 2 seeds, noisier than real use). This changed three things:
- *The seed-variance test (variance ratio, margin 2) almost never resolved.* Variance ratios of heavy-tailed statistics are too imprecise. It became the robust mean-absolute-difference ratio with a factor-2 margin.
- *A few slope tests on price statistics stayed undecided.* Their noise is very uneven across the domain. Slopes became noise-weighted.
- *Two heavy-tailed E2 statistics remained the only unreliable tests.* Relaxation time and late fluctuation are exactly 0 in the no-death regime and erratic near collapse. They were demoted to diagnostics; population dynamics stay tested through the whole N(t)/N₀ trajectory.

After these changes, an equivalent model passed every test in 10 of 12 random 256-point replications even in the noisier 2-vs-2 setup. The held-out positive controls were not used for any of these decisions.

**What a passing model can claim over the domain.** The `l2` upper bound gives an assumption-free coverage bound by Markov's inequality: the fraction of the domain where a statistic departs by more than *c* seed-SDs is at most UCB²/c². Audits report this per statistic and in total (a union bound).

---

## 3. Trade

### Precedent

- **The source results are contrasts.**
  - Epstein & Axtell's ch. IV results about trade are comparisons between trade and no trade:
    - trade raises carrying capacity (Fig. IV-6);
    - trade raises inequality (Fig. IV-13);
    - prices converge toward, but not exactly to, equilibrium (the variance of the trade price falls over time, Fig. IV-3, p. 109).
  - With finite lives and evolving preferences, prices never settle.
  - Mesa's own G1MT test suite treats IV-3 and IV-6 as the model's reproducibility targets, and lists IV-13 as a further target.
  - Mesa's README summarizes G1MT with one claim: "an emergent price equilibrium can happen via a decentralized dynamics."
- **Trade is the least well specified rule.**
  - Kehoe's formal Z specification of Sugarscape (2015) documents missing and ambiguous rule details.
  - The Multi-Level Mesa recreation reports a trade volume "much less" than the book's, with a heavily skewed distribution.
  - Bigbee, Cioffi-Revilla & Luke (2007) replicated the book in MASON. They noted that there was scant evidence of significant replication of its rules and outcomes in other toolkits.
  - Mesa's G1MT itself deviates from the book. Its code trades with every agent within vision, not only adjacent neighbours. It also searches diamond-shaped neighbourhoods, although its README describes the book's four lattice directions.
  - So the bargaining *details* are not a stable target even among faithful replications. The contrasts are.
- **Economics on how much intelligence exchange needs.**
  - *Zero-intelligence traders (Gode & Sunder 1993).* Traders who bid at random under a budget constraint reach near-100% allocative efficiency in double auctions: "market structure, not trader intelligence". Whether such traders also get price *convergence* is contested; later work reports that prices need not converge.
  - *Decentralized exchange (Axtell 2005).* Axtell contrasts the computational implausibility of Walrasian equilibrium with decentralized bilateral exchange. Decentralized exchange reaches stable but path-dependent states with price dispersion, allocations outside the core and a changed wealth distribution. His 1999 working paper puts bilateral exchange in polynomial time.
  - *Kinetic exchange models.* Random pairwise exchange alone shapes wealth distributions.

### What the experiment is trying to do

- The question is how much of G1MT can be thrown away while the society's features and their responses survive.
- If trade were optional, the minimal model would simply delete it and answer a different question: the minimal Sugarscape *without* trade, which is the book's ch. II society.
- If the *bargaining rule* were required, the experiment would assume its own answer. The ZI literature says this is exactly the part whose necessity is in doubt.

### Decision

1. **Trade is a required input, tested as a paired contrast.** Every sampled point is run with trade on and off.
   - The candidate must reproduce both societies and the trade effect: the `contrast` tests and the named responses IV-6c and IV-13b.
   - With trade off there must be no exchange. The `trading` indicator must be 0, which catches `ko_trade_always`.
2. **Price discovery (E3) is required when trade is on.** This covers late dispersion, dispersion decay, price level, exchange rate and the dispersion trajectory, plus IV-3 by name.
3. **The bargaining mechanism is not required.** The MRS-crossing stop, the geometric-mean price, repeated bargaining and the partner radius are all removable *if* the signatures survive.
4. **Uncertain knockouts are probes, not negative controls.** The old smoke calibration already found that removing the MRS-crossing stop passed every feature. Random-price trading with individual rationality (ZI-C traders) was undecided on price discovery.
   - Treating either as a must-fail control would bake in an answer the literature disputes. Both are now **probes**: run, reported, and excluded from the pass criterion.
   - The certain E3 negative is `null_noise_exchange`: one exchange per encounter at a random price with *no* welfare or MRS check. Its dispersion of log prices cannot decay.

---

## 4. What changed in the judge

| | Before | Now |
|---|---|---|
| Inputs | 8 one-at-a-time sweep points + 24 fixed LHS points over 4 inputs | the whole domain of all 8 inputs (min/span per trait) × trade on/off |
| Sampling | fixed points, reused for every candidate | a fresh random subset of a 384-point Sobol' bank per candidate; a separate i.i.d. holdout bank for audits |
| Seeds | 20 → 40 → 60 per point at 8 points | 2 per point at 32 → 96 → 256 points per world, α spent mostly on the last look |
| Feature equivalence | per-point TOST | noise-corrected domain RMS discrepancy (`l2`), robust seed-spread ratio, definedness |
| Responses (E5) | difference-of-differences between fixed sweep points | noise-weighted slope equivalence for every statistic × input, trade contrast, named literature responses |
| Statistics | Hill index raw; dispersion decay undefined at perfect convergence | log Hill index; ε-stabilized decay; presence indicators (0/1) whose domain-wide probabilities must match |
| Controls | 4 knockouts, 3 nulls | + 6 "ignore an input" knockouts, a certain-negative noise-exchange null; MRS-stop removal and ZI-C prices moved to probes |
| Audit | 32 points on holdout seeds | fresh i.i.d. points, adversarial re-test of the worst points, Markov coverage bound |

Code: `judge/domain.py` (domain and sampling), `judge/surface.py` (tests), `judge/pipeline.py`, `judge/design.json`, `calibrate_judge.py`, `judge/controls/__init__.py`.

## 5. Limits

- The domain is a modelling choice. Equivalence is claimed over *this* domain under *this* sampling measure, and nowhere else. The ranges are pre-registered in `judge/design.json` and hashed into every result.
- Slope tests check linear projections of the responses. Nonlinear shape differences are caught only through the `l2` test, which has less power for small effects.
- Run length is fixed at 200 steps and the landscape is fixed. Long-run behaviour beyond T and other landscapes are outside the claim.
- The coverage bound uses Markov's inequality: it is valid without assumptions but loose.

## Sources

- Axtell, Axelrod, Epstein & Cohen (1996), "Aligning simulation models: a case study and results", *Computational and Mathematical Organization Theory* 1: 123–141. [Deep Blue](https://deepblue.lib.umich.edu/items/e0299674-2147-4538-9019-6b349ce33915/full); [CAR replications](https://websites.umich.edu/~axe/CAR-replications.html)
- Wilensky & Rand (2007), "Making models match: replicating an agent-based model", *JASSS* 10(4) 2. [JASSS](https://www.jasss.org/10/4/2.html); follow-up on statistical power of replication tests: [JASSS 13(4) 1](https://jasss.soc.surrey.ac.uk/13/4/1.html)
- Robinson, Duursma & Marshall (2005), "A regression-based equivalence test for model validation: shifting the burden of proof", *Tree Physiology* 25(7): 903. [record](https://researchers.westernsydney.edu.au/en/publications/a-regression-based-equivalence-test-for-model-validation-shifting/); Robinson & Froese (2004), "Model validation using equivalence tests". [record](https://digitalcommons.mtu.edu/michigantech-p/6306)
- Grimm et al. (2005), "Pattern-oriented modeling of agent-based complex systems: lessons from ecology", *Science* 310: 987. [USGS record](https://pubs.usgs.gov/publication/70161783); [overview](https://Www.Wikipedia.org/wiki/Pattern-oriented_modeling)
- Grimm & Berger (2016), "Robustness analysis: deconstructing computational models for ecological theory and applications", *Ecological Modelling* 326: 162–167. [RePEc](https://ideas.repec.org/a/eee/ecomod/v326y2016icp162-167.html)
- Batterman & Rice (2014), "Minimal model explanations", *Philosophy of Science* 81(3). [repository](https://repository.brynmawr.edu/philosophy_pubs/15)
- Equifinality and POM in socio-environmental ABMs (2020). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7489104)
- Kinetic exchange models of wealth: [Patriarca & Chakraborti review](https://arxiv.org/pdf/1305.0768); [Money exchange model outlook](https://arxiv.org/pdf/physics/0505115)
- Bergstra & Bengio (2012), "Random search for hyper-parameter optimization", *JMLR* 13: 281–305. [JMLR](https://jmlr.org/papers/v13/bergstra12a.html)
- Lukoševičius (2012), "A practical guide to applying echo state networks". [PDF](https://www.ai.rug.nl/minds/uploads/PracticalESN.pdf)
- Hinaut & Trouvain (2021), "Which hype for my new task? Hints and random search for echo state networks hyperparameters", ICANN 2021. [HAL](https://hal-univ-tlse3.archives-ouvertes.fr/INRIA/hal-03203318); [ReservoirPy hyperparameter guide](https://reservoirpy.readthedocs.io/en/stable/user_guide/hyper.html)
- Thiele, Kurth & Grimm (2014), "Facilitating parameter estimation and sensitivity analysis of agent-based models: a cookbook using NetLogo and R", *JASSS* 17(3) 11. [JASSS](https://jasss.org/17/3/11.html)
- ten Broeke, van Voorn & Ligtenberg (2016), "Which sensitivity analysis method should I use for my agent-based model?", *JASSS* 19(1) 5. [JASSS](https://www.jasss.org/19/1/5/ten_broeke.html)
- Loeppky, Sacks & Welch (2009), "Choosing the sample size of a computer experiment: a practical guide", *Technometrics*. [UBC record](https://www.stat.ubc.ca/node/3744)
- Statistical model checking sample-size bounds (Okamoto / Clopper–Pearson): [Sound statistical model checking](https://arxiv.org/pdf/2411.00559)
- Chérel, Cottineau & Reuillon (2015), "Beyond corroboration: strengthening model validation by looking for unexpected patterns", *PLOS ONE* 10(9): e0138212. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4569327)
- Bryant & Lempert, scenario discovery with PRIM. [useR! 2009 slides](https://user2009.r-project.org/slides/Bryant.pdf)
- Epstein & Axtell (1996), *Growing Artificial Societies*, ch. II and IV. [Brookings](https://www.brookings.edu/books/growing-artificial-societies); secondary accounts of the trade results: [book notes](https://hubeventsnotes.blogspot.com/2016/01/growing-artificial-societies.html), [Science News summary](https://www.math.ucdavis.edu/~saito/data/science-news/original%20CD/software/SNData/10520); original parameter values: [Runestone chapter](https://runestone.academy/ns/books/published/complex/AgentBasedModels/Sugarscape.html)
- Mesa `sugarscape_g1mt` [docs](https://mesa.readthedocs.io/stable/examples/advanced/sugarscape_g1mt.html), and the example's `tests.py` (in `projectmesa/mesa-examples` before the examples moved into Mesa: `test_decreasing_price_variance` for Fig. IV-3, `test_carrying_capacity` for Fig. IV-6, and IV-12/IV-13 as listed TODOs)
- Kehoe (2015), "The specification of Sugarscape". [arXiv:1505.06012](https://arxiv.org/abs/1505.06012v3)
- Bigbee, Cioffi-Revilla & Luke (2007), "Replication of Sugarscape using MASON". [Academia](https://www.academia.edu/5043905/Replication_of_Sugarscape_Using_MASON); Multi-Level Mesa trade-volume discrepancy: [arXiv:1904.08315](https://arxiv.org/pdf/1904.08315)
- Gode & Sunder (1993), "Allocative efficiency of markets with zero-intelligence traders", *JPE* 101(1). [RePEc](https://econpapers.repec.org/RePEc:ucp:jpolec:v:101:y:1993:i:1:p:119-37); on contested price convergence: [Krannert WP](https://Krannert.purdue.edu/research/working-papers-series/2007/1204.pdf), [Ca' Foscari WP](https://www.unive.it/web/fileadmin/user_upload/dipartimenti/DMAN/pubblicazioni_scientifiche/working_papers/2008/2008wp168.pdf)
- Axtell (2005), "The complexity of exchange", *Economic Journal* 115(504): F193–F210. [RePEc](https://ideas.repec.org/a/ecj/econjl/v115y2005i504pf193-f210.html)
- Inverse generative social science, which evolves agent rules against macro targets with a parsimony objective and is the closest precedent for this loop: Vu et al. (2019), [PubMed](https://www.pubmed.gov/33083795/); evolutionary model discovery, Gunaratne & Garibay (2020), [arXiv](https://arxiv.org/abs/1802.00435)
- Kumar et al. (2024), "Automating the search for artificial life with foundation models" (ASAL). [arXiv:2412.17799](https://arxiv.org/pdf/2412.17799)
