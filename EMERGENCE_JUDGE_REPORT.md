# How Well Does the Emergence Judge Work?

**Scope:** The validation toolkit added in `66ce374` (Sep 9), which is `validation.py` plus `metrics.py`, and the September ablation runs on `ablation/sep10`, `sep11` and `sep12`. The May commits are excluded. `sep08` and `sep09` came before this toolkit existed.

**Analogy used throughout:** `strategy.py` is a **student** who keeps shortening their essay. `validation.py` is the **teacher** who checks that the essay still says the same things. Emergence means the big patterns (inequality, a stable population, prices settling, agents clustering) that appear from simple rules.

---

## TL;DR

On the recorded branches the teacher mostly checked that the student copied the same answers as before. It did not check whether the student still understood the material. Every candidate that ran to completion passed Stage 3. The student also managed to delete one of the exam questions. Ablations that removed rules from the book still passed. The book's actual experiments (trade on/off, vision, scarcity) never ran on the shortened models.

---

## Findings (most serious first)

### 1. The student deleted an exam question, and the teacher didn't notice
- **What happened:** In `sep11` Gen 8 (`3ce8da2`) the LLM removed the `#Traders` and `Price` datacollector reporters as "unused". Stage 3's population DTW check reads exactly those reporters. When they are missing, the check is **silently skipped** (`validation.py`, the `if baseline_pop_series and ...` guard). For Gens 8–25, `dtw_distance` is logged as `0.0000`, which reads like a perfect score.
- **ELI5:** The test had a question about how the population changed over time. The student tore that page out, and the teacher counted it as a perfect answer.
- **Pattern:** This is the second time it has happened. `0efa885` had already fixed the same check, which was looking up a key that didn't exist (`"Population"` instead of `"#Traders"`). The price DTW is computed but never used to reject anything.

### 2. The student grades their own homework
- **What happened:** All the raw numbers (wealths, positions, trade prices, trade volume) come from `run_model()` inside `strategy.py`, and the LLM is allowed to edit that file. Commits `fc68b91` and `bd6028d` are literally titled "Streamline the model factory and evaluation harness". Nothing outside the student watches the simulation.
- **ELI5:** The student fills in their own report card. Right now they are honest, but nothing forces them to be.

### 3. Rules from the book were removed and still passed
- **What happened:**
  - `sep11` Gen 25 (`abe6608`) deleted the MRS-crossing check (`s.mrs(...) > o.mrs(...)`). That is the Epstein & Axtell rule that stops trading once two agents' valuations cross, and it is the mechanism behind price convergence. The candidate passed all three stages. Trade volume rose to 6086 (from 5894) and Moran's I fell to −0.040.
  - Gen 12 (`a8209b2`) rewrote the movement tie-breaking rule. It had failed Stage 2 twice in Gens 10–11, and then a slightly different version passed. Every metric shifted after that (for example, price went from 0.995 to 1.085). DTW was already off at that point.
- **ELI5:** The student skipped a step in the recipe. The cake came out looking roughly the same size, so the teacher accepted it.
- **Why it slips through:** Each candidate is only checked against a fixed band around the Mesa baseline. As long as the numbers stay inside the band, behaviour can drift step by step.

### 4. The spatial-clustering check accepts random scatter
- **What happened:** Moran's I rejects only when I < −0.05. On the 5×5 grid it uses, pure randomness already gives about −1/24 ≈ **−0.042**, so a random scatter passes. The check also runs on just one seed (42) and 25 bins, and it counts empty bins as "wealth 0". Gen 25 scored −0.040, which is essentially random, and passed. Across near-identical code the value jumped between 0.001 and 0.168, so it is mostly noise.
- **ELI5:** The question was "are the kids standing in groups?" The bar was set so low that kids scattered at random also count as groups.

### 5. "Couldn't prove it's different" is treated as "proved it's the same"
- **What happened:** The Welch t-tests pass whenever they fail to find a difference. There is no equivalence test such as TOST. Stage 3 also **drops crashed seeds without saying so**, so a model that crashes more often gets a smaller, weaker test and passes more easily. Finally, `survival_rate` is just `final_population / 200`. Sweep A shows both with an identical effect size (d = 1.653). So the "7 metrics" are really about 5–6 independent ones.
- **ELI5:** "I didn't see you cheat" is not the same as "you didn't cheat". And skipping questions you got wrong made your score look *better*.

### 6. The book's real experiments ran once at the start, and the bar was lowered after they failed
- **What happened:** `parameter_sweeps.py` (trade vs no-trade, vision, carrying capacity, scarcity) runs only **once, at startup, on the un-ablated model**, and a failure only prints a warning (`autoresearch.py:828-838`). No ablated model is ever re-checked.
- The `sweep_results.json` from before `ad7494d` shows Sweep C failing `carrying_capacity_convergence` and `low_pop_high_survival`. `ad7494d` replaced those two checks with weaker ones (population > 10, survival between 20% and 50%). It also widened the bounds by √(10/n) and raised the price bound to 0.30. After that, the sweeps pass.
- **ELI5:** The science-fair experiments were done once, on the original project. When one failed, the passing grade was lowered. The shortened projects never repeated the experiments.

### 7. It takes snapshots, not movies
- **What happened:** All seven metrics are measured only at step 200. `mean_trade_price` uses only the **final step's** trades, because prices reset every step (`prepare.py` says "use final step prices"). As a result, the baseline price standard deviation is 0.27 on a mean of about 0.97. The only check that looks at change over time (DTW) is the one that got deleted (Finding 1).
- **ELI5:** Emergence is about *how* the pattern forms. The teacher only looks at the last photo of the day.

### 8. The test rewards copying the dice rolls, not reproducing the behaviour
- **What happened:** Candidates run on seeds 42–91, the same as the baseline (`SEED_BASE = 42`). A refactor that keeps every random draw in the same order reproduces its parent's numbers exactly. On `sep10`, 21 of 22 logged rows have identical metrics, and on `sep12`, 17 of 19 do.
- Meanwhile, the Stage 2 bounds are tighter than the noise of a 5-seed average. For example, the price bound is about 0.55σ, while the noise in a 5-seed mean is about 0.45σ. A rough estimate from `baseline_metrics.json` is that a model that is statistically identical but uses different random draws would be **rejected about 4 times in 10** by Stage 2 alone.
- **ELI5:** The teacher mostly checks whether you rolled the same dice in the same order. A genuinely simpler model that behaves the same but rolls the dice differently is likely to fail.
- **Consequence:** Most of the "passes" prove that a code refactor didn't change the output. That is a regression test, not evidence that the emergence survived.

### 9. Minor: bookkeeping makes the audit harder
- The `commit` column in `results.tsv` records the *parent* commit, not the candidate's.
- Generation numbers repeat, and runs restart Gen 1 on the same file.
- The committed `baseline_metrics.json` has 10 runs and no per-seed data. The real 50-seed baseline used at runtime was never committed, so these results can't be reproduced exactly from the repo.

---

## What would fix it (short list)

1. **Make missing data a failure, not a skip.** If DTW, KS or Bonferroni inputs are absent, reject the candidate.
2. **Move measurement out of `strategy.py`.** The judge should read model state directly with a fixed observer, not use `run_model()`'s self-report.
3. **Re-run the four canonical sweeps on candidates**, at least every N generations and on the final model, and restore the original carrying-capacity checks.
4. **Use equivalence tests (TOST)** with pre-registered margins, count crashed seeds as failures, and drop the duplicate `survival_rate`.
5. **Fix Moran's I.** Test significance against the null (for example with a permutation test) across many seeds, using agent density instead of zero-filled wealth.
6. **Measure dynamics.** Use price and Gini time series, time to carrying capacity, and price dispersion over time, not just step-200 snapshots.
7. **Use held-out seeds** that differ from the baseline seeds, so "same dice" can't be mistaken for "same behaviour".
8. **Add a rule checklist.** Flag removals of named book rules (the MRS-crossing stop, welfare-maximising movement) for human review, even if the metrics pass.
