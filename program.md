# Research Program: The Minimal Rules of a Sugarscape Society

You are a research agent searching for the **mechanistically minimal** agent-based model that produces the same **emergent features** as Epstein & Axtell's Sugarscape with trade (G1MT), as implemented in Mesa.

The goal is not a shorter copy of the same simulation. Your model may use different rules, data structures and random draws. It must reproduce the same *society-level* patterns, and those patterns must be statistically equivalent to Sugarscape's across many seeds and many settings of the inputs.

## What is fixed (the world and the measurements)

The judge supplies the world and does all measurement. You cannot see or change it.

- **The world:** the 50×50 sugar landscape and its mirrored spice landscape, the initial population, and the input parameters. The inputs are endowment range, metabolism range, vision range, trade on/off and regrowth rate.
- **The contract** your file must keep:

  ```
  init(env, rng)   -> state     # env holds the inputs and env["landscape"] (2 x 50 x 50 capacities)
  step(state, rng) -> None
  observe(state)   -> {"ids", "positions", "holdings", "exchanges", "resources"}
  ```

- **`exchanges`** lists this step's trades as `(i, j, dq)`, where `dq` is the change in agent i's holdings.
- **Randomness** must come from `rng`, or from a generator you seed from it.
- **Every input must be accepted and respected.** The judge varies all of them. If you remove the rule that uses an input, the model must still reproduce how Sugarscape *responds* to that input.

## What must be preserved (emergent features)

| Feature | What it means |
|---|---|
| E1 Wealth inequality | Wealth becomes unequal and right-skewed beyond the initial endowments, with Sugarscape's inequality level, tail and Lorenz curve. |
| E2 Carrying capacity | The population settles to a plateau set by the environment, with Sugarscape's level, timing and stability. |
| E3 Price discovery | Exchange ratios start dispersed and converge over time, to Sugarscape's price level, at Sugarscape's exchange rate. |
| E4 Spatial self-organization | Agents concentrate on resource-rich regions more than chance would produce, and they cluster the way Sugarscape agents do. |
| E5 Environmental response | Changing the inputs (trade, vision, density, scarcity) moves E1–E4 the way it moves them in Sugarscape. |

The judge tells you which features failed, where, and in which direction. It does not give you target numbers, so don't try to tune constants toward targets.

## What "simpler" means

Complexity is the **compressed size of a canonical form of all executed model logic**.

- Comments, docstrings, type hints and variable names do not count. Renaming or reformatting code saves nothing.
- **Every Mesa function your model executes is counted**, together with your file. Replacing Mesa machinery with small purpose-built code is worth it only when your code is smaller than what Mesa actually runs.
- Hard-coded constants count.

## How to work

Each round you are given move types to use. Think like a scientist doing ablation:

1. **State a hypothesis.** Name the rule you are changing, and say why the emergent features should survive.
2. **Prefer mechanism-level moves.** Remove a rule, substitute a simpler rule, change the representation, or collapse parallel structure. Pure refactors are allowed, and they are reported separately.
3. **Use the mechanism ledger.** Tag the rule you target. If a removal fails, that rule is probably load-bearing for some feature, and that is useful knowledge.
4. **Large rewrites are welcome** when they are coherent. A full-file rewrite in plain numpy that keeps the same society is a legitimate and valuable move.

## Hard rules

- Edit only `strategy.py`. Keep `init`, `step` and `observe` with the same signatures and `observe()` keys.
- Imports are limited to `math`, `numpy`, `mesa` and small standard-library helpers (`collections`, `itertools`, `functools`, `heapq`, `operator`, `dataclasses`, `typing`).
- Do not use file access, `exec` or `eval`, introspection tricks, or the global numpy or Python random state.
