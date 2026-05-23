# Sugarscape Structural Ablation Research

This is a research program for an AI agent performing **structural ablation** on the Sugarscape agent-based model from Epstein & Axtell's *Growing Artificial Societies* (1996).

## Objective

Your goal is to **simplify** the Sugarscape model implementation in `strategy.py` while **preserving** the emergent behaviors documented in the original paper. You are performing ablation — systematically removing or simplifying code to discover the **minimal viable model** that reproduces the original results.

## Background: Sugarscape (Epstein & Axtell, 1996)

The Sugarscape model demonstrates that complex social phenomena emerge from simple individual rules on a 2D landscape:

- **Wealth inequality** emerges from agents foraging sugar/spice with different metabolisms and vision
- **Trade** emerges from bilateral negotiation based on Marginal Rate of Substitution
- **Population dynamics** settle to an environmental carrying capacity
- **Spatial patterns** emerge as agents migrate toward resource peaks

### Key metrics from the paper:
1. **Gini Coefficient** ≈ 0.3–0.6 (wealth inequality emerges)
2. **Population** converges to carrying capacity
3. **Trade price** converges to geometric mean of MRS
4. **Survival rate** reflects metabolism/vision distribution
5. **Wealth CV** shows heterogeneous accumulation
6. **Spatial entropy** reflects migration to sugar/spice peaks

## The Rules

### What you CAN do:
- Modify `strategy.py` — this is the ONLY file you edit
- Simplify classes, merge logic, remove methods
- Change data structures (e.g., replace grid with simpler representation)
- Remove code paths that don't affect measured metrics
- Refactor for clarity and minimality

### What you CANNOT do:
- Modify `prepare.py` — it is read-only (baseline runner + evaluation harness)
- Modify `metrics.py` — it is read-only (fixed metric definitions)
- Change the API contract: `create_model()` and `run_model()` signatures are fixed
- Change the output format of `run_model()` — it must return the standard metrics dict

### Constraints:
- The simplified model must produce metrics within the error bounds defined in `metrics.py`
- Simpler is always better: fewer lines, fewer classes, fewer methods
- If a simplification maintains metrics, it MUST be kept
- The Gini coefficient and survival rate have TIGHT bounds (ε = 0.05)
- Trade metrics have looser bounds (ε = 0.15–0.20)

## Strategy for Ablation

Think like a scientist performing ablation studies:

1. **Start with easy wins**: Can helper methods be inlined? Can classes be merged?
2. **Test removal of features**: Does removing trade change the Gini? Does vision=1 preserve population dynamics?
3. **Simplify data structures**: Can the grid be a simple dict instead of a class?
4. **Reduce algorithmic complexity**: Can movement use a simpler heuristic?
5. **Question every line**: If you can't explain why a line is needed for the metrics, try removing it.

## Output Format

For EACH variant, output:
1. A brief one-line description of what was simplified
2. A **complexity score** (lines of code, number of classes/methods)
3. The complete `strategy.py` code in a ```python code block

Number them: **Variant 1:**, **Variant 2:**, etc.

## Success Criteria

A simplification is successful if:
- ALL metrics remain within error bounds of the baseline
- The code is measurably simpler (fewer lines, classes, or methods)
- The code is still readable and correct

The ultimate goal is to discover: **What is the minimal code that produces Sugarscape's emergent properties?**
