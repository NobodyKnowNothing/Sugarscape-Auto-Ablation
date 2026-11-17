# Sugarscape Structural Ablation — Agent Directive

You are an LLM agent inside a **auto research ratchet loop** (`autoresearch.py`). Propose surgical edits to `strategy.py` that reduce its complexity score while preserving Sugarscape G1MT emergent behavior.

## Architecture

- `autoresearch.py` — ratchet loop: calls you, evaluates, commits or reverts
- `edit_harness.py` — mini-SWE-agent: you output surgical edits, NOT full files
- `validation.py` — 3-stage fail-fast waterfall (tests your proposals)
- `complexity.py` — configurable scoring (your optimization target)
- `metrics.py` / `prepare.py`
- `strategy.py` — **THE ONLY FILE YOU EDIT**

## Validation Waterfall (All 3 Stages Must Pass)

| Stage | Seeds | Steps | Gate |
|-------|-------|-------|------|
| 1 Fast Filter | 1 | 50 | AST parse + no crash |
| 2 Shape Filter | 15 | 200 | KS wealth test + price-std convergence |
| 3 Heavy Gauntlet | 50 | 200 | Bonferroni t-tests, DTW, Moran's I, Wasserstein |


## API Contract (FIXED)

```python
def create_model(seed, steps, initial_population=200, endowment_min=25,
                 endowment_max=50, metabolism_min=1, metabolism_max=5,
                 vision_min=1, vision_max=5, enable_trade=True,
                 width=50, height=50) -> model

def run_model(model, steps) -> dict
    # agent_wealths, final_population, initial_population,
    # trade_prices, trade_volume, agent_positions,
    # grid_width, grid_height, steps_run
```

## Edit Commands (via edit harness)

- `replace_line <file> <line> "<content>"` / `replace_block <file> <start> <end>` + heredoc
- `str_replace <file>` + SEARCH/REPLACE block
- `insert <file> <after_line>` / `delete <file> <start> <end>`
- `check <file>` — syntax + complexity delta
- `eval <file>` — Stage 1 sanity check
- `submit "<description>"` — finalize

Always `check` after editing. Always `submit` with a one-line description.

## Rules

- **Never output the entire file** — surgical edits only
- **Never change `create_model()`/`run_model()` signatures or return schema**
- **Never modify `prepare.py`, `metrics.py`, or `validation.py`**
- **Every edit must reduce the complexity**
- **Don't break trade or spatial structure**
