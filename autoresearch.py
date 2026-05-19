#!/usr/bin/env python3
"""
Evolutionary Autoresearch Pipeline for Polymarket BTC 5-min Trading.

Adapts Karpathy's autoresearch pattern:
  - LLM (Gemma-4-26b via google-genai) acts as the mutation operator
  - strategy.py is the single mutable file (inside agent_repo/)
  - Git ratchet: winners get committed, losers get reverted
  - Parallel variants: multiple mutations tested per generation
  - Market resolution is the ground truth metric

Usage:
    export GOOGLE_API_KEY="your_key_here"
    python autoresearch.py

The loop runs indefinitely — kill with Ctrl+C.
"""

import json
import math
import os
import re
import subprocess
import sys
import textwrap
import time
import traceback
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "gemma-4-26b-a4b-it"
VARIANTS_PER_GENERATION = 5       # how many strategy mutations per round
AGENT_REPO = Path(__file__).parent / "agent_repo"
STRATEGY_FILE = AGENT_REPO / "strategy.py"
RESULTS_FILE = Path(__file__).parent / "results.tsv"
MIN_BOOK_DEPTH_USD = 10.0          # realistic liquidity threshold (increased from 1.0)
PROGRAM_FILE = Path(__file__).parent / "program.md"

# API key — falls back to env vars
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Git helpers (operate on agent_repo which has its own .git)
# ---------------------------------------------------------------------------
def git(cmd: str, cwd: str | None = None) -> str:
    """Run a git command in agent_repo and return stdout."""
    result = subprocess.run(
        f"git {cmd}",
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd or str(AGENT_REPO),
    )
    if result.returncode != 0 and "nothing to commit" not in result.stderr:
        # Only warn, don't crash — some git ops are benign failures
        pass
    return result.stdout.strip()


def git_commit(message: str):
    git("add strategy.py")
    git(f'commit -m "{message}"')


def git_revert_to(commit_hash: str):
    git(f"reset --hard {commit_hash}")


def git_current_hash() -> str:
    return git("rev-parse --short HEAD")


def git_create_branch(name: str):
    # If branch exists, just check it out
    existing = git("branch --list " + name)
    if existing.strip():
        git(f"checkout {name}")
    else:
        git(f"checkout -b {name}")


# ---------------------------------------------------------------------------
# Results tracking
# ---------------------------------------------------------------------------
def init_results():
    """Create results.tsv if it doesn't exist."""
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "generation\tvariant\tcommit\tprediction\tbet_amount\toutcome\tstatus\tprofit\troi\tdescription"
            "\tavg_fill\tbook_depth\tspread\tresolution_mid\n"
        )


def append_result(gen: int, var: int, commit: str, prediction: str, bet_amount: float,
                  outcome: str, status: str, profit: float, roi: float, desc: str,
                  avg_fill: float = 0.0, book_depth: float = 0.0, spread: float = 0.0,
                  resolution_mid: float = 0.0):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{gen}\t{var}\t{commit}\t{prediction}\t{bet_amount:.2f}\t{outcome}\t{status}\t{profit:.2f}\t{roi:.4f}\t{desc}"
                f"\t{avg_fill:.4f}\t{book_depth:.2f}\t{spread:.4f}\t{resolution_mid:.4f}\n")


def read_results_history(max_lines: int = 100) -> str:
    """Read recent results.tsv as a string for the LLM context, capped to avoid blowing up the context window."""
    if RESULTS_FILE.exists():
        lines = RESULTS_FILE.read_text().strip().split("\n")
        if len(lines) <= max_lines + 1:
            return "\n".join(lines)
        header = lines[0]
        recent_lines = lines[-max_lines:]
        return f"{header}\n... ({len(lines) - max_lines - 1} older rows truncated) ...\n" + "\n".join(recent_lines)
    return "(no results yet)"


def compute_stats(only_variant_zero: bool = True) -> tuple[int, int, float, float]:
    """Parse results.tsv and return (wins, total, rate, pnl).
    If only_variant_zero is True, only counts the performance of the 'control' strategy.
    """
    if not RESULTS_FILE.exists():
        return 0, 0, 0.0, 0.0
    lines = RESULTS_FILE.read_text().strip().split("\n")[1:]  # skip header
    wins = 0
    total = 0
    pnl = 0.0
    for l in lines:
        if not l.strip():
            continue
            
        parts = l.split("\t")
        
        # If true, we only track the real 'live' strategy, not the experimental mutations
        if only_variant_zero and len(parts) > 1 and not (parts[1] == "0" or str(parts[1]).startswith("L")):
            continue
            
        total += 1
        if "\twin\t" in l:
            wins += 1
            
        if len(parts) > 7:
            try:
                pnl += float(parts[7])
            except ValueError:
                pass
                
    rate = wins / total if total else 0.0
    return wins, total, rate, pnl


def compute_risk_metrics(only_variant_zero: bool = True) -> dict:
    """Compute risk/viability metrics from results.tsv.
    
    If only_variant_zero is True, only analyzes the control strategy.
    Otherwise analyzes all variants combined.
    
    Returns dict with: max_drawdown, longest_loss_streak, sharpe,
    profit_factor, avg_win, avg_loss, required_bankroll.
    """
    empty = {
        "max_drawdown": 0.0, "longest_loss_streak": 0, "sharpe": 0.0,
        "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "required_bankroll": 0.0,
    }
    if not RESULTS_FILE.exists():
        return empty
    
    lines = RESULTS_FILE.read_text().strip().split("\n")[1:]
    profits = []
    for l in lines:
        if not l.strip():
            continue
        parts = l.split("\t")
        if only_variant_zero and len(parts) > 1 and parts[1] != "0":
            continue
        if len(parts) > 7:
            try:
                profits.append(float(parts[7]))
            except ValueError:
                pass
    
    if len(profits) < 5:
        return empty
    
    n = len(profits)
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]
    
    # Max drawdown
    peak = -math.inf
    max_dd = 0.0
    cum = 0.0
    for p in profits:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    
    # Longest losing streak
    longest_streak = 0
    current_streak = 0
    for p in profits:
        if p < 0:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0
    
    # Sharpe ratio (per-trade)
    mean_p = sum(profits) / n
    variance = sum((p - mean_p) ** 2 for p in profits) / n
    std_dev = math.sqrt(variance) if variance > 0 else 0.0
    sharpe = mean_p / std_dev if std_dev > 0 else 0.0
    
    # Profit factor
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    
    return {
        "max_drawdown": max_dd,
        "longest_loss_streak": longest_streak,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "required_bankroll": max_dd * 1.5,
    }


# ---------------------------------------------------------------------------
# Data fetching (imported from our read-only module)
# ---------------------------------------------------------------------------
def fetch_features() -> list[dict]:
    """Fetch live BTC features from exchanges."""
    # Import here to avoid circular imports at module level
    sys.path.insert(0, str(Path(__file__).parent))
    from data_sources import get_btc_features
    return get_btc_features()


# ---------------------------------------------------------------------------
# Market interaction
# ---------------------------------------------------------------------------
def get_market_state() -> dict:
    """Get current Polymarket BTC 5-min market state."""
    sys.path.insert(0, str(Path(__file__).parent))
    from main import PolymarketBTC5m
    market = PolymarketBTC5m()
    return market.get_prices()


def determine_outcome(market_state: dict) -> str | None:
    """
    Determine whether 'Up' or 'Down' won based on final market prices.
    The outcome with the higher midpoint price is the winner.
    Returns None if we can't determine.
    """
    prices = market_state.get("prices") or market_state.get("outcome", {})
    if not prices:
        return None

    up_data = prices.get("Up", {})
    down_data = prices.get("Down", {})

    up_mid = up_data.get("midpoint", 0.0)
    down_mid = down_data.get("midpoint", 0.0)

    if up_mid <= 0 and down_mid <= 0:
        return None

    # The outcome with midpoint closer to 1.0 is the winner
    if up_mid > down_mid:
        return "Up"
    elif down_mid > up_mid:
        return "Down"
    return None


# ---------------------------------------------------------------------------
# Strategy execution (sandboxed)
# ---------------------------------------------------------------------------
def run_strategy(strategy_code: str, get_market_prices=None) -> tuple[str, float]:
    """
    Execute a strategy's predict() function safely.
    Returns ("Up"|"Down", 1.0), (None, 0.0) for abstain, or ("ERROR", 0.0) on crash.
    """
    try:
        from data_sources import get_btc_features
        import time
        namespace = {"__builtins__": __builtins__}
        exec(strategy_code, namespace)
        predict_fn = namespace.get("predict")
        if predict_fn is None:
            log("  Strategy has no predict() function!")
            return "ERROR", 0.0
        # Pass market prices callback; fall back for old strategies
        _market_cb = get_market_prices or (lambda: {})
        try:
            outcome = predict_fn(get_btc_features, time.sleep, _market_cb)
        except TypeError:
            # Backward compat: old strategy without market prices param
            outcome = predict_fn(get_btc_features, time.sleep)
        if outcome not in ("Up", "Down", None):
            log(f"  Strategy returned invalid outcome: {outcome}")
            return "ERROR", 0.0
        return outcome, 1.0 if outcome else 0.0
    except Exception as e:
        log(f"  Strategy execution error: {e}")
        return "ERROR", 0.0


# ---------------------------------------------------------------------------
# LLM Agent — Gemma-4-26b via google-genai
# ---------------------------------------------------------------------------
def create_client() -> genai.Client:
    """Create a Google GenAI client."""
    return genai.Client(api_key=API_KEY)


def generate_strategy_variants(
    client: genai.Client,
    current_strategy: str,
    results_history: str,
    features_sample: list[dict],
    market_prices_sample: dict,
    n_variants: int = VARIANTS_PER_GENERATION,
    temperature: float = 0.9,
    stagnation_notice: str = "",
) -> list[tuple[str, str]]:
    """
    Ask the LLM to generate N strategy variants.
    Returns list of (code, description) tuples.
    """
    program = PROGRAM_FILE.read_text() if PROGRAM_FILE.exists() else ""

    prompt = textwrap.dedent(f"""\
    {program}

    ---

    ## Current Best Strategy (strategy.py):
    ```python
    {current_strategy}
    ```

    ## Results History:
    ```
    {results_history}
    ```

    ## Current Market Features (sample):
    ```json
    {json.dumps(features_sample, indent=2, default=str)}
    ```

    ## Current Polymarket Odds (sample):
    ```json
    {json.dumps(market_prices_sample, indent=2, default=str)}
    ```

    {stagnation_notice}

    ---

    Generate {n_variants} different mutations of the current strategy.
    Each should try a meaningfully different approach or parameter tweak.
    Learn from the results history — if something consistently loses, avoid it.
    If something wins, try small variations of it.

    For EACH variant, output:
    1. A brief one-line description of what's different
    2. The complete strategy.py code in a ```python code block

    Number them like: **Variant 1:**, **Variant 2:**, etc.
    """)

    max_retries = 10
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,  # adaptive temp
                    max_output_tokens=8192,
                ),
            )

            if not response or not response.text:
                log("LLM returned empty response")
                return []

            return parse_variants(response.text, n_variants)

        except Exception as e:
            if "500" in str(e) or "ServerError" in type(e).__name__:
                log(f"LLM API 500 error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
            log(f"LLM API error: {e}")
            traceback.print_exc()
            return []
    return []


def parse_variants(text: str, expected: int) -> list[tuple[str, str]]:
    """
    Parse LLM output into (code, description) pairs.
    Looks for ```python ... ``` blocks with variant headers.
    """
    variants = []

    # First try: Split by variant markers
    parts = re.split(r'\*\*Variant\s+\d+[:\s]*\*\*', text)
    if len(parts) > 1:
        for part in parts[1:]:  # skip preamble before first variant
            lines = part.strip().split("\n")
            desc = ""
            for line in lines:
                line = line.strip().strip("*").strip("-").strip()
                if line and not line.startswith("```"):
                    desc = line.replace("\t", " ")[:300]  # Cap at 300 so it doesn't get ridiculously long, remove tabs
                    break

            code_match = re.search(r'```python\s*\n(.*?)```', part, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                if "def predict" in code:
                    variants.append((code, desc or "unnamed variant"))

    # Fallback: If no markers were found, extract text immediately preceding each code block
    if not variants:
        blocks = re.split(r'```python', text)
        for i in range(1, len(blocks)):
            pre_text = blocks[i-1]
            code_part = blocks[i]
            
            # Find the last meaningful line before the code block
            desc = f"variant {len(variants) + 1}"
            lines = pre_text.strip().split("\n")
            for line in reversed(lines):
                line = line.strip().strip("*").strip("-").strip()
                if line and not line.lower().startswith("here is") and not line.lower().startswith("variant"):
                    desc = line.replace("\t", " ")[:300]
                    break
                    
            code_match = re.search(r'^\s*\n(.*?)```', code_part, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                if "def predict" in code:
                    variants.append((code, desc))
            elif "def predict" in code_part:
                # If there's no closing ```
                code = code_part.strip()
                variants.append((code, desc))

    return variants[:expected]


# ---------------------------------------------------------------------------
# Main evolutionary loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main evolutionary loop (Steady-State)
# ---------------------------------------------------------------------------

MAX_POPULATION = 6

class StrategyVariant:
    def __init__(self, code, desc):
        self.code = code
        self.desc = desc
        self.pnl = 0.0
        self.wins = 0
        self.total = 0
        self.returns = [] # Track individual trade returns for Sharpe
        self.consecutive_losses = 0
        self.rounds_underwater = 0
        self.age = 0

    def get_score(self) -> float:
        """Score based on sharpe * sqrt(N) where N is trades done."""
        if not self.returns:
            # Cowardice penalty: slightly negative to allow fresh mutations
            # to displace old strategies that won't bet.
            return -0.01 * self.age
        n = len(self.returns)
        mean_p = sum(self.returns) / n
        variance = sum((p - mean_p) ** 2 for p in self.returns) / n
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        
        # Avoid division by zero; if all trades are identical and positive, 
        # it's a very good sign, but technically Sharpe is infinite.
        # We'll use a small epsilon or just return a high value.
        if std_dev == 0:
            # If they are all wins, score is high. If all losses, score is low/negative.
            return mean_p * math.sqrt(n)
        
        sharpe = mean_p / std_dev
        return sharpe * math.sqrt(n)

def run_steady_state_round(population: dict[int, StrategyVariant], generation: int, client: genai.Client):
    log(f"\n═══ Generation {generation} | Active Strategies: {len(population)} ═══")
    
    # 1. Replenish population
    if len(population) < MAX_POPULATION:
        needed = MAX_POPULATION - len(population)
        log(f"Population below max. Generating {needed} new mutations...")
        
        # Parent is the current best (by score)
        if population:
            parent_id = max(population.keys(), key=lambda k: population[k].get_score())
            parent_code = population[parent_id].code
        else:
            parent_code = STRATEGY_FILE.read_text() if STRATEGY_FILE.exists() else ""
            
        results_history = read_results_history()
        
        # Context
        try:
            from main import PolymarketBTC5m
            temp_mkt = PolymarketBTC5m()
            state = temp_mkt.get_prices()
            prices = state.get("prices", {})
            market_prices_sample = {side: prices.get(side, {}) for side in ("Up", "Down")}
            market_prices_sample["seconds_remaining"] = max(0, temp_mkt.market_end_time - time.time())
            features = fetch_features()
        except Exception:
            market_prices_sample = {}
            features = []

        # Adaptive Temperature Logic
        # Increase LLM temperature to encourage exploration when the control strategy plateaus,
        # reset it (lower) upon successful innovation to refine the edge.
        dynamic_temp = 0.9
        if population:
            parent = population[parent_id]
            if parent.pnl > 0 and parent.consecutive_losses == 0:
                dynamic_temp = 0.5  # refine successful strategy
            elif parent.pnl < -1.0 or parent.consecutive_losses > 0:
                dynamic_temp = 0.9  # explore wildly if bleeding or plateauing

        # Stagnation check: if the last 10 results are all 'abstain', sound the alarm
        stagnation_notice = ""
        recent_results = read_results_history(10)
        if "abstain" in recent_results and "win" not in recent_results and "loss" not in recent_results:
            stagnation_notice = textwrap.dedent("""
                ### CRITICAL: STAGNATION DETECTED
                Your recent strategies are all returning `None` (ABSTAIN). 
                While this avoids losses, it earns $0 and fails the task.
                **YOU MUST LOOSEN YOUR THRESHOLDS.** Try a strategy that is much more aggressive.
                Accept a lower confidence level to get some trades on the board.
            """)
            dynamic_temp = 1.0 # Max exploration

        variants = generate_strategy_variants(
            client, parent_code, results_history, features, market_prices_sample, 
            n_variants=needed, temperature=dynamic_temp, stagnation_notice=stagnation_notice
        )
        
        if variants:
            next_id = max(population.keys()) + 1 if population else 1
            for code, desc in variants:
                population[next_id] = StrategyVariant(code, desc)
                log(f"  + Added Variant {next_id}: {desc[:60]}")
                next_id += 1
                if len(population) >= MAX_POPULATION:
                    break

    # 2. Market Execution
    log("\n--- Executing Market Round ---")
    try:
        features = fetch_features()
    except Exception as e:
        log(f"Feature fetch failed: {e}")
        features = []

    try:
        market_state = get_market_state()
        market_ts = int(market_state.get("slug", "-0").split("-")[-1])
        log(f"Market: {market_state.get('slug', 'unknown')}, ends={market_state.get('end_time', '?')}")
    except Exception as e:
        log(f"Market fetch failed: {e}")
        return

    import concurrent.futures
    predictions = []

    def worker(p_idx):
        var = population[p_idx]
        from main import PolymarketBTC5m
        worker_mkt = PolymarketBTC5m()

        def _get_market_prices():
            try:
                state = worker_mkt.get_prices()
                prices = state.get("prices", {})
                res = {side: prices.get(side, {}) for side in ("Up", "Down")}
                res["seconds_remaining"] = max(0, worker_mkt.market_end_time - time.time())
                return res
            except Exception:
                return {}

        pred = run_strategy(var.code, get_market_prices=_get_market_prices)
        outcome_pred = pred[0] if pred else None
        bet_amount = pred[1] if pred else 0.0

        shares = 0.0; invested = 0.0; avg_fill_price = 0.0; book_depth = 0.0; spread = 0.0; thin_book = False
        
        if outcome_pred and outcome_pred != "ERROR":
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from paper_trade import position_size_detailed
                live_mkt = PolymarketBTC5m()
                fill = position_size_detailed(bet_amount, outcome=outcome_pred, market=live_mkt)
                shares = fill["shares"]; invested = bet_amount - fill["unfilled"]
                avg_fill_price = fill["avg_fill_price"]; book_depth = fill["book_depth_usd"]; spread = fill["spread"]
                if book_depth < MIN_BOOK_DEPTH_USD: thin_book = True
            except Exception as e:
                log(f"  Variant {p_idx}: immediate orderbook simulation failed: {e}")

        return {
            "idx": p_idx, "prediction": outcome_pred, "bet_amount": bet_amount,
            "shares": shares, "invested": invested, "avg_fill_price": avg_fill_price,
            "book_depth": book_depth, "spread": spread, "thin_book": thin_book
        }

    log("Running strategies concurrently...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(population)) as executor:
        futures = [executor.submit(worker, idx) for idx in population.keys()]
        for future in concurrent.futures.as_completed(futures):
            p = future.result()
            predictions.append(p)
            thin_tag = " [THIN]" if p.get("thin_book") else ""
            log(f"  Variant {p['idx']}: predicts {p['prediction']} bet=${p['bet_amount']:.2f} "
                f"fill@${p['avg_fill_price']:.4f}{thin_tag}")

    # Wait for resolution
    if market_state.get("status") == "live" and market_ts > 0:
        import time
        from main import PolymarketBTC5m
        temp_market = PolymarketBTC5m(initial_ts=market_ts)
        secs_remaining = temp_market.market_end_time - time.time()
        if secs_remaining > 0:
            log(f"Waiting {secs_remaining:.0f}s for market to expire...")
            while not temp_market.is_expired:
                rem = temp_market.market_end_time - time.time()
                if rem > 0:
                    wait = min(10, rem + 1)
                    time.sleep(wait)
                    if rem > 10: log(f"  {rem:.0f}s remaining...")
        time.sleep(3)

    log("Resolving market outcome...")
    resolution_mids = {}
    try:
        from main import PolymarketBTC5m
        resolved_market = PolymarketBTC5m(initial_ts=market_ts) if market_ts > 0 else PolymarketBTC5m()
        resolved_state = resolved_market.get_prices()
        actual_outcome = determine_outcome(resolved_state)
        res_prices = resolved_state.get("outcome") or resolved_state.get("prices", {})
        for side in ("Up", "Down"):
            resolution_mids[side] = res_prices.get(side, {}).get("midpoint", 0.0)
    except Exception as e:
        log(f"Resolution error: {e}")
        actual_outcome = None

    if actual_outcome:
        log(f"Market resolved: {actual_outcome} won!")
    else:
        actual_outcome = determine_outcome(market_state)
        if not actual_outcome:
            log("Cannot determine outcome. Skipping scoring.")
            return

    # Score
    parent_hash = git_current_hash()
    leader_idx_start = max(population.keys(), key=lambda k: population[k].get_score()) if population else None

    for p in predictions:
        idx = p["idx"]
        var = population[idx]
        var.age += 1
        
        outcome_pred = p["prediction"]
        res_mid = resolution_mids.get(outcome_pred, 0.0) if outcome_pred else 0.0
        status = "abstain"; profit = 0.0; roi = 0.0
        
        if outcome_pred == "ERROR":
            var.consecutive_losses = 999  # Force an immediate kill
            var.rounds_underwater = 999
            status = "crash"
        elif outcome_pred:
            var.total += 1
            correct = (outcome_pred == actual_outcome)
            status = "win" if correct else "loss"
            
            if correct: 
                var.wins += 1
            
            if p["invested"] > 0:
                revenue = p["shares"] * 1.00 if correct else 0.0
                profit = revenue - p["invested"]
                roi = profit / p["invested"]
            
            if p["thin_book"]:
                # Only ignore thin book *profits* (prevent fake wins).
                # Keep thin book *losses* so bad strategies are correctly penalized.
                if profit > 0:
                    profit = 0.0 
                
            var.pnl += profit
            var.returns.append(profit)
            
            # Strike Rule: Only penalized if it places a bet
            if profit < 0:
                var.consecutive_losses += 1
            else:
                var.consecutive_losses = 0
                
        if var.pnl <= 0:
            var.rounds_underwater += 1
        else:
            var.rounds_underwater = 0
            
        var_label = f"L{idx}" if idx == leader_idx_start else idx
        pred_label = "ABSTAIN" if outcome_pred is None else (outcome_pred if outcome_pred != "ERROR" else "CRASH")
        append_result(
            gen=generation, var=var_label, commit=parent_hash, prediction=pred_label,
            bet_amount=p["bet_amount"], outcome=actual_outcome, status=status, profit=profit, roi=roi,
            desc=var.desc, avg_fill=p["avg_fill_price"], book_depth=p["book_depth"],
            spread=p["spread"], resolution_mid=res_mid
        )
        
        emoji = "✅" if status == "win" else "❌"
        if not outcome_pred: emoji = "⏸️"
        log(f"  {emoji} Var {idx}: {status} | PnL: ${profit:.2f} | Total PnL: ${var.pnl:.2f} | Age: {var.age}")

    # 3. Culling
    log("\n--- Culling Phase ---")
    to_kill = []
    for idx, var in population.items():
        if var.rounds_underwater >= 5:
            log(f"  ☠️ Killed Var {idx} (Non-positive PnL for 5 consecutive rounds). Final PnL: ${var.pnl:.2f}")
            to_kill.append(idx)
        elif var.pnl <= -3.00:
            log(f"  ☠️ Killed Var {idx} (PnL Stop Loss hit). Final PnL: ${var.pnl:.2f}")
            to_kill.append(idx)
            
    # Always keep at least 1 alive, so if all die, keep the best one
    if len(to_kill) == len(population) and population:
        best_of_losers = max(population.keys(), key=lambda k: population[k].get_score())
        to_kill.remove(best_of_losers)
        log(f"  🛡️ Spared Var {best_of_losers} to prevent extinction.")
        
    for idx in to_kill:
        del population[idx]
        
    # 4. Promotion
    log("\n--- Leaderboard ---")
    sorted_pop = sorted(population.items(), key=lambda x: x[1].get_score(), reverse=True)
    for i, (idx, var) in enumerate(sorted_pop):
        win_rate = var.wins / var.total if var.total > 0 else 0
        log(f"  #{i+1}: Var {idx} | Score: {var.get_score():.2f} | PnL: ${var.pnl:.2f} | Wins: {var.wins}/{var.total} ({win_rate:.1%}) | Age: {var.age}")
        
    leader_idx, leader_var = sorted_pop[0]
    log(f"\n👑 Current Leader: Var {leader_idx}")
    STRATEGY_FILE.write_text(leader_var.code)
    git_commit(f"Leader Var {leader_idx}: Score {leader_var.get_score():.2f} | PnL ${leader_var.pnl:.2f} | Age {leader_var.age}")
    
    t_wins, t_rounds, t_rate, t_pnl = compute_stats(only_variant_zero=False)
    c_wins, c_rounds, c_rate, c_pnl = compute_stats(only_variant_zero=True)
    l_rate = leader_var.wins / leader_var.total if leader_var.total > 0 else 0.0

    log(f"\n📈 OVERALL PERFORMANCE TRACKER:")
    log(f"  Historical (All Variants):   {t_wins}/{t_rounds} wins ({t_rate:.1%}) | Total PnL: ${t_pnl:.2f}")
    log(f"  Historical (CEOs Only):      {c_wins}/{c_rounds} wins ({c_rate:.1%}) | Total PnL: ${c_pnl:.2f}")
    log(f"  Current CEO (Var {leader_idx} only): {leader_var.wins}/{leader_var.total} wins ({l_rate:.1%}) | Total PnL: ${leader_var.pnl:.2f}")


def main():
    log("╔══════════════════════════════════════════════════╗")
    log("║  Polymarket BTC 5-min Steady-State Pipeline      ║")
    log(f"║  Model: {MODEL} via google-genai      ║")
    log("╚══════════════════════════════════════════════════╝")

    if not API_KEY:
        log("ERROR: No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY env var.")
        sys.exit(1)

    client = create_client()
    
    if not (AGENT_REPO / ".git").exists():
        subprocess.run(["git", "init"], cwd=str(AGENT_REPO), capture_output=True)
        git_commit("Initial strategy")

    git_create_branch(f"autoresearch/{datetime.now().strftime('%b%d').lower()}")
    init_results()
    
    if not STRATEGY_FILE.exists() or "get_features" not in STRATEGY_FILE.read_text():
        STRATEGY_FILE.write_text(textwrap.dedent('''\
            def predict(get_features, sleep, get_market_prices):
                features = get_features()
                up = sum(1 for f in features if f.get("OBI", 0) > 0)
                down = sum(1 for f in features if f.get("OBI", 0) < 0)
                return "Up" if up >= down else "Down"
        '''))
        git_commit("Initial strategy")

    population = {
        0: StrategyVariant(STRATEGY_FILE.read_text(), "Initial Control Strategy")
    }

    generation = 1
    while True:
        try:
            run_steady_state_round(population, generation, client)
            generation += 1
        except KeyboardInterrupt:
            log("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            log(f"Generation {generation} failed: {e}")
            traceback.print_exc()
            time.sleep(30)

if __name__ == "__main__":
    main()

