"""
Code Complexity Scoring — AST Node Complexity + Cyclomatic Complexity

Replaces the simplistic line-count metric with a principled combined score:
  1. AST Node Complexity: total number of AST nodes in the parse tree
  2. Cyclomatic Complexity: M = E - N + 2P (decision points + 1 per function)

The combined score is a weighted sum normalized against a baseline,
giving a single "complexity score" that captures both structural depth
and control-flow branching.
"""

import ast
import gzip
import inspect
import math
import os
import re
import sys
from typing import Any, Callable

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# AST Node Complexity
# ---------------------------------------------------------------------------

def count_ast_nodes(code: str) -> int:
    """Count total AST nodes in a Python source string."""
    try:
        tree = ast.parse(code)
        return sum(1 for _ in ast.walk(tree))
    except SyntaxError:
        return -1


def ast_node_breakdown(code: str) -> dict:
    """
    Detailed breakdown of AST node types.
    Returns counts per category for reporting.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"error": "SyntaxError"}

    categories = {
        "statements": 0,
        "expressions": 0,
        "control_flow": 0,
        "definitions": 0,
        "operators": 0,
        "literals": 0,
        "other": 0,
    }

    control_flow_types = (
        ast.If, ast.For, ast.While, ast.Try, ast.With,
        ast.Break, ast.Continue, ast.Return, ast.Raise,
    )
    # Handle ExceptHandler separately since it's not in all versions
    try:
        control_flow_types = control_flow_types + (ast.ExceptHandler,)
    except AttributeError:
        pass

    definition_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    operator_types = (ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare)
    literal_types = (ast.Constant, ast.JoinedStr, ast.FormattedValue)

    for node in ast.walk(tree):
        if isinstance(node, control_flow_types):
            categories["control_flow"] += 1
        elif isinstance(node, definition_types):
            categories["definitions"] += 1
        elif isinstance(node, operator_types):
            categories["operators"] += 1
        elif isinstance(node, literal_types):
            categories["literals"] += 1
        elif isinstance(node, ast.expr):
            categories["expressions"] += 1
        elif isinstance(node, ast.stmt):
            categories["statements"] += 1
        else:
            categories["other"] += 1

    return categories


# ---------------------------------------------------------------------------
# Cyclomatic Complexity
# ---------------------------------------------------------------------------

class CyclomaticVisitor(ast.NodeVisitor):
    """
    Compute cyclomatic complexity for each function/method and the module.

    Cyclomatic complexity = number of decision points + 1 (per function).
    Decision points: if, elif, for, while, except, with, and, or,
                     assert, comprehension clauses.
    """

    def __init__(self):
        self.functions = {}  # name -> complexity
        self._current_func = None
        self._module_complexity = 1  # base complexity for module-level code

    def visit_FunctionDef(self, node):
        old = self._current_func
        self._current_func = node.name
        self.functions[node.name] = 1  # base
        self.generic_visit(node)
        self._current_func = old

    visit_AsyncFunctionDef = visit_FunctionDef

    def _increment(self, count=1):
        if self._current_func:
            self.functions[self._current_func] = (
                self.functions.get(self._current_func, 1) + count
            )
        else:
            self._module_complexity += count

    def visit_If(self, node):
        self._increment()
        self.generic_visit(node)

    def visit_For(self, node):
        self._increment()
        self.generic_visit(node)

    def visit_While(self, node):
        self._increment()
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self._increment()
        self.generic_visit(node)

    def visit_With(self, node):
        self._increment()
        self.generic_visit(node)

    def visit_Assert(self, node):
        self._increment()
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        # Each 'and'/'or' adds a decision path
        self._increment(len(node.values) - 1)
        self.generic_visit(node)

    def visit_IfExp(self, node):
        # Ternary expression: x if cond else y
        self._increment()
        self.generic_visit(node)

    def visit_comprehension(self, node):
        # Each comprehension clause (for + ifs)
        self._increment(1 + len(node.ifs))
        self.generic_visit(node)


def cyclomatic_complexity(code: str) -> dict:
    """
    Compute cyclomatic complexity for a Python source string.

    Returns:
        {
            "total": int,           # sum of all function complexities + module
            "module_level": int,    # module-level complexity
            "functions": {name: int},  # per-function complexity
            "max_function": int,    # highest single-function complexity
            "mean_function": float, # average function complexity
        }
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"total": -1, "error": "SyntaxError"}

    visitor = CyclomaticVisitor()
    visitor.visit(tree)

    func_values = list(visitor.functions.values())
    total = visitor._module_complexity + sum(func_values)

    return {
        "total": total,
        "module_level": visitor._module_complexity,
        "functions": visitor.functions,
        "max_function": max(func_values) if func_values else 0,
        "mean_function": (sum(func_values) / len(func_values)) if func_values else 0.0,
    }



# ---------------------------------------------------------------------------
# Gzip Compression Complexity Metrics
# ---------------------------------------------------------------------------

def gzip_compress_code(code: str, level: int = 9) -> bytes:
    """Compress UTF-8 source string using gzip."""
    return gzip.compress(code.encode("utf-8"), compresslevel=level)


def gzip_compressed_size(code: str, level: int = 9) -> int:
    """Return total bytes of gzipped UTF-8 source code (0 if code is empty)."""
    if not code:
        return 0
    return len(gzip_compress_code(code, level=level))


def gzip_compression_ratio(code: str, level: int = 9) -> float:
    """
    Compute gzip compression ratio: compressed_bytes / uncompressed_bytes.
    Returns 0.0 if code is empty.
    Lower values indicate higher compressibility / redundancy.
    """
    data = code.encode("utf-8")
    if not data:
        return 0.0
    compressed = gzip.compress(data, compresslevel=level)
    return len(compressed) / len(data)


def gzip_complexity(code: str, level: int = 9) -> dict[str, Any]:
    """
    Compute detailed gzip compression metrics for a Python source string.

    Returns:
        {
            "compressed_bytes": int,     # total size in bytes after gzip
            "uncompressed_bytes": int,   # total size in bytes of raw UTF-8 string
            "compression_ratio": float,  # compressed_bytes / uncompressed_bytes
            "space_saving": float,       # 1.0 - compression_ratio
            "compression_factor": float, # uncompressed_bytes / compressed_bytes
        }
    """
    data = code.encode("utf-8")
    raw_len = len(data)
    if raw_len == 0:
        return {
            "compressed_bytes": 0,
            "uncompressed_bytes": 0,
            "compression_ratio": 0.0,
            "space_saving": 0.0,
            "compression_factor": 0.0,
        }
    compressed = gzip.compress(data, compresslevel=level)
    comp_len = len(compressed)
    ratio = comp_len / raw_len
    return {
        "compressed_bytes": comp_len,
        "uncompressed_bytes": raw_len,
        "compression_ratio": ratio,
        "space_saving": 1.0 - ratio,
        "compression_factor": (raw_len / comp_len) if comp_len > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Pluggable Optimization & Complexity Scoring Functions
# ---------------------------------------------------------------------------

# Tunable weights for combining metrics into the default score
WEIGHT_AST_NODES: float = 0.4
WEIGHT_CYCLOMATIC: float = 0.6


def default_scoring_function(metrics: dict[str, Any]) -> float:
    """
    Default optimization objective:
    Weighted sum of AST nodes (structural size) and cyclomatic complexity (branching depth).
    Score = (WEIGHT_AST_NODES * ast_nodes) + (WEIGHT_CYCLOMATIC * cyclo).
    Lower is simpler.
    """
    ast_nodes = metrics.get("ast_nodes", -1)
    cyclo = metrics.get("cyclomatic_total", -1)
    if ast_nodes < 0 or cyclo < 0:
        return float("inf")
    return (WEIGHT_AST_NODES * ast_nodes) + (WEIGHT_CYCLOMATIC * cyclo)


def gzip_ratio_scoring_function(target: Any, metrics: dict[str, Any] | None = None) -> float:
    """
    Optimization objective minimizing gzip compression ratio (compressed / uncompressed).
    Expressed as percentage (0-100), e.g. 26.8 for 0.268 ratio.
    Lower indicates simpler / more compressible / more repetitive structure.
    Accepts (metrics: dict), (code: str), or (code: str, metrics: dict).
    """
    if isinstance(target, dict):
        if "gzip_compression_ratio" in target:
            return float(target["gzip_compression_ratio"]) * 100.0
        return 100.0
    elif isinstance(target, str):
        return gzip_compression_ratio(target) * 100.0
    elif metrics and "gzip_compression_ratio" in metrics:
        return float(metrics["gzip_compression_ratio"]) * 100.0
    return 100.0


def gzip_size_scoring_function(target: Any, metrics: dict[str, Any] | None = None) -> float:
    """
    Optimization objective minimizing raw gzipped byte size (Kolmogorov complexity proxy).
    Score = compressed byte count. Lower is simpler.
    Accepts (metrics: dict), (code: str), or (code: str, metrics: dict).
    """
    if isinstance(target, dict):
        if "gzip_compressed_bytes" in target:
            return float(target["gzip_compressed_bytes"])
        return float("inf")
    elif isinstance(target, str):
        return float(gzip_compressed_size(target))
    elif metrics and "gzip_compressed_bytes" in metrics:
        return float(metrics["gzip_compressed_bytes"])
    return float("inf")


def gzip_raw_ratio_scoring_function(target: Any, metrics: dict[str, Any] | None = None) -> float:
    """
    Optimization objective minimizing gzip compression ratio on raw 0.0-1.0 scale.
    """
    if isinstance(target, dict):
        return float(target.get("gzip_compression_ratio", 1.0))
    elif isinstance(target, str):
        return gzip_compression_ratio(target)
    elif metrics and "gzip_compression_ratio" in metrics:
        return float(metrics["gzip_compression_ratio"])
    return 1.0



# Active scoring function (optimization objective callable)
# Exposed as global variables so they can be inspected or switched in a notebook
ACTIVE_SCORING_FUNCTION: Callable = default_scoring_function
OPTIMIZATION_FUNCTION: Callable = default_scoring_function  # intuitive alias
active_scoring_function: Callable = default_scoring_function
optimization_function: Callable = default_scoring_function
SCORER_NAME: str = "weighted_ast_cyclomatic"
scorer_name: str = "weighted_ast_cyclomatic"  # lowercase alias


def _invoke_scorer(scorer: Callable, code: str, metrics: dict[str, Any]) -> float:
    """
    Flexibly invoke a complexity scoring function.
    Supports callables taking:
      - (metrics: dict) -> float
      - (code: str) -> float
      - (code: str, metrics: dict) -> float
    """
    try:
        sig = inspect.signature(scorer)
        params = list(sig.parameters.values())
        pos_params = [
            p for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]

        if len(pos_params) >= 2:
            return float(scorer(code, metrics))
        elif len(pos_params) == 1:
            p_name = pos_params[0].name.lower()
            if p_name in ("code", "source", "text", "src", "strategy_code"):
                return float(scorer(code))
            else:
                return float(scorer(metrics))
        else:
            # Varargs or parameterless: try metrics first, then code
            try:
                return float(scorer(metrics))
            except TypeError:
                return float(scorer(code))
    except Exception as e:
        # Fallback invocation attempts
        try:
            return float(scorer(metrics))
        except Exception:
            try:
                return float(scorer(code))
            except Exception:
                raise e


def set_scoring_function(scorer: Callable, name: str | None = None) -> Callable:
    """
    Switch the optimization function / complexity metric being minimized.

    The custom scorer can accept:
      - `metrics: dict` (access to 'ast_nodes', 'cyclomatic_total', 'lines', etc.)
      - `code: str` (the raw Python code, e.g. for custom metrics like Card & Agresti)
      - `code: str, metrics: dict`

    Example in a Jupyter notebook:
        import complexity

        # Example 1: Use a custom metric function (e.g. Card & Agresti complexity)
        def my_card_agresti(code: str, metrics: dict) -> float:
            # Custom calculation based on data flow, fan-in/fan-out, etc.
            return custom_score

        complexity.set_scoring_function(my_card_agresti, name="card_agresti")

        # Example 2: Simple lambda
        complexity.set_scoring_function(lambda m: m["ast_nodes"], name="pure_ast")
    """
    global ACTIVE_SCORING_FUNCTION, OPTIMIZATION_FUNCTION, SCORER_NAME
    global active_scoring_function, optimization_function, scorer_name
    ACTIVE_SCORING_FUNCTION = scorer
    OPTIMIZATION_FUNCTION = scorer
    active_scoring_function = scorer
    optimization_function = scorer
    SCORER_NAME = name or getattr(scorer, "__name__", "custom_scorer")
    scorer_name = SCORER_NAME
    return scorer


def set_weights(ast_nodes: float | None = None, cyclomatic: float | None = None) -> tuple[float, float]:
    """
    Conveniently adjust the default weights without replacing the scoring function.

    Example:
        import complexity
        complexity.set_weights(ast_nodes=0.2, cyclomatic=0.8)
    """
    global WEIGHT_AST_NODES, WEIGHT_CYCLOMATIC
    if ast_nodes is not None:
        WEIGHT_AST_NODES = float(ast_nodes)
    if cyclomatic is not None:
        WEIGHT_CYCLOMATIC = float(cyclomatic)
    return WEIGHT_AST_NODES, WEIGHT_CYCLOMATIC


def reset_default_scoring() -> None:
    """Reset the optimization function and weights back to default AST + Cyclomatic."""
    global WEIGHT_AST_NODES, WEIGHT_CYCLOMATIC, ACTIVE_SCORING_FUNCTION, OPTIMIZATION_FUNCTION, SCORER_NAME
    global active_scoring_function, optimization_function, scorer_name
    WEIGHT_AST_NODES = 0.4
    WEIGHT_CYCLOMATIC = 0.6
    ACTIVE_SCORING_FUNCTION = default_scoring_function
    OPTIMIZATION_FUNCTION = default_scoring_function
    active_scoring_function = default_scoring_function
    optimization_function = default_scoring_function
    SCORER_NAME = "weighted_ast_cyclomatic"
    scorer_name = "weighted_ast_cyclomatic"


def get_active_scoring_function() -> tuple[Callable, str]:
    """Return the currently active scoring function and its name."""
    return ACTIVE_SCORING_FUNCTION, SCORER_NAME


AVAILABLE_SCORERS: dict[str, Callable] = {
    "weighted_ast_cyclomatic": default_scoring_function,
    "default": default_scoring_function,
    "ast_cyclomatic": default_scoring_function,
    "gzip_ratio": gzip_ratio_scoring_function,
    "gzip_compression_ratio": gzip_ratio_scoring_function,
    "gzip_raw_ratio": gzip_raw_ratio_scoring_function,
    "gzip_size": gzip_size_scoring_function,
    "gzip_bytes": gzip_size_scoring_function,
    "gzip_compressed_bytes": gzip_size_scoring_function,
}


def select_scoring_function(scorer: Callable | str, name: str | None = None) -> Callable:
    """
    Select an active scoring function by name or callable.
    Named presets include:
      - 'default' / 'weighted_ast_cyclomatic' (0.4*AST + 0.6*Cyclomatic)
      - 'gzip_ratio' / 'gzip_compression_ratio' (percentage scale 0-100%, lower is simpler)
      - 'gzip_size' / 'gzip_bytes' (compressed byte count, lower is simpler)
      - 'gzip_raw_ratio' (raw 0.0-1.0 scale)
    """
    if isinstance(scorer, str):
        key = scorer.strip().lower()
        if key in AVAILABLE_SCORERS:
            resolved_name = name or key
            return set_scoring_function(AVAILABLE_SCORERS[key], name=resolved_name)
        raise ValueError(
            f"Unknown scorer preset '{scorer}'. Available presets: {list(AVAILABLE_SCORERS.keys())}"
        )
    return set_scoring_function(scorer, name=name)


def use_gzip_ratio_scoring() -> Callable:
    """Switch active optimization objective to gzip compression ratio (percentage)."""
    return set_scoring_function(gzip_ratio_scoring_function, name="gzip_compression_ratio")


def use_gzip_size_scoring() -> Callable:
    """Switch active optimization objective to gzip compressed byte size."""
    return set_scoring_function(gzip_size_scoring_function, name="gzip_compressed_bytes")


def use_default_scoring() -> Callable:
    """Reset to default weighted AST + Cyclomatic complexity scoring."""
    reset_default_scoring()
    return default_scoring_function


# Auto-configure initial scorer from environment variable if present
_ENV_SCORER = os.environ.get("COMPLEXITY_SCORER") or os.environ.get("OPTIMIZATION_OBJECTIVE")
if _ENV_SCORER and _ENV_SCORER.strip().lower() in AVAILABLE_SCORERS:
    select_scoring_function(_ENV_SCORER.strip().lower())


def combined_complexity_score(code: str, scoring_fn: Callable | None = None) -> dict[str, Any]:
    """
    Compute complexity metrics and evaluate the active optimization objective.

    Optionally pass `scoring_fn` to override the scoring function for a single evaluation.

    Returns dict with:
      - "combined_score": float (the objective value being minimized)
      - "optimization_score": float (alias of combined_score)
      - "scorer_name": str (name of the active objective function)
      - traditional metrics: lines, classes, methods, total_lines
      - AST metrics: ast_nodes, ast_breakdown
      - Cyclomatic metrics: cyclomatic_total, cyclomatic_max_function, cyclomatic_mean_function
      - Gzip metrics: gzip_compressed_bytes, gzip_uncompressed_bytes, gzip_compression_ratio, gzip_space_saving
    """
    # Traditional metrics (kept for backward compatibility)
    lines = [l for l in code.split("\n") if l.strip() and not l.strip().startswith("#")]
    classes = len(re.findall(r'^class\s+', code, re.MULTILINE))
    methods = len(re.findall(r'^\s+def\s+', code, re.MULTILINE))
    functions = len(re.findall(r'^def\s+', code, re.MULTILINE))

    # AST complexity
    ast_nodes = count_ast_nodes(code)
    ast_breakdown = ast_node_breakdown(code)

    # Cyclomatic complexity
    cc = cyclomatic_complexity(code)

    # Gzip compression complexity
    gz = gzip_complexity(code)

    base_metrics = {
        "lines": len(lines),
        "classes": classes,
        "methods": methods + functions,
        "total_lines": len(code.split("\n")),
        "ast_nodes": ast_nodes,
        "ast_breakdown": ast_breakdown,
        "cyclomatic_total": cc["total"],
        "cyclomatic_max_function": cc.get("max_function", 0),
        "cyclomatic_mean_function": cc.get("mean_function", 0.0),
        "cyclomatic_functions": cc.get("functions", {}),
        "gzip_compressed_bytes": gz["compressed_bytes"],
        "gzip_uncompressed_bytes": gz["uncompressed_bytes"],
        "gzip_compression_ratio": round(gz["compression_ratio"], 4),
        "gzip_space_saving": round(gz["space_saving"], 4),
        "gzip_compression_factor": round(gz["compression_factor"], 2),
    }

    # Evaluate optimization score using active or provided scoring function
    fn = scoring_fn if scoring_fn is not None else ACTIVE_SCORING_FUNCTION
    name = (
        getattr(scoring_fn, "__name__", "custom_override")
        if scoring_fn is not None
        else SCORER_NAME
    )

    try:
        score = _invoke_scorer(fn, code, base_metrics)
    except Exception:
        score = float("inf")

    base_metrics["combined_score"] = round(score, 2)
    base_metrics["optimization_score"] = round(score, 2)
    base_metrics["scorer_name"] = name
    return base_metrics


def format_complexity_report(complexity: dict) -> str:
    """Format complexity metrics as a human-readable report."""
    scorer_name = complexity.get("scorer_name", "Combined Score")
    lines = ["--- Code Complexity ---"]
    lines.append(f"  Lines of code:          {complexity['lines']}")
    lines.append(f"  Classes:                {complexity['classes']}")
    lines.append(f"  Methods/Functions:      {complexity['methods']}")
    lines.append(f"  AST Nodes:              {complexity['ast_nodes']}")
    lines.append(f"  Cyclomatic (total):     {complexity['cyclomatic_total']}")
    lines.append(f"  Cyclomatic (max func):  {complexity['cyclomatic_max_function']}")
    lines.append(f"  Cyclomatic (mean func): {complexity['cyclomatic_mean_function']:.1f}")
    if "gzip_compressed_bytes" in complexity:
        gz_comp = complexity["gzip_compressed_bytes"]
        gz_raw = complexity.get("gzip_uncompressed_bytes", 0)
        gz_ratio = complexity.get("gzip_compression_ratio", 0.0)
        lines.append(f"  Gzip Compressed Bytes:  {gz_comp} B (raw: {gz_raw} B)")
        lines.append(f"  Gzip Compression Ratio: {gz_ratio:.4f} ({gz_ratio * 100:.1f}%)")
    lines.append(f"  ╔═════════════════════════════════════════════════╗")
    lines.append(f"  ║  Optimization Score ({scorer_name:20s}): {complexity['combined_score']:>8.1f}  ║")
    lines.append(f"  ╚═════════════════════════════════════════════════╝")
    return "\n".join(lines)


# Alias for backward compatibility and convenience
count_complexity = combined_complexity_score


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compute code complexity and compression metrics.")
    parser.add_argument("file", nargs="?", default="strategy.py", help="Python file to analyze (default: strategy.py)")
    parser.add_argument(
        "--scorer", "-s",
        choices=list(AVAILABLE_SCORERS.keys()),
        default=None,
        help="Scorer preset to evaluate (default: weighted_ast_cyclomatic or $COMPLEXITY_SCORER)",
    )
    parser.add_argument("--json", "-j", action="store_true", help="Output metrics as JSON")
    args = parser.parse_args()

    target_path = Path(args.file)
    if not target_path.exists():
        print(f"Error: File not found: {target_path}", file=sys.stderr)
        sys.exit(1)

    code_content = target_path.read_text(encoding="utf-8")
    if args.scorer:
        select_scoring_function(args.scorer)

    results = combined_complexity_score(code_content)
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        print(f"Target: {target_path}")
        print(format_complexity_report(results))
