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
import inspect
import math
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
# Pluggable Optimization & Complexity Scoring Functions
# ---------------------------------------------------------------------------

# Tunable weights for combining metrics into the default score
WEIGHT_AST_NODES: float = 0.4
WEIGHT_CYCLOMATIC: float = 0.6


def default_scoring_function(metrics: dict[str, Any]) -> float:
    """
    Default optimization objective:
    Weighted sum of AST nodes (structural size) and cyclomatic complexity (branching depth).
    Score = (WEIGHT_AST_NODES * ast_nodes) + (WEIGHT_CYCLOMATIC * cyclomatic_total).
    Lower is simpler.
    """
    ast_nodes = metrics.get("ast_nodes", -1)
    cyclo = metrics.get("cyclomatic_total", -1)
    if ast_nodes < 0 or cyclo < 0:
        return float("inf")
    return (WEIGHT_AST_NODES * ast_nodes) + (WEIGHT_CYCLOMATIC * cyclo)


# Active scoring function (optimization objective callable)
# Exposed as global variables so they can be inspected or switched in a notebook
ACTIVE_SCORING_FUNCTION: Callable = default_scoring_function
OPTIMIZATION_FUNCTION: Callable = default_scoring_function  # intuitive alias
SCORER_NAME: str = "weighted_ast_cyclomatic"


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
    ACTIVE_SCORING_FUNCTION = scorer
    OPTIMIZATION_FUNCTION = scorer
    SCORER_NAME = name or getattr(scorer, "__name__", "custom_scorer")
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
    WEIGHT_AST_NODES = 0.4
    WEIGHT_CYCLOMATIC = 0.6
    ACTIVE_SCORING_FUNCTION = default_scoring_function
    OPTIMIZATION_FUNCTION = default_scoring_function
    SCORER_NAME = "weighted_ast_cyclomatic"


def get_active_scoring_function() -> tuple[Callable, str]:
    """Return the currently active scoring function and its name."""
    return ACTIVE_SCORING_FUNCTION, SCORER_NAME


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
    lines.append(f"  ╔═════════════════════════════════════════════════╗")
    lines.append(f"  ║  Optimization Score ({scorer_name:20s}): {complexity['combined_score']:>8.1f}  ║")
    lines.append(f"  ╚═════════════════════════════════════════════════╝")
    return "\n".join(lines)


# Alias for backward compatibility and convenience
count_complexity = combined_complexity_score
