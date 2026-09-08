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
import math


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
# Combined Complexity Score
# ---------------------------------------------------------------------------

# Weights for combining metrics into a single score
WEIGHT_AST_NODES = 0.4
WEIGHT_CYCLOMATIC = 0.6


def combined_complexity_score(code: str) -> dict:
    """
    Compute a combined complexity score from AST node count and
    cyclomatic complexity.

    Returns dict with all sub-metrics and the final combined score.
    The score is an absolute number (not normalized) — lower is simpler.
    """
    import re

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

    # Combined score: weighted sum
    # AST nodes capture structural size; cyclomatic captures branching depth
    if ast_nodes < 0 or cc["total"] < 0:
        combined = float("inf")
    else:
        combined = (WEIGHT_AST_NODES * ast_nodes) + (WEIGHT_CYCLOMATIC * cc["total"])

    return {
        # Traditional (backward compat)
        "lines": len(lines),
        "classes": classes,
        "methods": methods + functions,
        "total_lines": len(code.split("\n")),
        # New metrics
        "ast_nodes": ast_nodes,
        "ast_breakdown": ast_breakdown,
        "cyclomatic_total": cc["total"],
        "cyclomatic_max_function": cc.get("max_function", 0),
        "cyclomatic_mean_function": cc.get("mean_function", 0.0),
        "cyclomatic_functions": cc.get("functions", {}),
        # Combined
        "combined_score": round(combined, 2),
    }


def format_complexity_report(complexity: dict) -> str:
    """Format complexity metrics as a human-readable report."""
    lines = ["--- Code Complexity ---"]
    lines.append(f"  Lines of code:          {complexity['lines']}")
    lines.append(f"  Classes:                {complexity['classes']}")
    lines.append(f"  Methods/Functions:      {complexity['methods']}")
    lines.append(f"  AST Nodes:              {complexity['ast_nodes']}")
    lines.append(f"  Cyclomatic (total):     {complexity['cyclomatic_total']}")
    lines.append(f"  Cyclomatic (max func):  {complexity['cyclomatic_max_function']}")
    lines.append(f"  Cyclomatic (mean func): {complexity['cyclomatic_mean_function']:.1f}")
    lines.append(f"  ╔═══════════════════════════════════╗")
    lines.append(f"  ║  Combined Score: {complexity['combined_score']:>8.1f}          ║")
    lines.append(f"  ╚═══════════════════════════════════╝")
    return "\n".join(lines)


# Alias for backward compatibility and convenience
count_complexity = combined_complexity_score
