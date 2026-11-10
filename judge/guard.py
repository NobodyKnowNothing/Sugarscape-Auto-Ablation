"""
Stage 0 static guard.

Keeps all model logic inside what the complexity measure can see, and keeps the
model from touching the judge, the filesystem, or global randomness:
  - imports restricted to an allow-list (stdlib math/data helpers, numpy, mesa)
  - no dynamic code (exec/eval/compile/__import__), no introspection escapes
    (banned dunders as attributes, bare names, or getattr/setattr/delattr
    names; getattr/setattr/delattr only with a literal attribute name)
  - no file or OS access (the landscape arrives through `env`)
  - no global RNG (randomness must come from the judge's `rng` or the model's own seeded RNG)
  - no large opaque literals (encoded code/data)
  - the contract functions init/step/observe must exist at module level
"""

from __future__ import annotations

import ast

ALLOWED_IMPORT_ROOTS = {
    "__future__", "math", "numpy", "mesa", "typing", "dataclasses", "functools",
    "itertools", "collections", "heapq", "operator", "statistics", "enum", "bisect",
}
BANNED_CALLS = {
    "exec", "eval", "compile", "__import__", "open", "globals", "locals", "vars",
    "breakpoint", "input", "memoryview",
}
BANNED_ATTRS = {
    "__dict__", "__code__", "__globals__", "__builtins__", "__subclasses__",
    "__closure__", "__getattribute__", "__loader__", "__spec__", "f_globals", "f_locals",
}
# numpy.random entry points that create an explicitly seeded generator are fine;
# anything that draws from or reseeds the global numpy RNG is not.
ALLOWED_NP_RANDOM = {"default_rng", "Generator", "SeedSequence", "PCG64", "BitGenerator"}
ATTR_FUNCS = {"getattr", "setattr", "delattr"}
MAX_LITERAL_CHARS = 400
CONTRACT = {"init": 2, "step": 2, "observe": 1}


def check(code: str) -> list[str]:
    """Return a list of violations (empty list means the code passes the guard)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError at line {e.lineno}: {e.msg}"]

    problems: list[str] = []
    np_aliases = {"numpy"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "numpy" and a.name != "numpy.random":
                    np_aliases.add(a.asname or "numpy")
    # `np.random` nodes that are the base of an allowed `np.random.<X>` access
    allowed_np_random_bases = {
        id(n.value) for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr in ALLOWED_NP_RANDOM and _is_np_random(n.value, np_aliases)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    problems.append(f"line {node.lineno}: import of '{a.name}' is not allowed")
                if a.name == "numpy.random":
                    problems.append(f"line {node.lineno}: import numpy.random directly is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORT_ROOTS:
                problems.append(f"line {node.lineno}: import from '{node.module}' is not allowed")
            if node.module == "numpy.random" and any(a.name not in ALLOWED_NP_RANDOM for a in node.names):
                problems.append(f"line {node.lineno}: global numpy RNG functions are not allowed")
            if node.module == "numpy" and any(a.name == "random" for a in node.names):
                problems.append(f"line {node.lineno}: importing numpy's global 'random' module is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            problems.append(f"line {node.lineno}: call to '{node.func.id}' is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ATTR_FUNCS:
            name = node.args[1] if len(node.args) > 1 else None
            if not (isinstance(name, ast.Constant) and isinstance(name.value, str)):
                problems.append(f"line {node.lineno}: {node.func.id}() needs a literal attribute name")
            elif name.value in BANNED_ATTRS:
                problems.append(f"line {node.lineno}: access to '{name.value}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_ATTRS:
            problems.append(f"line {node.lineno}: access to '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                problems.append(f"line {node.lineno}: access to '{node.attr}' is not allowed")
            if _is_np_random(node, np_aliases) and id(node) not in allowed_np_random_bases:
                problems.append(f"line {node.lineno}: global numpy RNG '{node.value.id}.random' "
                                "is only allowed as <numpy>.random.default_rng/Generator/SeedSequence/...")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bytes):
                problems.append(f"line {node.lineno}: bytes literals are not allowed")
            elif isinstance(node.value, str) and len(node.value) > MAX_LITERAL_CHARS:
                # docstrings are allowed to be long; they are stripped before measurement
                pass
    problems.extend(_check_long_strings(tree))

    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name, arity in CONTRACT.items():
        fn = defs.get(name)
        if fn is None:
            problems.append(f"missing module-level contract function {name}()")
        elif len(fn.args.args) != arity:
            problems.append(f"{name}() must take exactly {arity} positional argument(s)")
    return problems


def _is_np_random(node: ast.AST, np_aliases: set[str]) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "random"
            and isinstance(node.value, ast.Name) and node.value.id in np_aliases)


def _check_long_strings(tree: ast.AST) -> list[str]:
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and len(node.value) > MAX_LITERAL_CHARS and id(node) not in docstrings):
            out.append(f"line {node.lineno}: string literal longer than {MAX_LITERAL_CHARS} chars")
    return out
