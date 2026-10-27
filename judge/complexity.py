"""
Complexity v2: canonical compressed size of all executed model logic.

    C(model) = min over {gzip, bz2, lzma} of |compress(canon(strategy.py + traced dependency code))|
               - C(empty contract template)

* Traced dependencies: the model is run briefly under sys.settrace and every
  Python function executed outside the "reference machine" is recorded. The
  reference machine is CPython + the standard library + numpy (fixed, like the
  reference machine of Kolmogorov complexity). Everything else that runs (Mesa,
  or any other third-party package) is extracted from source - the executed
  functions plus the class-level statements of their classes - and compressed
  jointly with strategy.py. Replacing general library machinery with small
  purpose-built rules is therefore rewarded only by the real difference.
* canon(): strips docstrings, comments, annotations and asserts, then
  alpha-renames every identifier in first-use order (one mapping across the whole
  bundle), so renaming, reformatting and comment edits give delta C = 0.
"""

from __future__ import annotations

import ast
import bz2
import gzip
import json
import keyword
import lzma
import os
import sys
import sysconfig
from itertools import count
from pathlib import Path

import numpy as np

from judge import JUDGE_DIR
from judge.contract import CACHE_DIR, code_hash, load_module, materialize, run_model

TEMPLATE = '''
def init(env, rng):
    return None

def step(state, rng):
    pass

def observe(state):
    return {"ids": [], "positions": [], "holdings": [], "exchanges": [], "resources": []}
'''

# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

_BODY_FIELDS = ("body", "orelse", "finalbody")


class _Strip(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        for field in _BODY_FIELDS:
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            if (field == "body" and isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                    and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]
            if field == "body" and not body and not isinstance(node, ast.Module):
                body = [ast.Pass()]
            setattr(node, field, body)
        return node

    def visit_FunctionDef(self, node):
        node.returns = None
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None
        return ast.copy_location(ast.Assign(targets=[node.target], value=self.visit(node.value)), node)

    def visit_Assert(self, node):
        return None


def _names():
    letters = "abcdefghijklmnopqrstuvwxyz"
    for n in count(1):
        for idx in np.ndindex(*([26] * n)):
            s = "".join(letters[i] for i in idx)
            if not keyword.iskeyword(s):
                yield s


class _Rename(ast.NodeTransformer):
    def __init__(self):
        self.mapping: dict[str, str] = {}
        self.gen = _names()

    def n(self, s: str) -> str:
        if s not in self.mapping:
            self.mapping[s] = next(self.gen)
        return self.mapping[s]

    def visit_Name(self, node):
        node.id = self.n(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self.n(node.arg)
        return node

    def visit_FunctionDef(self, node):
        node.name = self.n(node.name)
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node.attr = self.n(node.attr)
        return node

    def visit_keyword(self, node):
        if node.arg:
            node.arg = self.n(node.arg)
        return self.generic_visit(node)

    def visit_alias(self, node):
        if node.asname:
            node.asname = self.n(node.asname)
        return node

    def visit_Global(self, node):
        node.names = [self.n(x) for x in node.names]
        return node

    visit_Nonlocal = visit_Global

    def visit_ExceptHandler(self, node):
        if node.name:
            node.name = self.n(node.name)
        return self.generic_visit(node)


def canon_bundle(sources: list[str]) -> list[str]:
    """Canonicalize several sources with one shared alpha-renaming."""
    renamer = _Rename()
    out = []
    for src in sources:
        tree = _Strip().visit(ast.parse(src))
        tree = renamer.visit(tree)
        ast.fix_missing_locations(tree)
        out.append(ast.unparse(tree))
    return out


def canon(src: str) -> str:
    return canon_bundle([src])[0]


def compressed_sizes(text: str) -> dict[str, int]:
    data = text.encode("utf-8")
    return {
        "gzip": len(gzip.compress(data, compresslevel=9, mtime=0)),
        "bz2": len(bz2.compress(data, 9)),
        "lzma": len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)),
    }


_TEMPLATE_C = None


def template_size() -> int:
    global _TEMPLATE_C
    if _TEMPLATE_C is None:
        _TEMPLATE_C = min(compressed_sizes(canon(TEMPLATE)).values())
    return _TEMPLATE_C


# ---------------------------------------------------------------------------
# Dependency tracing
# ---------------------------------------------------------------------------

def _norm(p: str) -> str:
    return os.path.normcase(os.path.realpath(p))


def reference_dirs() -> list[str]:
    paths = sysconfig.get_paths()
    dirs = {paths["stdlib"], paths["platstdlib"], os.path.dirname(np.__file__), str(JUDGE_DIR)}
    return sorted(_norm(d) for d in dirs)


def _is_counted(filename: str, ref_dirs: list[str]) -> bool:
    if not filename or filename.startswith("<"):
        return False
    f = _norm(filename)
    if f.endswith(".pyc"):
        return False
    numpy_dir, judge_dir = _norm(os.path.dirname(np.__file__)), _norm(str(JUDGE_DIR))
    if f.startswith(numpy_dir + os.sep) or f.startswith(judge_dir + os.sep):
        return False
    parts = Path(f).parts
    if "site-packages" in parts or "dist-packages" in parts:
        return True   # third-party code (a venv's Lib dir can also look like a stdlib path)
    return not any(f.startswith(d + os.sep) for d in ref_dirs)


def _trace_worker(path: str, runs: list[tuple[dict, int, int]]) -> list[tuple[str, int, str]]:
    ref_dirs = reference_dirs()
    load_module(Path(path))  # imports happen before tracing: module bodies are infrastructure
    seen: set = set()
    records: list[tuple[str, int, str]] = []

    def tracer(frame, event, arg):
        if event == "call":
            co = frame.f_code
            if co not in seen:
                seen.add(co)
                if co.co_name != "<module>" and _is_counted(co.co_filename, ref_dirs):
                    records.append((co.co_filename, co.co_firstlineno, co.co_name))
        return None

    sys.settrace(tracer)
    try:
        for params, seed, steps in runs:
            run_model(Path(path), params, seed, steps)
    finally:
        sys.settrace(None)
    return records


def _first_line(node) -> int:
    decos = getattr(node, "decorator_list", None) or []
    return min([node.lineno] + [d.lineno for d in decos])


def extract_dependency_source(records: list[tuple[str, int, str]]) -> tuple[str, list[str]]:
    """Turn traced (file, firstlineno, name) records into one source text plus a manifest."""
    by_file: dict[str, set[int]] = {}
    for fn, line, _ in records:
        by_file.setdefault(fn, set()).add(line)

    chunks, manifest = [], []
    for fn in sorted(by_file, key=_norm):
        try:
            tree = ast.parse(Path(fn).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        parents = {}
        funcs = []
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append(node)
        by_first = {_first_line(f): f for f in funcs}

        def enclosing_func(line):
            best = None
            for f in funcs:
                if f.lineno <= line <= f.end_lineno and (best is None or f.lineno > best.lineno):
                    best = f
            return best

        selected = set()
        for line in by_file[fn]:
            node = by_first.get(line) or enclosing_func(line)
            if node is not None:
                selected.add(node)
        # drop functions nested inside other selected functions
        selected = {n for n in selected
                    if not any(o is not n and o.lineno <= n.lineno and n.end_lineno <= o.end_lineno for o in selected)}

        classes = {}
        for node in selected:
            p = parents.get(node)
            while p is not None:
                if isinstance(p, ast.ClassDef):
                    classes[p.lineno] = p
                p = parents.get(p)
        rel = _short(fn)
        items = []
        for cls in classes.values():
            skel = ast.ClassDef(
                name=cls.name, bases=cls.bases, keywords=cls.keywords, decorator_list=cls.decorator_list,
                body=[s for s in cls.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))] or [ast.Pass()],
                type_params=getattr(cls, "type_params", []),
            )
            items.append((cls.lineno, ast.unparse(ast.fix_missing_locations(skel))))
            manifest.append(f"{rel}:{cls.name} (class body)")
        for node in selected:
            items.append((node.lineno, ast.unparse(node)))
            owner = next((c.name for c in classes.values() if c.lineno <= node.lineno <= c.end_lineno), None)
            manifest.append(f"{rel}:{owner + '.' if owner else ''}{node.name}")
        chunks.extend(src for _, src in sorted(items))
    return "\n\n".join(chunks), sorted(manifest)


def _short(filename: str) -> str:
    f = Path(filename)
    parts = f.parts
    for marker in ("site-packages", "dist-packages"):
        if marker in parts:
            return "/".join(parts[parts.index(marker) + 1:])
    return f.name


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def trace_runs(profile) -> list[tuple[dict, int, int]]:
    from judge.design import seeds
    runs = []
    for point in ("default", "default_notrade"):
        for s in seeds("trace", "complexity", point, profile["trace_seeds"]):
            runs.append((profile.params(point), s, profile["trace_steps"]))
    return runs


def _trace_in_fresh_process(path: str, runs) -> list:
    """Trace in a brand-new interpreter so nothing else (test runners, earlier runs) leaks in."""
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
        return pool.apply_async(_trace_worker, (path, runs)).get(timeout=600)


def measure(code: str, profile, use_cache: bool = True) -> dict:
    """Full complexity measurement (requires a short traced run of the model in a fresh process)."""
    h = code_hash(code)
    cache = CACHE_DIR / f"complexity_{h}_{profile.name}.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    path = materialize(code)
    runs = trace_runs(profile)
    records = _trace_in_fresh_process(str(path), runs)
    dep_src, manifest = extract_dependency_source(records)
    strat_c, dep_c = canon_bundle([code, dep_src or "pass"])
    joint = strat_c + "\n\n" + dep_c if dep_src else strat_c
    sizes = compressed_sizes(joint)
    strategy_only = compressed_sizes(strat_c)
    t = template_size()
    result = {
        "C": min(sizes.values()) - t,
        "C_strategy_only": min(strategy_only.values()) - t,
        "compressor": min(sizes, key=sizes.get),
        "sizes": sizes,
        "template_bytes": t,
        "canonical_chars": len(joint),
        "dependency_functions": len([m for m in manifest if "(class body)" not in m]),
        "dependency_manifest": manifest,
        "code_hash": h,
    }
    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def static_size(code: str) -> dict:
    """Strategy-only canonical size (no tracing); for quick diagnostics and the edit harness."""
    sizes = compressed_sizes(canon(code))
    return {"C_strategy_only": min(sizes.values()) - template_size(), "sizes": sizes}
