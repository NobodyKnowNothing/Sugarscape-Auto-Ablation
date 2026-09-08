#!/usr/bin/env python3
"""
Mesa Sugarscape G1MT — Clone & Deterministic Flattening Pipeline

Clones Epstein & Axtell's Sugarscape with Traders (sugarscape_g1mt) from the
official Mesa repository and deterministically flattens its multi-file package
(agents.py, model.py, sugar-map.txt) into a single, self-contained Python module
fully compatible with this auto-ablation project's evaluation harness:
  - prepare.py (exports create_model, run_model returning the 9-key metrics dict)
  - parameter_sweeps.py (supports parameterized model runs)
  - complexity.py (valid AST, clean cyclomatic structure)
  - metrics.py (reproduces canonical emergent metrics)

Usage:
    # Clone and flatten from canonical Mesa main branch:
    python flatten_sugarscape.py

    # Output to a specific file with verification:
    python flatten_sugarscape.py --output canonical_strategy.py --verify

    # Overwrite strategy.py directly:
    python flatten_sugarscape.py --output strategy.py --overwrite --verify

    # Flatten from a local folder without cloning:
    python flatten_sugarscape.py --source-dir /path/to/sugarscape_g1mt
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import shutil
import subprocess
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_GITHUB_URL = (
    "https://github.com/mesa/mesa/tree/main/mesa/examples/advanced/sugarscape_g1mt"
)
DEFAULT_OUTPUT_FILE = "canonical_strategy.py"
DEFAULT_MAP_FILE = "sugar-map.txt"


# ---------------------------------------------------------------------------
# URL Parsing & Git Clone
# ---------------------------------------------------------------------------

def parse_github_url(url: str) -> tuple[str, str, str, str, str]:
    """
    Parse a GitHub tree URL into components:
    Returns: (owner, repo, branch, subpath, clone_url)
    """
    # Pattern: https://github.com/{owner}/{repo}/tree/{branch}/{subpath...}
    match = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)(?:/(?:tree|blob)/([^/]+)(?:/(.+))?)?",
        url.strip(),
    )
    if not match:
        raise ValueError(f"Invalid GitHub URL format: {url}")

    owner = match.group(1)
    repo = match.group(2)
    if repo.endswith(".git"):
        repo = repo[:-4]
    branch = match.group(3) or "main"
    subpath = match.group(4) or "mesa/examples/advanced/sugarscape_g1mt"
    clone_url = f"https://github.com/{owner}/{repo}.git"
    return owner, repo, branch, subpath, clone_url


def clone_sparse_subpath(clone_url: str, branch: str, subpath: str, dest_dir: Path) -> Path:
    """
    Perform a shallow, sparse git clone of only the target subpath.
    """
    print(f"[*] Cloning {clone_url} (branch: {branch}, subpath: {subpath})...")
    
    # 1. Clone with --filter=blob:none and --sparse
    cmd_clone = [
        "git", "clone",
        "--depth", "1",
        "--branch", branch,
        "--filter=blob:none",
        "--sparse",
        clone_url,
        str(dest_dir),
    ]
    res = subprocess.run(cmd_clone, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Git clone failed: {res.stderr.strip() or res.stdout.strip()}")

    # 2. Set sparse checkout path
    cmd_sparse = ["git", "-C", str(dest_dir), "sparse-checkout", "set", subpath]
    res = subprocess.run(cmd_sparse, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Git sparse-checkout failed: {res.stderr.strip()}")

    extracted_dir = dest_dir / Path(subpath)
    if not extracted_dir.exists():
        raise FileNotFoundError(f"Subpath '{subpath}' not found in cloned repository at {extracted_dir}")

    return extracted_dir


def download_github_raw_fallback(owner: str, repo: str, branch: str, subpath: str, dest_dir: Path) -> Path:
    """
    Fallback: download required files via raw.githubusercontent.com if git fails.
    """
    print(f"[*] Git unavailable or failed. Falling back to HTTP raw downloads for {owner}/{repo}@{branch}...")
    files_to_fetch = ["agents.py", "model.py", "sugar-map.txt", "__init__.py"]
    base_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{subpath}"
    
    target_dir = dest_dir / Path(subpath)
    target_dir.mkdir(parents=True, exist_ok=True)

    for fname in files_to_fetch:
        file_url = f"{base_url}/{fname}"
        local_path = target_dir / fname
        try:
            req = urllib.request.Request(file_url, headers={"User-Agent": "Sugarscape-Auto-Ablation"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                local_path.write_bytes(resp.read())
            print(f"    Downloaded {fname}")
        except urllib.error.HTTPError as e:
            if fname == "sugar-map.txt":
                # Try alternate filename or existing map
                print(f"    Notice: {fname} HTTP {e.code}, will check local repository")
            else:
                raise RuntimeError(f"Failed downloading required file {fname} from {file_url}: {e}")

    return target_dir


# ---------------------------------------------------------------------------
# Deterministic AST & Code Transformations
# ---------------------------------------------------------------------------

COMPATIBILITY_HEADER = '''# ===========================================================================
# Mesa Compatibility Bridge (supports both Mesa 3.x and Mesa 4.x)
# ===========================================================================
try:
    from mesa.discrete_space.property_layer import PropertyLayer

    if hasattr(PropertyLayer, "data"):
        if not hasattr(PropertyLayer, "__setitem__"):
            PropertyLayer.__setitem__ = lambda self, idx, val: setattr(self, "data", val)
        if not hasattr(PropertyLayer, "__add__"):
            PropertyLayer.__add__ = lambda self, other: self.data + other
            PropertyLayer.__radd__ = lambda self, other: other + self.data

    _orig_add_prop = OrthogonalVonNeumannGrid.add_property_layer

    def _compat_add_property_layer(self, name_or_layer, data=None):
        if data is not None and hasattr(PropertyLayer, "from_data"):
            return _orig_add_prop(self, PropertyLayer.from_data(name_or_layer, data))
        return _orig_add_prop(self, name_or_layer)

    OrthogonalVonNeumannGrid.add_property_layer = _compat_add_property_layer
except ImportError:
    pass
'''

PROJECT_HARNESS_CODE = '''# ===========================================================================
# Project Evaluation Harness Interface (prepare.py / parameter_sweeps.py)
# ===========================================================================

def create_model(seed: int = 42, **params) -> SugarscapeG1mt:
    """
    Instantiate SugarscapeG1mt model configured with seed and parameter overrides.
    Works seamlessly whether the model uses Mesa 4.x Scenario or Mesa 3.x kwargs.
    """
    p = dict(params)
    p.pop("steps", None)
    init_pop = p.get("initial_population", 200)

    if "SugarScapeScenario" in globals():
        scenario_kwargs = {k: v for k, v in p.items() if hasattr(SugarScapeScenario, k)}
        sc = SugarScapeScenario(rng=seed, **scenario_kwargs)
        model = SugarscapeG1mt(scenario=sc)
    else:
        if "rng" not in p and seed is not None:
            p["rng"] = seed
        model = SugarscapeG1mt(**p)

    model.initial_population = init_pop
    return model


def run_model(model: SugarscapeG1mt, steps: int = 200) -> dict[str, Any]:
    """
    Execute simulation steps and extract the standardized 9-key metrics dictionary.
    """
    try:
        model.run_model(step_count=steps)
    except (ValueError, IndexError) as e:
        return {"error": f"Mesa crash: {e}"}

    df = model.datacollector.get_model_vars_dataframe()
    cumul_trade_volume = int(df["Trade Volume"].sum()) if "Trade Volume" in df else 0

    agents = list(model.agents)
    agent_wealths = [float(a.sugar + a.spice) for a in agents]

    trade_prices = []
    for a in agents:
        if hasattr(a, "prices"):
            trade_prices.extend(a.prices)

    agent_positions = []
    for a in agents:
        if hasattr(a, "cell") and a.cell is not None:
            agent_positions.append(a.cell.coordinate)

    return {
        "agent_wealths": agent_wealths,
        "final_population": len(agents),
        "initial_population": getattr(model, "initial_population", 200),
        "trade_prices": trade_prices,
        "trade_volume": cumul_trade_volume,
        "agent_positions": agent_positions,
        "grid_width": getattr(model, "width", 50),
        "grid_height": getattr(model, "height", 50),
        "steps_run": steps,
    }
'''


def extract_ast_segments(source_code: str) -> tuple[list[str], list[str], list[str]]:
    """
    Parse Python source and extract:
      1. imports (raw source)
      2. top-level function source segments
      3. top-level class source segments
    """
    tree = ast.parse(source_code)
    imports: list[str] = []
    functions: list[str] = []
    classes: list[str] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(source_code, node)
            if seg:
                imports.append(seg.strip())
        elif isinstance(node, ast.FunctionDef):
            seg = ast.get_source_segment(source_code, node)
            if seg:
                functions.append(seg.strip())
        elif isinstance(node, ast.ClassDef):
            seg = ast.get_source_segment(source_code, node)
            if seg:
                classes.append(seg.strip())

    return imports, functions, classes


def build_deterministic_imports(all_raw_imports: list[str], uses_scenario: bool) -> str:
    """
    Filter out internal intra-package imports and assemble clean, deterministically
    sorted import statements adhering to PEP 8.
    """
    # Filter intra-package references to agents or sugarscape
    filtered = []
    for imp in all_raw_imports:
        if any(token in imp for token in ["sugarscape_g1mt", ".agents", "import agents"]):
            continue
        filtered.append(imp)

    # Standard imports guaranteed to be present deterministically
    stdlib_imports = [
        "import math",
        "from pathlib import Path",
        "from typing import Any",
    ]

    third_party_imports = [
        "import mesa",
        "from mesa.discrete_space import CellAgent, OrthogonalVonNeumannGrid",
        "import numpy as np",
    ]

    if uses_scenario:
        third_party_imports.append("from mesa.experimental.scenarios import Scenario")

    # Combine with any additional third party imports found (excluding duplicates)
    known = set(stdlib_imports) | set(third_party_imports)
    additional = []
    for imp in sorted(set(filtered)):
        if imp not in known:
            additional.append(imp)

    sections = [
        "\n".join(stdlib_imports),
        "\n".join(sorted(third_party_imports)),
    ]
    if additional:
        sections.append("\n".join(additional))

    return "\n\n".join(sections)


def flatten_sugarscape(source_dir: Path) -> str:
    """
    Deterministically flatten agents.py and model.py from source_dir into
    a single standalone module.
    """
    agents_file = source_dir / "agents.py"
    model_file = source_dir / "model.py"

    if not agents_file.exists():
        raise FileNotFoundError(f"Missing required agents.py in {source_dir}")
    if not model_file.exists():
        raise FileNotFoundError(f"Missing required model.py in {source_dir}")

    agents_src = agents_file.read_text(encoding="utf-8")
    model_src = model_file.read_text(encoding="utf-8")

    agents_imports, agents_funcs, agents_classes = extract_ast_segments(agents_src)
    model_imports, model_funcs, model_classes = extract_ast_segments(model_src)

    all_imports = agents_imports + model_imports
    uses_scenario = any("Scenario" in c for c in model_classes) or any("Scenario" in imp for imp in all_imports)

    import_block = build_deterministic_imports(all_imports, uses_scenario)

    # Deterministic assembly in strict dependency order:
    # 1. Imports
    # 2. Compatibility Bridge
    # 3. Agent Helper Functions (e.g. get_distance)
    # 4. Agent Classes (Trader)
    # 5. Model Helper Functions (e.g. flatten, geometric_mean, get_trade)
    # 6. Scenario Class (if present)
    # 7. Model Class (SugarscapeG1mt)
    # 8. Project Evaluation Harness (create_model, run_model)
    blocks: list[str] = [
        '"""\nCanonical Mesa Sugarscape G1MT — Deterministically Flattened Module.\n'
        "Generated directly from upstream Mesa Sugarscape with Traders repository.\n"
        '"""\n',
        import_block,
        COMPATIBILITY_HEADER,
        "# ===========================================================================\n"
        "# Agent Helper Functions & Trader Class (from agents.py)\n"
        "# ===========================================================================",
    ]

    for fn in agents_funcs:
        blocks.append(fn)

    for cls in agents_classes:
        blocks.append(cls)

    blocks.append(
        "# ===========================================================================\n"
        "# Model Helper Functions & Sugarscape Model (from model.py)\n"
        "# ==========================================================================="
    )

    for fn in model_funcs:
        blocks.append(fn)

    for cls in model_classes:
        blocks.append(cls)

    blocks.append(PROJECT_HARNESS_CODE)

    # Normalize line breaks and trailing whitespaces deterministically
    raw_output = "\n\n".join(b.strip() for b in blocks if b.strip()) + "\n"
    # Normalize CRLF -> LF
    normalized = "\n".join(line.rstrip() for line in raw_output.replace("\r\n", "\n").splitlines()) + "\n"
    return normalized


# ---------------------------------------------------------------------------
# Verification & Self-Test
# ---------------------------------------------------------------------------

def verify_flattened_code(code: str, sugar_map_path: Path) -> dict[str, Any]:
    """
    Validate Python syntax, instantiate model, run 5 simulation steps,
    and verify the 9-key output contract.
    """
    print("[*] Verifying flattened code...")
    # 1. Syntax check
    tree = ast.parse(code)
    print("    [✓] AST syntax check passed")

    # 2. Execution in isolated namespace
    ns: dict[str, Any] = {"__file__": str(sugar_map_path.resolve())}
    exec(code, ns)

    if "create_model" not in ns or "run_model" not in ns:
        raise AssertionError("Module must export create_model() and run_model()")
    print("    [✓] create_model and run_model exports confirmed")

    # 3. Model creation & run smoke test
    model = ns["create_model"](seed=42, width=50, height=50, initial_population=100)
    print(f"    [✓] Model initialized: {model}")

    metrics_dict = ns["run_model"](model, steps=5)
    expected_keys = {
        "agent_wealths", "final_population", "initial_population",
        "trade_prices", "trade_volume", "agent_positions",
        "grid_width", "grid_height", "steps_run"
    }

    missing = expected_keys - set(metrics_dict.keys())
    if missing:
        raise AssertionError(f"run_model() return missing expected keys: {missing}")

    print(f"    [✓] run_model() executed 5 steps successfully:")
    print(f"        Surviving agents: {metrics_dict['final_population']}/{metrics_dict['initial_population']}")
    print(f"        Trade volume:     {metrics_dict['trade_volume']}")
    print(f"        All 9 contract keys verified.")

    return {
        "ast_nodes": sum(1 for _ in ast.walk(tree)),
        "metrics_keys": list(metrics_dict.keys()),
        "final_pop": metrics_dict["final_population"],
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clone upstream Mesa sugarscape_g1mt and deterministically flatten into a project-compatible module."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_GITHUB_URL,
        help="GitHub URL to sugarscape_g1mt (default: canonical Mesa main branch)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Destination path for flattened file (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Flatten an existing local directory instead of cloning from GitHub",
    )
    parser.add_argument(
        "--map-output",
        type=str,
        default=DEFAULT_MAP_FILE,
        help=f"Destination for sugar-map.txt (default: ./{DEFAULT_MAP_FILE})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing destination output file",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Run syntax check and simulation smoke test on output (default: True)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_false",
        dest="verify",
        help="Disable verification check",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    map_output_path = Path(args.map_output).resolve()

    if output_path.exists() and not args.overwrite:
        print(f"[!] Output file {output_path} already exists. Use --overwrite to replace.")
        sys.exit(1)

    print("═══════════════════════════════════════════════════════════════")
    print("   Sugarscape G1MT Clone & Deterministic Flattening Pipeline   ")
    print("═══════════════════════════════════════════════════════════════")

    # Determine source directory (local vs clone)
    if args.source_dir:
        source_dir = Path(args.source_dir).resolve()
        if not source_dir.exists():
            print(f"[!] Error: Source directory {source_dir} does not exist.")
            sys.exit(1)
        print(f"[*] Using local source directory: {source_dir}")
        flattened_code = flatten_sugarscape(source_dir)
        source_map = source_dir / "sugar-map.txt"
        if source_map.exists() and not map_output_path.exists():
            shutil.copy2(source_map, map_output_path)
            print(f"[*] Copied {source_map.name} to {map_output_path}")
    else:
        owner, repo, branch, subpath, clone_url = parse_github_url(args.url)
        with tempfile.TemporaryDirectory(prefix="mesa_sugarscape_") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            try:
                extracted_dir = clone_sparse_subpath(clone_url, branch, subpath, temp_dir)
            except Exception as e:
                print(f"[!] Git clone failed ({e}). Attempting fallback download...")
                extracted_dir = download_github_raw_fallback(owner, repo, branch, subpath, temp_dir)

            source_map = extracted_dir / "sugar-map.txt"
            if source_map.exists() and not map_output_path.exists():
                shutil.copy2(source_map, map_output_path)
                print(f"[*] Copied canonical {source_map.name} to {map_output_path}")

            flattened_code = flatten_sugarscape(extracted_dir)

    # Compute deterministic SHA-256
    sha256 = hashlib.sha256(flattened_code.encode("utf-8")).hexdigest()
    print(f"[*] Deterministic SHA-256: {sha256}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(flattened_code, encoding="utf-8")
    print(f"[✓] Successfully wrote flattened module to: {output_path}")
    print(f"    File size: {len(flattened_code)} bytes, {len(flattened_code.splitlines())} lines")

    # Optional verification
    if args.verify:
        verify_flattened_code(flattened_code, map_output_path)
        print("[✓] All verification checks PASSED.")

    print("═══════════════════════════════════════════════════════════════")
    print(f"Ready for use: python prepare.py --compare {output_path.name}")
    print("═══════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
