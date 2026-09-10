"""
The fixed world and the model contract.

Same world, same measurements, free rules: the judge supplies the environment
(landscape, population, parameters), owns the run loop and the RNG, and reads
the model only through `observe()`. A candidate `strategy.py` must define:

    init(env: dict, rng: np.random.Generator) -> state
    step(state, rng: np.random.Generator) -> None
    observe(state) -> dict with keys
        ids        int[N]        unique agent ids
        positions  int[N, 2]     (x, y) cell of each agent
        holdings   float[N, G]   goods held by each agent (G >= 1)
        exchanges  list of (i, j, dq)  this step's exchanges; dq (length G) is
                                 the change in agent i's holdings (j gets -dq)
        resources  float[G', W, H]  current resource layers

Anything missing, malformed, non-finite or non-deterministic is a hard reject.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from judge import JUDGE_DIR, PROJECT_DIR

GRID_W = 50
GRID_H = 50
CACHE_DIR = JUDGE_DIR / "_cache"

DEFAULT_ENV = {
    "initial_population": 200,
    "endowment_min": 25,
    "endowment_max": 50,
    "metabolism_min": 1,
    "metabolism_max": 5,
    "vision_min": 1,
    "vision_max": 5,
    "enable_trade": True,
    "regrowth_rate": 1.0,
    "width": GRID_W,
    "height": GRID_H,
}

_LANDSCAPE = None


class ContractError(Exception):
    """The candidate violated the model contract."""


def landscape() -> np.ndarray:
    """Capacity layers (2, W, H): the canonical sugar map and its mirror for spice."""
    global _LANDSCAPE
    if _LANDSCAPE is None:
        sugar = np.genfromtxt(PROJECT_DIR / "sugar-map.txt")
        _LANDSCAPE = np.stack([sugar, np.flip(sugar, 1)])
        _LANDSCAPE.setflags(write=False)
    return _LANDSCAPE


def make_env(params: dict) -> dict:
    env = {**DEFAULT_ENV, **params}
    env["landscape"] = landscape()
    return env


@dataclass
class Snapshot:
    ids: np.ndarray
    positions: np.ndarray
    holdings: np.ndarray
    exchanges: list
    resources: np.ndarray

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def wealth(self) -> np.ndarray:
        return self.holdings.sum(axis=1) if self.n else np.zeros(0)


def validate_snapshot(obs, env: dict, n_goods: int | None) -> Snapshot:
    """Convert and validate an observe() result. Raises ContractError on any problem."""
    if not isinstance(obs, dict):
        raise ContractError(f"observe() must return a dict, got {type(obs).__name__}")
    missing = {"ids", "positions", "holdings", "exchanges", "resources"} - set(obs)
    if missing:
        raise ContractError(f"observe() missing keys: {sorted(missing)}")
    try:
        ids = np.asarray(obs["ids"], dtype=np.int64).reshape(-1)
        n = len(ids)
        positions = np.asarray(obs["positions"], dtype=np.int64).reshape(n, 2)
        holdings = np.asarray(obs["holdings"], dtype=np.float64)
        holdings = np.zeros((0, n_goods or 1)) if n == 0 else holdings.reshape(n, -1)
        resources = np.asarray(obs["resources"], dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ContractError(f"observe() fields have wrong shape/type: {e}") from e
    if len(np.unique(ids)) != n:
        raise ContractError("observe() ids are not unique")
    w, h = env["width"], env["height"]
    if n and (positions.min() < 0 or positions[:, 0].max() >= w or positions[:, 1].max() >= h):
        raise ContractError("observe() positions outside the grid")
    g = holdings.shape[1]
    if g < 1:
        raise ContractError("observe() holdings must have at least one good")
    if n_goods is not None and g != n_goods:
        raise ContractError(f"number of goods changed during the run ({n_goods} -> {g})")
    if not np.all(np.isfinite(holdings)):
        raise ContractError("observe() holdings contain non-finite values")
    if resources.ndim == 2:
        resources = resources[None]
    if resources.ndim != 3 or resources.shape[1:] != (w, h):
        raise ContractError(f"observe() resources must be (G, {w}, {h}), got {resources.shape}")
    if not np.all(np.isfinite(resources)):
        raise ContractError("observe() resources contain non-finite values")
    exchanges = []
    for ex in obs["exchanges"]:
        try:
            i, j, dq = ex
            dq = np.asarray(dq, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as e:
            raise ContractError(f"malformed exchange {ex!r}: {e}") from e
        if len(dq) != g or not np.all(np.isfinite(dq)):
            raise ContractError(f"exchange quantity vector must have {g} finite entries")
        exchanges.append((int(i), int(j), dq))
    return Snapshot(ids, positions, holdings, exchanges, resources)


def snapshot_digest(h, snap: Snapshot) -> None:
    """Feed a canonical byte representation of a snapshot into a hash (identity checks)."""
    order = np.argsort(snap.ids, kind="stable")
    h.update(snap.ids[order].tobytes())
    h.update(snap.positions[order].tobytes())
    h.update(np.round(snap.holdings[order], 9).tobytes())
    for i, j, dq in snap.exchanges:
        h.update(f"{i},{j},".encode())
        h.update(np.round(dq, 9).tobytes())


# ---------------------------------------------------------------------------
# Loading candidate code
# ---------------------------------------------------------------------------

def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def materialize(code: str) -> Path:
    """Write candidate code to the judge cache (so tracing and inspect can read it)."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"m_{code_hash(code)}.py"
    if not path.exists():
        path.write_text(code, encoding="utf-8")
    return path


_MODULES: dict[str, object] = {}


def load_module(path: Path):
    key = str(path)
    if key in _MODULES:
        return _MODULES[key]
    name = f"_judge_candidate_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    for fn in ("init", "step", "observe"):
        if not callable(getattr(mod, fn, None)):
            raise ContractError(f"strategy must define {fn}()")
    _MODULES[key] = mod
    return mod


def run_model(path: Path, params: dict, seed: int, steps: int, recorder=None,
              time_limit: float = 300.0, digest=None):
    """
    Run one simulation under the judge's loop. Calls `recorder.update(t, snap)`
    for t = 0..steps. Returns the number of goods. Raises ContractError (or the
    model's own exception) on failure.
    """
    mod = load_module(path)
    env = make_env(params)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    state = mod.init(env, rng)
    snap = validate_snapshot(mod.observe(state), env, None)
    n_goods = snap.holdings.shape[1] if snap.n else None
    if recorder is not None:
        recorder.update(0, snap)
    if digest is not None:
        snapshot_digest(digest, snap)
    for t in range(1, steps + 1):
        mod.step(state, rng)
        snap = validate_snapshot(mod.observe(state), env, n_goods)
        if n_goods is None and snap.n:
            n_goods = snap.holdings.shape[1]
        if recorder is not None:
            recorder.update(t, snap)
        if digest is not None:
            snapshot_digest(digest, snap)
        if time.perf_counter() - t0 > time_limit:
            raise TimeoutError(f"run exceeded {time_limit:.0f}s at step {t}")
    return n_goods
