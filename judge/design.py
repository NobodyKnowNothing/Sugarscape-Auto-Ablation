"""Design loading and deterministic, disjoint seed pools."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from judge import JUDGE_DIR

DESIGN_FILE = JUDGE_DIR / "design.json"

# Distinct pools so reference, perturbation, control, candidate and holdout seeds
# never coincide (63-bit seeds derived from SeedSequence; collisions are negligible).
POOLS = {"reference": 1, "perturbation": 2, "control": 3, "candidate": 4, "holdout": 5, "trace": 6}


@dataclass(frozen=True)
class Profile:
    name: str
    cfg: dict
    design: dict

    def __getitem__(self, key):
        return self.cfg[key]

    @property
    def steps(self) -> int:
        return self.cfg["steps"]

    @property
    def alpha(self) -> float:
        return self.design["alpha"]

    @property
    def margin_floor(self) -> dict:
        return {**self.design["margin_floor"], **self.cfg.get("margin_floor", {})}

    def params(self, point: str) -> dict:
        return dict(self.design["points"][point])

    @property
    def screening(self) -> list[str]:
        return list(self.cfg["screening_points"])

    @property
    def audit_points(self) -> list[str]:
        return list(self.cfg["audit_points"])


def load_design() -> dict:
    return json.loads(DESIGN_FILE.read_text(encoding="utf-8"))


def design_hash() -> str:
    return hashlib.sha256(DESIGN_FILE.read_bytes()).hexdigest()[:16]


def load_profile(name: str) -> Profile:
    design = load_design()
    if name not in design["profiles"]:
        raise KeyError(f"unknown profile '{name}', available: {list(design['profiles'])}")
    return Profile(name, design["profiles"][name], design)


def _key_int(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def seeds(pool: str, key: str, point: str, n: int, start: int = 0) -> list[int]:
    """Deterministic seeds for (pool, key, point); `start` lets sequential looks extend a list."""
    ss = np.random.SeedSequence([POOLS[pool], _key_int(key), _key_int(point)])
    state = ss.generate_state(2 * (start + n), dtype=np.uint32)
    vals = (state[0::2].astype(np.uint64) << np.uint64(31)) ^ state[1::2].astype(np.uint64)
    return [int(v) for v in vals[start:start + n]]
