"""
Emergence judge for the Sugarscape minimal-viable-model search.

The judge owns everything about measurement: it runs candidate models through a
fixed contract (judge/contract.py), observes them every step, computes
implementation-agnostic emergent-feature statistics (judge/features.py), and
decides statistical equivalence to canonical Sugarscape over a pre-registered
design of seeds and input parameters (judge/design.json, judge/pipeline.py).

Nothing in this package may be edited by the research loop.
"""

from pathlib import Path
import hashlib

JUDGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = JUDGE_DIR.parent


# Files that cannot change any calibrated quantity: the stage-0 static guard (calibration
# never calls it; it only screens candidate code) and this hashing module itself. Keeping
# them out of the hash lets the guard be tightened without invalidating calibration.
UNHASHED = {"__init__.py", "guard.py"}


def judge_hash() -> str:
    """Hash of every judge source file that affects measurement, plus the design, used to detect stale calibration."""
    h = hashlib.sha256()
    files = sorted(JUDGE_DIR.rglob("*.py")) + [JUDGE_DIR / "design.json"]
    for f in files:
        if "_cache" in f.parts or "reference" in f.parts:
            continue
        if f.parent == JUDGE_DIR and f.name in UNHASHED:
            continue
        h.update(f.relative_to(JUDGE_DIR).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]
