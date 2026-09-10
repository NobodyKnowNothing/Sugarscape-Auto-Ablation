"""
Acceptance pipeline.

Stage 0  static guard + traced canonical complexity (must drop by >= delta)
Stage 1  smoke + determinism (judge-owned loop, contract validation); identity
         digests label pure refactors (label only, never a reason to accept)
Stage 2  presence of E1-E4 at the default point on fresh seeds
Stage 3  sequential TOST of E1-E4 statistics at every screening design point
Stage 4  E5 response-surface equivalence across the screening design
Audit    full design (screening + Latin hypercube) on holdout seeds

Every comparison is against canonical Sugarscape (judge/reference), never the parent.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from judge import JUDGE_DIR, judge_hash
from judge import complexity as cx
from judge import guard
from judge.contract import code_hash, materialize
from judge.design import Profile, design_hash, seeds
from judge.features import FEATURE_NAMES, FEATURES, SCALAR_STATS, VECTOR_STATS, presence
from judge.parallel import job, run_jobs
from judge.stats import Comparison, applicable, effect_compare, scalar_compare, vector_compare

REFERENCE_ROOT = JUDGE_DIR / "reference"


# ---------------------------------------------------------------------------
# Running groups of seeds
# ---------------------------------------------------------------------------

class RunError(Exception):
    pass


def collect(code: str, profile: Profile, pool: str, key: str, points: list[str], n: int,
            start: int = 0, digest: bool = False) -> dict[str, list[dict]]:
    """Run `n` seeds (from `start`) at each point; raise RunError on any failed run."""
    path = materialize(code)
    jobs, index = [], []
    for p in points:
        for s in seeds(pool, key, p, n, start):
            jobs.append(job(path, p, profile.params(p), s, profile.steps, digest))
            index.append(p)
    results = run_jobs(jobs, timeout=max(600, 30 * len(jobs)))
    out: dict[str, list[dict]] = {p: [] for p in points}
    for p, j, r in zip(index, jobs, results):
        if not r["ok"]:
            raise RunError(f"run failed at point {p} (seed {j.seed}): {r['error']}")
        stats = r["stats"]
        stats["_seed"] = j.seed
        if digest:
            stats["_digest"] = r["digest"]
        out[p].append(stats)
    return out


# ---------------------------------------------------------------------------
# Reference / margins storage
# ---------------------------------------------------------------------------

def reference_dir(profile_name: str):
    return REFERENCE_ROOT / profile_name


def load_reference(profile_name: str) -> tuple[dict, dict, dict]:
    d = reference_dir(profile_name)
    ref = json.loads((d / "reference.json").read_text(encoding="utf-8"))
    margins = json.loads((d / "margins.json").read_text(encoding="utf-8"))
    calib = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
    return ref, margins, calib


def calibration_status(profile_name: str) -> tuple[bool, str]:
    """(usable, message). Usable means present, fresh and meeting the control criteria."""
    d = reference_dir(profile_name)
    if not (d / "calibration.json").exists():
        return False, f"no calibration for profile '{profile_name}' - run calibrate_judge.py --profile {profile_name}"
    calib = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
    if calib.get("judge_hash") != judge_hash():
        return False, "calibration is stale (judge code or design changed) - re-run calibrate_judge.py"
    if not calib.get("passed"):
        return False, "calibration exists but does not meet the positive/negative control criteria"
    return True, "calibration fresh and passing"


# ---------------------------------------------------------------------------
# Assessment (shared by the pipeline, calibration and certificates)
# ---------------------------------------------------------------------------

@dataclass
class Test:
    feature: str
    point: str
    stat: str
    kind: str          # scalar | vector | effect | presence
    margin: float
    status: str        # pass | fail | undecided
    comparison: dict | None = None
    note: str = ""

    def ratio(self) -> float:
        c = self.comparison or {}
        if self.kind == "presence" or not c:
            return 0.0
        hi = c.get("hi")
        lo = c.get("lo")
        if hi is None or lo is None:
            return math.inf
        ucb = hi if self.kind == "vector" else max(abs(lo), abs(hi))
        return ucb / self.margin if self.margin > 0 else math.inf


@dataclass
class Assessment:
    tests: list[Test] = field(default_factory=list)

    def feature_status(self) -> dict[str, str]:
        out = {}
        for f in FEATURES + ["E5"]:
            ts = [t for t in self.tests if t.feature == f]
            if any(t.status == "fail" for t in ts):
                out[f] = "fail"
            elif any(t.status == "undecided" for t in ts):
                out[f] = "undecided"
            else:
                out[f] = "pass"
        return out

    @property
    def all_pass(self) -> bool:
        return all(s == "pass" for s in self.feature_status().values())

    @property
    def any_clear_fail(self) -> bool:
        return any(t.status == "fail" for t in self.tests)

    def eq_distance(self) -> float:
        r = [t.ratio() for t in self.tests if t.kind != "presence"]
        return float(max(r)) if r else 0.0

    def feedback(self) -> dict[str, list[str]]:
        """Qualitative per-feature feedback (no numbers, no margins) for the proposer."""
        fb: dict[str, list[str]] = {}
        for t in self.tests:
            if t.status == "pass":
                continue
            where = f"at '{t.point}'"
            if t.kind == "presence":
                msg = f"{t.note} {where}"
            else:
                c = Comparison(**{k: (math.inf if v is None else v) for k, v in t.comparison.items()})
                verdict = "differs from Sugarscape" if t.status == "fail" else "not shown equivalent"
                msg = f"{t.stat} {verdict} ({c.direction()}) {where}"
            fb.setdefault(f"{t.feature} {FEATURE_NAMES[t.feature]}", []).append(msg)
        return fb


def _decide(comp: Comparison, margin: float) -> str:
    if comp.equivalent(margin):
        return "pass"
    if comp.clearly_different(margin):
        return "fail"
    return "undecided"


def assess(cand: dict[str, list[dict]], ref: dict, margins: dict, points: list[str], alpha: float,
           features: list[str] | None = None) -> Assessment:
    feats = features or (FEATURES + ["E5"])
    a = Assessment()
    active = margins["active"]
    for p in points:
        rruns = ref["points"][p]["runs"]
        cruns = cand[p]
        for f in FEATURES:
            if f not in feats:
                continue
            if ref["points"][p]["presence"].get(f):
                ok, why = presence(f, cruns, alpha)
                a.tests.append(Test(f, p, "presence", "presence", 0.0, "pass" if ok else "fail", None, why))
            for s in SCALAR_STATS[f]:
                rv = [r[s] for r in rruns]
                if not active.get(s, True) or not applicable(rv) or s not in margins["scalar"][p]:
                    continue
                comp = scalar_compare([r[s] for r in cruns], rv, alpha)
                m = margins["scalar"][p][s]
                a.tests.append(Test(f, p, s, "scalar", m, _decide(comp, m), comp.to_dict()))
            for s in VECTOR_STATS[f]:
                if not active.get(s, True) or s not in margins["vector"][p]:
                    continue
                comp = vector_compare([r[s] for r in cruns], [r[s] for r in rruns], alpha)
                m = margins["vector"][p][s]
                a.tests.append(Test(f, p, s, "vector", m, _decide(comp, m), comp.to_dict()))
    if "E5" in feats and "default" in points:
        for p in points:
            if p == "default" or p not in margins["effect"]:
                continue
            for s, m in margins["effect"][p].items():
                if not active.get(s, True):
                    continue
                comp = effect_compare([r[s] for r in cand[p]], [r[s] for r in cand["default"]],
                                      [r[s] for r in ref["points"][p]["runs"]],
                                      [r[s] for r in ref["points"]["default"]["runs"]], alpha)
                a.tests.append(Test("E5", p, s, "effect", m, _decide(comp, m), comp.to_dict()))
    return a


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    accepted: bool = False
    stage: int = 0
    reason: str = ""
    label: str = ""                 # 'refactor' (bit-identical to parent) or 'mechanism'
    complexity: dict = field(default_factory=dict)
    delta_C: float = 0.0
    features: dict = field(default_factory=dict)
    feedback: dict = field(default_factory=dict)
    eq_distance: float = math.nan
    seeds_per_point: int = 0
    n_runs: int = 0
    identity_digests: list = field(default_factory=list)
    tests: list = field(default_factory=list)
    code_hash: str = ""
    judge_hash: str = ""
    design_hash: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["complexity"] = {k: v for k, v in self.complexity.items() if k != "dependency_manifest"}
        return d


def required_delta(parent_C: float, profile: Profile) -> float:
    c = profile.design["complexity"]
    return max(c["delta_abs_bytes"], c["delta_rel"] * parent_C)


def identity_digests(code: str, profile: Profile) -> list[str]:
    res = collect(code, profile, "candidate", "identity", ["default"], 3, digest=True)
    return [r["_digest"] for r in res["default"]]


def evaluate_candidate(code: str, profile: Profile, parent: dict | None = None,
                       ref: dict | None = None, margins: dict | None = None,
                       log=print) -> JudgeResult:
    """
    Run the full acceptance pipeline. `parent` may carry {"C": float, "digests": [...]}
    for the complexity ratchet and the refactor label.
    """
    t0 = time.time()
    if ref is None or margins is None:
        ref, margins, _ = load_reference(profile.name)
    r = JudgeResult(code_hash=code_hash(code), judge_hash=judge_hash(), design_hash=design_hash())

    def done(stage, reason="", accepted=False):
        r.stage, r.reason, r.accepted = stage, reason, accepted
        r.seconds = round(time.time() - t0, 1)
        return r

    # Stage 0: guard + complexity -------------------------------------------
    problems = guard.check(code)
    if problems:
        return done(0, "guard: " + "; ".join(problems[:5]))
    try:
        r.complexity = cx.measure(code, profile)
    except Exception as e:  # noqa: BLE001
        return done(0, f"complexity trace crashed: {type(e).__name__}: {e}"[:400])
    if parent is not None:
        r.delta_C = parent["C"] - r.complexity["C"]
        need = required_delta(parent["C"], profile)
        if r.delta_C < need:
            return done(0, f"not simpler: C {r.complexity['C']} vs parent {parent['C']} (need a drop of >= {need:.0f} bytes)")
    log(f"    stage 0 ok: C={r.complexity['C']} (delta {r.delta_C:+.0f})")

    # Stage 1: smoke + determinism + identity label ----------------------------
    try:
        smoke = collect(code, profile, "candidate", "smoke", ["default"], 1, digest=True)
        again = collect(code, profile, "candidate", "smoke", ["default"], 1, digest=True)
        r.identity_digests = identity_digests(code, profile)
    except RunError as e:
        return done(1, str(e))
    if smoke["default"][0]["_digest"] != again["default"][0]["_digest"]:
        return done(1, "non-deterministic: the same seed produced different trajectories")
    r.label = "refactor" if parent and parent.get("digests") == r.identity_digests else "mechanism"
    log(f"    stage 1 ok ({r.label})")

    # Stage 2: presence at the default point ------------------------------------
    key = r.code_hash
    n_pres = profile["n_presence"]
    try:
        cand = collect(code, profile, "candidate", key, ["default"], n_pres)
    except RunError as e:
        return done(2, str(e))
    a = assess(cand, ref, margins, ["default"], profile.alpha, features=FEATURES)
    pres = [t for t in a.tests if t.kind == "presence" and t.status != "pass"]
    if pres:
        r.feedback = Assessment(pres).feedback()
        r.features = {t.feature: "fail" for t in pres}
        return done(2, "presence: " + "; ".join(f"{t.feature} {t.note}" for t in pres))
    log("    stage 2 ok: E1-E4 present")

    # Stages 3-4: sequential equivalence over the screening design --------------
    points = profile.screening
    schedule = profile["candidate_schedule"]
    alpha_look = profile.alpha / len(schedule)
    have = {p: (len(cand[p]) if p in cand else 0) for p in points}
    for p in points:
        cand.setdefault(p, [])
    for look, n in enumerate(schedule, 1):
        try:
            for p in points:
                if have[p] < n:
                    more = collect(code, profile, "candidate", key, [p], n - have[p], start=have[p])
                    cand[p].extend(more[p])
                    have[p] = n
        except RunError as e:
            return done(3, str(e))
        a = assess(cand, ref, margins, points, alpha_look)
        r.features = a.feature_status()
        r.seeds_per_point = n
        log(f"    look {look}/{len(schedule)} (n={n}/point): " + " ".join(f"{k}={v}" for k, v in r.features.items()))
        if a.all_pass:
            r.eq_distance = a.eq_distance()
            r.tests = [asdict(t) for t in a.tests]
            r.n_runs = sum(have.values())
            return done(4, "", accepted=True)
        if a.any_clear_fail or look == len(schedule):
            break
    r.feedback = a.feedback()
    r.tests = [asdict(t) for t in a.tests]
    r.n_runs = sum(have.values())
    failing = [k for k, v in r.features.items() if v != "pass"]
    stage = 4 if failing == ["E5"] else 3
    verdict = "differs" if a.any_clear_fail else "equivalence not demonstrated"
    return done(stage, f"{verdict}: " + ", ".join(f"{f} ({FEATURE_NAMES[f]})" for f in failing))


def audit(code: str, profile: Profile, ref: dict | None = None, margins: dict | None = None) -> dict:
    """Full-design audit on holdout seeds (single look)."""
    if ref is None or margins is None:
        ref, margins, _ = load_reference(profile.name)
    points = profile.audit_points
    try:
        cand = collect(code, profile, "holdout", code_hash(code), points, profile["n_holdout"])
    except RunError as e:
        return {"passed": False, "reason": str(e), "features": {}}
    a = assess(cand, ref, margins, points, profile.alpha)
    return {"passed": a.all_pass, "features": a.feature_status(), "feedback": a.feedback(),
            "eq_distance": a.eq_distance(), "points": points, "n_per_point": profile["n_holdout"]}
