"""
Acceptance pipeline.

Stage 0  static guard + traced canonical complexity (must drop by >= delta)
Stage 1  smoke + determinism (judge-owned loop, contract validation); identity
         digests label pure refactors (label only, never a reason to accept)
Stage 2  presence of E1-E4 at the default point on fresh seeds (cheap gate)
Stage 3  E1-E4 response-surface equivalence over a fresh random sample of the
         input domain, in both trade worlds, in sequential looks of growing size
Stage 4  E5 trait responses on the same sample: the effect of every input, the
         trade contrast and the literature-backed responses
Audit    a fresh sample of the i.i.d. holdout bank on holdout seeds, Markov
         coverage bounds, and adversarial re-testing of the worst points

Every comparison is against canonical Sugarscape (judge/reference), never the parent.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict, dataclass, field

import numpy as np

from judge import JUDGE_DIR, judge_hash
from judge import complexity as cx
from judge import guard
from judge import surface
from judge.contract import CACHE_DIR, code_hash, materialize
from judge.design import Profile, design_hash, seeds
from judge.domain import Point, subset_order
from judge.features import FEATURE_NAMES, FEATURES, SCALAR_STATS, TRADE_ONLY, presence
from judge.parallel import _run, _workers, job, pool, run_jobs, shutdown
from judge.stats import scalar_compare

REFERENCE_ROOT = JUDGE_DIR / "reference"


# ---------------------------------------------------------------------------
# Running groups of seeds
# ---------------------------------------------------------------------------

class RunError(Exception):
    pass


CACHE_CHUNK = 256       # results are committed to the run cache in batches of CACHE_CHUNK // 4
_MEASURE_HASH = None


def _measure_hash() -> str:
    """Hash of the code that turns a run into statistics (cached runs are keyed by it)."""
    global _MEASURE_HASH
    if _MEASURE_HASH is None:
        h = hashlib.sha256()
        for f in ("features.py", "contract.py", "parallel.py"):
            h.update((JUDGE_DIR / f).read_bytes())
        _MEASURE_HASH = h.hexdigest()[:16]
    return _MEASURE_HASH


def _cache_db():
    CACHE_DIR.mkdir(exist_ok=True)
    db = sqlite3.connect(CACHE_DIR / "runs.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS runs (k TEXT PRIMARY KEY, stats TEXT)")
    return db


def _run_key(code_h: str, params: dict, seed: int, steps: int) -> str:
    blob = json.dumps([code_h, _measure_hash(), steps, sorted(params.items()), seed], default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _cache_put(items: list[tuple[str, dict]]) -> None:
    fresh = [(k, json.dumps(r["stats"])) for k, r in items if r["ok"]]
    if fresh:
        with closing(_cache_db()) as db, db:
            db.executemany("INSERT OR REPLACE INTO runs VALUES (?, ?)", fresh)


def _indexed_run(item):
    i, j = item
    return i, _run(j)


def _cost(j) -> float:
    """Rough relative run time (agents x vision neighbourhood), to submit long runs first."""
    p = dict(j.params)
    return p.get("initial_population", 200) * (1 + p.get("vision_max", 5)) ** 2


def _stream(jobs: dict, keys: list[str]) -> dict:
    """Run jobs unordered on the pool, longest first, committing results to the cache as they arrive."""
    order = sorted(jobs, key=lambda i: -_cost(jobs[i]))
    results, pending = {}, []
    it = pool().imap_unordered(_indexed_run, [(i, jobs[i]) for i in order], chunksize=1)
    for _ in order:
        try:
            i, r = it.next(timeout=1200)
        except mp.TimeoutError:
            shutdown()
            raise RunError("a run produced no result within 1200 s")
        results[i] = r
        pending.append((keys[i], r))
        if len(pending) >= CACHE_CHUNK // 4:
            _cache_put(pending)
            pending = []
    _cache_put(pending)
    return results


def _run_all(code: str, profile: Profile, specs: list[tuple[str, dict, int]], digest: bool = False) -> list[dict]:
    """
    specs: (label, params, seed). Raises RunError on any failed run.

    Successful runs are cached by (model code, measurement code, inputs, seed, steps),
    so re-analysing a calibration never re-simulates. Digest runs (determinism and
    identity checks) always execute.
    """
    path = materialize(code)
    ch = code_hash(code)
    keys = [_run_key(ch, params, s, profile.steps) for _, params, s in specs]
    cached = {}
    if not digest:
        with closing(_cache_db()) as db:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = f"SELECT k, stats FROM runs WHERE k IN ({','.join('?' * len(chunk))})"
                cached.update({k: json.loads(v) for k, v in db.execute(q, chunk)})
    todo = [i for i, k in enumerate(keys) if k not in cached]
    jobs = {i: job(path, specs[i][0], specs[i][1], specs[i][2], profile.steps, digest) for i in todo}
    if digest or len(todo) <= 1 or _workers() == 1:
        results = dict(zip(todo, run_jobs([jobs[i] for i in todo], timeout=max(900, 60 * len(todo)))))
        if not digest:
            _cache_put([(keys[i], r) for i, r in results.items()])
    else:
        results = _stream(jobs, keys)
    bad = next(((i, results[i]) for i in todo if not results[i]["ok"]), None)
    if bad:
        label, _, s = specs[bad[0]]
        raise RunError(f"run failed at {label} (seed {s}): {bad[1]['error']}")
    out = []
    for i, ((label, _, s), k) in enumerate(zip(specs, keys)):
        if i in results:
            stats = results[i]["stats"]
            if digest:
                stats["_digest"] = results[i]["digest"]
        else:
            stats = cached[k]
        stats["_seed"] = s
        out.append(stats)
    return out


def collect(code: str, profile: Profile, pool: str, key: str, points: list[str], n: int,
            start: int = 0, digest: bool = False) -> dict[str, list[dict]]:
    """Run `n` seeds (from `start`) at each *named* point."""
    specs = [(p, profile.params(p), s) for p in points for s in seeds(pool, key, p, n, start)]
    res = _run_all(code, profile, specs, digest)
    out: dict[str, list[dict]] = {p: [] for p in points}
    for (p, _, _), r in zip(specs, res):
        out[p].append(r)
    return out


def collect_points(code: str, profile: Profile, pool: str, key: str, points: list[Point], n: int,
                   start: int = 0) -> dict[str, list[dict]]:
    """Run `n` seeds at each sampled domain point in every world -> {'pid|world': runs}."""
    specs, keys = [], []
    for pt in points:
        for w, wp in profile.worlds.items():
            k = surface.key(pt.id, w)
            for s in seeds(pool, key, k, n, start):
                specs.append((k, {**pt.params, **wp}, s))
                keys.append(k)
    res = _run_all(code, profile, specs)
    out: dict[str, list[dict]] = {}
    for k, r in zip(keys, res):
        out.setdefault(k, []).append(r)
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


def bank_points(ref: dict, which: str = "bank") -> list[Point]:
    return [Point(p["id"], p["params"], tuple(p["z"])) for p in ref[which]["points"]]


# ---------------------------------------------------------------------------
# Presence gate (stage 2)
# ---------------------------------------------------------------------------

@dataclass
class Test:
    feature: str
    point: str
    stat: str
    kind: str
    margin: float
    status: str
    comparison: dict | None = None
    note: str = ""


def presence_tests(cand_runs: list[dict], ref: dict, alpha: float) -> list[Test]:
    out = []
    for f in FEATURES:
        if ref["anchor"]["presence"].get(f):
            ok, why = presence(f, cand_runs, alpha)
            out.append(Test(f, "default", "presence", "presence", 0.0, "pass" if ok else "fail", None, why))
    return out


def screen(cand: dict, ref: dict, margins: dict, pts: list[Point], alpha: float, profile: Profile,
           features=None, which: str = "bank", seed: int = 0) -> surface.Assessment:
    Z = {p.id: p.z for p in pts}
    return surface.assess(cand, ref[which]["runs"], [p.id for p in pts], Z, margins, alpha, profile,
                          features=features, seed=seed)


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
    points: int = 0                 # domain points per world in the deciding look
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
    try:
        pres_runs = collect(code, profile, "candidate", key, ["default"], profile["n_presence"])["default"]
    except RunError as e:
        return done(2, str(e))
    pres = [t for t in presence_tests(pres_runs, ref, profile.alpha) if t.status != "pass"]
    if pres:
        r.feedback = {f"{t.feature} {FEATURE_NAMES[t.feature]}": [f"{t.note} at the default inputs"] for t in pres}
        r.features = {t.feature: "fail" for t in pres}
        return done(2, "presence: " + "; ".join(f"{t.feature} {t.note}" for t in pres))
    log("    stage 2 ok: E1-E4 present at the default inputs")

    # Stages 3-4: sequential looks over a fresh random sample of the domain -----
    pts_all = bank_points(ref)
    order = subset_order(len(pts_all), "screen:" + key)
    schedule = [min(n, len(pts_all)) for n in profile["look_points"]]
    look_alpha = profile["look_alpha"]       # alpha spending: sums to alpha; early looks mostly reject
    n_k = profile["candidate_seeds"]
    cand: dict[str, list[dict]] = {}
    have = 0
    a = None
    for look, n_pts in enumerate(schedule, 1):
        new = [pts_all[i] for i in order[have:n_pts]]
        try:
            cand.update(collect_points(code, profile, "candidate", key, new, n_k))
        except RunError as e:
            return done(3, str(e))
        have = n_pts
        pts = [pts_all[i] for i in order[:n_pts]]
        a = screen(cand, ref, margins, pts, look_alpha[look - 1], profile, seed=look)
        r.features = a.feature_status()
        r.points, r.seeds_per_point = n_pts, n_k
        r.n_runs = sum(len(v) for v in cand.values())
        log(f"    look {look}/{len(schedule)} ({n_pts} points x {len(profile.worlds)} worlds x {n_k} seeds): "
            + " ".join(f"{k}={v}" for k, v in r.features.items()))
        if a.all_pass:
            r.eq_distance = a.eq_distance()
            r.tests = [t.to_dict() for t in a.tests]
            return done(4, "", accepted=True)
        if a.any_clear_fail:
            break
    r.feedback = a.feedback()
    r.tests = [t.to_dict() for t in a.tests if t.status != "pass"]
    r.eq_distance = a.eq_distance()
    failing = [k for k, v in r.features.items() if v != "pass"]
    stage = 4 if failing == ["E5"] else 3
    verdict = "differs" if a.any_clear_fail else "equivalence not demonstrated"
    return done(stage, f"{verdict}: " + ", ".join(f"{f} ({FEATURE_NAMES[f]})" for f in failing))


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def coverage_bounds(a: surface.Assessment, c: float) -> dict:
    """
    Markov bound from the L2 tests: with E_x[delta^2] <= U (the upper confidence
    bound), the fraction of the domain where |delta| > c is at most U / c^2.
    Summed over statistics, it bounds the fraction of the domain where *any*
    statistic departs by more than c (union bound).
    """
    scalars = {s for f in FEATURES for s in SCALAR_STATS[f]}
    per = {}
    for t in a.tests:
        if t.kind == "l2" and t.stat in scalars and t.status == "pass" and math.isfinite(t.hi):
            per[f"{t.stat}|{t.world}"] = min(1.0, max(t.hi, 0.0) ** 2 / c ** 2)
    return {"per_statistic": per, "worst": max(per.values(), default=1.0),
            "any_statistic_union": min(1.0, sum(per.values()))}


def tested_scalars(profile: Profile, world: str) -> list[str]:
    diagnostic = set(profile.design.get("diagnostic_stats", []))
    return [s for f in FEATURES for s in SCALAR_STATS[f]
            if s not in diagnostic and not (world == "notrade" and s in TRADE_ONLY)]


def worst_points(cand: dict, ref_runs: dict, pts: list[Point], margins: dict, k: int,
                 profile: Profile) -> list[tuple[float, Point, str]]:
    scored = []
    for pt in pts:
        for w in margins["scales"]:
            best = 0.0
            for s in tested_scalars(profile, w):
                sc = margins["scales"][w].get(s)
                mk = surface._means(cand.get(surface.key(pt.id, w), []), s)
                mc = surface._means(ref_runs.get(surface.key(pt.id, w), []), s)
                if sc and math.isfinite(mk) and math.isfinite(mc):
                    best = max(best, abs(mk - mc) / sc)
            scored.append((best, pt, w))
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


def audit(code: str, profile: Profile, ref: dict | None = None, margins: dict | None = None) -> dict:
    """Holdout audit: fresh i.i.d. points, holdout seeds, single look, then adversarial refinement."""
    if ref is None or margins is None:
        ref, margins, _ = load_reference(profile.name)
    h = code_hash(code)
    pts_all = bank_points(ref, "holdout")
    order = subset_order(len(pts_all), "audit:" + h)
    pts = [pts_all[i] for i in order[:min(profile["audit_points"], len(pts_all))]]
    try:
        cand = collect_points(code, profile, "holdout", h, pts, profile["candidate_seeds"])
    except RunError as e:
        return {"passed": False, "reason": str(e), "features": {}}
    a = screen(cand, ref, margins, pts, profile.alpha, profile, which="holdout", seed=99)
    floor = profile.margin_floor

    # Adversarial refinement: re-test the most discrepant points with more seeds.
    located = []
    worst = worst_points(cand, ref["holdout"]["runs"], pts, margins, profile["refine_points"], profile)
    n_ref = profile["refine_seeds"]
    try:
        for _, pt, w in worst:
            label = surface.key(pt.id, w)
            params = {**pt.params, **profile.worlds[w]}
            rr = _run_all(canonical_code_text(), profile,
                          [(label, params, s) for s in seeds("reference", "refine", label, n_ref)])
            cr = _run_all(code, profile, [(label, params, s) for s in seeds("holdout", "refine:" + h, label, n_ref)])
            tested = tested_scalars(profile, w)
            for s in tested:
                comp = scalar_compare([x[s] for x in cr], [x[s] for x in rr], profile.alpha / len(tested))
                if comp.clearly_different(floor["point_sigma"]):
                    located.append({"point": pt.id, "world": w, "inputs": pt.params, "stat": s,
                                    "direction": comp.direction()})
    except RunError as e:
        return {"passed": False, "reason": str(e), "features": a.feature_status()}
    cov = coverage_bounds(a, floor["point_sigma"])
    passed = a.all_pass and not located
    return {"passed": passed, "features": a.feature_status(), "feedback": a.feedback(),
            "eq_distance": a.eq_distance(), "points": len(pts), "seeds_per_point": profile["candidate_seeds"],
            "located_failures": located, "coverage": cov,
            "reason": "" if passed else ("located failures: " + "; ".join(
                f"{x['stat']} {x['direction']} at {x['inputs']} ({x['world']})" for x in located[:3])
                if located else "not all features equivalent on the holdout sample")}


def canonical_code_text() -> str:
    from judge.controls import canonical_code
    return canonical_code()
