"""
Response-surface equivalence over a random sample of the input domain.

A candidate and canonical Sugarscape are run at the same sampled points, each in
both trade worlds. For every statistic the judge then tests, against margins
fixed before any candidate is seen:

  l2        root-mean-square discrepancy of the statistic's mean over the domain.
            E_x[(mu_cand(x) - mu_ref(x))^2] is estimated without the upward bias
            from seed noise, with a bootstrap-over-points confidence bound. It is
            the omnibus "distance between the two feature maps".
  slope     the response to each input: the per-point discrepancy is regressed on
            the input regressors and every coefficient must be equivalent to 0.
            Every trait response must match, including the *absence* of a
            response where Sugarscape has none (the regression-based equivalence
            test of Robinson, Duursma & Marshall 2005, applied to response
            surfaces; relational equivalence in the sense of Axtell et al. 1996).
  contrast  the trade on/off effect (difference of differences, paired by point).
  spread    the ratio of seed-to-seed variances, pooled over the domain.
  defined   where the statistic exists at all (agents survive, trade persists).
  named     literature-backed responses (Epstein & Axtell's figures) must show the
            right sign in the candidate itself.

Equivalence decisions are TOST-style: a test passes only when its confidence
interval lies inside the margin, so "no difference detected" never counts.
Requiring every test is an intersection-union test (Berger 1982), so the overall
error rate is alpha without a multiplicity correction.

Discrepancies are measured in units of canonical's typical seed-to-seed SD of the
statistic over the domain (`scales`), or in probability units for indicators.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats as sps

from judge.features import (FEATURE_NAMES, FEATURES, SCALAR_STATS, TRADE_ONLY, VECTOR_STATS,
                            stat_feature, world_stats)

REL_SD_FLOOR = 0.01
N_BOOT = 1000


def key(pid: str, world: str) -> str:
    return f"{pid}|{world}"


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def _vals(runs: list[dict], stat: str, comp: int | None = None) -> np.ndarray:
    if comp is None:
        a = np.array([r.get(stat, math.nan) for r in runs], dtype=float)
    else:
        a = np.array([(r.get(stat) or [])[comp] if len(r.get(stat) or []) > comp else math.nan for r in runs],
                     dtype=float)
    return a


def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def n_components(panel: dict, stat: str) -> int:
    for runs in panel.values():
        for r in runs:
            v = r.get(stat)
            if isinstance(v, list):
                return len(v)
    return 0


def scales(panel: dict, pids: list[str], worlds: list[str]) -> dict:
    """Canonical's root-mean within-point SD of every statistic, per world (indicators: 1)."""
    out = {}
    for w in worlds:
        out[w] = {}
        for f, items in world_stats(w == "trade").items():
            for s, kind in items:
                if kind == "indicator":
                    out[w][s] = 1.0
                elif kind == "scalar":
                    out[w][s] = _pooled_sd([_vals(panel.get(key(p, w), []), s) for p in pids])
                else:
                    out[w][s] = [_pooled_sd([_vals(panel.get(key(p, w), []), s, c) for p in pids])
                                 for c in range(n_components(panel, s))]
    return out


def _pooled_sd(groups: list[np.ndarray]) -> float:
    vs, ms = [], []
    for g in groups:
        g = _finite(g)
        if len(g) >= 2:
            vs.append(g.var(ddof=1))
        if len(g):
            ms.append(g.mean())
    if not vs:
        return math.nan
    sd = math.sqrt(float(np.mean(vs)))
    return float(max(sd, REL_SD_FLOOR * abs(float(np.mean(ms))) if ms else 0.0, 1e-9))


# ---------------------------------------------------------------------------
# Primitive estimators
# ---------------------------------------------------------------------------

def _boot_mean(t: np.ndarray, alpha: float, rng) -> tuple[float, float]:
    idx = rng.integers(0, len(t), (N_BOOT, len(t)))
    m = t[idx].mean(axis=1)
    return float(np.quantile(m, alpha)), float(np.quantile(m, 1 - alpha))


def _l2_terms(cand: dict, ref: dict, pids, w, stat, scale, comps=None, indicator_of=None):
    """Per-point noise-corrected squared standardized differences."""
    terms, missing, defined_ref = [], 0, 0
    for p in pids:
        ck, rc = cand.get(key(p, w), []), ref.get(key(p, w), [])
        if not ck or not rc:
            continue
        per = []
        for c, sc in (comps or [(None, scale)]):
            if indicator_of is not None:
                xk = np.isfinite(_vals(ck, indicator_of)).astype(float)
                xc = np.isfinite(_vals(rc, indicator_of)).astype(float)
            else:
                xk, xc = _finite(_vals(ck, stat, c)), _finite(_vals(rc, stat, c))
            if c is None and indicator_of is None:
                if len(xc) >= 2:
                    defined_ref += 1
                    if len(xk) == 0:
                        missing += 1
            if len(xk) < 2 or len(xc) < 2 or not (sc and math.isfinite(sc)):
                continue
            d = (xk.mean() - xc.mean()) / sc
            per.append(d * d - xk.var(ddof=1) / len(xk) / sc ** 2 - xc.var(ddof=1) / len(xc) / sc ** 2)
        if per:
            terms.append(float(np.mean(per)))
    return np.array(terms), missing, defined_ref


def _ols_hc3(y: np.ndarray, X: np.ndarray):
    """OLS with HC3 heteroscedasticity-robust standard errors."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    wts = e ** 2 / np.maximum(1 - h, 1e-6) ** 2
    cov = XtX_inv @ (X.T * wts) @ X @ XtX_inv
    return b, np.sqrt(np.maximum(np.diag(cov), 0.0)), n - k


def _means(runs: list[dict], stat: str) -> float:
    x = _finite(_vals(runs, stat))
    return float(x.mean()) if len(x) else math.nan


def response(panel: dict, pids, Z: dict, w: str, stat: str, scale: float):
    """A model's own response coefficients (per full input range, in scale units) and SEs."""
    y, X = [], []
    for p in pids:
        m = _means(panel.get(key(p, w), []), stat)
        if math.isfinite(m):
            y.append(m / scale)
            X.append((1.0,) + tuple(Z[p]))
    if len(y) < len(Z[pids[0]]) + 6:
        return None
    b, se, df = _ols_hc3(np.array(y), np.array(X))
    return {"b": b[1:].tolist(), "se": se[1:].tolist(), "df": df, "level": float(b[0]), "level_se": float(se[0])}


def own_contrast(panel: dict, pids, stat: str, scale: float):
    g = []
    for p in pids:
        on, off = _means(panel.get(key(p, "trade"), []), stat), _means(panel.get(key(p, "notrade"), []), stat)
        if math.isfinite(on) and math.isfinite(off):
            g.append((on - off) / scale)
    if len(g) < 5:
        return None
    g = np.array(g)
    return {"mean": float(g.mean()), "se": float(g.std(ddof=1) / math.sqrt(len(g))), "df": len(g) - 1}


def own_level(panel: dict, pids, w: str, stat: str):
    m = np.array([_means(panel.get(key(p, w), []), stat) for p in pids])
    m = m[np.isfinite(m)]
    if len(m) < 5:
        return None
    return {"mean": float(m.mean()), "se": float(m.std(ddof=1) / math.sqrt(len(m))), "df": len(m) - 1}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@dataclass
class STest:
    feature: str
    world: str
    stat: str
    kind: str            # l2 | defined | spread | slope | contrast | named
    term: str
    margin: float
    status: str          # pass | fail | undecided
    est: float = math.nan
    lo: float = math.nan
    hi: float = math.nan
    n: int = 0
    note: str = ""

    def ratio(self) -> float:
        """Upper confidence bound of the distance relative to the margin (< 1 means inside)."""
        if self.kind == "named" or not (self.margin > 0):
            return 0.0
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
            return math.inf
        if self.kind in ("l2", "defined"):
            return self.hi / self.margin
        return max(abs(self.lo), abs(self.hi)) / self.margin

    def se(self, alpha: float) -> float:
        z = sps.norm.ppf(1 - alpha)
        return (self.hi - self.lo) / (2 * z) if math.isfinite(self.hi - self.lo) else math.inf

    def p_exceeds(self, alpha: float) -> float:
        """One-sided p-value for 'the true distance exceeds the margin' (knockout certificates)."""
        if self.kind == "named":
            return 1.0
        if self.status == "fail" and not math.isfinite(self.est):
            return 0.0
        se = self.se(alpha)
        if not (math.isfinite(self.est) and se > 0 and math.isfinite(se)):
            return 1.0
        d = self.est if self.kind in ("l2", "defined") else abs(self.est)
        return float(sps.norm.sf((d - self.margin) / se))

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in asdict(self).items()}


def _rms(x: float) -> float:
    return math.copysign(math.sqrt(abs(x)), x)


def _two_sided(est, lo, hi, m) -> str:
    if -m < lo and hi < m:
        return "pass"
    if lo > m or hi < -m:
        return "fail"
    return "undecided"


def _one_sided(lo, hi, m) -> str:
    if hi < m:
        return "pass"
    if lo > m:
        return "fail"
    return "undecided"


def l2_test(cand, ref, pids, w, stat, kind, scale, margin, alpha, rng, missing_max) -> STest:
    f = stat_feature(stat)
    if kind == "vector":
        comps = [(c, s) for c, s in enumerate(scale) if s and math.isfinite(s)]
        terms, missing, dref = _l2_terms(cand, ref, pids, w, stat, None, comps=comps)
    else:
        terms, missing, dref = _l2_terms(cand, ref, pids, w, stat, scale)
    if kind == "scalar" and dref and missing / dref > missing_max:
        return STest(f, w, stat, "l2", "", margin, "fail", n=len(terms),
                     note="undefined in the candidate where Sugarscape defines it")
    if len(terms) < 3:
        if dref == 0:
            return STest(f, w, stat, "l2", "", margin, "pass", 0.0, 0.0, 0.0, 0, "not defined in this world")
        return STest(f, w, stat, "l2", "", margin, "undecided", n=len(terms), note="too few comparable points")
    est = float(terms.mean())
    lo, hi = _boot_mean(terms, alpha, rng)
    st = _one_sided(lo, hi, margin ** 2)
    return STest(f, w, stat, "l2", "", margin, st, _rms(est), _rms(lo), _rms(hi), len(terms))


def defined_test(cand, ref, pids, w, stat, margin, alpha, rng) -> STest:
    terms, _, _ = _l2_terms(cand, ref, pids, w, stat, 1.0, indicator_of=stat)
    f = stat_feature(stat)
    if len(terms) < 3:
        return STest(f, w, stat, "defined", "", margin, "undecided", n=len(terms))
    if np.all(terms == 0):
        return STest(f, w, stat, "defined", "", margin, "pass", 0.0, 0.0, 0.0, len(terms))
    est = float(terms.mean())
    lo, hi = _boot_mean(terms, alpha, rng)
    return STest(f, w, stat, "defined", "", margin, _one_sided(lo, hi, margin ** 2), _rms(est), _rms(lo), _rms(hi),
                 len(terms))


def spread_test(cand, ref, pids, w, stat, margin_ratio, alpha, rng) -> STest | None:
    rows = []
    for p in pids:
        xk = _finite(_vals(cand.get(key(p, w), []), stat))
        xc = _finite(_vals(ref.get(key(p, w), []), stat))
        if len(xk) >= 2 and len(xc) >= 2:
            rows.append(((len(xk) - 1) * xk.var(ddof=1), len(xk) - 1, (len(xc) - 1) * xc.var(ddof=1), len(xc) - 1))
    if len(rows) < 5:
        return None
    R = np.array(rows)
    if R[:, 2].sum() <= 1e-12:
        return None

    def logratio(A):
        num, den = A[:, 0].sum() / A[:, 1].sum(), A[:, 2].sum() / A[:, 3].sum()
        return math.log(max(num, 1e-300) / max(den, 1e-300))

    est = logratio(R)
    idx = rng.integers(0, len(R), (N_BOOT, len(R)))
    boots = np.array([logratio(R[i]) for i in idx])
    lo, hi = float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))
    m = math.log(margin_ratio)
    return STest(stat_feature(stat), w, stat, "spread", "", m, _two_sided(est, lo, hi, m), est, lo, hi, len(R))


def slope_tests(cand, ref, pids, Z, w, stat, scale, margins: dict, terms: list[str], alpha) -> list[STest]:
    y, X = [], []
    for p in pids:
        mk, mc = _means(cand.get(key(p, w), []), stat), _means(ref.get(key(p, w), []), stat)
        if math.isfinite(mk) and math.isfinite(mc):
            y.append((mk - mc) / scale)
            X.append((1.0,) + tuple(Z[p]))
    if len(y) < len(terms) + 6:
        return [STest("E5", w, stat, "slope", t, margins[t], "undecided", n=len(y), note="too few comparable points")
                for t in terms]
    b, se, df = _ols_hc3(np.array(y), np.array(X))
    tq = sps.t.ppf(1 - alpha, df)
    out = []
    for j, t in enumerate(terms, start=1):
        lo, hi = b[j] - tq * se[j], b[j] + tq * se[j]
        out.append(STest("E5", w, stat, "slope", t, margins[t], _two_sided(b[j], lo, hi, margins[t]),
                         float(b[j]), float(lo), float(hi), len(y)))
    return out


def contrast_test(cand, ref, pids, stat, scale, margin, alpha) -> STest | None:
    g = []
    for p in pids:
        vals = [_means(src.get(key(p, w), []), stat) for src in (cand, ref) for w in ("trade", "notrade")]
        if all(math.isfinite(v) for v in vals):
            g.append(((vals[0] - vals[1]) - (vals[2] - vals[3])) / scale)
    if len(g) < 5:
        return None
    g = np.array(g)
    est, se = float(g.mean()), float(g.std(ddof=1) / math.sqrt(len(g)))
    tq = sps.t.ppf(1 - alpha, len(g) - 1)
    lo, hi = est - tq * se, est + tq * se
    return STest("E5", "both", stat, "contrast", "trade", margin, _two_sided(est, lo, hi, margin), est, lo, hi, len(g))


def named_test(nr: dict, cand, pids, Z, terms, scales_: dict, alpha) -> STest:
    """Does the candidate itself show the literature response, with the right sign?"""
    stat, sign, term, w = nr["stat"], nr["sign"], nr["term"], nr["world"]
    if term == "trade":
        sc = math.sqrt((scales_["trade"][stat] ** 2 + scales_["notrade"][stat] ** 2) / 2)
        r = own_contrast(cand, pids, stat, sc)
        est, se, df = (r["mean"], r["se"], r["df"]) if r else (math.nan, math.inf, 1)
    elif term == "level":
        r = own_level(cand, pids, w, stat)
        est, se, df = (r["mean"], r["se"], r["df"]) if r else (math.nan, math.inf, 1)
    else:
        r = response(cand, pids, Z, w, stat, scales_[w][stat])
        j = terms.index(term)
        est, se, df = (r["b"][j], r["se"][j], r["df"]) if r else (math.nan, math.inf, 1)
    tq = sps.t.ppf(1 - alpha, max(df, 1))
    lo, hi = sign * est - tq * se, sign * est + tq * se
    if not math.isfinite(est):
        status = "undecided"
    elif lo > 0:
        status = "pass"
    elif hi < 0:
        status = "fail"
    else:
        status = "undecided"
    return STest("E5", w, stat, "named", nr["id"], 0.0, status, sign * est if math.isfinite(est) else math.nan,
                 lo, hi, note=nr["claim"])


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

TERM_LABELS = {
    "log_initial_population": "initial population",
    "log_regrowth_rate": "the resource regrowth rate",
    "vision_mean": "mean vision",
    "vision_spread": "the spread of vision",
    "metabolism_mean": "mean metabolism",
    "metabolism_spread": "the spread of metabolism",
    "endowment_mean": "mean initial endowment",
    "endowment_spread": "the spread of initial endowments",
}


@dataclass
class Assessment:
    tests: list[STest] = field(default_factory=list)
    canon: dict = field(default_factory=dict)        # canonical responses, for feedback wording

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
        r = [t.ratio() for t in self.tests if t.kind != "named"]
        return float(max(r)) if r else 0.0

    def _direction(self, t: STest) -> str:
        if t.kind == "l2":
            return "differs from Sugarscape across the input domain"
        if t.kind == "defined":
            return "exists in different parts of the input domain than in Sugarscape (survival or trade persistence differs)"
        if t.kind == "spread":
            return "varies " + ("more" if t.est > 0 else "less") + " from seed to seed than in Sugarscape"
        if t.kind == "named":
            return f"does not reproduce '{t.note}' ({t.term})"
        canon = (self.canon.get("contrast", {}).get(t.stat) if t.kind == "contrast"
                 else self.canon.get("slope", {}).get(t.world, {}).get(t.stat, {}).get(t.term))
        cb = canon[0] if canon else 0.0
        what = ("the effect of trade on " + t.stat) if t.kind == "contrast" else \
            f"the response of {t.stat} to {TERM_LABELS.get(t.term, t.term)}"
        if abs(cb) < 1.0:
            return f"{what} appears where Sugarscape has almost none"
        stronger = (t.est > 0) == (cb > 0)
        return f"{what} is too {'strong' if stronger else 'weak or reversed'}"

    def feedback(self) -> dict[str, list[str]]:
        """Qualitative per-feature feedback (no numbers, no margins) for the proposer."""
        fb: dict[str, list[str]] = {}
        for t in self.tests:
            if t.status == "pass":
                continue
            where = "" if t.world == "both" else f" ({'with' if t.world == 'trade' else 'without'} trade)"
            verdict = "" if t.status == "fail" else "not shown equivalent: "
            msg = t.note if t.note and t.kind == "l2" else self._direction(t)
            fb.setdefault(f"{t.feature} {FEATURE_NAMES[t.feature]}", []).append(f"{verdict}{t.stat}: {msg}{where}")
        return fb


def assess(cand: dict, ref: dict, pids: list[str], Z: dict, margins: dict, alpha: float, profile,
           features: list[str] | None = None, seed: int = 0) -> Assessment:
    """Run the full battery for one candidate panel against the canonical panel at points `pids`."""
    feats = set(features or (FEATURES + ["E5"]))
    rng = np.random.default_rng(seed)
    floors = profile.margin_floor
    terms = margins["terms"]
    a = Assessment(canon=margins.get("canon", {}))
    active = margins.get("active", {})
    worlds = [w for w in ("trade", "notrade") if w in margins["scales"]]
    for w in worlds:
        for f, items in world_stats(w == "trade").items():
            if f not in feats and "E5" not in feats:
                continue
            for s, kind in items:
                sc = margins["scales"][w].get(s)
                if sc is None or (kind != "vector" and not math.isfinite(sc)):
                    continue
                if f in feats and active.get(key(s, w), True):
                    m = margins["l2"][w][s]
                    a.tests.append(l2_test(cand, ref, pids, w, s, kind, sc, m, alpha, rng, floors["missing_max"]))
                    if kind == "scalar":
                        a.tests.append(defined_test(cand, ref, pids, w, s, margins["defined"][w][s], alpha, rng))
                        sp = spread_test(cand, ref, pids, w, s, margins["spread"][w][s], alpha, rng)
                        if sp is not None:
                            a.tests.append(sp)
                if "E5" in feats and kind == "scalar":
                    a.tests += slope_tests(cand, ref, pids, Z, w, s, sc, margins["slope"][w][s], terms, alpha)
    if "E5" in feats and len(worlds) == 2:
        for s, m in margins["contrast"].items():
            sc = math.sqrt((margins["scales"]["trade"][s] ** 2 + margins["scales"]["notrade"][s] ** 2) / 2)
            if not math.isfinite(sc):
                continue
            ct = contrast_test(cand, ref, pids, s, sc, m, alpha)
            if ct is not None:
                a.tests.append(ct)
        for nr in margins.get("named", []):
            if nr.get("confirmed"):
                a.tests.append(named_test(nr, cand, pids, Z, terms, margins["scales"], alpha))
    return a


def contrast_stats() -> list[str]:
    return [s for f in FEATURES for s in SCALAR_STATS[f] if s not in TRADE_ONLY]


def vector_stats() -> list[str]:
    return [s for f in FEATURES for s in VECTOR_STATS[f]]
