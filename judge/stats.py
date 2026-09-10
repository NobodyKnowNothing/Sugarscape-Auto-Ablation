"""
Equivalence statistics.

All comparisons are expressed in units of the reference (canonical) between-seed
standard deviation, so margins mean the same thing for every statistic.

Decisions:
  scalar   TOST: the (1-2a) confidence interval of the standardized difference d
           must lie inside (-margin, +margin).
  vector   sup-distance D = max_k |mean_cand_k - mean_ref_k| / sd_ref_k with a
           bootstrap upper confidence bound that must be below the margin.
  effect   TOST on the difference between the candidate's and the reference's
           response to an input change (difference of differences).

Requiring *every* test to pass is an intersection-union test (Berger 1982): the
overall error rate is controlled at alpha without a multiplicity correction.
Claims that need only *one* test to succeed (knockout certificates) use Holm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats as sps

UNDEFINED_SLACK = 0.2   # extra fraction of NaN runs tolerated relative to the reference
REL_SD_FLOOR = 0.01     # a reference SD below 1% of |mean| (e.g. N(0)/N0 == 1 exactly) is floored


def _scale(sd, mean):
    """Standardization scale with a floor so zero-variance reference values stay testable."""
    return np.maximum(sd, np.maximum(REL_SD_FLOOR * np.abs(mean), 1e-9))


@dataclass
class Comparison:
    d: float              # standardized difference (or sup-distance for vectors)
    lo: float             # lower confidence bound
    hi: float             # upper confidence bound
    se: float = math.nan
    df: float = math.nan
    n_cand: int = 0
    n_ref: int = 0
    kind: str = "scalar"

    @property
    def ucb_abs(self) -> float:
        return max(abs(self.lo), abs(self.hi)) if self.kind != "vector" else self.hi

    def equivalent(self, margin: float) -> bool:
        if self.kind == "vector":
            return self.hi < margin
        return -margin < self.lo and self.hi < margin

    def clearly_different(self, margin: float) -> bool:
        if self.kind == "vector":
            return self.lo > margin
        return self.lo > margin or self.hi < -margin

    def direction(self) -> str:
        if self.kind == "vector":
            return "shape differs"
        if math.isinf(self.d):
            return "undefined (missing in candidate)"
        return "too high" if self.d > 0 else "too low"

    def to_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in asdict(self).items()}


def _undefined(n_cand: int, n_ref: int, kind: str) -> Comparison:
    return Comparison(math.inf, math.inf, math.inf, math.inf, math.nan, n_cand, n_ref, kind)


def _finite(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def applicable(ref_vals) -> bool:
    """A statistic is testable at a point if the reference defines it in most runs."""
    a = np.asarray(ref_vals, dtype=float)
    return a.size > 0 and np.isfinite(a).mean() >= 0.8


def vector_applicable(ref_vecs) -> bool:
    R = np.asarray(ref_vecs, dtype=float)
    return R.ndim == 2 and len(R) >= 2 and bool((np.isfinite(R).mean(axis=0) >= 0.8).any())


def _nan_excess(cand, ref) -> bool:
    c = np.asarray(cand, dtype=float)
    r = np.asarray(ref, dtype=float)
    return (1 - np.isfinite(c).mean()) > (1 - np.isfinite(r).mean()) + UNDEFINED_SLACK


def scalar_compare(cand, ref, alpha: float) -> Comparison:
    if _nan_excess(cand, ref):
        return _undefined(len(cand), len(ref), "scalar")
    c, r = _finite(cand), _finite(ref)
    if len(c) < 2 or len(r) < 2:
        return _undefined(len(c), len(r), "scalar")
    scale = float(_scale(r.std(ddof=1), r.mean()))
    vc, vr = c.var(ddof=1) / len(c), r.var(ddof=1) / len(r)
    d = (c.mean() - r.mean()) / scale
    se = math.sqrt(vc + vr) / scale
    if se == 0:
        return Comparison(d, d, d, 0.0, math.inf, len(c), len(r))
    df = (vc + vr) ** 2 / ((vc ** 2 / (len(c) - 1) if vc else 0) + (vr ** 2 / (len(r) - 1) if vr else 0) or 1e-300)
    t = sps.t.ppf(1 - alpha, df)
    return Comparison(d, d - t * se, d + t * se, se, df, len(c), len(r))


def effect_compare(cand_x, cand_0, ref_x, ref_0, alpha: float) -> Comparison:
    """Compare the candidate's response to an input change with the reference's response."""
    groups = [cand_x, cand_0, ref_x, ref_0]
    if _nan_excess(cand_x, ref_x) or _nan_excess(cand_0, ref_0):
        return _undefined(len(cand_x), len(ref_x), "effect")
    g = [_finite(v) for v in groups]
    if any(len(v) < 2 for v in g):
        return _undefined(len(g[0]), len(g[2]), "effect")
    scale = float(_scale(math.sqrt((g[2].var(ddof=1) + g[3].var(ddof=1)) / 2), g[3].mean()))
    eff_c = g[0].mean() - g[1].mean()
    eff_r = g[2].mean() - g[3].mean()
    vs = [v.var(ddof=1) / len(v) for v in g]
    d = (eff_c - eff_r) / scale
    se = math.sqrt(sum(vs)) / scale
    if se == 0:
        return Comparison(d, d, d, 0.0, math.inf, len(g[0]), len(g[2]), "effect")
    den = sum((v ** 2) / (len(x) - 1) for v, x in zip(vs, g) if v)
    df = sum(vs) ** 2 / den if den else math.inf
    t = sps.t.ppf(1 - alpha, df)
    return Comparison(d, d - t * se, d + t * se, se, df, len(g[0]), len(g[2]), "effect")


def vector_compare(cand, ref, alpha: float, n_boot: int = 400, seed: int = 0) -> Comparison:
    C = np.asarray(cand, dtype=float)
    R = np.asarray(ref, dtype=float)
    if C.ndim != 2 or R.ndim != 2 or len(C) < 2 or len(R) < 2:
        return _undefined(len(C), len(R), "vector")
    keep = np.isfinite(R).mean(axis=0) >= 0.8
    if not keep.any():
        return _undefined(len(C), len(R), "vector")
    C, R = C[:, keep], R[:, keep]
    if any(_nan_excess(C[:, k], R[:, k]) for k in range(C.shape[1])):
        return _undefined(len(C), len(R), "vector")
    sd = _scale(np.nanstd(R, axis=0, ddof=1), np.nanmean(R, axis=0))

    def dist(Ci, Ri):
        with np.errstate(all="ignore"):
            return float(np.nanmax(np.abs(np.nanmean(Ci, axis=0) - np.nanmean(Ri, axis=0)) / sd))

    D = dist(C, R)
    rng = np.random.default_rng(seed)
    boots = np.array([
        dist(C[rng.integers(0, len(C), len(C))], R[rng.integers(0, len(R), len(R))])
        for _ in range(n_boot)
    ])
    boots = boots[np.isfinite(boots)]
    if len(boots) < n_boot // 2:
        return _undefined(len(C), len(R), "vector")
    return Comparison(D, float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha)),
                      float(boots.std()), math.nan, len(C), len(R), "vector")


def margin_from_perturbations(comparisons: list[Comparison], floor: float) -> float:
    """Largest shift that innocuous perturbations cannot be ruled out to produce (UCB), floored."""
    ucbs = [c.ucb_abs for c in comparisons if math.isfinite(c.ucb_abs)]
    return float(max([floor] + ucbs))


def nonequivalence_p(comp: Comparison, margin: float) -> float:
    """One-sided p-value for 'the true difference exceeds the margin' (used by certificates)."""
    if math.isinf(comp.d):
        return 0.0
    if comp.kind == "vector":
        # approximate: normal approximation from bootstrap spread
        if comp.se <= 0:
            return 0.0 if comp.d > margin else 1.0
        return float(sps.norm.sf((comp.d - margin) / comp.se))
    if comp.se == 0:
        return 0.0 if abs(comp.d) > margin else 1.0
    return float(sps.t.sf((abs(comp.d) - margin) / comp.se, comp.df))


def holm(pvals: list[float], alpha: float) -> list[bool]:
    order = np.argsort(pvals)
    m = len(pvals)
    reject = [False] * m
    for rank, idx in enumerate(order):
        if pvals[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject
