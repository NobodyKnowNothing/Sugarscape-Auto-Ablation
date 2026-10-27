"""
The input domain and its random samples.

Equivalence is demanded over a pre-registered *domain* of inputs, not at a few
hand-picked points. Every evaluation draws its own sample of the domain
(scrambled Sobol' points for screening, i.i.d. uniform points for audits), the
way random search samples hyperparameter spaces: each sampled point carries a
distinct value of every input, so the number of points needed does not grow
with the number of inputs the way a grid does.

Each agent trait is sampled as (min, span) so that max >= min always holds and
homogeneous populations (span 0) are part of the domain. Scale-like inputs
(population, regrowth) are log-uniform.

Every point is evaluated in both trade worlds (paired), so the trade contrast
is estimated within points.

Regressors (for trait-response tests) are the interpretable quantities a
modeller would plot: log population, log regrowth, and for each trait its mean
and its spread, each rescaled so that its full range in the domain has length 1
and its midpoint is 0. A regression coefficient is therefore "the change in a
statistic across the whole range of that input".
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

WORLDS = {"trade": {"enable_trade": True}, "notrade": {"enable_trade": False}}


def _key_int(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


@dataclass(frozen=True)
class Point:
    id: str
    params: dict          # model inputs (without enable_trade)
    z: tuple              # regressors in [-0.5, 0.5]


class Domain:
    def __init__(self, spec: dict):
        self.spec = spec
        self.factors = spec["factors"]
        self.names = list(self.factors)
        self.traits = spec["traits"]                 # trait -> [min factor, span factor]
        self.scales = spec["scales"]                 # model input -> factor (log-uniform scale inputs)

    # -- factor transforms ----------------------------------------------------

    def _value(self, name: str, u: float):
        f = self.factors[name]
        lo, hi = f["low"], f["high"]
        if f["dist"] == "loguniform":
            v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        else:
            v = lo + u * (hi - lo) if f["type"] == "float" else lo + math.floor(u * (hi - lo + 1))
        if f["type"] == "int":
            return int(min(hi, max(lo, round(v) if f["dist"] == "loguniform" else v)))
        return float(round(v, 4))

    def params(self, u) -> dict:
        x = {n: self._value(n, float(ui)) for n, ui in zip(self.names, u)}
        out = {inp: x[fac] for inp, fac in self.scales.items()}
        for trait, (fmin, fspan) in self.traits.items():
            out[f"{trait}_min"] = x[fmin]
            out[f"{trait}_max"] = x[fmin] + x[fspan]
        return out

    # -- regressors -----------------------------------------------------------

    @property
    def regressor_names(self) -> list[str]:
        names = [f"log_{inp}" for inp in self.scales]
        for trait in self.traits:
            names += [f"{trait}_mean", f"{trait}_spread"]
        return names

    def regressors(self, params: dict) -> tuple:
        z = []
        for inp, fac in self.scales.items():
            f = self.factors[fac]
            lo, hi = math.log(f["low"]), math.log(f["high"])
            z.append((math.log(params[inp]) - lo) / (hi - lo) - 0.5)
        for trait, (fmin, fspan) in self.traits.items():
            m, s = self.factors[fmin], self.factors[fspan]
            mean_lo, mean_hi = m["low"] + s["low"] / 2, m["high"] + s["high"] / 2
            mean = (params[f"{trait}_min"] + params[f"{trait}_max"]) / 2
            spread = params[f"{trait}_max"] - params[f"{trait}_min"]
            z.append((mean - mean_lo) / (mean_hi - mean_lo) - 0.5)
            z.append((spread - s["low"]) / ((s["high"] - s["low"]) or 1) - 0.5)
        return tuple(round(v, 6) for v in z)

    # -- sampling ---------------------------------------------------------------

    def sample(self, n: int, key: str, method: str = "sobol") -> list[Point]:
        """Deterministic sample of n points for `key` ('sobol' = scrambled Sobol', 'iid' = uniform)."""
        d = len(self.names)
        seed = _key_int(f"{method}:{key}")
        if method == "sobol":
            m = max(1, math.ceil(math.log2(max(n, 2))))
            U = qmc.Sobol(d, scramble=True, seed=seed).random_base2(m)[:n]
        elif method == "iid":
            U = np.random.default_rng(seed).random((n, d))
        else:
            raise ValueError(method)
        pts = []
        for i, u in enumerate(U):
            p = self.params(u)
            pts.append(Point(f"{method[0]}{_key_int(key) % 10**6:06d}-{i:04d}", p, self.regressors(p)))
        return pts


def subset_order(n: int, key: str) -> list[int]:
    """A key-specific random order of bank indices (looks take growing prefixes)."""
    return [int(i) for i in np.random.default_rng(_key_int("order:" + key)).permutation(n)]
