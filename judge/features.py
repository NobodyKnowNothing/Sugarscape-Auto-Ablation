"""
Emergent-feature statistics, computed by the judge from observed snapshots.

Every statistic is implementation-agnostic (it only needs the generic Snapshot),
dimensionless where possible, and time-resolved with windows defined relative to
the run length T, so the same definitions work for any model and any profile.

  E1 Wealth inequality          Gini / top-10% share / CV / Hill tail index / Lorenz curve
  E2 Carrying capacity          plateau N/N0, relaxation time, fluctuation, N(t)/N0 trajectory
  E3 Price discovery            dispersion of log exchange ratios over time, its decay, the
                                equilibrium price level, exchange rate per agent
  E4 Spatial self-organization  density-resource rank correlation over time, Moran's I of
                                density, correlation relative to a uniform-placement null
  E5 Trait responses            (judge/surface.py) how E1-E4 respond to every input across
                                the input domain, including the trade on/off contrast

Each feature has a qualitative presence check (group level), per-run presence
indicators (whose means are presence probabilities, compared across the domain)
and a set of statistics that must be equivalent to canonical Sugarscape within
margin.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

from judge.contract import Snapshot

BIN = 5                       # 5x5-cell bins -> 10x10 spatial grid on a 50x50 world
LORENZ_Q = np.linspace(0.1, 0.9, 9)
TRAJ_POINTS = 21              # samples of N(t)/N0 between t=0 and t=T
DISP_BINS = 8                 # time bins for the price-dispersion trajectory
NULL_DRAWS = 200
DISP_EPS = 0.01               # log-price dispersion below this counts as converged (keeps the decay finite)
PLATEAU_TOL = 0.1             # |relative population drift over the late window| counted as a plateau
CLUSTER_Z = 1.645             # density-resource correlation this many null SDs above chance counts as clustering

# Statistics that enter the equivalence test, by feature.
SCALAR_STATS = {
    "E1": ["gini_T", "gini_c1", "top10_T", "cv_T", "loghill_T"],
    "E2": ["plateau_frac", "tau_frac", "fluct_cv"],
    "E3": ["disp_late", "disp_decay", "price_level", "exch_rate"],
    "E4": ["drc_c1", "drc_T", "moran_T"],
}
VECTOR_STATS = {
    "E1": ["lorenz_T"],
    "E2": ["pop_traj"],
    "E3": ["disp_traj"],
    "E4": [],
}
# Per-run 0/1 indicators of each qualitative feature; their means are presence probabilities.
INDICATOR_STATS = {
    "E1": ["ineq_up"],
    "E2": ["plateaued"],
    "E3": ["trading", "disp_down"],
    "E4": ["clustered"],
}
# Statistics that only exist when trade is enabled (E3 except the 'trading' indicator,
# which must be 0 when trade is disabled).
TRADE_ONLY = {"disp_late", "disp_decay", "price_level", "exch_rate", "disp_traj", "disp_down"}
FEATURES = ["E1", "E2", "E3", "E4"]
FEATURE_NAMES = {
    "E1": "wealth inequality",
    "E2": "carrying capacity",
    "E3": "price discovery",
    "E4": "spatial self-organization",
    "E5": "trait responses",
}


def stat_feature(name: str) -> str:
    for f in FEATURES:
        if name in SCALAR_STATS[f] or name in VECTOR_STATS[f] or name in INDICATOR_STATS[f]:
            return f
    raise KeyError(name)


def world_stats(trade: bool) -> dict[str, list[tuple[str, str]]]:
    """(stat, kind) pairs tested in a world, by feature; kind is scalar | vector | indicator."""
    out = {}
    for f in FEATURES:
        items = [(s, "scalar") for s in SCALAR_STATS[f]] + [(s, "vector") for s in VECTOR_STATS[f]] \
            + [(s, "indicator") for s in INDICATOR_STATS[f]]
        out[f] = [(s, k) for s, k in items if trade or s not in TRADE_ONLY]
    return out


# ---------------------------------------------------------------------------
# Primitive statistics
# ---------------------------------------------------------------------------

def gini(w: np.ndarray) -> float:
    w = np.sort(np.asarray(w, dtype=float))
    n = len(w)
    if n < 2 or w.sum() <= 0:
        return math.nan
    i = np.arange(1, n + 1)
    return float(np.sum((2 * i - n - 1) * w) / (n * w.sum()))


def top_share(w: np.ndarray, frac: float = 0.1) -> float:
    w = np.sort(np.asarray(w, dtype=float))[::-1]
    if len(w) < 2 or w.sum() <= 0:
        return math.nan
    k = max(1, int(math.ceil(frac * len(w))))
    return float(w[:k].sum() / w.sum())


def cv(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    if len(w) < 2 or w.mean() <= 0:
        return math.nan
    return float(w.std() / w.mean())


def hill_index(w: np.ndarray) -> float:
    """Hill estimator of the upper-tail index on the top ~10% of wealth."""
    w = np.sort(np.asarray(w, dtype=float))[::-1]
    w = w[w > 0]
    if len(w) < 7:
        return math.nan
    k = min(max(5, len(w) // 10), len(w) - 1)
    logs = np.log(w[:k] / w[k])
    m = logs.mean()
    return float(1.0 / m) if m > 0 else math.nan


def lorenz(w: np.ndarray) -> np.ndarray:
    w = np.sort(np.asarray(w, dtype=float))
    if len(w) < 2 or w.sum() <= 0:
        return np.full(len(LORENZ_Q), math.nan)
    cum = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    x = np.linspace(0, 1, len(w) + 1)
    return np.interp(LORENZ_Q, x, cum)


def bin_counts(positions: np.ndarray, w: int, h: int) -> np.ndarray:
    nb_x, nb_y = w // BIN, h // BIN
    grid = np.zeros((nb_x, nb_y))
    if len(positions):
        bx = np.minimum(positions[:, 0] // BIN, nb_x - 1)
        by = np.minimum(positions[:, 1] // BIN, nb_y - 1)
        np.add.at(grid, (bx, by), 1)
    return grid


def bin_sum(layer: np.ndarray) -> np.ndarray:
    w, h = layer.shape
    return layer[: w // BIN * BIN, : h // BIN * BIN].reshape(w // BIN, BIN, h // BIN, BIN).sum(axis=(1, 3))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return math.nan
    return float(sps.spearmanr(a, b).statistic)


def _queen_weights(nx: int, ny: int) -> np.ndarray:
    n = nx * ny
    W = np.zeros((n, n))
    for i in range(nx):
        for j in range(ny):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if (di or dj) and 0 <= i + di < nx and 0 <= j + dj < ny:
                        W[i * ny + j, (i + di) * ny + (j + dj)] = 1.0
    return W / W.sum(axis=1, keepdims=True)


_W_CACHE: dict[tuple, np.ndarray] = {}


def morans_i(grid: np.ndarray) -> float:
    nx, ny = grid.shape
    W = _W_CACHE.setdefault((nx, ny), _queen_weights(nx, ny))
    z = grid.ravel() - grid.mean()
    denom = float(z @ z)
    if denom <= 1e-12:
        return math.nan
    return float(len(z) * (z @ W @ z) / (W.sum() * denom))


# ---------------------------------------------------------------------------
# Online recorder (one per simulation run)
# ---------------------------------------------------------------------------

class Recorder:
    """Accumulates what the emergent-feature statistics need, step by step."""

    def __init__(self, env: dict, steps: int, seed: int):
        self.T = steps
        self.seed = seed
        self.N0 = env["initial_population"]
        self.w, self.h = env["width"], env["height"]
        self.cap_bins = bin_sum(np.asarray(env["landscape"]).sum(axis=0))
        self.c1 = max(1, round(steps / 4))
        self.c2 = max(1, round(steps / 2))
        self.early = set(range(1, max(2, round(0.1 * steps)) + 1))
        self.late_len = max(5, round(steps / 4))
        self.late_start = steps - self.late_len + 1
        self.pop = np.zeros(steps + 1)
        self.n_ex = np.zeros(steps + 1)
        self.disp = np.full(steps + 1, math.nan)
        self.late_logp: list[float] = []
        self.wealth: dict[int, np.ndarray] = {}
        self.drc: dict[int, float] = {}
        self.final_counts = None
        self.trade_enabled = bool(env.get("enable_trade", True))

    def update(self, t: int, snap: Snapshot) -> None:
        self.pop[t] = snap.n
        self.n_ex[t] = len(snap.exchanges)
        logp = []
        for _, _, dq in snap.exchanges:
            if len(dq) >= 2 and dq[0] != 0 and dq[1] != 0 and np.sign(dq[0]) != np.sign(dq[1]):
                logp.append(math.log(abs(dq[1] / dq[0])))
        if len(logp) >= 2:
            self.disp[t] = float(np.std(logp))
        if t >= self.late_start:
            self.late_logp.extend(logp)
        if t in (0, self.c1, self.c2, self.T):
            self.wealth[t] = snap.wealth.copy()
        if t in (self.c1, self.c2, self.T):
            counts = bin_counts(snap.positions, self.w, self.h)
            self.drc[t] = spearman(counts, self.cap_bins)
            if t == self.T:
                self.final_counts = counts

    # -- statistics ---------------------------------------------------------

    def stats(self) -> dict:
        T, s = self.T, {}
        wT = self.wealth.get(T, np.zeros(0))
        extinct = len(wT) == 0

        # E1 wealth inequality
        s["gini_0"] = gini(self.wealth.get(0, np.zeros(0)))
        s["gini_c1"] = gini(self.wealth.get(self.c1, np.zeros(0)))
        s["gini_T"] = gini(wT)
        s["top10_T"] = top_share(wT)
        s["cv_T"] = cv(wT)
        s["hill_T"] = hill_index(wT)
        s["loghill_T"] = math.log(s["hill_T"]) if s["hill_T"] > 0 else math.nan
        s["skew_T"] = float(sps.skew(wT)) if len(wT) > 2 and np.ptp(wT) > 0 else math.nan
        s["lorenz_T"] = lorenz(wT).tolist()

        # E2 carrying capacity
        late = self.pop[self.late_start:]
        Np = float(late.mean())
        s["plateau_frac"] = Np / self.N0
        tol = max(0.05 * Np, 2.0)
        within = np.abs(self.pop - Np) <= tol
        # first step after which the population stays within tolerance of its plateau
        outside = np.where(~within)[0]
        tau = 0 if len(outside) == 0 else int(outside[-1]) + 1
        s["tau_frac"] = min(tau, T) / T
        s["fluct_cv"] = float(late.std() / Np) if Np > 0 else math.nan
        tt = np.arange(len(late))
        slope = np.polyfit(tt, late, 1)[0] if len(late) > 1 else 0.0
        s["plateau_slope_rel"] = float(slope * len(late) / Np) if Np > 0 else math.nan
        idx = np.round(np.linspace(0, T, TRAJ_POINTS)).astype(int)
        s["pop_traj"] = (self.pop[idx] / self.N0).tolist()

        # E3 price discovery (undefined when the environment disables trade)
        if self.trade_enabled:
            early_d = [self.disp[t] for t in sorted(self.early) if not math.isnan(self.disp[t])]
            late_d = [d for d in self.disp[self.late_start:] if not math.isnan(d)]
            de = float(np.mean(early_d)) if early_d else math.nan
            dl = float(np.mean(late_d)) if late_d else math.nan
            s["disp_early"] = de
            s["disp_late"] = dl
            s["disp_decay"] = math.log((dl + DISP_EPS) / (de + DISP_EPS)) if de >= 0 and dl >= 0 else math.nan
            s["price_level"] = float(np.mean(self.late_logp)) if self.late_logp else math.nan
            pl = self.pop[self.late_start:]
            s["exch_rate"] = float(np.mean(self.n_ex[self.late_start:] / np.maximum(pl, 1)))
            edges = np.linspace(1, T + 1, DISP_BINS + 1).astype(int)
            traj = []
            for a, b in zip(edges[:-1], edges[1:]):
                seg = self.disp[a:b]
                seg = seg[~np.isnan(seg)]
                traj.append(float(seg.mean()) if len(seg) else math.nan)
            s["disp_traj"] = traj
        else:
            for k in ("disp_early", "disp_late", "disp_decay", "price_level", "exch_rate"):
                s[k] = math.nan
            s["disp_traj"] = [math.nan] * DISP_BINS
        s["n_exchanges"] = float(self.n_ex.sum())

        # E4 spatial self-organization
        s["drc_c1"] = self.drc.get(self.c1, math.nan)
        s["drc_c2"] = self.drc.get(self.c2, math.nan)
        s["drc_T"] = self.drc.get(T, math.nan)
        if extinct or self.final_counts is None:
            s["moran_T"] = math.nan
            s["z_drc"] = math.nan
        else:
            s["moran_T"] = morans_i(self.final_counts)
            s["z_drc"] = self._null_z(int(len(wT)), s["drc_T"])
        s["extinct"] = float(extinct)

        # presence indicators (0 where the feature is absent, including after extinction)
        s["ineq_up"] = float(not extinct and s["gini_T"] > s["gini_0"] and s["skew_T"] > 0)
        s["plateaued"] = float(not extinct and abs(s["plateau_slope_rel"]) < PLATEAU_TOL)
        s["trading"] = float(s["n_exchanges"] > 0)
        s["disp_down"] = float(s["disp_decay"] < 0) if self.trade_enabled else math.nan
        s["clustered"] = float(s["z_drc"] > CLUSTER_Z)
        return s

    def _null_z(self, n: int, observed: float) -> float:
        """Observed density-resource correlation versus uniform random placement of n agents."""
        if math.isnan(observed) or n < 2:
            return math.nan
        rng = np.random.default_rng([self.seed, 7919])
        nb = self.cap_bins.size
        null = []
        for _ in range(NULL_DRAWS):
            counts = rng.multinomial(n, np.full(nb, 1.0 / nb))
            null.append(spearman(counts, self.cap_bins))
        null = np.array([v for v in null if not math.isnan(v)])
        if len(null) < 10 or null.std() == 0:
            return math.nan
        return float((observed - null.mean()) / null.std())


# ---------------------------------------------------------------------------
# Group-level presence checks
# ---------------------------------------------------------------------------

def _finite(vals) -> np.ndarray:
    a = np.asarray(vals, dtype=float)
    return a[np.isfinite(a)]


def _one_sided_p(x: np.ndarray, greater: bool) -> float:
    if len(x) < 3:
        return 1.0
    if np.all(x == x[0]):
        return 0.0 if (x[0] > 0) == greater and x[0] != 0 else 1.0
    alt = "greater" if greater else "less"
    return float(sps.ttest_1samp(x, 0.0, alternative=alt).pvalue)


STRICT_FACTOR = 0.5        # the reference must show a feature with this much buffer (tolerance x0.5, alpha /10)


def presence(feature: str, runs: list[dict], alpha: float = 0.05, strict: bool = False) -> tuple[bool, str]:
    """
    Does the qualitative emergent feature appear in this group of runs?

    `strict=True` is used for the reference: a feature is only demanded of
    candidates where canonical Sugarscape shows it with a
    buffer, so borderline points do not turn sampling noise into rejections.
    """
    if strict:
        alpha = alpha / 10
    if len(runs) < 3:
        return False, "too few runs to test presence"
    if any(r.get("extinct") for r in runs):
        return False, "population went extinct in some runs"
    if feature == "E1":
        d = _finite([r["gini_T"] - r["gini_0"] for r in runs])
        skew = _finite([r["skew_T"] for r in runs])
        if len(d) < len(runs) or _one_sided_p(d, True) >= alpha:
            return False, "inequality does not grow beyond the initial endowments"
        if len(skew) == 0 or skew.mean() <= 0:
            return False, "wealth is not right-skewed"
        return True, ""
    if feature == "E2":
        sl = _finite([r["plateau_slope_rel"] for r in runs])
        tol = PLATEAU_TOL * (STRICT_FACTOR if strict else 1.0)
        if len(sl) < len(runs) or np.median(np.abs(sl)) >= tol:
            return False, "population does not reach a plateau"
        return True, ""
    if feature == "E3":
        ex = _finite([r["exch_rate"] for r in runs])
        if len(ex) == 0 or ex.mean() <= 0:
            return False, "no exchanges take place"
        dd = _finite([r["disp_decay"] for r in runs])
        if len(dd) < 0.8 * len(runs) or _one_sided_p(dd, False) >= alpha:
            return False, "price dispersion does not decrease over time"
        return True, ""
    if feature == "E4":
        z = _finite([r["z_drc"] for r in runs])
        if len(z) < len(runs) or _one_sided_p(z, True) >= alpha:
            return False, "agents do not concentrate on resources beyond chance"
        return True, ""
    raise KeyError(feature)
