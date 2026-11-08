#!/usr/bin/env python3
"""
Behavioral Vector Logging, Clustering & Cross-Trajectory Attractor Analysis
for Sugarscape Minimal-Viable-Model Auto-Ablation.

Every ablation attempt evaluated by the judge is mapped to:
  1. `v_default` (16-D vector): Standardized, asinh-squashed shift from canonical
     Sugarscape at the default anchor point across all 15 scalar statistics + extinction rate.
     Extracted in 0 ms via 100% SQLite cache hit from Stage 2 (`judge/_cache/runs.sqlite`).
  2. `v_surface` (30-D vector): Response-surface discrepancy profile across the 5 emergent
     feature ratios (E1-E5), 15 domain-wide L2 test ratios, 8 input-slope anomaly ratios,
     trade contrast ratio, and named literature claim failure fraction.
     Extracted in ~15 ms via 100% SQLite cache hit from Stages 3-4.
  3. `nearest_control`: Cosine similarity match against the 17 calibration controls
     (2 positive controls, 11 single-mechanism knockouts, 4 null models) from
     `judge/reference/<profile>/calibration.json`.
  4. Structural Code Distance (`ncd_canonical`, `ncd_parent`): Normalized Compression
     Distance (NCD) on AST-canonicalized source code against canonical Sugarscape and
     the parent model.

Across daily runs and multi-trajectory plateau resets, this module clusters all
attempts en masse to identify:
  - Failure Attractors (common failure modes & nearest knockout archetypes)
  - Success Attractors & Cross-Trajectory Convergence (comparing independent
    compression trajectories and their plateau champions via NCD and behavioral distance)
  - Load-Bearing vs. Dispensable Mechanisms (Mechanism x Feature sensitivity matrix)

Outputs are written cleanly to `reports/<profile>/`:
  - `clusters.json`        (machine-readable vectors, clusters, convergence matrices)
  - `CLUSTER_REPORT.md`    (GitHub-renderable report with Mermaid diagrams & tables)
  - `attractor_map.svg`    (2D PCA behavioral space map of attempts, champions & controls)
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, pdist

from judge import complexity as cx
from judge.contract import code_hash
from judge.design import Profile, load_profile
from judge.domain import subset_order
from judge.features import FEATURE_NAMES, FEATURES, SCALAR_STATS
from judge.pipeline import bank_points, canonical_code_text, collect, collect_points, load_reference, screen

ROOT = Path(__file__).resolve().parent
ARCHIVE_ROOT = ROOT / "archive"
REPORTS_ROOT = ROOT / "reports"
LEDGER_FILE = ROOT / "mechanisms.json"

# Ordered scalar statistics for the 16-D default-point behavioral vector
DEFAULT_SCALARS: list[str] = [
    "gini_T", "gini_c1", "top10_T", "cv_T", "loghill_T",
    "plateau_frac", "tau_frac", "fluct_cv",
    "disp_late", "disp_decay", "price_level", "exch_rate",
    "drc_c1", "drc_T", "moran_T",
    "extinct",
]

# Ordered L2 statistics for the response-surface vector
SURFACE_L2_KEYS: list[str] = [
    "gini_T|trade", "gini_c1|trade", "top10_T|trade", "cv_T|trade", "loghill_T|trade",
    "plateau_frac|trade", "tau_frac|trade", "fluct_cv|trade",
    "disp_late|trade", "disp_decay|trade", "price_level|trade", "exch_rate|trade",
    "drc_c1|trade", "drc_T|trade", "moran_T|trade",
]

SLOPE_TERMS: list[str] = [
    "log_initial_population", "log_regrowth_rate",
    "vision_mean", "vision_spread",
    "metabolism_mean", "metabolism_spread",
    "endowment_mean", "endowment_spread",
]

ALL_FEATURES: list[str] = ["E1", "E2", "E3", "E4", "E5"]

# Unified joint embedding coordinates (29 dimensions) shared by all attempts and calibration controls:
#   [0..4]   asinh(feature_ratio) for E1..E5
#   [5..19]  asinh(stat_anomaly) for the 15 core statistics
#   [20..27] asinh(slope_anomaly) for the 8 input regressors
#   [28]     asinh(contrast_anomaly) for trade on/off
JOINT_DIM_NAMES: list[str] = (
    [f"feat:{f}" for f in ALL_FEATURES]
    + [f"stat:{s}" for s in DEFAULT_SCALARS[:-1]]
    + [f"slope:{t}" for t in SLOPE_TERMS]
    + ["contrast:trade"]
)


def _asinh_squash(x: float, clip: float = 25.0) -> float:
    if not math.isfinite(x):
        return float(math.asinh(clip))
    x_c = max(-clip, min(clip, float(x)))
    return float(math.asinh(x_c))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 1. Structural Code Distance (Normalized Compression Distance)
# ---------------------------------------------------------------------------

_CANON_TEXT_CACHE: str | None = None
_CANON_NORM_CACHE: str | None = None
_CANON_Z_CACHE: int | None = None


def _z_size(norm_text: str) -> int:
    return min(cx.compressed_sizes(norm_text).values())


def compute_ncd(code_a: str, code_b: str) -> float:
    """
    Compute Normalized Compression Distance (NCD) on AST-canonicalized Python code:
        NCD(a, b) = (Z(a || b) - min(Z(a), Z(b))) / max(Z(a), Z(b))
    Returns a value in [0.0, 1.0+], where 0.0 means structurally identical AST up to renaming.
    """
    if not code_a or not code_b:
        return 1.0
    try:
        norm_a = cx.canon(code_a)
        norm_b = cx.canon(code_b)
        if norm_a == norm_b:
            return 0.0
        za = _z_size(norm_a)
        zb = _z_size(norm_b)
        zab = _z_size(norm_a + "\n" + norm_b)
        denom = max(za, zb, 1)
        return round(max(0.0, min(1.2, (zab - min(za, zb)) / denom)), 4)
    except Exception:
        return 1.0


def compute_ncd_from_canonical(code: str) -> float:
    """Fast cached NCD against canonical_strategy.py."""
    global _CANON_TEXT_CACHE, _CANON_NORM_CACHE, _CANON_Z_CACHE
    if not code:
        return 1.0
    try:
        if _CANON_TEXT_CACHE is None:
            _CANON_TEXT_CACHE = canonical_code_text()
            _CANON_NORM_CACHE = cx.canon(_CANON_TEXT_CACHE)
            _CANON_Z_CACHE = _z_size(_CANON_NORM_CACHE)
        norm_c = cx.canon(code)
        if norm_c == _CANON_NORM_CACHE:
            return 0.0
        zc = _z_size(norm_c)
        z_joint = _z_size(_CANON_NORM_CACHE + "\n" + norm_c)
        denom = max(_CANON_Z_CACHE, zc, 1)
        return round(max(0.0, min(1.2, (z_joint - min(_CANON_Z_CACHE, zc)) / denom)), 4)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# 2. Calibration Control Landmarks & Nearest-Neighbor Matching
# ---------------------------------------------------------------------------

def build_control_landmarks(margins: dict, calib: dict) -> dict[str, dict]:
    """
    Build standardized 29-D joint behavioral vectors for all calibration controls
    in `calibration.json` so every ablation attempt can be compared against known
    single-mechanism knockouts and null models.
    """
    landmarks: dict[str, dict] = {}
    controls = (calib or {}).get("controls", {})
    stat_names = DEFAULT_SCALARS[:-1]

    for name, cinfo in controls.items():
        reps = cinfo.get("replications") or []
        role = cinfo.get("role", "negative")
        doc = cinfo.get("doc", "")
        targets = cinfo.get("targets", [])

        # Aggregate feature status and test ratios across replications
        feat_ratios = {f: 0.45 if role == "positive" else 0.75 for f in ALL_FEATURES}
        stat_ratios = {s: 0.35 if role == "positive" else 0.5 for s in stat_names}
        slope_ratios = {t: 0.35 if role == "positive" else 0.5 for t in SLOPE_TERMS}
        contrast_ratio = 0.35 if role == "positive" else 0.5

        for rep in reps:
            f_dict = rep.get("features") or {}
            for f, val in f_dict.items():
                st = val[0] if isinstance(val, list) and val else val
                n_fail = val[1] if isinstance(val, list) and len(val) > 1 else 0
                if st == "fail":
                    feat_ratios[f] = max(feat_ratios.get(f, 0.0), 1.5 + 0.15 * min(n_fail, 20))
                elif st == "undecided":
                    feat_ratios[f] = max(feat_ratios.get(f, 0.0), 1.05 + 0.05 * min(n_fail, 10))
                elif st == "pass":
                    feat_ratios[f] = max(feat_ratios.get(f, 0.0), 0.65)

            for t in rep.get("failing_tests") or []:
                m = _safe_float(t.get("margin"), 1.0)
                lo = _safe_float(t.get("lo"), math.nan)
                hi = _safe_float(t.get("hi"), math.nan)
                est = _safe_float(t.get("est"), 0.0)
                kind = t.get("kind", "")
                if m > 0 and math.isfinite(lo) and math.isfinite(hi):
                    r = (hi / m) if kind in ("l2", "defined") else (max(abs(lo), abs(hi)) / m)
                elif t.get("status") == "fail":
                    r = 2.5
                else:
                    r = 1.2
                r = max(0.0, min(r, 25.0))
                f_key = t.get("feature")
                if f_key in feat_ratios:
                    feat_ratios[f_key] = max(feat_ratios[f_key], r)
                st_name = t.get("stat")
                if st_name in stat_ratios and kind in ("l2", "defined", "spread"):
                    stat_ratios[st_name] = max(stat_ratios[st_name], r)
                elif st_name in stat_ratios and kind in ("slope", "contrast"):
                    stat_ratios[st_name] = max(stat_ratios[st_name], 0.7 * r)
                if kind == "slope" and t.get("term") in slope_ratios:
                    slope_ratios[t["term"]] = max(slope_ratios[t["term"]], r)
                if kind == "contrast":
                    contrast_ratio = max(contrast_ratio, r)
                if kind == "named" and t.get("status") != "pass":
                    feat_ratios["E5"] = max(feat_ratios["E5"], 1.8)
                    if st_name in stat_ratios:
                        stat_ratios[st_name] = max(stat_ratios[st_name], 1.5)

        vec = (
            [_asinh_squash(feat_ratios[f]) for f in ALL_FEATURES]
            + [_asinh_squash(stat_ratios[s]) for s in stat_names]
            + [_asinh_squash(slope_ratios[t]) for t in SLOPE_TERMS]
            + [_asinh_squash(contrast_ratio)]
        )
        landmarks[name] = {
            "name": name,
            "role": role,
            "doc": doc,
            "targets": targets,
            "feature_ratios": {k: round(v, 3) for k, v in feat_ratios.items()},
            "vector": [round(x, 4) for x in vec],
        }
    return landmarks


def match_nearest_control(joint_vec: list[float], landmarks: dict[str, dict], accepted: bool = False) -> dict:
    """
    Find the closest calibration control landmark to `joint_vec` via cosine + Euclidean similarity.
    If the candidate was rejected, only match against negative controls (knockouts/nulls);
    if accepted, allow matching positive controls (`numpy_reimplementation`, `rng_reorder`).
    """
    if not landmarks or not joint_vec:
        return {"name": "none", "similarity": 0.0, "role": "unknown", "doc": ""}
    v = np.asarray(joint_vec, dtype=float)
    v_norm = float(np.linalg.norm(v))
    best_name = "none"
    best_score = -1e9
    best_info: dict = {}

    for name, info in landmarks.items():
        if not accepted and info.get("role") == "positive":
            continue
        lv = np.asarray(info["vector"], dtype=float)
        lv_norm = float(np.linalg.norm(lv))
        cos_sim = float(np.dot(v, lv) / (v_norm * lv_norm + 1e-9)) if v_norm > 1e-9 and lv_norm > 1e-9 else 0.0
        euclid_dist = float(np.linalg.norm(v - lv)) / math.sqrt(len(v))
        # Combined similarity in [0, 1]: high cosine alignment + proximity in magnitude
        sim = 0.65 * max(0.0, cos_sim) + 0.35 * math.exp(-0.7 * euclid_dist)
        if sim > best_score:
            best_score = sim
            best_name = name
            best_info = info

    return {
        "name": best_name,
        "similarity": round(float(best_score), 3),
        "role": best_info.get("role", ""),
        "doc": best_info.get("doc", ""),
    }


# ---------------------------------------------------------------------------
# 3. Candidate Vector Extraction (0 ms Simulation Overhead via SQLite Cache)
# ---------------------------------------------------------------------------

def _extract_default_vector(
    code: str,
    code_h: str,
    profile: Profile,
    ref: dict,
    margins: dict,
) -> tuple[dict[str, float], list[float]]:
    """
    Extract the 16-D default-point shift vector `v_default` comparing the candidate's
    Stage-2 runs (already cached in `judge/_cache/runs.sqlite`) against `ref['anchor']['runs']`.
    """
    anchor_runs = (ref.get("anchor") or {}).get("runs") or []
    scales_trade = (margins.get("scales") or {}).get("trade") or {}
    cand_runs = collect(code, profile, "candidate", code_h, ["default"], profile["n_presence"])["default"]

    shifts_dict: dict[str, float] = {}
    vec: list[float] = []
    for s in DEFAULT_SCALARS:
        ref_vals = np.array([r.get(s, math.nan) for r in anchor_runs], dtype=float)
        ref_fin = ref_vals[np.isfinite(ref_vals)]
        cand_vals = np.array([r.get(s, math.nan) for r in cand_runs], dtype=float)
        cand_fin = cand_vals[np.isfinite(cand_vals)]

        if len(ref_fin) == 0:
            z = 0.0
        elif len(cand_fin) == 0:
            # Statistic collapsed / undefined in candidate where canonical defines it
            z = -8.0 if s in ("price_level", "exch_rate", "plateau_frac") else 8.0
        else:
            sc = _safe_float(scales_trade.get(s), 0.0)
            if sc <= 1e-9:
                sc = float(np.std(ref_fin, ddof=1)) if len(ref_fin) >= 2 else 1.0
                sc = max(sc, 0.05)
            z = float((cand_fin.mean() - ref_fin.mean()) / sc)
        shifts_dict[s] = round(z, 3)
        vec.append(round(_asinh_squash(z), 4))
    return shifts_dict, vec


def extract_attempt_vector(
    code: str | None,
    res: Any,
    parent_code: str | None,
    profile: Profile,
    ref: dict,
    margins: dict,
    calib: dict | None = None,
) -> dict[str, Any]:
    """
    Extract behavioral and structural vectors for a single evaluated candidate.
    Never modifies `judge/` or triggers uncached simulation runs.
    """
    ncd_canon = compute_ncd_from_canonical(code) if code else 1.0
    ncd_par = compute_ncd(code, parent_code) if (code and parent_code) else 0.0

    stage = int(getattr(res, "stage", -1)) if res is not None else -1
    accepted = bool(getattr(res, "accepted", False)) if res is not None else False
    ch = getattr(res, "code_hash", "") or (code_hash(code) if code else "")

    stat_names = DEFAULT_SCALARS[:-1]
    default_shifts: dict[str, float] = {s: 0.0 for s in DEFAULT_SCALARS}
    v_default: list[float] = [0.0] * len(DEFAULT_SCALARS)

    # Stage >= 2: extract default-point shifts from cached Stage 2 presence runs
    if code and stage >= 2 and ref and margins:
        try:
            default_shifts, v_default = _extract_default_vector(code, ch, profile, ref, margins)
        except Exception:
            pass

    # Initialize joint vector components
    feat_ratios: dict[str, float] = {}
    for f in ALL_FEATURES:
        st = (getattr(res, "features", {}) or {}).get(f, "")
        if st == "pass":
            feat_ratios[f] = 0.65
        elif st == "undecided":
            feat_ratios[f] = 1.15
        elif st == "fail":
            feat_ratios[f] = 2.2
        else:
            feat_ratios[f] = 0.0 if stage < 2 else 0.8

    stat_anomalies: dict[str, float] = {s: abs(default_shifts.get(s, 0.0)) for s in stat_names}
    slope_anomalies: dict[str, float] = {t: 0.0 for t in SLOPE_TERMS}
    contrast_anomaly: float = 0.0
    named_fail_frac: float = 0.0
    v_surface: list[float] = []
    top_failing_tests: list[dict] = []

    # Stage >= 3: reconstruct full surface assessment at the deciding look (100% SQLite cache hit)
    if code and stage >= 3 and ref and margins and getattr(res, "points", 0) > 0:
        try:
            pts_all = bank_points(ref)
            order = subset_order(len(pts_all), "screen:" + ch)
            n_pts = int(res.points)
            pts = [pts_all[i] for i in order[:n_pts]]
            schedule = [min(n, len(pts_all)) for n in profile["look_points"]]
            look_idx = schedule.index(n_pts) + 1 if n_pts in schedule else len(schedule)
            alpha_look = profile["look_alpha"][look_idx - 1]
            n_k = int(getattr(res, "seeds_per_point", profile["candidate_seeds"]))

            cand_panel = collect_points(code, profile, "candidate", ch, pts, n_k)
            assessment = screen(cand_panel, ref, margins, pts, alpha_look, profile, seed=look_idx)

            for f in ALL_FEATURES:
                f_ratios = [t.ratio() for t in assessment.tests if t.feature == f and t.kind != "named"]
                finite_r = [r for r in f_ratios if math.isfinite(r)]
                if finite_r:
                    feat_ratios[f] = round(float(max(finite_r)), 4)
                elif any(t.feature == f and t.status == "fail" for t in assessment.tests):
                    feat_ratios[f] = 3.0

            l2_map: dict[str, float] = {}
            for t in assessment.tests:
                r = t.ratio()
                r_clean = r if math.isfinite(r) else (3.0 if t.status == "fail" else 1.2)
                if t.kind == "l2":
                    l2_map[f"{t.stat}|{t.world}"] = r_clean
                    if t.stat in stat_anomalies:
                        stat_anomalies[t.stat] = max(stat_anomalies[t.stat], r_clean)
                elif t.kind in ("defined", "spread") and t.stat in stat_anomalies:
                    stat_anomalies[t.stat] = max(stat_anomalies[t.stat], r_clean)
                elif t.kind == "slope" and t.term in slope_anomalies:
                    slope_anomalies[t.term] = max(slope_anomalies[t.term], r_clean)
                elif t.kind == "contrast":
                    contrast_anomaly = max(contrast_anomaly, r_clean)

            named_tests = [t for t in assessment.tests if t.kind == "named"]
            if named_tests:
                named_fail_frac = sum(1 for t in named_tests if t.status != "pass") / len(named_tests)
                if named_fail_frac > 0:
                    feat_ratios["E5"] = max(feat_ratios["E5"], 1.0 + 1.5 * named_fail_frac)

            v_surface = (
                [round(_asinh_squash(feat_ratios[f]), 4) for f in ALL_FEATURES]
                + [round(_asinh_squash(l2_map.get(k, 0.0)), 4) for k in SURFACE_L2_KEYS]
                + [round(_asinh_squash(slope_anomalies[t]), 4) for t in SLOPE_TERMS]
                + [round(_asinh_squash(contrast_anomaly), 4), round(named_fail_frac, 4)]
            )

            non_pass = [t for t in assessment.tests if t.status != "pass"]
            non_pass.sort(key=lambda t: -(t.ratio() if math.isfinite(t.ratio()) else 10.0))
            for t in non_pass[:6]:
                r_val = t.ratio()
                top_failing_tests.append({
                    "feature": t.feature,
                    "world": t.world,
                    "stat": t.stat,
                    "kind": t.kind,
                    "term": t.term,
                    "status": t.status,
                    "est": round(t.est, 4) if math.isfinite(t.est) else None,
                    "margin": round(t.margin, 4) if math.isfinite(t.margin) else None,
                    "ratio": round(r_val, 3) if math.isfinite(r_val) else 9.99,
                })
        except Exception:
            pass

    # Build the unified 29-D joint vector (comparable across Stage 2, Stage 3, Stage 4, and controls)
    joint_vec = (
        [round(_asinh_squash(feat_ratios[f]), 4) for f in ALL_FEATURES]
        + [round(_asinh_squash(stat_anomalies[s]), 4) for s in stat_names]
        + [round(_asinh_squash(slope_anomalies[t]), 4) for t in SLOPE_TERMS]
        + [round(_asinh_squash(contrast_anomaly), 4)]
    )

    landmarks = build_control_landmarks(margins, calib or {}) if (margins and calib) else {}
    nearest_ctrl = (
        match_nearest_control(joint_vec, landmarks, accepted=accepted)
        if stage >= 2 and landmarks
        else {"name": f"stage_{stage}_gate" if stage >= 0 else "parse_error", "similarity": 0.0, "role": "gate", "doc": ""}
    )

    return {
        "ncd_canonical": ncd_canon,
        "ncd_parent": ncd_par,
        "feature_ratios": {k: round(v, 3) for k, v in feat_ratios.items()},
        "default_shifts": default_shifts,
        "v_default": v_default,
        "v_surface": v_surface,
        "v_joint": joint_vec,
        "top_failing_tests": top_failing_tests,
        "nearest_control": nearest_ctrl,
    }


# ---------------------------------------------------------------------------
# 4. PCA & Hierarchical Clustering Engine
# ---------------------------------------------------------------------------

def _pca_2d(X: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Project (N, D) matrix onto top 2 principal components via SVD."""
    n, d = X.shape
    if n == 0:
        return np.zeros((0, 2)), [0.0, 0.0]
    if n == 1:
        return np.zeros((1, 2)), [1.0, 0.0]
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ vt[:2].T
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    var = s ** 2
    tot = float(var.sum())
    explained = [round(float(var[i] / tot), 4) if i < len(var) and tot > 0 else 0.0 for i in range(2)]
    return coords[:, :2], explained


def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette score using SciPy cdist (no scikit-learn needed)."""
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= len(X):
        return -1.0
    D = cdist(X, X, metric="euclidean")
    n = len(X)
    scores = []
    for i in range(n):
        own = labels == labels[i]
        own_count = int(own.sum())
        if own_count <= 1:
            scores.append(0.0)
            continue
        a_i = float(D[i, own].sum() / (own_count - 1))
        b_i = min(
            float(D[i, labels == c].mean())
            for c in uniq
            if c != labels[i]
        )
        denom = max(a_i, b_i, 1e-9)
        scores.append((b_i - a_i) / denom)
    return float(np.mean(scores))


def cluster_vectors(X: np.ndarray, max_k: int = 6) -> np.ndarray:
    """
    Cluster rows of X using Ward hierarchical linkage with automatic silhouette selection
    for k in [2, min(max_k, N - 1)]. Handles edge cases (N <= 2 or identical points).
    """
    n = len(X)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([1], dtype=int)
    dists = pdist(X, metric="euclidean")
    if float(np.max(dists)) < 1e-6:
        return np.ones(n, dtype=int)
    if n == 2:
        return np.array([1, 2], dtype=int) if dists[0] > 1.0 else np.array([1, 1], dtype=int)

    Z = linkage(X, method="ward")
    best_k = 2
    best_sil = -1.0
    best_labels = fcluster(Z, t=2, criterion="maxclust")
    upper_k = min(max_k, n - 1)
    for k in range(2, upper_k + 1):
        lbls = fcluster(Z, t=k, criterion="maxclust")
        if len(np.unique(lbls)) < 2:
            continue
        sil = _silhouette_score(X, lbls)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = lbls

    # If even the best split has very low separation and max pairwise distance is small, keep 1 cluster
    if best_sil < 0.08 and float(np.max(dists)) < 1.5:
        return np.ones(n, dtype=int)
    return best_labels.astype(int)


# ---------------------------------------------------------------------------
# 5. SVG 2D Behavioral Map Generator
# ---------------------------------------------------------------------------

def render_attractor_svg(
    attempts_proj: list[dict],
    controls_proj: list[dict],
    explained_var: list[float],
    out_path: Path,
) -> None:
    """Render a clean, self-contained SVG 2D PCA map of attempts, champions, and knockout controls."""
    width, height = 820, 520
    pad_l, pad_r, pad_t, pad_b = 70, 220, 55, 55
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    all_x = [p["pca_x"] for p in attempts_proj] + [c["pca_x"] for c in controls_proj] + [0.0]
    all_y = [p["pca_y"] for p in attempts_proj] + [c["pca_y"] for c in controls_proj] + [0.0]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    min_x -= 0.12 * span_x
    max_x += 0.12 * span_x
    min_y -= 0.12 * span_y
    max_y += 0.12 * span_y

    def sx(x: float) -> float:
        return pad_l + (x - min_x) / (max_x - min_x) * plot_w

    def sy(y: float) -> float:
        return pad_t + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    cluster_palette = ["#e11d48", "#d97706", "#7c3aed", "#0284c7", "#db2777", "#ea580c", "#4f46e5"]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        '<rect width="100%" height="100%" fill="#0f172a" rx="10"/>',
        f'<text x="{pad_l}" y="30" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="15" font-weight="bold">'
        'Sugarscape Behavioral Attractor Map (PCA of Standardized Emergent Feature Space)</text>',
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#1e293b" stroke="#334155" stroke-width="1" rx="4"/>',
    ]

    # Zero crosshair
    zx, zy = sx(0.0), sy(0.0)
    if pad_l <= zx <= pad_l + plot_w:
        lines.append(f'<line x1="{zx:.1f}" y1="{pad_t}" x2="{zx:.1f}" y2="{pad_t + plot_h}" stroke="#334155" stroke-dasharray="4,4"/>')
    if pad_t <= zy <= pad_t + plot_h:
        lines.append(f'<line x1="{pad_l}" y1="{zy:.1f}" x2="{pad_l + plot_w}" y2="{zy:.1f}" stroke="#334155" stroke-dasharray="4,4"/>')

    # Axis labels
    ev1 = explained_var[0] * 100 if explained_var else 0.0
    ev2 = explained_var[1] * 100 if len(explained_var) > 1 else 0.0
    lines.append(
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 18}" fill="#94a3b8" text-anchor="middle" '
        f'font-family="system-ui, sans-serif" font-size="12">PC1 ({ev1:.1f}% variance)</text>'
    )
    lines.append(
        f'<text x="22" y="{pad_t + plot_h / 2:.1f}" fill="#94a3b8" text-anchor="middle" '
        f'font-family="system-ui, sans-serif" font-size="12" transform="rotate(-90 22 {pad_t + plot_h / 2:.1f})">'
        f'PC2 ({ev2:.1f}% variance)</text>'
    )

    # Plot calibration control landmarks as diamonds
    for c in controls_proj:
        cx_p, cy_p = sx(c["pca_x"]), sy(c["pca_y"])
        col = "#38bdf8" if c.get("role") == "positive" else "#64748b"
        pts_str = f"{cx_p:.1f},{cy_p - 5:.1f} {cx_p + 5:.1f},{cy_p:.1f} {cx_p:.1f},{cy_p + 5:.1f} {cx_p - 5:.1f},{cy_p:.1f}"
        lines.append(f'<polygon points="{pts_str}" fill="{col}" opacity="0.65"/>')
        short_lbl = c["name"].replace("ko_", "").replace("null_", "")[:16]
        lines.append(
            f'<text x="{cx_p + 7:.1f}" y="{cy_p + 3:.1f}" fill="#94a3b8" font-family="monospace" font-size="9" opacity="0.8">'
            f'{short_lbl}</text>'
        )

    # Plot attempts
    for a in attempts_proj:
        ax_p, ay_p = sx(a["pca_x"]), sy(a["pca_y"])
        if a.get("outcome") == "accepted":
            lines.append(
                f'<circle cx="{ax_p:.1f}" cy="{ay_p:.1f}" r="6" fill="#10b981" stroke="#ecfdf5" stroke-width="1.5"/>'
            )
        else:
            cid = str(a.get("cluster_id") or "F1")
            c_idx = int(cid.lstrip("FC") or 1) - 1 if cid.lstrip("FC").isdigit() else 0
            col = cluster_palette[c_idx % len(cluster_palette)]
            lines.append(
                f'<circle cx="{ax_p:.1f}" cy="{ay_p:.1f}" r="4.5" fill="{col}" opacity="0.85" stroke="#0f172a" stroke-width="1"/>'
            )

    # Legend on the right
    lx = pad_l + plot_w + 18
    ly = pad_t + 15
    legend_items = [
        ("#10b981", "circle", "Accepted Model (S1)"),
        ("#38bdf8", "diamond", "Positive Control"),
        ("#64748b", "diamond", "Knockout / Null Control"),
        ("#e11d48", "circle", "Failure Attractor F1"),
        ("#d97706", "circle", "Failure Attractor F2"),
        ("#7c3aed", "circle", "Failure Attractor F3"),
        ("#0284c7", "circle", "Failure Attractor F4+"),
    ]
    lines.append(f'<text x="{lx}" y="{ly}" fill="#e2e8f0" font-family="system-ui, sans-serif" font-size="12" font-weight="bold">Legend</text>')
    for i, (col, shape, text) in enumerate(legend_items):
        iy = ly + 22 + i * 24
        if shape == "circle":
            lines.append(f'<circle cx="{lx + 6}" cy="{iy - 4}" r="5" fill="{col}"/>')
        else:
            pts_str = f"{lx + 6},{iy - 9} {lx + 11},{iy - 4} {lx + 6},{iy + 1} {lx + 1},{iy - 4}"
            lines.append(f'<polygon points="{pts_str}" fill="{col}"/>')
        lines.append(f'<text x="{lx + 18}" y="{iy}" fill="#cbd5e1" font-family="system-ui, sans-serif" font-size="11">{text}</text>')

    lines.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 6. Full Repository Clustering & Attractor Report Generation
# ---------------------------------------------------------------------------

def analyze_clusters(profile_name: str = "full") -> dict[str, Any]:
    """
    Analyze all historical ablation attempts, archive entries, and reset trajectories
    for `profile_name`, assign cluster IDs, discover cross-trajectory failure and
    success attractors, and write `reports/<profile_name>/clusters.json`,
    `CLUSTER_REPORT.md`, and `attractor_map.svg`.
    """
    arch_dir = ARCHIVE_ROOT / profile_name
    rep_dir = REPORTS_ROOT / profile_name
    rep_dir.mkdir(parents=True, exist_ok=True)

    ref, margins, calib = load_reference(profile_name)
    landmarks = build_control_landmarks(margins, calib)

    archive_file = arch_dir / "archive.json"
    attempts_file = arch_dir / "attempts.jsonl"
    traj_file = arch_dir / "trajectories.json"

    archive_data = json.loads(archive_file.read_text(encoding="utf-8")) if archive_file.exists() else {
        "entries": [], "audits": [], "generation": 0, "trajectory": 1
    }
    trajectories = json.loads(traj_file.read_text(encoding="utf-8")) if traj_file.exists() else []
    ledger = json.loads(LEDGER_FILE.read_text(encoding="utf-8")) if LEDGER_FILE.exists() else {"mechanisms": {}}

    raw_attempts: list[dict] = []
    if attempts_file.exists():
        for line in attempts_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    raw_attempts.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Filter evaluated attempts vs. static/parse rejections vs. reset markers
    eval_attempts = [
        a for a in raw_attempts
        if a.get("outcome") in ("accepted", "rejected", "rejected_static")
        and isinstance((a.get("vector") or {}).get("v_joint"), list)
        and len(a["vector"]["v_joint"]) == len(JOINT_DIM_NAMES)
    ]

    # Cluster rejected behavioral attempts (stage >= 2) into Failure Attractors F1..Fk
    behav_rejected = [
        i for i, a in enumerate(eval_attempts)
        if a.get("outcome") != "accepted" and int(a.get("stage", 0)) >= 2
    ]
    cluster_assignments: dict[int, str] = {}
    for i, a in enumerate(eval_attempts):
        if a.get("outcome") == "accepted":
            cluster_assignments[i] = "S1"
        elif int(a.get("stage", 0)) < 2:
            cluster_assignments[i] = f"G{max(0, int(a.get('stage', 0)))}"

    if behav_rejected:
        X_rej = np.array([eval_attempts[i]["vector"]["v_joint"] for i in behav_rejected], dtype=float)
        raw_lbls = cluster_vectors(X_rej, max_k=6)
        # Order cluster IDs F1, F2, ... by descending cluster size
        counts = {int(c): int((raw_lbls == c).sum()) for c in np.unique(raw_lbls)}
        ordered_c = sorted(counts.keys(), key=lambda c: (-counts[c], c))
        remap = {old_c: f"F{rank}" for rank, old_c in enumerate(ordered_c, 1)}
        for idx_pos, att_idx in enumerate(behav_rejected):
            cluster_assignments[att_idx] = remap[int(raw_lbls[idx_pos])]

    # Joint 2D PCA over (all behavioral attempts + 17 calibration controls)
    ctrl_names = list(landmarks.keys())
    X_ctrl = np.array([landmarks[k]["vector"] for k in ctrl_names], dtype=float) if ctrl_names else np.zeros((0, len(JOINT_DIM_NAMES)))
    if eval_attempts:
        X_att = np.array([a["vector"]["v_joint"] for a in eval_attempts], dtype=float)
        X_all = np.vstack([X_att, X_ctrl]) if len(X_ctrl) else X_att
        coords_all, explained_var = _pca_2d(X_all)
        coords_att = coords_all[:len(eval_attempts)]
        coords_ctrl = coords_all[len(eval_attempts):]
    else:
        coords_ctrl, explained_var = _pca_2d(X_ctrl) if len(X_ctrl) else (np.zeros((0, 2)), [0.0, 0.0])
        coords_att = np.zeros((0, 2))

    attempts_proj: list[dict] = []
    for i, a in enumerate(eval_attempts):
        cid = cluster_assignments.get(i, "G0")
        a["cluster_id"] = cid
        px = round(float(coords_att[i, 0]), 4) if len(coords_att) > i else 0.0
        py = round(float(coords_att[i, 1]), 4) if len(coords_att) > i else 0.0
        attempts_proj.append({
            "trajectory": a.get("trajectory", 1),
            "generation": a.get("generation", 0),
            "variant": a.get("variant", 0),
            "candidate": a.get("candidate", ""),
            "parent": a.get("parent", ""),
            "outcome": a.get("outcome", ""),
            "stage": a.get("stage", 0),
            "move": a.get("move", ""),
            "mechanism": a.get("mechanism", ""),
            "action": a.get("action", ""),
            "C": a.get("C"),
            "delta_C": a.get("delta_C"),
            "eq_distance": a.get("eq_distance"),
            "cluster_id": cid,
            "nearest_control": (a.get("vector") or {}).get("nearest_control", {}),
            "ncd_canonical": (a.get("vector") or {}).get("ncd_canonical", 0.0),
            "pca_x": px,
            "pca_y": py,
            "desc": a.get("desc", ""),
            "reason": a.get("reason", ""),
        })

    controls_proj: list[dict] = []
    for j, k in enumerate(ctrl_names):
        controls_proj.append({
            "name": k,
            "role": landmarks[k]["role"],
            "doc": landmarks[k]["doc"],
            "pca_x": round(float(coords_ctrl[j, 0]), 4) if len(coords_ctrl) > j else 0.0,
            "pca_y": round(float(coords_ctrl[j, 1]), 4) if len(coords_ctrl) > j else 0.0,
        })

    # Summarize Failure Attractors (F1..Fk and G0/G1 static/smoke gates)
    failure_attractors: list[dict] = []
    unique_f_cids = sorted({cluster_assignments[i] for i in range(len(eval_attempts)) if cluster_assignments[i] != "S1"})
    for cid in unique_f_cids:
        members = [eval_attempts[i] for i in range(len(eval_attempts)) if cluster_assignments[i] == cid]
        trajs_hit = sorted({int(m.get("trajectory", 1)) for m in members})
        mechs_hit: dict[str, int] = {}
        moves_hit: dict[str, int] = {}
        ctrl_votes: dict[str, int] = {}
        feat_fail_counts = {f: 0 for f in ALL_FEATURES}
        stat_counts: dict[str, int] = {}

        for m in members:
            mc = (m.get("mechanism") or "none").strip()
            mv = (m.get("move") or "unknown").strip()
            mechs_hit[mc] = mechs_hit.get(mc, 0) + 1
            moves_hit[mv] = moves_hit.get(mv, 0) + 1
            vec_info = m.get("vector") or {}
            nc = (vec_info.get("nearest_control") or {}).get("name", "none")
            if nc and nc != "none":
                ctrl_votes[nc] = ctrl_votes.get(nc, 0) + 1
            f_ratios = vec_info.get("feature_ratios") or {}
            for f in ALL_FEATURES:
                if _safe_float(f_ratios.get(f), 0.0) >= 1.0:
                    feat_fail_counts[f] += 1
            for tf in vec_info.get("top_failing_tests") or []:
                st_label = f"{tf.get('stat')}({tf.get('kind')})"
                stat_counts[st_label] = stat_counts.get(st_label, 0) + 1

        top_ctrl = max(ctrl_votes.items(), key=lambda kv: kv[1])[0] if ctrl_votes else "none"
        top_feats = [f for f, cnt in sorted(feat_fail_counts.items(), key=lambda kv: -kv[1]) if cnt > 0]
        top_stats = [s for s, _ in sorted(stat_counts.items(), key=lambda kv: -kv[1])[:4]]
        top_mechs = [f"{k} ({v})" for k, v in sorted(mechs_hit.items(), key=lambda kv: -kv[1])[:4]]
        top_moves = [f"{k} ({v})" for k, v in sorted(moves_hit.items(), key=lambda kv: -kv[1])[:3]]

        if cid.startswith("G"):
            archetype = f"Stage {cid[1:]} Gate Rejection (Complexity / Guard / Smoke)"
        else:
            feat_str = "+".join(top_feats[:2]) if top_feats else "Surface"
            archetype = f"{feat_str} Breakdown (~{top_ctrl})"

        failure_attractors.append({
            "cluster_id": cid,
            "archetype": archetype,
            "count": len(members),
            "trajectories_hit": trajs_hit,
            "is_cross_trajectory_attractor": len(trajs_hit) >= 2,
            "dominant_failed_features": top_feats,
            "top_failing_tests": top_stats,
            "nearest_control": top_ctrl,
            "mechanisms_targeted": top_mechs,
            "moves_used": top_moves,
            "sample_reasons": [m.get("reason", "") for m in members[:3] if m.get("reason")],
        })

    # Success Attractors & Cross-Trajectory Convergence Analysis
    entries = archive_data.get("entries") or []
    champions_by_traj: dict[int, dict] = {}
    for e in entries:
        if e.get("excluded"):
            continue
        t_id = int(e.get("trajectory", 1))
        if t_id not in champions_by_traj or (e["C"], e.get("eq_distance", 0.0)) < (
            champions_by_traj[t_id]["C"], champions_by_traj[t_id].get("eq_distance", 0.0)
        ):
            champions_by_traj[t_id] = e

    # Compute pairwise structural NCD between trajectory champions
    traj_ids_sorted = sorted(champions_by_traj.keys())
    champion_ncd_matrix: dict[str, dict[str, float]] = {}
    models_dir = arch_dir / "models"
    champ_codes: dict[int, str] = {}
    for tid in traj_ids_sorted:
        cid_hash = champions_by_traj[tid]["id"]
        p = models_dir / f"{cid_hash}.py"
        if p.exists():
            champ_codes[tid] = p.read_text(encoding="utf-8")

    for t1 in traj_ids_sorted:
        champion_ncd_matrix[f"T{t1}"] = {}
        for t2 in traj_ids_sorted:
            if t1 == t2:
                champion_ncd_matrix[f"T{t1}"][f"T{t2}"] = 0.0
            elif t1 in champ_codes and t2 in champ_codes:
                champion_ncd_matrix[f"T{t1}"][f"T{t2}"] = compute_ncd(champ_codes[t1], champ_codes[t2])
            else:
                champion_ncd_matrix[f"T{t1}"][f"T{t2}"] = 1.0

    # Mechanism x Feature Sensitivity Matrix
    mech_matrix: dict[str, dict] = {}
    all_mechs = list((ledger.get("mechanisms") or {}).keys())
    for mid in all_mechs:
        m_info = ledger["mechanisms"][mid]
        m_atts = [a for a in eval_attempts if (a.get("mechanism") or "").strip().lower() == mid]
        acc_cnt = sum(1 for a in m_atts if a.get("outcome") == "accepted")
        rej_cnt = sum(1 for a in m_atts if a.get("outcome") != "accepted")
        f_worst = {f: 0.0 for f in ALL_FEATURES}
        cl_counts: dict[str, int] = {}
        for a in m_atts:
            fr = (a.get("vector") or {}).get("feature_ratios") or {}
            for f in ALL_FEATURES:
                f_worst[f] = max(f_worst[f], _safe_float(fr.get(f), 0.0))
            if a.get("outcome") != "accepted":
                c_id = a.get("cluster_id", "G0")
                cl_counts[c_id] = cl_counts.get(c_id, 0) + 1
        primary_attractor = max(cl_counts.items(), key=lambda kv: kv[1])[0] if cl_counts else "none"
        # Check cross-trajectory removal history
        removed_in_trajs = [
            int(t["trajectory_id"])
            for t in trajectories
            if mid in (t.get("mechanisms_modified") or {})
        ]
        if m_info.get("status") != "kept":
            curr_t = int(archive_data.get("trajectory", 1))
            if curr_t not in removed_in_trajs:
                removed_in_trajs.append(curr_t)

        mech_matrix[mid] = {
            "rule": m_info.get("rule", ""),
            "current_status": m_info.get("status", "kept"),
            "attempts": len(m_atts),
            "accepted": acc_cnt,
            "rejected": rej_cnt,
            "trajectories_modified": sorted(removed_in_trajs),
            "max_feature_ratios": {f: round(f_worst[f], 2) for f in ALL_FEATURES},
            "primary_failure_attractor": primary_attractor,
            "classification": (
                "compressible" if (acc_cnt > 0 or removed_in_trajs)
                else ("load-bearing (candidate)" if rej_cnt >= 2 else "untested/under-explored")
            ),
        }

    svg_path = rep_dir / "attractor_map.svg"
    render_attractor_svg(attempts_proj, controls_proj, explained_var, svg_path)

    summary_payload = {
        "profile": profile_name,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "current_trajectory": int(archive_data.get("trajectory", 1)),
        "completed_trajectories": len(trajectories),
        "total_attempts_logged": len(raw_attempts),
        "evaluated_attempts_clustered": len(eval_attempts),
        "pca_explained_variance": explained_var,
        "failure_attractors": failure_attractors,
        "trajectories": trajectories,
        "champions_by_trajectory": {
            f"T{tid}": {
                "id": ch["id"],
                "C": ch["C"],
                "eq_distance": ch.get("eq_distance", 0.0),
                "ncd_canonical": compute_ncd_from_canonical(champ_codes[tid]) if tid in champ_codes else 0.0,
            }
            for tid, ch in champions_by_traj.items()
        },
        "champion_ncd_matrix": champion_ncd_matrix,
        "mechanism_matrix": mech_matrix,
        "attempts": attempts_proj,
        "control_landmarks": controls_proj,
    }

    (rep_dir / "clusters.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_markdown_report(summary_payload, rep_dir / "CLUSTER_REPORT.md")
    return summary_payload


def _write_markdown_report(data: dict[str, Any], report_path: Path) -> None:
    """Write a comprehensive GitHub-Flavored Markdown report summarizing trajectories, clusters, and attractors."""
    prof = data["profile"]
    curr_t = data["current_trajectory"]
    trajs = data.get("trajectories") or []
    champs = data.get("champions_by_trajectory") or {}
    f_attr = data.get("failure_attractors") or []
    mech_mat = data.get("mechanism_matrix") or {}
    ncd_mat = data.get("champion_ncd_matrix") or {}

    lines = [
        f"# Sugarscape Auto-Ablation: Behavioral Clustering & Attractor Report (`{prof}`)",
        "",
        f"- **Updated**: `{data['updated_at']}`",
        f"- **Active Trajectory**: `Trajectory #{curr_t}` (`{data['completed_trajectories']}` plateau resets completed)",
        f"- **Total Attempts Logged**: `{data['total_attempts_logged']}` (`{data['evaluated_attempts_clustered']}` with full behavioral vectors)",
        "",
        "![Behavioral Attractor Map](./attractor_map.svg)",
        "",
        "---",
        "",
        "## 1. Multi-Trajectory Compression & Plateau Reset History",
        "",
        "Each trajectory starts from the canonical flattened Mesa Sugarscape (`canonical_strategy.py`), compresses complexity $C$ until a statistically significant plateau is reached, and resets to explore an independent ablation path.",
        "",
    ]

    if trajs or champs:
        lines.extend([
            "| Trajectory | Status | Generations | Attempts | Origin $C$ | Champion $C$ | Saved ($\\Delta C$) | NCD vs. Canonical | Modified Mechanisms | Plateau Reason |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |",
        ])
        logged_tids = set()
        for t in trajs:
            tid = t["trajectory_id"]
            logged_tids.add(tid)
            mechs_mod = ", ".join(f"{k}:{v}" for k, v in (t.get("mechanisms_modified") or {}).items()) or "none"
            ncd_c = (champs.get(f"T{tid}") or {}).get("ncd_canonical", 0.0)
            lines.append(
                f"| **T{tid}** | ✅ Plateaued & Reset | {t.get('generations_run', 0)} | {t.get('attempts_evaluated', 0)} | "
                f"`{t.get('origin_C')}` B | `{t.get('champion_C')}` B (`{t.get('champion_id')}`) | "
                f"**-{t.get('total_bytes_saved', 0)} B** | `{ncd_c:.3f}` | {mechs_mod} | {t.get('plateau_reason', '')} |"
            )
        for t_key, c_info in champs.items():
            tid = int(t_key.lstrip("T"))
            if tid not in logged_tids:
                lines.append(
                    f"| **T{tid}** | 🔄 Active | — | — | — | `{c_info['C']}` B (`{c_info['id']}`) | — | "
                    f"`{c_info.get('ncd_canonical', 0.0):.3f}` | (in progress) | active trajectory |"
                )
        lines.append("")

    # Cross-trajectory NCD matrix if >= 2 trajectories exist
    if len(ncd_mat) >= 2:
        t_keys = list(ncd_mat.keys())
        lines.extend([
            "### Cross-Trajectory Champion Convergence (Pairwise AST Normalized Compression Distance)",
            "",
            "| Champion | " + " | ".join(t_keys) + " |",
            "| :--- | " + " | ".join(":---:" for _ in t_keys) + " |",
        ])
        for tk1 in t_keys:
            row_cells = [f"`{ncd_mat[tk1][tk2]:.3f}`" for tk2 in t_keys]
            lines.append(f"| **{tk1}** | " + " | ".join(row_cells) + " |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 2. Failure Attractors (Clustered Rejection Modes)",
        "",
        "Rejected attempts are clustered in standardized behavioral response space and matched against the 17 pre-registered calibration controls/knockouts.",
        "",
    ])

    if f_attr:
        lines.extend([
            "| Cluster | Archetype | Count | Trajectories Hit | Cross-Trajectory? | Failed Features | Top Failing Statistics | Nearest Control | Targeted Mechanisms |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- |",
        ])
        for fa in f_attr:
            cross_badge = "🔥 **YES**" if fa["is_cross_trajectory_attractor"] else "No"
            trajs_str = ", ".join(f"T{t}" for t in fa["trajectories_hit"])
            feats_str = ", ".join(fa["dominant_failed_features"]) or "Stage 0/1"
            stats_str = ", ".join(fa["top_failing_tests"][:3]) or "—"
            mechs_str = ", ".join(fa["mechanisms_targeted"][:3]) or "—"
            lines.append(
                f"| **`{fa['cluster_id']}`** | {fa['archetype']} | `{fa['count']}` | {trajs_str} | {cross_badge} | "
                f"`{feats_str}` | `{stats_str}` | `{fa['nearest_control']}` | {mechs_str} |"
            )
        lines.append("")
    else:
        lines.append("*(No rejected attempts logged yet.)*\n")

    lines.extend([
        "---",
        "",
        "## 3. Mechanism Necessity & Feature Sensitivity Matrix",
        "",
        "Worst-case feature discrepancy ratio ($\\text{ratio} < 1.0$ passes equivalence; $\\ge 1.0$ fails) observed when each Sugarscape mechanism was targeted for ablation.",
        "",
        "| Mechanism | Classification | Attempts (Acc / Rej) | Trajectories Modified | Max E1 | Max E2 | Max E3 | Max E4 | Max E5 | Primary Failure Attractor |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for mid, m in mech_mat.items():
        fr = m["max_feature_ratios"]
        t_mod = ", ".join(f"T{t}" for t in m["trajectories_modified"]) or "—"
        lines.append(
            f"| **`{mid}`** | {m['classification']} | `{m['attempts']}` (`{m['accepted']}` / `{m['rejected']}`) | "
            f"{t_mod} | `{fr['E1']:.2f}` | `{fr['E2']:.2f}` | `{fr['E3']:.2f}` | `{fr['E4']:.2f}` | `{fr['E5']:.2f}` | "
            f"`{m['primary_failure_attractor']}` |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. Prompt Memory Formatter for the LLM Proposer
# ---------------------------------------------------------------------------

def format_cluster_memory_for_prompt(profile_name: str = "full") -> str:
    """
    Build a concise cross-trajectory attractor summary injected into `edit_harness.py`
    so the proposer avoids repeating known failure attractors and prioritizes under-explored
    or previously compressible mechanisms.
    """
    cluster_file = REPORTS_ROOT / profile_name / "clusters.json"
    if not cluster_file.exists():
        return "(No cross-trajectory cluster history yet.)"
    try:
        data = json.loads(cluster_file.read_text(encoding="utf-8"))
    except Exception:
        return "(Cluster history unreadable.)"

    out: list[str] = []
    trajs = data.get("trajectories") or []
    if trajs:
        out.append(f"Completed Compression Trajectories: {len(trajs)} (current: Trajectory #{data.get('current_trajectory', 1)})")
        for t in trajs[-3:]:
            mods = ", ".join(f"{k}({v})" for k, v in (t.get("mechanisms_modified") or {}).items()) or "refactor-only"
            out.append(
                f"  - Trajectory T{t['trajectory_id']}: C {t.get('origin_C')} -> {t.get('champion_C')} bytes "
                f"(-{t.get('total_bytes_saved')}B) | modified: {mods}"
            )

    f_attr = [fa for fa in (data.get("failure_attractors") or []) if not str(fa.get("cluster_id", "")).startswith("G")]
    if f_attr:
        out.append("Known Behavioral Failure Attractors (AVOID repeating naive removals that land here):")
        for fa in f_attr[:4]:
            out.append(
                f"  - [{fa['cluster_id']}] {fa['archetype']} (hit {fa['count']}x across trajectories {fa['trajectories_hit']}): "
                f"breaks {', '.join(fa['dominant_failed_features'])} on {', '.join(fa['top_failing_tests'][:2])} "
                f"when targeting {', '.join(fa['mechanisms_targeted'][:2])}"
            )

    mech_mat = data.get("mechanism_matrix") or {}
    untested = [k for k, v in mech_mat.items() if v.get("attempts", 0) == 0]
    compressible = [k for k, v in mech_mat.items() if v.get("classification") == "compressible"]
    if compressible:
        out.append(f"Previously Compressible Mechanisms (succeeded in at least one trajectory): {', '.join(compressible)}")
    if untested:
        out.append(f"Untested Mechanisms (high exploration value): {', '.join(untested)}")

    return "\n".join(out) if out else "(No behavioral clusters formed yet.)"


def main():
    ap = argparse.ArgumentParser(description="Cluster ablation attempts and generate attractor reports")
    ap.add_argument("--profile", default="full", help="Judge profile name (default: full)")
    args = ap.parse_args()
    summary = analyze_clusters(args.profile)
    print(
        f"[cluster_analysis] Profile '{args.profile}': clustered {summary['evaluated_attempts_clustered']} "
        f"evaluated attempts across {summary['current_trajectory']} trajectory(s). "
        f"Report written to reports/{args.profile}/CLUSTER_REPORT.md"
    )


if __name__ == "__main__":
    main()
