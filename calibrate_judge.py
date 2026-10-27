#!/usr/bin/env python3
"""
Calibrate the emergence judge before trusting it.

1. Reference banks: canonical Sugarscape at a scrambled-Sobol' sample of the input
   domain (screening bank) and at an i.i.d. sample (holdout bank, audits only),
   every point in both trade worlds; plus the default point (presence anchor).
2. Canonical trait responses: how every statistic responds to every input over
   the domain, the trade contrast, and whether each literature-backed response
   (Epstein & Axtell's figures) holds in canonical G1MT over this domain.
3. Margins: for every test, max(floor, relative floor x canonical's own response,
   largest upper confidence bound produced by an innocuous perturbation).
4. Separation: discrepancy statistics that no knockout or null society moves
   clearly beyond margin are flagged non-discriminative and dropped.
5. Controls: held-out positive controls must PASS (>= 95% of replications);
   knockouts and null societies must be REJECTED with every targeted feature
   failing. "Ignore an input" knockouts count only for inputs canonical responds
   to; probes are reported, never part of the criterion.

Writes judge/reference/<profile>/{reference,margins,calibration}.json and CALIBRATION.md.
The research loop refuses to run unless this calibration is fresh and passing.

Usage:
    python calibrate_judge.py --profile smoke
    JUDGE_WORKERS=4 python calibrate_judge.py --profile full
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scipy import stats as sps

from judge import judge_hash
from judge import complexity as cx
from judge import surface
from judge.controls import VARIANTS, by_role, canonical_code, variant_code
from judge.design import design_hash, load_profile
from judge.domain import subset_order
from judge.features import FEATURES, SCALAR_STATS, TRADE_ONLY, presence, world_stats
from judge.parallel import shutdown
from judge.pipeline import collect, collect_points, reference_dir


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _round(x):
    if isinstance(x, float):
        return float(f"{x:.6g}") if math.isfinite(x) else x
    if isinstance(x, list):
        return [_round(v) for v in x]
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return x


def subset(points, n, key):
    order = subset_order(len(points), key)
    return [points[i] for i in order[:min(n, len(points))]]


def scalar_items(world):
    return [s for f, items in world_stats(world == "trade").items() for s, k in items if k == "scalar"]


# ---------------------------------------------------------------------------
# Canonical responses and margins
# ---------------------------------------------------------------------------

def canonical_responses(profile, bank_runs, pts, sc):
    pids, Z = [p.id for p in pts], {p.id: p.z for p in pts}
    terms = profile.domain.regressor_names
    slope, contrast = {}, {}
    for w in profile.worlds:
        slope[w] = {}
        for s in scalar_items(w):
            r = surface.response(bank_runs, pids, Z, w, s, sc[w][s]) if math.isfinite(sc[w][s]) else None
            slope[w][s] = {t: [r["b"][j], r["se"][j]] for j, t in enumerate(terms)} if r else {}
    for s in surface.contrast_stats():
        both = math.sqrt((sc["trade"][s] ** 2 + sc["notrade"][s] ** 2) / 2)
        r = surface.own_contrast(bank_runs, pids, s, both) if math.isfinite(both) else None
        if r:
            contrast[s] = [r["mean"], r["se"]]
    named = []
    for nr in profile.named_responses:
        t = surface.named_test({**nr}, bank_runs, pids, Z, terms, sc, profile.alpha)
        named.append({**nr, "confirmed": t.status == "pass", "canon_est": t.est, "canon_lo": t.lo,
                      "canon_hi": t.hi})
    return {"slope": slope, "contrast": contrast}, named


def floor_margins(profile, sc, canon):
    fl = profile.margin_floor
    terms = profile.domain.regressor_names
    m = {"scales": sc, "terms": terms, "l2": {}, "defined": {}, "spread": {}, "slope": {}, "contrast": {},
         "canon": canon, "active": {}}
    for w in profile.worlds:
        m["l2"][w], m["defined"][w], m["spread"][w], m["slope"][w] = {}, {}, {}, {}
        for f, items in world_stats(w == "trade").items():
            for s, kind in items:
                m["l2"][w][s] = fl["indicator_rms"] if kind == "indicator" else fl["l2_sigma"]
                if kind == "scalar":
                    m["defined"][w][s] = fl["defined_rms"]
                    m["spread"][w][s] = fl["spread_ratio"]
                    cb = canon["slope"][w].get(s, {})
                    m["slope"][w][s] = {t: max(fl["slope_sigma"], fl["slope_rel"] * abs(cb.get(t, [0.0])[0]))
                                        for t in terms}
    for s, (cmean, _) in canon["contrast"].items():
        m["contrast"][s] = max(fl["contrast_sigma"], fl["contrast_rel"] * abs(cmean))
    return m


def widen_by_perturbations(margins, pert_assessments):
    """Margins become the largest upper confidence bound an innocuous perturbation produces (if above floor)."""
    src = {}
    for name, a in pert_assessments.items():
        for t in a.tests:
            if not (math.isfinite(t.lo) and math.isfinite(t.hi)):
                continue
            if t.kind in ("l2", "defined"):
                tab, k, ucb = margins[t.kind][t.world], t.stat, t.hi
            elif t.kind == "spread":
                tab, k, ucb = margins["spread"][t.world], t.stat, math.exp(max(abs(t.lo), abs(t.hi)))
            elif t.kind == "slope":
                tab, k, ucb = margins["slope"][t.world][t.stat], t.term, max(abs(t.lo), abs(t.hi))
            elif t.kind == "contrast":
                tab, k, ucb = margins["contrast"], t.stat, max(abs(t.lo), abs(t.hi))
            else:
                continue
            if k in tab and ucb > tab[k]:
                tab[k] = ucb
                src[f"{t.kind}:{t.world}:{t.stat}:{t.term}"] = name
    margins["widened_by"] = src
    return margins


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def run_control(profile, name, pts, rep):
    code = variant_code(name)
    return collect_points(code, profile, "control", f"{name}#{rep}", pts, profile["candidate_seeds"])


def oc_pass_probability(a: surface.Assessment, alpha: float) -> tuple[float, float]:
    """
    Plug-in operating characteristic: estimated probability that a fresh replication
    passes every test (independence approximation), and the smallest slack in SE units.
    """
    z = sps.norm.ppf(1 - alpha)
    logp, slack_min = 0.0, math.inf
    for t in a.tests:
        if t.kind == "named" or not (t.margin > 0):
            continue
        se = t.se(alpha)
        if not (se > 0 and math.isfinite(se)):
            continue
        theta = t.est if t.kind in ("l2", "defined") else abs(t.est)
        slack = (t.margin - theta) / se - z
        slack_min = min(slack_min, slack)
        logp += math.log(max(sps.norm.cdf(slack), 1e-300))
    return math.exp(logp), slack_min


def input_matters(canon, margins, terms):
    """Does canonical respond to any of these regressors beyond margin (with confidence)?"""
    hits = []
    for w, tab in canon["slope"].items():
        for s, coefs in tab.items():
            for t in terms:
                if t in coefs:
                    b, se = coefs[t]
                    m = margins["slope"][w][s][t]
                    if abs(b) - 2 * se > m:
                        hits.append(f"{s}~{t} ({w})")
    return hits


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(path, profile, calib, margins, canon, named):
    terms = margins["terms"]
    lines = [f"# Judge calibration: profile `{profile.name}`", "",
             f"- Result: **{'PASSED' if calib['passed'] else 'FAILED'}** "
             f"(positive controls min pass rate {calib['min_positive_pass_rate']:.2f}, "
             f"negative/null controls min fail rate {calib['min_negative_fail_rate']:.2f}; "
             f"required {calib['required']['positive']:.2f} / {calib['required']['negative']:.2f})",
             f"- Judge hash `{calib['judge_hash']}`, design hash `{calib['design_hash']}`, created {calib['created']}, "
             f"{calib['seconds'] / 3600:.2f} h",
             f"- Screening bank: {calib['banks']['bank_points']} scrambled-Sobol' points x 2 worlds x "
             f"{calib['banks']['bank_seeds']} seeds; holdout bank: {calib['banks']['holdout_points']} i.i.d. points x 2 x "
             f"{calib['banks']['holdout_seeds']}", "",
             "## Literature-backed responses in canonical G1MT over the domain", "",
             "Estimates are signed so that positive means the claimed direction: SD units across the input's "
             "whole range for responses, SD units for the trade effect, raw units (log dispersion ratio) for IV-3.", "",
             "| id | claim | source | canonical estimate | 95% one-sided bound | holds |", "|---|---|---|---|---|---|"]
    for nr in named:
        lines.append(f"| {nr['id']} | {nr['claim']} | {nr['source']} | {nr['canon_est']:+.2f} | "
                     f"{nr['canon_lo']:+.2f} | {'yes' if nr['confirmed'] else 'NO'} |")
    lines += ["", "## Canonical trait-response map", "",
              "Change of each statistic across the full range of each input, in units of its typical "
              "seed-to-seed SD (linear fit over the screening bank). Required to be reproduced within the "
              "slope margins.", ""]
    for w in ("trade", "notrade"):
        lines += [f"### {'With' if w == 'trade' else 'Without'} trade", "",
                  "| statistic | " + " | ".join(terms) + " |", "|---" * (len(terms) + 1) + "|"]
        for s, coefs in canon["slope"][w].items():
            if coefs:
                lines.append(f"| {s} | " + " | ".join(f"{coefs[t][0]:+.1f}" for t in terms) + " |")
        lines.append("")
    lines += ["### Trade contrast (trade minus no trade, domain average, in SD)", "",
              "| statistic | effect | margin |", "|---|---|---|"]
    for s, (cm, cse) in canon["contrast"].items():
        lines.append(f"| {s} | {cm:+.2f} ± {cse:.2f} | {margins['contrast'][s]:.2f} |")
    lines += ["", "## Controls", "", "| control | role | expected | outcome | features | est. pass prob. (positives) |",
              "|---|---|---|---|---|---|"]
    for name, o in calib["controls"].items():
        feats = " ".join(f"{k}={v[0]}" for k, v in o["replications"][-1]["features"].items())
        exp = {"positive": "accept", "margin": "-", "probe": "report"}.get(o["role"], "reject " + ",".join(o["targets"]))
        out = "as expected" if o.get("rate_as_expected", 0) == 1 else (
            "reported" if o["role"] == "probe" else f"rate {o.get('rate_as_expected', 0):.2f}")
        ocp = f"{o['oc_pass_probability']:.3f} (min slack {o['oc_min_slack']:.1f} SE)" if o["role"] == "positive" else ""
        lines.append(f"| {name} | {o['role']} | {exp} | {out} | {feats} | {ocp} |")
    lines += ["", f"Statistics dropped as non-discriminative: {', '.join(calib['dropped_statistics']) or 'none'}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Calibrate the emergence judge.")
    ap.add_argument("--profile", default="smoke")
    args = ap.parse_args()
    profile = load_profile(args.profile)
    alpha = profile.alpha
    t0 = time.time()
    out_dir = reference_dir(profile.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    canon_code = canonical_code()
    worlds = list(profile.worlds)
    alpha_look = profile["look_alpha"][-1]     # controls are judged as at the final look
    n_k = profile["candidate_seeds"]

    # 1. Reference banks ------------------------------------------------------
    bank, hold = profile.bank(), profile.holdout()
    log(f"Screening bank: canonical x {len(bank)} Sobol' points x {len(worlds)} worlds x {profile['bank_seeds']} seeds")
    bank_runs = collect_points(canon_code, profile, "reference", "bank", bank, profile["bank_seeds"])
    log(f"Holdout bank: canonical x {len(hold)} i.i.d. points x {len(worlds)} worlds x {profile['holdout_seeds']} seeds")
    hold_runs = collect_points(canon_code, profile, "reference", "holdout", hold, profile["holdout_seeds"])
    anchor = collect(canon_code, profile, "reference", "anchor", ["default"], profile["anchor_seeds"])["default"]
    anchor_presence = {f: presence(f, anchor, alpha, strict=True)[0] for f in FEATURES}
    log(f"Anchor presence at the default point: {anchor_presence}")
    ref = {"bank": {"points": [{"id": p.id, "params": p.params, "z": list(p.z)} for p in bank], "runs": bank_runs},
           "holdout": {"points": [{"id": p.id, "params": p.params, "z": list(p.z)} for p in hold], "runs": hold_runs},
           "anchor": {"runs": anchor, "presence": anchor_presence}}

    # 2. Scales and canonical responses -----------------------------------------
    pids = [p.id for p in bank]
    sc = surface.scales(bank_runs, pids, worlds)
    canon, named = canonical_responses(profile, bank_runs, bank, sc)
    for nr in named:
        unit = "(raw units)" if nr["term"] == "level" else "SD"
        log(f"  {nr['id']:12s} {nr['claim']:55s} canonical {nr['canon_est']:+.2f} {unit} "
            f"(lower bound {nr['canon_lo']:+.2f}) -> {'holds' if nr['confirmed'] else 'NOT confirmed'}")

    # 3. Margins from innocuous perturbations -----------------------------------------
    margins = floor_margins(profile, sc, canon)
    margins["named"] = named
    Zb = {p.id: p.z for p in bank}
    pert = {}
    for name in by_role("margin"):
        pts = subset(bank, profile["perturbation_points"], "perturbation:" + name)
        log(f"Perturbation {name}: {len(pts)} points x {len(worlds)} worlds x {n_k} seeds")
        runs = collect_points(variant_code(name), profile, "perturbation", name, pts, n_k)
        pert[name] = surface.assess(runs, bank_runs, [p.id for p in pts], Zb, margins, alpha_look, profile)
    margins = widen_by_perturbations(margins, pert)
    margins["perturbations"] = sorted(pert)

    # 4. Controls ----------------------------------------------------------------------
    control_runs = {}
    for name in by_role("positive"):
        reps = []
        for k in range(profile["positive_replications"]):
            pts = subset(bank, profile["positive_points"], f"control:{name}#{k}")
            log(f"Positive control {name} #{k}: {len(pts)} points")
            reps.append((pts, run_control(profile, name, pts, k)))
        control_runs[name] = reps
    for name in by_role("negative") + by_role("null") + by_role("probe"):
        pts = subset(bank, profile["negative_points"], f"control:{name}")
        log(f"{VARIANTS[name]['role'].capitalize()} control {name}: {len(pts)} points")
        control_runs[name] = [(pts, run_control(profile, name, pts, 0))]

    def assess_rep(pts, runs):
        return surface.assess(runs, bank_runs, [p.id for p in pts], Zb, margins, alpha_look, profile)

    # Separation: which discrepancy statistics does some negative/null control clearly move?
    hits = {}
    for name in by_role("negative") + by_role("null"):
        pts, runs = control_runs[name][0]
        for t in assess_rep(pts, runs).tests:
            if t.kind == "l2":
                hits.setdefault(surface.key(t.stat, t.world), [])
                if t.status == "fail":
                    hits[surface.key(t.stat, t.world)].append(name)
    margins["active"] = {k: bool(v) for k, v in hits.items()}
    margins["separation_hits"] = hits
    dropped = sorted(k for k, v in margins["active"].items() if not v)
    log(f"Non-discriminative statistics dropped: {dropped or 'none'}")

    # Outcomes
    outcomes = {}
    for name, reps in control_runs.items():
        spec = VARIANTS[name]
        role = spec["role"]
        targets = list(spec.get("targets", []))
        note = ""
        if role == "negative" and spec.get("terms"):
            matters = input_matters(canon, margins, spec["terms"])
            if not matters:
                role, note = "probe", "canonical does not respond to this input beyond margin: input not required"
            else:
                note = "canonical responds: " + ", ".join(matters[:4])
        res = []
        for pts, runs in reps:
            a = assess_rep(pts, runs)
            fs = a.feature_status()
            if role == "positive":
                ok = a.all_pass
            else:
                ok = (not a.all_pass) and all(fs[t] != "pass" for t in targets)
            p_oc, slack = oc_pass_probability(a, alpha_look)
            res.append({"features": {k: [v, sum(1 for t in a.tests if t.feature == k and t.status != "pass")]
                                     for k, v in fs.items()},
                        "as_expected": ok, "oc_pass_probability": p_oc, "oc_min_slack": slack,
                        "failing_tests": [t.to_dict() for t in a.tests if t.status != "pass"][:40]})
        rate = sum(r["as_expected"] for r in res) / len(res)
        outcomes[name] = {"role": role, "doc": spec["doc"], "targets": targets, "note": note,
                          "rate_as_expected": rate, "oc_pass_probability": min(r["oc_pass_probability"] for r in res),
                          "oc_min_slack": min(r["oc_min_slack"] for r in res), "replications": res}
        log(f"  {name:24s} {role:8s} as-expected {rate:.2f}  "
            + " ".join(f"{k}={v[0]}" for k, v in res[-1]["features"].items())
            + (f"  [{note}]" if note else ""))

    pos = [o["rate_as_expected"] for o in outcomes.values() if o["role"] == "positive"]
    neg = [o["rate_as_expected"] for o in outcomes.values() if o["role"] in ("negative", "null")]
    pos_rate, neg_rate = min(pos), min(neg)
    passed = pos_rate >= profile["required_positive_pass_rate"] and neg_rate >= profile["required_negative_fail_rate"]

    canon_c = cx.measure(canon_code, profile)
    calib = {
        "profile": profile.name,
        "passed": passed,
        "min_positive_pass_rate": pos_rate,
        "min_negative_fail_rate": neg_rate,
        "required": {"positive": profile["required_positive_pass_rate"], "negative": profile["required_negative_fail_rate"]},
        "controls": outcomes,
        "named_responses": named,
        "dropped_statistics": dropped,
        "banks": {k: profile[k] for k in ("bank_points", "bank_seeds", "holdout_points", "holdout_seeds")},
        "canonical_complexity": {k: v for k, v in canon_c.items() if k != "dependency_manifest"},
        "judge_hash": judge_hash(),
        "design_hash": design_hash(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
        "note": profile.cfg.get("comment", ""),
    }
    ref.update({"profile": profile.name, "judge_hash": calib["judge_hash"], "design_hash": calib["design_hash"]})
    (out_dir / "reference.json").write_text(json.dumps(_round(ref)), encoding="utf-8")
    (out_dir / "margins.json").write_text(json.dumps(_round(margins), indent=1), encoding="utf-8")
    (out_dir / "calibration.json").write_text(json.dumps(_round(calib), indent=1), encoding="utf-8")
    write_report(out_dir / "CALIBRATION.md", profile, calib, margins, canon, named)
    shutdown()

    log(f"Positive controls: min pass rate {pos_rate:.2f} (need {profile['required_positive_pass_rate']})")
    log(f"Negative/null controls: min fail rate {neg_rate:.2f} (need {profile['required_negative_fail_rate']})")
    log(f"Calibration {'PASSED' if passed else 'FAILED'} in {calib['seconds']}s -> {out_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
