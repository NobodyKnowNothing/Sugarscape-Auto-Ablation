#!/usr/bin/env python3
"""
Calibrate the emergence judge before trusting it.

1. Reference ensemble: canonical Sugarscape at every design point (reference seeds).
2. Margins: for every statistic and point, the largest shift (upper confidence
   bound) that the innocuous perturbations produce, floored at the design's
   minimum margin. Response-surface (E5) margins are derived the same way.
3. Separation: statistics that no knockout / null society moves clearly beyond
   their margin are flagged non-discriminative and dropped from the test set.
4. Controls: held-out positive controls must PASS every feature in >= 95% of
   replications; knockouts and null societies must FAIL their targeted features
   in >= 95% of replications.

Writes judge/reference/<profile>/{reference,margins,calibration}.json.
The research loop refuses to run unless this calibration is fresh and passing.

Usage:
    python calibrate_judge.py --profile smoke
    python calibrate_judge.py --profile full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from judge import judge_hash
from judge import complexity as cx
from judge.controls import VARIANTS, by_role, canonical_code, variant_code
from judge.design import design_hash, load_profile
from judge.features import FEATURES, SCALAR_STATS, VECTOR_STATS, presence
from judge.parallel import shutdown
from judge.pipeline import assess, collect, reference_dir
from judge.stats import (applicable, effect_compare, margin_from_perturbations,
                         scalar_compare, vector_applicable, vector_compare)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build_reference(profile, points):
    log(f"Reference ensemble: canonical x {len(points)} points x {profile['n_reference']} seeds")
    runs = collect(canonical_code(), profile, "reference", "canonical", points, profile["n_reference"])
    ref = {"points": {}}
    for p in points:
        pres = {f: presence(f, runs[p], profile.alpha, strict=True)[0] for f in FEATURES}
        ref["points"][p] = {"runs": runs[p], "presence": pres, "seeds": [r["_seed"] for r in runs[p]]}
    return ref


def build_margins(profile, ref, points, pert_runs):
    floors = profile.margin_floor
    alpha = profile.alpha
    margins = {"scalar": {}, "vector": {}, "effect": {}, "active": {}, "perturbations": sorted(pert_runs)}
    for p in points:
        margins["scalar"][p], margins["vector"][p] = {}, {}
        rruns = ref["points"][p]["runs"]
        for f in FEATURES:
            for s in SCALAR_STATS[f]:
                rv = [r[s] for r in rruns]
                if not applicable(rv):
                    continue
                comps = [scalar_compare([r[s] for r in pr[p]], rv, alpha) for pr in pert_runs.values()]
                margins["scalar"][p][s] = margin_from_perturbations(comps, floors["scalar_sigma"])
            for s in VECTOR_STATS[f]:
                if not vector_applicable([r[s] for r in rruns]):
                    continue
                comps = [vector_compare([r[s] for r in pr[p]], [r[s] for r in rruns], alpha) for pr in pert_runs.values()]
                margins["vector"][p][s] = margin_from_perturbations(comps, floors["vector_sigma"])
    if "default" in points:
        r0 = ref["points"]["default"]["runs"]
        for p in points:
            if p == "default":
                continue
            margins["effect"][p] = {}
            rx = ref["points"][p]["runs"]
            for f in FEATURES:
                for s in SCALAR_STATS[f]:
                    if not (applicable([r[s] for r in rx]) and applicable([r[s] for r in r0])):
                        continue
                    comps = [effect_compare([r[s] for r in pr[p]], [r[s] for r in pr["default"]],
                                            [r[s] for r in rx], [r[s] for r in r0], alpha)
                             for pr in pert_runs.values()]
                    margins["effect"][p][s] = margin_from_perturbations(comps, floors["effect_sigma"])
    return margins


def separation(profile, ref, margins, neg_runs, points):
    """Which statistics does at least one negative/null control move clearly beyond margin?"""
    alpha = profile.alpha
    hits = {s: [] for f in FEATURES for s in SCALAR_STATS[f] + VECTOR_STATS[f]}
    for name, runs in neg_runs.items():
        for p in points:
            rruns = ref["points"][p]["runs"]
            for s, m in margins["scalar"][p].items():
                if scalar_compare([r[s] for r in runs[p]], [r[s] for r in rruns], alpha).clearly_different(m):
                    hits[s].append(f"{name}@{p}")
            for s, m in margins["vector"][p].items():
                if vector_compare([r[s] for r in runs[p]], [r[s] for r in rruns], alpha).clearly_different(m):
                    hits[s].append(f"{name}@{p}")
            if p != "default" and p in margins["effect"]:
                for s, m in margins["effect"][p].items():
                    c = effect_compare([r[s] for r in runs[p]], [r[s] for r in runs["default"]],
                                       [r[s] for r in rruns], [r[s] for r in ref["points"]["default"]["runs"]], alpha)
                    if c.clearly_different(m):
                        hits[s].append(f"{name}@{p}(effect)")
    return hits


def main():
    ap = argparse.ArgumentParser(description="Calibrate the emergence judge.")
    ap.add_argument("--profile", default="smoke")
    args = ap.parse_args()
    profile = load_profile(args.profile)
    t0 = time.time()
    out_dir = reference_dir(profile.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_points = list(dict.fromkeys(profile.screening + profile.audit_points))
    screen = profile.screening
    alpha = profile.alpha

    ref = build_reference(profile, all_points)
    absent = {p: [f for f, ok in v["presence"].items() if not ok] for p, v in ref["points"].items()}
    log("Canonical presence (features absent per point): " + json.dumps({p: a for p, a in absent.items() if a}))

    log(f"Innocuous perturbations: {by_role('margin')} x {profile['n_perturbation']} seeds")
    pert_runs = {name: collect(variant_code(name), profile, "perturbation", name, all_points, profile["n_perturbation"])
                 for name in by_role("margin")}
    margins = build_margins(profile, ref, all_points, pert_runs)

    # Controls: replications on the screening design
    R, block = profile["control_replications"], profile["control_block"]
    control_names = by_role("positive") + by_role("negative") + by_role("null")
    log(f"Controls: {control_names} x {R} replication(s) x {block} seeds x {len(screen)} points")
    control_runs = {}
    for name in control_names:
        code = variant_code(name)
        control_runs[name] = [collect(code, profile, "control", f"{name}#{k}", screen, block) for k in range(R)]

    # Separation check on pooled negative/null data
    pooled = {}
    for name in by_role("negative") + by_role("null"):
        pooled[name] = {p: [r for rep in control_runs[name] for r in rep[p]] for p in screen}
    hits = separation(profile, ref, margins, pooled, screen)
    margins["active"] = {s: bool(h) for s, h in hits.items()}
    margins["separation_hits"] = hits
    dropped = [s for s, ok in margins["active"].items() if not ok]
    log(f"Non-discriminative statistics dropped: {dropped or 'none'}")

    # Control outcomes
    outcomes = {}
    for name in control_names:
        spec = VARIANTS[name]
        reps = []
        for rep in control_runs[name]:
            fs = assess(rep, ref, margins, screen, alpha).feature_status()
            if spec["role"] == "positive":
                ok = all(v == "pass" for v in fs.values())
            else:
                ok = all(fs[t] != "pass" for t in spec["targets"])
            reps.append({"features": fs, "as_expected": ok})
        rate = sum(r["as_expected"] for r in reps) / len(reps)
        outcomes[name] = {"role": spec["role"], "doc": spec["doc"], "targets": spec.get("targets", []),
                          "rate_as_expected": rate, "replications": reps}
        log(f"  {name:24s} {spec['role']:8s} as-expected rate {rate:.2f}  last={reps[-1]['features']}")

    pos = [o["rate_as_expected"] for o in outcomes.values() if o["role"] == "positive"]
    neg = [o["rate_as_expected"] for o in outcomes.values() if o["role"] in ("negative", "null")]
    pos_rate, neg_rate = min(pos), min(neg)
    passed = pos_rate >= profile["required_positive_pass_rate"] and neg_rate >= profile["required_negative_fail_rate"]

    canon_c = cx.measure(canonical_code(), profile)
    calib = {
        "profile": profile.name,
        "passed": passed,
        "min_positive_pass_rate": pos_rate,
        "min_negative_fail_rate": neg_rate,
        "required": {"positive": profile["required_positive_pass_rate"], "negative": profile["required_negative_fail_rate"]},
        "controls": outcomes,
        "dropped_statistics": dropped,
        "canonical_complexity": {k: v for k, v in canon_c.items() if k != "dependency_manifest"},
        "judge_hash": judge_hash(),
        "design_hash": design_hash(),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
        "note": profile.cfg.get("comment", ""),
    }
    ref.update({"profile": profile.name, "judge_hash": calib["judge_hash"], "design_hash": calib["design_hash"]})
    (out_dir / "reference.json").write_text(json.dumps(ref), encoding="utf-8")
    (out_dir / "margins.json").write_text(json.dumps(margins, indent=1), encoding="utf-8")
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=1), encoding="utf-8")
    shutdown()

    log(f"Positive controls: min pass rate {pos_rate:.2f} (need {profile['required_positive_pass_rate']})")
    log(f"Negative/null controls: min fail rate {neg_rate:.2f} (need {profile['required_negative_fail_rate']})")
    log(f"Calibration {'PASSED' if passed else 'FAILED'} in {calib['seconds']}s -> {out_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
