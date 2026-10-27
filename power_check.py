#!/usr/bin/env python3
"""
Design-stage power check: canonical Sugarscape against itself.

Splits the seeds at each screening-bank point into a "candidate" half and a
"reference" half, so the two panels come from the same model, and runs the full
test battery at the final-look size on random subsets of the bank. A judge that
cannot accept a model compared against itself cannot accept a faithful
re-implementation either. Each side gets half the bank's seeds (2 vs 2 with the
full profile), which is noisier than real use (2 candidate vs 4 reference
seeds), so the pass rate printed here is conservative.

No simulation is run; it needs a calibrated reference (calibrate_judge.py).
Writes judge/reference/<profile>/POWER.md.

Usage:
    python power_check.py --profile full --replications 12
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from judge import surface
from judge.design import load_profile
from judge.domain import subset_order
from judge.pipeline import bank_points, load_reference, reference_dir


def main():
    ap = argparse.ArgumentParser(description="Canonical-vs-itself power check of the judge.")
    ap.add_argument("--profile", default="full")
    ap.add_argument("--replications", type=int, default=12)
    args = ap.parse_args()
    profile = load_profile(args.profile)
    ref, margins, _ = load_reference(profile.name)
    pts_all = bank_points(ref)
    runs = ref["bank"]["runs"]
    n_pts = min(profile["look_points"][-1], len(pts_all))
    alpha = profile["look_alpha"][-1]
    Z = {p.id: p.z for p in pts_all}
    half = profile["bank_seeds"] // 2

    passes, nonpass = 0, Counter()
    lines = []
    for rep in range(args.replications):
        rng = np.random.default_rng(rep)
        pts = [pts_all[i] for i in subset_order(len(pts_all), f"power:{rep}")[:n_pts]]
        cand, refp = {}, {}
        for p in pts:
            for w in profile.worlds:
                k = surface.key(p.id, w)
                perm = rng.permutation(len(runs[k]))
                cand[k] = [runs[k][i] for i in perm[:half]]
                refp[k] = [runs[k][i] for i in perm[half:2 * half]]
        a = surface.assess(cand, refp, [p.id for p in pts], Z, margins, alpha, profile, seed=rep)
        passes += a.all_pass
        bad = [t for t in a.tests if t.status != "pass"]
        for t in bad:
            nonpass[f"{t.kind} {t.world} {t.stat} {t.term}".strip()] += 1
        lines.append(f"| {rep} | {'yes' if a.all_pass else 'no'} | {len(a.tests)} | "
                     f"{sum(t.status == 'undecided' for t in a.tests)} | {sum(t.status == 'fail' for t in a.tests)} |")
        print(f"replication {rep}: all pass = {a.all_pass}, non-pass tests = {len(bad)}", flush=True)

    rate = passes / args.replications
    out = [f"# Power check: canonical against itself (profile `{profile.name}`)", "",
           f"Each replication compares {half} seeds against {half} other seeds of canonical Sugarscape at "
           f"{n_pts} random screening-bank points per world, with the final-look alpha ({alpha}). "
           "Real candidates are compared against all reference seeds, so real power is higher.", "",
           f"**All tests passed in {passes} of {args.replications} replications ({rate:.0%}).**", "",
           "| replication | all pass | tests | undecided | fail |", "|---|---|---|---|---|", *lines, "",
           "Tests that did not pass, by frequency:", ""]
    out += [f"- {k}: {v}" for k, v in nonpass.most_common()] or ["- none"]
    path = reference_dir(profile.name) / "POWER.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"all-pass rate {rate:.2f} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
