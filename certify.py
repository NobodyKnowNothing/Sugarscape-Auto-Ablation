#!/usr/bin/env python3
"""
Irreducibility certificate for a model.

For every rule the ledger lists as surviving, run a knockout of that rule and
show that it breaks at least one emergent feature: some statistic's difference
from canonical Sugarscape exceeds its margin with confidence (one-sided tests,
Holm-corrected across all statistics, points and features, because the claim
is "at least one feature breaks"). Removed rules link to their passing evidence.

Knockout variants come from:
  --canonical          the judge's canonical knockouts (certifies the starting model)
  --knockouts DIR      files DIR/<mechanism_id>.py written for an evolved model
  --generate           ask the LLM to write DIR/<mechanism_id>.py for each surviving
                       rule of --model (requires GOOGLE_API_KEY), then evaluate them

Usage:
    python certify.py --profile smoke --canonical
    python certify.py --profile full --model strategy.py --knockouts knockouts/ --generate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from judge import judge_hash
from judge.controls import variant_code
from judge.contract import code_hash
from judge.design import load_profile
from judge.features import FEATURE_NAMES, FEATURES, SCALAR_STATS, VECTOR_STATS
from judge.parallel import shutdown
from judge.pipeline import RunError, collect, load_reference
from judge.stats import (applicable, holm, nonequivalence_p, scalar_compare,
                         vector_applicable, vector_compare)

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "mechanisms.json"


def knockout_evidence(code: str, profile, ref, margins) -> dict:
    """Holm-corrected evidence that `code` differs from canonical beyond margin somewhere."""
    points = profile.screening
    try:
        runs = collect(code, profile, "holdout", "ko-" + code_hash(code), points, profile["n_holdout"])
    except RunError as e:
        return {"breaks": True, "why": f"model fails to run: {e}", "hits": []}
    tests = []
    for p in points:
        rr = ref["points"][p]["runs"]
        for f in FEATURES:
            for s in SCALAR_STATS[f]:
                rv = [r[s] for r in rr]
                if not margins["active"].get(s, True) or not applicable(rv) or s not in margins["scalar"][p]:
                    continue
                c = scalar_compare([r[s] for r in runs[p]], rv, profile.alpha)
                tests.append((f, p, s, nonequivalence_p(c, margins["scalar"][p][s]), c.direction()))
            for s in VECTOR_STATS[f]:
                rv = [r[s] for r in rr]
                if s not in margins["vector"][p] or not vector_applicable(rv):
                    continue
                c = vector_compare([r[s] for r in runs[p]], rv, profile.alpha)
                tests.append((f, p, s, nonequivalence_p(c, margins["vector"][p][s]), c.direction()))
    rejects = holm([t[3] for t in tests], profile.alpha)
    hits = [{"feature": f, "point": p, "stat": s, "p": round(pv, 6), "direction": d}
            for (f, p, s, pv, d), rej in zip(tests, rejects) if rej]
    broken = sorted({h["feature"] for h in hits})
    return {"breaks": bool(hits), "features_broken": broken,
            "why": ", ".join(f"{f} ({FEATURE_NAMES[f]})" for f in broken) or "no feature shown to break",
            "hits": hits[:25], "n_tests": len(tests)}


def generate_knockouts(model_code: str, surviving: dict, out_dir: Path) -> None:
    from edit_harness import generate_knockout_variant
    out_dir.mkdir(parents=True, exist_ok=True)
    for mid, info in surviving.items():
        path = out_dir / f"{mid}.py"
        if path.exists():
            continue
        code = generate_knockout_variant(model_code, mid, info["rule"])
        if code:
            path.write_text(code, encoding="utf-8")
            print(f"  generated knockout for {mid}")
        else:
            print(f"  could not generate knockout for {mid}")


def main():
    ap = argparse.ArgumentParser(description="Knockout certificate for surviving mechanisms.")
    ap.add_argument("--profile", default="smoke")
    ap.add_argument("--model", default=str(ROOT / "strategy.py"))
    ap.add_argument("--canonical", action="store_true", help="use the judge's canonical knockouts")
    ap.add_argument("--knockouts", default=str(ROOT / "knockouts"))
    ap.add_argument("--generate", action="store_true", help="LLM-generate missing knockout files")
    ap.add_argument("--out", default=str(ROOT / "certificate.json"))
    args = ap.parse_args()

    profile = load_profile(args.profile)
    ref, margins, calib = load_reference(profile.name)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    mechs = ledger["mechanisms"]
    surviving = {k: v for k, v in mechs.items() if v["status"] == "kept"}
    model_code = Path(args.model).read_text(encoding="utf-8")

    if args.generate and not args.canonical:
        generate_knockouts(model_code, surviving, Path(args.knockouts))

    cert = {"model": Path(args.model).name, "model_hash": code_hash(model_code), "profile": profile.name,
            "judge_hash": judge_hash(), "calibration_passed": calib.get("passed"),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mechanisms": {}}
    for mid, info in mechs.items():
        entry = {"rule": info["rule"], "status": info["status"]}
        if info["status"] != "kept":
            entry["evidence"] = info.get("evidence", [])
            cert["mechanisms"][mid] = entry
            continue
        if args.canonical:
            ko = info.get("canonical_knockout")
            code = variant_code(ko) if ko else None
        else:
            path = Path(args.knockouts) / f"{mid}.py"
            code = path.read_text(encoding="utf-8") if path.exists() else None
        if code is None:
            entry["certified"] = None
            entry["why"] = "no knockout variant available"
        else:
            print(f"Knockout {mid} ...", flush=True)
            ev = knockout_evidence(code, profile, ref, margins)
            entry["certified"] = ev["breaks"]
            entry.update(ev)
            print(f"  -> {'NECESSARY' if ev['breaks'] else 'not shown necessary'}: {ev['why']}")
        cert["mechanisms"][mid] = entry
    shutdown()
    Path(args.out).write_text(json.dumps(cert, indent=1), encoding="utf-8")
    kept = [m for m in cert["mechanisms"].values() if m["status"] == "kept"]
    print(f"\nCertificate written to {args.out}")
    print(f"  kept rules: {len(kept)}; shown necessary: {sum(1 for m in kept if m.get('certified'))}; "
          f"not shown necessary: {sum(1 for m in kept if m.get('certified') is False)}; "
          f"no knockout: {sum(1 for m in kept if m.get('certified') is None)}")
    if not calib.get("passed"):
        print("  WARNING: the judge calibration for this profile did not pass; the certificate is not scientifically valid.")


if __name__ == "__main__":
    main()
