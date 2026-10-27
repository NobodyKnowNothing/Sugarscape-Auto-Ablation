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
from judge.domain import subset_order
from judge.features import FEATURE_NAMES
from judge.parallel import shutdown
from judge.pipeline import RunError, bank_points, collect_points, load_reference, screen
from judge.stats import holm

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / "mechanisms.json"


def knockout_evidence(code: str, profile, ref, margins) -> dict:
    """Holm-corrected evidence that `code` departs from canonical beyond some margin, on holdout points."""
    h = code_hash(code)
    pts_all = bank_points(ref, "holdout")
    order = subset_order(len(pts_all), "ko:" + h)
    pts = [pts_all[i] for i in order[:min(profile["audit_points"], len(pts_all))]]
    try:
        runs = collect_points(code, profile, "holdout", "ko-" + h, pts, profile["candidate_seeds"])
    except RunError as e:
        return {"breaks": True, "features_broken": [], "why": f"model fails to run: {e}", "hits": []}
    a = screen(runs, ref, margins, pts, profile.alpha, profile, which="holdout", seed=7)
    tests = [t for t in a.tests if t.kind != "named"]
    rejects = holm([t.p_exceeds(profile.alpha) for t in tests], profile.alpha)
    hits = [{"feature": t.feature, "world": t.world, "stat": t.stat, "kind": t.kind, "term": t.term,
             "p": round(t.p_exceeds(profile.alpha), 6)} for t, rej in zip(tests, rejects) if rej]
    broken = sorted({x["feature"] for x in hits})
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
