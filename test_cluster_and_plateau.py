"""Tests for behavioral vector extraction, clustering, plateau detection, and multi-trajectory resets."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from autoresearch import Archive, check_plateau, handle_plateau_reset, reset_ledger_for_new_trajectory
from cluster_analysis import (
    JOINT_DIM_NAMES,
    analyze_clusters,
    build_control_landmarks,
    cluster_vectors,
    compute_ncd,
    compute_ncd_from_canonical,
    extract_attempt_vector,
    format_cluster_memory_for_prompt,
    match_nearest_control,
)
from judge.controls import canonical_code, variant_code
from judge.design import load_profile
from judge.pipeline import JudgeResult, load_reference


def test_ncd_identities_and_separation():
    canon = canonical_code()
    assert compute_ncd_from_canonical(canon) == 0.0
    renamed = canon.replace("calculate_welfare", "w_fn")
    assert compute_ncd(canon, renamed) == 0.0
    np_impl = variant_code("numpy_reimplementation")
    ncd_np = compute_ncd(canon, np_impl)
    assert 0.25 < ncd_np <= 1.0


def test_control_landmarks_and_matching():
    ref, margins, calib = load_reference("full")
    landmarks = build_control_landmarks(margins, calib)
    assert "ko_no_trade" in landmarks
    assert "numpy_reimplementation" in landmarks
    assert len(landmarks["ko_no_trade"]["vector"]) == len(JOINT_DIM_NAMES)

    # A vector identical to ko_no_trade should match ko_no_trade as its top negative control
    m = match_nearest_control(landmarks["ko_no_trade"]["vector"], landmarks, accepted=False)
    assert m["name"] == "ko_no_trade"
    assert m["similarity"] > 0.95


def test_cluster_vectors_ward_and_edge_cases():
    assert len(cluster_vectors(np.zeros((0, 5)))) == 0
    assert list(cluster_vectors(np.ones((1, 5)))) == [1]
    assert list(cluster_vectors(np.ones((4, 5)))) == [1, 1, 1, 1]

    rng = np.random.default_rng(42)
    c1 = rng.normal(-4.0, 0.2, size=(6, 8))
    c2 = rng.normal(4.0, 0.2, size=(6, 8))
    X = np.vstack([c1, c2])
    lbls = cluster_vectors(X, max_k=4)
    assert len(np.unique(lbls)) == 2
    assert len(np.unique(lbls[:6])) == 1
    assert len(np.unique(lbls[6:])) == 1
    assert lbls[0] != lbls[6]


def test_extract_attempt_vector_and_cluster_report(tmp_path, monkeypatch):
    import autoresearch
    import cluster_analysis

    monkeypatch.setattr(autoresearch, "ARCHIVE_ROOT", tmp_path / "archive")
    monkeypatch.setattr(cluster_analysis, "ARCHIVE_ROOT", tmp_path / "archive")
    monkeypatch.setattr(cluster_analysis, "REPORTS_ROOT", tmp_path / "reports")

    profile = load_profile("smoke")
    ref, margins, calib = load_reference("smoke")
    canon = canonical_code()
    ko_trade = variant_code("ko_no_trade")

    res_fail = JudgeResult(
        accepted=False,
        stage=2,
        reason="presence: E3 trade failed",
        label="mechanism",
        complexity={"C": 9100},
        delta_C=100.0,
        features={"E3": "fail"},
        code_hash="deadbeef11111111",
    )
    vec1 = extract_attempt_vector(ko_trade, res_fail, canon, profile, ref, margins, calib)
    assert len(vec1["v_joint"]) == len(JOINT_DIM_NAMES)
    assert vec1["feature_ratios"]["E3"] >= 1.5
    assert vec1["ncd_canonical"] > 0.0

    arch = Archive("smoke")
    arch.add(canon, {"id": "origin00", "trajectory": 1, "parent": None, "C": 9200, "eq_distance": 0.0, "label": "origin"})
    for idx in range(3):
        arch.log_attempt({
            "trajectory": 1,
            "generation": 1,
            "variant": idx + 1,
            "candidate": f"cand_{idx}",
            "parent": "origin00",
            "outcome": "rejected",
            "stage": 2,
            "move": "remove_rule",
            "mechanism": "trade",
            "action": "remove",
            "C": 9100,
            "delta_C": 100.0,
            "reason": "presence: E3 failed",
            "desc": "remove trade loop",
            "vector": vec1,
        })

    summary = analyze_clusters("smoke")
    assert summary["evaluated_attempts_clustered"] == 3
    assert len(summary["failure_attractors"]) >= 1
    assert (tmp_path / "reports" / "smoke" / "CLUSTER_REPORT.md").exists()
    assert (tmp_path / "reports" / "smoke" / "attractor_map.svg").exists()
    prompt_mem = format_cluster_memory_for_prompt("smoke")
    assert "Failure Attractors" in prompt_mem


def test_plateau_detection_and_reset_cycle(tmp_path, monkeypatch):
    import autoresearch
    import cluster_analysis

    monkeypatch.setattr(autoresearch, "ARCHIVE_ROOT", tmp_path / "archive")
    monkeypatch.setattr(cluster_analysis, "ARCHIVE_ROOT", tmp_path / "archive")
    monkeypatch.setattr(cluster_analysis, "REPORTS_ROOT", tmp_path / "reports")
    monkeypatch.setattr(autoresearch, "RESULTS_FILE", tmp_path / "results.tsv")
    monkeypatch.setattr(autoresearch, "STRATEGY_FILE", tmp_path / "strategy.py")
    monkeypatch.setattr(autoresearch, "LEDGER_FILE", tmp_path / "mechanisms.json")

    canon = canonical_code()
    (tmp_path / "strategy.py").write_text(canon, encoding="utf-8")
    # Start from a pristine ledger: the repo's mechanisms.json carries live experiment state.
    ledger_src = json.loads((Path(__file__).parent / "mechanisms.json").read_text(encoding="utf-8"))
    for entry in ledger_src["mechanisms"].values():
        entry["status"] = "kept"
        entry["evidence"] = []
        entry.pop("cross_trajectory_history", None)
    (tmp_path / "mechanisms.json").write_text(json.dumps(ledger_src, indent=2), encoding="utf-8")

    autoresearch.init_results()
    profile = load_profile("smoke")
    ref, margins, calib = load_reference("smoke")
    arch = Archive("smoke")

    # Seed trajectory 1 without running traced complexity (fast synthetic origin + best)
    arch.add(canon, {
        "id": "orig_t1",
        "trajectory": 1,
        "parent": None,
        "C": 9200,
        "eq_distance": 0.0,
        "label": "origin",
        "delta_C": 0,
        "generation": 0,
        "audited": True,
    })
    arch.add(canon, {
        "id": "best_t1",
        "trajectory": 1,
        "parent": "orig_t1",
        "C": 8800,
        "eq_distance": 0.42,
        "label": "mechanism",
        "delta_C": 400,
        "generation": 1,
        "mechanism": "move_tiebreak",
        "action": "remove",
        "audited": True,
    })
    arch.data["last_best_improvement_gen"] = 1
    arch.data["attempts_since_improvement"] = 8
    arch.data["trajectory_attempts"] = 10
    arch.save()

    ledger = autoresearch.load_ledger()
    ledger["mechanisms"]["move_tiebreak"]["status"] = "removed"
    (tmp_path / "mechanisms.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    args = argparse.Namespace(
        plateau_generations=2,
        plateau_attempts=6,
        no_git=True,
        no_push=True,
    )

    # At gen=2 (only 1 gen since improvement at gen 1), not plateaued yet
    is_p, _ = check_plateau(arch, ledger, gen=2, args=args)
    assert not is_p

    # At gen=3 (2 gens and 8 attempts since improvement at gen 1), plateau triggers!
    is_p, reason = check_plateau(arch, ledger, gen=3, args=args)
    assert is_p and "frontier_stagnation" in reason

    # Mock seed_archive inside handle_plateau_reset to avoid slow Mesa tracing in unit test
    def _fast_seed(a, prof, force_canonical=False):
        code = canonical_code()
        (tmp_path / "strategy.py").write_text(code, encoding="utf-8")
        a.data["trajectory_start_gen"] = 3
        a.data["last_best_improvement_gen"] = 3
        a.data["attempts_since_improvement"] = 0
        a.data["trajectory_attempts"] = 0
        a.add(code, {
            "id": "orig_t2",
            "trajectory": a.current_trajectory,
            "parent": None,
            "C": 9200,
            "eq_distance": 0.0,
            "label": "origin",
            "delta_C": 0,
            "generation": 3,
            "audited": True,
        })

    monkeypatch.setattr(autoresearch, "seed_archive", _fast_seed)
    handle_plateau_reset(3, arch, profile, ref, margins, ledger, reason, args)

    # Verify trajectory incremented to 2, trajectories.json written, ledger reset to kept with T1 history, and results.tsv marked
    assert arch.current_trajectory == 2
    assert arch.best()["id"] == "orig_t2"
    trajs = arch.load_trajectories()
    assert len(trajs) == 1
    assert trajs[0]["trajectory_id"] == 1
    assert trajs[0]["champion_id"] == "best_t1"
    assert trajs[0]["total_bytes_saved"] == 400
    assert trajs[0]["mechanisms_modified"] == {"move_tiebreak": "removed"}
    assert (arch.champions / "traj_1_best_t1.py").exists()

    updated_ledger = autoresearch.load_ledger()
    assert updated_ledger["mechanisms"]["move_tiebreak"]["status"] == "kept"
    assert updated_ledger["mechanisms"]["move_tiebreak"]["cross_trajectory_history"]["T1"] == "removed"

    tsv_text = (tmp_path / "results.tsv").read_text(encoding="utf-8")
    assert "PLATEAU_RESET" in tsv_text
    assert "TRAJECTORY 1 PLATEAUED" in tsv_text


def test_empty_generation_does_not_count_toward_plateau(tmp_path, monkeypatch):
    """A generation with no proposals (e.g. API quota exhausted) gives its number back."""
    import autoresearch

    monkeypatch.setattr(autoresearch, "ARCHIVE_ROOT", tmp_path / "archive")
    monkeypatch.setattr(autoresearch.cx, "measure", lambda code, profile: {"C": 9200})

    class EmptyProposer:
        def propose(self, *a, **k):
            return []

    canon = canonical_code()
    arch = Archive("smoke")
    arch.add(canon, {"id": "orig", "trajectory": 1, "parent": None, "C": 9200, "eq_distance": 0.0,
                     "label": "origin", "delta_C": 0, "generation": 0, "audited": True})
    arch.data.update(generation=0, last_best_improvement_gen=0, attempts_since_improvement=20,
                     trajectory_attempts=20)
    args = argparse.Namespace(variants=4, plateau_generations=2, plateau_attempts=6, no_git=True, no_push=True)
    profile = load_profile("smoke")
    ledger = json.loads((Path(__file__).parent / "mechanisms.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)

    for _ in range(5):
        arch.data["generation"] += 1
        res = autoresearch.run_generation(arch.data["generation"], arch, EmptyProposer(), profile,
                                          None, None, None, ledger, rng, args)
        assert res is None

    assert arch.data["generation"] == 0
    assert Archive("smoke").data["generation"] == 0       # persisted
    assert arch.data["attempts_since_improvement"] == 20  # no attempts were added
    is_p, _ = check_plateau(arch, ledger, gen=arch.data["generation"], args=args)
    assert not is_p
