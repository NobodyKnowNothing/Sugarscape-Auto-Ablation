"""Tests for the emergence judge (run with: pytest test_judge.py -v)."""

import hashlib
import math

import numpy as np
import pytest

from judge import complexity as cx
from judge import guard
from judge.contract import ContractError, make_env, materialize, run_model, validate_snapshot
from judge.controls import VARIANTS, canonical_code, variant_code
from judge.design import load_profile, seeds
from judge.features import Recorder, gini, lorenz, presence, top_share
from judge.contract import Snapshot
from judge.stats import holm, scalar_compare, vector_compare, effect_compare

CANON = canonical_code()


# ---------------------------------------------------------------------------
# Static guard
# ---------------------------------------------------------------------------

def test_guard_accepts_canonical():
    assert guard.check(CANON) == []


@pytest.mark.parametrize("snippet, fragment", [
    ("import os\n", "import of 'os'"),
    ("x = eval('1')\n", "call to 'eval'"),
    ("import numpy as np\nx = np.random.rand()\n", "global numpy RNG"),
    ("from judge import pipeline\n", "import from 'judge'"),
    ("f = open('sugar-map.txt')\n", "call to 'open'"),
    ("x = b'abc'\n", "bytes literals"),
])
def test_guard_rejects(snippet, fragment):
    problems = guard.check(CANON + "\n" + snippet)
    assert any(fragment in p for p in problems), problems


def test_guard_requires_contract():
    code = CANON.replace("def observe(model):", "def look(model):")
    assert any("observe" in p for p in guard.check(code))


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------

def test_canon_invariant_to_names_comments_docstrings_annotations():
    renamed = (CANON.replace("calculate_welfare", "w").replace("metabolism_sugar", "ms")
               .replace("    def eat(self):", "    # comment\n    def eat(self):\n        '''doc'''"))
    assert cx.canon(CANON) == cx.canon(renamed)
    assert cx.static_size(CANON) == cx.static_size(renamed)


def test_canon_counts_real_logic():
    more = CANON.replace("        self.sugar -= self.metabolism_sugar\n",
                         "        self.sugar -= self.metabolism_sugar\n        self.sugar = max(self.sugar, -1e9) * 1.0 + 0.0\n")
    assert cx.static_size(more)["C_strategy_only"] > cx.static_size(CANON)["C_strategy_only"]


def test_traced_complexity_counts_mesa():
    profile = load_profile("smoke")
    m = cx.measure(CANON, profile, use_cache=False)
    assert m["dependency_functions"] > 20
    assert m["C"] > m["C_strategy_only"]
    assert any(x.startswith("mesa/") for x in m["dependency_manifest"])
    numpy_only = cx.measure(variant_code("numpy_reimplementation"), profile, use_cache=False)
    assert numpy_only["dependency_functions"] == 0


# ---------------------------------------------------------------------------
# Contract (fail closed)
# ---------------------------------------------------------------------------

ENV = make_env({})
GOOD = {"ids": [1, 2], "positions": [(0, 0), (3, 4)], "holdings": [(1.0, 2.0), (3.0, 4.0)],
        "exchanges": [(1, 2, (1, -2))], "resources": np.zeros((2, 50, 50))}


def test_snapshot_ok():
    snap = validate_snapshot(GOOD, ENV, None)
    assert snap.n == 2 and snap.holdings.shape == (2, 2)


@pytest.mark.parametrize("mutate", [
    lambda o: o.pop("exchanges"),
    lambda o: o.update(positions=[(0, 0), (50, 1)]),
    lambda o: o.update(holdings=[(1.0, math.nan), (3.0, 4.0)]),
    lambda o: o.update(ids=[1, 1]),
    lambda o: o.update(resources=np.zeros((2, 10, 10))),
    lambda o: o.update(exchanges=[(1, 2, (1,))]),
])
def test_snapshot_rejects(mutate):
    obs = {k: (list(v) if isinstance(v, list) else v) for k, v in GOOD.items()}
    mutate(obs)
    with pytest.raises(ContractError):
        validate_snapshot(obs, ENV, None)


def test_snapshot_allows_extinction():
    obs = dict(GOOD, ids=[], positions=[], holdings=[], exchanges=[])
    assert validate_snapshot(obs, ENV, 2).n == 0


def test_canonical_deterministic_given_seed():
    path = materialize(CANON)
    digests = []
    for _ in range(2):
        h = hashlib.sha256()
        run_model(path, {}, 777, 10, digest=h)
        digests.append(h.hexdigest())
    assert digests[0] == digests[1]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def test_gini_and_shares():
    assert gini(np.ones(10)) == pytest.approx(0.0)
    assert gini(np.r_[np.zeros(9), 1.0]) == pytest.approx(0.9)
    assert top_share(np.r_[np.ones(9), 91.0]) == pytest.approx(0.91)
    lz = lorenz(np.ones(100))
    assert np.allclose(lz, np.linspace(0.1, 0.9, 9))


def test_recorder_runs_and_presence_on_canonical():
    path = materialize(CANON)
    runs = []
    for s in range(4):
        rec = Recorder(make_env({}), 200, s)
        run_model(path, {}, s, 200, rec)
        runs.append(rec.stats())
    for f in ("E1", "E3", "E4"):
        assert presence(f, runs)[0], f


def test_presence_fails_on_extinction():
    runs = [{"extinct": 1.0}] * 5
    assert presence("E1", runs) == (False, "population went extinct in some runs")


# ---------------------------------------------------------------------------
# Equivalence statistics
# ---------------------------------------------------------------------------

def test_scalar_tost_equivalent_and_different():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 400), rng.normal(0, 1, 400)
    assert scalar_compare(a, b, 0.05).equivalent(0.5)
    c = rng.normal(3, 1, 400)
    comp = scalar_compare(c, b, 0.05)
    assert comp.clearly_different(1.0) and comp.direction() == "too high"


def test_scalar_undefined_when_candidate_missing():
    comp = scalar_compare([math.nan] * 10, list(range(10)), 0.05)
    assert math.isinf(comp.d) and not comp.equivalent(100)


def test_vector_and_effect_compare():
    rng = np.random.default_rng(1)
    R = rng.normal(0, 1, (200, 5))
    assert vector_compare(rng.normal(0, 1, (200, 5)), R, 0.05).equivalent(0.75)
    assert vector_compare(rng.normal(2, 1, (200, 5)), R, 0.05).clearly_different(0.75)
    eff = effect_compare(rng.normal(1, 1, 300), rng.normal(0, 1, 300), rng.normal(1, 1, 300), rng.normal(0, 1, 300), 0.05)
    assert eff.equivalent(0.7)


def test_holm():
    assert holm([0.001, 0.02, 0.5], 0.05) == [True, True, False]
    assert holm([0.04, 0.04], 0.05) == [False, False]


# ---------------------------------------------------------------------------
# Controls, seeds, proposals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(VARIANTS))
def test_controls_build_and_pass_guard(name):
    assert guard.check(variant_code(name)) == []


def test_seed_pools_deterministic_and_disjoint():
    a = seeds("reference", "canonical", "default", 50)
    assert a == seeds("reference", "canonical", "default", 50)
    assert seeds("reference", "canonical", "default", 10, start=40) == a[40:]
    b = seeds("candidate", "canonical", "default", 50)
    assert not set(a) & set(b)


def test_parse_research_variants():
    from edit_harness import parse_research_variants
    text = (
        "**Variant 1:** drop vision heterogeneity\nMOVE: remove_rule\nMECHANISM: vision\nACTION: remove\n"
        "HYPOTHESIS: foraging still concentrates agents\n```str_replace\n<<<<<<< SEARCH\n"
        "        self.vision = vision\n=======\n        self.vision = 1\n>>>>>>> REPLACE\n```\n"
        "**Variant 2:** rewrite\nMOVE: representation\nMECHANISM: none\nACTION: refactor\nHYPOTHESIS: x\n"
        "```full_file\ndef init(env, rng):\n    return 0\n\ndef step(s, rng):\n    pass\n\ndef observe(s):\n    return {}\n```\n"
    )
    out = parse_research_variants(text, CANON)
    assert out[0]["mechanism"] == "vision" and "self.vision = 1" in out[0]["code"]
    assert out[1]["move"] == "representation" and out[1]["code"].startswith("def init")


def test_pipeline_stage0_rejections():
    from judge.pipeline import evaluate_candidate
    profile = load_profile("smoke")
    r = evaluate_candidate(CANON + "\nimport os\n", profile, ref={}, margins={}, log=lambda *a: None)
    assert not r.accepted and r.stage == 0 and "guard" in r.reason
    c = cx.measure(CANON, profile)["C"]
    renamed = CANON.replace("calculate_welfare", "w")
    r = evaluate_candidate(renamed, profile, parent={"C": c}, ref={}, margins={}, log=lambda *a: None)

    assert not r.accepted and r.stage == 0 and "not simpler" in r.reason


# ---------------------------------------------------------------------------
# Input domain and response-surface equivalence
# ---------------------------------------------------------------------------

def test_domain_samples_valid_and_deterministic():
    profile = load_profile("full")
    dom = profile.domain
    pts = dom.sample(64, "t", "sobol")
    assert [p.params for p in pts] == [p.params for p in dom.sample(64, "t", "sobol")]
    assert [p.params for p in pts] != [p.params for p in dom.sample(64, "u", "sobol")]
    f = dom.factors
    for p in pts:
        q = p.params
        assert f["initial_population"]["low"] <= q["initial_population"] <= f["initial_population"]["high"]
        for t in ("vision", "metabolism", "endowment"):
            assert q[f"{t}_min"] <= q[f"{t}_max"]
        assert all(-0.5 - 1e-9 <= z <= 0.5 + 1e-9 for z in p.z)
        assert len(p.z) == len(dom.regressor_names)
    iid = dom.sample(32, "t", "iid")
    assert len({p.id for p in iid}) == 32
    # homogeneous populations (span 0) are part of the domain
    assert any(p.params["vision_min"] == p.params["vision_max"] for p in pts)


def _synthetic(pts, rng, n, effect=None, shift=0.0, noise=1.0):
    """Panel with stat 'plateau_frac' = 2*vision_mean + effect(z) + shift + noise, both worlds."""
    from judge.surface import key
    out = {}
    for p in pts:
        for w in ("trade", "notrade"):
            mu = 2.0 * p.z[2] + (0.5 if w == "trade" else 0.0) + shift + (effect(p.z) if effect else 0.0)
            out[key(p.id, w)] = [{"plateau_frac": float(mu + noise * rng.normal())} for _ in range(n)]
    return out


def _margins(profile, pts, ref):
    from judge.surface import scales
    terms = profile.domain.regressor_names
    sc = {w: {"plateau_frac": scales(ref, [p.id for p in pts], ["trade"])["trade"]["plateau_frac"]}
          for w in ("trade", "notrade")}
    return sc, terms


def test_surface_l2_and_slopes_identical_vs_different():
    from judge import surface
    profile = load_profile("full")
    rng = np.random.default_rng(0)
    pts = profile.domain.sample(192, "syn", "sobol")
    pids, Z = [p.id for p in pts], {p.id: p.z for p in pts}
    ref = _synthetic(pts, rng, 4)
    sc, terms = _margins(profile, pts, ref)
    s = sc["trade"]["plateau_frac"]
    same = _synthetic(pts, rng, 2)
    t = surface.l2_test(same, ref, pids, "trade", "plateau_frac", "scalar", s, 1.0, 0.05, rng, 0.2)
    assert t.status == "pass", t
    slopes = surface.slope_tests(same, ref, pids, Z, "trade", "plateau_frac", s, {k: 2.0 for k in terms}, terms, 0.05)
    assert all(x.status == "pass" for x in slopes), [x for x in slopes if x.status != "pass"]
    shifted = _synthetic(pts, rng, 2, shift=3.0)
    assert surface.l2_test(shifted, ref, pids, "trade", "plateau_frac", "scalar", s, 1.0, 0.05, rng, 0.2).status == "fail"
    # a candidate whose vision response is reversed fails the vision slope but not the others
    rev = _synthetic(pts, rng, 2, effect=lambda z: -4.0 * z[2])
    sl = {x.term: x for x in surface.slope_tests(rev, ref, pids, Z, "trade", "plateau_frac", s,
                                                 {k: 2.0 for k in terms}, terms, 0.05)}
    assert sl["vision_mean"].status == "fail"
    assert sl["metabolism_mean"].status == "pass"
    # a spurious response to an input canonical ignores is also a failure
    spur = _synthetic(pts, rng, 2, effect=lambda z: 6.0 * z[6])
    sl = {x.term: x for x in surface.slope_tests(spur, ref, pids, Z, "trade", "plateau_frac", s,
                                                 {k: 2.0 for k in terms}, terms, 0.05)}
    assert sl["endowment_mean"].status == "fail"


def test_surface_contrast_spread_named():
    from judge import surface
    profile = load_profile("full")
    rng = np.random.default_rng(1)
    pts = profile.domain.sample(192, "syn2", "sobol")
    pids, Z = [p.id for p in pts], {p.id: p.z for p in pts}
    ref = _synthetic(pts, rng, 4)
    s = surface.scales(ref, pids, ["trade"])["trade"]["plateau_frac"]
    same = _synthetic(pts, rng, 2)
    assert surface.contrast_test(same, ref, pids, "plateau_frac", s, 0.5, 0.05).status == "pass"
    # candidate in which trade has no effect
    no_trade_effect = _synthetic(pts, rng, 2, effect=lambda z: 0.0)
    for k in list(no_trade_effect):
        if k.endswith("|trade"):
            for r in no_trade_effect[k]:
                r["plateau_frac"] -= 0.5
    assert surface.contrast_test(no_trade_effect, ref, pids, "plateau_frac", s, 0.25, 0.05).status == "fail"
    noisy = _synthetic(pts, rng, 2, noise=3.0)
    assert surface.spread_test(noisy, ref, pids, "trade", "plateau_frac", 2.0, 0.05, rng).status == "fail"
    assert surface.spread_test(same, ref, pids, "trade", "plateau_frac", 2.0, 0.05, rng).status == "pass"
    terms = profile.domain.regressor_names
    nr = {"id": "x", "claim": "rises with vision", "stat": "plateau_frac", "world": "trade", "term": "vision_mean", "sign": 1}
    scs = {"trade": {"plateau_frac": s}, "notrade": {"plateau_frac": s}}
    assert surface.named_test(nr, same, pids, Z, terms, scs, 0.05).status == "pass"
    reversed_ = _synthetic(pts, rng, 2, effect=lambda z: -4.0 * z[2])
    assert surface.named_test(nr, reversed_, pids, Z, terms, scs, 0.05).status == "fail"


def test_recorder_indicators_on_canonical():
    path = materialize(CANON)
    rec = Recorder(make_env({}), 100, 3)
    run_model(path, {}, 3, 100, rec)
    s = rec.stats()
    for k in ("ineq_up", "plateaued", "trading", "disp_down", "clustered", "loghill_T"):
        assert k in s
    assert s["trading"] == 1.0
    rec = Recorder(make_env({"enable_trade": False}), 50, 3)
    run_model(path, {"enable_trade": False}, 3, 50, rec)
    s = rec.stats()
    assert s["trading"] == 0.0 and math.isnan(s["disp_down"])


def test_coverage_bound_from_l2():
    from judge.pipeline import coverage_bounds
    from judge.surface import Assessment, STest
    a = Assessment([STest("E2", "trade", "plateau_frac", "l2", "", 1.0, "pass", 0.1, 0.0, 0.5, 100)])
    cov = coverage_bounds(a, 2.5)
    assert cov["per_statistic"]["plateau_frac|trade"] == pytest.approx(0.25 / 6.25)


def test_run_cache_serves_repeats_but_not_digest_runs():
    from judge.pipeline import _run_all
    profile = load_profile("smoke")
    spec = [("t", {"initial_population": 60}, 424242)]
    a = _run_all(CANON, profile, spec)[0]
    b = _run_all(CANON, profile, spec)[0]
    assert a["gini_T"] == b["gini_T"] and b["_seed"] == 424242
    d = _run_all(CANON, profile, spec, digest=True)[0]
    assert "_digest" in d and d["gini_T"] == a["gini_T"]
