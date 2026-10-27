"""
Calibration controls.

Every control is defined relative to canonical_strategy.py (as exact text
patches) or as a standalone model file, so the set is fixed and reviewable.

roles
  margin    innocuous perturbations: changes that leave the society unchanged.
            Their shifts define the equivalence margins.
  positive  held-out "same society, different simulation" controls (not used to
            set margins). The judge must ACCEPT these >= 95% of the time.
  negative  knockouts whose effect is certain. The judge must REJECT them, with
            every targeted feature failing. "Ignore an input" knockouts (the model
            behaves as at Mesa's default whatever the input says) test that every
            trait response is required; they count as negatives only for inputs
            to which canonical Sugarscape responds beyond margin (`terms`).
  null      null societies (no social mechanism). Must fail the targeted features.
  probe     knockouts whose effect is an open scientific question (e.g. whether
            trading at random but individually rational prices still discovers
            prices, cf. Gode & Sunder 1993). Reported, never part of the pass
            criterion: tuning the judge until they fail would assume the answer.
"""

from __future__ import annotations

from pathlib import Path

from judge import PROJECT_DIR

CONTROLS_DIR = Path(__file__).resolve().parent
CANONICAL_FILE = PROJECT_DIR / "canonical_strategy.py"

VARIANTS: dict[str, dict] = {
    # -- innocuous perturbations (set the margins) --------------------------
    "regrowth_up5": {
        "role": "margin",
        "doc": "growback 5% faster",
        "patches": [("        regrowth = self.scenario.regrowth_rate\n",
                     "        regrowth = self.scenario.regrowth_rate * 1.05\n")],
    },
    "regrowth_down5": {
        "role": "margin",
        "doc": "growback 5% slower",
        "patches": [("        regrowth = self.scenario.regrowth_rate\n",
                     "        regrowth = self.scenario.regrowth_rate * 0.95\n")],
    },
    "growback_after_agents": {
        "role": "margin",
        "doc": "resources regrow after agents act instead of before",
        "patches": [(
            "        regrowth = self.scenario.regrowth_rate\n"
            "        self.grid.sugar[:] = np.minimum(self.grid.sugar + regrowth, self.sugar_distribution)\n"
            "        self.grid.spice[:] = np.minimum(self.grid.spice + regrowth, self.spice_distribution)\n"
            "\n"
            "        # step trader agents\n"
            "        self.agents.shuffle_do(\"step\")\n"
            "        if self.enable_trade:\n"
            "            self.agents.shuffle_do(\"trade_with_neighbors\")\n",
            "        regrowth = self.scenario.regrowth_rate\n"
            "\n"
            "        # step trader agents\n"
            "        self.agents.shuffle_do(\"step\")\n"
            "        if self.enable_trade:\n"
            "            self.agents.shuffle_do(\"trade_with_neighbors\")\n"
            "        self.grid.sugar[:] = np.minimum(self.grid.sugar + regrowth, self.sugar_distribution)\n"
            "        self.grid.spice[:] = np.minimum(self.grid.spice + regrowth, self.spice_distribution)\n",
        )],
    },
    "endowment_jitter": {
        "role": "margin",
        "doc": "initial endowment range widened by one unit each side",
        "patches": [(
            "            sugar=self.rng.integers(\n                self.scenario.endowment_min,\n                self.scenario.endowment_max,\n",
            "            sugar=self.rng.integers(\n                self.scenario.endowment_min - 1,\n                self.scenario.endowment_max + 1,\n",
        ), (
            "            spice=self.rng.integers(\n                self.scenario.endowment_min,\n                self.scenario.endowment_max,\n",
            "            spice=self.rng.integers(\n                self.scenario.endowment_min - 1,\n                self.scenario.endowment_max + 1,\n",
        )],
    },
    # -- held-out positive controls --------------------------------------------
    "numpy_reimplementation": {
        "role": "positive",
        "doc": "independent plain-numpy implementation of the same rules (no Mesa)",
        "file": "numpy_g1mt.py",
    },
    "rng_reorder": {
        "role": "positive",
        "doc": "same rules, different dice: trait draws reordered and model seed remixed",
        "patches": [(
            "        rng=int(rng.integers(2**31)), **{k: env[k] for k in SCENARIO_KEYS}",
            "        rng=int(rng.integers(2**31)) ^ 0x5BD1E995, **{k: env[k] for k in SCENARIO_KEYS}",
        ), (
            "            vision=self.rng.integers(",
            "            vision=self.rng.permutation(self.rng.integers(",
        ), (
            "                self.scenario.vision_max,\n                (n,),\n                endpoint=True,\n            ),\n        )",
            "                self.scenario.vision_max,\n                (n,),\n                endpoint=True,\n            )),\n        )",
        )],
    },
    # -- rule knockouts ------------------------------------------------------
    "ko_no_trade": {
        "role": "negative",
        "doc": "trade rule T removed (agents never exchange)",
        "targets": ["E3", "E5"],
        "patches": [("        for a in self.cell.get_neighborhood(radius=self.vision).agents:\n            self.trade(a)\n",
                     "        pass\n")],
    },
    "ko_no_growback": {
        "role": "negative",
        "doc": "resources never regrow",
        "targets": ["E2"],
        "patches": [("        regrowth = self.scenario.regrowth_rate\n",
                     "        regrowth = 0\n")],
    },
    "ko_random_move": {
        "role": "negative",
        "doc": "movement rule M replaced by a random empty cell within vision",
        "targets": ["E4"],
        "patches": [("        # 2. determine which move maximizes welfare\n",
                     "        self.cell = self.random.choice(neighboring_cells)\n        return\n        # 2. determine which move maximizes welfare\n")],
    },
    "ko_no_mrs_stop": {
        "role": "probe",
        "doc": "MRS-crossing stop removed from the trade rule",
        "targets": ["E3"],
        "patches": [("        if not (both_agents_better_off and mrs_not_crossing):",
                     "        if not both_agents_better_off:")],
    },
    "ko_uniform_vision": {
        "role": "negative",
        "doc": "vision fixed at 1 regardless of the vision inputs",
        "targets": ["E5"],
        "terms": ["vision_mean", "vision_spread"],
        "patches": [("        self.vision = vision\n", "        self.vision = 1\n")],
    },
    # -- input-ignoring knockouts: every trait response is required ----------------
    "ko_ignore_population": {
        "role": "negative",
        "doc": "always starts 200 agents whatever initial_population says",
        "targets": ["E5"],
        "terms": ["log_initial_population"],
        "patches": [("        n = self.scenario.initial_population\n        Trader.create_agents(\n            self,\n"
                     "            self.scenario.initial_population,\n",
                     "        n = 200\n        Trader.create_agents(\n            self,\n            n,\n")],
    },
    "ko_ignore_regrowth": {
        "role": "negative",
        "doc": "resources always regrow 1 unit per step whatever regrowth_rate says",
        "targets": ["E5"],
        "terms": ["log_regrowth_rate"],
        "patches": [("        regrowth = self.scenario.regrowth_rate\n", "        regrowth = 1\n")],
    },
    "ko_ignore_vision": {
        "role": "negative",
        "doc": "vision always drawn from Mesa's default 1-5 whatever the vision inputs say",
        "targets": ["E5"],
        "terms": ["vision_mean", "vision_spread"],
        "patches": [("            vision=self.rng.integers(\n                self.scenario.vision_min,\n"
                     "                self.scenario.vision_max,\n",
                     "            vision=self.rng.integers(\n                1,\n                5,\n")],
    },
    "ko_ignore_metabolism": {
        "role": "negative",
        "doc": "metabolisms always drawn from Mesa's default 1-5 whatever the metabolism inputs say",
        "targets": ["E5"],
        "terms": ["metabolism_mean", "metabolism_spread"],
        "patches": [(f"            metabolism_{g}=self.rng.integers(\n                self.scenario.metabolism_min,\n"
                     "                self.scenario.metabolism_max,\n",
                     f"            metabolism_{g}=self.rng.integers(\n                1,\n                5,\n")
                    for g in ("sugar", "spice")],
    },
    "ko_ignore_endowment": {
        "role": "negative",
        "doc": "endowments always drawn from Mesa's default 25-50 whatever the endowment inputs say",
        "targets": ["E5"],
        "terms": ["endowment_mean", "endowment_spread"],
        "patches": [(f"            {g}=self.rng.integers(\n                self.scenario.endowment_min,\n"
                     "                self.scenario.endowment_max,\n",
                     f"            {g}=self.rng.integers(\n                25,\n                50,\n")
                    for g in ("sugar", "spice")],
    },
    "ko_trade_always": {
        "role": "negative",
        "doc": "agents trade even when enable_trade is false (the trade switch is ignored)",
        "targets": ["E3", "E5"],
        "patches": [("        if self.enable_trade:\n", "        if True:\n")],
    },
    # -- null societies ------------------------------------------------------
    "null_random_walk": {
        "role": "null",
        "doc": "random walkers with multiplicative random wealth; no resources, no interaction",
        "targets": ["E3", "E4"],
        "file": "null_random_walk.py",
    },
    "null_static_agents": {
        "role": "null",
        "doc": "agents never move (no foraging)",
        "targets": ["E4"],
        "patches": [("        # 1. identify all possible moves\n", "        return\n        # 1. identify all possible moves\n")],
    },
    "null_noise_exchange": {
        "role": "null",
        "doc": "one exchange per encounter at a random price, with no welfare or MRS check (noise trading)",
        "targets": ["E3"],
        "patches": [("        price = math.sqrt(mrs_self * mrs_other)\n",
                     "        price = math.exp(self.random.uniform(-2.0, 2.0))\n"),
                    ("        if not (both_agents_better_off and mrs_not_crossing):\n            return False\n", ""),
                    ("        # continue trading\n        self.trade(other)\n", "        return\n")],
    },
    # -- probes (reported, not part of the pass criterion) ------------------------
    "null_random_price": {
        "role": "probe",
        "doc": "random price unrelated to MRS, but trades still need both parties better off "
               "(zero-intelligence traders with a budget constraint, Gode & Sunder 1993)",
        "targets": ["E3"],
        "patches": [("        price = math.sqrt(mrs_self * mrs_other)\n",
                     "        price = math.exp(self.random.uniform(-2.0, 2.0))\n")],
    },
}


def canonical_code() -> str:
    return CANONICAL_FILE.read_text(encoding="utf-8")


def variant_code(name: str) -> str:
    spec = VARIANTS[name]
    if "file" in spec:
        return (CONTROLS_DIR / spec["file"]).read_text(encoding="utf-8")
    code = canonical_code()
    for old, new in spec["patches"]:
        n = code.count(old)
        if n != 1:
            raise ValueError(f"control '{name}': patch anchor found {n} times (expected 1): {old[:60]!r}")
        code = code.replace(old, new)
    return code


def by_role(role: str) -> list[str]:
    return [k for k, v in VARIANTS.items() if v["role"] == role]
