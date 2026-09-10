"""
Calibration controls.

Every control is defined relative to canonical_strategy.py (as exact text
patches) or as a standalone model file, so the set is fixed and reviewable.

roles
  margin    innocuous perturbations: changes that leave the society unchanged.
            Their shifts define the equivalence margins.
  positive  held-out "same society, different simulation" controls (not used to
            set margins). The judge must ACCEPT these >= 95% of the time.
  negative  rule knockouts. The judge must REJECT the targeted features >= 95%.
  null      null societies (no social mechanism). Must fail the targeted features.
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
        "targets": ["E3"],
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
        "role": "negative",
        "doc": "MRS-crossing stop removed from the trade rule",
        "targets": ["E3"],
        "patches": [("        if not (both_agents_better_off and mrs_not_crossing):",
                     "        if not both_agents_better_off:")],
    },
    "ko_uniform_vision": {
        "role": "negative",
        "doc": "vision fixed at 1 regardless of the vision inputs",
        "targets": ["E5"],
        "patches": [("        self.vision = vision\n", "        self.vision = 1\n")],
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
    "null_random_price": {
        "role": "null",
        "doc": "exchanges happen at a random price unrelated to MRS",
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
