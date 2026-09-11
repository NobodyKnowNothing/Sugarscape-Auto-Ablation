"""
Canonical Mesa Sugarscape G1MT — Deterministically Flattened Module.
Generated directly from upstream Mesa Sugarscape with Traders repository.
"""

import math
from pathlib import Path
from typing import Any

from mesa.discrete_space import CellAgent, OrthogonalVonNeumannGrid
from mesa.experimental.scenarios import Scenario
import mesa
import numpy as np
# No redundant imports needed here
# ===========================================================================
# Mesa Compatibility Bridge (supports both Mesa 3.x and Mesa 4.x)
# ===========================================================================

# ===========================================================================
# Agent Helper Functions & Trader Class (from agents.py)
# ===========================================================================


class Trader(CellAgent):
    def __init__(self, model, cell, sugar=0, spice=0, metabolism_sugar=0, metabolism_spice=0, vision=0):
        super().__init__(model)
        self.cell, self.sugar, self.spice = cell, sugar, spice
        self.metabolism_sugar, self.metabolism_spice, self.vision = metabolism_sugar, metabolism_spice, vision
        self.prices, self.trade_count = [], 0
        m_tot = metabolism_sugar + metabolism_spice
        self.welfare = lambda s, p: s**(metabolism_sugar/m_tot) * p**(metabolism_spice/m_tot)
        self.mrs = lambda s, p: (p/metabolism_spice) / (s/metabolism_sugar)

    def trade(self, other):
        m_s, m_o = self.mrs(self.sugar, self.spice), other.mrs(other.sugar, other.spice)
        if math.isclose(m_s, m_o): return
        price = math.sqrt(m_s * m_o)
        s, o = (self, other) if m_s > m_o else (other, self)
        s_ex, p_ex = (1, int(price)) if price >= 1 else (int(1 / price), 1)
        s_s, o_s, s_p, o_p = s.sugar + s_ex, o.sugar - s_ex, s.spice - p_ex, o.spice + p_ex
        if min(s_s, o_s, s_p, o_p) > 0 and \
           s.welfare(s_s, s_p) > s.welfare(s.sugar, s.spice) and \
           o.welfare(o_s, o_p) > o.welfare(o.sugar, o.spice) and \
           s.mrs(s_s, s_p) > o.mrs(o_s, o_p):
            s.sugar, o.sugar, s.spice, o.spice = s_s, o_s, s_p, o_p
            s.prices.append(price)
            s.trade_partners.append(o.unique_id)
            s.trade(o)

    def step(self):
        self.prices, self.trade_partners = [], []
        ns = [c for c in self.cell.get_neighborhood(self.vision, include_center=True) if c.is_empty]
        if ns:
            self.random.shuffle(ns)
            self.cell = max(ns, key=lambda c: (self.welfare(self.sugar + c.sugar, self.spice + c.spice), -math.dist(self.cell.coordinate, c.coordinate)))
        self.sugar += self.cell.sugar - self.metabolism_sugar
        self.spice += self.cell.spice - self.metabolism_spice
        self.cell.sugar = self.cell.spice = 0
        if self.sugar <= 0 or self.spice <= 0: self.remove()

    def trade_with_neighbors(self):
        for a in self.cell.get_neighborhood(radius=self.vision).agents:
            self.trade(a)
# ===========================================================================
# Model Helper Functions & Sugarscape Model (from model.py)
# ===========================================================================

class SugarScapeScenario(Scenario):
    """Sugarscape scenario class."""

    initial_population: int = 200
    endowment_min: int = 25
    endowment_max: int = 50
    metabolism_min: int = 1
    metabolism_max: int = 5
    vision_min: int = 1
    vision_max: int = 5
    enable_trade: bool = True

class SugarscapeG1mt(mesa.Model):
    """
    Manager class to run Sugarscape with Traders
    """

    def __init__(self, scenario: SugarScapeScenario = SugarScapeScenario):
        super().__init__(scenario=scenario)
        # Initiate width and height of sugarscape
        self.width = 50
        self.height = 50

        # Initiate population attributes
        self.enable_trade = self.scenario.enable_trade
        self.running = True

        # initiate mesa grid class
        self.grid = OrthogonalVonNeumannGrid(
            (self.width, self.height), torus=False, random=self.random
        )
        # initiate datacollector
        self.datacollector = mesa.DataCollector(
            model_reporters={"Trade Volume": lambda m: sum(len(a.trade_partners) for a in m.agents)}
        )

        # read in landscape file from supplementary material
        self.sugar_distribution = np.genfromtxt(Path(__file__).parent / "sugar-map.txt")
        self.spice_distribution = np.flip(self.sugar_distribution, 1)

        self.grid.add_property_layer("sugar", self.sugar_distribution.copy())
        self.grid.add_property_layer("spice", self.spice_distribution.copy())

        n = self.scenario.initial_population
        agent_args = {k: self.rng.integers(getattr(self.scenario, f"{v}_min"), getattr(self.scenario, f"{v}_max"), (n,), endpoint=True) 
                      for k, v in [("sugar", "endowment"), ("spice", "endowment"), ("metabolism_sugar", "metabolism"), ("metabolism_spice", "metabolism"), ("vision", "vision")]}
        Trader.create_agents(self, n, self.random.choices(self.grid.all_cells.cells, k=n), **agent_args)

    def step(self):
        """
        Unique step function that does staged activation of sugar and spice
        and then randomly activates traders
        """
        self.grid.sugar[:] = np.minimum(self.grid.sugar + 1, self.sugar_distribution)
        self.grid.spice[:] = np.minimum(self.grid.spice + 1, self.spice_distribution)

        # step trader agents
        self.agents.shuffle_do("step")
        if self.enable_trade:
            self.agents.shuffle_do("trade_with_neighbors")
        self.datacollector.collect(self)

# ===========================================================================
# Project Evaluation Harness Interface (prepare.py / parameter_sweeps.py)
# ===========================================================================

def create_model(seed: int = 42, **params) -> SugarscapeG1mt:
    """Instantiate SugarscapeG1mt model with seed and parameter overrides."""
    init_pop = params.pop("initial_population", 200)
    params.pop("steps", None)
    sc = SugarScapeScenario(rng=seed, **{k: v for k, v in params.items() if hasattr(SugarScapeScenario, k)})
    model = SugarscapeG1mt(scenario=sc)
    model.initial_population = init_pop
    return model

def run_model(model: SugarscapeG1mt, steps: int = 200) -> dict[str, Any]:
    """
    Execute simulation steps and extract the standardized 9-key metrics dictionary.
    """
    try:
        for _ in range(steps): model.step()
    except (ValueError, IndexError) as e:
        return {"error": f"Mesa crash: {e}"}

    df = model.datacollector.get_model_vars_dataframe()
    cumul_trade_volume = int(df["Trade Volume"].sum()) if "Trade Volume" in df else 0

    agents = list(model.agents)
    agent_wealths = [float(a.sugar + a.spice) for a in agents]
    trade_prices = [p for a in agents for p in getattr(a, "prices", [])]
    agent_positions = [a.cell.coordinate for a in agents if getattr(a, "cell", None)]
    return {
        "agent_wealths": agent_wealths,
        "final_population": len(agents),
        "initial_population": getattr(model, "initial_population", 200),
        "trade_prices": trade_prices,
        "trade_volume": cumul_trade_volume,
        "agent_positions": agent_positions,
        "grid_width": getattr(model, "width", 50),
        "grid_height": getattr(model, "height", 50),
        "steps_run": steps,
    }
