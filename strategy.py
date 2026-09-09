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

from mesa.discrete_space import CellAgent
from mesa.discrete_space import OrthogonalVonNeumannGrid

# ===========================================================================
# Mesa Compatibility Bridge (supports both Mesa 3.x and Mesa 4.x)
# ===========================================================================
try:
    from mesa.discrete_space.property_layer import PropertyLayer

    if hasattr(PropertyLayer, "data"):
        if not hasattr(PropertyLayer, "__setitem__"):
            PropertyLayer.__setitem__ = lambda self, idx, val: setattr(self, "data", val)
        if not hasattr(PropertyLayer, "__add__"):
            PropertyLayer.__add__ = lambda self, other: self.data + other
            PropertyLayer.__radd__ = lambda self, other: other + self.data

    _orig_add_prop = OrthogonalVonNeumannGrid.add_property_layer

    def _compat_add_property_layer(self, name_or_layer, data=None):
        if data is not None and hasattr(PropertyLayer, "from_data"):
            return _orig_add_prop(self, PropertyLayer.from_data(name_or_layer, data))
        return _orig_add_prop(self, name_or_layer)

    OrthogonalVonNeumannGrid.add_property_layer = _compat_add_property_layer
except ImportError:
    pass

# ===========================================================================
# Agent Helper Functions & Trader Class (from agents.py)
# ===========================================================================

get_distance = lambda c1, c2: math.sqrt((c1.coordinate[0]-c2.coordinate[0])**2 + (c1.coordinate[1]-c2.coordinate[1])**2)

class Trader(CellAgent):
    """
    Trader:
    - has a metabolism of sugar and spice
    - harvest and trade sugar and spice to survive
    """

    def __init__(
        self,
        model,
        cell,
        sugar=0,
        spice=0,
        metabolism_sugar=0,
        metabolism_spice=0,
        vision=0,
    ):
        super().__init__(model)
        self.cell = cell
        self.sugar = sugar
        self.spice = spice
        self.metabolism_sugar = metabolism_sugar
        self.metabolism_spice = metabolism_spice
        self.vision = vision
        self.prices = []
        self.trade_partners = []

    def calculate_welfare(self, sugar, spice):
        m = self.metabolism_sugar + self.metabolism_spice
        return sugar**(self.metabolism_sugar / m) * spice**(self.metabolism_spice / m)


    def calculate_MRS(self, sugar, spice):
        """
        Helper function for
          - self.trade()
          - self.maybe_self_spice()

        Determines what trader agent needs and can give up
        """

        return (spice / self.metabolism_spice) / (sugar / self.metabolism_sugar)

    def trade(self, other):
        mrs_s, mrs_o = self.calculate_MRS(self.sugar, self.spice), other.calculate_MRS(other.sugar, other.spice)
        if math.isclose(mrs_s, mrs_o): return
        price = math.sqrt(mrs_s * mrs_o)
        s, b = (self, other) if mrs_s > mrs_o else (other, self)
        s_ex, p_ex = (1, int(price)) if price >= 1 else (int(1 / price), 1)
        s_s, o_s, s_p, o_p = s.sugar + s_ex, b.sugar - s_ex, s.spice - p_ex, b.spice + p_ex
        if all(v > 0 for v in (s_s, o_s, s_p, o_p)) and \
           s.calculate_welfare(s.sugar, s.spice) < s.calculate_welfare(s_s, s_p) and \
           b.calculate_welfare(b.sugar, b.spice) < b.calculate_welfare(o_s, o_p) and \
           s.calculate_MRS(s_s, s_p) > b.calculate_MRS(o_s, o_p):
            s.sugar, b.sugar, s.spice, b.spice = s_s, o_s, s_p, o_p
            self.prices.append(price); self.trade_partners.append(other.unique_id)
            self.trade(other)
    ######################################################################
    #                                                                    #
    #                      MAIN TRADE FUNCTIONS                          #
    #                                                                    #
    ######################################################################

    def move(self):
        """Function for trader agent to identify optimal move."""
        neighbors = [c for c in self.cell.get_neighborhood(self.vision, include_center=True) if c.is_empty]
        if not neighbors: return

        welfares = [self.calculate_welfare(self.sugar + c.sugar, self.spice + c.spice) for c in neighbors]
        max_w = max(welfares)
        candidates = [c for c, w in zip(neighbors, welfares) if math.isclose(w, max_w)]

        min_d = min(get_distance(self.cell, c) for c in candidates)
        self.cell = self.random.choice([c for c in candidates if math.isclose(get_distance(self.cell, c), min_d, rel_tol=1e-2)])
    def step(self):
        """Agent step method."""
        self.prices, self.trade_partners = [], []
        self.move()
        # Eat
        self.sugar += self.cell.sugar
        self.cell.sugar = 0
        self.sugar -= self.metabolism_sugar
        self.spice += self.cell.spice
        self.cell.spice = 0
        self.spice -= self.metabolism_spice
        # Maybe die
        if self.sugar <= 0 or self.spice <= 0:
            self.remove()
    def trade_with_neighbors(self):
        """
        Function for trader agents to decide who to trade with in three parts

        1- identify neighbors who can trade
        2- trade (2 sessions)
        3- collect data
        """
        # iterate through traders in neighboring cells and trade
        for a in self.cell.get_neighborhood(radius=self.vision).agents:
            self.trade(a)

        return

# ===========================================================================
# Model Helper Functions & Sugarscape Model (from model.py)
# ===========================================================================

def flatten(list_of_lists):
    """
    helper function for model datacollector for trade price
    collapses agent price list into one list
    """
    return [item for sublist in list_of_lists for item in sublist]

def geometric_mean(list_of_prices):
    """
    find the geometric mean of a list of prices
    """
    # protects against an invalid value if no prices
    if len(list_of_prices) == 0:
        return -1
    return np.exp(np.log(list_of_prices).mean())

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
            model_reporters={
                "#Traders": lambda m: len(m.agents),
                "Trade Volume": lambda m: sum(len(a.trade_partners) for a in m.agents),
                "Price": lambda m: geometric_mean(
                    flatten([a.prices for a in m.agents])
                ),
            },
            agent_reporters={"Trade Network": "trade_partners"},
        )

        # read in landscape file from supplementary material
        self.sugar_distribution = np.genfromtxt(Path(__file__).parent / "sugar-map.txt")
        self.spice_distribution = np.flip(self.sugar_distribution, 1)

        self.grid.add_property_layer("sugar", self.sugar_distribution.copy())
        self.grid.add_property_layer("spice", self.spice_distribution.copy())

        n = self.scenario.initial_population
        Trader.create_agents(
            self,
            self.scenario.initial_population,
            self.random.choices(self.grid.all_cells.cells, k=n),
            sugar=self.rng.integers(
                self.scenario.endowment_min,
                self.scenario.endowment_max,
                (n,),
                endpoint=True,
            ),
            spice=self.rng.integers(
                self.scenario.endowment_min,
                self.scenario.endowment_max,
                (n,),
                endpoint=True,
            ),
            metabolism_sugar=self.rng.integers(
                self.scenario.metabolism_min,
                self.scenario.metabolism_max,
                (n,),
                endpoint=True,
            ),
            metabolism_spice=self.rng.integers(
                self.scenario.metabolism_min,
                self.scenario.metabolism_max,
                (n,),
                endpoint=True,
            ),
            vision=self.rng.integers(
                self.scenario.vision_min,
                self.scenario.vision_max,
                (n,),
                endpoint=True,
            ),
        )

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

    def run_model(self, step_count=1000):
        for _ in range(step_count):
            self.step()

# ===========================================================================
# Project Evaluation Harness Interface (prepare.py / parameter_sweeps.py)
# ===========================================================================

def create_model(seed: int = 42, **params) -> SugarscapeG1mt:
    """
    Instantiate SugarscapeG1mt model configured with seed and parameter overrides.
    Works seamlessly whether the model uses Mesa 4.x Scenario or Mesa 3.x kwargs.
    """
    p = dict(params)
    p.pop("steps", None)
    init_pop = p.get("initial_population", 200)

    if "SugarScapeScenario" in globals():
        scenario_kwargs = {k: v for k, v in p.items() if hasattr(SugarScapeScenario, k)}
        sc = SugarScapeScenario(rng=seed, **scenario_kwargs)
        model = SugarscapeG1mt(scenario=sc)
    else:
        if "rng" not in p and seed is not None:
            p["rng"] = seed
        model = SugarscapeG1mt(**p)

    model.initial_population = init_pop
    return model


def run_model(model: SugarscapeG1mt, steps: int = 200) -> dict[str, Any]:
    """
    Execute simulation steps and extract the standardized 9-key metrics dictionary.
    """
    try:
        model.run_model(step_count=steps)
    except (ValueError, IndexError) as e:
        return {"error": f"Mesa crash: {e}"}

    df = model.datacollector.get_model_vars_dataframe()
    cumul_trade_volume = int(df["Trade Volume"].sum()) if "Trade Volume" in df else 0

    agents = list(model.agents)
    agent_wealths = [float(a.sugar + a.spice) for a in agents]

    trade_prices = []
    for a in agents:
        if hasattr(a, "prices"):
            trade_prices.extend(a.prices)

    agent_positions = []
    for a in agents:
        if hasattr(a, "cell") and a.cell is not None:
            agent_positions.append(a.cell.coordinate)

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
