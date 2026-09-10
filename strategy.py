"""
Canonical Mesa Sugarscape G1MT — flattened module ported to the judge contract.

Generated from upstream Mesa Sugarscape with Traders (via flatten_sugarscape.py),
then ported to the judge's model contract (see judge/contract.py):

    init(env, rng) -> state
    step(state, rng) -> None
    observe(state) -> dict

Port changes (behaviour-neutral at the default environment):
  1. The landscape comes from env["landscape"] instead of reading sugar-map.txt.
  2. Growback adds env["regrowth_rate"] per step (default 1, as upstream).
  3. Each executed trade is appended to model.exchanges so the judge can observe it.
  4. Measurement-only code (DataCollector, prices/trade_partners logs) is removed;
     the judge observes the model itself.
"""

import math

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

def get_distance(cell_1, cell_2):
    """
    Calculate the Euclidean distance between two positions

    used in trade.move()
    """

    x1, y1 = cell_1.coordinate
    x2, y2 = cell_2.coordinate
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx**2 + dy**2)

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

    def get_trader(self, cell):
        """
        helper function used in self.trade_with_neighbors()
        """

        for agent in cell.agents:
            if isinstance(agent, Trader):
                return agent

    def calculate_welfare(self, sugar, spice):
        """
        helper function

        part 2 self.move()
        self.trade()
        """

        # calculate total resources
        m_total = self.metabolism_sugar + self.metabolism_spice
        # Cobb-Douglas functional form; starting on p. 97
        # on Growing Artificial Societies
        return sugar ** (self.metabolism_sugar / m_total) * spice ** (
            self.metabolism_spice / m_total
        )

    def is_starved(self):
        """
        Helper function for self.maybe_die()
        """

        return (self.sugar <= 0) or (self.spice <= 0)

    def calculate_MRS(self, sugar, spice):
        """
        Helper function for
          - self.trade()
          - self.maybe_self_spice()

        Determines what trader agent needs and can give up
        """

        return (spice / self.metabolism_spice) / (sugar / self.metabolism_sugar)

    def calculate_sell_spice_amount(self, price):
        """
        helper function for self.maybe_sell_spice() which is called from
        self.trade()
        """

        if price >= 1:
            sugar = 1
            spice = int(price)
        else:
            sugar = int(1 / price)
            spice = 1
        return sugar, spice

    def sell_spice(self, other, sugar, spice):
        """
        used in self.maybe_sell_spice()

        exchanges sugar and spice between traders
        """

        self.sugar += sugar
        other.sugar -= sugar
        self.spice -= spice
        other.spice += spice
        self.model.exchanges.append((self.unique_id, other.unique_id, (sugar, -spice)))

    def maybe_sell_spice(self, other, price, welfare_self, welfare_other):
        """
        helper function for self.trade()
        """

        sugar_exchanged, spice_exchanged = self.calculate_sell_spice_amount(price)

        # Assess new sugar and spice amount - what if change did occur
        self_sugar = self.sugar + sugar_exchanged
        other_sugar = other.sugar - sugar_exchanged
        self_spice = self.spice - spice_exchanged
        other_spice = other.spice + spice_exchanged

        # double check to ensure agents have resources

        if (
            (self_sugar <= 0)
            or (other_sugar <= 0)
            or (self_spice <= 0)
            or (other_spice <= 0)
        ):
            return False

        # trade criteria #1 - are both agents better off?
        both_agents_better_off = (
            welfare_self < self.calculate_welfare(self_sugar, self_spice)
        ) and (welfare_other < other.calculate_welfare(other_sugar, other_spice))

        # trade criteria #2 is their mrs crossing with potential trade
        mrs_not_crossing = self.calculate_MRS(
            self_sugar, self_spice
        ) > other.calculate_MRS(other_sugar, other_spice)

        if not (both_agents_better_off and mrs_not_crossing):
            return False

        # criteria met, execute trade
        self.sell_spice(other, sugar_exchanged, spice_exchanged)

        return True

    def trade(self, other):
        """
        helper function used in trade_with_neighbors()

        other is a trader agent object
        """

        # sanity check to verify code is working as expected
        assert self.sugar > 0
        assert self.spice > 0
        assert other.sugar > 0
        assert other.spice > 0

        # calculate marginal rate of substitution in Growing Artificial Societies p. 101
        mrs_self = self.calculate_MRS(self.sugar, self.spice)
        mrs_other = other.calculate_MRS(other.sugar, other.spice)

        # calculate each agents welfare
        welfare_self = self.calculate_welfare(self.sugar, self.spice)
        welfare_other = other.calculate_welfare(other.sugar, other.spice)

        if math.isclose(mrs_self, mrs_other):
            return

        # calculate price
        price = math.sqrt(mrs_self * mrs_other)

        if mrs_self > mrs_other:
            # self is a sugar buyer, spice seller
            sold = self.maybe_sell_spice(other, price, welfare_self, welfare_other)
            # no trade - criteria not met
            if not sold:
                return
        else:
            # self is a spice buyer, sugar seller
            sold = other.maybe_sell_spice(self, price, welfare_other, welfare_self)
            # no trade - criteria not met
            if not sold:
                return

        # continue trading
        self.trade(other)

    ######################################################################
    #                                                                    #
    #                      MAIN TRADE FUNCTIONS                          #
    #                                                                    #
    ######################################################################

    def move(self):
        """
        Function for trader agent to identify optimal move for each step in 4 parts
        1 - identify all possible moves
        2 - determine which move maximizes welfare
        3 - find closest best option
        4 - move
        """

        # 1. identify all possible moves

        neighboring_cells = [
            cell
            for cell in self.cell.get_neighborhood(self.vision, include_center=True)
            if cell.is_empty
        ]

        if not neighboring_cells:
            # all neighboring cells are occupied
            return

        # 2. determine which move maximizes welfare

        welfares = [
            self.calculate_welfare(
                self.sugar + cell.sugar,
                self.spice + cell.spice,
            )
            for cell in neighboring_cells
        ]

        # 3. Find closest best option

        # find the highest welfare in welfares
        max_welfare = max(welfares)
        # Get cells with the highest welfare
        candidates = [
            cell
            for cell, welfare in zip(neighboring_cells, welfares)
            if math.isclose(welfare, max_welfare)
        ]

        min_dist = min(get_distance(self.cell, cell) for cell in candidates)

        final_candidates = [
            cell
            for cell in candidates
            if math.isclose(get_distance(self.cell, cell), min_dist, rel_tol=1e-02)
        ]

        # 4. Move Agent
        self.cell = self.random.choice(final_candidates)

    def eat(self):
        self.sugar += self.cell.sugar
        self.cell.sugar = 0
        self.sugar -= self.metabolism_sugar

        self.spice += self.cell.spice
        self.cell.spice = 0
        self.spice -= self.metabolism_spice

    def maybe_die(self):
        """
        Function to remove Traders who have consumed all their sugar or spice
        """

        if self.is_starved():
            self.remove()

    def step(self):
        """Agent step method."""
        self.move()
        self.eat()
        self.maybe_die()

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
# Sugarscape Model (from model.py)
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
    regrowth_rate: float = 1


class SugarscapeG1mt(mesa.Model):
    """
    Manager class to run Sugarscape with Traders
    """

    def __init__(self, landscape, scenario: SugarScapeScenario = SugarScapeScenario):
        super().__init__(scenario=scenario)
        # Initiate width and height of sugarscape
        self.width = landscape.shape[1]
        self.height = landscape.shape[2]

        # Initiate population attributes
        self.enable_trade = self.scenario.enable_trade
        self.running = True
        self.exchanges = []

        # initiate mesa grid class
        self.grid = OrthogonalVonNeumannGrid(
            (self.width, self.height), torus=False, random=self.random
        )

        # landscape supplied by the environment (sugar-map.txt and its mirror)
        self.sugar_distribution = np.array(landscape[0], dtype=float)
        self.spice_distribution = np.array(landscape[1], dtype=float)

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
        self.exchanges = []
        regrowth = self.scenario.regrowth_rate
        self.grid.sugar[:] = np.minimum(self.grid.sugar + regrowth, self.sugar_distribution)
        self.grid.spice[:] = np.minimum(self.grid.spice + regrowth, self.spice_distribution)

        # step trader agents
        self.agents.shuffle_do("step")
        if self.enable_trade:
            self.agents.shuffle_do("trade_with_neighbors")


# ===========================================================================
# Judge contract
# ===========================================================================

SCENARIO_KEYS = (
    "initial_population", "endowment_min", "endowment_max", "metabolism_min",
    "metabolism_max", "vision_min", "vision_max", "enable_trade", "regrowth_rate",
)


def init(env, rng):
    scenario = SugarScapeScenario(
        rng=int(rng.integers(2**31)), **{k: env[k] for k in SCENARIO_KEYS}
    )
    return SugarscapeG1mt(env["landscape"], scenario=scenario)


def step(model, rng):
    model.step()


def observe(model):
    agents = list(model.agents)
    return {
        "ids": [a.unique_id for a in agents],
        "positions": [a.cell.coordinate for a in agents],
        "holdings": [(a.sugar, a.spice) for a in agents],
        "exchanges": model.exchanges,
        "resources": np.stack([model.grid.sugar.data, model.grid.spice.data]),
    }
