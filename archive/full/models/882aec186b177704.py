import math
import numpy as np

class Agent:
    def __init__(self, uid, x, y, sugar, spice, m_sugar, m_spice, vision):
        self.uid = uid
        self.x, self.y = x, y
        self.sugar = sugar
        self.spice = spice
        self.m_sugar = m_sugar
        self.m_spice = m_spice
        self.vision = vision

    def welfare(self, s, p):
        mt = self.m_sugar + self.m_spice
        return (s ** (self.m_sugar / mt)) * (p ** (self.m_spice / mt))

    def mrs(self, s, p):
        return (p / self.m_spice) / (s / self.m_sugar)

    def move(self, model, rng):
        best_cell = None
        best_val = (float('inf'), float('inf'), float('inf'))
        
        x, y = self.x, self.y
        v = self.vision
        for dx in range(-v, v + 1):
            for dy in range(-(v - abs(dx)), (v - abs(dx)) + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < model.w and 0 <= ny < model.h:
                    if model.agent_grid[nx, ny] is None:
                        w = self.welfare(self.sugar + model.sugar_map[nx, ny], 
                                         self.spice + model.spice_map[nx, ny])
                        dist = math.sqrt(dx*dx + dy*dy)
                        val = (-w, dist, rng.random())
                        if val < best_val:
                            best_val = val
                            best_cell = (nx, ny)
        
        if best_cell:
            model.agent_grid[self.x, self.y] = None
            self.x, self.y = best_cell
            model.agent_grid[self.x, self.y] = self

    def eat(self, model):
        self.sugar += model.sugar_map[self.x, self.y]
        model.sugar_map[self.x, self.y] = 0
        self.sugar -= self.m_sugar
        self.spice += model.spice_map[self.x, self.y]
        model.spice_map[self.x, self.y] = 0
        self.spice -= self.m_spice

    def trade(self, other, model):
        if self.sugar <= 0 or self.spice <= 0 or other.sugar <= 0 or other.spice <= 0:
            return
        
        m_s = self.mrs(self.sugar, self.spice)
        m_o = other.mrs(other.sugar, other.spice)
        if m_s == m_o: return
        
        price = math.sqrt(m_s * m_o)
        buyer, seller = (self, other) if m_s > m_o else (other, self)
        
        sq, sp = (1, int(price)) if price >= 1 else (int(1 / price), 1)
        
        # Check resources and welfare
        if buyer.sugar < sq or seller.spice < sp: return
        
        w_b = buyer.welfare(buyer.sugar, buyer.spice)
        w_s = seller.welfare(seller.sugar, seller.spice)
        
        if (buyer.welfare(buyer.sugar + sq, buyer.spice - sp) > w_b and 
            seller.welfare(seller.sugar - sq, seller.spice + sp) > w_s and 
            buyer.mrs(buyer.sugar + sq, buyer.spice - sp) > seller.mrs(seller.sugar - sq, seller.spice + sp)):
            
            buyer.sugar += sq
            buyer.spice -= sp
            seller.sugar -= sq
            seller.spice += sp
            model.exchanges.append((seller.uid, buyer.uid, (sq, -sp)) if seller == self else (buyer.uid, seller.uid, (-sq, sp)))
            self.trade(other, model)

class SugarscapeG1mt:
    def __init__(self, env, rng):
        self.w, self.h = 50, 50
        self.sugar_cap = np.array(env["landscape"][0], dtype=float)
        self.spice_cap = np.array(env["landscape"][1], dtype=float)
        self.sugar_map = self.sugar_cap.copy()
        self.spice_map = self.spice_cap.copy()
        self.regrowth = env["regrowth_rate"]
        self.enable_trade = env["enable_trade"]
        self.exchanges = []
        self.agent_grid = np.full((self.w, self.h), None)
        
        n = env["initial_population"]
        self.agents = []
        coords = [(x, y) for x in range(self.w) for y in range(self.h)]
        chosen_coords = rng.choice(coords, n, replace=False)
        
        for i in range(n):
            a = Agent(i, *chosen_coords[i], 
                      rng.integers(env["endowment_min"], env["endowment_max"] + 1),
                      rng.integers(env["endowment_min"], env["endowment_max"] + 1),
                      rng.integers(env["metabolism_min"], env["metabolism_max"] + 1),
                      rng.integers(env["metabolism_min"], env["metabolism_max"] + 1),
                      rng.integers(env["vision_min"], env["vision_max"] + 1))
            self.agents.append(a)
            self.agent_grid[a.x, a.y] = a

    def step(self, rng):
        self.exchanges = []
        self.sugar_map[:] = np.minimum(self.sugar_map + self.regrowth, self.sugar_cap)
        self.spice_map[:] = np.minimum(self.spice_map + self.regrowth, self.spice_cap)
        
        # Phase 1: Move, Eat, Die
        order = list(self.agents)
        rng.shuffle(order)
        for a in order:
            if a in self.agents:
                a.move(self, rng)
                a.eat(self)
                if a.sugar <= 0 or a.spice <= 0:
                    self.agent_grid[a.x, a.y] = None
                    self.agents.remove(a)
        
        # Phase 2: Trade
        if self.enable_trade:
            order = list(self.agents)
            rng.shuffle(order)
            for a in order:
                v = a.vision
                for dx in range(-v, v + 1):
                    for dy in range(-(v - abs(dx)), (v - abs(dx)) + 1):
                        nx, ny = a.x + dx, a.y + dy
                        if 0 <= nx < self.w and 0 <= ny < self.h:
                            other = self.agent_grid[nx, ny]
                            if other and other != a:
                                a.trade(other, self)

def init(env, rng):
    return SugarscapeG1mt(env, rng)

def step(model, rng):
    model.step(rng)

def observe(model):
    return {
        "ids": [a.uid for a in model.agents],
        "positions": [(a.x, a.y) for a in model.agents],
        "holdings": [(a.sugar, a.spice) for a in model.agents],
        "exchanges": model.exchanges,
        "resources": np.stack([model.sugar_map, model.spice_map]),
    }
