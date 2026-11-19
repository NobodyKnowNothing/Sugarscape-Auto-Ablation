import math
import numpy as np

class Agent:
    def __init__(self, uid, x, y, h, m, vision):
        self.uid = uid
        self.x, self.y = x, y
        self.h = h # [sugar, spice]
        self.m = m # [m_sugar, m_spice]
        self.vision = vision

    def welfare(self, h):
        mt = sum(self.m)
        return (h[0] ** (self.m[0] / mt)) * (h[1] ** (self.m[1] / mt))

    def mrs(self, h):
        return (h[1] / self.m[1]) / (h[0] / self.m[0])

    def move(self, model, rng):
        best_cell = None
        best_val = (float('inf'), float('inf'), float('inf'))
        x, y = self.x, self.y
        v = self.vision
        for dx in range(-v, v + 1):
            for dy in range(-(v - abs(dx)), (v - abs(dx)) + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 50 and 0 <= ny < 50 and model.agent_grid[nx, ny] is None:
                    h_next = [self.h[0] + model.maps[0, nx, ny], self.h[1] + model.maps[1, nx, ny]]
                    w = self.welfare(h_next)
                    val = (-w, dx*dx + dy*dy, rng.random())
                    if val < best_val:
                        best_val, best_cell = val, (nx, ny)
        if best_cell:
            model.agent_grid[self.x, self.y] = None
            self.x, self.y = best_cell
            model.agent_grid[self.x, self.y] = self

    def eat(self, model):
        for i in range(2):
            self.h[i] += model.maps[i, self.x, self.y]
            model.maps[i, self.x, self.y] = 0
            self.h[i] -= self.m[i]

    def trade(self, other, model):
        if any(v <= 0 for v in self.h + other.h): return
        m_s, m_o = self.mrs(self.h), other.mrs(other.h)
        if m_s == m_o: return
        price = math.sqrt(m_s * m_o)
        buyer, seller = (self, other) if m_s > m_o else (other, self)
        sq, sp = (1, int(price)) if price >= 1 else (int(1 / price), 1)
        if buyer.h[0] < sq or seller.h[1] < sp: return
        w_b, w_s = buyer.welfare(buyer.h), seller.welfare(seller.h)
        nb_h, ns_h = [buyer.h[0] + sq, buyer.h[1] - sp], [seller.h[0] - sq, seller.h[1] + sp]
        if (buyer.welfare(nb_h) > w_b and seller.welfare(ns_h) > w_s and 
            buyer.mrs(nb_h) > seller.mrs(ns_h)):
            buyer.h[0] += sq; buyer.h[1] -= sp
            seller.h[0] -= sq; seller.h[1] += sp
            model.exchanges.append((seller.uid, buyer.uid, (sq, -sp)) if seller == self else (buyer.uid, seller.uid, (-sq, sp)))
            self.trade(other, model)

class SugarscapeG1mt:
    def __init__(self, env, rng):
        self.w, self.h = 50, 50
        self.caps = np.array(env["landscape"], dtype=float)
        self.maps = self.caps.copy()
        self.regrowth = env["regrowth_rate"]
        self.enable_trade = env["enable_trade"]
        self.exchanges = []
        self.agent_grid = np.full((50, 50), None)
        n = env["initial_population"]
        self.agents = []
        coords = rng.choice([(x, y) for x in range(50) for y in range(50)], n, replace=False)
        for i in range(n):
            a = Agent(i, *coords[i], 
                      [rng.integers(env["endowment_min"], env["endowment_max"] + 1) for _ in range(2)],
                      [rng.integers(env["metabolism_min"], env["metabolism_max"] + 1) for _ in range(2)],
                      rng.integers(env["vision_min"], env["vision_max"] + 1))
            self.agents.append(a)
            self.agent_grid[a.x, a.y] = a

    def step(self, rng):
        self.exchanges = []
        self.maps[:] = np.minimum(self.maps + self.regrowth, self.caps)
        order = list(self.agents)
        rng.shuffle(order)
        for a in order:
            if a in self.agents:
                a.move(self, rng)
                a.eat(self)
                if a.h[0] <= 0 or a.h[1] <= 0:
                    self.agent_grid[a.x, a.y] = None
                    self.agents.remove(a)
        if self.enable_trade:
            order = list(self.agents)
            rng.shuffle(order)
            for a in order:
                v = a.vision
                for dx in range(-v, v + 1):
                    for dy in range(-(v - abs(dx)), (v - abs(dx)) + 1):
                        nx, ny = a.x + dx, a.y + dy
                        if 0 <= nx < 50 and 0 <= ny < 50:
                            other = self.agent_grid[nx, ny]
                            if other and other != a: a.trade(other, self)

def init(env, rng): return SugarscapeG1mt(env, rng)
def step(model, rng): model.step(rng)
def observe(model):
    return {
        "ids": [a.uid for a in model.agents],
        "positions": [(a.x, a.y) for a in model.agents],
        "holdings": [tuple(a.h) for a in model.agents],
        "exchanges": model.exchanges,
        "resources": model.maps,
    }
