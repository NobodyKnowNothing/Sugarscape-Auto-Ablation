"""
Independent re-implementation of Mesa's Sugarscape G1MT (positive control).

Written from the rules, not from the Mesa code path: plain Python + numpy, no
Mesa, its own data structures and its own random streams. It implements the same
rules as the canonical model, so a valid judge must accept it as the same
society even though no trajectory is shared.

Rules (as in Mesa's sugarscape_g1mt):
  * growback: every cell regrows `regrowth_rate` of each good up to capacity
  * agents act in random order: move, eat, die if out of sugar or spice
  * move: among empty cells within Manhattan distance `vision` (the agent's own
    cell is occupied, so it is never a candidate), pick maximal Cobb-Douglas
    welfare, then nearest (euclidean, 1% tolerance), ties at random
  * trade (if enabled), agents in random order: with every agent on another cell
    within `vision`, repeatedly exchange at price sqrt(MRS_a * MRS_b) while both
    gain welfare, MRSs do not cross and holdings stay positive
"""

import math

import numpy as np


class World:
    def __init__(self, env, rng):
        self.rng = rng
        self.cap = np.array(env["landscape"], dtype=float)
        self.res = self.cap.copy()
        self.regrowth = env["regrowth_rate"]
        self.trade = env["enable_trade"]
        self.w, self.h = self.cap.shape[1], self.cap.shape[2]
        n = env["initial_population"]

        def draw(lo, hi):
            return rng.integers(env[lo], env[hi] + 1, n)

        self.pos = {}
        self.goods = {}
        self.metab = {}
        self.vision = {}
        self.occ = np.zeros((self.w, self.h), dtype=int)
        cells = rng.integers(0, self.w * self.h, n)
        sugar, spice = draw("endowment_min", "endowment_max"), draw("endowment_min", "endowment_max")
        ms, mp = draw("metabolism_min", "metabolism_max"), draw("metabolism_min", "metabolism_max")
        vis = draw("vision_min", "vision_max")
        for a in range(n):
            x, y = divmod(int(cells[a]), self.h)
            self.pos[a] = (x, y)
            self.occ[x, y] += 1
            self.goods[a] = [float(sugar[a]), float(spice[a])]
            self.metab[a] = (int(ms[a]), int(mp[a]))
            self.vision[a] = int(vis[a])
        self.exchanges = []
        self._balls = {}

    def ball(self, r):
        if r not in self._balls:
            self._balls[r] = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                              if abs(dx) + abs(dy) <= r]
        return self._balls[r]

    def cells_within(self, a, include_center):
        x, y = self.pos[a]
        for dx, dy in self.ball(self.vision[a]):
            if (dx or dy or include_center) and 0 <= x + dx < self.w and 0 <= y + dy < self.h:
                yield x + dx, y + dy

    def welfare(self, a, s, p):
        ms, mp = self.metab[a]
        return s ** (ms / (ms + mp)) * p ** (mp / (ms + mp))

    def mrs(self, a, s, p):
        ms, mp = self.metab[a]
        return (p / mp) / (s / ms)

    def move(self, a):
        options = [c for c in self.cells_within(a, True) if self.occ[c] == 0]
        if not options:
            return
        s, p = self.goods[a]
        wel = [self.welfare(a, s + self.res[0][c], p + self.res[1][c]) for c in options]
        best = max(wel)
        options = [c for c, v in zip(options, wel) if math.isclose(v, best)]
        here = self.pos[a]
        dist = [math.dist(here, c) for c in options]
        near = min(dist)
        options = [c for c, d in zip(options, dist) if math.isclose(d, near, rel_tol=1e-2)]
        target = options[int(self.rng.integers(len(options)))]
        self.occ[here] -= 1
        self.occ[target] += 1
        self.pos[a] = target

    def eat_or_die(self, a):
        c = self.pos[a]
        g = self.goods[a]
        for k in (0, 1):
            g[k] += self.res[k][c] - self.metab[a][k]
            self.res[k][c] = 0.0
        if g[0] <= 0 or g[1] <= 0:
            self.occ[c] -= 1
            for d in (self.pos, self.goods, self.metab, self.vision):
                del d[a]

    def bargain(self, a, b):
        while True:
            ga, gb = self.goods[a], self.goods[b]
            ma, mb = self.mrs(a, *ga), self.mrs(b, *gb)
            if math.isclose(ma, mb):
                return
            price = math.sqrt(ma * mb)
            seller, buyer = (a, b) if ma > mb else (b, a)   # seller gives spice, receives sugar
            sug, spi = (1, int(price)) if price >= 1 else (int(1 / price), 1)
            gs, gbuy = self.goods[seller], self.goods[buyer]
            new_s = (gs[0] + sug, gs[1] - spi)
            new_b = (gbuy[0] - sug, gbuy[1] + spi)
            if min(new_s + new_b) <= 0:
                return
            if not (self.welfare(seller, *gs) < self.welfare(seller, *new_s)
                    and self.welfare(buyer, *gbuy) < self.welfare(buyer, *new_b)):
                return
            if not self.mrs(seller, *new_s) > self.mrs(buyer, *new_b):
                return
            gs[0], gs[1] = new_s
            gbuy[0], gbuy[1] = new_b
            self.exchanges.append((seller, buyer, (sug, -spi)))

    def step(self):
        self.exchanges = []
        np.minimum(self.res + self.regrowth, self.cap, out=self.res)
        for a in self.rng.permutation(list(self.pos)):
            a = int(a)
            self.move(a)
            self.eat_or_die(a)
        if self.trade:
            by_cell = {}
            for a, c in self.pos.items():
                by_cell.setdefault(c, []).append(a)
            for a in self.rng.permutation(list(self.pos)):
                a = int(a)
                for c in self.cells_within(a, False):
                    for b in by_cell.get(c, ()):
                        self.bargain(a, b)


def init(env, rng):
    return World(env, rng)


def step(world, rng):
    world.step()


def observe(world):
    ids = list(world.pos)
    return {
        "ids": ids,
        "positions": [world.pos[a] for a in ids],
        "holdings": [world.goods[a] for a in ids],
        "exchanges": world.exchanges,
        "resources": world.res,
    }
