"""
Null society (negative control): inequality without a society.

Agents random-walk on the grid, ignore resources, never interact, and their
wealth follows a multiplicative random process (a classic way to get a skewed
wealth distribution with no social mechanism at all). Agents die when wealth
falls below a threshold. A valid judge must reject this as Sugarscape: it has
no exchanges (E3) and no concentration on resources (E4).
"""

import numpy as np


class World:
    def __init__(self, env, rng):
        n = env["initial_population"]
        self.w, self.h = env["width"], env["height"]
        self.pos = rng.integers(0, [self.w, self.h], size=(n, 2))
        self.wealth = rng.integers(env["endowment_min"], env["endowment_max"] + 1, size=(n, 2)).astype(float)
        self.ids = np.arange(n)
        self.res = np.array(env["landscape"], dtype=float)

    def step(self, rng):
        moves = rng.integers(-1, 2, size=self.pos.shape)
        self.pos = np.clip(self.pos + moves, 0, [self.w - 1, self.h - 1])
        self.wealth *= rng.lognormal(-0.01, 0.12, size=self.wealth.shape)
        alive = (self.wealth > 5).all(axis=1)
        self.pos, self.wealth, self.ids = self.pos[alive], self.wealth[alive], self.ids[alive]


def init(env, rng):
    return World(env, rng)


def step(world, rng):
    world.step(rng)


def observe(world):
    return {
        "ids": world.ids,
        "positions": world.pos,
        "holdings": world.wealth,
        "exchanges": [],
        "resources": world.res,
    }
