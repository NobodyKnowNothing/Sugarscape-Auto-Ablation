import math
import numpy as np

class State:
    def __init__(self, env, rng):
        self.rng = rng
        self.params = env
        self.w, self.h = 50, 50
        self.landscape = np.array(env["landscape"], dtype=float)
        self.caps = self.landscape.copy()
        self.sugar = self.landscape[0].copy()
        self.spice = self.landscape[1].copy()
        self.exchanges = []
        
        n = env["initial_population"]
        self.ids = np.arange(n)
        self.pos = np.array([self.rng.integers(0, 50, 2) for _ in range(n)])
        self.sugar_h = self.rng.integers(env["endowment_min"], env["endowment_max"] + 1, n).astype(float)
        self.spice_h = self.rng.integers(env["endowment_min"], env["endowment_max"] + 1, n).astype(float)
        self.met_s = self.rng.integers(env["metabolism_min"], env["metabolism_max"] + 1, n).astype(float)
        self.met_p = self.rng.integers(env["metabolism_min"], env["metabolism_max"] + 1, n).astype(float)
        self.vision = self.rng.integers(env["vision_min"], env["vision_max"] + 1, n)
        self.alive = np.ones(n, dtype=bool)

def get_welfare(s, p, ms, mp):
    return (s** (ms / (ms + mp))) * (p** (mp / (ms + mp)))

def get_mrs(s, p, ms, mp):
    return (p / mp) / (s / ms)

def init(env, rng):
    return State(env, rng)

def step(state, rng):
    state.exchanges = []
    regrowth = state.params["regrowth_rate"]
    state.sugar = np.minimum(state.sugar + regrowth, state.caps[0])
    state.spice = np.minimum(state.spice + regrowth, state.caps[1])
    
    indices = rng.permutation(len(state.ids))
    
    occupied = np.zeros((50, 50), dtype=bool)
    for i in range(len(state.ids)):
        if state.alive[i]: occupied[state.pos[i][0], state.pos[i][1]] = True

    for i in indices:
        if not state.alive[i]: continue
        
        # Move
        v = state.vision[i]
        x, y = state.pos[i]
        best_w, best_dist, best_cell = -1.0, float('inf'), None
        candidates = []
        
        for dx in range(-v, v + 1):
            for dy in range(-(v - abs(dx)), (v - abs(dx)) + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 50 and 0 <= ny < 50 and not occupied[nx, ny]:
                    w = get_welfare(state.sugar_h[i] + state.sugar[nx, ny], state.spice_h[i] + state.spice[nx, ny], state.met_s[i], state.met_p[i])
                    dist = math.dist((x, y), (nx, ny))
                    if (w, -dist) > (best_w, -best_dist):
                        best_w, best_dist, candidates = w, dist, [(nx, ny)]
                    elif (w, -dist) == (best_w, -best_dist):
                        candidates.append((nx, ny))
        
        if candidates:
            occupied[x, y] = False
            new_pos = candidates[rng.integers(0, len(candidates))]
            state.pos[i] = new_pos
            occupied[new_pos[0], new_pos[1]] = True
            
        # Eat
        cx, cy = state.pos[i]
        state.sugar_h[i] += state.sugar[cx, cy]
        state.spice_h[i] += state.spice[cx, cy]
        state.sugar[cx, cy] = 0
        state.spice[cx, cy] = 0
        state.sugar_h[i] -= state.met_s[i]
        state.spice_h[i] -= state.met_p[i]
        
        if state.sugar_h[i] <= 0 or state.spice_h[i] <= 0:
            state.alive[i] = False
            occupied[cx, cy] = False

    if state.params["enable_trade"]:
        rng.shuffle(indices)
        for i in indices:
            if not state.alive[i]: continue
            v = state.vision[i]
            x, y = state.pos[i]
            for j in indices:
                if i == j or not state.alive[j]: continue
                if abs(x - state.pos[j][0]) + abs(y - state.pos[j][1]) <= v:
                    while True:
                        s1, p1, ms1, mp1 = state.sugar_h[i], state.spice_h[i], state.met_s[i], state.met_p[i]
                        s2, p2, ms2, mp2 = state.sugar_h[j], state.spice_h[j], state.met_s[j], state.met_p[j]
                        mrs1, mrs2 = get_mrs(s1, p1, ms1, mp1), get_mrs(s2, p2, ms2, mp2)
                        if math.isclose(mrs1, mrs2): break
                        price = math.sqrt(mrs1 * mrs2)
                        s_ex, p_ex = (1, int(price)) if price >= 1 else (int(1/price), 1)
                        w1, w2 = get_welfare(s1, p1, ms1, mp1), get_welfare(s2, p2, ms2, mp2)
                        if mrs1 > mrs2:
                            if s1+s_ex > 0 and s2-s_ex > 0 and p1-p_ex > 0 and p2+p_ex > 0 and \
                               get_welfare(s1+s_ex, p1-p_ex, ms1, mp1) > w1 and \
                               get_welfare(s2-s_ex, p2+p_ex, ms2, mp2) > w2:
                                state.sugar_h[i] += s_ex; state.sugar_h[j] -= s_ex
                                state.spice_h[i] -= p_ex; state.spice_h[j] += p_ex
                                state.exchanges.append((state.ids[i], state.ids[j], (s_ex, -p_ex)))
                            else: break
                        else:
                            if s1-s_ex > 0 and s2+s_ex > 0 and p1+p_ex > 0 and p2-p_ex > 0 and \
                               get_welfare(s1-s_ex, p1+p_ex, ms1, mp1) > w1 and \
                               get_welfare(s2+s_ex, p2-p_ex, ms2, mp2) > w2:
                                state.sugar_h[i] -= s_ex; state.sugar_h[j] += s_ex
                                state.spice_h[i] += p_ex; state.spice_h[j] -= p_ex
                                state.exchanges.append((state.ids[i], state.ids[j], (-s_ex, p_ex)))
                            else: break

def observe(state):
    alive = state.alive
    return {
        "ids": state.ids[alive].tolist(),
        "positions": state.pos[alive].tolist(),
        "holdings": list(zip(state.sugar_h[alive], state.spice_h[alive])),
        "exchanges": state.exchanges,
        "resources": np.stack([state.sugar, state.spice]),
    }
