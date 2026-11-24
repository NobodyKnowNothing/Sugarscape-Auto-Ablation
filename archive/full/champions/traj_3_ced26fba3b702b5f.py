import numpy as np

def init(env, rng):
    n = env["initial_population"]
    state = {
        "n": n,
        "caps": np.array(env["landscape"], dtype=float),
        "maps": np.array(env["landscape"], dtype=float),
        "regrowth": env["regrowth_rate"],
        "enable_trade": env["enable_trade"],
        "exchanges": [],
        "grid": np.full((50, 50), -1, dtype=int),
        "pos": np.zeros((n, 2), dtype=int),
        "h": np.zeros((n, 2), dtype=float),
        "m": np.zeros((n, 2), dtype=float),
        "v": np.zeros(n, dtype=int),
        "alive": np.ones(n, dtype=bool)
    }
    coords = rng.choice([x*50+y for x in range(50) for y in range(50)], n, replace=False)
    for i in range(n):
        x, y = divmod(coords[i], 50)
        state["pos"][i] = [x, y]
        state["grid"][x, y] = i
        state["h"][i] = [rng.integers(env["endowment_min"], env["endowment_max"] + 1) for _ in range(2)]
        state["m"][i] = [rng.integers(env["metabolism_min"], env["metabolism_max"] + 1) for _ in range(2)]
        state["v"][i] = rng.integers(env["vision_min"], env["vision_max"] + 1)
    return state

def step(s, rng):
    s["exchanges"] = []
    s["maps"] = np.minimum(s["maps"] + s["regrowth"], s["caps"])
    
    order = np.arange(s["n"])
    rng.shuffle(order)
    for i in order:
        if not s["alive"][i]: continue
        x, y = s["pos"][i]
        v = s["v"][i]
        best_cell, best_val = None, (float('inf'), float('inf'), float('inf'))
        for dx in range(-v, v + 1):
            for dy in range(-(v - abs(dx)), (v - abs(dx)) + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 50 and 0 <= ny < 50 and s["grid"][nx, ny] == -1:
                    h_next = s["h"][i] + s["maps"][:, nx, ny]
                    w = (h_next[0]**s["m"][i, 0]) * (h_next[1]**s["m"][i, 1])
                    val = (-w, dx*dx + dy*dy, rng.random())
                    if val < best_val: best_val, best_cell = val, (nx, ny)
        if best_cell:
            s["grid"][x, y] = -1
            s["pos"][i] = best_cell
            s["grid"][best_cell[0], best_cell[1]] = i
        
        s["h"][i] += s["maps"][:, s["pos"][i][0], s["pos"][i][1]]
        s["maps"][:, s["pos"][i][0], s["pos"][i][1]] = 0
        s["h"][i] -= s["m"][i]
        if np.any(s["h"][i] <= 0):
            s["grid"][s["pos"][i][0], s["pos"][i][1]] = -1
            s["alive"][i] = False

    if s["enable_trade"]:
        rng.shuffle(order)
        for i in order:
            if not s["alive"][i]: continue
            x, y = s["pos"][i]
            v = s["v"][i]
            for dx in range(-v, v + 1):
                for dy in range(-(v - abs(dx)), (v - abs(dx)) + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50:
                        j = s["grid"][nx, ny]
                        if j != -1 and j != i:
                            while True:
                                if np.any(s["h"][i] <= 0) or np.any(s["h"][j] <= 0): break
                                mrs_i = (s["h"][i, 1] * s["m"][i, 0]) / (s["h"][i, 0] * s["m"][i, 1])
                                mrs_j = (s["h"][j, 1] * s["m"][j, 0]) / (s["h"][j, 0] * s["m"][j, 1])
                                price = (mrs_i * mrs_j)**0.5
                                b, sl = (i, j) if mrs_i > mrs_j else (j, i)
                                sq, sp = (1, int(price)) if price >= 1 else (int(1 / price), 1)
                                if s["h"][b, 0] < sq or s["h"][sl, 1] < sp: break
                                w_b, w_sl = (s["h"][b, 0]**s["m"][b, 0] * s["h"][b, 1]**s["m"][b, 1]), (s["h"][sl, 0]**s["m"][sl, 0] * s["h"][sl, 1]**s["m"][sl, 1])
                                nb_h, ns_h = [s["h"][b, 0] + sq, s["h"][b, 1] - sp], [s["h"][sl, 0] - sq, s["h"][sl, 1] + sp]
                                if (nb_h[0]**s["m"][b, 0] * nb_h[1]**s["m"][b, 1] > w_b and 
                                    ns_h[0]**s["m"][sl, 0] * ns_h[1]**s["m"][sl, 1] > w_sl and 
                                    (nb_h[1] * s["m"][b, 0]) / (nb_h[0] * s["m"][b, 1]) > (ns_h[1] * s["m"][sl, 0]) / (ns_h[0] * s["m"][sl, 1])):
                                    s["h"][b] += [sq, -sp]
                                    s["h"][sl] += [-sq, sp]
                                    s["exchanges"].append((sl, b, (sq, -sp)) if sl == i else (b, sl, (-sq, sp)))
                                else: break

def observe(s):
    alive_idx = np.where(s["alive"])[0]
    return {
        "ids": alive_idx.tolist(),
        "positions": s["pos"][alive_idx].tolist(),
        "holdings": s["h"][alive_idx].tolist(),
        "exchanges": s["exchanges"],
        "resources": s["maps"],
    }
