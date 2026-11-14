# Sugarscape Auto-Ablation: Behavioral Clustering & Attractor Report (`full`)

- **Updated**: `2026-11-14T14:27:16+00:00`
- **Active Trajectory**: `Trajectory #2` (`1` plateau resets completed)
- **Total Attempts Logged**: `67` (`48` with full behavioral vectors)

![Behavioral Attractor Map](./attractor_map.svg)

---

## 1. Multi-Trajectory Compression & Plateau Reset History

Each trajectory starts from the canonical flattened Mesa Sugarscape (`canonical_strategy.py`), compresses complexity $C$ until a statistically significant plateau is reached, and resets to explore an independent ablation path.

| Trajectory | Status | Generations | Attempts | Origin $C$ | Champion $C$ | Saved ($\Delta C$) | NCD vs. Canonical | Modified Mechanisms | Plateau Reason |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **T1** | ✅ Plateaued & Reset | 10 | 21 | `6158` B | `6142` B (`6bcb34abcfe2fdf2`) | **-16 B** | `0.173` | mrs_stop:removed | diminishing_returns: relative C reduction over last 6 generations is 0.0000% (< 0.20% threshold) with 0 mechanism-level acceptances |
| **T2** | 🔄 Active | — | — | — | `1175` B (`5869928c303c094b`) | — | `0.858` | (in progress) | active trajectory |

### Cross-Trajectory Champion Convergence (Pairwise AST Normalized Compression Distance)

| Champion | T1 | T2 |
| :--- | :---: | :---: |
| **T1** | `0.000` | `0.860` |
| **T2** | `0.867` | `0.000` |

---

## 2. Failure Attractors (Clustered Rejection Modes)

Rejected attempts are clustered in standardized behavioral response space and matched against the 17 pre-registered calibration controls/knockouts.

| Cluster | Archetype | Count | Trajectories Hit | Cross-Trajectory? | Failed Features | Top Failing Statistics | Nearest Control | Targeted Mechanisms |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **`F1`** | E1+E2 Breakdown (~ko_ignore_regrowth) | `2` | T1, T2 | 🔥 **YES** | `E1, E2, E3, E4, E5` | `disp_late(slope), loghill_T(slope), moran_T(slope)` | `ko_ignore_regrowth` | mesa_framework (1), growback (1) |
| **`F2`** | E2+E3 Breakdown (~ko_no_mrs_stop) | `2` | T2 | No | `E2, E3` | `pop_traj(l2), plateau_frac(l2), disp_traj(l2)` | `ko_no_mrs_stop` | move_tiebreak (2) |
| **`F3`** | E1+E2 Breakdown (~ko_ignore_endowment) | `2` | T1, T2 | 🔥 **YES** | `E1, E2, E3, E4, E5` | `pop_traj(l2), plateau_frac(l2), disp_traj(l2)` | `ko_ignore_endowment` | move_tiebreak (1), two_goods (1) |
| **`F4`** | E1+E2 Breakdown (~ko_ignore_endowment) | `1` | T2 | No | `E1, E2, E3, E4` | `—` | `ko_ignore_endowment` | heterogeneity (1) |
| **`F5`** | E1+E2 Breakdown (~ko_ignore_regrowth) | `1` | T2 | No | `E1, E2, E3, E4, E5` | `plateau_frac(slope), pop_traj(l2)` | `ko_ignore_regrowth` | heterogeneity (1) |
| **`G0`** | Stage 0 Gate Rejection (Complexity / Guard / Smoke) | `30` | T1, T2 | 🔥 **YES** | `Stage 0/1` | `—` | `stage_0_gate` | mesa_framework (9), none (5), two_goods (4) |
| **`G1`** | Stage 1 Gate Rejection (Complexity / Guard / Smoke) | `1` | T2 | No | `Stage 0/1` | `—` | `stage_1_gate` | none (1) |

---

## 3. Mechanism Necessity & Feature Sensitivity Matrix

Worst-case feature discrepancy ratio ($\text{ratio} < 1.0$ passes equivalence; $\ge 1.0$ fails) observed when each Sugarscape mechanism was targeted for ablation.

| Mechanism | Classification | Attempts (Acc / Rej) | Trajectories Modified | Max E1 | Max E2 | Max E3 | Max E4 | Max E5 | Primary Failure Attractor |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`growback`** | untested/under-explored | `1` (`0` / `1`) | — | `1.99` | `1.61` | `1.81` | `1.58` | `3.13` | `F1` |
| **`move_welfare`** | untested/under-explored | `0` (`0` / `0`) | — | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `none` |
| **`move_tiebreak`** | compressible | `6` (`2` / `4`) | — | `1.27` | `2.22` | `1.43` | `1.01` | `1.55` | `F2` |
| **`welfare`** | untested/under-explored | `1` (`0` / `1`) | — | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `G0` |
| **`metabolism`** | untested/under-explored | `1` (`0` / `1`) | — | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `G0` |
| **`death`** | untested/under-explored | `0` (`0` / `0`) | — | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `none` |
| **`vision`** | untested/under-explored | `0` (`0` / `0`) | — | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `none` |
| **`trade`** | load-bearing (candidate) | `3` (`0` / `3`) | — | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `G0` |
| **`mrs_stop`** | compressible | `3` (`2` / `1`) | T1 | `0.79` | `0.55` | `0.79` | `0.54` | `0.78` | `G0` |
| **`price_rule`** | compressible | `2` (`1` / `1`) | — | `0.70` | `0.62` | `0.49` | `0.63` | `0.62` | `G0` |
| **`heterogeneity`** | load-bearing (candidate) | `3` (`0` / `3`) | — | `2.98` | `9.85` | `4.36` | `4.94` | `14.59` | `G0` |
| **`two_goods`** | load-bearing (candidate) | `5` (`0` / `5`) | — | `1.06` | `1.77` | `1.36` | `1.22` | `1.64` | `G0` |
| **`random_activation`** | compressible | `4` (`1` / `3`) | — | `0.65` | `0.53` | `0.87` | `0.56` | `0.99` | `G0` |
| **`mesa_framework`** | compressible | `11` (`1` / `10`) | T2 | `1.73` | `1.75` | `1.85` | `1.34` | `4.53` | `G0` |
