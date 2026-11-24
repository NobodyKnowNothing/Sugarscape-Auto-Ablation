# Sugarscape Auto-Ablation: Behavioral Clustering & Attractor Report (`full`)

- **Updated**: `2026-11-24T17:20:20+00:00`
- **Active Trajectory**: `Trajectory #4` (`3` plateau resets completed)
- **Total Attempts Logged**: `144` (`102` with full behavioral vectors)

![Behavioral Attractor Map](./attractor_map.svg)

---

## 1. Multi-Trajectory Compression & Plateau Reset History

Each trajectory starts from the canonical flattened Mesa Sugarscape (`canonical_strategy.py`), compresses complexity $C$ until a statistically significant plateau is reached, and resets to explore an independent ablation path.

| Trajectory | Status | Generations | Attempts | Origin $C$ | Champion $C$ | Saved ($\Delta C$) | NCD vs. Canonical | Modified Mechanisms | Plateau Reason |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **T1** | ✅ Plateaued & Reset | 10 | 21 | `6158` B | `6142` B (`6bcb34abcfe2fdf2`) | **-16 B** | `0.173` | mrs_stop:removed | diminishing_returns: relative C reduction over last 6 generations is 0.0000% (< 0.20% threshold) with 0 mechanism-level acceptances |
| **T2** | ✅ Plateaued & Reset | 15 | 61 | `6158` B | `1175` B (`5869928c303c094b`) | **-4983 B** | `0.858` | mesa_framework:replaced | frontier_stagnation: 6 generations (27 attempts) without C_min improvement (95% upper bound on per-attempt improvement rate p < 0.105) |
| **T3** | ✅ Plateaued & Reset | 17 | 59 | `6158` B | `1074` B (`ced26fba3b702b5f`) | **-5084 B** | `0.875` | move_tiebreak:replaced, mesa_framework:replaced | diminishing_returns: relative C reduction over last 6 generations is 0.0000% (< 0.20% threshold) with 0 mechanism-level acceptances |
| **T4** | 🔄 Active | — | — | — | `6158` B (`af314acdce5789e2`) | — | `0.000` | (in progress) | active trajectory |

### Cross-Trajectory Champion Convergence (Pairwise AST Normalized Compression Distance)

| Champion | T1 | T2 | T3 | T4 |
| :--- | :---: | :---: | :---: | :---: |
| **T1** | `0.000` | `0.860` | `0.877` | `0.169` |
| **T2** | `0.867` | `0.000` | `0.785` | `0.867` |
| **T3** | `0.882` | `0.776` | `0.000` | `0.882` |
| **T4** | `0.173` | `0.858` | `0.875` | `0.000` |

---

## 2. Failure Attractors (Clustered Rejection Modes)

Rejected attempts are clustered in standardized behavioral response space and matched against the 17 pre-registered calibration controls/knockouts.

| Cluster | Archetype | Count | Trajectories Hit | Cross-Trajectory? | Failed Features | Top Failing Statistics | Nearest Control | Targeted Mechanisms |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **`F1`** | E2+E3 Breakdown (~ko_no_mrs_stop) | `16` | T1, T2, T3 | 🔥 **YES** | `E2, E3, E1, E4, E5` | `pop_traj(l2), loghill_T(slope), plateau_frac(l2)` | `ko_no_mrs_stop` | move_tiebreak (6), mesa_framework (3), two_goods (2) |
| **`F2`** | E1+E2 Breakdown (~ko_ignore_regrowth) | `2` | T2 | No | `E1, E2, E3, E4, E5` | `plateau_frac(slope), pop_traj(l2), disp_late(slope)` | `ko_ignore_regrowth` | heterogeneity (1), none (1) |
| **`G0`** | Stage 0 Gate Rejection (Complexity / Guard / Smoke) | `58` | T1, T2, T3 | 🔥 **YES** | `Stage 0/1` | `—` | `stage_0_gate` | mesa_framework (15), none (9), two_goods (8) |
| **`G1`** | Stage 1 Gate Rejection (Complexity / Guard / Smoke) | `3` | T2, T3 | 🔥 **YES** | `Stage 0/1` | `—` | `stage_1_gate` | mesa_framework (2), none (1) |

---

## 3. Mechanism Necessity & Feature Sensitivity Matrix

Worst-case feature discrepancy ratio ($\text{ratio} < 1.0$ passes equivalence; $\ge 1.0$ fails) observed when each Sugarscape mechanism was targeted for ablation.

| Mechanism | Classification | Attempts (Acc / Rej) | Trajectories Modified | Max E1 | Max E2 | Max E3 | Max E4 | Max E5 | Primary Failure Attractor |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`growback`** | compressible | `2` (`1` / `1`) | — | `1.99` | `1.61` | `1.81` | `1.58` | `3.13` | `F1` |
| **`move_welfare`** | untested/under-explored | `0` (`0` / `0`) | — | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `none` |
| **`move_tiebreak`** | compressible | `16` (`5` / `11`) | T3 | `1.27` | `2.22` | `1.43` | `1.01` | `1.55` | `F1` |
| **`welfare`** | compressible | `2` (`1` / `1`) | — | `0.75` | `0.80` | `0.63` | `0.63` | `0.63` | `G0` |
| **`metabolism`** | untested/under-explored | `1` (`0` / `1`) | — | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `G0` |
| **`death`** | untested/under-explored | `0` (`0` / `0`) | — | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `none` |
| **`vision`** | untested/under-explored | `0` (`0` / `0`) | — | `0.00` | `0.00` | `0.00` | `0.00` | `0.00` | `none` |
| **`trade`** | compressible | `7` (`2` / `5`) | — | `0.66` | `0.71` | `0.62` | `0.65` | `0.66` | `G0` |
| **`mrs_stop`** | compressible | `5` (`3` / `2`) | T1 | `0.79` | `0.55` | `0.81` | `0.56` | `0.78` | `G0` |
| **`price_rule`** | compressible | `4` (`1` / `3`) | — | `2.26` | `1.32` | `3.12` | `1.35` | `4.16` | `G0` |
| **`heterogeneity`** | load-bearing (candidate) | `4` (`0` / `4`) | — | `2.98` | `9.85` | `4.36` | `4.94` | `14.59` | `G0` |
| **`two_goods`** | compressible | `11` (`1` / `10`) | — | `1.06` | `1.77` | `1.36` | `1.22` | `1.64` | `G0` |
| **`random_activation`** | compressible | `7` (`1` / `6`) | — | `1.36` | `0.79` | `0.87` | `0.91` | `1.19` | `G0` |
| **`mesa_framework`** | compressible | `23` (`3` / `20`) | T2, T3 | `1.73` | `1.75` | `1.85` | `1.34` | `4.53` | `G0` |
