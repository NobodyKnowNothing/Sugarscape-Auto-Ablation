# Judge calibration: profile `smoke`

- Result: **FAILED** (positive controls min pass rate 0.00, negative/null controls min fail rate 1.00; required 0.95 / 0.95)
- Judge hash `94620be625cb0d3f`, design hash `c9763bbbb1ed6cda`, created 2026-10-27T20:59:23+00:00, 0.23 h
- Screening bank: 48 scrambled-Sobol' points x 2 worlds x 2 seeds; holdout bank: 24 i.i.d. points x 2 x 2

## Literature-backed responses in canonical G1MT over the domain

| id | claim | source | canonical estimate (signed, in SD) | 95% one-sided bound | holds |
|---|---|---|---|---|---|
| IV-6a | carrying capacity rises with mean vision (with trade) | Epstein & Axtell 1996, Fig. IV-6; Mesa sugarscape_g1mt tests.py | +13.14 | +5.19 | yes |
| IV-6b | carrying capacity rises with mean vision (without trade) | Epstein & Axtell 1996, Fig. IV-6; Mesa sugarscape_g1mt tests.py | +17.42 | +7.36 | yes |
| IV-6c | carrying capacity is higher with trade | Epstein & Axtell 1996, Fig. IV-6; Mesa sugarscape_g1mt tests.py | +0.96 | +0.49 | yes |
| IV-13a | inequality falls with mean vision | Epstein & Axtell 1996, Fig. IV-13 (listed as a target in Mesa's tests.py) | -0.90 | -2.94 | NO |
| IV-13b | inequality is higher with trade | Epstein & Axtell 1996, Fig. IV-13 (listed as a target in Mesa's tests.py) | +0.81 | +0.55 | yes |
| IV-3 | price dispersion falls over time | Epstein & Axtell 1996, Fig. IV-3 (p. 109); Mesa sugarscape_g1mt tests.py | +0.44 | +0.09 | yes |
| II-met | carrying capacity falls with mean metabolism | Epstein & Axtell 1996, ch. II (carrying capacity) | +59.56 | +50.10 | yes |
| II-growback | carrying capacity rises with the regrowth rate | Epstein & Axtell 1996, ch. II (growback rule G) | +9.46 | +1.61 | yes |

## Canonical trait-response map

Change of each statistic across the full range of each input, in units of its typical seed-to-seed SD (linear fit over the screening bank). Required to be reproduced within the slope margins.

### With trade

| statistic | log_initial_population | log_regrowth_rate | vision_mean | vision_spread | metabolism_mean | metabolism_spread | endowment_mean | endowment_spread |
|---|---|---|---|---|---|---|---|---|
| gini_T | +1.1 | -0.5 | +0.9 | +0.9 | +0.4 | +2.4 | -1.9 | +0.7 |
| gini_c1 | +3.6 | -1.5 | -1.6 | +1.2 | +4.4 | +3.0 | -4.6 | +0.7 |
| top10_T | +0.5 | -1.6 | +0.8 | -0.3 | +4.5 | -0.7 | -1.4 | +0.2 |
| cv_T | +0.8 | -0.7 | +0.9 | +0.7 | +0.8 | +1.7 | -1.7 | +0.4 |
| loghill_T | +0.5 | +0.8 | +1.3 | +0.4 | -5.4 | +0.4 | +1.6 | -1.0 |
| plateau_frac | -8.2 | +8.1 | +13.1 | -2.9 | -52.1 | +7.0 | +14.8 | +2.1 |
| tau_frac | +5.5 | -0.1 | -2.5 | +2.0 | +4.3 | +2.4 | +1.0 | +2.4 |
| fluct_cv | -1.8 | +1.3 | +3.0 | -2.2 | +7.2 | -0.6 | +3.1 | -2.4 |
| disp_late | +2.8 | +2.8 | -5.7 | +0.1 | +0.8 | -0.9 | +0.4 | +0.3 |
| disp_decay | +1.8 | +0.9 | -3.0 | +1.9 | -2.0 | -1.3 | +5.2 | -1.8 |
| price_level | +0.4 | +0.8 | -0.2 | -1.1 | +1.5 | -0.6 | -0.9 | -0.1 |
| exch_rate | +1.9 | +2.3 | -4.1 | +1.9 | -4.1 | +2.0 | +1.7 | +1.7 |
| drc_c1 | +6.6 | +4.4 | +0.1 | +3.5 | -5.8 | +0.7 | +4.4 | +0.5 |
| drc_T | +6.8 | +4.6 | +3.3 | +2.4 | -23.8 | +8.0 | +1.9 | +2.0 |
| moran_T | +2.7 | +3.5 | +2.2 | +0.5 | -8.2 | +1.5 | +1.3 | +0.8 |

### Without trade

| statistic | log_initial_population | log_regrowth_rate | vision_mean | vision_spread | metabolism_mean | metabolism_spread | endowment_mean | endowment_spread |
|---|---|---|---|---|---|---|---|---|
| gini_T | +1.1 | -0.2 | +1.2 | +1.8 | -3.6 | +5.1 | -2.9 | +1.6 |
| gini_c1 | +5.3 | -0.8 | -3.7 | +3.2 | +8.3 | +5.5 | -4.7 | +1.2 |
| top10_T | -0.7 | -1.5 | +1.8 | -1.4 | +4.3 | -0.9 | -0.9 | -0.1 |
| cv_T | +1.0 | -0.5 | +1.4 | +1.6 | -3.3 | +5.0 | -3.0 | +1.4 |
| loghill_T | +1.1 | +0.0 | +0.0 | +1.2 | -4.8 | -1.2 | +2.4 | -2.4 |
| plateau_frac | -9.7 | +9.5 | +17.4 | -4.6 | -59.6 | +6.1 | +15.6 | +2.0 |
| tau_frac | +4.4 | -0.6 | +0.2 | +0.7 | +3.9 | +0.8 | +0.9 | +1.6 |
| fluct_cv | +0.4 | -0.1 | +1.3 | -0.4 | +3.4 | -0.8 | +1.8 | -0.9 |
| drc_c1 | +7.2 | +4.4 | -0.0 | +1.7 | -3.0 | -1.3 | +4.8 | -0.5 |
| drc_T | +6.0 | +5.1 | +4.2 | +1.0 | -21.7 | +7.4 | -0.6 | +1.8 |
| moran_T | +2.5 | +4.2 | +2.1 | -0.1 | -6.4 | +0.1 | -0.7 | +1.0 |

### Trade contrast (trade minus no trade, domain average, in SD)

| statistic | effect | margin |
|---|---|---|
| gini_T | +0.81 ± 0.15 | 2.00 |
| gini_c1 | +1.12 ± 0.18 | 2.00 |
| top10_T | +0.27 ± 0.17 | 2.00 |
| cv_T | +0.74 ± 0.15 | 2.00 |
| loghill_T | -0.31 ± 0.16 | 2.00 |
| plateau_frac | +0.96 ± 0.28 | 2.00 |
| tau_frac | +0.45 ± 0.18 | 2.00 |
| fluct_cv | +0.70 ± 0.45 | 6.78 |
| drc_c1 | -0.12 ± 0.25 | 2.00 |
| drc_T | +0.34 ± 0.23 | 2.00 |
| moran_T | +0.36 ± 0.19 | 2.00 |

## Controls

| control | role | expected | outcome | features | est. pass prob. (positives) |
|---|---|---|---|---|---|
| numpy_reimplementation | positive | accept | rate 0.00 | E1=pass E2=undecided E3=undecided E4=undecided E5=undecided | 0.000 (min slack -2.3 SE) |
| rng_reorder | positive | accept | rate 0.00 | E1=undecided E2=undecided E3=pass E4=undecided E5=undecided | 0.000 (min slack -3.1 SE) |
| ko_no_trade | negative | reject E3,E5 | as expected | E1=undecided E2=undecided E3=fail E4=undecided E5=fail |  |
| ko_no_growback | negative | reject E2 | as expected | E1=fail E2=fail E3=fail E4=fail E5=undecided |  |
| ko_random_move | negative | reject E4 | as expected | E1=fail E2=fail E3=fail E4=fail E5=undecided |  |
| ko_uniform_vision | probe | report | as expected | E1=undecided E2=fail E3=fail E4=fail E5=undecided |  |
| ko_ignore_population | negative | reject E5 | as expected | E1=undecided E2=fail E3=undecided E4=fail E5=undecided |  |
| ko_ignore_regrowth | probe | report | as expected | E1=undecided E2=undecided E3=undecided E4=undecided E5=undecided |  |
| ko_ignore_vision | probe | report | as expected | E1=undecided E2=fail E3=fail E4=undecided E5=undecided |  |
| ko_ignore_metabolism | negative | reject E5 | as expected | E1=fail E2=fail E3=undecided E4=fail E5=undecided |  |
| ko_ignore_endowment | probe | report | as expected | E1=undecided E2=fail E3=pass E4=undecided E5=undecided |  |
| ko_trade_always | negative | reject E3,E5 | as expected | E1=undecided E2=undecided E3=fail E4=undecided E5=undecided |  |
| null_random_walk | null | reject E3,E4 | as expected | E1=fail E2=fail E3=fail E4=fail E5=undecided |  |
| null_static_agents | null | reject E4 | as expected | E1=fail E2=fail E3=fail E4=fail E5=undecided |  |
| null_noise_exchange | null | reject E3 | as expected | E1=undecided E2=undecided E3=fail E4=undecided E5=undecided |  |
| ko_no_mrs_stop | probe | report | as expected | E1=undecided E2=undecided E3=undecided E4=undecided E5=undecided |  |
| null_random_price | probe | report | reported | E1=undecided E2=undecided E3=pass E4=undecided E5=undecided |  |

Statistics dropped as non-discriminative: disp_down|trade, ineq_up|notrade, lorenz_T|trade, plateaued|trade
