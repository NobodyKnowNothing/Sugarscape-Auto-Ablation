# Judge calibration: profile `full`

- Result: **PASSED** (positive controls min pass rate 1.00, negative/null controls min fail rate 1.00; required 0.95 / 0.95)
- Judge hash `c144304071e9ad4e`, design hash `e2ca5bf9ad7dd15d`, created 2026-10-28T01:26:08+00:00, 0.22 h
- Screening bank: 384 scrambled-Sobol' points x 2 worlds x 4 seeds; holdout bank: 384 i.i.d. points x 2 x 4

## Literature-backed responses in canonical G1MT over the domain

Estimates are signed so that positive means the claimed direction: SD units across the input's whole range for responses, SD units for the trade effect, raw units (log dispersion ratio) for IV-3.

| id | claim | source | canonical estimate | 95% one-sided bound | holds |
|---|---|---|---|---|---|
| IV-6a | carrying capacity rises with mean vision (with trade) | Epstein & Axtell 1996, Fig. IV-6; Mesa sugarscape_g1mt tests.py | +7.56 | +5.07 | yes |
| IV-6b | carrying capacity rises with mean vision (without trade) | Epstein & Axtell 1996, Fig. IV-6; Mesa sugarscape_g1mt tests.py | +8.06 | +5.58 | yes |
| IV-6c | carrying capacity is higher with trade | Epstein & Axtell 1996, Fig. IV-6; Mesa sugarscape_g1mt tests.py | +0.90 | +0.79 | yes |
| IV-13a | inequality falls with mean vision | Epstein & Axtell 1996, Fig. IV-13 (listed as a target in Mesa's tests.py) | -0.51 | -1.37 | NO |
| IV-13b | inequality is higher with trade | Epstein & Axtell 1996, Fig. IV-13 (listed as a target in Mesa's tests.py) | +0.64 | +0.56 | yes |
| IV-3 | price dispersion falls over time | Epstein & Axtell 1996, Fig. IV-3 (p. 109); Mesa sugarscape_g1mt tests.py | +1.34 | +1.24 | yes |
| II-met | carrying capacity falls with mean metabolism | Epstein & Axtell 1996, ch. II (carrying capacity) | +48.26 | +45.97 | yes |
| II-growback | carrying capacity rises with the regrowth rate | Epstein & Axtell 1996, ch. II (growback rule G) | +4.58 | +2.98 | yes |

## Canonical trait-response map

Change of each statistic across the full range of each input, in units of its typical seed-to-seed SD (linear fit over the screening bank). Required to be reproduced within the slope margins.

### With trade

| statistic | log_initial_population | log_regrowth_rate | vision_mean | vision_spread | metabolism_mean | metabolism_spread | endowment_mean | endowment_spread |
|---|---|---|---|---|---|---|---|---|
| gini_T | +0.9 | +0.0 | +0.5 | +0.5 | -2.7 | +4.3 | -0.4 | +0.4 |
| gini_c1 | +1.4 | -0.7 | -0.1 | +0.6 | +2.4 | +3.0 | -0.5 | +1.0 |
| top10_T | -0.1 | -1.9 | -0.9 | -0.1 | +6.0 | -1.2 | -0.4 | +0.1 |
| cv_T | +0.8 | -0.1 | +0.7 | +0.4 | -2.2 | +3.9 | -0.5 | +0.3 |
| loghill_T | +0.9 | -0.0 | +2.4 | -0.4 | -6.8 | +1.4 | +0.2 | -0.9 |
| plateau_frac | -5.6 | +5.5 | +7.6 | -0.4 | -49.0 | +6.8 | +4.5 | -1.3 |
| tau_frac | +1.8 | +0.4 | -0.1 | +0.8 | +2.9 | +0.4 | +4.6 | +0.8 |
| fluct_cv | +0.0 | -0.3 | -0.1 | -0.0 | +1.1 | -0.5 | +0.5 | +0.2 |
| disp_late | +1.4 | +2.0 | -4.9 | +0.8 | +0.7 | -1.9 | +1.0 | -0.1 |
| disp_decay | +0.8 | +0.9 | -1.3 | +0.7 | -0.7 | -1.7 | +2.8 | -0.4 |
| price_level | +0.2 | +0.3 | -0.4 | +0.4 | -0.2 | +0.2 | +0.2 | +0.1 |
| exch_rate | +0.9 | +1.0 | -1.6 | +0.2 | -5.4 | +0.3 | +1.4 | +0.0 |
| drc_c1 | +5.4 | +3.0 | +2.8 | +0.7 | -15.2 | +3.9 | +4.9 | +0.1 |
| drc_T | +5.7 | +3.1 | +2.7 | +0.6 | -25.6 | +8.8 | +1.5 | -0.4 |
| moran_T | +3.4 | +3.7 | +1.5 | +0.7 | -9.1 | +1.8 | +0.6 | -0.2 |

### Without trade

| statistic | log_initial_population | log_regrowth_rate | vision_mean | vision_spread | metabolism_mean | metabolism_spread | endowment_mean | endowment_spread |
|---|---|---|---|---|---|---|---|---|
| gini_T | +0.6 | -0.3 | +1.4 | +0.3 | -4.6 | +5.7 | -1.1 | +0.9 |
| gini_c1 | +0.7 | -0.8 | +1.3 | +0.2 | +1.4 | +3.3 | -0.6 | +1.6 |
| top10_T | -0.8 | -1.8 | -1.1 | -0.4 | +6.7 | -1.5 | -0.7 | +0.5 |
| cv_T | +0.6 | -0.4 | +1.4 | +0.2 | -3.9 | +5.2 | -1.1 | +0.8 |
| loghill_T | +1.4 | -0.3 | +1.4 | -0.2 | -5.5 | +0.3 | +0.5 | -1.3 |
| plateau_frac | -5.5 | +4.6 | +8.1 | -0.6 | -48.3 | +6.4 | +4.3 | -1.4 |
| tau_frac | +2.0 | -0.1 | +0.9 | +0.2 | +2.1 | +0.3 | +4.9 | +0.7 |
| fluct_cv | +0.0 | -0.3 | -0.1 | -0.1 | +1.1 | -0.6 | +0.8 | +0.1 |
| drc_c1 | +6.0 | +2.9 | +3.1 | +0.4 | -15.6 | +3.3 | +5.7 | -0.3 |
| drc_T | +5.9 | +2.8 | +2.9 | +0.4 | -26.7 | +9.2 | +1.4 | -0.5 |
| moran_T | +3.0 | +3.1 | +1.9 | +0.3 | -7.8 | +1.3 | +0.6 | -0.2 |

### Trade contrast (trade minus no trade, domain average, in SD)

| statistic | effect | margin |
|---|---|---|
| gini_T | +0.64 ± 0.05 | 0.50 |
| gini_c1 | +0.87 ± 0.05 | 0.50 |
| top10_T | +0.14 ± 0.05 | 0.50 |
| cv_T | +0.60 ± 0.05 | 0.50 |
| loghill_T | -0.38 ± 0.05 | 0.50 |
| plateau_frac | +0.90 ± 0.07 | 0.50 |
| tau_frac | +0.29 ± 0.06 | 0.50 |
| fluct_cv | +0.09 ± 0.05 | 0.50 |
| drc_c1 | +0.16 ± 0.05 | 0.50 |
| drc_T | +0.14 ± 0.05 | 0.50 |
| moran_T | +0.36 ± 0.04 | 0.50 |

## Controls

| control | role | expected | outcome | features | est. pass prob. (positives) |
|---|---|---|---|---|---|
| numpy_reimplementation | positive | accept | as expected | E1=pass E2=pass E3=pass E4=pass E5=pass | 0.865 (min slack 1.2 SE) |
| rng_reorder | positive | accept | as expected | E1=pass E2=pass E3=pass E4=pass E5=pass | 0.972 (min slack 1.9 SE) |
| ko_no_trade | negative | reject E3,E5 | as expected | E1=undecided E2=fail E3=fail E4=undecided E5=fail |  |
| ko_no_growback | negative | reject E2 | as expected | E1=fail E2=fail E3=fail E4=fail E5=fail |  |
| ko_random_move | negative | reject E4 | as expected | E1=fail E2=fail E3=fail E4=fail E5=fail |  |
| ko_uniform_vision | negative | reject E5 | as expected | E1=fail E2=fail E3=fail E4=fail E5=fail |  |
| ko_ignore_population | negative | reject E5 | as expected | E1=undecided E2=fail E3=undecided E4=fail E5=fail |  |
| ko_ignore_regrowth | negative | reject E5 | as expected | E1=undecided E2=fail E3=fail E4=fail E5=fail |  |
| ko_ignore_vision | negative | reject E5 | as expected | E1=undecided E2=fail E3=fail E4=fail E5=fail |  |
| ko_ignore_metabolism | negative | reject E5 | as expected | E1=fail E2=fail E3=fail E4=fail E5=fail |  |
| ko_ignore_endowment | negative | reject E5 | as expected | E1=undecided E2=fail E3=undecided E4=fail E5=undecided |  |
| ko_trade_always | negative | reject E3,E5 | as expected | E1=undecided E2=fail E3=fail E4=pass E5=fail |  |
| null_random_walk | null | reject E3,E4 | as expected | E1=fail E2=fail E3=fail E4=fail E5=fail |  |
| null_static_agents | null | reject E4 | as expected | E1=fail E2=fail E3=fail E4=fail E5=fail |  |
| null_noise_exchange | null | reject E3 | as expected | E1=fail E2=fail E3=fail E4=undecided E5=fail |  |
| ko_no_mrs_stop | probe | verdict | would be accepted | E1=pass E2=pass E3=pass E4=pass E5=pass | 0.979 (min slack 2.2 SE) |
| null_random_price | probe | verdict | rejected (E3, E5) | E1=pass E2=pass E3=fail E4=pass E5=fail |  |

Statistics dropped as non-discriminative: none

## Probes: open mechanism questions

Each probe is judged exactly like a candidate at the final look. 'Would be accepted' means the mechanism it removes is not needed for any required feature or trait response over this domain.

- **ko_no_mrs_stop** (MRS-crossing stop removed from the trade rule): would be accepted.
- **null_random_price** (random price unrelated to MRS, but trades still need both parties better off (zero-intelligence traders with a budget constraint, Gode & Sunder 1993)): rejected (E3, E5).
  - E3: l2 disp_late  (trade) estimate +1.34 [+1.09, +1.56], margin 1.00
  - E5: slope disp_late vision_mean (trade) estimate -3.02 [-3.89, -2.16], margin 2.00
