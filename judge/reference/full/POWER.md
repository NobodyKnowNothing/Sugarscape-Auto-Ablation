# Power check: canonical against itself (profile `full`)

Each replication compares 2 seeds against 2 other seeds of canonical Sugarscape at 256 random screening-bank points per world, with the final-look alpha (0.04). Real candidates are compared against all reference seeds, so real power is higher.

**All tests passed in 10 of 12 replications (83%).**

| replication | all pass | tests | undecided | fail |
|---|---|---|---|---|
| 0 | yes | 272 | 0 | 0 |
| 1 | yes | 272 | 0 | 0 |
| 2 | yes | 272 | 0 | 0 |
| 3 | yes | 272 | 0 | 0 |
| 4 | yes | 272 | 0 | 0 |
| 5 | yes | 272 | 0 | 0 |
| 6 | no | 272 | 2 | 0 |
| 7 | yes | 272 | 0 | 0 |
| 8 | yes | 272 | 0 | 0 |
| 9 | yes | 272 | 0 | 0 |
| 10 | yes | 272 | 0 | 0 |
| 11 | no | 272 | 1 | 0 |

Tests that did not pass, by frequency:

- l2 trade top10_T: 1
- l2 notrade plateaued: 1
- l2 notrade ineq_up: 1
