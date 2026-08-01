# Response-Fibre Surface-Code Audit — LaTeX Package (v0.6.1 evidence sync)

Paper source and compiled PDF, with figure data and an independent Wilson-bound
verification script. All numerical content is synchronised with the archived
v0.6.1 Monte Carlo report (`results/v0.6.0/report.json.gz`).

## Contents

| File | Role |
|---|---|
| `main.tex` | Paper source (compile with any LaTeX distribution, e.g. `tectonic main.tex`) |
| `main.pdf` | Compiled 8-page PDF |
| `figure_data_seed_20290107.csv` | Figure 1 data, seed 20290107 (`d, ref%, flow%, wilson_ref, wilson_flow`) |
| `figure_data_seed_20290119.csv` | Figure 1 data, seed 20290119 |
| `figure_data_seed_20290203.csv` | Figure 1 data, seed 20290203 |
| `verify_wilson.py` | Recomputes the one-sided 95% Wilson upper bounds from the failure counts and checks the frozen crossover claim |

## Headline results (v0.6.1)

- Minimum relative reduction of decoded logical failure: **17.86%**
  (exact: 17.8643035%, seed 20290203, d=3: 11.5213% -> 9.4631%)
- Minimum two-sample z-score: **24.86**
  (exact: 24.8586212, seed 20290203, d=11)
- 15/15 fixed-distance comparisons pass the secondary gates
  (rho >= 5%, z >= 1.645)
- All three seeds: minimum tested distance crossover **d=11 -> d=9**,
  45.34% saving in the declared active-coordinate qubit-round proxy

## Hashes (SHA-256)

| Artifact | Hash |
|---|---|
| Protocol | `fec91e30001712f3d9ac84c0e45a6b70f2d5ae7189d3c9ac6d1096d47505cbf6` |
| Canonical report certificate (before self field) | `2db9620419ac5a7ff64510c65e0d391c4603b6c361fdd8aadd2d9f96165cbc79` |
| report.json.gz | `f9edf8692aaa0f116cc6584507e7f326d184831f251669aa6e2dd2dd143bb95a` |
| Claim certificate | `6754415ceb7bb662982c052d2045c7fafced8a7568bf86db7f94a86338a0f00f` |

Archived run: 478.948 seconds. Full run: Python 3.12.13, NumPy 2.0.2,
Stim 1.16.0, PyMatching 2.4.0.

## Verify

```
python3 verify_wilson.py
```

Recomputes all 30 Wilson upper bounds from the decoded failure counts
(N = 10^6, z_0.95 = 1.6448536) and confirms d*_ref = 11 -> d*_flow = 9
for every seed.

## Scope

This is a LaTeX submission package, not the full reproduction bundle.
The complete audit code, compact reference summary, and certificates live
in the repository: https://github.com/papasop/quantum
