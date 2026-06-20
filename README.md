# A Metric-Compatible Path-Capacity Principle for Quantum Control Time

Validation code and supporting materials for the manuscript *A Metric-Compatible
Path-Capacity Principle for Quantum Control Time*.

## Quick summary

Elapsed quantum control time is reconstructed from a realized state-space path over the
local capacity of the dynamics that generates it,

```
dt_info = dΦ_g / H_cap^(g,G)
```

The nontrivial content is the **capacity-selection rule**: once a state-space metric
`g_ρ` and the true physical generator `G_t` are fixed, the capacity is determined by the
same metric that defines the path element,

```
H_cap^(g,G) = sqrt( g_ρ( G_t[ρ], G_t[ρ] ) )
```

so the path increment and the capacity must be **metric-compatible**. The principle is
falsifiable: wrong generators, wrong metrics, and endpoint-only estimates do not
reconstruct the time.

## Core validation: Q3 / Q4 / Q5

All core formulas are validated by closed-system and open-system **state-trajectory**
simulations in `npjquantum.py`. Run:

```bash
python3 npjquantum.py                       # quick mode (~5 s)
python3 -c "import npjquantum; npjquantum.main(run_full_checks=True)"   # full anti-cheat
python3 -c "import npjquantum; npjquantum.generate_figures('.')"        # regenerate figures
```

| Test | What it establishes | Key numbers |
|------|---------------------|-------------|
| **Q3** non-commuting drive | FS selection rule; correct vs wrong generator | correct ~3e-8; wrong no-z RMSE 1.3e-1; `\|v_proj − ΔE/ℏ\|` ~1e-15 |
| **Q4** open dephasing | Bures/Liouvillian capacity; metric-compatibility | correct 5.7e-4; ΔE 3.5e-1, γ=0 6.0e-2, **HS (wrong metric) 3.2e-1**, endpoint 3.1e-1 |
| **Q5** entangling + closed loop | path ≠ endpoint | Bell time exact; ratio 5/3; **closed loop D_end=0, Φ=π** |

The selection rule is verified both positively (FS to ~1e-15, Bures to 5.7e-4) and
negatively (every wrong generator / wrong metric / endpoint-only comparator fails by
3–7 orders of magnitude).

## Hardware-facing extension

The geometry implies a prediction, separate from the core principle:

```
D_endpoint = 0   does NOT imply   zero executed hardware exposure.
```

Endpoint-equivalent native-ZZ identity-loop circuits (which return to the initial ray,
so the endpoint distance is zero) can still accumulate residual error scaling with the
number of executed native operations. This is the hardware-facing content the IonQ
materials below probe — **and the only thing they probe.**

## IonQ simulator preflight

Ideal simulator gives `P_odd = 0`. Forte-like noise simulators show `P_odd` increasing
mainly with the executed ZZ count `N_ZZ`. Fitting

```
P_odd = α + β_N · N_ZZ + η_θ · θ_L1
```

gives `β_N > 0` with `R² ≈ 0.95–0.96`, while the continuous-angle ledger `θ_L1` is not
clearly resolved. The effective Forte-like simulator ledger is therefore an
**operation-count exposure**,

```
Φ_exec^(Forte,sim) ~ N_ZZ      (not clearly Σ_i |θ_i|).
```

This is a preflight on the noise model, not a measurement.

## Preliminary IonQ Forte Enterprise QPU pilot

Backend `qpu.forte-enterprise-1`, `dry_run=false`, 200 shots per circuit per run.
Odd-parity residual on endpoint-equivalent native-ZZ identity loops:

| Circuit | Run 1 | Run 2 | Combined (odd/400) |
|---------|-------|-------|--------------------|
| `ZZ(0)^1` | 0.010 | 0.005 | 3/400 = **0.0075** |
| `ZZ(0)^4` | 0.020 | 0.015 | 7/400 = **0.0175** |
| `ZZ(0)^8` | 0.105 | 0.055 | 32/400 = **0.0800** |

> Two independent IonQ Forte Enterprise QPU pilot runs show a reproducible increase of
> odd-parity residual with executed native-ZZ identity-loop depth.

Reproduce the recorded pilot summary:

```bash
python3 -c "import npjquantum; npjquantum.print_ionq_pilot_summary(npjquantum.ionq_qpu_pilot())"
```

## What is validated where

| Claim | Validated by |
|-------|--------------|
| `dt_info = dΦ_g / H_cap` | Q3/Q4/Q5 state-trajectory validation |
| Selection rule `H_cap = sqrt(g(G,G))` | Q3 (FS) + Q4 (Bures) trajectory validation |
| FS speed identity / Bures–Liouvillian capacity | Q3 / Q4 trajectory validation |
| path ≠ endpoint; closed loop D=0, Φ=π | Q5 + closed-loop validation |
| `D_endpoint=0 ⇏ zero executed exposure` | IonQ simulator preflight + QPU pilot (preliminary) |

## What this does **not** claim

- The IonQ simulator and QPU pilot **do not** verify `dt_info = dΦ_g / H_cap`, the
  capacity-selection rule, the Fubini–Study speed identity, or the Bures–Liouvillian
  capacity. Those are established only by the Q3/Q4/Q5 state-trajectory validation.
- The QPU data are a **preliminary pilot**, not a complete hardware benchmark.
- The pilot supports only the hardware-facing prediction that endpoint-equivalent
  circuits can carry nonzero executed-operation exposure.
- The Forte-like simulator resolves an operation-count ledger (`~N_ZZ`); it does not
  clearly resolve a continuous-angle ledger (`Σ|θ_i|`).

## Reproducibility notes

- `npjquantum.py` is self-contained: it runs the validation and generates all figures.
- Quick mode skips the slow anti-cheat scans (independent RK4 integrator, fidelity-
  formula crosscheck, ε-stability and step-size convergence); enable with
  `run_full_checks=True`.
- Q4 propagation enforces only Hermiticity and trace (no positivity projection); the
  minimum eigenvalue stays ≥ 0 to machine precision.
- The Bures-speed finite difference uses ε = time step by default; the ε-scan is stable
  and below 2e-4 over the well-conditioned range, with ε ≲ 1e-6 limited by
  floating-point cancellation.
- Pilot odd-counts are stored as integers out of 200 shots and verified against the
  reported fractions.

## Recommended next steps

- Larger shot counts and more circuits per depth.
- Randomized / interleaved circuit ordering across depths.
- Calibration controls and repeated sessions across days.
- A dedicated hardware study to separate an operation-count ledger from a continuous-
  angle ledger.

## One-sentence takeaway

Endpoint-equivalent quantum circuits can still carry different executed hardware
exposure: in preliminary IonQ Forte Enterprise QPU pilot runs, native-ZZ identity loops
showed increasing odd-parity residual with executed native-ZZ depth.


Y.Y.N., L. (2026, June 19). A Metric-Compatible Path-Capacity Principle for Quantum Control Time. Zenodo. https://doi.org/10.5281/zenodo.20767791



