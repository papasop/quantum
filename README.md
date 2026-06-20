# A Metric-Compatible Path-Capacity Principle for Quantum Control Time

Validation code, figures, simulator checks, and preliminary hardware-facing pilot data for the manuscript:

> **A Metric-Compatible Path-Capacity Principle for Quantum Control Time**

This project studies an operational reconstruction of elapsed quantum control time from a realized state-space path divided by the local capacity of the dynamics that generates it,

[
dt_{\rm info}
=============

\frac{d\Phi_g}{H_{\rm cap}^{(g,\mathcal G)}}.
]

The nontrivial content is the **capacity-selection rule**: once a state-space metric (g_\rho) and the true physical generator (\mathcal G_t) are fixed, the capacity is selected by the same metric that defines the path element,

[
H_{\rm cap}^{(g,\mathcal G)}(\rho,t)
====================================

\sqrt{
g_\rho
\left(
\mathcal G_t[\rho],
\mathcal G_t[\rho]
\right)
}.
]

Thus the path increment and the capacity must be **metric-compatible**. The principle is falsifiable: wrong generators, wrong metrics, and endpoint-only estimates do not reconstruct elapsed time.

---

## Quick summary

This repository has three layers:

| Layer                    | Purpose                                                                               | Status      |
| ------------------------ | ------------------------------------------------------------------------------------- | ----------- |
| Core validation          | Tests the metric-compatible path/capacity reconstruction rule                         | Complete    |
| IonQ simulator preflight | Tests hardware-facing path-exposure behavior in ideal and Forte-like noise simulators | Complete    |
| IonQ QPU pilot           | Tests endpoint-equivalent native-ZZ identity loops on a real QPU                      | Preliminary |

The core formulas are validated by controlled closed-system and open-system state-trajectory simulations.

The IonQ materials test only the hardware-facing consequence:

[
D_{\rm endpoint}=0
\not\Rightarrow
\text{zero executed hardware exposure}.
]

They do **not** directly verify the main path-capacity selection rule.

---

## Repository contents

A typical repository layout is:

```text
.
├── README.md
├── npjquantum.py
├── info_time.tex
├── figures/
│   ├── fig1_principle_schematic.pdf
│   ├── fig2_noncommuting_drive.pdf
│   ├── fig3_open_system.pdf
│   ├── fig4_entangling_gate.pdf
│   └── figS1_commuting_sanity.pdf
└── results/
    ├── core_validation_summary.txt
    ├── ionq_simulator_preflight_summary.txt
    └── ionq_qpu_pilot_summary.txt
```

The exact file names may differ depending on the release version.

---

## How to run the core validation

The core validation is contained in:

```text
npjquantum.py
```

Run:

```bash
python3 npjquantum.py
```

This reproduces the main Q3/Q4/Q5 validation outputs and anti-cheating checks included in the script version.

If the repository version includes extra helper functions for figure generation or full scans, see the comments at the top of `npjquantum.py`.

---

# 1. Core validation: Q3 / Q4 / Q5

The core theory is validated using state-trajectory simulations.

The tested reconstruction is:

[
T_{\rm rec}
===========

\int_\Gamma
\frac{d\Phi_g}{H_{\rm cap}^{(g,\mathcal G)}}.
]

The key negative controls are:

* wrong generator;
* wrong metric;
* endpoint-only estimate.

---

## Q3: Non-commuting pure-state drive

A single qubit is driven by a non-commuting two-axis Hamiltonian,

[
H(t)
====

\frac{\hbar}{2}
\left[
\Omega_x(t)\sigma_x
+
\Omega_z(t)\sigma_z
\right].
]

The correct Fubini--Study capacity is

[
H_{\rm cap}
===========

\Delta E(t)/\hbar.
]

Representative result:

```text
[Q3: NON-COMMUTING PURE DRIVE]
correct relative error          ≈ 3.214e-08
endpoint/mean RMSE              ≈ 2.974e-01
wrong no-z model RMSE           ≈ 1.285e-01
path/endpoint ratio             ≈ 1.344966
FS speed identity max error     ≈ 1.693e-15
```

Interpretation:

```text
The correct metric-compatible capacity reconstructs elapsed time.
A capacity built from the wrong generator fails.
Endpoint distance is not a reliable substitute for realized path length.
```

---

## Q4: Open dephasing

The open-system test evolves a qubit under

[
\dot\rho
========

-\frac{i}{\hbar}[H(t),\rho]
+
\gamma(\sigma_z\rho\sigma_z-\rho).
]

The correct capacity is the Bures speed of the full Liouvillian,

[
H_{\mathcal L}(t)
=================

\sqrt{
g_{{\rm B},\rho}
\left(
\mathcal L_t[\rho],
\mathcal L_t[\rho]
\right)
}.
]

Representative result:

```text
[Q4: OPEN DEPHASING]
correct Liouvillian rel error   ≈ 5.692e-04
wrong DeltaE RMSE               ≈ 3.460e-01
wrong gamma=0 RMSE              ≈ 5.968e-02
wrong Hilbert-Schmidt RMSE      ≈ 3.193e-01
endpoint/mean RMSE              ≈ 3.138e-01
purity final                    ≈ 0.633112
path/endpoint ratio             ≈ 1.353378
```

Interpretation:

```text
For open dynamics, the capacity is not the closed-system energy uncertainty.
It is the Bures metric speed of the full Liouvillian.
Wrong generator and wrong metric assignments fail.
```

---

## Q5: Two-qubit entangling gate

The two-qubit interaction is

[
H_{\rm int}
===========

\frac{\hbar J}{2}
\sigma_z\otimes\sigma_z.
]

It generates entanglement from (|+\rangle\otimes|+\rangle). The path-capacity reconstruction recovers the Bell time and distinguishes realized path from endpoint distance.

Representative result:

```text
[Q5: TWO-QUBIT ENTANGLING]
correct relative error          ≈ 1.096e-10
endpoint/mean RMSE              ≈ 3.123e-01
max concurrence                 = 1.00000000
max entropy                     = 1.00000000
Bell time error                 = 0
path/endpoint ratio             = 1.666667
```

---

## Closed-loop endpoint failure

A closed entangling loop gives the strongest endpoint failure:

[
D_{\rm endpoint}=0,
\qquad
\Phi_{\rm path}=\pi.
]

Representative result:

```text
[Closed entangling loop]
endpoint distance final          = 0
path length final                ≈ 3.14159265
path error vs analytic pi        ≈ 2.002e-10
reconstruction rel error on loop ≈ 6.379e-11
endpoint-only RMSE on loop       ≈ 1.974
```

Interpretation:

```text
The endpoint can return to the initial ray while the executed path remains nonzero.
Endpoint-only cost can vanish even when a full physical path was executed.
```

---

# 2. What the core validation establishes

| Test                   | What it establishes                                      | Key numbers                                                                                                                                                  |
| ---------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Q3 non-commuting drive | Fubini--Study selection rule; correct vs wrong generator | correct (\sim 3\times10^{-8}); wrong no-z RMSE (\sim 1.3\times10^{-1}); FS speed identity (\sim10^{-15})                                                     |
| Q4 open dephasing      | Bures/Liouvillian capacity; metric compatibility         | correct (\sim5.7\times10^{-4}); (\Delta E) RMSE (\sim3.5\times10^{-1}); (\gamma=0) RMSE (\sim6.0\times10^{-2}); Hilbert--Schmidt RMSE (\sim3.2\times10^{-1}) |
| Q5 entangling gate     | path cost differs from endpoint cost                     | Bell time recovered; path/endpoint ratio (5/3); closed loop (D_{\rm endpoint}=0,\ \Phi=\pi)                                                                  |

The selection rule is verified both positively and negatively: the correct metric-compatible assignments reconstruct elapsed time, while wrong generators, wrong metrics, and endpoint-only comparators fail by orders of magnitude, with the size of the separation depending on the test.

---

# 3. Hardware-facing extension

The geometry implies a hardware-facing prediction separate from the core path-capacity reconstruction:

[
D_{\rm endpoint}=0
\not\Rightarrow
\text{zero executed hardware exposure}.
]

Endpoint-equivalent native-ZZ identity-loop circuits can have the same ideal endpoint while executing different physical native-operation paths.

The hardware-facing observable used here is the odd-parity residual,

[
P_{\rm odd}
===========

P(01)+P(10).
]

The IonQ simulator and QPU materials below probe only this hardware-facing statement. They do **not** verify the main formula

[
dt_{\rm info}
=============

d\Phi_g/H_{\rm cap}^{(g,\mathcal G)}.
]

---

# 4. IonQ simulator preflight

IonQ ideal and Forte-like noise simulators were used as pre-QPU checks.

## Ideal simulator

The ideal simulator gives

```text
P_odd = 0
```

for endpoint-equivalent identity-loop circuits.

This confirms that odd-parity residual is not generated by the ideal endpoint.

---

## Forte-like noise simulators

Forte-like noise simulators show (P_{\rm odd}) increasing mainly with the executed native-ZZ operation count (N_{\rm ZZ}).

A linear model was fit:

[
P_{\rm odd}
===========

\alpha
+
\beta_N N_{\rm ZZ}
+
\eta_\theta \theta_{L1},
]

where

[
\theta_{L1}
===========

\sum_i |\theta_i|.
]

Representative simulator regression:

```text
P_odd = 0.002636 + 0.007664 N_ZZ + 0.005420 theta_L1
R^2   = 0.9590
```

A broader multi-noise-model comparison gave:

```text
ideal:
  beta_N = 0
  eta_theta = 0

forte-1:
  beta_N > 0
  R^2 ≈ 0.95

forte-enterprise-1:
  beta_N > 0
  R^2 ≈ 0.96
```

Interpretation:

```text
The Forte-like noise simulators support an executed-operation exposure model.
The effective simulator ledger is primarily N_ZZ rather than a clearly resolved
continuous angle ledger sum_i |theta_i|.
```

Important limitation:

```text
This is simulator preflight, not QPU evidence.
```

---

# 5. Preliminary IonQ Forte Enterprise QPU pilot

A preliminary QPU pilot was run on:

```text
backend = qpu.forte-enterprise-1
dry_run = false
shots   = 200 per circuit per run
```

The tested native-ZZ identity-loop circuits were:

```text
ZZ(0)^1
ZZ(0)^4
ZZ(0)^8
```

All have the same ideal endpoint (|00\rangle), but different executed native-ZZ depths.

---

## Run 1

| Circuit   | Probabilities                                | (P_{\rm odd}) |
| --------- | -------------------------------------------- | ------------: |
| (ZZ(0)^1) | (P_{00}=0.990,\ P_{10}=0.010)                |         0.010 |
| (ZZ(0)^4) | (P_{00}=0.980,\ P_{01}=0.010,\ P_{10}=0.010) |         0.020 |
| (ZZ(0)^8) | (P_{00}=0.895,\ P_{01}=0.050,\ P_{10}=0.055) |         0.105 |

---

## Run 2

| Circuit   | Probabilities                                | (P_{\rm odd}) |
| --------- | -------------------------------------------- | ------------: |
| (ZZ(0)^1) | (P_{00}=0.995,\ P_{10}=0.005)                |         0.005 |
| (ZZ(0)^4) | (P_{00}=0.985,\ P_{01}=0.015)                |         0.015 |
| (ZZ(0)^8) | (P_{00}=0.945,\ P_{01}=0.045,\ P_{10}=0.010) |         0.055 |

---

## Combined QPU pilot result

Each combined point uses 400 total shots.

| Circuit   | Odd counts | Total shots | (P_{\rm odd}) |
| --------- | ---------: | ----------: | ------------: |
| (ZZ(0)^1) |          3 |         400 |        0.0075 |
| (ZZ(0)^4) |          7 |         400 |        0.0175 |
| (ZZ(0)^8) |         32 |         400 |        0.0800 |

The combined trend is

[
P_{\rm odd}(ZZ(0)^8)

>

P_{\rm odd}(ZZ(0)^4)

>

P_{\rm odd}(ZZ(0)^1).
]

A conservative interpretation is:

```text
Two repeated IonQ Forte Enterprise QPU pilot runs show a reproducible increase
of odd-parity residual with executed native-ZZ identity-loop depth.
```

Important limitation:

```text
These are preliminary QPU pilot data, not a full hardware benchmark.
No calibration correction or drift model is applied.
```

---

# 6. What is validated where?

| Claim                                                                     | Validated by                                      |
| ------------------------------------------------------------------------- | ------------------------------------------------- |
| (dt_{\rm info}=d\Phi_g/H_{\rm cap}) reconstructs elapsed time             | Q3/Q4/Q5 state-trajectory validation              |
| (H_{\rm cap}=\sqrt{g(\mathcal G,\mathcal G)}) selection rule              | Q3 Fubini--Study + Q4 Bures trajectory validation |
| Wrong generator / wrong metric / endpoint-only assignments fail           | Q3/Q4/Q5 negative controls                        |
| Closed loop can have (D_{\rm endpoint}=0) but (\Phi_{\rm path}=\pi)       | Q5 closed-loop validation                         |
| Endpoint-equivalent circuits can have nonzero executed hardware exposure  | IonQ simulator preflight + preliminary QPU pilot  |
| Forte-like effective ledger is mainly executed ZZ operation count         | IonQ simulator regression                         |
| Real QPU odd-parity residual increases with native-ZZ identity-loop depth | Preliminary IonQ Forte Enterprise QPU pilot       |

---

# 7. What this does not claim

This repository does **not** claim that the IonQ pilot fully verifies the whole theory.

Specifically:

* The IonQ simulator and QPU pilot do **not** verify

[
dt_{\rm info}
=============

d\Phi_g/H_{\rm cap}^{(g,\mathcal G)}.
]

* They do **not** verify the capacity-selection rule

[
H_{\rm cap}^{(g,\mathcal G)}
============================

\sqrt{
g_\rho
\left(
\mathcal G_t[\rho],
\mathcal G_t[\rho]
\right)
}.
]

* They do **not** verify the Fubini--Study speed identity or the Bures--Liouvillian capacity.

Those core statements are established by the Q3/Q4/Q5 state-trajectory validation.

The QPU pilot supports only the hardware-facing prediction that endpoint-equivalent circuits can carry nonzero executed-operation exposure.

The QPU data are preliminary and should not be interpreted as a complete hardware benchmark.

---

# 8. Reproducibility notes

## Core validation

The core validation is self-contained in `npjquantum.py`.

```bash
python3 npjquantum.py
```

The script performs the state-trajectory tests and prints the reported validation values.

Depending on the repository version, additional options may be provided for full anti-cheating scans and figure generation.

---

## Q4 open-system numerical notes

The Q4 propagation enforces only Hermiticity and trace normalization.

No positivity projection is applied to the trajectory.

The minimum eigenvalue remains nonnegative to machine precision in the reported run.

The Bures-speed finite difference uses the time step as the default (\epsilon). The (\epsilon)-scan remains stable over the well-conditioned finite-difference range; very small (\epsilon) is limited by floating-point cancellation.

---

## QPU pilot reproducibility

The QPU pilot summary stores processed probabilities and integer odd counts.

Direct reruns of the IonQ QPU pilot require:

* independent IonQ Cloud access;
* backend availability for `qpu.forte-enterprise-1`;
* user-provided API credentials;
* user responsibility for any QPU cost.

No API keys, private account identifiers, or private project metadata are included.

The native-ZZ identity-loop circuit family is:

```python
def zz0_list(n):
    return [{"gate": "zz", "targets": [0, 1], "angle": 0.0} for _ in range(n)]
```

The odd-parity residual is:

```python
def p_odd(probs):
    return float(probs.get("1", 0.0)) + float(probs.get("2", 0.0))
```

---

# 9. Recommended next steps

A dedicated hardware study should include:

1. larger shot counts;
2. randomized or interleaved circuit ordering across depths;
3. calibration controls;
4. repeated sessions across days;
5. nonzero-angle controls, such as (ZZ(0.25)^4) and (ZZ(0.25)^8);
6. a direct fit of

[
P_{\rm odd}
===========

\alpha
+
\beta_N N_{\rm ZZ}
+
\eta_\theta \theta_{L1};
]

7. comparison between simulator and QPU coefficients.

---

# 10. Current status

```text
Core path-capacity validation: complete
IonQ simulator preflight: complete
IonQ Forte Enterprise QPU pilot: preliminary success
Full hardware benchmark: future work
```

---

# 11. One-sentence takeaway

Endpoint-equivalent quantum circuits can still carry different executed hardware exposure: in preliminary IonQ Forte Enterprise QPU pilot runs, native-ZZ identity loops showed increasing odd-parity residual with executed native-ZZ depth.



Y.Y.N., L. (2026, June 19). A Metric-Compatible Path-Capacity Principle for Quantum Control Time. Zenodo. https://doi.org/10.5281/zenodo.20767791



