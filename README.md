The QPU pilot is reproducible at the protocol and analysis level from this repository.
Direct reruns of IonQ QPU jobs require independent IonQ Cloud access, backend availability,
and user-provided API credentials. No API keys or private account metadata are included.




# Metric-Compatible Path-Capacity Principle for Quantum Control Time

This repository contains validation code, figures, simulator checks, and preliminary QPU pilot data for the manuscript:

> **A Metric-Compatible Path-Capacity Principle for Quantum Control Time**

The project studies whether elapsed quantum control time can be reconstructed from a **metric-compatible path/capacity pair**:

[
dt_{\rm info}
=============

\frac{d\Phi_g}{H_{\rm cap}^{(g,\mathcal G)}} .
]

The key point is that the capacity is **not a fitted parameter**. Once a state-space metric (g_\rho) and a physical generator (\mathcal G_t) are fixed, the capacity is selected by the same metric that defines the path element:

[
H_{\rm cap}^{(g,\mathcal G)}(\rho,t)
====================================

\sqrt{
g_\rho!\left(\mathcal G_t[\rho],\mathcal G_t[\rho]\right)
}.
]

---

## Quick summary

This repository has three layers:

| Layer                    | Purpose                                                             | Status      |
| ------------------------ | ------------------------------------------------------------------- | ----------- |
| Core validation          | Tests (dt=d\Phi/H) and metric-compatible capacity selection         | Complete    |
| IonQ simulator extension | Tests hardware-facing path-exposure predictions in noise simulators | Complete    |
| IonQ QPU pilot           | Tests endpoint-equivalent native-ZZ identity loops on real QPU      | Preliminary |

The core theory is validated using controlled state-trajectory simulations.
The IonQ tests examine a hardware-facing consequence:

[
D_{\rm endpoint}=0
\not\Rightarrow
\text{zero executed hardware exposure}.
]

---

## Repository contents

Suggested structure:

```text
.
├── README.md
├── info_time.tex
├── path_capacity_validation.py
├── ionq_forte_noise_path_regression.py
├── ionq_qpu_pilot_results.csv
├── figures/
│   ├── fig1_principle_schematic.pdf
│   ├── fig2_noncommuting_drive.pdf
│   ├── fig3_open_system.pdf
│   ├── fig4_entangling_gate.pdf
│   └── figS1_commuting_sanity.pdf
└── results/
    ├── core_validation_summary.txt
    ├── ionq_multi_noise_regression.csv
    └── ionq_qpu_pilot_summary.csv
```

---

# 1. Core path-capacity validation

The main validation script tests the path-capacity principle in three settings.

---

## Q3: Non-commuting pure-state drive

A single qubit is driven by a non-commuting two-axis Hamiltonian:

[
H(t)
====

\frac{\hbar}{2}
\left[
\Omega_x(t)\sigma_x+\Omega_z(t)\sigma_z
\right].
]

The correct Fubini--Study capacity,

[
H_{\rm cap}
===========

\Delta E(t)/\hbar,
]

reconstructs elapsed time, while wrong-generator and endpoint-only estimates fail.

Representative result:

```text
[Q3: NON-COMMUTING PURE DRIVE]
correct relative error          = 3.214e-08
endpoint/mean RMSE              = 2.974e-01
wrong no-z model RMSE           = 1.285e-01
path/endpoint ratio             = 1.344966
speed identity max error        = 1.693e-15
```

Interpretation:

```text
The correct metric-compatible capacity reconstructs time.
A capacity built from the wrong generator fails.
Endpoint distance is not a reliable proxy for the realized path.
```

---

## Q4: Open dephasing

The open-system test evolves a mixed state under:

[
\dot\rho
========

-\frac{i}{\hbar}[H(t),\rho]
+
\gamma(\sigma_z\rho\sigma_z-\rho).
]

The correct capacity is the Bures speed of the full Liouvillian:

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
correct Liouvillian rel error   = 5.692e-04
wrong DeltaE RMSE               = 3.460e-01
wrong gamma=0 RMSE              = 5.968e-02
wrong Hilbert-Schmidt RMSE      = 3.193e-01
endpoint/mean RMSE              = 3.138e-01
purity final                    = 0.633112
path/endpoint ratio             = 1.353378
```

Interpretation:

```text
For open dynamics, the capacity is not the closed-system energy uncertainty.
It is the Bures metric speed of the full Liouvillian.
Wrong generator and wrong metric assignments fail.
```

---

## Q5: Two-qubit entangling gate

The entangling interaction is:

[
H_{\rm int}
===========

\frac{\hbar J}{2}\sigma_z\otimes\sigma_z.
]

It generates maximal entanglement from (|+\rangle\otimes|+\rangle). The path-capacity reconstruction recovers the Bell time and distinguishes realized path from endpoint distance.

Representative result:

```text
[Q5: TWO-QUBIT ENTANGLING]
correct relative error          = 1.096e-10
endpoint/mean RMSE              = 3.123e-01
max concurrence                 = 1.00000000
max entropy                     = 1.00000000
Bell time error                 = 0.000e+00
path/endpoint ratio             = 1.666667
```

---

## Closed-loop endpoint failure

The strongest endpoint failure is a closed entangling loop:

[
D_{\rm endpoint}=0,
\qquad
\Phi_{\rm path}=\pi.
]

Numerically:

```text
[Closed entangling loop]
endpoint distance final          = 0.000e+00
path length final                = 3.14159265
path error vs analytic pi        = 2.002e-10
reconstruction rel error on loop = 6.379e-11
endpoint-only RMSE on loop       = 1.974e+00
```

Interpretation:

```text
The endpoint can return to the initial ray while the executed path remains nonzero.
Endpoint-only cost can vanish even when a full physical path was executed.
```

---

# 2. Hardware-facing extension

The manuscript predicts that endpoint-equivalent circuits may carry different executed hardware costs.

In noisy hardware, a possible path-exposure relation is:

[
P_{\rm err}
\sim
1-\exp(-\lambda_{\rm path}\Phi_{\rm exec}).
]

The hardware-facing tests here study native-ZZ identity loops:

[
ZZ(0)^1,\qquad
ZZ(0)^4,\qquad
ZZ(0)^8.
]

All have the same ideal endpoint (|00\rangle), but they execute different numbers of native ZZ operations.

The measured observable is:

[
P_{\rm odd}
===========

P(01)+P(10).
]

This tests the hardware-facing statement:

[
D_{\rm endpoint}=0
\not\Rightarrow
\text{zero executed hardware exposure}.
]

---

# 3. IonQ simulator preflight

IonQ ideal and Forte-like noise simulators were used as pre-QPU checks.

## Ideal simulator

The ideal simulator gives:

```text
P_odd = 0
```

for all endpoint-equivalent identity loops.

This confirms that the residual is not caused by the ideal circuit endpoint.

---

## Forte-like noise simulators

Forte-like noise simulators show that (P_{\rm odd}) increases mainly with the number of executed native ZZ operations.

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

Representative regression:

```text
P_odd = 0.002636 + 0.007664 N_ZZ + 0.005420 theta_L1
R^2   = 0.9590
```

A broader multi-noise-model comparison gives:

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
The effective hardware ledger is primarily N_ZZ rather than sum |theta_i|.
```

Important limitation:

```text
This is simulator preflight, not QPU evidence.
```

---

# 4. Preliminary IonQ Forte Enterprise QPU pilot

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

```text
ZZ(0)^1:  P_odd = 0.010
ZZ(0)^4:  P_odd = 0.020
ZZ(0)^8:  P_odd = 0.105
```

## Run 2

```text
ZZ(0)^1:  P_odd = 0.005
ZZ(0)^4:  P_odd = 0.015
ZZ(0)^8:  P_odd = 0.055
```

---

## Combined QPU pilot result

Each combined point uses 400 total shots.

| Circuit   | Odd counts | Total shots | (P_{\rm odd}) |
| --------- | ---------: | ----------: | ------------: |
| (ZZ(0)^1) |          3 |         400 |        0.0075 |
| (ZZ(0)^4) |          7 |         400 |        0.0175 |
| (ZZ(0)^8) |         32 |         400 |        0.0800 |

The combined trend is:

[
P_{\rm odd}(ZZ(0)^8)

>

P_{\rm odd}(ZZ(0)^4)

>

P_{\rm odd}(ZZ(0)^1).
]

Interpretation:

```text
The preliminary QPU data support the hardware-facing prediction that endpoint-equivalent native-ZZ identity circuits can accumulate residual error with executed native-operation exposure.
```

Important limitation:

```text
These are preliminary QPU pilot data, not a full hardware benchmark.
```

---

# 5. What is validated where?

| Claim                                                               | Validation source                           |
| ------------------------------------------------------------------- | ------------------------------------------- |
| (dt_{\rm info}=d\Phi_g/H_{\rm cap}) reconstructs elapsed time       | Q3/Q4/Q5 core simulations                   |
| (H_{\rm cap}) is selected by the metric speed of the true generator | Core selection-rule validation              |
| Wrong generator / wrong metric / endpoint-only assignments fail     | Core negative controls                      |
| Closed loops can have (D_{\rm endpoint}=0) but nonzero path         | Two-qubit closed-loop validation            |
| Endpoint-equivalent circuits can have nonzero hardware exposure     | IonQ simulator and QPU pilot                |
| Forte-like effective ledger is mainly executed ZZ operation count   | IonQ simulator regression                   |
| Real QPU residual increases with native-ZZ identity-loop depth      | Preliminary IonQ Forte Enterprise QPU pilot |

---

# 6. What this does not claim

This repository does **not** claim that the IonQ pilot fully verifies the whole theory.

The QPU pilot tests only the hardware-facing extension:

[
\text{endpoint equivalence}
\not\Rightarrow
\text{zero executed hardware exposure}.
]

The core path-capacity equations are validated by controlled state-trajectory simulations.

The current QPU pilot is preliminary and should be interpreted as:

```text
pilot hardware evidence for executed-operation exposure
```

not as:

```text
a complete hardware benchmark
```

or:

```text
a full experimental proof of the path-capacity principle
```

---

# 7. Reproducibility notes

Do not commit API keys.

Recommended setup:

```python
import os
from getpass import getpass

os.environ["IONQ_API_KEY"] = getpass("Paste IonQ API key: ")
```

Native-ZZ circuit format:

```json
{
  "type": "ionq.circuit.v1",
  "backend": "simulator",
  "shots": 1000,
  "input": {
    "qubits": 2,
    "gateset": "native",
    "circuit": [
      {"gate": "zz", "targets": [0, 1], "angle": 0.0}
    ]
  }
}
```

Identity-loop helper:

```python
def zz0_list(n):
    return [{"gate": "zz", "targets": [0, 1], "angle": 0.0} for _ in range(n)]
```

Odd-parity residual:

```python
def p_odd(probs):
    return float(probs.get("1", 0.0)) + float(probs.get("2", 0.0))
```

---

# 8. Recommended next steps

1. Repeat the three-point QPU pilot in another session:

[
ZZ(0)^1,\quad ZZ(0)^4,\quad ZZ(0)^8.
]

2. Interleave circuit order to reduce drift:

```text
8ZZ → 1ZZ → 4ZZ
4ZZ → 8ZZ → 1ZZ
1ZZ → 8ZZ → 4ZZ
```

3. Increase shots after confirming the trend remains stable.

4. Add nonzero-angle controls:

[
ZZ(0.25)^4,\qquad ZZ(0.25)^8.
]

5. Fit a hardware ledger:

[
P_{\rm odd}
===========

\alpha
+
\beta_N N_{\rm ZZ}
+
\eta_\theta \theta_{L1}.
]

6. Compare QPU coefficients with simulator coefficients.

---

# 9. Current status

```text
Core path-capacity validation: complete
IonQ simulator preflight: complete
IonQ Forte Enterprise QPU pilot: preliminary success
Full hardware benchmark: future work
```

---

# 10. One-sentence takeaway

Endpoint-equivalent quantum circuits can still carry different executed hardware exposure: in preliminary IonQ Forte Enterprise QPU pilot runs, native-ZZ identity loops showed increasing odd-parity residual with executed native-ZZ depth.



Y.Y.N., L. (2026, June 19). A Metric-Compatible Path-Capacity Principle for Quantum Control Time. Zenodo. https://doi.org/10.5281/zenodo.20767791



