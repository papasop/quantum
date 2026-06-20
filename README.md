# Metric-Compatible Path-Capacity Principle for Quantum Control Time

This repository contains validation code and hardware-facing pilot tests for the manuscript:

**A Metric-Compatible Path-Capacity Principle for Quantum Control Time**

The central idea is that elapsed quantum control time can be reconstructed from a metric-compatible path/capacity pair,

[
dt_{\rm info}
=============

\frac{d\Phi_g}{H_{\rm cap}^{(g,\mathcal G)}} ,
]

where the capacity is not a free parameter. Once a state-space metric (g_\rho) and a physical generator (\mathcal G_t) are specified, the local capacity is fixed by the same metric that defines the path element:

[
H_{\rm cap}^{(g,\mathcal G)}(\rho,t)
====================================

\sqrt{
g_\rho!\left(\mathcal G_t[\rho],\mathcal G_t[\rho]\right)
}.
]

The repository is organized around two layers:

1. **Core path-capacity validation**
   Numerical tests of the metric-compatible reconstruction rule in closed and open quantum dynamics.

2. **Hardware-facing path-exposure tests**
   Simulator and preliminary QPU pilot tests of endpoint-equivalent native-ZZ identity loops on IonQ/Forte-style backends.

---

## 1. Core validation

The main validation script checks the path-capacity principle in three settings:

### Q3: Non-commuting pure-state drive

A single qubit is driven by a non-commuting two-axis Hamiltonian,

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
H_{\rm cap}=\Delta E(t)/\hbar,
]

reconstructs the elapsed time, while a wrong-generator capacity and endpoint-only estimate fail.

Representative result:

```text
correct relative error          = 3.214e-08
endpoint/mean RMSE              = 2.974e-01
wrong no-z model RMSE           = 1.285e-01
path/endpoint ratio             = 1.344966
speed identity max error        = 1.693e-15
```

This verifies that the reconstruction is not just a formal identity: using the wrong generator fails.

---

### Q4: Open dephasing

The open-system test evolves a mixed state under a Lindblad equation,

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

Wrong assignments fail: using (\Delta E/\hbar), setting (\gamma=0), using Hilbert--Schmidt speed, or using endpoint distance does not reconstruct the elapsed time.

Representative result:

```text
correct Liouvillian rel error   = 5.692e-04
wrong DeltaE RMSE               = 3.460e-01
wrong gamma=0 RMSE              = 5.968e-02
wrong Hilbert-Schmidt RMSE      = 3.193e-01
endpoint/mean RMSE              = 3.138e-01
purity final                    = 0.633112
path/endpoint ratio             = 1.353378
```

This verifies the open-system part of the selection rule: the capacity must be the metric speed of the true Liouvillian in the same metric used to define the path.

---

### Q5: Two-qubit entangling gate

A two-qubit interaction,

[
H_{\rm int}
===========

\frac{\hbar J}{2}\sigma_z\otimes\sigma_z,
]

generates entanglement from (|+\rangle\otimes|+\rangle). The path-capacity reconstruction recovers the Bell time and distinguishes real executed path from endpoint distance.

Representative result:

```text
correct relative error          = 1.096e-10
endpoint/mean RMSE              = 3.123e-01
max concurrence                 = 1.00000000
max entropy                     = 1.00000000
Bell time error                 = 0.000e+00
path/endpoint ratio             = 1.666667
```

A closed entangling loop gives the strongest endpoint failure:

[
D_{\rm endpoint}=0,
\qquad
\Phi_{\rm path}=\pi.
]

Numerically:

```text
endpoint distance final          = 0.000e+00
path length final                = 3.14159265
path error vs analytic pi        = 2.002e-10
reconstruction rel error on loop = 6.379e-11
endpoint-only RMSE on loop       = 1.974e+00
```

---

## 2. Hardware-facing extension

The manuscript predicts that mathematically equivalent circuits may carry different executed hardware costs. In noisy hardware, this can appear as a residual error exposure,

[
P_{\rm err}
\sim
1-\exp(-\lambda_{\rm path}\Phi_{\rm exec}).
]

The hardware-facing tests in this repository focus on native-ZZ identity loops. Ideally, all of the following circuits have the same endpoint (|00\rangle), but they execute different numbers of native ZZ operations:

[
ZZ(0)^1,\qquad
ZZ(0)^4,\qquad
ZZ(0)^8.
]

The measured observable is odd-parity residual,

[
P_{\rm odd}=P(01)+P(10).
]

This is a hardware-facing test of the statement:

[
D_{\rm endpoint}=0
\not\Rightarrow
\text{zero executed hardware exposure}.
]

---

## 3. IonQ simulator preflight

IonQ ideal and Forte-like noise simulators were used as pre-QPU checks.

The ideal simulator gives zero odd-parity residual for endpoint-equivalent identity loops.

Forte-like noise models show a strong increase of (P_{\rm odd}) with the number of executed native ZZ operations. A linear regression of the form

[
P_{\rm odd}
===========

\alpha
+
\beta_N N_{\rm ZZ}
+
\eta_\theta \theta_{L1}
]

shows that the dominant predictor is the executed native-ZZ operation count (N_{\rm ZZ}), not the continuous angle ledger (\theta_{L1}=\sum_i|\theta_i|).

Representative simulator regression:

```text
P_odd = 0.002636 + 0.007664 N_ZZ + 0.005420 theta_L1
R^2   = 0.9590
```

A broader multi-noise-model check gives:

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
The IonQ/Forte noise simulators support an executed-operation exposure model.
The effective hardware ledger is primarily N_ZZ rather than sum |theta_i|.
```

This simulator result is not QPU evidence. It is a preflight check motivating real QPU tests.

---

## 4. Preliminary IonQ Forte Enterprise QPU pilot

A preliminary IonQ Forte Enterprise QPU pilot was run using native-ZZ identity-loop circuits.

Backend:

```text
qpu.forte-enterprise-1
```

Execution mode:

```text
dry_run = false
```

Shots:

```text
200 shots per circuit per run
```

The tested circuits were:

```text
ZZ(0)^1
ZZ(0)^4
ZZ(0)^8
```

Two independent 200-shot runs were performed. The odd-parity residual increased reproducibly with the number of executed native ZZ operations.

### Run 1

```text
ZZ(0)^1:  P_odd = 0.010
ZZ(0)^4:  P_odd = 0.020
ZZ(0)^8:  P_odd = 0.105
```

### Run 2

```text
ZZ(0)^1:  P_odd = 0.005
ZZ(0)^4:  P_odd = 0.015
ZZ(0)^8:  P_odd = 0.055
```

### Combined result

Each combined point uses 400 shots.

| Circuit   | Odd counts | Total shots | (P_{\rm odd}) |
| --------- | ---------: | ----------: | ------------: |
| (ZZ(0)^1) |          3 |         400 |        0.0075 |
| (ZZ(0)^4) |          7 |         400 |        0.0175 |
| (ZZ(0)^8) |         32 |         400 |        0.0800 |

The combined trend is:

[
P_{\rm odd}!\left(ZZ(0)^8\right)

>

P_{\rm odd}!\left(ZZ(0)^4\right)

>

P_{\rm odd}!\left(ZZ(0)^1\right).
]

Interpretation:

```text
The preliminary QPU data support the hardware-facing prediction that endpoint-equivalent native-ZZ identity circuits can accumulate residual error with executed native operation exposure.
```

This is preliminary QPU pilot evidence, not a full hardware benchmark. Larger shot counts, interleaved randomized ordering, calibration controls, and repeated sessions are needed for a dedicated hardware study.

---

## 5. What is validated where?

| Claim                                                             | Validation source                           |
| ----------------------------------------------------------------- | ------------------------------------------- |
| (dt_{\rm info}=d\Phi_g/H_{\rm cap}) reconstructs elapsed time     | Core Q3/Q4/Q5 numerical validation          |
| (H_{\rm cap}) is fixed by the metric speed of the true generator  | Core selection-rule validation              |
| Wrong generator / wrong metric / endpoint-only assignments fail   | Core negative controls                      |
| Closed loops can have (D_{\rm endpoint}=0) but nonzero path       | Two-qubit closed-loop validation            |
| Endpoint-equivalent circuits can have nonzero hardware exposure   | IonQ simulator and preliminary QPU pilot    |
| Forte-like effective ledger is mainly executed ZZ operation count | IonQ simulator regression                   |
| True hardware residual increases with executed native ZZ depth    | Preliminary IonQ Forte Enterprise QPU pilot |

---

## 6. Important limitations

The IonQ QPU data in this repository are preliminary.

They should be interpreted as:

```text
pilot evidence for the hardware-facing path-exposure prediction
```

not as:

```text
a complete experimental verification of the full theory
```

The core path-capacity equations are validated by controlled state-trajectory simulations. The IonQ experiments test a hardware-facing consequence: endpoint-equivalent circuits may accumulate different residual errors because they execute different native-operation paths.

The current QPU pilot supports an operation-count exposure ledger,

[
\Phi_{\rm exec}^{\rm Forte}
\sim
N_{\rm ZZ},
]

rather than a clearly resolved continuous-angle ledger,

[
\Phi_{\rm exec}
\sim
\sum_i|\theta_i|.
]

---

## 7. Reproducibility notes

Do not commit API keys to this repository.

Recommended local setup:

```python
import os
from getpass import getpass

os.environ["IONQ_API_KEY"] = getpass("Paste IonQ API key: ")
```

The IonQ API key should be stored only as an environment variable or secret.

The QPU pilot jobs used native gates:

```json
{
  "gateset": "native",
  "circuit": [
    {"gate": "zz", "targets": [0, 1], "angle": 0.0}
  ]
}
```

For the identity-loop depth tests:

```python
def zz0_list(n):
    return [{"gate": "zz", "targets": [0, 1], "angle": 0.0} for _ in range(n)]
```

---

## 8. Recommended next steps

1. Repeat the three-point QPU test in another session:

[
ZZ(0)^1,\quad ZZ(0)^4,\quad ZZ(0)^8.
]

2. Increase shots after confirming the trend remains stable.

3. Interleave circuit order to suppress drift:

```text
8ZZ → 1ZZ → 4ZZ
4ZZ → 8ZZ → 1ZZ
1ZZ → 8ZZ → 4ZZ
```

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

6. Compare simulator and QPU coefficients.

---

## 9. Summary

The core path-capacity principle is validated by direct state-trajectory simulations.

IonQ/Forte simulator tests support the hardware-facing extension: endpoint-equivalent native-ZZ circuits accumulate residual error primarily with executed native operation count.

Preliminary IonQ Forte Enterprise QPU pilot data reproduce the same qualitative trend:

[
P_{\rm odd}(8ZZ)>P_{\rm odd}(4ZZ)>P_{\rm odd}(1ZZ).
]

This provides first hardware-facing support for the idea that endpoint equivalence does not imply zero executed physical cost.



Y.Y.N., L. (2026, June 19). A Metric-Compatible Path-Capacity Principle for Quantum Control Time. Zenodo. https://doi.org/10.5281/zenodo.20767791



