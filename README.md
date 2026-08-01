# Response-Fibre Fault Tolerance

[![Structural checks](https://github.com/papasop/quantum/actions/workflows/structural-checks.yml/badge.svg)](https://github.com/papasop/quantum/actions/workflows/structural-checks.yml)

Can schedule freedom reduce the code distance needed at a fixed logical
reliability while preserving a declared response? This repository develops a
fail-closed numerical test of that question using a synthetic
schedule-to-fault map, Stim rotated-surface-code circuits, and PyMatching.

## Main result

In the frozen prospective v0.6.0 experiment, documented and evidence-synchronised
in v0.6.2, response-fibre schedule optimization changed the minimum **tested**
distance satisfying the one-sided 95% Wilson criterion at
`p_L <= 0.025` from

```text
reference: d = 11
flow:      d = 9
```

on all three frozen prospective seeds. With syndrome rounds set equal to code
distance, this corresponds to a 45.34% reduction in the declared
active-coordinate qubit-round proxy:

```text
1 - (161 x 9) / (241 x 11) = 0.4534138061.
```

| Frozen prospective gate | Result |
|---|---:|
| Prospective seeds passing | 3 / 3 |
| Fixed-distance cases passing | 15 / 15 |
| Minimum decoded-failure reduction | 17.86% |
| Minimum two-sample z-score | 24.86 |
| Minimum tested-distance crossover | 11 to 9 |
| Active-coordinate qubit-round proxy reduction | 45.34% |

The complete eight-page paper is available as
[`paper/v0.6.2/response_fibre_fault_tolerance_v0_6_2.pdf`](paper/v0.6.2/response_fibre_fault_tolerance_v0_6_2.pdf),
with its LaTeX source, plotted CSV data, and independent Wilson-bound checker in
the same directory.

## What is preserved, and what changes

Each data-qubit schedule row satisfies the declared affine response

```text
mean_j u[q,j] = 1,
```

with fixed amplitude bounds. Tangent-projected descent preserves this response
while reducing a synthetic exposure objective. The resulting single-data and
adjacent-pair fault probabilities are injected into a constructively inserted
Stim identity layer and evaluated by matched PyMatching decoders.

The schedule does **not** generate a different physical unitary evolution in
this repository. The tested fibre is an affine schedule constraint, not a
complete-unitary equivalence class.

## Claim boundary

Supported:

- a frozen prospective numerical mechanism in the declared synthetic model;
- positive decoded-logical-failure reduction in 15/15 fixed-distance cases;
- a 3/3 minimum-tested-distance crossover from 11 to 9;
- a 45.34% saving in the declared active-coordinate qubit-round proxy.

Not supported:

- a fault-tolerance threshold or asymptotic exponent improvement;
- a compiler-derived or calibrated schedule-to-noise relation;
- a hardware, QPU, or physical resource advantage;
- complete-unitary preservation by physical control schedules;
- out-of-model prediction validation;
- a formal interval-arithmetic theorem.

The analytic fault map optimized by the flow is also the map injected into
Stim. Monte Carlo outcomes are held out from optimization, but the experiment
is not an independent validation of the synthetic map itself. See
[`docs/CLAIM_SCOPE.md`](docs/CLAIM_SCOPE.md) for the exact statement.

## Reproduce and verify

Python 3.10 or newer is recommended. The claim-bearing environment used Python
3.12.13 with NumPy 2.0.2, Stim 1.16.0, and PyMatching 2.4.0.

```bash
python -m pip install -r requirements.txt
python tools/verify_release.py
```

Run the frozen full audit:

```bash
python src/ft_unit_change_time_rotated_surface_code_v0_6_0.py
```

The run evaluates 30 million decoded samples and writes `protocol.json` and
`report.json`. A lightweight pipeline test is available:

```bash
python src/ft_unit_change_time_rotated_surface_code_v0_6_0.py --quick
```

`--quick` changes the protocol and is ineligible for the prospective claim.

The repository already archives the frozen protocol and complete compressed
Monte Carlo report. Verify the paper's Wilson bounds independently with:

```bash
cd paper/v0.6.2
python verify_wilson.py
```

## Hash-bound evidence

| Artifact | SHA-256 |
|---|---|
| Protocol | `fec91e30001712f3d9ac84c0e45a6b70f2d5ae7189d3c9ac6d1096d47505cbf6` |
| Canonical report certificate, before self field | `2db9620419ac5a7ff64510c65e0d391c4603b6c361fdd8aadd2d9f96165cbc79` |
| `results/v0.6.0/report.json.gz` | `f9edf8692aaa0f116cc6584507e7f326d184831f251669aa6e2dd2dd143bb95a` |
| Claim certificate | `6754415ceb7bb662982c052d2045c7fafced8a7568bf86db7f94a86338a0f00f` |

The archived report contains all raw per-case failure counts, Wilson upper
bounds, and crossover decisions. The archived run completed in 478.948 seconds
on the reported environment; runtime is hardware- and batch-dependent.

## Evidence ladder

| Version | Experiment | Status |
|---|---|---|
| v0.1.0 | Exact-probability toy repetition code | Mechanism supported |
| v0.2.0 | Phenomenological repeated-syndrome graph | Mechanism supported |
| v0.3.0 | Prospective Stim/PyMatching gate-level repetition code | Mechanism supported |
| v0.4.0 | Fixed-reliability repetition-code crossover | Distance 9 to 7 supported |
| v0.5.0 | First rotated-surface-code resource audit | Failed closed; initialization stalls exposed |
| v0.5.1 | Feasible-initialization repair on reused seeds | Regression repair only |
| v0.5.2 | Lower-noise, rounds-equal-distance calibration | Development-only target interval |
| v0.6.0 | Frozen prospective rotated-surface-code audit | Distance 11 to 9; 45.34% proxy saving |
| v0.6.1 | Evidence and claim-boundary repair | Full report and exact claim certificate archived |
| v0.6.2 | Paper and README evidence synchronization | Eight-page manuscript, tables, figure data, and hashes aligned |

Negative and development-stage results are retained intentionally. See
[`docs/EVIDENCE_HISTORY.md`](docs/EVIDENCE_HISTORY.md).

## Repository structure

```text
src/                 versioned standalone numerical audits
results/v0.6.0/      frozen protocol, full report, summary, and certificate
docs/                claim boundary and evidence history
tools/               integrity and syntax verification
paper/v0.6.2/        complete paper, LaTeX, figure data, and Wilson checker
.github/workflows/   structural checks
```

## Relation to geometric flow

The upstream geometric principle is to move inside an implementation fibre
while decreasing a secondary objective:

```text
R(theta(s)) = c,
dV(theta(s))/ds < 0.
```

This repository tests a fault-tolerance application of that principle. The
theory and certified local-flow work remain in
[`papasop/Geometric-Flow`](https://github.com/papasop/Geometric-Flow).

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Until the
paper receives a persistent identifier, cite the repository by release tag and
commit together with the companion manuscript.

Released under the [MIT License](LICENSE).
