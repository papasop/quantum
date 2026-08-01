# Response-Fibre Fault Tolerance

Numerical audits of whether a control flow can preserve a declared ideal
logical task while reducing decoded logical failure and the physical resources
needed to reach a fixed reliability target.

## Prospective v0.6.0 result

On a frozen rotated-surface-code experiment using Stim and PyMatching, the
response-fibre schedule lowered the minimum resolved distance at the declared
logical-failure target:

\[
d_{\mathrm{reference}}=11 \longrightarrow d_{\mathrm{flow}}=9,
\qquad p_L^{\mathrm{target}}=0.025.
\]

This occurred on all three new prospective seeds. With 241 active qubits at
distance 11 and 161 at distance 9, and with the number of syndrome rounds set
equal to the code distance, the resolved physical-qubit-round saving was

\[
1-\frac{161\times 9}{241\times 11}=45.34\%.
\]

| Frozen prospective gate | Result |
|---|---:|
| Prospective seeds passing | 3 / 3 |
| Fixed-distance cases passing | 15 / 15 |
| Minimum decoded-failure reduction | 17.95% |
| Minimum two-sample z score | 23.65 |
| Distance crossover | 11 to 9 |
| Physical-qubit-round saving | 45.34% |

Protocol SHA-256:

```text
fec91e30001712f3d9ac84c0e45a6b70f2d5ae7189d3c9ac6d1096d47505cbf6
```

Reference report certificate SHA-256 (before its self field):

```text
e4658480b6a4cb71caadfb6532453c4a31ba11f732dfaf36ffd66cc10921138c
```

## Claim boundary

The result is a prospective numerical mechanism result under a frozen,
synthetic schedule-to-fault map. It is **not**:

- a fault-tolerance threshold theorem;
- evidence that the asymptotic distance-suppression exponent changed;
- a compiler-derived or calibrated physical-noise result;
- a hardware or QPU advantage claim;
- a formal interval-arithmetic certificate.

The fitted exponent differences changed sign across the three seeds, so the
supported interpretation is a finite-reliability resource/intercept advantage,
not an asymptotic scaling improvement.

See [docs/CLAIM_SCOPE.md](docs/CLAIM_SCOPE.md) for the exact statement.

## Reproduce v0.6.0

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
python src/ft_unit_change_time_rotated_surface_code_v0_6_0.py
```

The full frozen run evaluates 30 million decoded samples in total and may take
roughly 20 minutes, depending on CPU and memory performance. It writes
`protocol.json` and `report.json` under
`ft_unit_change_time_v0_6_0_results/`.

A lightweight pipeline check is available:

```bash
python src/ft_unit_change_time_rotated_surface_code_v0_6_0.py --quick
```

`--quick` changes the frozen protocol and is therefore never eligible for the
prospective v0.6.0 claim.

Verify the repository files and compile every Python source:

```bash
python tools/verify_release.py
```

## Evidence ladder

| Version | Experiment | Status |
|---|---|---|
| v0.1.0 | Exact-probability toy repetition code | Mechanism supported |
| v0.2.0 | Phenomenological repeated-syndrome graph | Mechanism supported |
| v0.3.0 | Prospective Stim/PyMatching gate-level repetition code | Mechanism supported |
| v0.4.0 | Fixed-reliability repetition-code crossover | Distance 9 to 7 supported |
| v0.5.0 | First rotated-surface-code resource audit | Fail-closed; initialization stalls exposed |
| v0.5.1 | Feasible-initialization repair on reused seeds | Regression repair supported; no new prospective claim |
| v0.5.2 | Lower-noise, rounds-equal-distance development calibration | Development-only target interval identified |
| v0.6.0 | Frozen prospective rotated-surface-code audit | Distance 11 to 9 and 45.34% resource saving supported |

The v0.5.0 negative result is retained intentionally. The v0.5.1 projection
repair is part of the methodological record and was completed before the
v0.6.0 prospective seeds were evaluated.

## Repository structure

```text
src/                 versioned standalone audits
results/v0.6.0/      compact frozen reference summary
docs/                claim boundary and evidence history
tools/               integrity and syntax verification
paper/               manuscript handoff notes
.github/workflows/   lightweight structural checks
```

## Relation to geometric flow

The upstream geometric idea is to move within an implementation fibre while
decreasing a secondary objective:

\[
R_{\mathrm{logical}}(\theta(s))=c,
\qquad \frac{d}{ds}V_{\mathrm{fault}}(\theta(s))<0.
\]

This repository tests that idea as a fault-tolerance application. The theory
and certified local-flow work remain in
[papasop/Geometric-Flow](https://github.com/papasop/Geometric-Flow).

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Until the
companion manuscript receives a persistent identifier, cite this repository by
version and commit or release tag.

## License

No open-source license is asserted in this release. Copyright remains with the
author unless a license file is added explicitly.
