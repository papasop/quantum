# Claim scope

## Supported v0.6.0 statement

For the declared rotated-surface-code logical Z-memory task, frozen synthetic
schedule-to-fault map, Stim circuit construction, independently sampled
reference and optimized arms, and PyMatching decoder constructed from each
arm's detector error model, the preregistered v0.6.0 cohort supports:

1. response-fibre schedule optimization at every tested seed and distance;
2. positive decoded-logical-failure reduction in 15 of 15 fixed-distance
   cases;
3. a reduction of the minimum Wilson-resolved distance from 11 to 9 at
   \(p_L^{\mathrm{target}}=0.025\) on 3 of 3 new prospective seeds;
4. a corresponding 45.34% reduction in active-physical-qubit-round volume.

The target 0.025 was frozen before v0.6.0 from the common development interval
identified by v0.5.2:

```text
[0.022859700317492494, 0.028340340056867457)
```

The prospective seeds were:

```text
20290107, 20290119, 20290203
```

## Unsupported statements

The repository does not establish:

- an accuracy threshold or threshold improvement;
- an asymptotic logical-error exponent improvement;
- universality across surface-code circuits, decoders, or noise models;
- a compiler-derived pulse-to-fault relation;
- agreement with calibrated hardware noise;
- an experimental or QPU resource advantage;
- an interval-certified theorem.

The response fibre in this experiment fixes the normalized mean of each
identity-layer schedule. It should not be described as preservation of an
arbitrary compiled physical gate under every hardware constraint.

## Correct concise wording

> In a frozen synthetic rotated-surface-code audit, response-fibre schedule
> optimization prospectively reduced the minimum Wilson-resolved distance from
> 11 to 9 at a target logical failure probability of 0.025 on all three new
> seeds, corresponding to a 45.34% active-physical-qubit-round saving. No
> threshold, exponent, compiler, or hardware claim is made.
