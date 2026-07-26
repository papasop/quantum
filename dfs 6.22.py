#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFS OPERATIONAL PROTOCOL + CHANNEL SUPPORT AUDIT v6.2.2
=======================================================

Protocol-first upgrade of v5.1.  The operational primitive is a predeclared
preparation/evolution/readout protocol Pi, not an isolated scalar chosen after
seeing a desired zero.  The script distinguishes four objects:

  1. local model rate       j_Pi(rho) = Tr(L_0^dagger L_0 rho),
  2. accumulated cost       J_Pi[rho] = integral j_Pi(rho_t) dt,
  3. finite channel witness E_ch      = 1/2 ||J(E_noisy)-J(E_ideal)||_1,
  4. abstract tangent F     = NOT CONSTRUCTED by this script.

What v6.2.2 closes in v6.2.1
----------------------------
Symmetry, not correctness.  v6.2.1 gave Layer 2 a point-by-point comparison
against its commitment, but Layer 2B still verified only the receipt hash.
Execution order there was already right -- the joint transfer predictions were
written and hashed before the data was drawn -- so this was never a run-time
defect.  The gap was semantic: a hash proves the FILE is intact, not that the
values that actually entered the pulls came from it.

v6.2.2 applies the same coverage-then-deviation structure to Layer 2B:

  * `joint_commitment_coverage` re-reads frozen_joint_predictions.json and
    compares every scored `predicted_visibility` against it by
    (offset, duration), and additionally checks that the committed gamma_hat
    and delta0_hat are the estimates actually used to build the predictions.
  * the new gate `every_scored_joint_prediction_matches_the_commitment_on_disk`
    joins `prediction_freezes_verify_from_disk`, so the sentence "every scored
    prediction matches the commitment" now holds for both layers.

No number changes; the deviation is 0.0 in the reference run.

What v6.2.1 closed in v6.2
--------------------------
An audit gap, not a scientific one.  v6.2's gate
`every_scored_prediction_matches_the_commitment_on_disk` compared the Ramsey
held-out and logical-X_L transfer predictions point by point, but for the E_ch
payload it recorded only a COUNT.  The values that actually entered the error
calculation -- `predicted_encoded_cost_closed_form` -- were never checked
against the file, so the gate name promised more than it verified.  Accurately,
v6.2 established: every scored VISIBILITY prediction matches the commitment,
while the encoded-cost payload is independently hashed and preserved.

v6.2.1 closes the semantics:

  * `maximum_encoded_cost_commitment_deviation` compares each E_ch prediction
    against `committed["encoded_cost_extrapolation"]` by delta, and the gate
    requires BOTH deviations to be exactly zero and BOTH families to be fully
    covered.  The gate name is now no stronger than what is checked.
  * coverage is established BEFORE any deviation is computed.  v6.2 evaluated
    max(...) first, so a scored point absent from the commitment raised a
    KeyError and crashed the run rather than failing the gate.  A missing
    commitment is now reported as `missing_visibility_points` /
    `missing_encoded_cost_points` and fails the gate.

No number changes; the E_ch deviation is 0.0 in the reference run, which is
what v6.2 assumed without checking.

What v6.2 changed in v6.1
-------------------------
Three refinements.  No physics, no tolerance and no estimator is touched, and
every reported number is bit-identical to v6.1 and to v6.0.

  1  NAMING OF THE GUARANTEE.  `commitment_verifies_from_disk` establishes only
     that the serialized payload can be rebuilt from disk and rehashes to the
     recorded digest.  It does NOT independently establish that the payload was
     written before any outcome was computed -- there is no external timestamp
     and no third party.  The guarantee is EXECUTION-ORDER FREEZING WITHIN THE
     EXECUTABLE AUDIT, and it must not be written up as formal preregistration.
     v6.1 said this only in a source comment; v6.2 states it in the receipt
     file, in `gate_bucket_semantics`, and in `claim_boundary`, so the wording
     travels with the artefact.

  2  ONE COMMITMENT FOR EVERY LAYER-2 HELD-OUT PREDICTION.  v6.1 committed the
     E_ch extrapolation and the Layer-2B joint transfer, but the same-observable
     Ramsey held-out points and the cross-observable X_L transfer points relied
     on execution order alone with nothing on disk to check them against.  The
     order is now: calibration fit -> build every held-out prediction -> one
     commitment file -> draw all held-out counts -> score.  The new gate
     `every_scored_prediction_matches_the_commitment_on_disk` re-reads that file
     and compares it against every value actually scored, so a prediction
     recomputed after the fact, or a scored point that was never committed,
     makes the gate fail.

  3  SYMBOL ALIGNMENT WITH THE PAPER.  The contraction family is written H_s =
     s H in the manuscript, so `contraction_epsilons` -> `contraction_scales`,
     the row key `epsilon` -> `scale_s`, and
     `every_positive_epsilon_path_nonconstant` ->
     `every_positive_scale_path_nonconstant`.  Values and semantics unchanged.

  NOTE.  Item 3 renames a key inside the protocol manifest, so
  `protocol_sha256` necessarily differs from the v6.0/v6.1 value
  7c465d17bd212f5af4106432a13929a1de30a9f94db92f8003bdeefb9e7c1749.  That is a
  rename of a frozen field, not a change of protocol.

What v6.1 fixed in v6.0
-----------------------
The physics, the Lindblad model, the estimators and every tolerance are
unchanged, and v6.0's numbers reproduce.  What was broken was the AUDITING
MACHINERY -- the parts that decide whether a gate can fail at all.

  B1  freeze_and_commit().  v6.0 wrote frozen["predictions"] = prediction_rows
      BY REFERENCE and then mutated those same dicts when the truth was
      revealed, so the in-memory frozen object no longer hashed to its own
      receipt (observed: ef985ffa... -> cec041c2...), and no gate ever re-read
      the file.  Payloads are now deep-copied, hashed, read back from disk and
      re-hashed, and the result is a gate that can fail.

  B2  Layer 2B committed nothing.  It drew the held-out X_L data, computed the
      residuals and the pulls, and only afterwards wrote a file named
      "frozen_joint_predictions.json" -- with no hash and no receipt.  That
      file documented a comparison already made.  Predictions are now written
      and hashed BEFORE the transfer data is drawn.

  B3  identifiability_witness() evaluated the Fisher matrix at cfg.true_gamma
      and cfg.true_delta0, and its gates VOTE.  An experiment design may not
      read the truth.  The voting instance now sits at the stated design prior;
      a second, explicitly non-voting instance reports the achieved information
      at the fitted values.

  B4  The single-setting rank-deficiency test used single[0]/single[-1], which
      is NEGATIVE (observed: -1.090e-17) for a numerically rank-one matrix, so
      "ratio <= 1e-6" passed on SIGN rather than on magnitude and would also
      have passed for a genuinely indefinite matrix.  It now uses magnitudes.

  B5  operational_negative_control().  v6.0 introduced sixteen voting Layer-0
      gates and not one mutation able to trip any of them, which contradicts
      this script's own doctrine that a gate never observed to fail is not
      evidence.  Four mutations are added.  M7 (H = 0) is the one that matters:
      it demonstrates that `finite_nonconstant_path` really does exclude
      standing still, which is the only thing separating the zero-cost claim
      from a tautology.

  B6  operational_gates["protocol_frozen_before_outcomes"] was hardcoded True.
      That is an unfailable gate -- precisely the defect v3.1 was criticised
      for.  It is replaced by a disk round-trip verification and renamed to
      state only what it actually checks.

Robustness, non-scientific: NumPy 1.x fallback for the trapezoid rule; explicit
errors instead of ZeroDivisionError when a control offset cancels the prior
imbalance; explicit error instead of a NumPy exception when the joint coverage
sweep yields fewer than two usable replicates; stdout layer labels aligned with
the certificate keys; the positive control gated on its own cost threshold
rather than borrowing the trace-distance threshold.

Checked and deliberately NOT changed: the kappa=0 false-alarm gate and
kappa_detect are single-draw statistics.  Over 300 reseeds the kappa=0 transfer
max-pull had mean 1.87, p95 2.87, max 3.88 against a 4-sigma threshold (0/300
false alarms), and kappa_detect was 5e-4 in 40/40 reseeds.  At these shot counts
the single draw is not a defect and replicating it would only hide an already
quantified fact.

What v6 added over v5.1
-----------------------
  L0    A frozen protocol manifest is serialized and hashed before any outcome
        is computed.
  L1A   The collective-dephasing DFS is certified algebraically as the kernel
        of the predeclared local rate and as an invariant subspace of H.
  L1B   A finite nonconstant trajectory is propagated inside that kernel.  Its
        accumulated declared jump cost and dissipator activity vanish.
  L1C   The same frozen meter gives positive cost on a control state.
  L1D   A contraction family H_s = s H gives nonconstant zero-cost paths for
        every s>0 and approaches the constant path as s->0.  This is the
        finite-attainment/integrability witness.
  L2+   The v5.1 channel reduction, symmetry-opening law, calibration, joint
        identification, negative controls, and misspecification sweeps remain
        as model support rather than as a universal Principle-R claim.

Inherited v5 additions
----------------------
  L1.0  The encoded channel is REDUCED, exactly, to a one-qubit channel
        (H_L = J X_L, single jump sqrt(gamma) delta Z_L).  Everything else in
        Layer 1 -- the exact zero at delta=0, the absence of leakage, the
        delta^2 law -- is then a corollary of an algebraic identity.  The
        script is a witness for a two-line proof, not a discovery engine.
  L2B   delta is no longer assumed known.  An unknown intrinsic imbalance
        delta0 is identified jointly with gamma from three KNOWN control
        offsets, with an explicit Fisher rank witness showing that a single
        setting leaves the pair unidentifiable (the model depends on them only
        through gamma*(2+delta0+offset)^2).  Transfer residuals are compared
        against shot noise AND propagated parameter uncertainty.
  L3    Model misspecification, not just code defects.  The data generator is
        given a term the estimator does not know about (non-collective
        dephasing, or amplitude damping) and kappa is swept.  The reported
        kappa_detect is the honest sensitivity of the protocol to unmodelled
        physics, measured separately for the same-observable held-out test and
        the cross-experiment transfer test.

Layer 1 -- channel-level structural audit
-----------------------------------------
Logical qubit encoded in span{|01>,|10>} of H = (J/2)(XX + YY) with a single
imbalanced collective-dephasing jump

    L_delta = sqrt(gamma) [ Z1 + (1 + delta) Z2 ].

The finite channel witness is the normalized Choi-state trace distance between
the encoded noisy and encoded ideal channels,
E_ch = 1/2 || J(noisy) - J(ideal) ||_1.  E_ch is not the local cost density.

Gates:
  L1.1  E_ch(delta=0) = 0 over a *duration sweep*, not one time.
  L1.2  the full four-dimensional physical channel is not the ideal channel.
  L1.3  L_delta leaves the code space exactly invariant (no leakage) for all
        tested delta, so the encoded channel is a genuine qubit channel.
  L1.4  QUANTITATIVE symmetry-breaking law:  E_ch = gamma * delta^2 * t
        to leading order.  Verified as a log-log slope and prefactor, not as
        "greater than 1e-8".
  L1.5  CPTP validity of the simulated superoperator: Choi positivity, unit
        trace, trace preservation, and the pre-symmetrization Hermiticity
        residual that v3.1 silently discarded.
  L1.6  Lindblad-representation regression tests.  v3.1 only tested one jump
        operator against copies of itself, which is invariant for algebraic
        reasons independent of the physics.  Added here: unitary mixing of two
        *linearly independent* jumps, and the inhomogeneous gauge freedom
        L -> L + c I with H -> H - (i/2)(conj(c) L - c L^dagger), which is the
        representation freedom that actually has content.

Layer 2 -- cross-experiment calibration
---------------------------------------
v3.1 fit gamma to synthetic counts generated from the *same closed form* used
to fit, then "predicted" a quantity that is exactly linear in gamma.  That
prediction gate could not fail whenever the gamma gate passed.

Here:
  * counts are generated by exact Liouvillian evolution, and the closed forms
    used by the estimator are verified against that evolution (gate L2.0), so
    the model itself is under test;
  * calibration uses the |00>,|11> coherence, which is an exact eigenpair of H
    (both are annihilated by XX + YY), so the closed form V = exp(-8 gamma t)
    holds with the Hamiltonian ON.  v3.1's implied |00>,|01> pair is rotated by
    H and its stated formula is wrong by ~0.26 in visibility at t = 0.9;
  * gamma_hat is frozen and used to predict a *different observable in a
    different subspace at unseen delta*: the logical X_L decay inside the code
    space, rate 2 gamma delta^2, against fresh finite-shot data (gate L2.5);
  * coverage of the asymptotic interval is measured over a seed sweep instead
    of a single draw;
  * predictions are serialized, hashed and re-verified from disk before the
    truth is computed, so the freeze is auditable rather than asserted.

Honest boundary
---------------
Exact NumPy/SciPy model evidence plus finite-shot synthetic data.  Not QPU
data, not laboratory calibration, not zero total energy, not a universal
realizability or Lorentzian claim.  "Blind" here means software-enforced
information separation over synthetic counts.  In Layer 2 delta is treated as a
known control parameter and only gamma is inferred; Layer 2B relaxes that.

Run:
    pip install -U numpy scipy matplotlib
    python dfs_operational_v6_2_2.py
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dfs_channel_v5_matplotlib"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

VERSION = "DFS-OPERATIONAL-PROTOCOL-v6.2.2"


def integrate(values: Any, times: Any) -> float:
    """FIX v6.1: np.trapezoid is NumPy>=2.0 only; fall back on NumPy 1.x."""
    rule = getattr(np, "trapezoid", None)
    if rule is None:
        rule = np.trapz
    return float(rule(values, times))


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    exchange_J: float = 1.0
    true_gamma: float = 0.20
    target_duration: float = math.pi / 2.0

    # --- Layer 0/1: protocol-first operational audit ---
    # The path is sampled only after the protocol manifest has been written.
    trajectory_samples: int = 401
    # RENAMED v6.2 (was contraction_epsilons): the paper writes H_s = s H,
    # so the code uses the same symbol.  Values and semantics unchanged.
    contraction_scales: tuple[float, ...] = (
        1.0, 0.5, 0.25, 0.125, 0.0625,
    )
    nonconstant_trace_distance_minimum: float = 1.0e-8
    contraction_monotonic_tolerance: float = 1.0e-13
    # FIX v6.1: the positive control is an accumulated COST, not a trace
    # distance.  v6.0 gated it against nonconstant_trace_distance_minimum,
    # which is a threshold on a different quantity in different units.
    same_meter_minimum_accumulated_cost: float = 1.0e-6

    # --- Layer 1 ---
    duration_sweep: tuple[float, ...] = (0.1, 0.5, math.pi / 2.0, 2.0, 5.0, 25.0)
    selection_deltas: tuple[float, ...] = (-0.08, -0.04, 0.0, 0.04, 0.08)
    leakage_deltas: tuple[float, ...] = (0.0, 0.04, 0.25, 0.5, 2.0)
    powerlaw_deltas: tuple[float, ...] = (
        0.005, 0.008, 0.012, 0.020, 0.032, 0.050,
    )
    # Per-slice scaling audit: every (gamma, t) slice is fitted separately and
    # the WORST slice is reported, instead of one aggregate number.
    scaling_gammas: tuple[float, ...] = (0.05, 0.20, 0.80)
    scaling_durations: tuple[float, ...] = (0.5, math.pi / 2.0, 3.0)
    scaling_exponent_tolerance: float = 0.01
    scaling_coefficient_tolerance: float = 0.02
    scaling_parity_tolerance: float = 1.0e-13

    # --- Layer 2: calibration on the H-invariant |00>,|11> coherence ---
    calibration_delta: float = 0.0
    calibration_times: tuple[float, ...] = (
        0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.85, 1.20,
    )
    heldout_times: tuple[float, ...] = (0.08, 0.25, 0.50, 0.95)
    shots_per_time: int = 200_000
    master_seed: int = 20260726

    # --- Layer 2: cross-experiment transfer to the logical X_L decay ---
    # (delta, times) pairs.  Rate is 2*gamma*delta^2, i.e. 16/delta^2 times
    # slower than the calibration rate 8*gamma.
    transfer_schedule: tuple[tuple[float, tuple[float, ...]], ...] = (
        (0.50, (2.0, 5.0, 9.0, 14.0, 20.0)),
        (0.25, (8.0, 20.0, 36.0, 56.0, 80.0)),
    )

    # --- Layer 2: E_ch extrapolation grid (delta is a known control knob) ---
    heldout_deltas: tuple[float, ...] = (
        -0.12, -0.06, -0.03, -0.015, 0.015, 0.03, 0.06, 0.12,
    )

    # --- Layer 2B: joint (gamma, delta0) identification ---
    # delta0 is now an UNKNOWN intrinsic imbalance.  The experimenter can add
    # a KNOWN control offset, so the total imbalance is delta0 + offset.
    true_delta0: float = 0.05
    control_offsets: tuple[float, ...] = (-0.8, 0.0, 0.8)
    joint_shots_per_time: int = 500_000
    # Time grids are designed from a stated PRIOR, never from the truth.
    design_prior_gamma: float = 0.25
    design_prior_delta0: float = 0.0
    design_time_units: tuple[float, ...] = (
        0.06, 0.15, 0.30, 0.50, 0.80, 1.20, 1.70, 2.40,
    )
    joint_transfer_offsets: tuple[float, ...] = (0.40, 0.15)
    joint_transfer_time_units: tuple[float, ...] = (0.2, 0.5, 0.9, 1.4, 2.0)
    joint_coverage_replicates: int = 400
    delta0_absolute_error_tolerance: float = 0.02
    identifiability_minimum_eigenvalue: float = 1.0e2
    identifiability_rank_deficiency_maximum: float = 1.0e-6

    # --- Layer 3: model-misspecification sensitivity ---
    misspecification_kappas: tuple[float, ...] = (
        0.0, 5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3, 1.0e-2, 2.0e-2, 5.0e-2,
    )
    misspecification_detection_sigma: float = 4.0

    # --- coverage sweep ---
    coverage_replicates: int = 1200
    coverage_nominal: float = 0.95
    coverage_sigma_allowance: float = 3.0

    # --- tolerances ---
    exact_zero_tolerance: float = 1.0e-13
    leakage_tolerance: float = 1.0e-13
    full_channel_minimum: float = 1.0e-3
    powerlaw_slope_tolerance: float = 0.01
    powerlaw_prefactor_tolerance: float = 5.0e-3
    cptp_tolerance: float = 1.0e-12
    representation_tolerance: float = 1.0e-12
    model_closed_form_tolerance: float = 1.0e-11

    gamma_relative_error_tolerance: float = 0.02
    residual_sigma_tolerance: float = 4.0
    transfer_sigma_tolerance: float = 4.0
    likelihood_bounds: tuple[float, float] = (1.0e-6, 2.0)
    boundary_margin: float = 1.0e-3

    # E_ch extrapolation: deliberately NOT looser than the gamma gate.
    # See `extrapolation_is_not_an_independent_test` in the certificate.
    ech_relative_error_tolerance: float = 0.02
    ech_model_relative_error_tolerance: float = 0.02


# ----------------------------------------------------------------------------
# io helpers
# ----------------------------------------------------------------------------
def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_object(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(clean(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """FIX (v3.1 bug): field names are the union of all keys, not rows[0]."""
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(clean(rows))


def freeze_and_commit(output: Path, name: str, payload: Any) -> tuple[str, bool]:
    """FIX v6.1 (B1): make a freeze falsifiable instead of asserted.

    v6.0 stored `frozen["predictions"] = prediction_rows` BY REFERENCE and then
    mutated those same dicts when the truth was revealed, so the in-memory
    frozen object no longer hashed to its own receipt, and no gate ever re-read
    the file.  Here the payload is deep-copied before serialization, the hash is
    taken over the copy, and the file is read BACK from disk and re-hashed.  The
    returned `verified` flag is a gate that can fail: tampering with the file
    after the fact makes the recomputed hash disagree with the receipt.

    Scope note (v6.2, item 1).  This proves the commitment is reproducible from
    disk.  It does NOT prove that no outcome was computed first: there is no
    external timestamp and no third party.  The ordering is enforced by where
    this function is called, and a reviewer must check the call site.  The
    correct write-up is "execution-order freezing within the executable audit",
    never "preregistration".
    """
    snapshot = copy.deepcopy(payload)
    path = output / f"{name}.json"
    save_json(path, snapshot)
    digest = sha256_object(snapshot)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    verified = bool(sha256_object(reloaded) == digest)
    save_json(
        output / f"{name}_receipt.json",
        {
            "sha256_of_canonical_payload": digest,
            "recomputed_from_disk_matches": verified,
            "guarantee": "execution-order freezing within the executable audit",
            "not_a_guarantee_of": "formal preregistration",
            "note": (
                "Hash is taken over the canonical (sorted-key, compact) JSON "
                "of the payload, not over the indented file bytes.  Written "
                "and hashed BEFORE any truth value or held-out draw was "
                "computed.  This establishes only that the payload can be "
                "rebuilt from disk and rehashes to the recorded digest.  There "
                "is no external timestamp and no third party, so it does NOT "
                "independently establish that the write preceded every "
                "computation; that ordering is enforced by the call site and "
                "must be checked there.  Do not describe this as "
                "preregistration."
            ),
        },
    )
    return digest, verified


def create_unique_output_dir(requested: str | None) -> Path:
    base = Path(
        requested
        or f"dfs_operational_v6_2_2_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    for candidate in [base] + [
        base.with_name(f"{base.name}_run{i:02d}") for i in range(2, 1000)
    ]:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            if candidate != base:
                print(f"[output] {base} exists; preserving it, using {candidate}")
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate an output directory from {base}.")


# ----------------------------------------------------------------------------
# quantum model
# ----------------------------------------------------------------------------
def ket(index: int, dimension: int = 4) -> np.ndarray:
    state = np.zeros(dimension, dtype=complex)
    state[index] = 1.0
    return state


def build_operators(cfg: Config) -> dict[str, np.ndarray]:
    identity = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return {
        "H": 0.5 * cfg.exchange_J * (np.kron(x, x) + np.kron(y, y)),
        "Z1": np.kron(z, identity),
        "Z2": np.kron(identity, z),
        "X1": np.kron(x, identity),
        "SM1": np.kron(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex), identity),
        "SM2": np.kron(identity, np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)),
        "V": np.column_stack([ket(1), ket(2)]),
        "I4": np.eye(4, dtype=complex),
    }


def protocol_manifest(cfg: Config) -> dict[str, Any]:
    """Predeclare the operational experiment before computing any outcome."""
    return {
        "protocol_id": "collective-dephasing-DFS-Pi-v1",
        "model_domain": "exact two-qubit Lindblad model",
        "basis_order": ["|00>", "|01>", "|10>", "|11>"],
        "preparation": {
            "zero_path_input": "|01>",
            "positive_control_input": "(|00>+|01>)/sqrt(2)",
        },
        "hamiltonian": "H=(J/2)(XX+YY)",
        "declared_meter": {
            "jump": "L0=sqrt(gamma)(Z1+Z2)",
            "local_rate": "j_Pi(rho)=Tr(L0^dagger L0 rho)",
            "accumulated_cost": "J_Pi[rho]=integral_0^T j_Pi(rho_t) dt",
            "dissipator_activity": "||D[L0](rho_t)||_F",
        },
        "finite_channel_witness": (
            "E_ch=1/2||J(encoded noisy channel)-J(encoded ideal channel)||_1"
        ),
        "frozen_parameters": {
            "exchange_J": cfg.exchange_J,
            "gamma": cfg.true_gamma,
            "duration": cfg.target_duration,
            "trajectory_samples": cfg.trajectory_samples,
            "contraction_scales": cfg.contraction_scales,
        },
        "decision_rules": {
            "exact_zero_tolerance": cfg.exact_zero_tolerance,
            "leakage_tolerance": cfg.leakage_tolerance,
            "nonconstant_trace_distance_minimum": (
                cfg.nonconstant_trace_distance_minimum
            ),
        },
        "claim_boundary": (
            "The protocol defines a model-local jump cost. It does not define "
            "zero total energy, a universal tangent density F, Principle R, "
            "or Lorentzian signature."
        ),
    }


def dissipator(jump: np.ndarray, rho: np.ndarray) -> np.ndarray:
    kernel = jump.conj().T @ jump
    return jump @ rho @ jump.conj().T - 0.5 * (kernel @ rho + rho @ kernel)


def pure_trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    overlap = abs(np.vdot(left, right)) ** 2
    return float(math.sqrt(max(0.0, 1.0 - min(1.0, overlap))))


def operational_zero_mode_audit(
    cfg: Config,
    operators: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Certify a finite, integrable zero-cost trajectory for frozen protocol Pi."""
    hamiltonian = operators["H"]
    encoding = operators["V"]
    projector = encoding @ encoding.conj().T
    complement = operators["I4"] - projector
    jump = jump_for_delta(cfg.true_gamma, 0.0, operators)
    kernel = jump.conj().T @ jump

    analytic = {
        "norm_L0_P_DFS": float(np.linalg.norm(jump @ projector, ord="fro")),
        "norm_K0_P_DFS": float(np.linalg.norm(kernel @ projector, ord="fro")),
        "norm_commutator_H_P_DFS": float(
            np.linalg.norm(hamiltonian @ projector - projector @ hamiltonian)
        ),
        "norm_Q_H_P_DFS": float(
            np.linalg.norm(complement @ hamiltonian @ projector)
        ),
        "DFS_rank": int(round(float(np.real(np.trace(projector))))),
        "cost_kernel_eigenvalues": np.linalg.eigvalsh(kernel),
    }
    analytic_gates = {
        "L0_annihilates_DFS": analytic["norm_L0_P_DFS"]
        <= cfg.exact_zero_tolerance,
        "K0_annihilates_DFS": analytic["norm_K0_P_DFS"]
        <= cfg.exact_zero_tolerance,
        "Hamiltonian_preserves_DFS": (
            analytic["norm_commutator_H_P_DFS"] <= cfg.exact_zero_tolerance
            and analytic["norm_Q_H_P_DFS"] <= cfg.exact_zero_tolerance
        ),
        "DFS_equals_cost_kernel_dimension": int(
            np.count_nonzero(
                np.abs(analytic["cost_kernel_eigenvalues"])
                <= cfg.exact_zero_tolerance
            )
        )
        == analytic["DFS_rank"],
    }

    initial = ket(1)
    initial_rho = np.outer(initial, initial.conj())
    times = np.linspace(0.0, cfg.target_duration, cfg.trajectory_samples)
    trajectory_rows: list[dict[str, Any]] = []
    jump_rates: list[float] = []
    activities: list[float] = []
    distances: list[float] = []
    state_leakages: list[float] = []
    tangent_leakages: list[float] = []
    tangent_kernel_residuals: list[float] = []

    for point in times:
        state = expm(-1.0j * hamiltonian * point) @ initial
        rho = np.outer(state, state.conj())
        tangent = -1.0j * hamiltonian @ state
        rate = max(0.0, float(np.real(np.trace(kernel @ rho))))
        activity = float(np.linalg.norm(dissipator(jump, rho), ord="fro"))
        distance = pure_trace_distance(initial, state)
        state_leakage = float(np.real(np.vdot(state, complement @ state)))
        tangent_leakage = float(np.linalg.norm(complement @ tangent))
        tangent_kernel = float(np.linalg.norm(kernel @ tangent))
        jump_rates.append(rate)
        activities.append(activity)
        distances.append(distance)
        state_leakages.append(abs(state_leakage))
        tangent_leakages.append(tangent_leakage)
        tangent_kernel_residuals.append(tangent_kernel)
        trajectory_rows.append(
            {
                "time": point,
                "trace_distance_from_initial": distance,
                "jump_rate": rate,
                "dissipator_activity": activity,
                "DFS_state_leakage": abs(state_leakage),
                "DFS_tangent_leakage": tangent_leakage,
                "cost_kernel_tangent_residual": tangent_kernel,
            }
        )

    accumulated_cost = integrate(jump_rates, times)
    accumulated_activity = integrate(activities, times)
    trajectory = {
        "maximum_trace_distance_from_initial": max(distances),
        "accumulated_declared_jump_cost": accumulated_cost,
        "accumulated_dissipator_activity": accumulated_activity,
        "maximum_DFS_state_leakage": max(state_leakages),
        "maximum_DFS_tangent_leakage": max(tangent_leakages),
        "maximum_cost_kernel_tangent_residual": max(tangent_kernel_residuals),
    }
    trajectory_gates = {
        "finite_nonconstant_path": trajectory[
            "maximum_trace_distance_from_initial"
        ]
        >= cfg.nonconstant_trace_distance_minimum,
        "zero_accumulated_declared_cost": accumulated_cost
        <= cfg.exact_zero_tolerance,
        "zero_accumulated_dissipator_activity": accumulated_activity
        <= cfg.exact_zero_tolerance,
        "state_path_stays_in_DFS": trajectory["maximum_DFS_state_leakage"]
        <= cfg.leakage_tolerance,
        "path_tangent_stays_in_DFS": trajectory["maximum_DFS_tangent_leakage"]
        <= cfg.leakage_tolerance,
        "path_tangent_stays_in_cost_kernel": trajectory[
            "maximum_cost_kernel_tangent_residual"
        ]
        <= cfg.exact_zero_tolerance,
    }

    positive_initial = (ket(0) + ket(1)) / math.sqrt(2.0)
    positive_rates: list[float] = []
    positive_activities: list[float] = []
    for point in times:
        state = expm(-1.0j * hamiltonian * point) @ positive_initial
        rho = np.outer(state, state.conj())
        positive_rates.append(float(np.real(np.trace(kernel @ rho))))
        positive_activities.append(
            float(np.linalg.norm(dissipator(jump, rho), ord="fro"))
        )
    positive_cost = integrate(positive_rates, times)
    analytic_positive_cost = 2.0 * cfg.true_gamma * cfg.target_duration
    positive_control = {
        "initial_state": "(|00>+|01>)/sqrt(2)",
        "accumulated_declared_jump_cost": positive_cost,
        "analytic_accumulated_jump_cost": analytic_positive_cost,
        "relative_error": abs(positive_cost - analytic_positive_cost)
        / analytic_positive_cost,
        "accumulated_dissipator_activity": integrate(positive_activities, times),
    }
    positive_control_gates = {
        # FIX v6.1: gated on same_meter_minimum_accumulated_cost, which is a
        # threshold on a cost, instead of on the trace-distance threshold.
        "same_meter_cost_positive": positive_cost
        > cfg.same_meter_minimum_accumulated_cost,
        "same_meter_activity_positive": positive_control[
            "accumulated_dissipator_activity"
        ]
        > cfg.same_meter_minimum_accumulated_cost,
        "same_meter_cost_matches_analytic_calibration": positive_control[
            "relative_error"
        ]
        <= cfg.model_closed_form_tolerance,
    }

    contraction_rows: list[dict[str, Any]] = []
    for scale_s in cfg.contraction_scales:
        endpoint = expm(
            -1.0j * scale_s * hamiltonian * cfg.target_duration
        ) @ initial
        endpoint_distance = pure_trace_distance(initial, endpoint)
        maximum_distance = 0.0
        rates: list[float] = []
        for point in times:
            state = expm(-1.0j * scale_s * hamiltonian * point) @ initial
            maximum_distance = max(
                maximum_distance, pure_trace_distance(initial, state)
            )
            rho = np.outer(state, state.conj())
            rates.append(max(0.0, float(np.real(np.trace(kernel @ rho)))))
        contraction_rows.append(
            {
                "scale_s": scale_s,
                "endpoint_trace_distance_from_constant_path": endpoint_distance,
                "maximum_trace_distance_from_initial": maximum_distance,
                "accumulated_declared_jump_cost": integrate(rates, times),
                "nonconstant": maximum_distance
                >= cfg.nonconstant_trace_distance_minimum,
            }
        )
    ordered_by_scale = sorted(contraction_rows, key=lambda row: row["scale_s"])
    endpoint_distances = [
        row["endpoint_trace_distance_from_constant_path"]
        for row in ordered_by_scale
    ]
    monotone = all(
        endpoint_distances[index + 1] + cfg.contraction_monotonic_tolerance
        >= endpoint_distances[index]
        for index in range(len(endpoint_distances) - 1)
    )
    contraction = {
        "family": "H_s = s H, s > 0, fixed duration",
        "rows": contraction_rows,
        "smallest_scale_endpoint_distance": ordered_by_scale[0][
            "endpoint_trace_distance_from_constant_path"
        ],
        "all_members_nonconstant": all(row["nonconstant"] for row in contraction_rows),
        "all_members_zero_declared_cost": all(
            row["accumulated_declared_jump_cost"] <= cfg.exact_zero_tolerance
            for row in contraction_rows
        ),
        "endpoint_distance_monotone_with_scale": monotone,
    }
    contraction_gates = {
        "finite_zero_cost_family": contraction["all_members_zero_declared_cost"],
        "every_positive_scale_path_nonconstant": contraction[
            "all_members_nonconstant"
        ],
        "family_contracts_toward_constant_path": monotone
        and ordered_by_scale[0][
            "endpoint_trace_distance_from_constant_path"
        ]
        < ordered_by_scale[-1][
            "endpoint_trace_distance_from_constant_path"
        ],
    }

    gates = {
        **analytic_gates,
        **trajectory_gates,
        **positive_control_gates,
        **contraction_gates,
    }
    report = {
        "status": (
            "DECLARED_PROTOCOL_INTEGRABLE_NONCONSTANT_ZERO_COST_PATH_SUPPORTED"
            if all(gates.values())
            else "OPERATIONAL_ZERO_MODE_GATE_FAILURE"
        ),
        "object_hierarchy": {
            "operational_primitive": "predeclared protocol Pi",
            "model_local_rate": "j_Pi(rho)=Tr(K0 rho), K0=L0^dagger L0",
            "accumulated_model_cost": "J_Pi[rho]=integral j_Pi(rho_t)dt",
            "finite_channel_witness": "E_ch (audited separately in Layer 1)",
            "abstract_tangent_density_F": "NOT_CONSTRUCTED",
            "universal_Principle_R_bridge": "NOT_ESTABLISHED",
        },
        "analytic_DFS_certificate": analytic,
        "finite_zero_cost_trajectory": trajectory,
        "same_meter_positive_control": positive_control,
        "contraction_family": contraction,
        "gates": gates,
        "falsifiability_note": (
            "These gates are exercised against injected defects in "
            "`operational_negative_control`; v6.0 shipped them with no "
            "mutation able to trip any of them."
        ),
    }
    return report, trajectory_rows, contraction_rows


def liouvillian(hamiltonian: np.ndarray, jumps: list[np.ndarray]) -> np.ndarray:
    """Column-major vectorization: vec(A rho B) = (B^T (x) A) vec(rho)."""
    dimension = hamiltonian.shape[0]
    identity = np.eye(dimension, dtype=complex)
    generator = -1.0j * (
        np.kron(identity, hamiltonian) - np.kron(hamiltonian.T, identity)
    )
    for jump in jumps:
        kernel = jump.conj().T @ jump
        generator += (
            np.kron(jump.conj(), jump)
            - 0.5 * np.kron(identity, kernel)
            - 0.5 * np.kron(kernel.T, identity)
        )
    return generator


def apply_superoperator(superoperator: np.ndarray, rho: np.ndarray) -> np.ndarray:
    dimension = rho.shape[0]
    return (superoperator @ rho.reshape(-1, order="F")).reshape(
        dimension, dimension, order="F"
    )


def channel_superoperator(
    hamiltonian: np.ndarray, jumps: list[np.ndarray], duration: float
) -> np.ndarray:
    return expm(liouvillian(hamiltonian, jumps) * duration)


def choi_state(
    superoperator: np.ndarray, encoding: np.ndarray
) -> tuple[np.ndarray, float]:
    """Normalized Choi state and the Hermiticity residual BEFORE symmetrizing.

    v3.1 symmetrized unconditionally and discarded the residual, which would
    have masked a non-Hermiticity bug.  The residual is now returned and gated.
    """
    logical_dimension = encoding.shape[1]
    physical_dimension = encoding.shape[0]
    size = logical_dimension * physical_dimension
    choi = np.zeros((size, size), dtype=complex)
    for a in range(logical_dimension):
        for b in range(logical_dimension):
            logical_basis = np.zeros(
                (logical_dimension, logical_dimension), dtype=complex
            )
            logical_basis[a, b] = 1.0
            physical_basis = np.outer(encoding[:, a], encoding[:, b].conj())
            choi += np.kron(
                logical_basis, apply_superoperator(superoperator, physical_basis)
            ) / logical_dimension
    residual = float(np.linalg.norm(choi - choi.conj().T, ord="fro"))
    return 0.5 * (choi + choi.conj().T), residual


def trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = 0.5 * ((left - right) + (left - right).conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


def jump_for_delta(
    gamma: float, delta: float, operators: dict[str, np.ndarray]
) -> np.ndarray:
    return math.sqrt(gamma) * (operators["Z1"] + (1.0 + delta) * operators["Z2"])


def encoded_channel_cost(
    gamma: float, delta: float, duration: float, operators: dict[str, np.ndarray]
) -> float:
    hamiltonian = operators["H"]
    ideal, _ = choi_state(
        channel_superoperator(hamiltonian, [], duration), operators["V"]
    )
    noisy, _ = choi_state(
        channel_superoperator(
            hamiltonian, [jump_for_delta(gamma, delta, operators)], duration
        ),
        operators["V"],
    )
    return trace_distance(noisy, ideal)


# ----------------------------------------------------------------------------
# Layer 1
# ----------------------------------------------------------------------------
def cptp_report(
    superoperator: np.ndarray, operators: dict[str, np.ndarray], cfg: Config
) -> dict[str, Any]:
    full_choi, hermiticity_residual = choi_state(superoperator, operators["I4"])
    eigenvalues = np.linalg.eigvalsh(full_choi)
    identity_vector = operators["I4"].reshape(-1, order="F")
    trace_preservation_residual = float(
        np.linalg.norm(superoperator.conj().T @ identity_vector - identity_vector)
    )
    return {
        "choi_minimum_eigenvalue": float(np.min(eigenvalues)),
        "choi_trace": float(np.real(np.trace(full_choi))),
        "choi_hermiticity_residual_before_symmetrization": hermiticity_residual,
        "trace_preservation_residual": trace_preservation_residual,
        "complete_positivity_ok": bool(
            np.min(eigenvalues) >= -cfg.cptp_tolerance
        ),
        "trace_preservation_ok": bool(
            trace_preservation_residual <= cfg.cptp_tolerance
            and abs(float(np.real(np.trace(full_choi))) - 1.0) <= cfg.cptp_tolerance
        ),
        "hermiticity_ok": bool(hermiticity_residual <= cfg.cptp_tolerance),
    }


def representation_regression(
    cfg: Config, operators: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Regression tests on the Lindblad representation freedom.

    NOTE ON INTERPRETATION.  E_ch is defined from the superoperator, so its
    independence of the unravelling is true by construction.  These are code
    regression tests, NOT physical evidence, and are labelled as such.
    """
    hamiltonian = operators["H"]
    duration = cfg.target_duration
    reference_jump = jump_for_delta(cfg.true_gamma, 0.0, operators)
    reference = channel_superoperator(hamiltonian, [reference_jump], duration)
    size = reference.shape[0]

    def deviation(candidate: np.ndarray) -> float:
        return float(np.linalg.norm(reference - candidate, ord="fro") / size)

    # (a) global phase on a single jump -- algebraically trivial
    phase_deviation = deviation(
        channel_superoperator(
            hamiltonian, [np.exp(1.0j * 0.731) * reference_jump], duration
        )
    )

    # (b) split into two identical halves -- algebraically trivial
    halves = [reference_jump / math.sqrt(2.0)] * 2
    split_deviation = deviation(
        channel_superoperator(hamiltonian, halves, duration)
    )

    theta, phi = 0.417, 0.913
    cosine, sine = math.cos(theta), math.sin(theta)
    unitary = np.array(
        [
            [cosine, np.exp(1.0j * phi) * sine],
            [-np.exp(-1.0j * phi) * sine, cosine],
        ],
        dtype=complex,
    )
    unitarity_residual = float(
        np.linalg.norm(unitary.conj().T @ unitary - np.eye(2), ord="fro")
    )

    # (c) mixing identical halves -- this is what v3.1 called its strongest
    #     test; it is degenerate because sum_a |sum_b U_ab|^2 = 2 regardless.
    degenerate_mix = [
        unitary[a, 0] * halves[0] + unitary[a, 1] * halves[1] for a in range(2)
    ]
    degenerate_deviation = deviation(
        channel_superoperator(hamiltonian, degenerate_mix, duration)
    )

    # (d) mixing two LINEARLY INDEPENDENT jumps.
    second_jump = math.sqrt(0.05) * operators["X1"]
    distinct = [reference_jump, second_jump]
    distinct_reference = channel_superoperator(hamiltonian, distinct, duration)
    distinct_mixed = [
        unitary[a, 0] * distinct[0] + unitary[a, 1] * distinct[1] for a in range(2)
    ]
    distinct_deviation = float(
        np.linalg.norm(
            distinct_reference
            - channel_superoperator(hamiltonian, distinct_mixed, duration),
            ord="fro",
        )
        / size
    )

    # (e) inhomogeneous gauge freedom, the representation freedom that
    #     actually has content:  L -> L + c I,  H -> H - (i/2)(c* L - c L^dag).
    shift = 0.37 + 0.21j
    shifted_jump = reference_jump + shift * operators["I4"]
    shifted_hamiltonian = hamiltonian - 0.5j * (
        np.conj(shift) * reference_jump - shift * reference_jump.conj().T
    )
    gauge_deviation = deviation(
        channel_superoperator(shifted_hamiltonian, [shifted_jump], duration)
    )

    tolerance = cfg.representation_tolerance
    return {
        "role": "code_regression_only_not_physical_evidence",
        "single_jump_global_phase_deviation": phase_deviation,
        "split_into_identical_halves_deviation": split_deviation,
        "degenerate_identical_jump_mixing_deviation": degenerate_deviation,
        "distinct_jump_unitary_mixing_deviation": distinct_deviation,
        "inhomogeneous_gauge_shift_deviation": gauge_deviation,
        "mixing_unitarity_residual": unitarity_residual,
        "gates": {
            "trivial_single_jump_phase": phase_deviation <= tolerance,
            "trivial_identical_split": split_deviation <= tolerance,
            "degenerate_identical_mixing": (
                degenerate_deviation <= tolerance
                and unitarity_residual <= tolerance
            ),
            "nondegenerate_distinct_jump_mixing": distinct_deviation <= tolerance,
            "inhomogeneous_gauge_invariance": gauge_deviation <= tolerance,
        },
    }


def compressed_encoded_choi(
    superoperator: np.ndarray, encoding: np.ndarray
) -> np.ndarray:
    """Encoded Choi state written in the 2x2 logical output basis.

    Valid only when the code space is exactly invariant, which is gated
    separately by `code_space_exactly_invariant_no_leakage`.
    """
    choi = np.zeros((4, 4), dtype=complex)
    for a in range(2):
        for b in range(2):
            logical_basis = np.zeros((2, 2), dtype=complex)
            logical_basis[a, b] = 1.0
            output = apply_superoperator(
                superoperator,
                np.outer(encoding[:, a], encoding[:, b].conj()),
            )
            choi += (
                np.kron(logical_basis, encoding.conj().T @ output @ encoding) / 2.0
            )
    return 0.5 * (choi + choi.conj().T)


def logical_reduction_report(
    cfg: Config, operators: dict[str, np.ndarray]
) -> dict[str, Any]:
    """L1.0 -- the encoded channel reduces EXACTLY to a one-qubit channel.

    L_delta acts inside span{|01>,|10>} as sqrt(gamma)*delta*diag(-1,+1) and H
    acts as J*X_L.  Hence the encoded channel is the qubit channel

        H_L = J X_L ,   single jump  sqrt(gamma) delta Z_L ,

    for every gamma, delta and t.  Everything Layer 1 reports downstream --
    the exact zero at delta=0, the absence of leakage, the delta^2 law -- is a
    corollary of this reduction, not an empirical discovery.  The numbers below
    are a witness for an algebraic statement, and should be presented that way.
    """
    logical_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    logical_z = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=complex)
    rows: list[dict[str, Any]] = []
    for gamma, delta, duration in (
        (cfg.true_gamma, 0.0, cfg.target_duration),
        (cfg.true_gamma, 0.06, cfg.target_duration),
        (0.70, -0.30, 2.0),
        (0.05, 1.50, 11.0),
    ):
        physical = compressed_encoded_choi(
            channel_superoperator(
                operators["H"],
                [jump_for_delta(gamma, delta, operators)],
                duration,
            ),
            operators["V"],
        )
        logical, _ = choi_state(
            channel_superoperator(
                cfg.exchange_J * logical_x,
                [math.sqrt(gamma) * delta * logical_z],
                duration,
            ),
            np.eye(2, dtype=complex),
        )
        rows.append(
            {
                "gamma": gamma,
                "delta": delta,
                "duration": duration,
                "frobenius_deviation": float(
                    np.linalg.norm(physical - logical, ord="fro")
                ),
            }
        )
    worst = max(row["frobenius_deviation"] for row in rows)
    return {
        "statement": (
            "encoded channel == qubit channel (H_L = J X_L, jump "
            "sqrt(gamma)*delta*Z_L), exactly, for all gamma/delta/t"
        ),
        "role": "numerical_witness_for_an_algebraic_identity",
        "rows": rows,
        "maximum_frobenius_deviation": worst,
        "gates": {
            "encoded_channel_reduces_to_one_qubit_channel": worst
            <= cfg.representation_tolerance
        },
    }


def scaling_audit(
    cfg: Config, operators: dict[str, np.ndarray]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """L1.4 -- per-slice audit of the symmetry-breaking law E_ch = gamma delta^2 t.

    Each (gamma, t) slice is fitted independently on the delta grid.  The
    certificate reports the full range of fitted exponents, the worst
    coefficient error, and the worst delta <-> -delta parity error, so a
    reviewer can see the spread rather than a single aggregate number.

    Parity is an exact prediction of the L1.0 reduction: the encoded jump is
    sqrt(gamma)*delta*Z_L and the dissipator depends on |sqrt(gamma)*delta|^2,
    so E_ch must be EXACTLY even in delta.  Its tolerance is set at machine
    precision, not as a fitted tolerance.
    """
    rows: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    for gamma in cfg.scaling_gammas:
        for duration in cfg.scaling_durations:
            costs: list[float] = []
            parity_errors: list[float] = []
            for delta in cfg.powerlaw_deltas:
                positive = encoded_channel_cost(gamma, delta, duration, operators)
                negative = encoded_channel_cost(gamma, -delta, duration, operators)
                costs.append(positive)
                parity_errors.append(abs(positive - negative) / positive)
                rows.append(
                    {
                        "gamma": gamma,
                        "duration": duration,
                        "delta": delta,
                        "encoded_cost_positive_delta": positive,
                        "encoded_cost_negative_delta": negative,
                        "parity_relative_error": parity_errors[-1],
                        "cost_over_gamma_delta2_t": positive
                        / (gamma * delta**2 * duration),
                        "data_role": "layer1_scaling_slice",
                    }
                )
            exponent, intercept = np.polyfit(
                np.log(np.array(cfg.powerlaw_deltas)), np.log(np.array(costs)), 1
            )
            coefficient = float(np.exp(intercept))
            expected = gamma * duration
            slices.append(
                {
                    "gamma": gamma,
                    "duration": duration,
                    "fitted_delta_exponent": float(exponent),
                    "fitted_coefficient": coefficient,
                    "expected_coefficient_gamma_times_t": float(expected),
                    "coefficient_relative_error": float(
                        abs(coefficient - expected) / expected
                    ),
                    "maximum_parity_relative_error": float(max(parity_errors)),
                }
            )

    exponents = [entry["fitted_delta_exponent"] for entry in slices]
    coefficient_errors = [
        entry["coefficient_relative_error"] for entry in slices
    ]
    parity = [entry["maximum_parity_relative_error"] for entry in slices]
    minimum_exponent, maximum_exponent = min(exponents), max(exponents)
    worst_coefficient = max(coefficient_errors)
    worst_parity = max(parity)
    worst_index = int(np.argmax(coefficient_errors))

    return {
        "law": "E_ch = gamma * delta^2 * t + O(gamma^2 delta^4 t^2)",
        "number_of_slices": len(slices),
        "slices": slices,
        "minimum_fitted_delta_exponent": minimum_exponent,
        "maximum_fitted_delta_exponent": maximum_exponent,
        "maximum_coefficient_relative_error": worst_coefficient,
        "worst_coefficient_slice": {
            "gamma": slices[worst_index]["gamma"],
            "duration": slices[worst_index]["duration"],
        },
        "maximum_parity_relative_error": worst_parity,
        "note": (
            "The coefficient error grows with gamma*t because the leading-order "
            "law is the first term of a saturating expansion; the trend is "
            "visible slice by slice in `slices` and is not hidden by averaging."
        ),
        "gates": {
            "delta_exponent_within_tolerance_on_every_slice": (
                abs(minimum_exponent - 2.0) <= cfg.scaling_exponent_tolerance
                and abs(maximum_exponent - 2.0) <= cfg.scaling_exponent_tolerance
            ),
            "coefficient_matches_gamma_t_on_every_slice": worst_coefficient
            <= cfg.scaling_coefficient_tolerance,
            "exactly_even_under_delta_sign_flip": worst_parity
            <= cfg.scaling_parity_tolerance,
        },
    }, rows


def layer1_audit(
    cfg: Config, operators: dict[str, np.ndarray]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    hamiltonian = operators["H"]
    encoding = operators["V"]
    projector = encoding @ encoding.conj().T

    # L1.1 duration sweep of the exact encoded zero
    duration_rows = []
    for duration in cfg.duration_sweep:
        duration_rows.append(
            {
                "duration": duration,
                "encoded_cost_at_delta_zero": encoded_channel_cost(
                    cfg.true_gamma, 0.0, duration, operators
                ),
            }
        )
    maximum_zero = max(row["encoded_cost_at_delta_zero"] for row in duration_rows)

    # L1.2 the full physical channel is not ideal
    reference = channel_superoperator(
        hamiltonian,
        [jump_for_delta(cfg.true_gamma, 0.0, operators)],
        cfg.target_duration,
    )
    ideal_full, _ = choi_state(
        channel_superoperator(hamiltonian, [], cfg.target_duration),
        operators["I4"],
    )
    noisy_full, _ = choi_state(reference, operators["I4"])
    full_channel_cost = trace_distance(noisy_full, ideal_full)

    # L1.3 leakage out of the code space
    leakage_rows = []
    for delta in cfg.leakage_deltas:
        superoperator = channel_superoperator(
            hamiltonian,
            [jump_for_delta(cfg.true_gamma, delta, operators)],
            cfg.target_duration,
        )
        worst = 0.0
        for column in range(encoding.shape[1]):
            output = apply_superoperator(
                superoperator,
                np.outer(encoding[:, column], encoding[:, column].conj()),
            )
            worst = max(
                worst,
                abs(
                    float(
                        np.real(np.trace((operators["I4"] - projector) @ output))
                    )
                ),
            )
        leakage_rows.append({"delta": delta, "leakage_population": worst})
    maximum_leakage = max(row["leakage_population"] for row in leakage_rows)

    # L1.0 exact reduction to a one-qubit channel
    reduction = logical_reduction_report(cfg, operators)

    # L1.4 quantitative symmetry-breaking law, audited slice by slice
    powerlaw, powerlaw_rows = scaling_audit(cfg, operators)

    # L1.5 CPTP validity
    cptp = cptp_report(reference, operators, cfg)

    # L1.6 representation regression
    representation = representation_regression(cfg, operators)

    selection_rows = [
        {
            "delta": delta,
            "encoded_choi_trace_distance": encoded_channel_cost(
                cfg.true_gamma, delta, cfg.target_duration, operators
            ),
            "data_role": "layer1_selection",
        }
        for delta in cfg.selection_deltas
    ]

    gates = {
        **reduction["gates"],
        "encoded_zero_exact_over_duration_sweep": maximum_zero
        <= cfg.exact_zero_tolerance,
        "full_physical_channel_not_ideal": full_channel_cost
        >= cfg.full_channel_minimum,
        "code_space_exactly_invariant_no_leakage": maximum_leakage
        <= cfg.leakage_tolerance,
        **{f"scaling_{k}": v for k, v in powerlaw["gates"].items()},
        "cptp_complete_positivity": cptp["complete_positivity_ok"],
        "cptp_trace_preservation": cptp["trace_preservation_ok"],
        "cptp_hermiticity": cptp["hermiticity_ok"],
    }
    # Representation invariance is a property of the superoperator by
    # construction.  These checks verify the CODE, not the physics, and do not
    # vote on declared model support.
    nonvoting = dict(representation["gates"])
    return (
        {
            "status": (
                "CHANNEL_STRUCTURE_SUPPORTED"
                if all(gates.values())
                else "CHANNEL_STRUCTURE_AUDIT_FAILED"
            ),
            "finite_channel_witness_definition": (
                "E_ch = 1/2 ||J(encoded noisy) - J(encoded ideal)||_1"
            ),
            "logical_reduction": reduction,
            "duration_sweep": duration_rows,
            "maximum_encoded_cost_at_delta_zero": maximum_zero,
            "full_physical_channel_cost": full_channel_cost,
            "leakage_sweep": leakage_rows,
            "maximum_leakage_population": maximum_leakage,
            "symmetry_breaking_law": powerlaw,
            "cptp": cptp,
            "representation_regression": representation,
            "gates": gates,
            "nonvoting_regression_checks": nonvoting,
            "interpretation": (
                "The zero is an encoded-channel statement about a leakage-free "
                "qubit channel; it is not a claim that the full physical "
                "channel is noiseless.  Note further that the whole of Layer 1 "
                "follows algebraically from `logical_reduction`; these numbers "
                "are a witness, not a discovery, and should be written up as a "
                "two-line proof with the script cited as verification."
            ),
        },
        selection_rows,
        powerlaw_rows,
    )


# ----------------------------------------------------------------------------
# Layer 2 -- observables, exact simulation, and closed forms
# ----------------------------------------------------------------------------
def simulate_parity_visibility(
    gamma: float,
    delta: float,
    duration: float,
    operators: dict[str, np.ndarray],
    extra_jumps: tuple[np.ndarray, ...] = (),
) -> float:
    """|00>,|11> coherence under the FULL Liouvillian, Hamiltonian ON.

    Both |00> and |11> are annihilated by XX + YY, so this pair is an exact
    H-invariant subspace and the closed form below holds with H switched on.
    """
    state = np.zeros(4, dtype=complex)
    state[0] = state[3] = 1.0 / math.sqrt(2.0)
    output = apply_superoperator(
        channel_superoperator(
            operators["H"],
            [jump_for_delta(gamma, delta, operators)] + list(extra_jumps),
            duration,
        ),
        np.outer(state, state.conj()),
    )
    return float(2.0 * abs(output[0, 3]))


def closed_form_parity_visibility(gamma: float, delta: float, duration: float) -> float:
    return float(math.exp(-2.0 * gamma * (2.0 + delta) ** 2 * duration))


def simulate_logical_x_visibility(
    gamma: float,
    delta: float,
    duration: float,
    operators: dict[str, np.ndarray],
    extra_jumps: tuple[np.ndarray, ...] = (),
) -> float:
    """Logical X_L expectation inside the code space, Hamiltonian ON.

    (|01>+|10>)/sqrt(2) is an eigenvector of H, and X_L commutes with the
    logical Hamiltonian J X_L, so <X_L> decays purely at rate 2 gamma delta^2.
    """
    state = np.zeros(4, dtype=complex)
    state[1] = state[2] = 1.0 / math.sqrt(2.0)
    output = apply_superoperator(
        channel_superoperator(
            operators["H"],
            [jump_for_delta(gamma, delta, operators)] + list(extra_jumps),
            duration,
        ),
        np.outer(state, state.conj()),
    )
    return float(np.real(output[1, 2] + output[2, 1]))


def closed_form_logical_x_visibility(
    gamma: float, delta: float, duration: float
) -> float:
    return float(math.exp(-2.0 * gamma * delta**2 * duration))


def model_validity_report(
    cfg: Config, operators: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Gate L2.0: the estimator's closed forms must match exact evolution."""
    rows: list[dict[str, Any]] = []
    for delta, times in (
        (cfg.calibration_delta, cfg.calibration_times + cfg.heldout_times),
    ):
        for duration in times:
            rows.append(
                {
                    "observable": "parity_00_11",
                    "delta": delta,
                    "duration": duration,
                    "simulated": simulate_parity_visibility(
                        cfg.true_gamma, delta, duration, operators
                    ),
                    "closed_form": closed_form_parity_visibility(
                        cfg.true_gamma, delta, duration
                    ),
                }
            )
    for delta, times in cfg.transfer_schedule:
        for duration in times:
            rows.append(
                {
                    "observable": "logical_x",
                    "delta": delta,
                    "duration": duration,
                    "simulated": simulate_logical_x_visibility(
                        cfg.true_gamma, delta, duration, operators
                    ),
                    "closed_form": closed_form_logical_x_visibility(
                        cfg.true_gamma, delta, duration
                    ),
                }
            )
    for row in rows:
        row["absolute_deviation"] = abs(row["simulated"] - row["closed_form"])
    worst = max(row["absolute_deviation"] for row in rows)

    # Counter-check: the v3.1 formula on the pair it implicitly used.
    v31_state = np.zeros(4, dtype=complex)
    v31_state[0] = v31_state[1] = 1.0 / math.sqrt(2.0)
    v31_output = apply_superoperator(
        channel_superoperator(
            operators["H"],
            [jump_for_delta(cfg.true_gamma, 0.0, operators)],
            0.90,
        ),
        np.outer(v31_state, v31_state.conj()),
    )
    v31_simulated = float(2.0 * abs(v31_output[0, 1]))
    v31_formula = math.exp(-2.0 * cfg.true_gamma * 0.90)

    return {
        "rows": rows,
        "maximum_absolute_deviation": worst,
        "v31_regression_witness": {
            "note": (
                "v3.1 used V = exp(-2 gamma t), which corresponds to the "
                "|00>,|01> coherence.  |01> is not an eigenvector of H, so the "
                "formula is inconsistent with the Layer 1 Liouvillian."
            ),
            "duration": 0.90,
            "simulated_with_hamiltonian_on": v31_simulated,
            "v31_closed_form": float(v31_formula),
            "absolute_deviation": abs(v31_simulated - v31_formula),
        },
        "gates": {
            "closed_forms_match_exact_liouvillian": worst
            <= cfg.model_closed_form_tolerance,
            "v31_formula_demonstrably_inconsistent": abs(
                v31_simulated - v31_formula
            )
            > 1.0e-3,
        },
    }


# ----------------------------------------------------------------------------
# Layer 2 -- estimation
# ----------------------------------------------------------------------------
def draw_counts(
    visibility: float, shots: int, rng: np.random.Generator
) -> tuple[int, float]:
    probability_plus = float(np.clip(0.5 * (1.0 + visibility), 0.0, 1.0))
    plus = int(rng.binomial(shots, probability_plus))
    return plus, 2.0 * plus / shots - 1.0


def generate_rows(
    gamma: float,
    delta: float,
    times: tuple[float, ...],
    shots: int,
    rng: np.random.Generator,
    role: str,
    simulator: Callable[..., float],
    operators: dict[str, np.ndarray],
    extra_jumps: tuple[np.ndarray, ...] = (),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for duration in times:
        visibility = simulator(gamma, delta, duration, operators, extra_jumps)
        plus, measured = draw_counts(visibility, shots, rng)
        rows.append(
            {
                "duration": duration,
                "delta": delta,
                "shots": shots,
                "plus_counts": plus,
                "minus_counts": shots - plus,
                "exact_visibility": visibility,
                "measured_visibility": measured,
                "shot_noise_sigma": math.sqrt(
                    max(1.0 - visibility**2, 0.0) / shots
                ),
                "data_role": role,
            }
        )
    return rows


def fit_gamma_mle(
    rows: list[dict[str, Any]], delta: float, cfg: Config
) -> tuple[float, float, dict[str, Any]]:
    def negative_log_likelihood(gamma: float) -> float:
        total = 0.0
        for row in rows:
            visibility = closed_form_parity_visibility(
                gamma, delta, row["duration"]
            )
            probability = float(
                np.clip(0.5 * (1.0 + visibility), 1.0e-12, 1.0 - 1.0e-12)
            )
            total -= row["plus_counts"] * math.log(probability)
            total -= row["minus_counts"] * math.log(1.0 - probability)
        return total

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=cfg.likelihood_bounds,
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"gamma MLE failed: {result.message}")
    gamma_hat = float(result.x)

    lower, upper = cfg.likelihood_bounds
    span = upper - lower
    interior = bool(
        gamma_hat > lower + cfg.boundary_margin * span
        and gamma_hat < upper - cfg.boundary_margin * span
    )

    # FIX (v3.1 misnomer): this is the EXPECTED Fisher information evaluated at
    # gamma_hat, computed from the model, not the observed information.
    expected_fisher = 0.0
    for row in rows:
        duration = row["duration"]
        visibility = closed_form_parity_visibility(gamma_hat, delta, duration)
        probability = 0.5 * (1.0 + visibility)
        derivative = -(2.0 + delta) ** 2 * duration * visibility  # d p / d gamma
        expected_fisher += (
            row["shots"]
            * derivative**2
            / max(probability * (1.0 - probability), 1.0e-15)
        )
    standard_error = float(1.0 / math.sqrt(expected_fisher))
    return (
        gamma_hat,
        standard_error,
        {
            "optimizer_success": bool(result.success),
            "optimum_is_interior": interior,
            "negative_log_likelihood": float(result.fun),
            "expected_fisher_information_at_gamma_hat": float(expected_fisher),
            "asymptotic_standard_error": standard_error,
            "asymptotic_95_percent_interval": [
                gamma_hat - 1.96 * standard_error,
                gamma_hat + 1.96 * standard_error,
            ],
        },
    )


def coverage_sweep(
    cfg: Config, operators: dict[str, np.ndarray], rng: np.random.Generator
) -> dict[str, Any]:
    """FIX (v3.1): interval coverage was a single Bernoulli draw."""
    exact = {
        duration: simulate_parity_visibility(
            cfg.true_gamma, cfg.calibration_delta, duration, operators
        )
        for duration in cfg.calibration_times
    }
    covered = 0
    relative_errors: list[float] = []
    pulls: list[float] = []
    for _ in range(cfg.coverage_replicates):
        rows = []
        for duration, visibility in exact.items():
            plus, _ = draw_counts(visibility, cfg.shots_per_time, rng)
            rows.append(
                {
                    "duration": duration,
                    "shots": cfg.shots_per_time,
                    "plus_counts": plus,
                    "minus_counts": cfg.shots_per_time - plus,
                }
            )
        gamma_hat, standard_error, diagnostics = fit_gamma_mle(
            rows, cfg.calibration_delta, cfg
        )
        low, high = diagnostics["asymptotic_95_percent_interval"]
        covered += int(low <= cfg.true_gamma <= high)
        relative_errors.append(abs(gamma_hat - cfg.true_gamma) / cfg.true_gamma)
        pulls.append((gamma_hat - cfg.true_gamma) / standard_error)

    n = cfg.coverage_replicates
    empirical = covered / n
    sigma = math.sqrt(cfg.coverage_nominal * (1 - cfg.coverage_nominal) / n)
    pull_mean = float(np.mean(pulls))
    pull_std = float(np.std(pulls, ddof=1))
    return {
        "replicates": n,
        "empirical_coverage": empirical,
        "nominal_coverage": cfg.coverage_nominal,
        "coverage_binomial_sigma": sigma,
        "coverage_deviation_in_sigma": abs(empirical - cfg.coverage_nominal) / sigma,
        "median_relative_error": float(np.median(relative_errors)),
        "pull_mean": pull_mean,
        "pull_standard_deviation": pull_std,
        "gates": {
            "interval_coverage_consistent_with_nominal": abs(
                empirical - cfg.coverage_nominal
            )
            <= cfg.coverage_sigma_allowance * sigma,
            "pull_distribution_unbiased": abs(pull_mean)
            <= cfg.coverage_sigma_allowance / math.sqrt(n),
            "pull_distribution_unit_width": abs(pull_std - 1.0) <= 0.15,
        },
    }


def residual_sigma_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    normalized = [
        row["visibility_residual"] / max(row["shot_noise_sigma"], 1e-15)
        for row in rows
    ]
    return {
        "root_mean_square_residual": float(
            np.sqrt(np.mean([row["visibility_residual"] ** 2 for row in rows]))
        ),
        "maximum_absolute_pull": float(np.max(np.abs(normalized))),
        "root_mean_square_pull": float(np.sqrt(np.mean(np.square(normalized)))),
    }


def layer2_calibration(
    cfg: Config, operators: dict[str, np.ndarray], output: Path
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    # FIX (v3.1): independent RNG streams instead of one shared sequence, so
    # changing one experiment does not silently reshuffle another.
    streams = np.random.default_rng(cfg.master_seed).spawn(4)
    calibration_rng, heldout_rng, transfer_rng, coverage_rng = streams

    model = model_validity_report(cfg, operators)

    calibration_rows = generate_rows(
        cfg.true_gamma,
        cfg.calibration_delta,
        cfg.calibration_times,
        cfg.shots_per_time,
        calibration_rng,
        "calibration_fit",
        simulate_parity_visibility,
        operators,
    )
    gamma_hat, standard_error, fit_diagnostics = fit_gamma_mle(
        calibration_rows, cfg.calibration_delta, cfg
    )
    gamma_relative_error = abs(gamma_hat - cfg.true_gamma) / cfg.true_gamma

    # ---- freeze: EVERY held-out prediction is built and committed BEFORE
    #      any held-out datum is drawn ------------------------------------
    #
    # FIX v6.2 (item 2).  v6.1 committed only the E_ch extrapolation.  The
    # same-observable Ramsey held-out points and the cross-observable X_L
    # transfer points were predicted correctly but relied on execution order
    # alone, with nothing on disk to check them against.  All three families
    # now go into a single commitment file, the counts are drawn afterwards,
    # and `every_scored_prediction_matches_the_commitment_on_disk` re-reads
    # that file and compares it against every value actually scored.
    heldout_predictions = [
        {
            "family": "heldout_same_observable",
            "observable": "parity_00_11",
            "delta": cfg.calibration_delta,
            "duration": duration,
            "predicted_visibility": closed_form_parity_visibility(
                gamma_hat, cfg.calibration_delta, duration
            ),
        }
        for duration in cfg.heldout_times
    ]
    transfer_predictions = [
        {
            "family": "cross_experiment_transfer",
            "observable": "logical_x",
            "delta": delta,
            "duration": duration,
            "predicted_visibility": closed_form_logical_x_visibility(
                gamma_hat, delta, duration
            ),
        }
        for delta, times in cfg.transfer_schedule
        for duration in times
    ]
    # E_ch extrapolation over the control grid.  This is NOT an independent
    # test of gamma_hat; see `extrapolation_is_not_an_independent_test`.
    prediction_rows = [
        {
            "delta": delta,
            "predicted_encoded_cost_closed_form": 0.5
            * (
                1.0
                - math.exp(
                    -2.0 * gamma_hat * delta**2 * cfg.target_duration
                )
            ),
            "data_role": "frozen_prediction",
        }
        for delta in cfg.heldout_deltas
    ]
    # FIX v6.1 (B1): deep-copied, hashed, and re-verified from disk.  v6.0
    # hashed an object that aliased prediction_rows and was mutated below.
    frozen_hash, frozen_verified = freeze_and_commit(
        output,
        "frozen_predictions",
        {
            "gamma_hat": gamma_hat,
            "target_duration": cfg.target_duration,
            "heldout_same_observable": heldout_predictions,
            "cross_experiment_transfer": transfer_predictions,
            "encoded_cost_extrapolation": prediction_rows,
        },
    )

    # ---- data drawn only after the commitment ---------------------------
    heldout_rows = generate_rows(
        cfg.true_gamma,
        cfg.calibration_delta,
        cfg.heldout_times,
        cfg.shots_per_time,
        heldout_rng,
        "heldout_parity",
        simulate_parity_visibility,
        operators,
    )
    for row, prediction in zip(heldout_rows, heldout_predictions):
        if row["duration"] != prediction["duration"]:
            raise RuntimeError(
                "held-out draw order does not match the commitment order"
            )
        row["predicted_visibility"] = prediction["predicted_visibility"]
        row["visibility_residual"] = (
            row["measured_visibility"] - row["predicted_visibility"]
        )
    heldout_summary = residual_sigma_summary(heldout_rows)

    # ---- cross-experiment transfer: different observable, different
    #      subspace, unseen delta, rate slower by 16/delta^2 ---------------
    committed_transfer = {
        (entry["delta"], entry["duration"]): entry["predicted_visibility"]
        for entry in transfer_predictions
    }
    transfer_rows: list[dict[str, Any]] = []
    for delta, times in cfg.transfer_schedule:
        block = generate_rows(
            cfg.true_gamma,
            delta,
            times,
            cfg.shots_per_time,
            transfer_rng,
            "transfer_logical_x",
            simulate_logical_x_visibility,
            operators,
        )
        for row in block:
            key = (delta, row["duration"])
            if key not in committed_transfer:
                raise RuntimeError(f"transfer point {key} was never committed")
            row["predicted_visibility"] = committed_transfer[key]
            row["visibility_residual"] = (
                row["measured_visibility"] - row["predicted_visibility"]
            )
            row["rate_ratio_versus_calibration"] = (
                (2.0 + cfg.calibration_delta) ** 2 / delta**2
            )
        transfer_rows.extend(block)
    transfer_summary = residual_sigma_summary(transfer_rows)

    # ---- truth revealed only now ----------------------------------------
    extrapolation_errors: list[float] = []
    model_only_errors: list[float] = []
    for row in prediction_rows:
        truth = encoded_channel_cost(
            cfg.true_gamma, row["delta"], cfg.target_duration, operators
        )
        model_at_truth = 0.5 * (
            1.0
            - math.exp(
                -2.0 * cfg.true_gamma * row["delta"] ** 2 * cfg.target_duration
            )
        )
        row["truth_exact_liouvillian"] = truth
        row["closed_form_at_true_gamma"] = model_at_truth
        row["relative_error_total"] = (
            abs(row["predicted_encoded_cost_closed_form"] - truth) / truth
        )
        row["relative_error_model_only"] = abs(model_at_truth - truth) / truth
        extrapolation_errors.append(row["relative_error_total"])
        model_only_errors.append(row["relative_error_model_only"])

    # honest sensitivity diagnostic: d ln E_ch / d ln gamma at the operating
    # point.  If this is ~1 the extrapolation gate carries no information
    # beyond the gamma gate, and the certificate says so explicitly.
    probe_delta = 0.06
    base = encoded_channel_cost(
        cfg.true_gamma, probe_delta, cfg.target_duration, operators
    )
    bumped = encoded_channel_cost(
        cfg.true_gamma * 1.001, probe_delta, cfg.target_duration, operators
    )
    log_sensitivity = float((bumped - base) / base / 0.001)

    # Re-read the commitment from disk and confirm that every value actually
    # scored above came from it.  This is the falsifiable form of "the
    # predictions were frozen": a prediction recomputed after the fact, or a
    # held-out point never committed, makes this gate fail.
    committed = json.loads(
        (output / "frozen_predictions.json").read_text(encoding="utf-8")
    )
    committed_visibility = {
        (entry["observable"], entry["delta"], entry["duration"]): entry[
            "predicted_visibility"
        ]
        for family in ("heldout_same_observable", "cross_experiment_transfer")
        for entry in committed[family]
    }
    committed_encoded_cost = {
        entry["delta"]: entry["predicted_encoded_cost_closed_form"]
        for entry in committed["encoded_cost_extrapolation"]
    }
    scored_visibility = [("parity_00_11", row) for row in heldout_rows] + [
        ("logical_x", row) for row in transfer_rows
    ]

    # FIX v6.2.1: coverage is established BEFORE any deviation is computed.
    # v6.2 evaluated max(...) first, so a point that was never committed raised
    # a KeyError and crashed the run instead of failing the gate.  A missing
    # commitment is a gate failure, not a traceback.
    missing_visibility = [
        [observable, row["delta"], row["duration"]]
        for observable, row in scored_visibility
        if (observable, row["delta"], row["duration"]) not in committed_visibility
    ]
    missing_encoded_cost = [
        row["delta"]
        for row in prediction_rows
        if row["delta"] not in committed_encoded_cost
    ]

    # FIX v6.2.1: the E_ch payload is now compared POINT BY POINT against the
    # commitment, not merely counted.  v6.2 recorded only
    # committed_encoded_cost_predictions, so the values that actually entered
    # the error calculation were never checked against disk and the gate name
    # `every_scored_prediction_...` was stronger than what was verified.
    maximum_visibility_deviation = max(
        (
            abs(
                row["predicted_visibility"]
                - committed_visibility[
                    (observable, row["delta"], row["duration"])
                ]
            )
            for observable, row in scored_visibility
            if (observable, row["delta"], row["duration"]) in committed_visibility
        ),
        default=0.0,
    )
    maximum_encoded_cost_deviation = max(
        (
            abs(
                row["predicted_encoded_cost_closed_form"]
                - committed_encoded_cost[row["delta"]]
            )
            for row in prediction_rows
            if row["delta"] in committed_encoded_cost
        ),
        default=0.0,
    )

    commitment_coverage = {
        "committed_visibility_predictions": len(committed_visibility),
        "scored_visibility_predictions": len(scored_visibility),
        "committed_encoded_cost_predictions": len(committed_encoded_cost),
        "scored_encoded_cost_predictions": len(prediction_rows),
        "missing_visibility_points": missing_visibility,
        "missing_encoded_cost_points": missing_encoded_cost,
        "every_scored_point_was_committed": (
            not missing_visibility and not missing_encoded_cost
        ),
        "maximum_visibility_commitment_deviation": maximum_visibility_deviation,
        "maximum_encoded_cost_commitment_deviation": (
            maximum_encoded_cost_deviation
        ),
        "note": (
            "Deviations are computed only over points that ARE present in the "
            "commitment; absence is reported separately by "
            "every_scored_point_was_committed, and the gate requires both."
        ),
    }

    gates = {
        **model["gates"],
        "mle_converged": fit_diagnostics["optimizer_success"],
        "mle_optimum_interior": fit_diagnostics["optimum_is_interior"],
        "gamma_recovered_from_calibration_only": gamma_relative_error
        <= cfg.gamma_relative_error_tolerance,
        "heldout_residuals_within_shot_noise": heldout_summary[
            "maximum_absolute_pull"
        ]
        <= cfg.residual_sigma_tolerance,
        "cross_experiment_transfer_within_shot_noise": transfer_summary[
            "maximum_absolute_pull"
        ]
        <= cfg.transfer_sigma_tolerance,
        "ech_extrapolation_within_tolerance": max(extrapolation_errors)
        <= cfg.ech_relative_error_tolerance,
        "ech_closed_form_model_error_small": max(model_only_errors)
        <= cfg.ech_model_relative_error_tolerance,
        "frozen_prediction_receipt_verifies_from_disk": frozen_verified,
        "every_scored_prediction_matches_the_commitment_on_disk": (
            commitment_coverage["every_scored_point_was_committed"]
            and commitment_coverage["maximum_visibility_commitment_deviation"]
            == 0.0
            and commitment_coverage["maximum_encoded_cost_commitment_deviation"]
            == 0.0
            and commitment_coverage["scored_visibility_predictions"]
            == commitment_coverage["committed_visibility_predictions"]
            and commitment_coverage["scored_encoded_cost_predictions"]
            == commitment_coverage["committed_encoded_cost_predictions"]
        ),
    }

    coverage = coverage_sweep(cfg, operators, coverage_rng)
    gates.update({f"coverage_{k}": v for k, v in coverage["gates"].items()})

    return (
        {
            "status": (
                "CROSS_EXPERIMENT_CALIBRATION_SUPPORTED"
                if all(gates.values())
                else "CALIBRATION_AUDIT_FAILED"
            ),
            "protocol": (
                "Counts are drawn from exact Liouvillian evolution.  gamma is "
                "fit by binomial MLE on the H-invariant |00>,|11> coherence at "
                "delta=0, then frozen and used to predict (a) disjoint times of "
                "the same observable, (b) the logical X_L decay inside the code "
                "space at unseen delta, and (c) the encoded channel cost."
            ),
            "model_validity": model,
            "gamma_true_hidden_from_fit": cfg.true_gamma,
            "gamma_hat": gamma_hat,
            "gamma_relative_error": gamma_relative_error,
            "gamma_standard_error": standard_error,
            "fit_diagnostics": fit_diagnostics,
            "heldout_residual_summary": heldout_summary,
            "cross_experiment_transfer_summary": transfer_summary,
            "frozen_prediction_sha256": frozen_hash,
            "frozen_prediction_receipt_verified": frozen_verified,
            "commitment_coverage": commitment_coverage,
            "ech_extrapolation": {
                "maximum_relative_error_total": max(extrapolation_errors),
                "maximum_relative_error_model_only": max(model_only_errors),
                "d_log_Ech_d_log_gamma": log_sensitivity,
                "extrapolation_is_not_an_independent_test": (
                    "d ln E_ch / d ln gamma is ~1, so the relative error of "
                    "this extrapolation is pinned to the relative error of "
                    "gamma_hat.  v3.1 set a LOOSER tolerance here (0.06) than "
                    "on gamma (0.03), which made the gate unfailable.  The "
                    "tolerance is now equal to the gamma tolerance, and the "
                    "only genuinely new information is "
                    "maximum_relative_error_model_only, which isolates the "
                    "closed-form approximation from the estimation error."
                ),
            },
            "coverage_sweep": coverage,
            "gates": gates,
            "boundary": (
                "Software-enforced separation over synthetic counts.  Not "
                "independent laboratory calibration.  delta is a known control "
                "parameter; only gamma is inferred."
            ),
        },
        {
            "calibration": calibration_rows,
            "heldout": heldout_rows,
            "transfer": transfer_rows,
            "predictions": prediction_rows,
        },
    )


# ----------------------------------------------------------------------------
# Layer 2B -- joint identification of (gamma, delta0)
# ----------------------------------------------------------------------------
def design_times(cfg: Config, offset: float) -> tuple[float, ...]:
    """Time grid built from the STATED PRIOR, never from the truth."""
    rate = 2.0 * cfg.design_prior_gamma * (
        2.0 + cfg.design_prior_delta0 + offset
    ) ** 2
    if not (rate > 0.0):
        raise ValueError(
            f"control offset {offset} cancels 2+design_prior_delta0; the "
            "predicted parity decay rate is zero and no time grid exists."
        )
    return tuple(float(u / rate) for u in cfg.design_time_units)


def transfer_design_times(cfg: Config, offset: float) -> tuple[float, ...]:
    rate = 2.0 * cfg.design_prior_gamma * (
        cfg.design_prior_delta0 + offset
    ) ** 2
    # FIX v6.1: an offset that cancels the prior imbalance gave a bare
    # ZeroDivisionError rather than a diagnosable failure.
    if not (rate > 0.0):
        raise ValueError(
            f"transfer offset {offset} cancels design_prior_delta0; the "
            "predicted X_L decay rate is zero and no transfer grid exists."
        )
    return tuple(float(u / rate) for u in cfg.joint_transfer_time_units)


def joint_parity_visibility(
    gamma: float, delta0: float, offset: float, duration: float
) -> float:
    return float(
        math.exp(-2.0 * gamma * (2.0 + delta0 + offset) ** 2 * duration)
    )


def joint_logical_x_visibility(
    gamma: float, delta0: float, offset: float, duration: float
) -> float:
    return float(math.exp(-2.0 * gamma * (delta0 + offset) ** 2 * duration))


def joint_fisher(
    gamma: float, delta0: float, rows: list[dict[str, Any]]
) -> np.ndarray:
    """Expected 2x2 Fisher information for (gamma, delta0)."""
    information = np.zeros((2, 2))
    for row in rows:
        offset, duration = row["offset"], row["duration"]
        scale = 2.0 + delta0 + offset
        visibility = joint_parity_visibility(gamma, delta0, offset, duration)
        probability = 0.5 * (1.0 + visibility)
        jacobian = np.array(
            [
                -(scale**2) * duration * visibility,
                -2.0 * gamma * scale * duration * visibility,
            ]
        )
        information += (
            row["shots"]
            * np.outer(jacobian, jacobian)
            / max(probability * (1.0 - probability), 1.0e-15)
        )
    return information


def fit_joint_mle(
    rows: list[dict[str, Any]], cfg: Config
) -> tuple[float, float, np.ndarray, dict[str, Any]]:
    from scipy.optimize import minimize

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        gamma, delta0 = float(parameters[0]), float(parameters[1])
        if gamma <= 0.0 or not math.isfinite(gamma):
            return 1.0e18
        total = 0.0
        for row in rows:
            visibility = joint_parity_visibility(
                gamma, delta0, row["offset"], row["duration"]
            )
            probability = float(
                np.clip(0.5 * (1.0 + visibility), 1.0e-12, 1.0 - 1.0e-12)
            )
            total -= row["plus_counts"] * math.log(probability)
            total -= row["minus_counts"] * math.log(1.0 - probability)
        return total

    result = minimize(
        negative_log_likelihood,
        np.array([cfg.design_prior_gamma, cfg.design_prior_delta0]),
        method="Nelder-Mead",
        options={
            "xatol": 1.0e-12,
            "fatol": 1.0e-9,
            "maxiter": 40_000,
            "maxfev": 40_000,
        },
    )
    if not result.success:
        raise RuntimeError(f"joint MLE failed: {result.message}")
    gamma_hat, delta0_hat = float(result.x[0]), float(result.x[1])
    information = joint_fisher(gamma_hat, delta0_hat, rows)
    covariance = np.linalg.inv(information)
    return (
        gamma_hat,
        delta0_hat,
        covariance,
        {
            "optimizer_success": bool(result.success),
            "negative_log_likelihood": float(result.fun),
            "expected_fisher_eigenvalues": np.linalg.eigvalsh(information).tolist(),
            "standard_errors": [
                float(math.sqrt(covariance[0, 0])),
                float(math.sqrt(covariance[1, 1])),
            ],
            "correlation": float(
                covariance[0, 1]
                / math.sqrt(covariance[0, 0] * covariance[1, 1])
            ),
        },
    )


def identifiability_witness(
    cfg: Config, gamma: float, delta0: float, evaluation_point: str
) -> dict[str, Any]:
    """A single control setting cannot identify (gamma, delta0); three can.

    With one setting the model depends on the pair only through the product
    gamma*(2+delta0+offset)^2, so the Fisher matrix is exactly rank one.  This
    is the reason the v4 protocol had to treat delta as known.

    FIX v6.1 (B3).  v6.0 evaluated this at cfg.true_gamma and cfg.true_delta0
    while its gates VOTED.  An experiment design may not read the truth, so the
    voting instance is now evaluated at the stated design prior; a second,
    explicitly non-voting instance is reported at the fitted values as the
    achieved information.

    FIX v6.1 (B4).  v6.0 computed single[0]/single[-1], which is NEGATIVE
    (observed -1.090e-17) for a numerically rank-one matrix, so the gate
    `ratio <= 1e-6` passed on SIGN rather than on magnitude, and would also have
    passed for a genuinely indefinite matrix.  The ratio now uses magnitudes.
    """
    def rows_for(offsets: tuple[float, ...]) -> list[dict[str, Any]]:
        return [
            {
                "offset": offset,
                "duration": duration,
                "shots": cfg.joint_shots_per_time,
            }
            for offset in offsets
            for duration in design_times(cfg, offset)
        ]

    single = np.linalg.eigvalsh(joint_fisher(gamma, delta0, rows_for((0.0,))))
    multiple = np.linalg.eigvalsh(
        joint_fisher(gamma, delta0, rows_for(cfg.control_offsets))
    )
    largest = float(abs(single[-1]))
    single_ratio = float(abs(single[0]) / largest) if largest > 0.0 else float("inf")
    return {
        "evaluation_point": evaluation_point,
        "evaluated_at_gamma": float(gamma),
        "evaluated_at_delta0": float(delta0),
        "single_setting_fisher_eigenvalues": single.tolist(),
        "single_setting_eigenvalue_ratio": single_ratio,
        "multi_setting_fisher_eigenvalues": multiple.tolist(),
        "multi_setting_minimum_eigenvalue": float(multiple[0]),
        "gates": {
            "single_control_setting_is_rank_deficient": single_ratio
            <= cfg.identifiability_rank_deficiency_maximum,
            "multiple_control_settings_identify_both": float(multiple[0])
            >= cfg.identifiability_minimum_eigenvalue,
        },
    }


def layer2b_joint_identification(
    cfg: Config, operators: dict[str, np.ndarray], output: Path
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    streams = np.random.default_rng(cfg.master_seed + 77).spawn(3)
    calibration_rng, transfer_rng, coverage_rng = streams

    # Voting instance: design-time, evaluated at the STATED PRIOR only.
    identifiability = identifiability_witness(
        cfg,
        cfg.design_prior_gamma,
        cfg.design_prior_delta0,
        "design_prior_pre_data",
    )

    calibration_rows: list[dict[str, Any]] = []
    for offset in cfg.control_offsets:
        for duration in design_times(cfg, offset):
            exact = simulate_parity_visibility(
                cfg.true_gamma, cfg.true_delta0 + offset, duration, operators
            )
            plus, measured = draw_counts(
                exact, cfg.joint_shots_per_time, calibration_rng
            )
            calibration_rows.append(
                {
                    "offset": offset,
                    "duration": duration,
                    "shots": cfg.joint_shots_per_time,
                    "plus_counts": plus,
                    "minus_counts": cfg.joint_shots_per_time - plus,
                    "exact_visibility": exact,
                    "closed_form_visibility": joint_parity_visibility(
                        cfg.true_gamma, cfg.true_delta0, offset, duration
                    ),
                    "measured_visibility": measured,
                    "data_role": "joint_calibration",
                }
            )
    model_deviation = max(
        abs(row["exact_visibility"] - row["closed_form_visibility"])
        for row in calibration_rows
    )

    gamma_hat, delta0_hat, covariance, diagnostics = fit_joint_mle(
        calibration_rows, cfg
    )
    gamma_relative_error = abs(gamma_hat - cfg.true_gamma) / cfg.true_gamma
    delta0_absolute_error = abs(delta0_hat - cfg.true_delta0)

    # ---- frozen two-parameter transfer prediction ------------------------
    # The predicted rate is 2*gamma*(delta0+offset)^2, so it needs BOTH
    # estimates.  Uncertainty is propagated by the delta method and combined
    # with shot noise, instead of comparing against shot noise alone.
    #
    # FIX v6.1 (B2).  v6.0 drew the held-out data, computed residuals and
    # pulls, and only THEN wrote "frozen_joint_predictions.json" -- unhashed.
    # That file recorded a comparison already made.  The predictions are now
    # committed and hashed FIRST; the transfer data is drawn afterwards.
    predicted_rows: list[dict[str, Any]] = []
    for offset in cfg.joint_transfer_offsets:
        for duration in transfer_design_times(cfg, offset):
            predicted_rows.append(
                {
                    "offset": offset,
                    "duration": duration,
                    "predicted_visibility": joint_logical_x_visibility(
                        gamma_hat, delta0_hat, offset, duration
                    ),
                }
            )
    joint_frozen_hash, joint_frozen_verified = freeze_and_commit(
        output,
        "frozen_joint_predictions",
        {
            "gamma_hat": gamma_hat,
            "delta0_hat": delta0_hat,
            "covariance": covariance.tolist(),
            "predictions": predicted_rows,
        },
    )

    transfer_rows: list[dict[str, Any]] = []
    for entry in predicted_rows:
        offset = entry["offset"]
        duration = entry["duration"]
        predicted = entry["predicted_visibility"]
        exact = simulate_logical_x_visibility(
            cfg.true_gamma, cfg.true_delta0 + offset, duration, operators
        )
        plus, measured = draw_counts(exact, cfg.shots_per_time, transfer_rng)
        total = delta0_hat + offset
        jacobian = np.array(
            [
                -2.0 * total**2 * duration * predicted,
                -4.0 * gamma_hat * total * duration * predicted,
            ]
        )
        parameter_variance = float(jacobian @ covariance @ jacobian)
        shot_variance = max(1.0 - exact**2, 0.0) / cfg.shots_per_time
        sigma = math.sqrt(shot_variance + parameter_variance)
        transfer_rows.append(
            {
                "offset": offset,
                "total_delta": total,
                "duration": duration,
                "shots": cfg.shots_per_time,
                "exact_visibility": exact,
                "measured_visibility": measured,
                "predicted_visibility": predicted,
                "visibility_residual": measured - predicted,
                "shot_noise_sigma": math.sqrt(shot_variance),
                "parameter_propagation_sigma": math.sqrt(parameter_variance),
                "combined_sigma": sigma,
                "pull": (measured - predicted) / sigma,
                "data_role": "joint_transfer",
            }
        )
    worst_pull = max(abs(row["pull"]) for row in transfer_rows)

    # ---- re-read the joint commitment and compare it point by point ------
    #
    # FIX v6.2.2.  v6.2.1 gave Layer 2 a point-by-point check against its
    # commitment but left Layer 2B verifying only the receipt hash.  Execution
    # order was already correct, so this was never a run-time defect -- but the
    # semantics were asymmetric: a hash proves the FILE is intact, not that the
    # values actually scored came from it.  The same coverage-then-deviation
    # structure is applied here, so "every scored prediction matches the
    # commitment" now holds for both layers rather than for one.
    joint_committed = json.loads(
        (output / "frozen_joint_predictions.json").read_text(encoding="utf-8")
    )
    committed_joint_lookup = {
        (entry["offset"], entry["duration"]): entry["predicted_visibility"]
        for entry in joint_committed["predictions"]
    }
    # Coverage FIRST: a scored point absent from the commitment must fail the
    # gate, not raise KeyError.
    missing_joint = [
        [row["offset"], row["duration"]]
        for row in transfer_rows
        if (row["offset"], row["duration"]) not in committed_joint_lookup
    ]
    maximum_joint_deviation = max(
        (
            abs(
                row["predicted_visibility"]
                - committed_joint_lookup[(row["offset"], row["duration"])]
            )
            for row in transfer_rows
            if (row["offset"], row["duration"]) in committed_joint_lookup
        ),
        default=0.0,
    )
    joint_commitment_coverage = {
        "committed_transfer_predictions": len(committed_joint_lookup),
        "scored_transfer_predictions": len(transfer_rows),
        "missing_transfer_points": missing_joint,
        "every_scored_point_was_committed": not missing_joint,
        "maximum_transfer_commitment_deviation": maximum_joint_deviation,
        "committed_parameters_match_the_estimates_used": (
            joint_committed["gamma_hat"] == gamma_hat
            and joint_committed["delta0_hat"] == delta0_hat
        ),
        "note": (
            "The deviation is computed only over points that ARE present in "
            "the commitment; absence is reported separately by "
            "every_scored_point_was_committed, and the gate requires both."
        ),
    }

    # ---- coverage of the joint asymptotic intervals ----------------------
    exact_cache = {
        (offset, duration): simulate_parity_visibility(
            cfg.true_gamma, cfg.true_delta0 + offset, duration, operators
        )
        for offset in cfg.control_offsets
        for duration in design_times(cfg, offset)
    }
    gamma_covered = delta0_covered = 0
    gamma_pulls: list[float] = []
    delta0_pulls: list[float] = []
    for _ in range(cfg.joint_coverage_replicates):
        replicate_rows = []
        for (offset, duration), exact in exact_cache.items():
            plus, _ = draw_counts(exact, cfg.joint_shots_per_time, coverage_rng)
            replicate_rows.append(
                {
                    "offset": offset,
                    "duration": duration,
                    "shots": cfg.joint_shots_per_time,
                    "plus_counts": plus,
                    "minus_counts": cfg.joint_shots_per_time - plus,
                }
            )
        try:
            g_hat, d_hat, cov, _ = fit_joint_mle(replicate_rows, cfg)
        except RuntimeError:
            continue
        gamma_sigma = math.sqrt(cov[0, 0])
        delta0_sigma = math.sqrt(cov[1, 1])
        gamma_pulls.append((g_hat - cfg.true_gamma) / gamma_sigma)
        delta0_pulls.append((d_hat - cfg.true_delta0) / delta0_sigma)
        gamma_covered += int(abs(g_hat - cfg.true_gamma) <= 1.96 * gamma_sigma)
        delta0_covered += int(abs(d_hat - cfg.true_delta0) <= 1.96 * delta0_sigma)
    replicates = len(gamma_pulls)
    # FIX v6.1: np.std(..., ddof=1) raises on fewer than two replicates; the
    # v6.0 max(replicates, 1) guard protected the ratios but not the spread.
    if replicates < 2:
        raise RuntimeError(
            f"joint coverage sweep produced {replicates} usable replicates; "
            "the joint MLE is not converging."
        )
    sigma_band = math.sqrt(
        cfg.coverage_nominal * (1 - cfg.coverage_nominal) / replicates
    )
    coverage = {
        "replicates": replicates,
        "gamma_empirical_coverage": gamma_covered / replicates,
        "delta0_empirical_coverage": delta0_covered / replicates,
        "binomial_sigma": sigma_band,
        "gamma_pull_standard_deviation": float(np.std(gamma_pulls, ddof=1)),
        "delta0_pull_standard_deviation": float(np.std(delta0_pulls, ddof=1)),
    }

    gates = {
        **identifiability["gates"],
        "joint_closed_form_matches_exact_liouvillian": model_deviation
        <= cfg.model_closed_form_tolerance,
        "joint_mle_converged": diagnostics["optimizer_success"],
        "gamma_recovered_jointly": gamma_relative_error
        <= cfg.gamma_relative_error_tolerance,
        "delta0_recovered_jointly": delta0_absolute_error
        <= cfg.delta0_absolute_error_tolerance,
        "two_parameter_transfer_within_combined_uncertainty": worst_pull
        <= cfg.transfer_sigma_tolerance,
        "joint_frozen_prediction_receipt_verifies_from_disk": (
            joint_frozen_verified
        ),
        "every_scored_joint_prediction_matches_the_commitment_on_disk": (
            joint_commitment_coverage["every_scored_point_was_committed"]
            and joint_commitment_coverage[
                "maximum_transfer_commitment_deviation"
            ]
            == 0.0
            and joint_commitment_coverage["scored_transfer_predictions"]
            == joint_commitment_coverage["committed_transfer_predictions"]
            and joint_commitment_coverage[
                "committed_parameters_match_the_estimates_used"
            ]
        ),
        "joint_gamma_coverage_consistent": abs(
            coverage["gamma_empirical_coverage"] - cfg.coverage_nominal
        )
        <= cfg.coverage_sigma_allowance * sigma_band,
        "joint_delta0_coverage_consistent": abs(
            coverage["delta0_empirical_coverage"] - cfg.coverage_nominal
        )
        <= cfg.coverage_sigma_allowance * sigma_band,
    }
    return (
        {
            "status": (
                "JOINT_IDENTIFICATION_SUPPORTED"
                if all(gates.values())
                else "JOINT_IDENTIFICATION_FAILED"
            ),
            "protocol": (
                "delta0 is unknown; the experimenter adds known control "
                "offsets.  Three settings of the |00>,|11> coherence identify "
                "(gamma, delta0) jointly; the logical X_L decay at unseen "
                "offsets is kept entirely as held-out data, and the two-"
                "parameter prediction for it is hashed to disk before that "
                "data is drawn."
            ),
            "identifiability_design_prior_voting": identifiability,
            "identifiability_at_estimates_nonvoting": identifiability_witness(
                cfg, gamma_hat, delta0_hat, "fitted_estimates_post_data"
            ),
            "true_gamma_hidden_from_fit": cfg.true_gamma,
            "true_delta0_hidden_from_fit": cfg.true_delta0,
            "gamma_hat": gamma_hat,
            "delta0_hat": delta0_hat,
            "gamma_relative_error": gamma_relative_error,
            "delta0_absolute_error": delta0_absolute_error,
            "fit_diagnostics": diagnostics,
            "maximum_closed_form_deviation": model_deviation,
            "transfer_maximum_absolute_pull": worst_pull,
            "joint_frozen_prediction_sha256": joint_frozen_hash,
            "joint_frozen_prediction_receipt_verified": joint_frozen_verified,
            "joint_commitment_coverage": joint_commitment_coverage,
            "coverage": coverage,
            "gates": gates,
            "design_note": (
                "Time grids come from design_prior_gamma/design_prior_delta0, "
                "not from the true values, and so does the voting "
                "identifiability witness.  Identification and the held-out "
                "test use disjoint observables: using X_L data to pin delta0 "
                "would buy identifiability at the cost of the held-out test, "
                "and that trade-off is a real constraint of the protocol."
            ),
        },
        {"joint_calibration": calibration_rows, "joint_transfer": transfer_rows},
    )


# ----------------------------------------------------------------------------
# Layer 3 -- model-misspecification sensitivity
# ----------------------------------------------------------------------------
def misspecification_sensitivity(
    cfg: Config, operators: dict[str, np.ndarray]
) -> dict[str, Any]:
    """How much unmodelled physics does this protocol actually detect?

    M1-M4 inject CODE defects.  This injects MODEL defects: the data generator
    contains a term the estimator does not know about.  For each mechanism the
    sweep reports the smallest kappa at which the protocol notices, separately
    for the same-observable held-out test and for the cross-experiment
    transfer test.  The gap between the two is the argument for keeping the
    transfer test at all.

    v6.1 note: kappa_detect and the kappa=0 false-alarm gate are single-draw
    statistics.  This was audited rather than assumed: over 300 reseeds the
    kappa=0 transfer max-pull had mean 1.87, p95 2.87 and max 3.88 against the
    4-sigma threshold (0/300 false alarms), and kappa_detect for the
    non-collective mechanism was 5e-4 in 40/40 reseeds.  At these shot counts
    the single draw is stable and is deliberately left as is.
    """
    mechanisms: dict[str, list[np.ndarray]] = {
        "non_collective_dephasing_Z1": [operators["Z1"]],
        "amplitude_damping_both_qubits": [operators["SM1"], operators["SM2"]],
    }
    report: dict[str, Any] = {
        "note": (
            "Under leakage the two-outcome binomial model for <X_L> is itself "
            "an approximation; leakage is reported alongside so the detection "
            "mechanism is visible."
        )
    }
    for name, generators in mechanisms.items():
        rng = np.random.default_rng(cfg.master_seed + 4242)
        rows: list[dict[str, Any]] = []
        for kappa in cfg.misspecification_kappas:
            extra = tuple(math.sqrt(kappa) * g for g in generators) if kappa > 0 else ()

            calibration = generate_rows(
                cfg.true_gamma,
                cfg.calibration_delta,
                cfg.calibration_times,
                cfg.shots_per_time,
                rng,
                "misspec_calibration",
                simulate_parity_visibility,
                operators,
                extra,
            )
            gamma_hat, gamma_sigma, _ = fit_gamma_mle(
                calibration, cfg.calibration_delta, cfg
            )

            def pulls(
                times: tuple[float, ...],
                delta: float,
                simulator: Callable[..., float],
                closed_form: Callable[[float, float, float], float],
            ) -> tuple[float, float]:
                worst = 0.0
                worst_leakage = 0.0
                for duration in times:
                    exact = simulator(
                        cfg.true_gamma, delta, duration, operators, extra
                    )
                    _, measured = draw_counts(exact, cfg.shots_per_time, rng)
                    predicted = closed_form(gamma_hat, delta, duration)
                    jacobian = (
                        closed_form(gamma_hat * 1.0001, delta, duration)
                        - predicted
                    ) / (gamma_hat * 0.0001)
                    sigma = math.sqrt(
                        max(1.0 - exact**2, 0.0) / cfg.shots_per_time
                        + (jacobian * gamma_sigma) ** 2
                    )
                    worst = max(worst, abs(measured - predicted) / sigma)
                    projector = operators["V"] @ operators["V"].conj().T
                    state = np.zeros(4, dtype=complex)
                    state[1] = state[2] = 1.0 / math.sqrt(2.0)
                    out = apply_superoperator(
                        channel_superoperator(
                            operators["H"],
                            [jump_for_delta(cfg.true_gamma, delta, operators)]
                            + list(extra),
                            duration,
                        ),
                        np.outer(state, state.conj()),
                    )
                    worst_leakage = max(
                        worst_leakage,
                        abs(
                            float(
                                np.real(
                                    np.trace((operators["I4"] - projector) @ out)
                                )
                            )
                        ),
                    )
                return worst, worst_leakage

            same_pull, _ = pulls(
                cfg.heldout_times,
                cfg.calibration_delta,
                simulate_parity_visibility,
                closed_form_parity_visibility,
            )
            transfer_pull = 0.0
            leakage = 0.0
            for delta, times in cfg.transfer_schedule:
                pull, leak = pulls(
                    times,
                    delta,
                    simulate_logical_x_visibility,
                    closed_form_logical_x_visibility,
                )
                transfer_pull = max(transfer_pull, pull)
                leakage = max(leakage, leak)

            rows.append(
                {
                    "kappa": kappa,
                    "kappa_over_gamma": kappa / cfg.true_gamma,
                    "gamma_hat": gamma_hat,
                    "gamma_absorption_bias": gamma_hat - cfg.true_gamma,
                    "same_observable_heldout_max_pull": same_pull,
                    "cross_experiment_transfer_max_pull": transfer_pull,
                    "maximum_code_space_leakage": leakage,
                }
            )

        threshold = cfg.misspecification_detection_sigma

        def first_detected(key: str) -> float | None:
            for row in rows:
                if row["kappa"] > 0.0 and row[key] > threshold:
                    return float(row["kappa"])
            return None

        report[name] = {
            "sweep": rows,
            "detection_threshold_sigma": threshold,
            "kappa_detect_same_observable": first_detected(
                "same_observable_heldout_max_pull"
            ),
            "kappa_detect_cross_experiment_transfer": first_detected(
                "cross_experiment_transfer_max_pull"
            ),
        }

    baselines = [
        row
        for entry in report.values()
        if isinstance(entry, dict) and "sweep" in entry
        for row in entry["sweep"]
        if row["kappa"] == 0.0
    ]
    detected = [
        entry["kappa_detect_cross_experiment_transfer"] is not None
        for entry in report.values()
        if isinstance(entry, dict) and "sweep" in entry
    ]
    report["gates"] = {
        "no_false_alarm_at_zero_misspecification": all(
            row["cross_experiment_transfer_max_pull"]
            <= cfg.misspecification_detection_sigma
            and row["same_observable_heldout_max_pull"]
            <= cfg.misspecification_detection_sigma
            for row in baselines
        ),
        "every_mechanism_detected_somewhere_in_the_sweep": all(detected),
    }
    report["interpretation"] = (
        "kappa_detect is the honest sensitivity of this protocol to unmodelled "
        "physics.  Where kappa_detect_same_observable is None but "
        "kappa_detect_cross_experiment_transfer is finite, the calibration "
        "curve absorbed the defect into gamma_hat without any residual "
        "signature, and only the transfer test saw it.  That case is the "
        "entire justification for the cross-experiment design."
    )
    return report


# ----------------------------------------------------------------------------
# negative control for the OPERATIONAL (Layer 0) gates
# ----------------------------------------------------------------------------
def operational_negative_control(
    cfg: Config, operators: dict[str, np.ndarray]
) -> dict[str, Any]:
    """FIX v6.1 (B5): give the Layer-0 gates something that can kill them.

    v6.0 introduced sixteen voting operational gates and not one mutation able
    to trip any of them.  By this script's own doctrine -- a gate that has never
    been observed to fail is not evidence -- the entire new layer was
    unwitnessed.  Each mutation below names the operational gate it targets.

    M7 is the one that matters.  A constant path has zero declared cost
    trivially; unless `finite_nonconstant_path` can actually fail, the
    zero-cost claim is a tautology.  M7 exhibits exactly that failure.
    """
    encoding = operators["V"]
    projector = encoding @ encoding.conj().T
    complement = operators["I4"] - projector
    times = np.linspace(0.0, cfg.target_duration, cfg.trajectory_samples)

    def probe(
        hamiltonian: np.ndarray, jump: np.ndarray, initial: np.ndarray
    ) -> dict[str, float]:
        kernel = jump.conj().T @ jump
        rates: list[float] = []
        distances: list[float] = []
        leakages: list[float] = []
        for point in times:
            state = expm(-1.0j * hamiltonian * point) @ initial
            rho = np.outer(state, state.conj())
            rates.append(max(0.0, float(np.real(np.trace(kernel @ rho)))))
            distances.append(pure_trace_distance(initial, state))
            leakages.append(
                abs(float(np.real(np.vdot(state, complement @ state))))
            )
        return {
            "accumulated_declared_jump_cost": integrate(rates, times),
            "maximum_trace_distance_from_initial": max(distances),
            "maximum_DFS_state_leakage": max(leakages),
        }

    collective = jump_for_delta(cfg.true_gamma, 0.0, operators)
    dfs_state = ket(1)
    control_state = (ket(0) + ket(1)) / math.sqrt(2.0)
    results: dict[str, Any] = {}

    # M5: a single-qubit jump is not collective, so the DFS is not in the
    #     kernel of the declared meter.  Expect J_Pi = gamma * T exactly.
    m5 = probe(
        operators["H"], math.sqrt(cfg.true_gamma) * operators["Z1"], dfs_state
    )
    results["M5_single_qubit_jump_breaks_the_cost_kernel"] = {
        "targets_gate": "zero_accumulated_declared_cost",
        "accumulated_declared_jump_cost": m5["accumulated_declared_jump_cost"],
        "analytic_expectation_gamma_times_T": cfg.true_gamma * cfg.target_duration,
        "tolerance": cfg.exact_zero_tolerance,
        "gate_fires": m5["accumulated_declared_jump_cost"]
        > cfg.exact_zero_tolerance,
    }

    # M6: a Hamiltonian that does not commute with the code projector drives
    #     the state out of the DFS.
    m6 = probe(
        operators["H"] + cfg.exchange_J * operators["X1"], collective, dfs_state
    )
    results["M6_hamiltonian_does_not_preserve_the_DFS"] = {
        "targets_gate": "state_path_stays_in_DFS",
        "maximum_DFS_state_leakage": m6["maximum_DFS_state_leakage"],
        "tolerance": cfg.leakage_tolerance,
        "gate_fires": m6["maximum_DFS_state_leakage"] > cfg.leakage_tolerance,
    }

    # M7: THE TAUTOLOGY CHECK.  H = 0 gives a constant path with zero cost.
    m7 = probe(np.zeros((4, 4), dtype=complex), collective, dfs_state)
    results["M7_constant_path_is_zero_cost_but_trivial"] = {
        "targets_gate": "finite_nonconstant_path",
        "maximum_trace_distance_from_initial": m7[
            "maximum_trace_distance_from_initial"
        ],
        "accumulated_declared_jump_cost": m7["accumulated_declared_jump_cost"],
        "threshold": cfg.nonconstant_trace_distance_minimum,
        "gate_fires": m7["maximum_trace_distance_from_initial"]
        < cfg.nonconstant_trace_distance_minimum,
    }

    # M8: preparing the positive control INSIDE the DFS makes the same-meter
    #     control vacuous; the gate must notice.
    m8 = probe(operators["H"], collective, dfs_state)
    results["M8_positive_control_prepared_inside_the_DFS"] = {
        "targets_gate": "same_meter_cost_positive",
        "declared_input": "|01> instead of (|00>+|01>)/sqrt(2)",
        "accumulated_declared_jump_cost": m8["accumulated_declared_jump_cost"],
        "threshold": cfg.same_meter_minimum_accumulated_cost,
        "gate_fires": not (
            m8["accumulated_declared_jump_cost"]
            > cfg.same_meter_minimum_accumulated_cost
        ),
    }

    reference = probe(operators["H"], collective, control_state)
    results["reference_unmutated_protocol"] = {
        "role": "sanity_only_not_a_mutation",
        "positive_control_cost": reference["accumulated_declared_jump_cost"],
    }
    results["all_operational_mutations_detected"] = all(
        entry["gate_fires"]
        for entry in results.values()
        if isinstance(entry, dict) and "gate_fires" in entry
    )
    return results


# ----------------------------------------------------------------------------
# negative control: prove the model gates can fail
# ----------------------------------------------------------------------------
def negative_control(
    cfg: Config, operators: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Deliberately inject four defects and confirm the matching gate fires.

    A gate that has never been observed to fail is not evidence.  Each entry
    below names the gate it is meant to kill and reports the observed value.
    """
    hamiltonian = operators["H"]
    rng = np.random.default_rng(cfg.master_seed + 1)
    results: dict[str, Any] = {}

    # M1: v3.1's Ramsey pair and closed form.  Fit exp(-2 gamma t) to data
    # taken on the |00>,|01> coherence with the Hamiltonian on.
    state = np.zeros(4, dtype=complex)
    state[0] = state[1] = 1.0 / math.sqrt(2.0)
    rows = []
    for duration in cfg.calibration_times:
        output = apply_superoperator(
            channel_superoperator(
                hamiltonian,
                [jump_for_delta(cfg.true_gamma, 0.0, operators)],
                duration,
            ),
            np.outer(state, state.conj()),
        )
        plus, _ = draw_counts(
            float(2.0 * abs(output[0, 1])), cfg.shots_per_time, rng
        )
        rows.append(
            {
                "duration": duration,
                "shots": cfg.shots_per_time,
                "plus_counts": plus,
                "minus_counts": cfg.shots_per_time - plus,
            }
        )

    def v31_nll(gamma: float) -> float:
        total = 0.0
        for row in rows:
            probability = float(
                np.clip(
                    0.5 * (1.0 + math.exp(-2.0 * gamma * row["duration"])),
                    1e-12,
                    1 - 1e-12,
                )
            )
            total -= row["plus_counts"] * math.log(probability)
            total -= row["minus_counts"] * math.log(1.0 - probability)
        return total

    bad_gamma = float(
        minimize_scalar(
            v31_nll, bounds=cfg.likelihood_bounds, method="bounded"
        ).x
    )
    bad_relative_error = abs(bad_gamma - cfg.true_gamma) / cfg.true_gamma
    results["M1_v31_ramsey_pair_and_formula"] = {
        "targets_gate": "gamma_recovered_from_calibration_only",
        "gamma_hat": bad_gamma,
        "relative_error": bad_relative_error,
        "tolerance": cfg.gamma_relative_error_tolerance,
        "gate_fires": bad_relative_error > cfg.gamma_relative_error_tolerance,
    }

    # M2: independent instead of collective dephasing -> DFS is destroyed.
    independent = [
        math.sqrt(cfg.true_gamma) * operators["Z1"],
        math.sqrt(cfg.true_gamma) * operators["Z2"],
    ]
    ideal, _ = choi_state(
        channel_superoperator(hamiltonian, [], cfg.target_duration),
        operators["V"],
    )
    broken, _ = choi_state(
        channel_superoperator(hamiltonian, independent, cfg.target_duration),
        operators["V"],
    )
    broken_cost = trace_distance(broken, ideal)
    results["M2_independent_not_collective_dephasing"] = {
        "targets_gate": "encoded_zero_exact_over_duration_sweep",
        "encoded_cost_at_delta_zero": broken_cost,
        "tolerance": cfg.exact_zero_tolerance,
        "gate_fires": broken_cost > cfg.exact_zero_tolerance,
    }

    # M3: sign error in the anticommutator term of the dissipator.
    dimension = 4
    identity = np.eye(dimension, dtype=complex)
    jump = jump_for_delta(cfg.true_gamma, 0.04, operators)
    kernel = jump.conj().T @ jump
    corrupted = -1.0j * (
        np.kron(identity, hamiltonian) - np.kron(hamiltonian.T, identity)
    )
    corrupted += (
        np.kron(jump.conj(), jump)
        + 0.5 * np.kron(identity, kernel)  # sign bug
        - 0.5 * np.kron(kernel.T, identity)
    )
    corrupted_superoperator = expm(corrupted * cfg.target_duration)
    corrupted_report = cptp_report(corrupted_superoperator, operators, cfg)
    results["M3_dissipator_sign_error"] = {
        "targets_gate": "cptp_trace_preservation",
        "trace_preservation_residual": corrupted_report[
            "trace_preservation_residual"
        ],
        "choi_trace": corrupted_report["choi_trace"],
        "gate_fires": not corrupted_report["trace_preservation_ok"],
    }

    # M4: a 20 percent biased gamma_hat must break the transfer prediction.
    biased_gamma = 1.20 * cfg.true_gamma
    pulls = []
    for delta, times in cfg.transfer_schedule:
        for duration in times:
            visibility = simulate_logical_x_visibility(
                cfg.true_gamma, delta, duration, operators
            )
            _, measured = draw_counts(visibility, cfg.shots_per_time, rng)
            sigma = math.sqrt(
                max(1.0 - visibility**2, 0.0) / cfg.shots_per_time
            )
            predicted = closed_form_logical_x_visibility(
                biased_gamma, delta, duration
            )
            pulls.append(abs(measured - predicted) / max(sigma, 1e-15))
    worst_pull = float(max(pulls))
    results["M4_gamma_biased_20_percent"] = {
        "targets_gate": "cross_experiment_transfer_within_shot_noise",
        "maximum_absolute_pull": worst_pull,
        "tolerance": cfg.transfer_sigma_tolerance,
        "gate_fires": worst_pull > cfg.transfer_sigma_tolerance,
    }

    results["all_mutations_detected"] = all(
        entry["gate_fires"]
        for entry in results.values()
        if isinstance(entry, dict) and "gate_fires" in entry
    )
    return results


# ----------------------------------------------------------------------------
# plotting (non-voting)
# ----------------------------------------------------------------------------
def save_plot(
    path: Path,
    powerlaw_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    transfer_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    gamma_hat: float,
    cfg: Config,
) -> str | None:
    figure = None
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 4, figsize=(17.5, 3.9))

        reference = [
            row
            for row in powerlaw_rows
            if abs(row["gamma"] - cfg.true_gamma) < 1e-12
            and abs(row["duration"] - cfg.target_duration) < 1e-12
        ]
        deltas = np.array([row["delta"] for row in reference])
        costs = np.array([row["encoded_cost_positive_delta"] for row in reference])
        axes[0].loglog(deltas, costs, "o", label="exact")
        axes[0].loglog(
            deltas,
            cfg.true_gamma * deltas**2 * cfg.target_duration,
            "-",
            label=r"$\gamma\delta^2 t$",
        )
        axes[0].set_title("Symmetry-breaking law")
        axes[0].set_xlabel(r"$\delta$")
        axes[0].set_ylabel(r"$E_{\rm ch}$")
        axes[0].legend()

        grid = np.linspace(
            0.0,
            max(
                max(r["duration"] for r in calibration_rows),
                max(r["duration"] for r in heldout_rows),
            ),
            300,
        )
        axes[1].plot(
            grid,
            [
                closed_form_parity_visibility(gamma_hat, cfg.calibration_delta, t)
                for t in grid
            ],
            label="frozen fit",
        )
        axes[1].scatter(
            [r["duration"] for r in calibration_rows],
            [r["measured_visibility"] for r in calibration_rows],
            label="calibration",
        )
        axes[1].scatter(
            [r["duration"] for r in heldout_rows],
            [r["measured_visibility"] for r in heldout_rows],
            marker="x",
            label="held out",
        )
        axes[1].set_title(r"Calibration: $|00\rangle,|11\rangle$")
        axes[1].set_xlabel("time")
        axes[1].set_ylabel("visibility")
        axes[1].legend()

        for delta, times in cfg.transfer_schedule:
            block = [r for r in transfer_rows if r["delta"] == delta]
            grid = np.linspace(0.0, max(times), 300)
            line = axes[2].plot(
                grid,
                [
                    closed_form_logical_x_visibility(gamma_hat, delta, t)
                    for t in grid
                ],
                label=rf"$\delta={delta}$ frozen",
            )
            axes[2].scatter(
                [r["duration"] for r in block],
                [r["measured_visibility"] for r in block],
                marker="x",
                color=line[0].get_color(),
            )
        axes[2].set_title(r"Transfer: logical $X_L$")
        axes[2].set_xlabel("time")
        axes[2].set_ylabel(r"$\langle X_L\rangle$")
        axes[2].legend(fontsize=8)

        axes[3].plot(
            [r["delta"] for r in prediction_rows],
            [r["truth_exact_liouvillian"] for r in prediction_rows],
            "o-",
            label="exact",
        )
        axes[3].plot(
            [r["delta"] for r in prediction_rows],
            [r["predicted_encoded_cost_closed_form"] for r in prediction_rows],
            "x--",
            label="frozen prediction",
        )
        axes[3].set_title("Encoded cost extrapolation")
        axes[3].set_xlabel(r"$\delta$")
        axes[3].set_ylabel(r"$E_{\rm ch}$")
        axes[3].legend()

        for axis in axes:
            axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        return path.name
    except Exception as exc:
        print(f"[plot] non-voting plot failed; gates unaffected: {exc!r}")
        return None
    finally:
        if figure is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(figure)
            except Exception:
                pass


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DFS operational-protocol + channel support audit v6.2.2"
    )
    parser.add_argument("--output-dir")
    raw, cleaned, ignored, index = sys.argv[1:], [], [], 0
    while index < len(raw):
        if raw[index] == "-f" and index + 1 < len(raw):
            ignored.extend(raw[index : index + 2])
            index += 2
        elif raw[index].startswith("-f="):
            ignored.append(raw[index])
            index += 1
        else:
            cleaned.append(raw[index])
            index += 1
    if ignored:
        print(f"[notebook] ignored kernel arguments: {ignored}")
    return parser.parse_args(cleaned)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = create_unique_output_dir(args.output_dir)
    script_value = globals().get("__file__")
    script_path = (
        Path(script_value).resolve()
        if script_value and Path(script_value).is_file()
        else None
    )
    summary: dict[str, Any] = {
        "version": VERSION,
        "status": "RUNNING",
        "provenance": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "matplotlib": package_version("matplotlib"),
            "script_path": str(script_path) if script_path else None,
            "script_sha256": sha256_file(script_path),
        },
    }
    save_json(output / "summary.json", summary)

    print("\n" + "=" * 100)
    print("DFS OPERATIONAL PROTOCOL + CHANNEL SUPPORT AUDIT v6.2.2")
    print("=" * 100)
    print("backend=exact NumPy/SciPy | cloud access=none")

    try:
        cfg = Config()

        # The manifest is deliberately serialized before operators are built or
        # any outcome is evaluated.  FIX v6.1 (B6): the commitment is now read
        # back from disk and re-hashed, instead of being asserted True.
        frozen_protocol = protocol_manifest(cfg)
        frozen_protocol_sha256, protocol_commitment_verified = freeze_and_commit(
            output, "frozen_protocol", frozen_protocol
        )
        operators = build_operators(cfg)

        print("\n[COMMITMENT] Frozen operational protocol manifest")
        print(
            json.dumps(
                {
                    "protocol_id": frozen_protocol["protocol_id"],
                    "protocol_sha256": frozen_protocol_sha256,
                    "commitment_verifies_from_disk": protocol_commitment_verified,
                },
                indent=2,
            )
        )

        print(
            "\n[LAYER 0] Integrable operational zero-mode audit "
            "-> certificate.operational_zero_mode"
        )
        operational, trajectory_rows, contraction_rows = (
            operational_zero_mode_audit(cfg, operators)
        )
        print(json.dumps(clean(operational["gates"]), indent=2))
        print(
            "  max path distance="
            f"{operational['finite_zero_cost_trajectory']['maximum_trace_distance_from_initial']:.6f}"
            "  J_Pi="
            f"{operational['finite_zero_cost_trajectory']['accumulated_declared_jump_cost']:.3e}"
            "  positive-control J_Pi="
            f"{operational['same_meter_positive_control']['accumulated_declared_jump_cost']:.6f}"
        )

        print("\n[LAYER 1] Channel-level structural audit -> certificate.layer1")
        layer1, selection_rows, powerlaw_rows = layer1_audit(cfg, operators)
        print(json.dumps(clean(layer1["gates"]), indent=2))
        law = layer1["symmetry_breaking_law"]
        print(f"  scaling audit over {law['number_of_slices']} (gamma, t) slices:")
        print(
            f"{'gamma':>9}{'t':>10}{'exponent':>12}"
            f"{'coef_relerr':>14}{'parity_err':>13}"
        )
        for entry in law["slices"]:
            print(
                f"{entry['gamma']:9.3f}{entry['duration']:10.4f}"
                f"{entry['fitted_delta_exponent']:12.6f}"
                f"{entry['coefficient_relative_error']:14.3e}"
                f"{entry['maximum_parity_relative_error']:13.2e}"
            )
        print(
            f"  exponent range = [{law['minimum_fitted_delta_exponent']:.6f}, "
            f"{law['maximum_fitted_delta_exponent']:.6f}]"
            f"   max coef rel err = {law['maximum_coefficient_relative_error']:.3e}"
            f" at gamma={law['worst_coefficient_slice']['gamma']},"
            f" t={law['worst_coefficient_slice']['duration']:.4f}"
        )
        print(
            f"  max delta<->-delta parity err = "
            f"{law['maximum_parity_relative_error']:.3e}"
            f"   max_leakage = {layer1['maximum_leakage_population']:.3e}"
        )
        print("  [non-voting] representation regression:")
        print("   ", json.dumps(clean(layer1["nonvoting_regression_checks"])))

        print("\n[DIAGNOSTIC CONTROL] Injected defects must trip their gates")
        operational_control = operational_negative_control(cfg, operators)
        print("  operational (Layer 0) mutations:")
        print("   ", json.dumps(clean({
            k: v["gate_fires"] for k, v in operational_control.items()
            if isinstance(v, dict) and "gate_fires" in v
        })))
        control = negative_control(cfg, operators)
        print("  model mutations:")
        print("   ", json.dumps(clean({
            k: v["gate_fires"] for k, v in control.items()
            if isinstance(v, dict) and "gate_fires" in v
        })))

        print("\n[LAYER 2] Cross-experiment calibration -> certificate.layer2")
        layer2, data = layer2_calibration(cfg, operators, output)
        print(json.dumps(clean(layer2["gates"]), indent=2))
        print(
            f"  gamma_hat={layer2['gamma_hat']:.8f}"
            f"  rel_err={layer2['gamma_relative_error']:.3e}"
            f"  coverage={layer2['coverage_sweep']['empirical_coverage']:.3f}"
            f"  d_ln_E/d_ln_g="
            f"{layer2['ech_extrapolation']['d_log_Ech_d_log_gamma']:.4f}"
        )

        print(
            "\n[LAYER 2B] Joint identification of (gamma, delta0) "
            "-> certificate.layer2b_joint_identification"
        )
        layer2b, joint_data = layer2b_joint_identification(cfg, operators, output)
        print(json.dumps(clean(layer2b["gates"]), indent=2))
        print(
            f"  gamma_hat={layer2b['gamma_hat']:.8f}"
            f"  delta0_hat={layer2b['delta0_hat']:.6f}"
            f" (true {cfg.true_delta0})"
            f"  SE={layer2b['fit_diagnostics']['standard_errors']}"
            f"  corr={layer2b['fit_diagnostics']['correlation']:+.4f}"
            f"  transfer_max_pull={layer2b['transfer_maximum_absolute_pull']:.2f}"
        )
        print(
            "  identifiability evaluated at "
            f"{layer2b['identifiability_design_prior_voting']['evaluation_point']}"
            " (voting):"
            " single_ratio="
            f"{layer2b['identifiability_design_prior_voting']['single_setting_eigenvalue_ratio']:.3e}"
            "   [non-voting] at estimates: single_ratio="
            f"{layer2b['identifiability_at_estimates_nonvoting']['single_setting_eigenvalue_ratio']:.3e}"
        )

        print(
            "\n[LAYER 3] Model-misspecification sensitivity "
            "-> certificate.layer3_misspecification_sensitivity"
        )
        misspecification = misspecification_sensitivity(cfg, operators)
        for name, entry in misspecification.items():
            if not isinstance(entry, dict) or "sweep" not in entry:
                continue
            print(f"  {name}:")
            print(
                "    kappa   k/gamma  gamma_bias   same_obs_pull  "
                "transfer_pull   leakage"
            )
            for row in entry["sweep"]:
                print(
                    f"    {row['kappa']:.4f}  {row['kappa_over_gamma']:7.4f}"
                    f"  {row['gamma_absorption_bias']:+.6f}"
                    f"  {row['same_observable_heldout_max_pull']:13.2f}"
                    f"  {row['cross_experiment_transfer_max_pull']:13.2f}"
                    f"  {row['maximum_code_space_leakage']:9.2e}"
                )
            print(
                f"    kappa_detect: same_observable="
                f"{entry['kappa_detect_same_observable']}"
                f"  transfer={entry['kappa_detect_cross_experiment_transfer']}"
            )
        print(json.dumps(clean(misspecification["gates"]), indent=2))

        l1, l2, l2b = layer1["gates"], layer2["gates"], layer2b["gates"]

        # ---- bucket 1: protocol-first operational claim -------------------
        operational_gates = {
            # FIX v6.1 (B6): v6.0 hardcoded this to True -- an unfailable gate,
            # exactly the defect v3.1 was criticised for.  It now re-reads the
            # manifest from disk and re-hashes it.  Honest scope: this verifies
            # that the commitment is reproducible, NOT that no outcome was
            # computed first; that ordering is enforced by code position.
            "protocol_commitment_verifies_from_disk": (
                protocol_commitment_verified
            ),
            "analytic_DFS_kernel_and_invariance": all(
                operational["gates"][key]
                for key in (
                    "L0_annihilates_DFS",
                    "K0_annihilates_DFS",
                    "Hamiltonian_preserves_DFS",
                    "DFS_equals_cost_kernel_dimension",
                )
            ),
            "finite_integrable_nonconstant_zero_cost_path": all(
                operational["gates"][key]
                for key in (
                    "finite_nonconstant_path",
                    "zero_accumulated_declared_cost",
                    "zero_accumulated_dissipator_activity",
                    "state_path_stays_in_DFS",
                    "path_tangent_stays_in_DFS",
                    "path_tangent_stays_in_cost_kernel",
                )
            ),
            "same_meter_positive_control": all(
                operational["gates"][key]
                for key in (
                    "same_meter_cost_positive",
                    "same_meter_activity_positive",
                    "same_meter_cost_matches_analytic_calibration",
                )
            ),
            "contraction_family": all(
                operational["gates"][key]
                for key in (
                    "finite_zero_cost_family",
                    "every_positive_scale_path_nonconstant",
                    "family_contracts_toward_constant_path",
                )
            ),
        }

        # ---- bucket 2: gates that vote on declared model support ----------
        model_gates = {
            "logical_reduction_to_one_qubit_channel": l1[
                "encoded_channel_reduces_to_one_qubit_channel"
            ],
            "encoded_zero": l1["encoded_zero_exact_over_duration_sweep"],
            "no_leakage": l1["code_space_exactly_invariant_no_leakage"],
            "full_physical_channel_not_ideal": l1[
                "full_physical_channel_not_ideal"
            ],
            "delta_squared_opening": all(
                v for k, v in l1.items() if k.startswith("scaling_")
            ),
            "cptp": all(v for k, v in l1.items() if k.startswith("cptp_")),
            "calibration_model_validity": (
                l2["closed_forms_match_exact_liouvillian"]
                and l2b["joint_closed_form_matches_exact_liouvillian"]
            ),
            "gamma_identification": (
                l2["mle_converged"]
                and l2["mle_optimum_interior"]
                and l2["gamma_recovered_from_calibration_only"]
                and l2["heldout_residuals_within_shot_noise"]
            ),
            "interval_coverage": all(
                v for k, v in l2.items() if k.startswith("coverage_")
            )
            and l2b["joint_gamma_coverage_consistent"]
            and l2b["joint_delta0_coverage_consistent"],
            "cross_experiment_transfer": (
                l2["cross_experiment_transfer_within_shot_noise"]
                and l2b["two_parameter_transfer_within_combined_uncertainty"]
            ),
            "encoded_cost_extrapolation": (
                l2["ech_extrapolation_within_tolerance"]
                and l2["ech_closed_form_model_error_small"]
            ),
            "joint_identification": (
                l2b["single_control_setting_is_rank_deficient"]
                and l2b["multiple_control_settings_identify_both"]
                and l2b["joint_mle_converged"]
                and l2b["gamma_recovered_jointly"]
                and l2b["delta0_recovered_jointly"]
            ),
            "prediction_freezes_verify_from_disk": (
                l2["frozen_prediction_receipt_verifies_from_disk"]
                and l2["every_scored_prediction_matches_the_commitment_on_disk"]
                and l2b["joint_frozen_prediction_receipt_verifies_from_disk"]
                and l2b[
                    "every_scored_joint_prediction_matches_the_commitment_on_disk"
                ]
            ),
            "misspecification_sensitivity": all(
                misspecification["gates"].values()
            ),
        }

        # ---- bucket 3: does the diagnostic system actually detect faults? --
        # These validate the INSTRUMENT.  They are not physical findings and
        # they do not vote on declared model support.
        diagnostic_validation_checks = {
            "injected_single_qubit_jump_detected": operational_control[
                "M5_single_qubit_jump_breaks_the_cost_kernel"
            ]["gate_fires"],
            "injected_DFS_breaking_hamiltonian_detected": operational_control[
                "M6_hamiltonian_does_not_preserve_the_DFS"
            ]["gate_fires"],
            "trivial_constant_path_rejected": operational_control[
                "M7_constant_path_is_zero_cost_but_trivial"
            ]["gate_fires"],
            "positive_control_inside_DFS_rejected": operational_control[
                "M8_positive_control_prepared_inside_the_DFS"
            ]["gate_fires"],
            "injected_ramsey_error_detected": control[
                "M1_v31_ramsey_pair_and_formula"
            ]["gate_fires"],
            "injected_noncollective_dephasing_detected": control[
                "M2_independent_not_collective_dephasing"
            ]["gate_fires"],
            "injected_sign_error_detected": control["M3_dissipator_sign_error"][
                "gate_fires"
            ],
            "injected_gamma_bias_detected": control[
                "M4_gamma_biased_20_percent"
            ]["gate_fires"],
            "v31_ramsey_formula_inconsistent_with_liouvillian": l2[
                "v31_formula_demonstrably_inconsistent"
            ],
        }

        # ---- bucket 4: code regression only -------------------------------
        # E_ch is defined from the superoperator, so invariance under a change
        # of Lindblad representation is true by construction.  Verifying it
        # tests the implementation, not the physics.  Non-voting by design.
        nonvoting_regression_checks = dict(layer1["nonvoting_regression_checks"])

        operational_support = all(operational_gates.values())
        declared_model_support = all(model_gates.values())
        diagnostics_validated = all(diagnostic_validation_checks.values())
        regression_clean = all(nonvoting_regression_checks.values())
        scientific_status = (
            "DECLARED_DFS_PROTOCOL_HAS_AN_INTEGRABLE_NONCONSTANT_ZERO_COST_PATH"
            if operational_support and declared_model_support
            else "OPERATIONAL_OR_MODEL_GATE_FAILURE"
        )
        global_gates = {
            "declared_operational_protocol_support": operational_support,
            "declared_model_support": declared_model_support,
            "diagnostic_system_validated": diagnostics_validated,
            "code_regression_clean": regression_clean,
        }
        certificate = {
            "version": VERSION,
            "scientific_status": scientific_status,
            "declared_operational_protocol_support": operational_support,
            "declared_model_support": declared_model_support,
            "support_type": (
                "exact-model + synthetic-counts; NOT physical measurement. "
                "`declared_model_support` means every voting model gate passed "
                "for the declared Lindblad model with synthetic finite-shot "
                "data.  It is not evidence about hardware."
            ),
            "frozen_protocol": frozen_protocol,
            "frozen_protocol_sha256": frozen_protocol_sha256,
            "protocol_commitment_verified_from_disk": protocol_commitment_verified,
            "object_hierarchy": operational["object_hierarchy"],
            "operational_gates": operational_gates,
            "model_gates": model_gates,
            "diagnostic_validation_checks": diagnostic_validation_checks,
            "nonvoting_regression_checks": nonvoting_regression_checks,
            "gate_bucket_semantics": {
                "freezing_semantics": (
                    "Every commitment in this run is EXECUTION-ORDER FREEZING "
                    "WITHIN THE EXECUTABLE AUDIT: the payload is serialized and "
                    "hashed at a point in the code that precedes the "
                    "corresponding draw or truth evaluation, and the hash is "
                    "re-verified from disk.  With no external timestamp and no "
                    "third party this is NOT formal preregistration, and must "
                    "not be written up as such."
                ),
                "operational_gates": (
                    "Vote on the restricted operational statement: the frozen "
                    "two-qubit protocol has a finite nonconstant path with zero "
                    "accumulated declared jump cost, plus a same-meter positive "
                    "control and a contracting family.  Every one of them is "
                    "exercised against an injected defect in "
                    "`operational_negative_control`."
                ),
                "model_gates": (
                    "Vote on declared_model_support.  Each is a falsifiable "
                    "statement about the declared model or the estimation "
                    "protocol."
                ),
                "diagnostic_validation_checks": (
                    "Confirm that injected faults are detected.  They validate "
                    "the instrument, not the physics, and must not be reported "
                    "as findings."
                ),
                "nonvoting_regression_checks": (
                    "Representation-change invariance, true by construction for "
                    "a superoperator-defined cost.  Implementation regression "
                    "only.  Non-voting."
                ),
            },
            "diagnostics_validated": diagnostics_validated,
            "code_regression_clean": regression_clean,
            "frozen_config": asdict(cfg),
            "operational_zero_mode": operational,
            "layer1": layer1,
            "layer2": layer2,
            "layer2b_joint_identification": layer2b,
            "layer3_misspecification_sensitivity": misspecification,
            "negative_control": control,
            "operational_negative_control": operational_control,
            "global_gates": global_gates,
            "changes_from_v31": [
                "v6: added a protocol-first Layer 0; the preparation, "
                "Hamiltonian, declared meter, readouts, tolerances, and "
                "decision rules are serialized and hashed before outcomes.",
                "v6: separated the local model rate j_Pi, accumulated path "
                "cost J_Pi, and finite encoded-channel witness E_ch.",
                "v6: added an algebraic DFS kernel/invariance certificate, a "
                "finite nonconstant zero-cost trajectory, a same-meter "
                "positive control, and a contracting H_s = s H family.",
                "v6: explicitly records abstract tangent F as NOT_CONSTRUCTED "
                "and the bridge to universal Principle R as NOT_ESTABLISHED.",
                "Ramsey observable moved to the H-invariant |00>,|11> pair; "
                "v3.1's exp(-2 gamma t) belongs to the |00>,|01> coherence, "
                "which H rotates.  Witness included in layer2.model_validity.",
                "Counts are now generated by exact Liouvillian evolution "
                "rather than by the same closed form used to fit.",
                "Removed the unfailable channel-prediction gate; replaced by a "
                "cross-observable transfer test and by an explicit reporting "
                "of d ln E_ch / d ln gamma.",
                "Replaced 'E_ch > 1e-8' with the quantitative law "
                "E_ch = gamma delta^2 t (slope and prefactor).",
                "Added CPTP gates and the pre-symmetrization Hermiticity "
                "residual that v3.1 discarded.",
                "Added non-degenerate representation tests (distinct jumps, "
                "inhomogeneous gauge) and relabelled all of them as code "
                "regression rather than physical evidence.",
                "Interval coverage measured over a seed sweep, not one draw.",
                "Predictions hashed before truth is computed.",
                "Added a negative-control mutation suite so that every gate "
                "class has been observed to fail on an injected defect.",
                "v5: Layer 1 now carries an explicit reduction of the encoded "
                "channel to a one-qubit channel, so the zero, the absence of "
                "leakage and the delta^2 law are corollaries of an algebraic "
                "identity rather than numerical discoveries.",
                "v5: added joint identification of (gamma, delta0) from three "
                "known control offsets, with an explicit Fisher rank witness "
                "showing that a single setting cannot identify the pair.",
                "v5: transfer residuals are now compared against shot noise "
                "AND propagated parameter uncertainty, not shot noise alone.",
                "v5: added a model-misspecification sweep reporting "
                "kappa_detect for unmodelled non-collective dephasing and "
                "amplitude damping, separately for the same-observable and "
                "cross-experiment tests.",
                "v5.1: renamed physical_support -> declared_model_support; the "
                "evidence is exact-model plus synthetic counts and must not be "
                "labelled physical.",
                "v5.1: gates split into three buckets - model_gates (voting), "
                "diagnostic_validation_checks (instrument validation), and "
                "nonvoting_regression_checks (representation invariance, true "
                "by construction).  Representation changes are not physical "
                "evidence and injected faults are not findings.",
                "v5.1: the delta^2 law is audited slice by slice over a "
                "(gamma, t) grid, reporting the exponent range, the worst "
                "coefficient error and its slice, and the exact "
                "delta <-> -delta parity error.",
                "Fixed: expected-vs-observed Fisher naming, MLE boundary "
                "check, CSV field-name union, independent RNG streams.",
                "v6.1 B1: freeze_and_commit deep-copies, hashes and re-reads "
                "each payload from disk.  v6.0 hashed an object that aliased "
                "prediction_rows and was mutated when the truth was revealed, "
                "so the frozen object stopped matching its own receipt "
                "(ef985ffa... -> cec041c2...) and nothing re-verified it.",
                "v6.1 B2: Layer 2B now commits and hashes its two-parameter "
                "transfer prediction BEFORE the held-out data is drawn.  v6.0 "
                "wrote that file, unhashed, only after the pulls were computed.",
                "v6.1 B3: the voting identifiability witness is evaluated at "
                "the stated design prior, not at cfg.true_gamma/true_delta0; "
                "the achieved information at the fitted values is reported "
                "separately and does not vote.",
                "v6.1 B4: the single-setting rank-deficiency ratio uses "
                "magnitudes.  v6.0's single[0]/single[-1] was -1.090e-17, so "
                "the gate passed on sign and would also have passed for an "
                "indefinite matrix.",
                "v6.1 B5: added operational_negative_control (M5-M8).  v6.0 "
                "shipped sixteen voting Layer-0 gates with no mutation able to "
                "trip any of them.  M7 (H=0) is the tautology check: it shows "
                "that finite_nonconstant_path really can fail.",
                "v6.1 B6: replaced the hardcoded "
                "protocol_frozen_before_outcomes=True with a disk round-trip "
                "verification, renamed to state only what it checks.",
                "v6.1 robustness: NumPy 1.x trapezoid fallback; explicit "
                "errors for offsets that cancel the prior imbalance and for "
                "coverage sweeps with fewer than two usable replicates; stdout "
                "layer labels aligned with certificate keys; the positive "
                "control gated on a cost threshold instead of the "
                "trace-distance threshold.",
                "v6.2 item 1: the freezing guarantee is named "
                "'execution-order freezing within the executable audit' in the "
                "receipts, the gate-bucket semantics and the claim boundary; "
                "hashing without an external timestamp is not preregistration.",
                "v6.2 item 2: one commitment file now covers the "
                "same-observable held-out points, the cross-observable transfer "
                "points and the E_ch extrapolation.  Order is calibration fit "
                "-> build every held-out prediction -> commit -> draw -> score, "
                "and every scored value is re-checked against the file on disk.",
                "v6.2 item 3: contraction_epsilons -> contraction_scales, "
                "epsilon -> scale_s, every_positive_epsilon_path_nonconstant -> "
                "every_positive_scale_path_nonconstant, to match H_s = s H in "
                "the manuscript.  This renames a frozen manifest field, so "
                "protocol_sha256 differs from the v6.0/v6.1 value by "
                "construction.",
                "v6.2.1: the E_ch payload is compared point by point against "
                "the commitment instead of merely counted, and the gate "
                "requires both the visibility and the encoded-cost deviation "
                "to be exactly zero; v6.2's gate name was stronger than what "
                "it verified.",
                "v6.2.1: commitment coverage is established before any "
                "deviation is computed, so a scored point missing from the "
                "commitment fails the gate instead of raising KeyError.",
                "v6.2.2: Layer 2B now compares every scored joint-transfer "
                "prediction against its on-disk commitment point by point, and "
                "checks that the committed (gamma_hat, delta0_hat) are the "
                "estimates used.  v6.2.1 verified only the receipt hash there, "
                "so the point-by-point semantics held for Layer 2 alone.",
                "v6.1 audited and deliberately unchanged: kappa=0 false-alarm "
                "and kappa_detect are single draws, but over 300 reseeds the "
                "kappa=0 transfer max-pull had mean 1.87 / p95 2.87 / max 3.88 "
                "against a 4-sigma threshold, and kappa_detect was 5e-4 in "
                "40/40 reseeds.",
            ],
            "claim_boundary": (
                "For the frozen exact two-qubit collective-dephasing protocol, "
                "a finite nonconstant trajectory remains in the kernel of the "
                "predeclared local jump rate and has zero accumulated declared "
                "jump cost; the same frozen meter is positive on a control, "
                "and a finite s-family (H_s = s H) contracts toward the "
                "constant path.  That the nonconstancy requirement is not vacuous is "
                "itself witnessed: an H=0 mutation trips it.  The encoded "
                "Choi-distance zero is a separate finite channel-level "
                "statement about a leakage-free logical qubit, exact over the "
                "tested duration sweep.  Symmetry breaking follows E_ch = "
                "gamma delta^2 t at leading order.  gamma is inferred from "
                "finite-shot synthetic data on an H-invariant non-code-space "
                "coherence and transfers to an unseen observable at unseen "
                "delta, and every Layer 2 held-out prediction is written to a "
                "single commitment file and re-checked against every scored "
                "value.  All freezing here is execution-order freezing within "
                "the executable audit, not formal preregistration.  With three "
                "known control offsets, gamma and an "
                "unknown intrinsic imbalance delta0 are jointly identifiable, "
                "and the protocol's sensitivity to unmodelled physics is "
                "quantified as kappa_detect rather than assumed.  No QPU, no "
                "laboratory calibration, no zero-total-energy, no universal "
                "realizability, and no Lorentzian claim.  Generator and "
                "estimator still share a model family; kappa_detect bounds how "
                "far that assumption can be wrong before this protocol notices. "
                "The script does not construct a universal tangent density F "
                "and does not establish Principle R or Lorentzian signature."
            ),
        }

        save_json(output / "certificate.json", certificate)
        save_csv(output / "operational_trajectory.csv", trajectory_rows)
        save_csv(output / "contraction_family.csv", contraction_rows)
        save_csv(output / "layer1_selection.csv", selection_rows)
        save_csv(output / "layer1_scaling_slices.csv", powerlaw_rows)
        save_csv(
            output / "layer1_scaling_slice_fits.csv",
            layer1["symmetry_breaking_law"]["slices"],
        )
        save_csv(output / "calibration_counts.csv", data["calibration"])
        save_csv(output / "heldout_counts.csv", data["heldout"])
        save_csv(output / "transfer_counts.csv", data["transfer"])
        save_csv(output / "ech_extrapolation.csv", data["predictions"])
        save_csv(output / "joint_calibration.csv", joint_data["joint_calibration"])
        save_csv(output / "joint_transfer.csv", joint_data["joint_transfer"])
        save_csv(
            output / "misspecification_sweep.csv",
            [
                {"mechanism": name, **row}
                for name, entry in misspecification.items()
                if isinstance(entry, dict) and "sweep" in entry
                for row in entry["sweep"]
            ],
        )
        figure = save_plot(
            output / "diagnostic.png",
            powerlaw_rows,
            data["calibration"],
            data["heldout"],
            data["transfer"],
            data["predictions"],
            layer2["gamma_hat"],
            cfg,
        )

        summary.update(
            {
                "status": "COMPLETE",
                "scientific_status": scientific_status,
                "declared_model_support": declared_model_support,
                "declared_operational_protocol_support": operational_support,
                "diagnostics_validated": diagnostics_validated,
                "code_regression_clean": regression_clean,
                "figure": figure,
            }
        )

        print("\n" + "=" * 100)
        print("GLOBAL VERDICT")
        print("=" * 100)
        failed = [
            k
            for k, v in {
                **operational["gates"],
                **layer1["gates"],
                **layer2["gates"],
                **layer2b["gates"],
                **misspecification["gates"],
                **layer1["nonvoting_regression_checks"],
                **diagnostic_validation_checks,
            }.items()
            if not v
        ]
        print(
            json.dumps(
                clean(
                    {
                        "scientific_status": scientific_status,
                        "declared_operational_protocol_support": (
                            operational_support
                        ),
                        "declared_model_support": declared_model_support,
                        "support_type": (
                            "exact-model + synthetic-counts; "
                            "NOT physical measurement"
                        ),
                        "operational_gates": operational_gates,
                        "model_gates": model_gates,
                        "diagnostic_validation_checks": (
                            diagnostic_validation_checks
                        ),
                        "nonvoting_regression_checks": (
                            nonvoting_regression_checks
                        ),
                        "global_gates": global_gates,
                        "failed_gates": failed,
                        "claim_boundary": certificate["claim_boundary"],
                    }
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        if not (
            operational_support
            and declared_model_support
            and diagnostics_validated
            and regression_clean
        ):
            raise AssertionError(f"Gates failed: {failed}")
    except Exception as exc:
        summary.update(
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        save_json(output / "summary.json", summary)
        print(f"elapsed={summary['elapsed_seconds']:.2f}s")
        print(f"outputs={output}")


if __name__ == "__main__":
    main()
