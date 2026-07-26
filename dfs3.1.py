#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFS CHANNEL-COST + SYNTHETIC CALIBRATION v3.3
=======================================

Layer 1 — encoded-channel witness and numerical channel audit
------------------------------------------------
For a logical qubit encoded in span{|01>,|10>}, compare the finite-time noisy
channel with the ideal unitary channel using normalized Choi-state trace
distance:

    E_ch = 1/2 ||J(noisy) - J(ideal)||_1.

This quantity is computed from the channel itself, not from a chosen Lindblad
unravelling.  The script checks:

  * E_ch = 0 for exactly collective dephasing on the encoded DFS;
  * the corresponding full four-dimensional physical channel is not equal to
    the ideal channel;
  * a controlled coupling imbalance
        L_delta = sqrt(gamma)[Z1 + (1+delta)Z2]
    opens a positive encoded-channel loss;
  * CPTP conditions are checked before any Choi Hermitian symmetrization;
  * nontrivial Lindblad representation changes are regression-tested.

Layer 2 — separated synthetic calibration
--------------------------------------------------------
The true gamma is used only by a data generator.  A calibration module sees
Ramsey counts from (|00>+|11>)/sqrt(2), propagated by the same full
Liouvillian used in Layer 1, and estimates gamma by binomial maximum
likelihood.  Both basis states are annihilated by the exchange Hamiltonian, so
the exact visibility exp(-8 gamma t) remains valid with H switched on.  The
fitted gamma is then frozen and used to predict disjoint held-out Ramsey data.

Layer 3 — falsifiable symmetry-breaking law
--------------------------------------------------------
The encoded Choi witness is scanned over disjoint gamma, |delta|, and time
grids.  The audit tests the finite-range log-log delta exponent, reflection
evenness, and the independently derived weak-imbalance coefficient

    E_ch / (gamma delta^2 t) -> 1.

This replaces the v3.1 channel-relative-error gate, which was algebraically
redundant with gamma recovery in the linear regime.

The prediction module is passed gamma_hat only; it does not receive gamma_true.

Boundary
--------
This is exact NumPy/SciPy model evidence plus finite-shot synthetic
calibration.  It is not QPU data, not an experimentally calibrated cost, not
zero total energy, and not a universal realizability or Lorentzian theorem.

Run:
    pip install -U numpy scipy matplotlib
    python dfs_channel_cost_calibration_v3_3.py

Jupyter/Colab ``-f kernel.json`` arguments are ignored.
"""
from __future__ import annotations

import argparse
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dfs_channel_v3_matplotlib"),
)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar


VERSION = "DFS-CHANNEL-COST-CALIBRATION-v3.3"


@dataclass(frozen=True)
class Config:
    exchange_J: float = 1.0
    true_gamma: float = 0.20
    target_duration: float = math.pi / 2.0

    calibration_shots_per_time: int = 50_000
    master_seed: int = 20260726
    coverage_repetitions: int = 256
    calibration_times: tuple[float, ...] = (
        0.20, 0.40, 0.65, 0.90, 1.20, 1.55, 1.95, 2.40
    )
    heldout_times: tuple[float, ...] = (0.30, 0.75, 1.35, 2.15)
    heldout_shots_per_time: int = 50_000

    selection_deltas: tuple[float, ...] = (-0.08, -0.04, 0.0, 0.04, 0.08)
    scaling_abs_deltas: tuple[float, ...] = (
        0.005, 0.008, 0.012, 0.018, 0.027, 0.040,
    )
    scaling_gammas: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.80)
    scaling_times: tuple[float, ...] = (0.50, 1.00, math.pi / 2.0, 2.00)

    zero_channel_tolerance: float = 3.0e-13
    positive_channel_minimum: float = 1.0e-8
    representation_invariance_tolerance: float = 3.0e-13
    choi_hermiticity_tolerance: float = 3.0e-13
    choi_cp_eigenvalue_tolerance: float = 3.0e-13
    choi_tp_tolerance: float = 3.0e-13
    gamma_relative_error_tolerance: float = 0.03
    heldout_visibility_rmse_tolerance: float = 0.012
    coverage_lower: float = 0.90
    coverage_upper: float = 0.99
    delta_exponent_lower: float = 1.98
    delta_exponent_upper: float = 2.02
    weak_coefficient_relative_tolerance: float = 0.01
    reflection_evenness_tolerance: float = 3.0e-12
    likelihood_bounds: tuple[float, float] = (1.0e-6, 2.0)


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


def sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(clean(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(clean(rows))


def create_unique_output_dir(requested: str | None) -> Path:
    base = Path(
        requested
        or f"dfs_channel_calibration_v3_3_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    candidates = [base]
    candidates.extend(
        base.with_name(f"{base.name}_run{index:02d}")
        for index in range(2, 1000)
    )
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            if candidate != base:
                print(
                    f"[output] directory exists: {base}\n"
                    f"[output] preserving it; this run uses: {candidate}"
                )
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate an output directory from {base}.")


def ket(index: int, dimension: int = 4) -> np.ndarray:
    state = np.zeros(dimension, dtype=complex)
    state[index] = 1.0
    return state


def build_operators(cfg: Config) -> dict[str, np.ndarray]:
    identity = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    z1 = np.kron(z, identity)
    z2 = np.kron(identity, z)
    h = 0.5 * cfg.exchange_J * (
        np.kron(x, x) + np.kron(y, y)
    )
    encoding = np.column_stack([ket(1), ket(2)])
    return {
        "H": h,
        "Z1": z1,
        "Z2": z2,
        "V": encoding,
    }


def liouvillian(
    hamiltonian: np.ndarray,
    jumps: list[np.ndarray],
) -> np.ndarray:
    """Column-major vectorization: vec(A rho B)=(B^T⊗A)vec(rho)."""
    dimension = hamiltonian.shape[0]
    identity = np.eye(dimension, dtype=complex)
    generator = -1.0j * (
        np.kron(identity, hamiltonian)
        - np.kron(hamiltonian.T, identity)
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
    return (
        superoperator @ rho.reshape(-1, order="F")
    ).reshape(dimension, dimension, order="F")


def channel_superoperator(
    hamiltonian: np.ndarray,
    jumps: list[np.ndarray],
    duration: float,
) -> np.ndarray:
    return expm(liouvillian(hamiltonian, jumps) * duration)


def encoded_choi(
    superoperator: np.ndarray,
    encoding: np.ndarray,
) -> np.ndarray:
    """Normalized Choi state for logical input dimension d=2."""
    logical_dimension = encoding.shape[1]
    physical_dimension = encoding.shape[0]
    choi = np.zeros(
        (
            logical_dimension * physical_dimension,
            logical_dimension * physical_dimension,
        ),
        dtype=complex,
    )
    for a in range(logical_dimension):
        for b in range(logical_dimension):
            logical_basis = np.zeros(
                (logical_dimension, logical_dimension), dtype=complex
            )
            logical_basis[a, b] = 1.0
            physical_basis = np.outer(
                encoding[:, a], encoding[:, b].conj()
            )
            output = apply_superoperator(superoperator, physical_basis)
            choi += np.kron(logical_basis, output) / logical_dimension
    # Return the raw matrix.  Diagnostics must see any anti-Hermitian defect;
    # silently symmetrizing here would hide propagation/reshaping bugs.
    return choi


def full_choi(superoperator: np.ndarray, dimension: int = 4) -> np.ndarray:
    return encoded_choi(superoperator, np.eye(dimension, dtype=complex))


def trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = 0.5 * ((left - right) + (left - right).conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


def partial_trace_output_choi(
    choi: np.ndarray,
    input_dimension: int,
    output_dimension: int,
) -> np.ndarray:
    """Trace normalized Choi state over output; TP target is I/d_in."""
    tensor = choi.reshape(
        input_dimension,
        output_dimension,
        input_dimension,
        output_dimension,
    )
    return np.einsum("iaja->ij", tensor)


def choi_cptp_diagnostics(
    choi: np.ndarray,
    input_dimension: int,
    output_dimension: int,
) -> dict[str, float]:
    hermiticity = float(
        np.linalg.norm(choi - choi.conj().T, ord="fro")
    )
    hermitian_copy = 0.5 * (choi + choi.conj().T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian_copy)))
    reduced = partial_trace_output_choi(
        choi, input_dimension, output_dimension
    )
    tp_target = np.eye(input_dimension, dtype=complex) / input_dimension
    tp_residual = float(np.linalg.norm(reduced - tp_target, ord="fro"))
    trace_residual = float(abs(np.trace(choi) - 1.0))
    return {
        "hermiticity_residual": hermiticity,
        "minimum_eigenvalue": minimum_eigenvalue,
        "trace_preservation_residual": tp_residual,
        "normalized_trace_residual": trace_residual,
    }


def jump_for_delta(
    gamma: float,
    delta: float,
    operators: dict[str, np.ndarray],
) -> np.ndarray:
    return math.sqrt(gamma) * (
        operators["Z1"] + (1.0 + delta) * operators["Z2"]
    )


def encoded_channel_cost(
    gamma: float,
    delta: float,
    duration: float,
    operators: dict[str, np.ndarray],
) -> float:
    h = operators["H"]
    ideal = channel_superoperator(h, [], duration)
    noisy = channel_superoperator(
        h, [jump_for_delta(gamma, delta, operators)], duration
    )
    ideal_choi = encoded_choi(ideal, operators["V"])
    noisy_choi = encoded_choi(noisy, operators["V"])
    return trace_distance(noisy_choi, ideal_choi)


def layer1_channel_audit(
    cfg: Config,
    operators: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    h = operators["H"]
    ideal = channel_superoperator(h, [], cfg.target_duration)
    jump_zero = jump_for_delta(cfg.true_gamma, 0.0, operators)
    noisy_zero = channel_superoperator(
        h, [jump_zero], cfg.target_duration
    )

    encoded_zero = trace_distance(
        encoded_choi(noisy_zero, operators["V"]),
        encoded_choi(ideal, operators["V"]),
    )
    full_space_cost = trace_distance(
        full_choi(noisy_zero),
        full_choi(ideal),
    )

    # Same dissipator represented by a phase-rotated single jump.
    phase = np.exp(1.0j * 0.731)
    noisy_phase = channel_superoperator(
        h, [phase * jump_zero], cfg.target_duration
    )
    phase_representation_change = float(
        np.linalg.norm(noisy_zero - noisy_phase, ord="fro")
        / noisy_zero.shape[0]
    )

    # Regression 2: mix two linearly independent jump operators.  This avoids
    # the degenerate v3.1 test that mixed two identical copies of one jump.
    distinct_jumps = [
        math.sqrt(cfg.true_gamma) * operators["Z1"],
        math.sqrt(0.37 * cfg.true_gamma) * operators["Z2"],
    ]
    mixing_theta = 0.417
    mixing_phase = 0.913
    c_mix = math.cos(mixing_theta)
    s_mix = math.sin(mixing_theta)
    unitary_mix = np.array(
        [
            [c_mix, np.exp(1.0j * mixing_phase) * s_mix],
            [-np.exp(-1.0j * mixing_phase) * s_mix, c_mix],
        ],
        dtype=complex,
    )
    mixed_jumps = [
        sum(
            (
                unitary_mix[a, b] * distinct_jumps[b]
                for b in range(2)
            ),
            np.zeros_like(jump_zero),
        )
        for a in range(2)
    ]
    noisy_distinct = channel_superoperator(
        h, distinct_jumps, cfg.target_duration
    )
    noisy_mixed = channel_superoperator(
        h, mixed_jumps, cfg.target_duration
    )
    unitary_mixing_change = float(
        np.linalg.norm(noisy_distinct - noisy_mixed, ord="fro")
        / noisy_zero.shape[0]
    )
    mixing_unitarity_residual = float(
        np.linalg.norm(
            unitary_mix.conj().T @ unitary_mix
            - np.eye(2, dtype=complex),
            ord="fro",
        )
    )

    # Regression 3: inhomogeneous Lindblad gauge freedom
    # L' = L + c I,
    # H' = H + (c* L - c L^dagger)/(2 i).
    # The Hamiltonian correction is essential and its sign follows the
    # master-equation convention implemented in liouvillian().
    gauge_c = 0.231 + 0.173j
    identity4 = np.eye(h.shape[0], dtype=complex)
    gauge_jump = jump_zero + gauge_c * identity4
    gauge_h = h + (
        gauge_c.conjugate() * jump_zero
        - gauge_c * jump_zero.conj().T
    ) / (2.0j)
    gauge_generator_difference = float(
        np.linalg.norm(
            liouvillian(h, [jump_zero])
            - liouvillian(gauge_h, [gauge_jump]),
            ord="fro",
        )
        / noisy_zero.shape[0]
    )
    noisy_gauge = channel_superoperator(
        gauge_h, [gauge_jump], cfg.target_duration
    )
    gauge_channel_difference = float(
        np.linalg.norm(noisy_zero - noisy_gauge, ord="fro")
        / noisy_zero.shape[0]
    )

    # CPTP diagnostics are performed on raw Choi matrices before any
    # Hermitian projection.  We test ideal/noisy encoded and full channels.
    choi_cases = {
        "encoded_ideal": (
            encoded_choi(ideal, operators["V"]), 2, 4
        ),
        "encoded_collective_noisy": (
            encoded_choi(noisy_zero, operators["V"]), 2, 4
        ),
        "full_ideal": (full_choi(ideal), 4, 4),
        "full_collective_noisy": (full_choi(noisy_zero), 4, 4),
    }
    cptp = {
        name: choi_cptp_diagnostics(choi, d_in, d_out)
        for name, (choi, d_in, d_out) in choi_cases.items()
    }
    cptp_gate = all(
        values["hermiticity_residual"]
        <= cfg.choi_hermiticity_tolerance
        and values["minimum_eigenvalue"]
        >= -cfg.choi_cp_eigenvalue_tolerance
        and values["trace_preservation_residual"]
        <= cfg.choi_tp_tolerance
        and values["normalized_trace_residual"]
        <= cfg.choi_tp_tolerance
        for values in cptp.values()
    )

    rows: list[dict[str, Any]] = []
    for delta in cfg.selection_deltas:
        cost = encoded_channel_cost(
            cfg.true_gamma, delta, cfg.target_duration, operators
        )
        rows.append({
            "delta": delta,
            "encoded_choi_trace_distance": cost,
            "data_role": "layer1_selection",
        })

    nonzero = [
        x["encoded_choi_trace_distance"]
        for x in rows
        if x["delta"] != 0.0
    ]
    zero_row = next(x for x in rows if x["delta"] == 0.0)
    model_gates = {
        "encoded_DFS_channel_exactly_ideal": (
            encoded_zero <= cfg.zero_channel_tolerance
            and zero_row["encoded_choi_trace_distance"]
            <= cfg.zero_channel_tolerance
        ),
        "full_physical_channel_not_globally_ideal": (
            full_space_cost >= cfg.positive_channel_minimum
        ),
        "symmetry_breaking_opens_encoded_channel_loss": all(
            x >= cfg.positive_channel_minimum for x in nonzero
        ),
        "raw_choi_channels_are_CPTP": cptp_gate,
    }
    regression_checks = {
        "jump_phase_representation_invariance": (
            phase_representation_change
            <= cfg.representation_invariance_tolerance
        ),
        "distinct_two_jump_unitary_mixing_regression": (
            unitary_mixing_change
            <= cfg.representation_invariance_tolerance
            and mixing_unitarity_residual
            <= cfg.representation_invariance_tolerance
        ),
        "inhomogeneous_lindblad_gauge_regression": (
            gauge_generator_difference
            <= cfg.representation_invariance_tolerance
            and gauge_channel_difference
            <= cfg.representation_invariance_tolerance
        ),
    }
    regression_checks_pass = all(regression_checks.values())
    return {
        "status": (
            "ENCODED_CHANNEL_ZERO_AND_CPTP_AUDIT_SUPPORTED"
            if all(model_gates.values())
            else "CHANNEL_LEVEL_DFS_AUDIT_FAILED"
        ),
        "regression_status": (
            "NUMERICAL_REPRESENTATION_REGRESSIONS_PASS"
            if regression_checks_pass
            else "NUMERICAL_REPRESENTATION_REGRESSION_WARNING"
        ),
        "cost_definition": (
            "E_ch=1/2||J(encoded noisy channel)"
            "-J(encoded ideal channel)||_1"
        ),
        "encoded_collective_point_cost": encoded_zero,
        "full_physical_space_collective_point_cost": full_space_cost,
        "equivalent_jump_phase_superoperator_difference":
            phase_representation_change,
        "distinct_two_jump_unitary_mixing_superoperator_difference":
            unitary_mixing_change,
        "jump_mixing_unitarity_residual": mixing_unitarity_residual,
        "inhomogeneous_gauge_generator_difference":
            gauge_generator_difference,
        "inhomogeneous_gauge_channel_difference":
            gauge_channel_difference,
        "raw_choi_CPTP_diagnostics": cptp,
        "model_gates": model_gates,
        "nonvoting_regression_checks": regression_checks,
        "nonvoting_regression_checks_pass": regression_checks_pass,
        "interpretation": (
            "The zero is an encoded-channel statement, not a claim that the "
            "full physical channel is noiseless. Representation changes are "
            "non-voting numerical regression tests because E_ch is defined "
            "from the channel itself."
        ),
    }, rows


def ramsey_visibility(gamma: float, time_value: float) -> float:
    # Coherence between |00> and |11>.  Their collective-jump eigenvalue
    # difference is 4 sqrt(gamma), hence exp[-(difference^2/2)t].
    return float(np.exp(-8.0 * gamma * time_value))


def ramsey_probability_from_liouvillian(
    gamma: float,
    time_value: float,
    operators: dict[str, np.ndarray],
) -> float:
    """Full-model Ramsey return probability with exchange H kept on."""
    psi = (ket(0) + ket(3)) / math.sqrt(2.0)
    rho0 = np.outer(psi, psi.conj())
    projector_plus = rho0
    channel = channel_superoperator(
        operators["H"],
        [jump_for_delta(gamma, 0.0, operators)],
        time_value,
    )
    rho_t = apply_superoperator(channel, rho0)
    probability = float(np.real(np.trace(projector_plus @ rho_t)))
    return float(np.clip(probability, 0.0, 1.0))


def generate_ramsey_rows(
    gamma_true: float,
    times: tuple[float, ...],
    shots: int,
    rng: np.random.Generator,
    role: str,
    operators: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for time_value in times:
        probability_plus = ramsey_probability_from_liouvillian(
            gamma_true, time_value, operators
        )
        visibility = 2.0 * probability_plus - 1.0
        analytic_visibility = ramsey_visibility(gamma_true, time_value)
        plus_counts = int(rng.binomial(shots, probability_plus))
        measured_visibility = 2.0 * plus_counts / shots - 1.0
        rows.append({
            "time": time_value,
            "shots": shots,
            "plus_counts": plus_counts,
            "minus_counts": shots - plus_counts,
            "measured_visibility": measured_visibility,
            "liouvillian_visibility": visibility,
            "analytic_visibility": analytic_visibility,
            "liouvillian_vs_analytic_error":
                abs(visibility - analytic_visibility),
            "data_role": role,
        })
    return rows


def fit_gamma_mle(
    rows: list[dict[str, Any]],
    bounds: tuple[float, float],
) -> tuple[float, float, dict[str, Any]]:
    """Fit gamma using calibration rows only."""
    def negative_log_likelihood(gamma: float) -> float:
        value = 0.0
        for row in rows:
            visibility = ramsey_visibility(gamma, row["time"])
            probability = np.clip(
                0.5 * (1.0 + visibility), 1.0e-12, 1.0 - 1.0e-12
            )
            plus = row["plus_counts"]
            minus = row["minus_counts"]
            value -= plus * math.log(probability)
            value -= minus * math.log(1.0 - probability)
        return value

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=bounds,
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"gamma MLE failed: {result.message}")
    gamma_hat = float(result.x)

    fisher = 0.0
    for row in rows:
        time_value = row["time"]
        visibility = ramsey_visibility(gamma_hat, time_value)
        probability = 0.5 * (1.0 + visibility)
        # p=(1+exp(-8 gamma t))/2, so dp/dgamma=-4 t visibility.
        derivative = -4.0 * time_value * visibility
        fisher += (
            row["shots"]
            * derivative**2
            / max(probability * (1.0 - probability), 1.0e-15)
        )
    standard_error = float(1.0 / math.sqrt(fisher))
    diagnostics = {
        "optimizer_success": bool(result.success),
        "negative_log_likelihood": float(result.fun),
        "expected_fisher_information_at_mle": fisher,
        "asymptotic_standard_error": standard_error,
        "asymptotic_95_percent_interval": [
            max(0.0, gamma_hat - 1.96 * standard_error),
            gamma_hat + 1.96 * standard_error,
        ],
    }
    return gamma_hat, standard_error, diagnostics


def layer2_synthetic_calibration(
    cfg: Config,
    operators: dict[str, np.ndarray],
    output: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    seed_sequence = np.random.SeedSequence(cfg.master_seed)
    primary_cal_seed, primary_heldout_seed, coverage_root = (
        seed_sequence.spawn(3)
    )
    calibration_rng = np.random.default_rng(primary_cal_seed)
    heldout_rng = np.random.default_rng(primary_heldout_seed)
    calibration_rows = generate_ramsey_rows(
        cfg.true_gamma,
        cfg.calibration_times,
        cfg.calibration_shots_per_time,
        calibration_rng,
        "calibration_fit",
        operators,
    )

    gamma_hat, standard_error, fit_diagnostics = fit_gamma_mle(
        calibration_rows, cfg.likelihood_bounds
    )
    lower_bound, upper_bound = cfg.likelihood_bounds
    mle_is_interior = (
        lower_bound * 1.001 < gamma_hat < upper_bound * 0.999
    )
    gamma_relative_error = abs(gamma_hat - cfg.true_gamma) / cfg.true_gamma

    # Freeze predictions on disk before the held-out generator is invoked.
    frozen_payload = {
        "version": VERSION,
        "master_seed": cfg.master_seed,
        "gamma_hat": gamma_hat,
        "heldout_times": list(cfg.heldout_times),
        "predicted_visibilities": [
            ramsey_visibility(gamma_hat, value)
            for value in cfg.heldout_times
        ],
        "statement": (
            "Written before held-out synthetic counts are generated."
        ),
    }
    frozen_path = output / "frozen_predictions_pretruth.json"
    save_json(frozen_path, frozen_payload)
    frozen_hash = sha256(frozen_path)

    heldout_rows = generate_ramsey_rows(
        cfg.true_gamma,
        cfg.heldout_times,
        cfg.heldout_shots_per_time,
        heldout_rng,
        "heldout_ramsey_test",
        operators,
    )
    heldout_visibility_errors: list[float] = []
    for row in heldout_rows:
        prediction = ramsey_visibility(gamma_hat, row["time"])
        row["predicted_visibility"] = prediction
        row["visibility_residual"] = (
            row["measured_visibility"] - prediction
        )
        heldout_visibility_errors.append(row["visibility_residual"])
    heldout_visibility_rmse = float(
        np.sqrt(np.mean(np.square(heldout_visibility_errors)))
    )

    # Empirical interval coverage and held-out calibration over independent
    # repetitions.  Calibration and held-out streams are separate per seed.
    coverage_rows: list[dict[str, Any]] = []
    covered_count = 0
    coverage_children = coverage_root.spawn(cfg.coverage_repetitions)
    for repetition, repetition_seed in enumerate(coverage_children):
        cal_seed, test_seed = repetition_seed.spawn(2)
        cal_rows = generate_ramsey_rows(
            cfg.true_gamma,
            cfg.calibration_times,
            cfg.calibration_shots_per_time,
            np.random.default_rng(cal_seed),
            "coverage_calibration",
            operators,
        )
        rep_hat, rep_se, rep_diag = fit_gamma_mle(
            cal_rows, cfg.likelihood_bounds
        )
        rep_interval = rep_diag["asymptotic_95_percent_interval"]
        covered = rep_interval[0] <= cfg.true_gamma <= rep_interval[1]
        covered_count += int(covered)
        test_rows = generate_ramsey_rows(
            cfg.true_gamma,
            cfg.heldout_times,
            cfg.heldout_shots_per_time,
            np.random.default_rng(test_seed),
            "coverage_heldout",
            operators,
        )
        residuals = [
            row["measured_visibility"]
            - ramsey_visibility(rep_hat, row["time"])
            for row in test_rows
        ]
        coverage_rows.append({
            "repetition": repetition,
            "gamma_hat": rep_hat,
            "gamma_standard_error": rep_se,
            "interval_low": rep_interval[0],
            "interval_high": rep_interval[1],
            "covers_true_gamma": covered,
            "gamma_relative_error":
                abs(rep_hat - cfg.true_gamma) / cfg.true_gamma,
            "heldout_visibility_rmse":
                float(np.sqrt(np.mean(np.square(residuals)))),
        })
    empirical_coverage = covered_count / cfg.coverage_repetitions
    mean_coverage_rmse = float(np.mean([
        row["heldout_visibility_rmse"] for row in coverage_rows
    ]))
    maximum_model_formula_error = max(
        row["liouvillian_vs_analytic_error"]
        for row in calibration_rows + heldout_rows
    )
    interval = fit_diagnostics["asymptotic_95_percent_interval"]
    gates = {
        "calibration_optimizer_succeeded":
            fit_diagnostics["optimizer_success"],
        "mle_is_strictly_inside_bounds": mle_is_interior,
        "full_liouvillian_matches_ramsey_formula": (
            maximum_model_formula_error <= 3.0e-13
        ),
        "gamma_recovered_from_calibration_only": (
            gamma_relative_error <= cfg.gamma_relative_error_tolerance
        ),
        "disjoint_heldout_ramsey_prediction": (
            heldout_visibility_rmse
            <= cfg.heldout_visibility_rmse_tolerance
        ),
        "empirical_95_percent_interval_coverage": (
            cfg.coverage_lower
            <= empirical_coverage
            <= cfg.coverage_upper
        ),
        "prediction_frozen_before_heldout_generation":
            frozen_hash is not None,
    }
    return {
        "status": (
            "SYNTHETIC_SEPARATED_CALIBRATION_AND_COVERAGE_SUPPORTED"
            if all(gates.values())
            else "SYNTHETIC_CALIBRATION_AUDIT_FAILED"
        ),
        "calibration_protocol": (
            "Full-Liouvillian counts for (|00>+|11>)/sqrt(2); one fitted "
            "parameter gamma in a declared response family; gamma_hat and "
            "held-out predictions written before held-out generation."
        ),
        "ramsey_state": "(|00>+|11>)/sqrt(2)",
        "ramsey_visibility_formula": "V(t)=exp(-8 gamma t), with H on",
        "gamma_true_used_only_by_synthetic_generator": cfg.true_gamma,
        "gamma_hat": gamma_hat,
        "gamma_relative_error": gamma_relative_error,
        "gamma_standard_error": standard_error,
        "fit_diagnostics": fit_diagnostics,
        "heldout_visibility_rmse": heldout_visibility_rmse,
        "maximum_liouvillian_vs_formula_error":
            maximum_model_formula_error,
        "coverage_repetitions": cfg.coverage_repetitions,
        "empirical_95_percent_interval_coverage": empirical_coverage,
        "mean_repeated_heldout_visibility_rmse": mean_coverage_rmse,
        "frozen_prediction_file": frozen_path.name,
        "frozen_prediction_sha256": frozen_hash,
        "gates": gates,
        "boundary": (
            "Generator and estimator are separated in software, but counts "
            "are synthetic. This is parameter-recovery and predictive-"
            "calibration evidence, not independent laboratory calibration "
            "or a blind model-discovery test."
        ),
    }, calibration_rows, heldout_rows, coverage_rows


def layer3_symmetry_breaking_scaling(
    cfg: Config,
    operators: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_slice: list[dict[str, Any]] = []
    maximum_evenness_error = 0.0
    for gamma in cfg.scaling_gammas:
        for duration in cfg.scaling_times:
            positive_costs: list[float] = []
            for delta in cfg.scaling_abs_deltas:
                positive = encoded_channel_cost(
                    gamma, delta, duration, operators
                )
                negative = encoded_channel_cost(
                    gamma, -delta, duration, operators
                )
                evenness = abs(positive - negative) / max(
                    positive, negative, 1.0e-15
                )
                maximum_evenness_error = max(
                    maximum_evenness_error, evenness
                )
                positive_costs.append(positive)
                rows.append({
                    "gamma": gamma,
                    "duration": duration,
                    "abs_delta": delta,
                    "positive_delta_cost": positive,
                    "negative_delta_cost": negative,
                    "reflection_evenness_relative_error": evenness,
                    "cost_over_gamma_delta2_time":
                        positive / (gamma * delta**2 * duration),
                    "data_role": "symmetry_breaking_scaling",
                })
            exponent, log_prefactor = np.polyfit(
                np.log(np.asarray(cfg.scaling_abs_deltas)),
                np.log(np.asarray(positive_costs)),
                1,
            )
            # The smallest delta is the declared weak-imbalance coefficient
            # check; larger deltas test finite-range exponent stability.
            weak_ratio = (
                positive_costs[0]
                / (
                    gamma
                    * cfg.scaling_abs_deltas[0] ** 2
                    * duration
                )
            )
            per_slice.append({
                "gamma": gamma,
                "duration": duration,
                "delta_loglog_exponent": float(exponent),
                "fitted_prefactor": float(np.exp(log_prefactor)),
                "small_delta_cost_over_gamma_delta2_time": weak_ratio,
                "small_delta_coefficient_relative_error":
                    abs(weak_ratio - 1.0),
            })

    exponents = [row["delta_loglog_exponent"] for row in per_slice]
    coefficient_errors = [
        row["small_delta_coefficient_relative_error"]
        for row in per_slice
    ]
    gates = {
        "delta_squared_exponent_on_all_gamma_time_slices": all(
            cfg.delta_exponent_lower <= value <= cfg.delta_exponent_upper
            for value in exponents
        ),
        "weak_imbalance_coefficient_matches_gamma_delta2_time": (
            max(coefficient_errors)
            <= cfg.weak_coefficient_relative_tolerance
        ),
        "delta_reflection_evenness": (
            maximum_evenness_error
            <= cfg.reflection_evenness_tolerance
        ),
        "gamma_time_grid_is_nontrivial": (
            len(cfg.scaling_gammas) >= 3
            and len(cfg.scaling_times) >= 3
        ),
    }
    return {
        "status": (
            "ENCODED_CHANNEL_DELTA_SQUARED_OPENING_SUPPORTED"
            if all(gates.values())
            else "SYMMETRY_BREAKING_SCALING_AUDIT_FAILED"
        ),
        "tested_relation": (
            "E_ch(delta)=gamma*delta^2*t+o(delta^2)"
        ),
        "gamma_grid": cfg.scaling_gammas,
        "duration_grid": cfg.scaling_times,
        "absolute_delta_grid": cfg.scaling_abs_deltas,
        "minimum_delta_loglog_exponent": min(exponents),
        "maximum_delta_loglog_exponent": max(exponents),
        "maximum_weak_coefficient_relative_error":
            max(coefficient_errors),
        "maximum_delta_reflection_evenness_relative_error":
            maximum_evenness_error,
        "slice_fits": per_slice,
        "gates": gates,
    }, rows


def save_plot(
    path: Path,
    selection_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
    gamma_hat: float,
) -> str | None:
    figure = None
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
        axes[0].plot(
            [x["delta"] for x in selection_rows],
            [x["encoded_choi_trace_distance"] for x in selection_rows],
            "o-",
        )
        axes[0].set_title("Encoded channel cost")
        axes[0].set_xlabel(r"$\delta$")
        axes[0].set_ylabel("Choi trace distance")

        all_times = np.linspace(
            0.0,
            max(
                max(x["time"] for x in calibration_rows),
                max(x["time"] for x in heldout_rows),
            ),
            200,
        )
        axes[1].plot(
            all_times,
            [ramsey_visibility(gamma_hat, t) for t in all_times],
            label="frozen fit",
        )
        axes[1].scatter(
            [x["time"] for x in calibration_rows],
            [x["measured_visibility"] for x in calibration_rows],
            label="calibration",
        )
        axes[1].scatter(
            [x["time"] for x in heldout_rows],
            [x["measured_visibility"] for x in heldout_rows],
            marker="x",
            label="held out",
        )
        axes[1].set_title("Separated synthetic calibration")
        axes[1].set_xlabel("time")
        axes[1].set_ylabel("visibility")
        axes[1].legend()

        selected_scaling = [
            row for row in scaling_rows
            if abs(row["gamma"] - 0.20) < 1.0e-15
            and abs(row["duration"] - math.pi / 2.0) < 1.0e-15
        ]
        axes[2].loglog(
            [x["abs_delta"] for x in selected_scaling],
            [x["positive_delta_cost"] for x in selected_scaling],
            "o-",
            label=r"exact $E_{\rm ch}$",
        )
        axes[2].loglog(
            [x["abs_delta"] for x in selected_scaling],
            [
                x["gamma"] * x["abs_delta"] ** 2 * x["duration"]
                for x in selected_scaling
            ],
            "--",
            label=r"$\gamma\delta^2t$",
        )
        axes[2].set_title("Symmetry-breaking opening")
        axes[2].set_xlabel(r"$|\delta|$")
        axes[2].set_ylabel("encoded Choi witness")
        axes[2].legend()

        for axis in axes:
            axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(path, dpi=180)
        return path.name
    except Exception as exc:
        print(
            "[plot] non-voting plot failed; certificates remain valid: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    finally:
        if figure is not None:
            try:
                plt.close(figure)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    raw = sys.argv[1:]
    cleaned: list[str] = []
    ignored: list[str] = []
    index = 0
    while index < len(raw):
        if raw[index] == "-f" and index + 1 < len(raw):
            ignored.extend(raw[index:index + 2])
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
            "script_sha256": sha256(script_path),
        },
    }
    save_json(output / "summary.json", summary)

    print("\n" + "=" * 100)
    print("DFS CHANNEL-COST + SYNTHETIC CALIBRATION v3.3")
    print("=" * 100)
    print("backend=exact NumPy/SciPy | cloud access=none")

    try:
        cfg = Config()
        operators = build_operators(cfg)

        print("\n[LAYER 1] Encoded-channel witness, CPTP, regressions")
        layer1, selection_rows = layer1_channel_audit(cfg, operators)
        print(json.dumps(clean(layer1), indent=2, ensure_ascii=False))

        print("\n[LAYER 2] Full-Liouvillian synthetic calibration")
        (
            layer2,
            calibration_rows,
            heldout_rows,
            coverage_rows,
        ) = layer2_synthetic_calibration(cfg, operators, output)
        print(json.dumps(clean(layer2), indent=2, ensure_ascii=False))

        print("\n[LAYER 3] Multi-gamma/time delta-squared opening")
        layer3, scaling_rows = layer3_symmetry_breaking_scaling(
            cfg, operators
        )
        print(json.dumps(clean(layer3), indent=2, ensure_ascii=False))

        global_gates = {
            "encoded_channel_zero_and_CPTP":
                layer1["status"]
                == "ENCODED_CHANNEL_ZERO_AND_CPTP_AUDIT_SUPPORTED",
            "synthetic_separated_calibration_and_coverage":
                layer2["status"]
                == (
                    "SYNTHETIC_SEPARATED_CALIBRATION_AND_"
                    "COVERAGE_SUPPORTED"
                ),
            "delta_squared_symmetry_breaking_opening":
                layer3["status"]
                == "ENCODED_CHANNEL_DELTA_SQUARED_OPENING_SUPPORTED",
        }
        declared_model_support = all(global_gates.values())
        nonvoting_regression_checks_pass = bool(
            layer1["nonvoting_regression_checks_pass"]
        )
        scientific_status = (
            "DFS_CHANNEL_ZERO_CALIBRATION_AND_SCALING_SUPPORTED"
            if declared_model_support
            else "DFS_CHANNEL_CALIBRATION_AUDIT_FAILED"
        )
        certificate = {
            "version": VERSION,
            "scientific_status": scientific_status,
            "declared_model_support": declared_model_support,
            "nonvoting_regression_checks_pass":
                nonvoting_regression_checks_pass,
            "frozen_config": asdict(cfg),
            "layer1_channel_cost": layer1,
            "layer2_synthetic_calibration": layer2,
            "layer3_symmetry_breaking_scaling": layer3,
            "global_gates": global_gates,
            "claim_boundary": (
                "The normalized encoded Choi-state trace-distance zero is a "
                "channel-level witness, not a diamond-distance estimate. "
                "Raw Choi CPTP conditions are audited. Representation changes "
                "are regression tests, not physical evidence. Gamma is the "
                "only fitted parameter in a declared synthetic Ramsey family; "
                "predictions are frozen before held-out generation. The "
                "delta-squared opening is tested over multiple gamma and time "
                "values. No QPU, laboratory calibration, zero-total-energy, "
                "universal realizability, or Lorentzian claim is made."
            ),
        }

        save_json(output / "dfs_channel_v3_3_certificate.json", certificate)
        save_csv(output / "layer1_channel_selection.csv", selection_rows)
        save_csv(output / "calibration_ramsey_counts.csv", calibration_rows)
        save_csv(output / "heldout_ramsey_counts.csv", heldout_rows)
        save_csv(output / "coverage_repetitions.csv", coverage_rows)
        save_csv(output / "symmetry_breaking_scaling.csv", scaling_rows)
        figure = save_plot(
            output / "dfs_channel_v3_3_diagnostic.png",
            selection_rows,
            calibration_rows,
            heldout_rows,
            scaling_rows,
            layer2["gamma_hat"],
        )

        summary.update({
            "status": "COMPLETE",
            "scientific_status": scientific_status,
            "declared_model_support": declared_model_support,
            "nonvoting_regression_checks_pass":
                nonvoting_regression_checks_pass,
            "outputs": {
                "certificate": "dfs_channel_v3_3_certificate.json",
                "layer1": "layer1_channel_selection.csv",
                "calibration": "calibration_ramsey_counts.csv",
                "heldout_ramsey": "heldout_ramsey_counts.csv",
                "coverage": "coverage_repetitions.csv",
                "scaling": "symmetry_breaking_scaling.csv",
                "frozen_predictions":
                    "frozen_predictions_pretruth.json",
                "figure": figure,
            },
        })

        print("\n" + "=" * 100)
        print("GLOBAL VERDICT")
        print("=" * 100)
        print(json.dumps(clean({
            "scientific_status": scientific_status,
            "declared_model_support": declared_model_support,
            "nonvoting_regression_checks_pass":
                nonvoting_regression_checks_pass,
            "global_gates": global_gates,
            "claim_boundary": certificate["claim_boundary"],
        }), indent=2, ensure_ascii=False))
        if not declared_model_support:
            raise AssertionError(
                "At least one frozen v3.3 model gate failed; inspect "
                "certificate."
            )
    except Exception as exc:
        summary.update({
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        save_json(output / "summary.json", summary)
        print(f"elapsed={summary['elapsed_seconds']:.2f}s")
        print(f"outputs={output}")


if __name__ == "__main__":
    main()
