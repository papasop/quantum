#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFS CHANNEL-COST + BLIND-CALIBRATION v3
=======================================

Layer 1 — representation-invariant channel cost
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
  * the channel result is invariant under a unitary mixing/phase change of an
    equivalent jump-operator representation.

Layer 2 — independently separated synthetic calibration
--------------------------------------------------------
The true gamma is used only by a data generator.  A calibration module sees
Ramsey counts from a non-DFS coherence and estimates gamma by binomial maximum
likelihood.  The fitted gamma is then frozen and used to predict:

  * disjoint held-out Ramsey times;
  * encoded Choi losses for an unseen delta grid.

The prediction module is passed gamma_hat only; it does not receive gamma_true.

Boundary
--------
This is exact NumPy/SciPy model evidence plus a finite-shot synthetic blind
calibration.  It is not QPU data, not an experimentally calibrated cost, not
zero total energy, and not a universal realizability or Lorentzian theorem.

Run:
    pip install -U numpy scipy matplotlib
    python dfs_channel_cost_calibration_v3_1.py

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


VERSION = "DFS-CHANNEL-COST-CALIBRATION-v3.1"


@dataclass(frozen=True)
class Config:
    exchange_J: float = 1.0
    true_gamma: float = 0.20
    target_duration: float = math.pi / 2.0

    calibration_shots_per_time: int = 50_000
    calibration_seed: int = 20260726
    calibration_times: tuple[float, ...] = (
        0.20, 0.40, 0.65, 0.90, 1.20, 1.55, 1.95, 2.40
    )
    heldout_times: tuple[float, ...] = (0.30, 0.75, 1.35, 2.15)
    heldout_shots_per_time: int = 50_000

    selection_deltas: tuple[float, ...] = (-0.08, -0.04, 0.0, 0.04, 0.08)
    heldout_deltas: tuple[float, ...] = (
        -0.12, -0.06, -0.03, -0.015,
        0.015, 0.03, 0.06, 0.12,
    )

    zero_channel_tolerance: float = 3.0e-13
    positive_channel_minimum: float = 1.0e-8
    representation_invariance_tolerance: float = 3.0e-13
    gamma_relative_error_tolerance: float = 0.03
    heldout_visibility_rmse_tolerance: float = 0.012
    heldout_channel_max_relative_error_tolerance: float = 0.06
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
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(rows))


def create_unique_output_dir(requested: str | None) -> Path:
    base = Path(
        requested
        or f"dfs_channel_calibration_v3_1_{time.strftime('%Y%m%d_%H%M%S')}"
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
    return 0.5 * (choi + choi.conj().T)


def full_choi(superoperator: np.ndarray, dimension: int = 4) -> np.ndarray:
    return encoded_choi(superoperator, np.eye(dimension, dtype=complex))


def trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = 0.5 * ((left - right) + (left - right).conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(difference))))


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

    # Embed the same dissipator into two equivalent jump components and apply
    # a nontrivial 2x2 unitary mixing.  Lindblad sums are invariant under
    # L'_a = sum_b U_ab L_b for unitary U.
    split_jumps = [jump_zero / math.sqrt(2.0)] * 2
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
                unitary_mix[a, b] * split_jumps[b]
                for b in range(2)
            ),
            np.zeros_like(jump_zero),
        )
        for a in range(2)
    ]
    noisy_split = channel_superoperator(
        h, split_jumps, cfg.target_duration
    )
    noisy_mixed = channel_superoperator(
        h, mixed_jumps, cfg.target_duration
    )
    split_matches_single = float(
        np.linalg.norm(noisy_zero - noisy_split, ord="fro")
        / noisy_zero.shape[0]
    )
    unitary_mixing_change = float(
        np.linalg.norm(noisy_split - noisy_mixed, ord="fro")
        / noisy_zero.shape[0]
    )
    mixing_unitarity_residual = float(
        np.linalg.norm(
            unitary_mix.conj().T @ unitary_mix
            - np.eye(2, dtype=complex),
            ord="fro",
        )
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
    gates = {
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
        "jump_phase_representation_invariance": (
            phase_representation_change
            <= cfg.representation_invariance_tolerance
        ),
        "split_jump_representation_matches_single_jump": (
            split_matches_single
            <= cfg.representation_invariance_tolerance
        ),
        "two_jump_unitary_mixing_invariance": (
            unitary_mixing_change
            <= cfg.representation_invariance_tolerance
            and mixing_unitarity_residual
            <= cfg.representation_invariance_tolerance
        ),
    }
    return {
        "status": (
            "REPRESENTATION_INVARIANT_ENCODED_CHANNEL_ZERO_SUPPORTED"
            if all(gates.values())
            else "CHANNEL_LEVEL_DFS_AUDIT_FAILED"
        ),
        "cost_definition": (
            "E_ch=1/2||J(encoded noisy channel)"
            "-J(encoded ideal channel)||_1"
        ),
        "encoded_collective_point_cost": encoded_zero,
        "full_physical_space_collective_point_cost": full_space_cost,
        "equivalent_jump_phase_superoperator_difference":
            phase_representation_change,
        "equivalent_split_vs_single_superoperator_difference":
            split_matches_single,
        "equivalent_two_jump_unitary_mixing_superoperator_difference":
            unitary_mixing_change,
        "jump_mixing_unitarity_residual": mixing_unitarity_residual,
        "gates": gates,
        "interpretation": (
            "The zero is an encoded-channel statement, not a claim that the "
            "full physical channel is noiseless."
        ),
    }, rows


def ramsey_visibility(gamma: float, time_value: float) -> float:
    # Coherence between L eigenvalues 2 sqrt(gamma) and 0.
    return float(np.exp(-2.0 * gamma * time_value))


def generate_ramsey_rows(
    gamma_true: float,
    times: tuple[float, ...],
    shots: int,
    rng: np.random.Generator,
    role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for time_value in times:
        visibility = ramsey_visibility(gamma_true, time_value)
        probability_plus = 0.5 * (1.0 + visibility)
        plus_counts = int(rng.binomial(shots, probability_plus))
        measured_visibility = 2.0 * plus_counts / shots - 1.0
        rows.append({
            "time": time_value,
            "shots": shots,
            "plus_counts": plus_counts,
            "minus_counts": shots - plus_counts,
            "measured_visibility": measured_visibility,
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
        derivative = -time_value * visibility
        fisher += (
            row["shots"]
            * derivative**2
            / max(probability * (1.0 - probability), 1.0e-15)
        )
    standard_error = float(1.0 / math.sqrt(fisher))
    diagnostics = {
        "optimizer_success": bool(result.success),
        "negative_log_likelihood": float(result.fun),
        "observed_fisher_information": fisher,
        "asymptotic_standard_error": standard_error,
        "asymptotic_95_percent_interval": [
            max(0.0, gamma_hat - 1.96 * standard_error),
            gamma_hat + 1.96 * standard_error,
        ],
    }
    return gamma_hat, standard_error, diagnostics


def predict_heldout_channels(
    gamma_hat: float,
    deltas: tuple[float, ...],
    duration: float,
    operators: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Prediction interface deliberately receives gamma_hat, not gamma_true."""
    rows: list[dict[str, Any]] = []
    for delta in deltas:
        rows.append({
            "delta": delta,
            "predicted_encoded_choi_trace_distance":
                encoded_channel_cost(gamma_hat, delta, duration, operators),
            "data_role": "heldout_channel_prediction",
        })
    return rows


def layer2_blind_calibration(
    cfg: Config,
    operators: dict[str, np.ndarray],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rng = np.random.default_rng(cfg.calibration_seed)
    calibration_rows = generate_ramsey_rows(
        cfg.true_gamma,
        cfg.calibration_times,
        cfg.calibration_shots_per_time,
        rng,
        "calibration_fit",
    )
    heldout_rows = generate_ramsey_rows(
        cfg.true_gamma,
        cfg.heldout_times,
        cfg.heldout_shots_per_time,
        rng,
        "heldout_ramsey_test",
    )

    gamma_hat, standard_error, fit_diagnostics = fit_gamma_mle(
        calibration_rows, cfg.likelihood_bounds
    )
    gamma_relative_error = abs(gamma_hat - cfg.true_gamma) / cfg.true_gamma

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

    predicted_channels = predict_heldout_channels(
        gamma_hat,
        cfg.heldout_deltas,
        cfg.target_duration,
        operators,
    )
    channel_relative_errors: list[float] = []
    for row in predicted_channels:
        # Truth is revealed only after predictions have been constructed.
        truth = encoded_channel_cost(
            cfg.true_gamma,
            row["delta"],
            cfg.target_duration,
            operators,
        )
        row["truth_encoded_choi_trace_distance"] = truth
        row["absolute_error"] = abs(
            row["predicted_encoded_choi_trace_distance"] - truth
        )
        row["relative_error"] = row["absolute_error"] / max(truth, 1.0e-15)
        channel_relative_errors.append(row["relative_error"])

    maximum_channel_relative_error = max(channel_relative_errors)
    interval = fit_diagnostics["asymptotic_95_percent_interval"]
    gates = {
        "calibration_optimizer_succeeded":
            fit_diagnostics["optimizer_success"],
        "gamma_recovered_from_calibration_only": (
            gamma_relative_error <= cfg.gamma_relative_error_tolerance
        ),
        "true_gamma_inside_asymptotic_95_percent_interval": (
            interval[0] <= cfg.true_gamma <= interval[1]
        ),
        "disjoint_heldout_ramsey_prediction": (
            heldout_visibility_rmse
            <= cfg.heldout_visibility_rmse_tolerance
        ),
        "frozen_gamma_predicts_unseen_channel_grid": (
            maximum_channel_relative_error
            <= cfg.heldout_channel_max_relative_error_tolerance
        ),
    }
    return {
        "status": (
            "SYNTHETIC_BLIND_CALIBRATION_AND_HELDOUT_PREDICTION_SUPPORTED"
            if all(gates.values())
            else "BLIND_CALIBRATION_AUDIT_FAILED"
        ),
        "calibration_protocol": (
            "Binomial MLE on non-DFS Ramsey counts; gamma_hat frozen before "
            "held-out Ramsey and encoded-channel evaluation."
        ),
        "gamma_true_hidden_from_fit": cfg.true_gamma,
        "gamma_hat": gamma_hat,
        "gamma_relative_error": gamma_relative_error,
        "gamma_standard_error": standard_error,
        "fit_diagnostics": fit_diagnostics,
        "heldout_visibility_rmse": heldout_visibility_rmse,
        "maximum_heldout_channel_relative_error":
            maximum_channel_relative_error,
        "gates": gates,
        "boundary": (
            "The separation is enforced in software, but the counts are "
            "synthetic; this is not independent laboratory calibration."
        ),
    }, calibration_rows, heldout_rows, predicted_channels


def save_plot(
    path: Path,
    selection_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
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
        axes[1].set_title("Blind Ramsey calibration")
        axes[1].set_xlabel("time")
        axes[1].set_ylabel("visibility")
        axes[1].legend()

        axes[2].plot(
            [x["delta"] for x in channel_rows],
            [
                x["truth_encoded_choi_trace_distance"]
                for x in channel_rows
            ],
            "o-",
            label="hidden truth",
        )
        axes[2].plot(
            [x["delta"] for x in channel_rows],
            [
                x["predicted_encoded_choi_trace_distance"]
                for x in channel_rows
            ],
            "x--",
            label="frozen prediction",
        )
        axes[2].set_title("Unseen channel grid")
        axes[2].set_xlabel(r"$\delta$")
        axes[2].set_ylabel("Choi trace distance")
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
    print("DFS CHANNEL-COST + SYNTHETIC BLIND-CALIBRATION v3.1")
    print("=" * 100)
    print("backend=exact NumPy/SciPy | cloud access=none")

    try:
        cfg = Config()
        operators = build_operators(cfg)

        print("\n[LAYER 1] Representation-invariant encoded-channel cost")
        layer1, selection_rows = layer1_channel_audit(cfg, operators)
        print(json.dumps(clean(layer1), indent=2, ensure_ascii=False))

        print("\n[LAYER 2] Calibration-only fit and frozen held-out prediction")
        (
            layer2,
            calibration_rows,
            heldout_rows,
            channel_rows,
        ) = layer2_blind_calibration(cfg, operators)
        print(json.dumps(clean(layer2), indent=2, ensure_ascii=False))

        global_gates = {
            "representation_invariant_channel_cost":
                layer1["status"]
                == "REPRESENTATION_INVARIANT_ENCODED_CHANNEL_ZERO_SUPPORTED",
            "synthetic_blind_calibration_and_heldout_prediction":
                layer2["status"]
                == (
                    "SYNTHETIC_BLIND_CALIBRATION_AND_"
                    "HELDOUT_PREDICTION_SUPPORTED"
                ),
        }
        physical_support = all(global_gates.values())
        scientific_status = (
            "DFS_CHANNEL_ZERO_AND_BLIND_CALIBRATION_SUPPORTED"
            if physical_support
            else "DFS_CHANNEL_CALIBRATION_AUDIT_FAILED"
        )
        certificate = {
            "version": VERSION,
            "scientific_status": scientific_status,
            "physical_support": physical_support,
            "frozen_config": asdict(cfg),
            "layer1_channel_cost": layer1,
            "layer2_blind_calibration": layer2,
            "global_gates": global_gates,
            "claim_boundary": (
                "The encoded Choi-distance zero is a channel-level statement "
                "and is unchanged under the tested equivalent single-jump "
                "phase and two-jump unitary-mixing representations. Gamma is "
                "estimated from disjoint finite-shot synthetic Ramsey data and "
                "then frozen for held-out predictions. No QPU, laboratory "
                "calibration, zero-total-energy, universal realizability, or "
                "Lorentzian claim is made."
            ),
        }

        save_json(output / "dfs_channel_v3_certificate.json", certificate)
        save_csv(output / "layer1_channel_selection.csv", selection_rows)
        save_csv(output / "calibration_ramsey_counts.csv", calibration_rows)
        save_csv(output / "heldout_ramsey_counts.csv", heldout_rows)
        save_csv(output / "heldout_channel_predictions.csv", channel_rows)
        figure = save_plot(
            output / "dfs_channel_v3_diagnostic.png",
            selection_rows,
            calibration_rows,
            heldout_rows,
            channel_rows,
            layer2["gamma_hat"],
        )

        summary.update({
            "status": "COMPLETE",
            "scientific_status": scientific_status,
            "physical_support": physical_support,
            "outputs": {
                "certificate": "dfs_channel_v3_certificate.json",
                "layer1": "layer1_channel_selection.csv",
                "calibration": "calibration_ramsey_counts.csv",
                "heldout_ramsey": "heldout_ramsey_counts.csv",
                "heldout_channels": "heldout_channel_predictions.csv",
                "figure": figure,
            },
        })

        print("\n" + "=" * 100)
        print("GLOBAL VERDICT")
        print("=" * 100)
        print(json.dumps(clean({
            "scientific_status": scientific_status,
            "physical_support": physical_support,
            "global_gates": global_gates,
            "claim_boundary": certificate["claim_boundary"],
        }), indent=2, ensure_ascii=False))
        if not physical_support:
            raise AssertionError(
                "At least one frozen v3 gate failed; inspect certificate."
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
