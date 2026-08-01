#!/usr/bin/env python3
"""Stim/PyMatching rotated-surface-code scaling-regime calibration v0.5.2.

The ideal task is one protected logical Z-memory interval in a distance-d
rotated surface code. Every schedule has the same ideal identity endpoint. A
projected flow optimises a frozen identity-layer pulse-risk score.
Initial and optimised schedule-dependent faults are then inserted into a Stim
rotated-surface-code syndrome-extraction circuit containing reset, measurement,
idle, and two-qubit Clifford noise.  PyMatching is built independently from
each circuit's detector error model, and held-out detector samples decide the
claim.

The primary endpoint is no longer a percentage reduction at fixed distance.
For a frozen target logical error, a one-sided Wilson upper confidence bound
decides the smallest code distance that closes the target.  The claim passes
only if fibre optimisation lowers that minimum distance by one odd-distance
step on the preregistered seed fraction.

This is a development-only calibration of a lower-noise, rounds-equal-distance
regime. It scans candidate fixed-reliability targets and distance slopes before
any v0.6 prospective protocol is frozen. Its seeds and selected target are
ineligible for a confirmatory claim. It is not a threshold theorem, formal
interval certificate, compiler result, or hardware validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TITLE = "FAULT-TOLERANT RESPONSE-FIBRE ROTATED-SURFACE-CODE SCALING-REGIME CALIBRATION"
VERSION = "0.5.2"
CALIBRATION_SEEDS = (20281107, 20281119, 20281203)
DEVELOPMENT_SEEDS_EXCLUDED = (
    20280107, 20280119, 20280203,
    20280307, 20280319, 20280403,
    20280507, 20280519, 20280603,
    20280707, 20280719, 20280803,
    20280907, 20280919, 20281003,
    20281107, 20281119, 20281203,
)


def canonical_json(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_obj(x: Any) -> str:
    return hashlib.sha256(canonical_json(x)).hexdigest()


def load_qec_packages(auto_install: bool):
    try:
        stim = importlib.import_module("stim")
        pymatching = importlib.import_module("pymatching")
        return stim, pymatching
    except ModuleNotFoundError:
        if not auto_install:
            raise RuntimeError(
                "Stim/PyMatching are required. Run: pip install 'stim>=1.15,<2' "
                "'pymatching>=2.3,<3'"
            )
    print("[setup] installing Stim and PyMatching into the active Python environment...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "stim>=1.15,<2",
            "pymatching>=2.3,<3",
        ]
    )
    importlib.invalidate_caches()
    return importlib.import_module("stim"), importlib.import_module("pymatching")


@dataclass
class Hardware:
    drive: np.ndarray
    xtalk: float


def make_hardware(n_data: int, segments: int, seed: int, distance: int) -> Hardware:
    rng = np.random.default_rng(seed + 1009 * distance)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=(n_data, 1))
    t = (np.arange(segments) + 0.5)[None, :] / segments
    drive = 0.024 * (
        1.18
        + 0.52 * np.sin(2.0 * math.pi * t + phase)
        + 0.22 * np.cos(4.0 * math.pi * t - 0.7 * phase)
    )
    drive *= rng.uniform(0.80, 1.20, size=(n_data, 1))
    return Hardware(drive=np.maximum(drive, 0.003), xtalk=0.0065)


def project_row_to_box_and_mean(
    row: np.ndarray,
    minimum_amplitude: float,
    maximum_amplitude: float,
    target_mean: float = 1.0,
) -> np.ndarray:
    """Euclidean projection onto {lo <= x <= hi, mean(x)=target_mean}."""
    if not minimum_amplitude <= target_mean <= maximum_amplitude:
        raise ValueError("target mean is outside the amplitude box")
    row = np.asarray(row, dtype=float)
    lower_shift = minimum_amplitude - float(np.max(row)) - 1.0
    upper_shift = maximum_amplitude - float(np.min(row)) + 1.0
    target_sum = target_mean * row.size
    for _ in range(100):
        middle = 0.5 * (lower_shift + upper_shift)
        candidate = np.clip(row + middle, minimum_amplitude, maximum_amplitude)
        if float(np.sum(candidate)) < target_sum:
            lower_shift = middle
        else:
            upper_shift = middle
    projected = np.clip(
        row + 0.5 * (lower_shift + upper_shift),
        minimum_amplitude,
        maximum_amplitude,
    )
    residual = target_sum - float(np.sum(projected))
    free = (projected > minimum_amplitude + 1e-14) & (
        projected < maximum_amplitude - 1e-14
    )
    if np.any(free):
        projected[free] += residual / int(np.count_nonzero(free))
    if abs(float(np.mean(projected)) - target_mean) > 5e-14:
        raise ArithmeticError("box/mean projection did not close")
    if projected.min() < minimum_amplitude - 1e-14 or projected.max() > maximum_amplitude + 1e-14:
        raise ArithmeticError("box/mean projection escaped the amplitude box")
    return projected


def initial_schedule(
    n_data: int,
    segments: int,
    seed: int,
    distance: int,
    minimum_amplitude: float,
    maximum_amplitude: float,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed + 7919 * distance)
    raw = np.exp(rng.normal(0.0, 0.44, size=(n_data, segments)))
    normalized = raw / raw.mean(axis=1, keepdims=True)
    projected = np.vstack(
        [
            project_row_to_box_and_mean(
                row, minimum_amplitude, maximum_amplitude, 1.0
            )
            for row in normalized
        ]
    )
    diagnostics = {
        "raw_normalized_minimum": float(np.min(normalized)),
        "raw_normalized_maximum": float(np.max(normalized)),
        "projected_minimum": float(np.min(projected)),
        "projected_maximum": float(np.max(projected)),
        "maximum_mean_residual": float(
            np.max(np.abs(projected.mean(axis=1) - 1.0))
        ),
        "projection_displacement_norm": float(np.linalg.norm(projected - normalized)),
    }
    return projected, diagnostics


def pulse_risk_and_gradient(
    u: np.ndarray, hw: Hardware, edges: list[tuple[int, int]]
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    _, segments = u.shape
    dt = 1.0 / segments
    exposure = dt * np.sum(hw.drive * u * u, axis=1)
    pair_exposure = np.array(
        [dt * hw.xtalk * np.sum(u[i] * u[j]) for i, j in edges]
    )
    single_p = 0.5 * (1.0 - np.exp(-2.0 * exposure))
    pair_p = 1.0 - np.exp(-pair_exposure)
    score = float(np.sum(single_p) + 2.5 * np.sum(pair_p))

    grad = dt * (2.0 * hw.drive * u) * np.exp(-2.0 * exposure)[:, None]
    for k, (i, j) in enumerate(edges):
        coeff = 2.5 * math.exp(-pair_exposure[k]) * dt * hw.xtalk
        grad[i] += coeff * u[j]
        grad[j] += coeff * u[i]
    return score, grad, single_p, pair_p


def optimise(
    u0: np.ndarray,
    hw: Hardware,
    edges: list[tuple[int, int]],
    iterations: int,
    step_radius: float,
    min_amp: float,
    max_amp: float,
) -> tuple[np.ndarray, list[dict[str, float]], dict[str, Any]]:
    u = u0.copy()
    history: list[dict[str, float]] = []
    accepted_steps = 0
    all_checkpoints_inside_box = bool(
        u.min() >= min_amp - 1e-14 and u.max() <= max_amp + 1e-14
    )
    for k in range(iterations + 1):
        value, grad, _, _ = pulse_risk_and_gradient(u, hw, edges)
        if k % max(1, iterations // 20) == 0 or k == iterations:
            history.append({"iteration": k, "design_score": value})
        if k == iterations:
            break
        direction = -(grad - grad.mean(axis=1, keepdims=True))
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm):
            raise ArithmeticError("projected gradient is non-finite")
        if norm <= 1e-15:
            break
        direction /= norm
        directional = float(np.sum(grad * direction))
        alpha = step_radius
        accepted = False
        for _ in range(40):
            trial = u + alpha * direction
            if trial.min() >= min_amp and trial.max() <= max_amp:
                trial_value, _, _, _ = pulse_risk_and_gradient(trial, hw, edges)
                if trial_value <= value + 1e-4 * alpha * directional:
                    u = trial
                    accepted_steps += 1
                    all_checkpoints_inside_box = all_checkpoints_inside_box and bool(
                        u.min() >= min_amp - 1e-14
                        and u.max() <= max_amp + 1e-14
                    )
                    accepted = True
                    break
            alpha *= 0.5
        if not accepted:
            break
    return u, history, {
        "accepted_steps": accepted_steps,
        "all_checkpoints_inside_amplitude_box": all_checkpoints_inside_box,
        "final_minimum_amplitude": float(np.min(u)),
        "final_maximum_amplitude": float(np.max(u)),
    }


def insert_after_first_tick(stim, base, inserted):
    """Insert a circuit block after reset/initialisation and its first TICK."""
    lines = str(base).splitlines()
    tick_positions = [i for i, line in enumerate(lines) if line.strip() == "TICK"]
    if not tick_positions:
        raise RuntimeError("generated Stim circuit has no TICK after initialisation")
    cut = tick_positions[0] + 1
    prefix = stim.Circuit("\n".join(lines[:cut]) + "\n")
    suffix = stim.Circuit("\n".join(lines[cut:]) + "\n")
    return prefix + inserted + suffix


def surface_layout(
    base, distance: int
) -> tuple[list[int], list[tuple[int, int]], dict[int, tuple[float, float]]]:
    """Recover rotated-code data qubits and nearest-neighbour data edges."""
    raw = base.get_final_qubit_coordinates()
    coordinates = {
        int(q): (float(values[0]), float(values[1]))
        for q, values in raw.items()
        if len(values) >= 2
    }
    data = sorted(
        q for q, (x, y) in coordinates.items()
        if int(round(x)) % 2 == 1 and int(round(y)) % 2 == 1
    )
    if len(data) != distance * distance:
        raise RuntimeError(
            f"unexpected rotated-code data layout: {len(data)} data qubits "
            f"for distance {distance}; expected {distance * distance}"
        )
    edges: list[tuple[int, int]] = []
    for i in range(len(data)):
        xi, yi = coordinates[data[i]]
        for j in range(i + 1, len(data)):
            xj, yj = coordinates[data[j]]
            if abs(xi - xj) + abs(yi - yj) == 2.0:
                edges.append((i, j))
    expected_edges = 2 * distance * (distance - 1)
    if len(edges) != expected_edges:
        raise RuntimeError(
            f"unexpected rotated-code data graph: {len(edges)} edges "
            f"for distance {distance}; expected {expected_edges}"
        )
    return data, edges, coordinates


def generated_surface_base(
    stim,
    distance: int,
    rounds: int,
    after_clifford_depolarization: float,
    before_round_data_depolarization: float,
    before_measure_flip_probability: float,
    after_reset_flip_probability: float,
):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=after_clifford_depolarization,
        before_round_data_depolarization=before_round_data_depolarization,
        before_measure_flip_probability=before_measure_flip_probability,
        after_reset_flip_probability=after_reset_flip_probability,
    )


def build_circuit(
    stim,
    d: int,
    rounds: int,
    single_p: np.ndarray,
    pair_p: np.ndarray,
    after_clifford_depolarization: float,
    before_round_data_depolarization: float,
    before_measure_flip_probability: float,
    after_reset_flip_probability: float,
):
    base = generated_surface_base(
        stim,
        d,
        rounds,
        after_clifford_depolarization,
        before_round_data_depolarization,
        before_measure_flip_probability,
        after_reset_flip_probability,
    )
    data, edges, _ = surface_layout(base, d)
    if len(single_p) != len(data) or len(pair_p) != len(edges):
        raise ValueError(
            f"fault-vector shape mismatch: singles={len(single_p)}/{len(data)}, "
            f"pairs={len(pair_p)}/{len(edges)}"
        )
    inserted = stim.Circuit()
    inserted.append("I", data)  # explicit ideal-identity control layer
    inserted.append("TICK")
    for q, p in zip(data, single_p):
        inserted.append("X_ERROR", [q], float(p))
    for (i, j), p in zip(edges, pair_p):
        inserted.append(
            "E",
            [stim.target_x(data[i]), stim.target_x(data[j])],
            float(p),
        )
    inserted.append("TICK")
    return insert_after_first_tick(stim, base, inserted)


def noiseless_endpoint_gate(stim, d: int, rounds: int) -> dict[str, Any]:
    base = generated_surface_base(stim, d, rounds, 0.0, 0.0, 0.0, 0.0)
    data, edges, _ = surface_layout(base, d)
    zero_single = np.zeros(len(data))
    zero_pair = np.zeros(len(edges))
    circuit = build_circuit(stim, d, rounds, zero_single, zero_pair, 0.0, 0.0, 0.0, 0.0)
    sampler = circuit.compile_detector_sampler(seed=1234567 + d)
    det, obs = sampler.sample(shots=64, separate_observables=True)
    return {
        "all_noiseless_detectors_zero": not bool(np.any(det)),
        "all_noiseless_observable_flips_zero": not bool(np.any(obs)),
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "stim_register_width": circuit.num_qubits,
        "active_physical_qubits": len(circuit.get_final_qubit_coordinates()),
        "num_data_qubits": len(data),
        "num_data_edges": len(edges),
    }


def evaluate_circuit(
    circuit,
    pymatching,
    shots: int,
    sample_seed: int,
    correlated_decoding: bool,
    batch_shots: int,
) -> dict[str, Any]:
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(
        dem, enable_correlations=correlated_decoding
    )
    sampler = circuit.compile_detector_sampler(seed=sample_seed)
    failure_count = 0
    remaining = shots
    while remaining:
        take = min(batch_shots, remaining)
        detectors, actual = sampler.sample(
            shots=take, separate_observables=True, bit_packed=False
        )
        predicted = matching.decode_batch(
            detectors,
            enable_correlations=correlated_decoding,
            bit_packed_predictions=False,
        )
        if predicted.shape != actual.shape:
            raise RuntimeError(
                f"decoder/observable shape mismatch: {predicted.shape} versus {actual.shape}"
            )
        failure_count += int(np.count_nonzero(np.any(predicted != actual, axis=1)))
        remaining -= take
    p = failure_count / shots
    se = math.sqrt(max(p * (1.0 - p), 0.25 / shots) / shots)
    return {
        "decoded_logical_failure_probability": p,
        "failure_count": failure_count,
        "shots": shots,
        "wald_standard_error": se,
        "stim_register_width": circuit.num_qubits,
        "active_physical_qubits": len(circuit.get_final_qubit_coordinates()),
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "circuit_instruction_count": len(circuit),
        "detector_error_model_instruction_count": len(dem),
        "correlated_decoding": correlated_decoding,
        "sampling_batch_shots": batch_shots,
    }


def independent_difference_stats(
    failures0: int, n0: int, failures1: int, n1: int
) -> dict[str, float]:
    p0 = failures0 / n0
    p1 = failures1 / n1
    se = math.sqrt(
        max(p0 * (1.0 - p0), 0.25 / n0) / n0
        + max(p1 * (1.0 - p1), 0.25 / n1) / n1
    )
    absolute = p0 - p1
    return {
        "absolute_failure_reduction": absolute,
        "relative_failure_reduction": absolute / p0 if p0 > 0 else 0.0,
        "independent_standard_error": se,
        "z_score": absolute / se if se > 0 else (math.inf if absolute > 0 else 0.0),
    }


def wilson_upper(failures: int, shots: int, z: float = 1.6448536269514722) -> float:
    """One-sided Wilson upper confidence bound for a binomial proportion."""
    if shots <= 0 or failures < 0 or failures > shots:
        raise ValueError("invalid binomial count")
    p = failures / shots
    z2 = z * z
    denominator = 1.0 + z2 / shots
    centre = p + z2 / (2.0 * shots)
    radius = z * math.sqrt(p * (1.0 - p) / shots + z2 / (4.0 * shots * shots))
    return (centre + radius) / denominator


def analyse_seed_crossover(
    cases: list[dict[str, Any]],
    target: float,
    minimum_distance_reduction: int,
) -> dict[str, Any]:
    ordered = sorted(cases, key=lambda c: c["distance"])
    table = []
    for case in ordered:
        i = case["initial"]
        o = case["optimised"]
        ui = wilson_upper(i["failure_count"], i["shots"])
        uo = wilson_upper(o["failure_count"], o["shots"])
        table.append(
            {
                "distance": case["distance"],
                "rounds": case["rounds"],
                "physical_qubits": i["active_physical_qubits"],
                "physical_qubit_rounds": i["active_physical_qubits"]
                * case["rounds"],
                "initial_point_estimate": i["decoded_logical_failure_probability"],
                "initial_wilson_upper_95_one_sided": ui,
                "initial_target_closed": ui <= target,
                "optimised_point_estimate": o["decoded_logical_failure_probability"],
                "optimised_wilson_upper_95_one_sided": uo,
                "optimised_target_closed": uo <= target,
            }
        )
    d_initial = next((r["distance"] for r in table if r["initial_target_closed"]), None)
    d_optimised = next((r["distance"] for r in table if r["optimised_target_closed"]), None)
    resolved = d_initial is not None and d_optimised is not None
    distance_reduction = (
        int(d_initial - d_optimised) if resolved else None
    )
    if resolved:
        initial_row = next(
            r for r in table if r["distance"] == d_initial
        )
        optimised_row = next(
            r for r in table if r["distance"] == d_optimised
        )
        initial_qubits = initial_row["physical_qubits"]
        optimised_qubits = optimised_row["physical_qubits"]
        initial_qubit_rounds = initial_row["physical_qubit_rounds"]
        optimised_qubit_rounds = optimised_row["physical_qubit_rounds"]
        qubit_round_saving = 1.0 - optimised_qubit_rounds / initial_qubit_rounds
    else:
        initial_qubits = optimised_qubits = None
        initial_qubit_rounds = optimised_qubit_rounds = None
        qubit_round_saving = None
    gates = {
        "reference_target_distance_resolved": d_initial is not None,
        "optimised_target_distance_resolved": d_optimised is not None,
        "minimum_distance_reduced_by_declared_amount": (
            resolved and distance_reduction >= minimum_distance_reduction
        ),
        "qubit_round_resource_strictly_reduced": (
            qubit_round_saving is not None and qubit_round_saving > 0.0
        ),
    }
    return {
        "seed": ordered[0]["seed"],
        "target_logical_failure_probability": target,
        "decision_rule": "one-sided 95% Wilson upper bound",
        "minimum_reference_distance": d_initial,
        "minimum_optimised_distance": d_optimised,
        "distance_reduction": distance_reduction,
        "reference_physical_qubits_at_target": initial_qubits,
        "optimised_physical_qubits_at_target": optimised_qubits,
        "reference_physical_qubit_rounds_at_target": initial_qubit_rounds,
        "optimised_physical_qubit_rounds_at_target": optimised_qubit_rounds,
        "relative_physical_qubit_round_saving": qubit_round_saving,
        "distance_table": table,
        "gates": gates,
        "pass": all(gates.values()),
    }


def effective_logical_error_per_round(total_error: float, rounds: int) -> float:
    """Convert a parity-flip probability into an iid-equivalent per-round rate."""
    clipped = min(max(total_error, 0.0), 0.5 - 1e-15)
    return 0.5 * (1.0 - (1.0 - 2.0 * clipped) ** (1.0 / rounds))


def fit_distance_scaling(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(cases, key=lambda case: case["distance"])
    x = np.array([case["distance"] for case in ordered], dtype=float)
    result: dict[str, Any] = {"seed": ordered[0]["seed"]}
    for arm in ("initial", "optimised"):
        total = np.array(
            [case[arm]["decoded_logical_failure_probability"] for case in ordered]
        )
        per_round = np.array(
            [
                effective_logical_error_per_round(
                    case[arm]["decoded_logical_failure_probability"], case["rounds"]
                )
                for case in ordered
            ]
        )
        floor = np.array([0.5 / case[arm]["shots"] for case in ordered])
        y = np.log(np.maximum(per_round, floor))
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        result[arm] = {
            "distances": x.astype(int).tolist(),
            "total_logical_failure_probabilities": total.tolist(),
            "iid_equivalent_per_round_probabilities": per_round.tolist(),
            "log_per_round_intercept": float(intercept),
            "log_per_round_slope": float(slope),
            "suppression_exponent_beta": float(-slope),
            "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0,
            "total_error_strictly_decreases_with_distance": bool(
                np.all(np.diff(total) < 0.0)
            ),
            "per_round_error_strictly_decreases_with_distance": bool(
                np.all(np.diff(per_round) < 0.0)
            ),
        }
    result["beta_flow_minus_reference"] = (
        result["optimised"]["suppression_exponent_beta"]
        - result["initial"]["suppression_exponent_beta"]
    )
    result["both_arms_have_positive_suppression_exponent"] = (
        result["initial"]["suppression_exponent_beta"] > 0.0
        and result["optimised"]["suppression_exponent_beta"] > 0.0
    )
    return result


def run_case(stim, pymatching, d: int, seed: int, args) -> dict[str, Any]:
    case_rounds = d if args.round_policy == "equal_distance" else args.fixed_rounds
    layout_base = generated_surface_base(stim, d, case_rounds, 0.0, 0.0, 0.0, 0.0)
    data, edges, coordinates = surface_layout(layout_base, d)
    hw = make_hardware(len(data), args.segments, seed, d)
    initial, initialisation = initial_schedule(
        len(data),
        args.segments,
        seed,
        d,
        args.minimum_amplitude,
        args.maximum_amplitude,
    )
    optimised, history, optimiser_diagnostics = optimise(
        initial,
        hw,
        edges,
        args.iterations,
        args.step_radius,
        args.minimum_amplitude,
        args.maximum_amplitude,
    )
    score0, _, single0, pair0 = pulse_risk_and_gradient(initial, hw, edges)
    score1, _, single1, pair1 = pulse_risk_and_gradient(optimised, hw, edges)

    c0 = build_circuit(
        stim, d, case_rounds, single0, pair0,
        args.after_clifford_depolarization,
        args.before_round_data_depolarization,
        args.before_measure_flip_probability,
        args.after_reset_flip_probability,
    )
    c1 = build_circuit(
        stim, d, case_rounds, single1, pair1,
        args.after_clifford_depolarization,
        args.before_round_data_depolarization,
        args.before_measure_flip_probability,
        args.after_reset_flip_probability,
    )
    endpoint = noiseless_endpoint_gate(stim, d, case_rounds)
    m0 = evaluate_circuit(
        c0,
        pymatching,
        args.shots,
        seed + 1000003 * d,
        args.correlated_decoding,
        args.batch_shots,
    )
    m1 = evaluate_circuit(
        c1,
        pymatching,
        args.shots,
        seed + 2000003 * d,
        args.correlated_decoding,
        args.batch_shots,
    )
    stats = independent_difference_stats(
        m0["failure_count"], m0["shots"], m1["failure_count"], m1["shots"]
    )
    p0 = m0["decoded_logical_failure_probability"]
    p1 = m1["decoded_logical_failure_probability"]
    physical_qubits = m0["active_physical_qubits"]
    cost0 = physical_qubits * case_rounds / (1.0 - p0)
    cost1 = physical_qubits * case_rounds / (1.0 - p1)
    cost_reduction = (cost0 - cost1) / cost0
    area_residual = float(np.max(np.abs(optimised.mean(axis=1) - 1.0)))
    gates = {
        "initial_schedule_inside_amplitude_box": (
            initialisation["projected_minimum"] >= args.minimum_amplitude - 1e-14
            and initialisation["projected_maximum"] <= args.maximum_amplitude + 1e-14
        ),
        "initial_schedule_mean_constraint_closed": (
            initialisation["maximum_mean_residual"] <= args.maximum_area_residual
        ),
        "at_least_one_flow_step_accepted": optimiser_diagnostics["accepted_steps"] > 0,
        "all_schedule_checkpoints_inside_box": optimiser_diagnostics[
            "all_checkpoints_inside_amplitude_box"
        ],
        "noiseless_logical_endpoint_verified": (
            endpoint["all_noiseless_detectors_zero"]
            and endpoint["all_noiseless_observable_flips_zero"]
        ),
        "pulse_area_fibre_preserved": area_residual <= args.maximum_area_residual,
        "circuit_structure_matched": (
            m0["active_physical_qubits"] == m1["active_physical_qubits"]
            and m0["stim_register_width"] == m1["stim_register_width"]
            and m0["num_detectors"] == m1["num_detectors"]
            and m0["num_observables"] == m1["num_observables"]
            and m0["circuit_instruction_count"] == m1["circuit_instruction_count"]
        ),
        "matched_decoder_built_from_each_dem": True,
        "hook_capable_two_qubit_noise_present": args.after_clifford_depolarization > 0.0,
        "flow_nontrivial": float(np.linalg.norm(optimised - initial)) >= args.minimum_flow_displacement,
        "design_score_reduced": score1 < score0,
        "decoded_failure_reduction_gate": stats["relative_failure_reduction"] >= args.minimum_failure_reduction,
        "significance_gate": stats["z_score"] >= args.minimum_z_score,
        "reliable_unit_change_cost_reduced": cost_reduction > 0.0,
    }
    return {
        "seed": seed,
        "distance": d,
        "rounds": case_rounds,
        "layout": {
            "physical_qubits": physical_qubits,
            "data_qubits": len(data),
            "nearest_neighbour_data_edges": len(edges),
            "data_qubit_stim_indices": data,
            "data_qubit_coordinates": {
                str(q): list(coordinates[q]) for q in data
            },
        },
        "noiseless_endpoint": endpoint,
        "initial": {
            **m0,
            "design_score": score0,
            "pulse_single_fault_probabilities": single0.tolist(),
            "pulse_pair_fault_probabilities": pair0.tolist(),
        },
        "optimised": {
            **m1,
            "design_score": score1,
            "pulse_single_fault_probabilities": single1.tolist(),
            "pulse_pair_fault_probabilities": pair1.tolist(),
        },
        "difference_statistics": stats,
        "initial_reliable_unit_change_qubit_cycles": cost0,
        "optimised_reliable_unit_change_qubit_cycles": cost1,
        "relative_reliable_unit_change_cost_reduction": cost_reduction,
        "maximum_pulse_area_residual": area_residual,
        "flow_displacement_norm": float(np.linalg.norm(optimised - initial)),
        "initialisation_diagnostics": initialisation,
        "optimiser_diagnostics": optimiser_diagnostics,
        "gates": gates,
        "pass": all(gates.values()),
        "design_history": history,
        "initial_schedule": initial.tolist(),
        "optimised_schedule": optimised.tolist(),
    }


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--output", default="ft_unit_change_time_v0_5_2_results")
    p.add_argument("--seeds", default=",".join(map(str, CALIBRATION_SEEDS)))
    p.add_argument("--distances", default="3,5,7,9,11")
    p.add_argument("--segments", type=int, default=24)
    p.add_argument("--round-policy", choices=("equal_distance", "fixed"), default="equal_distance")
    p.add_argument("--fixed-rounds", type=int, default=5)
    p.add_argument("--shots", type=int, default=200000)
    p.add_argument("--batch-shots", type=int, default=20000)
    p.add_argument("--iterations", type=int, default=240)
    p.add_argument("--step-radius", type=float, default=0.035)
    p.add_argument("--minimum-amplitude", type=float, default=0.02)
    p.add_argument("--maximum-amplitude", type=float, default=3.5)
    p.add_argument("--after-clifford-depolarization", type=float, default=0.0015)
    p.add_argument("--before-round-data-depolarization", type=float, default=0.0010)
    p.add_argument("--before-measure-flip-probability", type=float, default=0.0050)
    p.add_argument("--after-reset-flip-probability", type=float, default=0.0025)
    p.add_argument("--maximum-area-residual", type=float, default=1e-12)
    p.add_argument("--minimum-flow-displacement", type=float, default=0.1)
    p.add_argument("--minimum-failure-reduction", type=float, default=0.05)
    p.add_argument("--minimum-z-score", type=float, default=1.645)
    p.add_argument("--target-grid", default="0.05,0.02,0.01,0.005,0.001")
    p.add_argument("--minimum-distance-reduction", type=int, default=2)
    p.add_argument("--minimum-successful-seed-fraction", type=float, default=2.0 / 3.0)
    p.add_argument("--no-correlated-decoding", action="store_true")
    p.add_argument("--no-auto-install", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p


def main() -> int:
    args, unknown = make_parser().parse_known_args()
    if unknown:
        print(f"[notice] ignored notebook arguments: {unknown}")
    args.correlated_decoding = not args.no_correlated_decoding
    if args.quick:
        args.shots = min(args.shots, 10000)
        args.iterations = min(args.iterations, 100)
        args.batch_shots = min(args.batch_shots, 5000)
    stim, pymatching = load_qec_packages(auto_install=not args.no_auto_install)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    distances = tuple(int(x) for x in args.distances.split(",") if x.strip())
    targets = tuple(float(x) for x in args.target_grid.split(",") if x.strip())
    if any(d < 3 or d % 2 == 0 for d in distances):
        raise ValueError("distances must be odd and at least three")
    if args.shots <= 0 or args.batch_shots <= 0:
        raise ValueError("shots and batch-shots must be positive")
    if not targets or any(not 0.0 < target < 1.0 for target in targets):
        raise ValueError("target-grid entries must lie strictly between zero and one")

    frozen_default_calibration = (
        seeds == CALIBRATION_SEEDS
        and distances == (3, 5, 7, 9, 11)
        and targets == (0.05, 0.02, 0.01, 0.005, 0.001)
        and not args.quick
        and args.shots == 200_000
        and args.batch_shots == 20_000
        and args.round_policy == "equal_distance"
        and args.fixed_rounds == 5
        and args.segments == 24
        and args.iterations == 240
        and args.step_radius == 0.035
        and args.minimum_amplitude == 0.02
        and args.maximum_amplitude == 3.5
        and args.maximum_area_residual == 1e-12
        and args.minimum_flow_displacement == 0.1
        and args.minimum_failure_reduction == 0.05
        and args.minimum_z_score == 1.645
        and args.correlated_decoding
        and args.minimum_distance_reduction == 2
        and args.minimum_successful_seed_fraction == 2.0 / 3.0
        and args.after_clifford_depolarization == 0.0015
        and args.before_round_data_depolarization == 0.0010
        and args.before_measure_flip_probability == 0.0050
        and args.after_reset_flip_probability == 0.0025
    )
    protocol = {
        "title": TITLE,
        "version": VERSION,
        "formal_interval_arithmetic": False,
        "purpose": "calibrate a lower-noise rounds-equal-distance regime, scan fixed-reliability targets, and estimate distance slopes before freezing v0.6",
        "logical_task": "one protected rotated-surface-code logical Z-memory interval (ideal identity logical channel)",
        "implementation_fibre": "fixed normalized identity-layer schedule mean on every data qubit",
        "circuit_engine": "Stim generated surface_code:rotated_memory_z with an inserted ideal-identity layer and schedule-dependent faults",
        "decoder": "PyMatching constructed separately from each Stim detector error model",
        "hook_error_model": "two-qubit Clifford depolarization in syndrome-extraction gates",
        "correlated_schedule_faults": "Stim E instructions on adjacent data pairs",
        "schedule_to_fault_map": "frozen synthetic heterogeneous exposure map; not a pulse compiler or calibrated hardware model",
        "design_evaluation_separation": "projected flow uses only the analytic exposure score; claims use independently seeded Stim detector samples and matched decoding",
        "resource_measure": "active coordinated Stim qubits times the distance-dependent syndrome rounds; sparse Stim register width is reported but not charged as hardware",
        "source_v051_protocol_sha256": "d44fa5affc21c243d25bd812f926bb76570373cf039b998262dfe68af18d9894",
        "source_v051_certificate_sha256": "cb41aac02ab5960496ae490f93383675edfe6f4cc66123dddead4f14c872dae6",
        "development_seeds_excluded_from_new_claims": list(DEVELOPMENT_SEEDS_EXCLUDED),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "new_prospective_cohort_used": False,
        "frozen_default_calibration_used": frozen_default_calibration,
        "seeds": list(seeds),
        "distances": list(distances),
        "segments": args.segments,
        "round_policy": args.round_policy,
        "fixed_rounds_if_selected": args.fixed_rounds,
        "shots_per_schedule": args.shots,
        "sampling_batch_shots": args.batch_shots,
        "noise": {
            "after_clifford_depolarization": args.after_clifford_depolarization,
            "before_round_data_depolarization": args.before_round_data_depolarization,
            "before_measure_flip_probability": args.before_measure_flip_probability,
            "after_reset_flip_probability": args.after_reset_flip_probability,
        },
        "correlated_decoding": args.correlated_decoding,
        "independent_heldout_sampler_seeds": True,
        "target_calibration": {
            "candidate_logical_failure_probabilities": list(targets),
            "confidence_rule": "one-sided 95% Wilson upper bound must not exceed target",
            "minimum_distance_reduction": args.minimum_distance_reduction,
            "minimum_successful_seed_fraction": args.minimum_successful_seed_fraction,
        },
        "gates": {
            "maximum_area_residual": args.maximum_area_residual,
            "minimum_flow_displacement": args.minimum_flow_displacement,
            "secondary_fixed_distance_diagnostic_minimum_reduction": args.minimum_failure_reduction,
            "secondary_fixed_distance_diagnostic_minimum_z": args.minimum_z_score,
        },
        "initialisation_repair": "rowwise Euclidean projection onto the intersection of the declared amplitude box and the exact mean-one affine fibre",
        "quick_mode": args.quick,
        "software": {"stim": stim.__version__, "pymatching": pymatching.__version__},
        "scope": "lower-noise rounds-equal-distance development calibration; not a new prospective advantage claim, threshold theorem, interval certificate, compiler result, or hardware validation",
    }
    protocol_hash = sha256_obj(protocol)
    print("=" * 112)
    print(f"{TITLE} v{VERSION}")
    print("=" * 112)
    print(json.dumps(protocol, indent=2))
    print(f"protocol_sha256 = {protocol_hash}")

    start = time.time()
    cases = []
    for seed in seeds:
        for d in distances:
            case = run_case(stim, pymatching, d, seed, args)
            cases.append(case)
            print(
                f"[seed={seed} d={d}] repair_flow_steps="
                f"{case['optimiser_diagnostics']['accepted_steps']} "
                f"secondary_fixed_distance_pass={case['pass']} "
                f"pL={case['initial']['decoded_logical_failure_probability']:.6e}"
                f"->{case['optimised']['decoded_logical_failure_probability']:.6e} "
                f"reduction={case['difference_statistics']['relative_failure_reduction']:.2%} "
                f"z={case['difference_statistics']['z_score']:.3f}"
            )

    target_diagnostics = []
    for target in targets:
        crossovers = []
        for seed in seeds:
            seed_cases = [c for c in cases if c["seed"] == seed]
            crossover = analyse_seed_crossover(
                seed_cases, target, args.minimum_distance_reduction
            )
            crossovers.append(crossover)
            print(
                f"[target={target:g} seed={seed} crossover] pass={crossover['pass']} "
                f"d_ref={crossover['minimum_reference_distance']} "
                f"d_flow={crossover['minimum_optimised_distance']} "
                f"qubit_round_saving={crossover['relative_physical_qubit_round_saving']}"
            )
        fraction = sum(item["pass"] for item in crossovers) / len(crossovers)
        savings = [
            item["relative_physical_qubit_round_saving"]
            for item in crossovers
            if item["relative_physical_qubit_round_saving"] is not None
        ]
        target_diagnostics.append(
            {
                "target": target,
                "successful_seed_fraction": fraction,
                "development_candidate_gate_pass": (
                    fraction >= args.minimum_successful_seed_fraction
                ),
                "minimum_resolved_physical_qubit_round_saving": (
                    min(savings) if savings else None
                ),
                "seed_crossovers": crossovers,
            }
        )

    scaling_fits = [
        fit_distance_scaling([case for case in cases if case["seed"] == seed])
        for seed in seeds
    ]
    positive_scaling_fraction = sum(
        fit["both_arms_have_positive_suppression_exponent"] for fit in scaling_fits
    ) / len(scaling_fits)
    for fit in scaling_fits:
        print(
            f"[seed={fit['seed']} scaling] "
            f"beta_ref={fit['initial']['suppression_exponent_beta']:.6f} "
            f"beta_flow={fit['optimised']['suppression_exponent_beta']:.6f} "
            f"delta={fit['beta_flow_minus_reference']:.6f}"
        )

    architecture_keys = (
        "initial_schedule_inside_amplitude_box",
        "initial_schedule_mean_constraint_closed",
        "at_least_one_flow_step_accepted",
        "all_schedule_checkpoints_inside_box",
        "noiseless_logical_endpoint_verified",
        "pulse_area_fibre_preserved",
        "circuit_structure_matched",
        "matched_decoder_built_from_each_dem",
        "hook_capable_two_qubit_noise_present",
        "flow_nontrivial",
        "design_score_reduced",
    )
    architecture_pass = all(
        all(case["gates"][key] for key in architecture_keys) for case in cases
    )
    candidate_targets = [
        item["target"]
        for item in target_diagnostics
        if item["development_candidate_gate_pass"]
    ]
    scaling_regime_found = (
        positive_scaling_fraction >= args.minimum_successful_seed_fraction
    )
    rounds_equal_distance_gate = all(
        case["rounds"] == case["distance"] for case in cases
    )
    calibration_candidate_found = (
        architecture_pass
        and rounds_equal_distance_gate
        and scaling_regime_found
        and bool(candidate_targets)
    )
    calibration_pass = frozen_default_calibration and calibration_candidate_found
    report = {
        "scientific_status": (
            "DEVELOPMENT_ROTATED_SURFACE_CODE_SCALING_AND_CROSSOVER_CANDIDATE_FOUND"
            if calibration_pass
            else (
                "DEVELOPMENT_ROTATED_SURFACE_CODE_SCALING_REGIME_FOUND_CROSSOVER_UNRESOLVED"
                if architecture_pass and scaling_regime_found
                else "DEVELOPMENT_ROTATED_SURFACE_CODE_SCALING_REGIME_NOT_FOUND"
            )
        ),
        "all_gates_pass": calibration_candidate_found,
        "architecture_gates_pass": architecture_pass,
        "rounds_equal_distance_gate_pass": rounds_equal_distance_gate,
        "formal_interval_arithmetic": False,
        "frozen_default_calibration_used": frozen_default_calibration,
        "new_prospective_cohort_used": False,
        "new_prospective_advantage_claimed": False,
        "fixed_reliability_distance_reduction_claimed": False,
        "fixed_reliability_qubit_cycle_advantage_claimed": False,
        "fault_tolerance_threshold_claimed": False,
        "development_calibration_result_claimed": calibration_pass,
        "hardware_advantage_claimed": False,
        "protocol_sha256": protocol_hash,
        "cases_declared": len(cases),
        "secondary_fixed_distance_cases_passing": sum(c["pass"] for c in cases),
        "cases_with_accepted_flow_steps": sum(
            c["optimiser_diagnostics"]["accepted_steps"] > 0 for c in cases
        ),
        "minimum_accepted_flow_steps": min(
            c["optimiser_diagnostics"]["accepted_steps"] for c in cases
        ),
        "cases_whose_unrepaired_raw_schedule_was_outside_box": sum(
            c["initialisation_diagnostics"]["raw_normalized_minimum"]
            < args.minimum_amplitude
            or c["initialisation_diagnostics"]["raw_normalized_maximum"]
            > args.maximum_amplitude
            for c in cases
        ),
        "seeds_declared": len(seeds),
        "candidate_targets_scanned": list(targets),
        "candidate_targets_passing_development_gate": candidate_targets,
        "most_stringent_candidate_target_for_v060": (
            min(candidate_targets) if candidate_targets else None
        ),
        "positive_distance_scaling_seed_fraction": positive_scaling_fraction,
        "distance_scaling_regime_found": scaling_regime_found,
        "minimum_declared_distance_reduction": args.minimum_distance_reduction,
        "minimum_relative_decoded_failure_reduction": min(
            c["difference_statistics"]["relative_failure_reduction"] for c in cases
        ),
        "minimum_z_score": min(c["difference_statistics"]["z_score"] for c in cases),
        "minimum_relative_reliable_unit_change_cost_reduction": min(
            c["relative_reliable_unit_change_cost_reduction"] for c in cases
        ),
        "elapsed_seconds": time.time() - start,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "stim": stim.__version__,
            "pymatching": pymatching.__version__,
            "platform": platform.platform(),
        },
        "next_required_step": "if a target closes the development crossover gate and both arms show positive distance suppression, freeze exactly one target, the noise tuple, rounds=d, distance set, gates, and entirely new seeds in v0.6; otherwise refine only the declared development regime",
        "scope": protocol["scope"],
        "target_diagnostics": target_diagnostics,
        "scaling_fits": scaling_fits,
        "cases": cases,
    }
    report["certificate_sha256_before_self_field"] = sha256_obj(report)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_bytes(canonical_json(protocol) + b"\n")
    (out / "report.json").write_bytes(canonical_json(report) + b"\n")
    print("\n" + "=" * 112)
    print("FINAL RESULT")
    print("=" * 112)
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "target_diagnostics", "scaling_fits")}, indent=2))
    if calibration_pass:
        print("\nPASS: a development scaling/crossover regime is available for freezing in a new v0.6 prospective protocol.")
    elif architecture_pass and scaling_regime_found:
        print("\nPARTIAL: distance suppression is present, but no candidate target closed the development crossover gate.")
    else:
        print("\nFAIL-CLOSED: the declared development scaling/crossover regime was not resolved.")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL-CLOSED: {type(exc).__name__}: {exc}")
        raise
