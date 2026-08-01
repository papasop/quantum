#!/usr/bin/env python3
"""Stim/PyMatching rotated-surface-code resource audit v0.5.0.

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

This is a prospective numerical rotated-surface-code resource audit. It is
not a threshold theorem, formal interval certificate, compiler result, or
hardware validation.
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


TITLE = "FAULT-TOLERANT RESPONSE-FIBRE ROTATED-SURFACE-CODE RESOURCE AUDIT"
VERSION = "0.5.0"
PROSPECTIVE_SEEDS = (20280907, 20280919, 20281003)
DEVELOPMENT_SEEDS_EXCLUDED = (
    20280107, 20280119, 20280203,
    20280307, 20280319, 20280403,
    20280507, 20280519, 20280603,
    20280707, 20280719, 20280803,
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


def initial_schedule(n_data: int, segments: int, seed: int, distance: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 7919 * distance)
    raw = np.exp(rng.normal(0.0, 0.44, size=(n_data, segments)))
    return raw / raw.mean(axis=1, keepdims=True)


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
) -> tuple[np.ndarray, list[dict[str, float]]]:
    u = u0.copy()
    history: list[dict[str, float]] = []
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
                    accepted = True
                    break
            alpha *= 0.5
        if not accepted:
            break
    return u, history


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
                "physical_qubits": i["active_physical_qubits"],
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
        initial_qubits = next(
            r["physical_qubits"] for r in table if r["distance"] == d_initial
        )
        optimised_qubits = next(
            r["physical_qubits"] for r in table if r["distance"] == d_optimised
        )
        qubit_cycle_saving = 1.0 - optimised_qubits / initial_qubits
    else:
        initial_qubits = optimised_qubits = None
        qubit_cycle_saving = None
    gates = {
        "reference_target_distance_resolved": d_initial is not None,
        "optimised_target_distance_resolved": d_optimised is not None,
        "minimum_distance_reduced_by_declared_amount": (
            resolved and distance_reduction >= minimum_distance_reduction
        ),
        "qubit_cycle_resource_strictly_reduced": (
            qubit_cycle_saving is not None and qubit_cycle_saving > 0.0
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
        "relative_qubit_cycle_saving_at_fixed_rounds": qubit_cycle_saving,
        "distance_table": table,
        "gates": gates,
        "pass": all(gates.values()),
    }


def run_case(stim, pymatching, d: int, seed: int, args) -> dict[str, Any]:
    layout_base = generated_surface_base(stim, d, args.rounds, 0.0, 0.0, 0.0, 0.0)
    data, edges, coordinates = surface_layout(layout_base, d)
    hw = make_hardware(len(data), args.segments, seed, d)
    initial = initial_schedule(len(data), args.segments, seed, d)
    optimised, history = optimise(
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
        stim, d, args.rounds, single0, pair0,
        args.after_clifford_depolarization,
        args.before_round_data_depolarization,
        args.before_measure_flip_probability,
        args.after_reset_flip_probability,
    )
    c1 = build_circuit(
        stim, d, args.rounds, single1, pair1,
        args.after_clifford_depolarization,
        args.before_round_data_depolarization,
        args.before_measure_flip_probability,
        args.after_reset_flip_probability,
    )
    endpoint = noiseless_endpoint_gate(stim, d, args.rounds)
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
    cost0 = physical_qubits * args.rounds / (1.0 - p0)
    cost1 = physical_qubits * args.rounds / (1.0 - p1)
    cost_reduction = (cost0 - cost1) / cost0
    area_residual = float(np.max(np.abs(optimised.mean(axis=1) - 1.0)))
    gates = {
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
        "rounds": args.rounds,
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
        "gates": gates,
        "pass": all(gates.values()),
        "design_history": history,
        "initial_schedule": initial.tolist(),
        "optimised_schedule": optimised.tolist(),
    }


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--output", default="ft_unit_change_time_v0_5_0_results")
    p.add_argument("--seeds", default=",".join(map(str, PROSPECTIVE_SEEDS)))
    p.add_argument("--distances", default="3,5,7,9")
    p.add_argument("--segments", type=int, default=24)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--shots", type=int, default=1000000)
    p.add_argument("--batch-shots", type=int, default=50000)
    p.add_argument("--iterations", type=int, default=240)
    p.add_argument("--step-radius", type=float, default=0.035)
    p.add_argument("--minimum-amplitude", type=float, default=0.02)
    p.add_argument("--maximum-amplitude", type=float, default=3.5)
    p.add_argument("--after-clifford-depolarization", type=float, default=0.004)
    p.add_argument("--before-round-data-depolarization", type=float, default=0.003)
    p.add_argument("--before-measure-flip-probability", type=float, default=0.012)
    p.add_argument("--after-reset-flip-probability", type=float, default=0.006)
    p.add_argument("--maximum-area-residual", type=float, default=1e-12)
    p.add_argument("--minimum-flow-displacement", type=float, default=0.1)
    p.add_argument("--minimum-failure-reduction", type=float, default=0.05)
    p.add_argument("--minimum-z-score", type=float, default=1.645)
    p.add_argument("--target-logical-failure", type=float, default=1e-3)
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
    if any(d < 3 or d % 2 == 0 for d in distances):
        raise ValueError("distances must be odd and at least three")
    if args.shots <= 0 or args.batch_shots <= 0:
        raise ValueError("shots and batch-shots must be positive")

    frozen_default_cohort = (
        seeds == PROSPECTIVE_SEEDS
        and distances == (3, 5, 7, 9)
        and not args.quick
        and args.shots == 1_000_000
        and args.batch_shots == 50_000
        and args.rounds == 5
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
        and args.target_logical_failure == 1e-3
        and args.minimum_distance_reduction == 2
        and args.minimum_successful_seed_fraction == 2.0 / 3.0
        and args.after_clifford_depolarization == 0.004
        and args.before_round_data_depolarization == 0.003
        and args.before_measure_flip_probability == 0.012
        and args.after_reset_flip_probability == 0.006
    )
    protocol = {
        "title": TITLE,
        "version": VERSION,
        "formal_interval_arithmetic": False,
        "purpose": "test whether an ideal-identity response-fibre flow lowers the minimum rotated-surface-code distance and qubit-cycle volume required at a fixed logical reliability",
        "logical_task": "one protected rotated-surface-code logical Z-memory interval (ideal identity logical channel)",
        "implementation_fibre": "fixed normalized identity-layer schedule mean on every data qubit",
        "circuit_engine": "Stim generated surface_code:rotated_memory_z with an inserted ideal-identity layer and schedule-dependent faults",
        "decoder": "PyMatching constructed separately from each Stim detector error model",
        "hook_error_model": "two-qubit Clifford depolarization in syndrome-extraction gates",
        "correlated_schedule_faults": "Stim E instructions on adjacent data pairs",
        "schedule_to_fault_map": "frozen synthetic heterogeneous exposure map; not a pulse compiler or calibrated hardware model",
        "design_evaluation_separation": "projected flow uses only the analytic exposure score; claims use independently seeded Stim detector samples and matched decoding",
        "resource_measure": "active coordinated Stim qubits times fixed syndrome rounds; sparse Stim register width is reported but not charged as hardware",
        "development_seeds_excluded_from_claims": list(DEVELOPMENT_SEEDS_EXCLUDED),
        "prospective_seeds_frozen_before_run": list(PROSPECTIVE_SEEDS),
        "frozen_default_cohort_used": frozen_default_cohort,
        "seeds": list(seeds),
        "distances": list(distances),
        "segments": args.segments,
        "rounds": args.rounds,
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
        "target_decision": {
            "target_logical_failure_probability": args.target_logical_failure,
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
        "quick_mode": args.quick,
        "software": {"stim": stim.__version__, "pymatching": pymatching.__version__},
        "scope": "fixed-target rotated-surface-code numerical distance/resource audit; not a threshold theorem, interval certificate, compiler result, or hardware validation",
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
                f"[seed={seed} d={d}] secondary_fixed_distance_pass={case['pass']} "
                f"pL={case['initial']['decoded_logical_failure_probability']:.6e}"
                f"->{case['optimised']['decoded_logical_failure_probability']:.6e} "
                f"reduction={case['difference_statistics']['relative_failure_reduction']:.2%} "
                f"z={case['difference_statistics']['z_score']:.3f}"
            )

    seed_crossovers = []
    for seed in seeds:
        seed_cases = [c for c in cases if c["seed"] == seed]
        crossover = analyse_seed_crossover(
            seed_cases,
            args.target_logical_failure,
            args.minimum_distance_reduction,
        )
        seed_crossovers.append(crossover)
        print(
            f"[seed={seed} crossover] pass={crossover['pass']} "
            f"d_ref={crossover['minimum_reference_distance']} "
            f"d_flow={crossover['minimum_optimised_distance']} "
            f"saving={crossover['relative_qubit_cycle_saving_at_fixed_rounds']}"
        )

    architecture_keys = (
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
    successful_seed_fraction = sum(x["pass"] for x in seed_crossovers) / len(seed_crossovers)
    cohort_pass = (
        architecture_pass
        and successful_seed_fraction >= args.minimum_successful_seed_fraction
    )
    claim_eligible = frozen_default_cohort and cohort_pass
    resolved_savings = [
        x["relative_qubit_cycle_saving_at_fixed_rounds"]
        for x in seed_crossovers
        if x["relative_qubit_cycle_saving_at_fixed_rounds"] is not None
    ]
    report = {
        "scientific_status": (
            "PROSPECTIVE_ROTATED_SURFACE_CODE_FIXED_RELIABILITY_RESOURCE_ADVANTAGE_SUPPORTED"
            if claim_eligible
            else (
                "ROTATED_SURFACE_CODE_RESOURCE_CROSSOVER_PREFLIGHT_SUPPORTED"
                if cohort_pass
                else "ROTATED_SURFACE_CODE_FIXED_RELIABILITY_RESOURCE_ADVANTAGE_NOT_SUPPORTED"
            )
        ),
        "all_gates_pass": cohort_pass,
        "architecture_gates_pass": architecture_pass,
        "formal_interval_arithmetic": False,
        "claim_eligible_frozen_prospective_cohort": frozen_default_cohort,
        "fixed_reliability_distance_reduction_claimed": claim_eligible,
        "fixed_reliability_qubit_cycle_advantage_claimed": claim_eligible,
        "fault_tolerance_threshold_claimed": False,
        "rotated_surface_code_numerical_result_claimed": claim_eligible,
        "hardware_advantage_claimed": False,
        "protocol_sha256": protocol_hash,
        "cases_declared": len(cases),
        "secondary_fixed_distance_cases_passing": sum(c["pass"] for c in cases),
        "seeds_declared": len(seed_crossovers),
        "crossover_seeds_passing": sum(x["pass"] for x in seed_crossovers),
        "crossover_successful_seed_fraction": successful_seed_fraction,
        "crossover_cohort_gate_pass": cohort_pass,
        "target_logical_failure_probability": args.target_logical_failure,
        "minimum_declared_distance_reduction": args.minimum_distance_reduction,
        "minimum_resolved_relative_qubit_cycle_saving": (
            min(resolved_savings) if resolved_savings else None
        ),
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
        "next_required_step": "if the rotated-surface-code resource crossover passes, freeze a physical-noise-rate grid and test a scaling region or replace the synthetic schedule-to-fault map by compiler-derived pulse noise; do not call either result a threshold theorem without the required scaling analysis",
        "scope": protocol["scope"],
        "seed_crossovers": seed_crossovers,
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
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "seed_crossovers")}, indent=2))
    if claim_eligible:
        print("\nPASS: the frozen prospective rotated-surface-code distance and qubit-cycle advantage is supported.")
    elif cohort_pass:
        print("\nPREFLIGHT PASS: the crossover gates pass, but this is not the frozen default protocol.")
    else:
        print("\nFAIL-CLOSED: the preregistered cohort does not support a fixed-reliability distance/resource advantage.")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL-CLOSED: {type(exc).__name__}: {exc}")
        raise
