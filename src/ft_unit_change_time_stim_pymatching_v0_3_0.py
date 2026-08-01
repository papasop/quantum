#!/usr/bin/env python3
"""Stim/PyMatching gate-level response-fibre audit v0.3.0.

The ideal task is one transversal logical X on a distance-d repetition code.
Every schedule has the same fixed pulse area on each data qubit, so the ideal
endpoint is unchanged.  A projected flow optimises a frozen pulse-risk score.
Initial and optimised schedule-dependent faults are then inserted into a Stim
repetition-code syndrome-extraction circuit containing reset, measurement,
idle, and two-qubit Clifford noise.  PyMatching is built independently from
each circuit's detector error model, and held-out detector samples decide the
claim.

This is a prospective numerical gate-level repetition-code experiment.  It is
not a threshold theorem, surface-code result, formal interval certificate, or
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


TITLE = "FAULT-TOLERANT RESPONSE-FIBRE STIM/PYMATCHING GATE-LEVEL AUDIT"
VERSION = "0.3.0"
PROSPECTIVE_SEEDS = (20280507, 20280519, 20280603)
DEVELOPMENT_SEEDS_EXCLUDED = (
    20280107, 20280119, 20280203,
    20280307, 20280319, 20280403,
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


def make_hardware(d: int, segments: int, seed: int) -> Hardware:
    rng = np.random.default_rng(seed + 1009 * d)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=(d, 1))
    t = (np.arange(segments) + 0.5)[None, :] / segments
    drive = 0.024 * (
        1.18
        + 0.52 * np.sin(2.0 * math.pi * t + phase)
        + 0.22 * np.cos(4.0 * math.pi * t - 0.7 * phase)
    )
    drive *= rng.uniform(0.80, 1.20, size=(d, 1))
    return Hardware(drive=np.maximum(drive, 0.003), xtalk=0.0065)


def initial_schedule(d: int, segments: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 7919 * d)
    raw = np.exp(rng.normal(0.0, 0.44, size=(d, segments)))
    return raw / raw.mean(axis=1, keepdims=True)


def pulse_risk_and_gradient(
    u: np.ndarray, hw: Hardware
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    d, segments = u.shape
    dt = 1.0 / segments
    exposure = dt * np.sum(hw.drive * u * u, axis=1)
    pair_exposure = np.array(
        [dt * hw.xtalk * np.sum(u[i] * u[i + 1]) for i in range(d - 1)]
    )
    single_p = 0.5 * (1.0 - np.exp(-2.0 * exposure))
    pair_p = 1.0 - np.exp(-pair_exposure)
    score = float(np.sum(single_p) + 2.5 * np.sum(pair_p))

    grad = dt * (2.0 * hw.drive * u) * np.exp(-2.0 * exposure)[:, None]
    for i in range(d - 1):
        coeff = 2.5 * math.exp(-pair_exposure[i]) * dt * hw.xtalk
        grad[i] += coeff * u[i + 1]
        grad[i + 1] += coeff * u[i]
    return score, grad, single_p, pair_p


def optimise(
    u0: np.ndarray,
    hw: Hardware,
    iterations: int,
    step_radius: float,
    min_amp: float,
    max_amp: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    u = u0.copy()
    history: list[dict[str, float]] = []
    for k in range(iterations + 1):
        value, grad, _, _ = pulse_risk_and_gradient(u, hw)
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
                trial_value, _, _, _ = pulse_risk_and_gradient(trial, hw)
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
    base = stim.Circuit.generated(
        "repetition_code:memory",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=after_clifford_depolarization,
        before_round_data_depolarization=before_round_data_depolarization,
        before_measure_flip_probability=before_measure_flip_probability,
        after_reset_flip_probability=after_reset_flip_probability,
    )
    if base.num_qubits != 2 * d - 1:
        raise RuntimeError(
            f"unexpected repetition-code layout: {base.num_qubits} qubits for distance {d}"
        )
    data = list(range(0, 2 * d - 1, 2))
    inserted = stim.Circuit()
    inserted.append("X", data)  # exact ideal transversal logical X
    inserted.append("TICK")
    for q, p in zip(data, single_p):
        inserted.append("X_ERROR", [q], float(p))
    for i, p in enumerate(pair_p):
        inserted.append(
            "E",
            [stim.target_x(data[i]), stim.target_x(data[i + 1])],
            float(p),
        )
    inserted.append("TICK")
    return insert_after_first_tick(stim, base, inserted)


def noiseless_endpoint_gate(stim, d: int, rounds: int) -> dict[str, Any]:
    zero_single = np.zeros(d)
    zero_pair = np.zeros(d - 1)
    circuit = build_circuit(stim, d, rounds, zero_single, zero_pair, 0.0, 0.0, 0.0, 0.0)
    sampler = circuit.compile_detector_sampler(seed=1234567 + d)
    det, obs = sampler.sample(shots=64, separate_observables=True)
    return {
        "all_noiseless_detectors_zero": not bool(np.any(det)),
        "all_noiseless_observable_flips_zero": not bool(np.any(obs)),
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "num_qubits": circuit.num_qubits,
    }


def evaluate_circuit(
    circuit,
    pymatching,
    shots: int,
    sample_seed: int,
    correlated_decoding: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(
        dem, enable_correlations=correlated_decoding
    )
    sampler = circuit.compile_detector_sampler(seed=sample_seed)
    detectors, actual = sampler.sample(
        shots=shots, separate_observables=True, bit_packed=False
    )
    predicted = matching.decode_batch(
        detectors, enable_correlations=correlated_decoding, bit_packed_predictions=False
    )
    if predicted.shape != actual.shape:
        raise RuntimeError(
            f"decoder/observable shape mismatch: {predicted.shape} versus {actual.shape}"
        )
    failures = np.any(predicted != actual, axis=1)
    p = float(np.mean(failures))
    se = math.sqrt(max(p * (1.0 - p), 0.25 / shots) / shots)
    return {
        "decoded_logical_failure_probability": p,
        "failure_count": int(failures.sum()),
        "shots": shots,
        "wald_standard_error": se,
        "num_qubits": circuit.num_qubits,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "circuit_instruction_count": len(circuit),
        "detector_error_model_instruction_count": len(dem),
        "correlated_decoding": correlated_decoding,
    }, failures


def independent_difference_stats(f0: np.ndarray, f1: np.ndarray) -> dict[str, float]:
    p0 = float(np.mean(f0))
    p1 = float(np.mean(f1))
    n0, n1 = f0.size, f1.size
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


def run_case(stim, pymatching, d: int, seed: int, args) -> dict[str, Any]:
    hw = make_hardware(d, args.segments, seed)
    initial = initial_schedule(d, args.segments, seed)
    optimised, history = optimise(
        initial,
        hw,
        args.iterations,
        args.step_radius,
        args.minimum_amplitude,
        args.maximum_amplitude,
    )
    score0, _, single0, pair0 = pulse_risk_and_gradient(initial, hw)
    score1, _, single1, pair1 = pulse_risk_and_gradient(optimised, hw)

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
    m0, f0 = evaluate_circuit(
        c0, pymatching, args.shots, seed + 1000003 * d, args.correlated_decoding
    )
    m1, f1 = evaluate_circuit(
        c1, pymatching, args.shots, seed + 2000003 * d, args.correlated_decoding
    )
    stats = independent_difference_stats(f0, f1)
    p0 = m0["decoded_logical_failure_probability"]
    p1 = m1["decoded_logical_failure_probability"]
    physical_qubits = 2 * d - 1
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
            m0["num_qubits"] == m1["num_qubits"]
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
    p.add_argument("--output", default="ft_unit_change_time_v0_3_0_results")
    p.add_argument("--seeds", default=",".join(map(str, PROSPECTIVE_SEEDS)))
    p.add_argument("--distances", default="3,5,7")
    p.add_argument("--segments", type=int, default=24)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--shots", type=int, default=250000)
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
    stim, pymatching = load_qec_packages(auto_install=not args.no_auto_install)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    distances = tuple(int(x) for x in args.distances.split(",") if x.strip())
    if any(d < 3 or d % 2 == 0 for d in distances):
        raise ValueError("distances must be odd and at least three")

    frozen_default_cohort = seeds == PROSPECTIVE_SEEDS and not args.quick
    protocol = {
        "title": TITLE,
        "version": VERSION,
        "formal_interval_arithmetic": False,
        "purpose": "test whether a same-logical-X fibre flow survives gate-level syndrome extraction and matched decoding",
        "logical_task": "one ideal transversal repetition-code logical X",
        "implementation_fibre": "fixed normalized same-axis pulse area on every data qubit",
        "circuit_engine": "Stim generated repetition_code:memory with inserted logical X and schedule-dependent faults",
        "decoder": "PyMatching constructed separately from each Stim detector error model",
        "hook_error_model": "two-qubit Clifford depolarization in syndrome-extraction gates",
        "correlated_schedule_faults": "Stim E instructions on adjacent data pairs",
        "development_seeds_excluded_from_claims": list(DEVELOPMENT_SEEDS_EXCLUDED),
        "prospective_seeds_frozen_before_run": list(PROSPECTIVE_SEEDS),
        "frozen_default_cohort_used": frozen_default_cohort,
        "seeds": list(seeds),
        "distances": list(distances),
        "segments": args.segments,
        "rounds": args.rounds,
        "shots_per_schedule": args.shots,
        "noise": {
            "after_clifford_depolarization": args.after_clifford_depolarization,
            "before_round_data_depolarization": args.before_round_data_depolarization,
            "before_measure_flip_probability": args.before_measure_flip_probability,
            "after_reset_flip_probability": args.after_reset_flip_probability,
        },
        "correlated_decoding": args.correlated_decoding,
        "independent_heldout_sampler_seeds": True,
        "gates": {
            "maximum_area_residual": args.maximum_area_residual,
            "minimum_relative_decoded_failure_reduction": args.minimum_failure_reduction,
            "minimum_z_score": args.minimum_z_score,
            "minimum_flow_displacement": args.minimum_flow_displacement,
        },
        "quick_mode": args.quick,
        "software": {"stim": stim.__version__, "pymatching": pymatching.__version__},
        "scope": "gate-level repetition-code numerical audit; not a threshold theorem, surface code, interval certificate, compiler result, or hardware validation",
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
                f"[seed={seed} d={d}] pass={case['pass']} "
                f"pL={case['initial']['decoded_logical_failure_probability']:.6e}"
                f"->{case['optimised']['decoded_logical_failure_probability']:.6e} "
                f"reduction={case['difference_statistics']['relative_failure_reduction']:.2%} "
                f"z={case['difference_statistics']['z_score']:.3f}"
            )

    cohort_pass = all(c["pass"] for c in cases)
    claim_eligible = frozen_default_cohort and cohort_pass
    report = {
        "scientific_status": (
            "PROSPECTIVE_GATE_LEVEL_FT_RESPONSE_FIBRE_MECHANISM_SUPPORTED"
            if claim_eligible
            else (
                "GATE_LEVEL_FT_RESPONSE_FIBRE_PREFLIGHT_SUPPORTED"
                if cohort_pass
                else "GATE_LEVEL_FT_RESPONSE_FIBRE_MECHANISM_NOT_SUPPORTED"
            )
        ),
        "all_gates_pass": cohort_pass,
        "formal_interval_arithmetic": False,
        "claim_eligible_frozen_prospective_cohort": frozen_default_cohort,
        "gate_level_repetition_code_mechanism_claimed": claim_eligible,
        "fault_tolerance_threshold_claimed": False,
        "surface_code_claimed": False,
        "hardware_advantage_claimed": False,
        "protocol_sha256": protocol_hash,
        "cases_declared": len(cases),
        "cases_passing": sum(c["pass"] for c in cases),
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
        "next_required_step": "run a rotated surface-code memory cohort and test whether the flow lowers the code distance or qubit-cycle volume required for a fixed target logical error",
        "scope": protocol["scope"],
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
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    if claim_eligible:
        print("\nPASS: the frozen prospective gate-level repetition-code fibre mechanism is supported.")
    elif cohort_pass:
        print("\nPREFLIGHT PASS: gates pass, but this run is not the frozen prospective cohort.")
    else:
        print("\nFAIL-CLOSED: the preregistered gate-level cohort does not support the mechanism.")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL-CLOSED: {type(exc).__name__}: {exc}")
        raise
