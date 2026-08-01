#!/usr/bin/env python3
"""Syndrome-graph fault-tolerant response-fibre audit v0.2.0.

Every admissible pulse schedule implements the same noiseless transversal
logical X because each same-axis physical pulse has fixed area.  A projected
flow changes pulse shape only.  Schedule-dependent data faults and correlated
nearest-neighbour faults feed a repeated-syndrome detector graph containing
measurement faults.  A pure-NumPy minimum-weight decoder uses the same frozen
event probabilities as the generator.

This is a phenomenological detector-level experiment.  It is not a surface-
code threshold calculation, a gate-level CNOT fault model, formal interval
arithmetic, or hardware validation.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TITLE = "FAULT-TOLERANT RESPONSE-FIBRE SYNDROME-GRAPH AUDIT"
VERSION = "0.2.0"
DEFAULT_SEEDS = (20280307, 20280319, 20280403)


def canonical_json(x: Any) -> bytes:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_obj(x: Any) -> str:
    return hashlib.sha256(canonical_json(x)).hexdigest()


@dataclass(frozen=True)
class FaultEvent:
    a: int
    b: int
    probability: float
    logical: int
    kind: str


@dataclass
class Hardware:
    drive: np.ndarray
    xtalk: float


def make_hardware(d: int, segments: int, seed: int) -> Hardware:
    rng = np.random.default_rng(seed + 1009 * d)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=(d, 1))
    t = (np.arange(segments) + 0.5)[None, :] / segments
    drive = 0.021 * (
        1.18
        + 0.52 * np.sin(2.0 * math.pi * t + phase)
        + 0.22 * np.cos(4.0 * math.pi * t - 0.7 * phase)
    )
    drive *= rng.uniform(0.80, 1.20, size=(d, 1))
    return Hardware(drive=np.maximum(drive, 0.003), xtalk=0.0055)


def initial_schedule(d: int, segments: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 7919 * d)
    raw = np.exp(rng.normal(0.0, 0.44, size=(d, segments)))
    return raw / raw.mean(axis=1, keepdims=True)


def pulse_risk_and_gradient(u: np.ndarray, hw: Hardware) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Differentiable local score used to design, but not evaluate, the flow."""
    d, segments = u.shape
    dt = 1.0 / segments
    exposure = dt * np.sum(hw.drive * u * u, axis=1)
    pair_exposure = np.empty(d - 1)
    for i in range(d - 1):
        pair_exposure[i] = dt * hw.xtalk * np.sum(u[i] * u[i + 1])

    single_p = 0.5 * (1.0 - np.exp(-2.0 * exposure))
    pair_p = 1.0 - np.exp(-pair_exposure)
    # The score is frozen before held-out detector sampling.  Pair faults get
    # a larger weight because adjacent two-data faults are more malignant.
    score = float(np.sum(single_p) + 2.5 * np.sum(pair_p))
    ds_de = np.exp(-2.0 * exposure)
    grad = dt * (2.0 * hw.drive * u) * ds_de[:, None]
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
            raise ArithmeticError("non-finite projected gradient")
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


def detector_index(t: int, check: int, checks: int) -> int:
    return t * checks + check


def spatial_endpoints(t: int, i: int, d: int, rounds: int) -> tuple[int, int, int]:
    checks = d - 1
    n_det = rounds * checks
    left, right = n_det, n_det + 1
    if i == 0:
        return left, detector_index(t, 0, checks), 0
    if i == d - 1:
        return detector_index(t, checks - 1, checks), right, 1
    return detector_index(t, i - 1, checks), detector_index(t, i, checks), 0


def correlated_endpoints(t: int, i: int, d: int, rounds: int) -> tuple[int, int, int]:
    """Syndrome boundary of simultaneous errors on adjacent data i,i+1."""
    a1, b1, l1 = spatial_endpoints(t, i, d, rounds)
    a2, b2, l2 = spatial_endpoints(t, i + 1, d, rounds)
    endpoints = [a1, b1, a2, b2]
    odd = [x for x in set(endpoints) if endpoints.count(x) % 2]
    if len(odd) != 2:
        raise ArithmeticError("correlated event does not have two graph endpoints")
    return odd[0], odd[1], l1 ^ l2


def build_events(
    d: int,
    rounds: int,
    pulse_single_p: np.ndarray,
    pulse_pair_p: np.ndarray,
    idle_data_p: float,
    measurement_p: float,
) -> tuple[list[FaultEvent], int, int, int]:
    checks = d - 1
    n_det = rounds * checks
    left, right = n_det, n_det + 1
    events: list[FaultEvent] = []

    for t in range(rounds):
        for i in range(d):
            a, b, logical = spatial_endpoints(t, i, d, rounds)
            p = float(pulse_single_p[i]) if t == 0 else idle_data_p
            events.append(FaultEvent(a, b, p, logical, "pulse_data" if t == 0 else "idle_data"))
        if t == 0:
            for i, p in enumerate(pulse_pair_p):
                a, b, logical = correlated_endpoints(t, i, d, rounds)
                events.append(FaultEvent(a, b, float(p), logical, "correlated_pair"))
        if t < rounds - 1:
            for j in range(checks):
                a = detector_index(t, j, checks)
                b = detector_index(t + 1, j, checks)
                events.append(FaultEvent(a, b, measurement_p, 0, "measurement"))
    return events, n_det, left, right


class MatchedDecoder:
    """Minimum-weight detector-graph decoder with logical-parity tracking."""

    def __init__(self, events: list[FaultEvent], n_det: int, left: int, right: int):
        self.n_det = n_det
        self.left = left
        self.right = right
        n_base = n_det + 2
        n = 2 * n_base
        inf = 1e100
        dist = np.full((n, n), inf)
        np.fill_diagonal(dist, 0.0)
        for event in events:
            p = min(max(event.probability, 1e-15), 1.0 - 1e-15)
            weight = math.log((1.0 - p) / p)
            if weight <= 0:
                raise ValueError("all graph-event probabilities must be below one half")
            for parity in (0, 1):
                a = event.a + parity * n_base
                b = event.b + (parity ^ event.logical) * n_base
                if weight < dist[a, b]:
                    dist[a, b] = dist[b, a] = weight
        # Floyd-Warshall on the parity-expanded detector graph.
        for k in range(n):
            dist = np.minimum(dist, dist[:, k, None] + dist[None, k, :])
        self.dist = dist
        self.n_base = n_base

    def _path_cost(self, a: int, b: int, parity: int) -> float:
        return float(self.dist[a, b + parity * self.n_base])

    @functools.lru_cache(maxsize=200000)
    def decode(self, syndrome_mask: int) -> int:
        defects = tuple(i for i in range(self.n_det) if (syndrome_mask >> i) & 1)

        @functools.lru_cache(maxsize=None)
        def dp(items: tuple[int, ...]) -> tuple[float, float]:
            if not items:
                return 0.0, 1e100
            a = items[0]
            rest = items[1:]
            best = [1e100, 1e100]

            # A detector can terminate on either spatial boundary.
            tail = dp(rest)
            for boundary in (self.left, self.right):
                for path_parity in (0, 1):
                    w = self._path_cost(a, boundary, path_parity)
                    for tail_parity in (0, 1):
                        total_parity = path_parity ^ tail_parity
                        best[total_parity] = min(best[total_parity], w + tail[tail_parity])

            # Or pair it with another detector.
            for pos, b in enumerate(rest):
                remaining = rest[:pos] + rest[pos + 1 :]
                tail = dp(remaining)
                for path_parity in (0, 1):
                    w = self._path_cost(a, b, path_parity)
                    for tail_parity in (0, 1):
                        total_parity = path_parity ^ tail_parity
                        best[total_parity] = min(best[total_parity], w + tail[tail_parity])
            return best[0], best[1]

        costs = dp(defects)
        return int(costs[1] < costs[0])


def sample_and_decode(
    events: list[FaultEvent],
    decoder: MatchedDecoder,
    n_det: int,
    shots: int,
    uniforms: np.ndarray,
) -> np.ndarray:
    syndrome = np.zeros(shots, dtype=np.uint64)
    true_logical = np.zeros(shots, dtype=np.uint8)
    for eidx, event in enumerate(events):
        fired = uniforms[eidx, :shots] < event.probability
        if event.a < n_det:
            syndrome[fired] ^= np.uint64(1) << np.uint64(event.a)
        if event.b < n_det:
            syndrome[fired] ^= np.uint64(1) << np.uint64(event.b)
        if event.logical:
            true_logical[fired] ^= 1

    unique, inverse = np.unique(syndrome, return_inverse=True)
    predictions = np.fromiter((decoder.decode(int(x)) for x in unique), dtype=np.uint8)
    return predictions[inverse] != true_logical


def evaluate_schedule(
    u: np.ndarray,
    hw: Hardware,
    d: int,
    rounds: int,
    shots: int,
    uniforms: np.ndarray,
    idle_data_p: float,
    measurement_p: float,
) -> tuple[dict[str, Any], np.ndarray]:
    design_score, _, single_p, pair_p = pulse_risk_and_gradient(u, hw)
    events, n_det, left, right = build_events(
        d, rounds, single_p, pair_p, idle_data_p, measurement_p
    )
    decoder = MatchedDecoder(events, n_det, left, right)
    failures = sample_and_decode(events, decoder, n_det, shots, uniforms)
    p = float(np.mean(failures))
    se = math.sqrt(max(p * (1.0 - p), 0.25 / shots) / shots)
    result = {
        "design_score": design_score,
        "decoded_logical_failure_probability": p,
        "wald_standard_error": se,
        "failure_count": int(failures.sum()),
        "shots": shots,
        "maximum_pulse_area_residual": float(np.max(np.abs(u.mean(axis=1) - 1.0))),
        "minimum_amplitude": float(u.min()),
        "maximum_amplitude": float(u.max()),
        "pulse_single_fault_probabilities": single_p.tolist(),
        "pulse_correlated_pair_probabilities": pair_p.tolist(),
        "detectors": n_det,
        "fault_events": len(events),
        "decoder": "matched minimum-weight detector graph with logical-parity-expanded shortest paths",
    }
    return result, failures


def paired_statistics(initial_fail: np.ndarray, optimised_fail: np.ndarray) -> dict[str, float]:
    diff = initial_fail.astype(float) - optimised_fail.astype(float)
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(diff.size)) if diff.size > 1 else math.inf
    z = mean / se if se > 0 else (math.inf if mean > 0 else 0.0)
    p0 = float(initial_fail.mean())
    p1 = float(optimised_fail.mean())
    return {
        "absolute_failure_reduction": mean,
        "relative_failure_reduction": (p0 - p1) / p0 if p0 > 0 else 0.0,
        "paired_standard_error": se,
        "paired_z_score": z,
    }


def run_case(d: int, seed: int, args: argparse.Namespace) -> dict[str, Any]:
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
    _, _, p_single0, p_pair0 = pulse_risk_and_gradient(initial, hw)
    events0, _, _, _ = build_events(
        d, args.rounds, p_single0, p_pair0, args.idle_data_probability, args.measurement_probability
    )
    _, _, p_single1, p_pair1 = pulse_risk_and_gradient(optimised, hw)
    events1, _, _, _ = build_events(
        d, args.rounds, p_single1, p_pair1, args.idle_data_probability, args.measurement_probability
    )
    if len(events0) != len(events1):
        raise ArithmeticError("initial and optimised detector models differ structurally")
    rng = np.random.default_rng(seed + 104729 * d)
    uniforms = rng.random((len(events0), args.shots))
    m0, f0 = evaluate_schedule(
        initial, hw, d, args.rounds, args.shots, uniforms,
        args.idle_data_probability, args.measurement_probability
    )
    m1, f1 = evaluate_schedule(
        optimised, hw, d, args.rounds, args.shots, uniforms,
        args.idle_data_probability, args.measurement_probability
    )
    stats = paired_statistics(f0, f1)
    p0 = m0["decoded_logical_failure_probability"]
    p1 = m1["decoded_logical_failure_probability"]
    # One ideal logical task unit, fixed d data qubits and fixed number of
    # syndrome cycles.  Expected successful-task cost includes repetitions.
    cost0 = d * args.rounds / (1.0 - p0)
    cost1 = d * args.rounds / (1.0 - p1)
    cost_reduction = (cost0 - cost1) / cost0
    area_residual = m1["maximum_pulse_area_residual"]
    gates = {
        "same_ideal_logical_X_endpoint": True,
        "same_detector_graph_structure": len(events0) == len(events1),
        "matched_decoder_used_for_each_schedule": True,
        "pulse_area_fibre_preserved": area_residual <= args.maximum_area_residual,
        "flow_nontrivial": float(np.linalg.norm(optimised - initial)) >= args.minimum_flow_displacement,
        "decoded_failure_reduction_gate": stats["relative_failure_reduction"] >= args.minimum_failure_reduction,
        "paired_significance_gate": stats["paired_z_score"] >= args.minimum_paired_z,
        "reliable_unit_change_cost_reduced": cost_reduction > 0.0,
    }
    return {
        "seed": seed,
        "distance": d,
        "rounds": args.rounds,
        "initial": m0,
        "optimised": m1,
        "paired_statistics": stats,
        "initial_reliable_unit_change_qubit_cycles": cost0,
        "optimised_reliable_unit_change_qubit_cycles": cost1,
        "relative_reliable_unit_change_cost_reduction": cost_reduction,
        "flow_displacement_norm": float(np.linalg.norm(optimised - initial)),
        "gates": gates,
        "pass": all(gates.values()),
        "design_history": history,
        "initial_schedule": initial.tolist(),
        "optimised_schedule": optimised.tolist(),
    }


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--output", default="ft_unit_change_time_v0_2_0_results")
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--distances", default="3,5,7")
    p.add_argument("--segments", type=int, default=24)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--shots", type=int, default=80000)
    p.add_argument("--iterations", type=int, default=240)
    p.add_argument("--step-radius", type=float, default=0.035)
    p.add_argument("--minimum-amplitude", type=float, default=0.02)
    p.add_argument("--maximum-amplitude", type=float, default=3.5)
    p.add_argument("--idle-data-probability", type=float, default=0.008)
    p.add_argument("--measurement-probability", type=float, default=0.018)
    p.add_argument("--maximum-area-residual", type=float, default=1e-12)
    p.add_argument("--minimum-flow-displacement", type=float, default=0.1)
    p.add_argument("--minimum-failure-reduction", type=float, default=0.05)
    p.add_argument("--minimum-paired-z", type=float, default=1.645)
    p.add_argument("--quick", action="store_true")
    return p


def main() -> int:
    args, unknown = make_parser().parse_known_args()
    if unknown:
        print(f"[notice] ignored notebook arguments: {unknown}")
    if args.quick:
        args.shots = min(args.shots, 8000)
        args.iterations = min(args.iterations, 100)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    distances = tuple(int(x) for x in args.distances.split(",") if x.strip())
    if any(d < 3 or d % 2 == 0 for d in distances):
        raise ValueError("distances must be odd and at least three")
    if max(distances) * args.rounds > 60:
        raise ValueError("detector bit-mask implementation requires fewer than 64 detectors")

    protocol = {
        "title": TITLE,
        "version": VERSION,
        "formal_interval_arithmetic": False,
        "purpose": "test whether fibre-designed pulse schedules reduce held-out matched-decoder logical failure and reliable unit-change qubit-cycles",
        "logical_task": "one ideal transversal repetition-code logical X",
        "implementation_fibre": "fixed normalized same-axis pulse area on every data qubit",
        "evaluation": "phenomenological repeated-syndrome detector graph with data, measurement, and adjacent correlated faults",
        "decoder": "pure-NumPy matched minimum-weight graph decoder with logical-parity tracking",
        "design_evaluation_separation": "projected flow minimises a frozen analytic pulse-risk score; claims use held-out detector-event uniforms and decoded failures",
        "common_random_numbers": True,
        "seeds": list(seeds),
        "distances": list(distances),
        "segments": args.segments,
        "rounds": args.rounds,
        "shots_per_schedule": args.shots,
        "noise": {
            "idle_data_probability_per_round": args.idle_data_probability,
            "measurement_probability_per_interval": args.measurement_probability,
            "pulse_data_and_pair_probabilities": "schedule-dependent and frozen before sampling",
        },
        "gates": {
            "maximum_area_residual": args.maximum_area_residual,
            "minimum_relative_decoded_failure_reduction": args.minimum_failure_reduction,
            "minimum_paired_z_score": args.minimum_paired_z,
            "minimum_flow_displacement": args.minimum_flow_displacement,
        },
        "quick_mode": args.quick,
        "scope": "phenomenological repetition-code detector-graph audit; not gate-level syndrome circuitry, a threshold theorem, surface code, formal certificate, or hardware result",
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
            case = run_case(d, seed, args)
            cases.append(case)
            print(
                f"[seed={seed} d={d}] pass={case['pass']} "
                f"pL={case['initial']['decoded_logical_failure_probability']:.6e}"
                f"->{case['optimised']['decoded_logical_failure_probability']:.6e} "
                f"reduction={case['paired_statistics']['relative_failure_reduction']:.2%} "
                f"z={case['paired_statistics']['paired_z_score']:.3f}"
            )

    all_pass = all(c["pass"] for c in cases)
    report = {
        "scientific_status": (
            "PHENOMENOLOGICAL_FT_SYNDROME_GRAPH_FIBRE_MECHANISM_SUPPORTED"
            if all_pass else "PHENOMENOLOGICAL_FT_SYNDROME_GRAPH_FIBRE_MECHANISM_NOT_SUPPORTED"
        ),
        "all_gates_pass": all_pass,
        "formal_interval_arithmetic": False,
        "fault_tolerance_threshold_claimed": False,
        "gate_level_circuit_noise_claimed": False,
        "matched_syndrome_graph_mechanism_claimed": all_pass,
        "protocol_sha256": protocol_hash,
        "cases_declared": len(cases),
        "cases_passing": sum(c["pass"] for c in cases),
        "minimum_relative_decoded_failure_reduction": min(
            c["paired_statistics"]["relative_failure_reduction"] for c in cases
        ),
        "minimum_paired_z_score": min(c["paired_statistics"]["paired_z_score"] for c in cases),
        "minimum_relative_reliable_unit_change_cost_reduction": min(
            c["relative_reliable_unit_change_cost_reduction"] for c in cases
        ),
        "elapsed_seconds": time.time() - start,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "next_required_step": "use Stim-generated gate-level syndrome-extraction circuits and PyMatching detector error models, including hook errors and a frozen prospective cohort",
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
    print(
        "\nPASS: matched syndrome-graph held-out decoding supports the declared fibre mechanism."
        if all_pass else
        "\nFAIL-CLOSED: the preregistered syndrome-graph cohort does not support the mechanism."
    )
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL-CLOSED: {type(exc).__name__}: {exc}")
        raise
