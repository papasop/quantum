#!/usr/bin/env python3
"""Fail-closed numerical preflight for fault-tolerant unit-change time.

The model is deliberately small and transparent.  A distance-d repetition
code implements one ideal logical X.  Each physical qubit receives a
same-axis pulse with fixed area, so every admissible schedule has exactly the
same noiseless endpoint.  Time-dependent independent Pauli-X faults are then
integrated analytically and majority-decoding failure is evaluated exactly.

Projected descent changes only the pulse shapes while preserving every pulse
area.  This is a numerical response-fibre experiment, not a threshold
theorem, surface-code simulation, or hardware result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


TITLE = "FAULT-TOLERANT UNIT-CHANGE-TIME RESPONSE-FIBRE PREFLIGHT"
VERSION = "0.1.0"
DEFAULT_SEEDS = (20280107, 20280119, 20280203)


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def distribution(ps: np.ndarray) -> np.ndarray:
    """Poisson-binomial count distribution, evaluated without sampling."""
    out = np.array([1.0])
    for p in ps:
        nxt = np.zeros(out.size + 1)
        nxt[:-1] += out * (1.0 - p)
        nxt[1:] += out * p
        out = nxt
    return out


def logical_failure_and_dp(ps: np.ndarray) -> tuple[float, np.ndarray]:
    """Majority-decoding failure and exact derivatives with respect to p_i."""
    d = ps.size
    threshold = (d + 1) // 2
    fail = float(distribution(ps)[threshold:].sum())
    grad = np.empty(d)
    for i in range(d):
        others = np.delete(ps, i)
        dist = distribution(others)
        grad[i] = dist[threshold - 1]
    return fail, grad


@dataclass
class Hardware:
    idle: np.ndarray
    drive: np.ndarray
    xtalk: float


def make_hardware(d: int, segments: int, seed: int) -> Hardware:
    """Frozen heterogeneous, time-dependent toy hardware coefficients."""
    rng = np.random.default_rng(seed + 1009 * d)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=(d, 1))
    t = (np.arange(segments) + 0.5)[None, :] / segments
    # Heterogeneity makes schedule shape matter while all coefficients remain
    # positive and fixed before optimisation.
    drive = 0.018 * (
        1.15
        + 0.48 * np.sin(2.0 * math.pi * t + phase)
        + 0.20 * np.cos(4.0 * math.pi * t - 0.7 * phase)
    )
    drive *= rng.uniform(0.82, 1.18, size=(d, 1))
    drive = np.maximum(drive, 0.003)
    idle = rng.uniform(0.0015, 0.0030, size=d)
    return Hardware(idle=idle, drive=drive, xtalk=0.0045)


def exposure_and_gradient(
    u: np.ndarray, hardware: Hardware
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return exposures, physical error probabilities, p_L and d p_L / d u."""
    d, segments = u.shape
    dt = 1.0 / segments
    exposure = hardware.idle.copy()
    exposure += dt * np.sum(hardware.drive * u * u, axis=1)

    # Nearest-neighbour ring crosstalk.  Each endpoint accumulates the pair
    # exposure; this is an explicitly declared phenomenological noise model.
    for i in range(d):
        j = (i + 1) % d
        exposure[i] += dt * hardware.xtalk * np.sum(u[i] * u[j])
        exposure[j] += dt * hardware.xtalk * np.sum(u[i] * u[j])

    ps = 0.5 * (1.0 - np.exp(-2.0 * exposure))
    p_logical, dpl_dp = logical_failure_and_dp(ps)
    dpl_de = dpl_dp * np.exp(-2.0 * exposure)

    grad = dt * (2.0 * hardware.drive * u) * dpl_de[:, None]
    for i in range(d):
        j = (i + 1) % d
        pair_weight = dt * hardware.xtalk * (dpl_de[i] + dpl_de[j])
        grad[i] += pair_weight * u[j]
        grad[j] += pair_weight * u[i]
    return exposure, ps, p_logical, grad


def initial_schedule(d: int, segments: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 7919 * d)
    raw = np.exp(rng.normal(0.0, 0.42, size=(d, segments)))
    return raw / raw.mean(axis=1, keepdims=True)


def flat_schedule(d: int, segments: int) -> np.ndarray:
    return np.ones((d, segments))


def metrics(u: np.ndarray, hardware: Hardware) -> dict[str, Any]:
    exposure, ps, p_logical, grad = exposure_and_gradient(u, hardware)
    d = u.shape[0]
    logical_success = 1.0 - p_logical
    # The ideal logical task distance is declared to be one.  Gate-window time
    # is one, so d qubits occupy d qubit-time units.
    reliable_unit_change_cost = d / logical_success
    tangent_grad = grad - grad.mean(axis=1, keepdims=True)
    return {
        "logical_failure_probability": float(p_logical),
        "logical_success_probability": float(logical_success),
        "reliable_unit_change_qubit_time": float(reliable_unit_change_cost),
        "maximum_pulse_area_residual": float(np.max(np.abs(u.mean(axis=1) - 1.0))),
        "minimum_amplitude": float(u.min()),
        "maximum_amplitude": float(u.max()),
        "physical_error_probabilities": ps.tolist(),
        "integrated_exposures": exposure.tolist(),
        "projected_gradient_norm": float(np.linalg.norm(tangent_grad)),
    }


def optimise(
    u0: np.ndarray,
    hardware: Hardware,
    iterations: int,
    step_radius: float,
    min_amp: float,
    max_amp: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    u = u0.copy()
    history: list[dict[str, float]] = []
    previous = math.inf

    for k in range(iterations + 1):
        _, _, value, grad = exposure_and_gradient(u, hardware)
        if value > previous + 1e-14:
            raise ArithmeticError("logical failure increased along accepted flow")
        previous = value
        if k % max(1, iterations // 20) == 0 or k == iterations:
            history.append(
                {
                    "iteration": k,
                    "logical_failure_probability": float(value),
                    "maximum_area_residual": float(np.max(np.abs(u.mean(axis=1) - 1.0))),
                }
            )
        if k == iterations:
            break

        direction = -(grad - grad.mean(axis=1, keepdims=True))
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm):
            raise ArithmeticError("projected gradient is non-finite")
        if norm <= 1e-15:
            break
        direction /= norm

        alpha = step_radius
        accepted = False
        directional = float(np.sum(grad * direction))
        for _ in range(40):
            trial = u + alpha * direction
            # Row sums are preserved algebraically; reject rather than clip so
            # the endpoint response constraint is not silently changed.
            if trial.min() >= min_amp and trial.max() <= max_amp:
                _, _, trial_value, _ = exposure_and_gradient(trial, hardware)
                if trial_value <= value + 1e-4 * alpha * directional:
                    u = trial
                    accepted = True
                    break
            alpha *= 0.5
        if not accepted:
            # A failed Armijo search at machine precision is treated as
            # numerical convergence, not as evidence for a theorem.  The
            # preregistered reduction and displacement gates still decide the
            # scientific result fail-closed.
            break
    return u, history


def run_case(d: int, segments: int, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    hardware = make_hardware(d, segments, seed)
    initial = initial_schedule(d, segments, seed)
    flat = flat_schedule(d, segments)
    optimised, history = optimise(
        initial,
        hardware,
        args.iterations,
        args.step_radius,
        args.minimum_amplitude,
        args.maximum_amplitude,
    )
    m0 = metrics(initial, hardware)
    mf = metrics(flat, hardware)
    m1 = metrics(optimised, hardware)

    failure_reduction = (m0["logical_failure_probability"] - m1["logical_failure_probability"]) / m0[
        "logical_failure_probability"
    ]
    cost_reduction = (
        m0["reliable_unit_change_qubit_time"] - m1["reliable_unit_change_qubit_time"]
    ) / m0["reliable_unit_change_qubit_time"]
    gates = {
        "same_noiseless_logical_X_endpoint": True,
        "pulse_area_fibre_preserved": m1["maximum_pulse_area_residual"] <= args.maximum_area_residual,
        "amplitude_bounds_preserved": (
            m1["minimum_amplitude"] >= args.minimum_amplitude
            and m1["maximum_amplitude"] <= args.maximum_amplitude
        ),
        "logical_failure_strictly_reduced": failure_reduction >= args.minimum_failure_reduction,
        "reliable_unit_change_cost_strictly_reduced": cost_reduction > 0.0,
        "flow_is_nontrivial": float(np.linalg.norm(optimised - initial)) >= args.minimum_flow_displacement,
        "all_outputs_finite": bool(np.all(np.isfinite(optimised))),
    }
    return {
        "seed": seed,
        "distance": d,
        "segments": segments,
        "initial": m0,
        "flat_reference": mf,
        "optimised": m1,
        "relative_logical_failure_reduction": float(failure_reduction),
        "relative_reliable_unit_change_cost_reduction": float(cost_reduction),
        "flow_displacement_norm": float(np.linalg.norm(optimised - initial)),
        "gates": gates,
        "pass": all(gates.values()),
        "history": history,
        "initial_schedule": initial.tolist(),
        "optimised_schedule": optimised.tolist(),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--output", default="ft_unit_change_time_v0_1_0_results")
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--distances", default="3,5,7")
    p.add_argument("--segments", type=int, default=24)
    p.add_argument("--iterations", type=int, default=240)
    p.add_argument("--step-radius", type=float, default=0.035)
    p.add_argument("--minimum-amplitude", type=float, default=0.02)
    p.add_argument("--maximum-amplitude", type=float, default=3.5)
    p.add_argument("--maximum-area-residual", type=float, default=1e-12)
    p.add_argument("--minimum-failure-reduction", type=float, default=0.05)
    p.add_argument("--minimum-flow-displacement", type=float, default=0.1)
    return p


def main() -> int:
    args, unknown = parser().parse_known_args()
    if unknown:
        print(f"[notice] ignored notebook arguments: {unknown}")
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    distances = tuple(int(x) for x in args.distances.split(",") if x.strip())
    if any(d < 3 or d % 2 == 0 for d in distances):
        raise ValueError("repetition-code distances must be odd and at least three")

    protocol = {
        "title": TITLE,
        "version": VERSION,
        "formal_interval_arithmetic": False,
        "exact_probability_evaluation": True,
        "purpose": "test fibre-preserving reduction of logical failure and reliable unit-change qubit-time",
        "logical_task": "one ideal repetition-code logical X",
        "implementation_fibre": "each physical same-axis pulse has fixed normalized area one",
        "noise_model": "declared heterogeneous time-dependent independent Pauli-X rates with nearest-neighbour drive crosstalk",
        "decoder": "exact majority decoder",
        "unit_change_metric": "qubit-time divided by decoded logical success for a task distance declared equal to one",
        "seeds": list(seeds),
        "distances": list(distances),
        "segments": args.segments,
        "gates": {
            "maximum_area_residual": args.maximum_area_residual,
            "minimum_relative_logical_failure_reduction": args.minimum_failure_reduction,
            "minimum_flow_displacement": args.minimum_flow_displacement,
        },
        "scope": "toy repetition-code numerical preflight; not a threshold theorem, surface-code result, correlated decoder study, compiler result, or hardware validation",
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
            case = run_case(d, args.segments, seed, args)
            cases.append(case)
            print(
                f"[seed={seed} d={d}] pass={case['pass']} "
                f"pL={case['initial']['logical_failure_probability']:.6e}"
                f"->{case['optimised']['logical_failure_probability']:.6e} "
                f"failure_reduction={case['relative_logical_failure_reduction']:.3%} "
                f"unit_cost_reduction={case['relative_reliable_unit_change_cost_reduction']:.3%}"
            )

    cohort_pass = all(c["pass"] for c in cases)
    report = {
        "scientific_status": (
            "TOY_FT_UNIT_CHANGE_TIME_FIBRE_MECHANISM_SUPPORTED"
            if cohort_pass
            else "TOY_FT_UNIT_CHANGE_TIME_FIBRE_MECHANISM_NOT_SUPPORTED"
        ),
        "all_gates_pass": cohort_pass,
        "formal_interval_arithmetic": False,
        "fault_tolerance_threshold_claimed": False,
        "hardware_advantage_claimed": False,
        "unit_change_time_fibre_mechanism_claimed_on_declared_model": cohort_pass,
        "protocol_sha256": protocol_hash,
        "cases_declared": len(cases),
        "cases_passing": sum(c["pass"] for c in cases),
        "minimum_relative_logical_failure_reduction": min(
            c["relative_logical_failure_reduction"] for c in cases
        ),
        "minimum_relative_reliable_unit_change_cost_reduction": min(
            c["relative_reliable_unit_change_cost_reduction"] for c in cases
        ),
        "elapsed_seconds": time.time() - start,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "next_required_step": "replace the independent repetition-code model by a circuit-level syndrome-extraction experiment with correlated faults and a matched decoder; preserve the same-task and qubit-time gates",
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
    summary = {k: v for k, v in report.items() if k != "cases"}
    print(json.dumps(summary, indent=2))
    if cohort_pass:
        print("\nPASS: the declared toy model supports fibre-preserving reduction of logical failure and unit-change cost.")
    else:
        print("\nFAIL-CLOSED: the declared cohort does not support the mechanism.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(f"FAIL-CLOSED: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
