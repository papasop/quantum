#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N=2 117-point pulse-order scan plus fixed-H versus path-aware
Hamiltonian-learning audit.

This standalone script closes four predeclared questions without importing
historical scripts, cached results, or external data:

  (1) Does an independent 117-point exact-state scan resolve pulse-order
      information at matched total duration and weighted-average detuning?
  (2) Does the effect survive in the minimal interacting N=2 model?
  (3) Can any one shared time-independent generator represent both orderings,
      or must a summary-only model average them into one endpoint?
  (4) Where does that assumption fail in held-out learning, and does an ordered
      piecewise model recover unseen reverses, unseen schedules, and the hidden
      physical parameters?

Question (3) is answered by an architecture-independent analytic lower bound.
For a fixed input and a matched pair with the same duration and summary, every
shared fixed-H representation produces one common output ray.  If the two
target rays have overlap c, the smallest possible worst-case pure-state trace
distance of ANY such common output is

    D_minimax = sqrt((1-c)/2).

The script constructs the optimal projective midpoint and verifies this bound
numerically.  This is stronger than failure of one restricted optimizer.

Question (4) is kept separate.  Both learned models use the same two unknown
calibration parameters from FORWARD schedules only:

    theta = (Omega, detuning calibration offset).

The interaction strength is held fixed as a geometry/device parameter.  This
prevents weak identifiability of the |11> interaction from being conflated with
the schedule-representation question.

Before the learning comparison, the script runs a separate 117-point N=2
benchmark (9 detuning gaps x 13 clock-aligned fractions) at fixed total
duration and weighted-average detuning.  This scan characterizes the physical
response but is NOT used as the learning training set: every scan point has the
same average-only summary, which would make the learning conflict tautological.
The held-out audit therefore retains a separate varied-(T, average-detuning)
design.

The models receive the same training examples and use the same optimizer,
parameter bounds, starts, and physical Hamiltonian family.  They differ only
in the schedule representation:

    average-only:
        U = exp[-i H(weighted-average detuning; theta) T]

    path-aware:
        U = product_j exp[-i H(detuning_j; theta) duration_j]

The decisive orientation test consists of the REVERSES of the training
schedules, which neither model sees during fitting.  A second test uses wholly
unseen schedule parameters.  Targets are generated independently before each
fit from the declared piecewise model.

Scientific outcomes never raise AssertionError.  Assertions are reserved for
implementation validity (clock alignment, matched summaries, exact gap-zero
control, deterministic split, optimizer finiteness, and output integrity).

Dependencies: numpy, scipy, matplotlib.
Notebook-safe: unknown Jupyter arguments such as ``-f kernel.json`` are ignored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import expm
from scipy.optimize import least_squares


SCRIPT_VERSION = "3.1"
SEED = 20260725
CLOCK_NS = 4
SPACING_UM = 8.0

# Declared exact target parameters.  Frequencies are angular frequencies.
TRUE_OMEGA = 1.22  # rad / us
TRUE_DETUNING_OFFSET = 0.017  # rad / us; hidden calibration offset
FIXED_INTERACTION = 20.70  # rad / us for the |11> projector
TRUE_THETA = np.array([TRUE_OMEGA, TRUE_DETUNING_OFFSET], dtype=float)

PARAMETER_NAMES = ("Omega_rad_per_us", "detuning_offset_rad_per_us")
LOWER_BOUNDS = np.array([0.75, -0.12], dtype=float)
UPPER_BOUNDS = np.array([1.65, +0.12], dtype=float)

# These are scientific reporting thresholds, not implementation assertions.
PATH_MAX_TEST_D_TARGET = 1.0e-6
PATH_PARAMETER_RELATIVE_ERROR_TARGET = 1.0e-4
MIN_RESOLVED_ORDER_D = 1.0e-4
MIN_PATH_ADVANTAGE_FACTOR = 10.0

# Independent N=2 117-point benchmark.  8088 ns is divisible by 24 ns, so the
# 4 ns clock grid, exact half split, thirds, and f -> 1-f reflection coexist.
SCAN_TOTAL_NS = 8088
SCAN_CENTER = -0.31
SCAN_GAPS = tuple(float(x) for x in np.linspace(0.0, 0.24, 9))
SCAN_FRACTIONS = tuple(float(x) for x in np.linspace(0.10, 0.90, 13))
SCAN_EXPECTED_POINTS = 117


@dataclass(frozen=True)
class Spec:
    name: str
    total_ns: int
    center: float
    fraction_requested: float
    gap: float


@dataclass(frozen=True)
class Example:
    example_id: str
    split: str
    orientation: str
    total_ns: int
    duration1_ns: int
    duration2_ns: int
    fraction_actual: float
    gap: float
    center: float
    delta1: float
    delta2: float
    segments: tuple[tuple[float, int], tuple[float, int]]
    target_state: np.ndarray


# Each training summary (T, center) is unique.  Only FORWARD paths are used
# for fitting; the corresponding reverses form the orientation-flip test.
TRAIN_SPECS = (
    Spec("tr01", 7200, -0.380, 0.25, 0.08),
    Spec("tr02", 7488, -0.360, 0.35, 0.13),
    Spec("tr03", 7776, -0.340, 0.45, 0.18),
    Spec("tr04", 8064, -0.320, 0.55, 0.10),
    Spec("tr05", 8352, -0.300, 0.65, 0.15),
    Spec("tr06", 8640, -0.280, 0.75, 0.20),
    Spec("tr07", 8928, -0.260, 0.30, 0.17),
    Spec("tr08", 9216, -0.240, 0.70, 0.12),
)

# Both orientations of these new parameter settings are held out.
HELDOUT_SPECS = (
    Spec("ho01", 7344, -0.350, 0.20, 0.16),
    Spec("ho02", 7920, -0.330, 0.40, 0.22),
    Spec("ho03", 8496, -0.290, 0.60, 0.09),
    Spec("ho04", 9072, -0.250, 0.80, 0.14),
    Spec("ho05", 8784, -0.370, 0.50, 0.19),
    Spec("ho06", 7632, -0.270, 0.33, 0.11),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="N=2 fixed-H versus path-aware learning audit."
    )
    parser.add_argument(
        "--starts", type=int, default=8,
        help="Deterministic multi-start count per model (default: 8)."
    )
    parser.add_argument(
        "--max-nfev", type=int, default=2500,
        help="Maximum least-squares evaluations per start."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory. Default: timestamped directory."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[notebook] ignored kernel arguments: {unknown}")
    if args.starts < 2:
        parser.error("--starts must be at least 2")
    return args


def script_path_if_available() -> Path | None:
    value = globals().get("__file__")
    if not value:
        return None
    path = Path(value).resolve()
    return path if path.exists() else None


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [[float(z.real), float(z.imag)] for z in value.ravel()]
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def normalize_state(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128).ravel()
    norm = float(np.linalg.norm(psi))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Invalid state norm.")
    return psi / norm


def projective_residual(psi: np.ndarray, phi: np.ndarray) -> float:
    """Phase-aligned state residual, stable near identical states."""
    psi = normalize_state(psi)
    phi = normalize_state(phi)
    overlap = np.vdot(psi, phi)
    if abs(overlap) > 0.0:
        phi = phi * np.exp(-1j * np.angle(overlap))
    return float(np.linalg.norm(psi - phi))


def pure_trace_distance(psi: np.ndarray, phi: np.ndarray) -> float:
    """Cancellation-free D=sqrt(1-|<psi|phi>|^2)."""
    r = min(projective_residual(psi, phi), math.sqrt(2.0))
    return float(r * math.sqrt(max(0.0, 1.0 - 0.25 * r * r)))


def probabilities(psi: np.ndarray) -> np.ndarray:
    p = np.abs(normalize_state(psi)) ** 2
    return p / p.sum()


def tvd(psi: np.ndarray, phi: np.ndarray) -> float:
    return float(0.5 * np.abs(probabilities(psi) - probabilities(phi)).sum())


def shared_endpoint_minimax(
    psi: np.ndarray, phi: np.ndarray
) -> dict[str, Any]:
    """
    Exact obstruction for any order-blind shared fixed-H endpoint.

    With one fixed input and one common duration/summary, a shared
    time-independent generator returns one ray for both schedules.  The
    projective midpoint minimizes the larger trace distance to the two target
    rays.  Its optimum is sqrt((1-|<psi|phi>|)/2).
    """
    psi = normalize_state(psi)
    phi = normalize_state(phi)
    overlap = np.vdot(psi, phi)
    overlap_abs_direct = float(np.clip(abs(overlap), 0.0, 1.0))
    if abs(overlap) > 0.0:
        phi_aligned = phi * np.exp(-1j * np.angle(overlap))
    else:
        phi_aligned = phi
    pair_residual = min(
        float(np.linalg.norm(psi - phi_aligned)),
        math.sqrt(2.0),
    )
    # Stable identities:
    #   r^2 = 2(1-|<psi|phi>|),
    #   D_minimax = sqrt((1-|<psi|phi>|)/2) = r/2.
    # They avoid catastrophic cancellation in 1-|overlap| at gap=0.
    overlap_abs = float(np.clip(
        1.0 - 0.5 * pair_residual * pair_residual, 0.0, 1.0
    ))
    analytic_bound = 0.5 * pair_residual
    mean_infidelity_bound = 0.25 * pair_residual * pair_residual
    midpoint = normalize_state(psi + phi_aligned)
    distance_to_first = pure_trace_distance(midpoint, psi)
    distance_to_second = pure_trace_distance(midpoint, phi)
    verification_error = max(
        abs(distance_to_first - analytic_bound),
        abs(distance_to_second - analytic_bound),
    )
    return {
        "target_overlap_abs": overlap_abs,
        "target_overlap_abs_direct": overlap_abs_direct,
        "overlap_estimator_abs_difference": abs(
            overlap_abs - overlap_abs_direct
        ),
        "target_projective_residual": pair_residual,
        "target_pair_D": pure_trace_distance(psi, phi),
        "analytic_minimax_D_lower_bound": analytic_bound,
        "analytic_minimum_mean_infidelity": mean_infidelity_bound,
        "constructed_midpoint_D_to_first": distance_to_first,
        "constructed_midpoint_D_to_second": distance_to_second,
        "bound_verification_error": verification_error,
        "midpoint_state": midpoint,
    }


def operators_n2() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X1+X2, N=n1+n2, and V=n1*n2 in |00>,|01>,|10>,|11>."""
    ident = np.eye(2, dtype=np.complex128)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    n = np.diag([0.0, 1.0]).astype(np.complex128)
    xsum = np.kron(x, ident) + np.kron(ident, x)
    nsum = np.kron(n, ident) + np.kron(ident, n)
    nn = np.kron(n, n)
    return xsum, nsum, nn


XSUM, NSUM, NN = operators_n2()
PSI0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)


def hamiltonian(detuning: float, theta: Sequence[float]) -> np.ndarray:
    omega, detuning_offset = np.asarray(theta, dtype=float)
    return (
        0.5 * omega * XSUM
        - (float(detuning) + detuning_offset) * NSUM
        + FIXED_INTERACTION * NN
    )


def evolve_path(
    segments: Sequence[tuple[float, int]], theta: Sequence[float]
) -> np.ndarray:
    psi = PSI0.copy()
    for detuning, duration_ns in segments:
        psi = expm(
            -1j * hamiltonian(detuning, theta) * (duration_ns / 1000.0)
        ) @ psi
    return normalize_state(psi)


def evolve_average(
    segments: Sequence[tuple[float, int]], theta: Sequence[float]
) -> np.ndarray:
    total_ns = sum(duration for _, duration in segments)
    average = sum(
        detuning * duration for detuning, duration in segments
    ) / total_ns
    return normalize_state(
        expm(-1j * hamiltonian(average, theta) * (total_ns / 1000.0))
        @ PSI0
    )


def clock_split(total_ns: int, fraction: float) -> tuple[int, int]:
    if total_ns % CLOCK_NS != 0:
        raise ValueError(f"T={total_ns} ns is not clock aligned.")
    duration1 = int(round(total_ns * fraction / CLOCK_NS) * CLOCK_NS)
    duration1 = max(CLOCK_NS, min(total_ns - CLOCK_NS, duration1))
    duration2 = total_ns - duration1
    if duration1 % CLOCK_NS or duration2 % CLOCK_NS:
        raise AssertionError("Clock split failed.")
    return duration1, duration2


def schedules_from_spec(
    spec: Spec,
) -> tuple[
    tuple[tuple[float, int], tuple[float, int]],
    tuple[tuple[float, int], tuple[float, int]],
    float,
]:
    d1, d2 = clock_split(spec.total_ns, spec.fraction_requested)
    f = d1 / spec.total_ns
    delta1 = spec.center - (1.0 - f) * spec.gap
    delta2 = spec.center + f * spec.gap
    forward = ((float(delta1), d1), (float(delta2), d2))
    reverse = ((float(delta2), d2), (float(delta1), d1))
    for schedule in (forward, reverse):
        average = sum(x * d for x, d in schedule) / spec.total_ns
        if abs(average - spec.center) > 2.0e-15:
            raise AssertionError("Weighted-average matching failed.")
        if sum(d for _, d in schedule) != spec.total_ns:
            raise AssertionError("Duration matching failed.")
    return forward, reverse, f


def make_example(
    spec: Spec, split: str, orientation: str, target_theta: np.ndarray
) -> Example:
    forward, reverse, f = schedules_from_spec(spec)
    segments = forward if orientation == "forward" else reverse
    state = evolve_path(segments, target_theta)
    return Example(
        example_id=f"{spec.name}_{orientation}",
        split=split,
        orientation=orientation,
        total_ns=spec.total_ns,
        duration1_ns=forward[0][1],
        duration2_ns=forward[1][1],
        fraction_actual=f,
        gap=spec.gap,
        center=spec.center,
        delta1=forward[0][0],
        delta2=forward[1][0],
        segments=segments,
        target_state=state,
    )


def build_dataset() -> dict[str, list[Example]]:
    train = [
        make_example(spec, "train_forward", "forward", TRUE_THETA)
        for spec in TRAIN_SPECS
    ]
    orientation = [
        make_example(spec, "orientation_reverse", "reverse", TRUE_THETA)
        for spec in TRAIN_SPECS
    ]
    heldout: list[Example] = []
    for spec in HELDOUT_SPECS:
        heldout.append(
            make_example(spec, "heldout_new", "forward", TRUE_THETA)
        )
        heldout.append(
            make_example(spec, "heldout_new", "reverse", TRUE_THETA)
        )

    train_ids = {x.example_id for x in train}
    test_ids = {x.example_id for x in orientation + heldout}
    if train_ids & test_ids:
        raise AssertionError("Train/test leakage.")
    return {
        "train_forward": train,
        "orientation_reverse": orientation,
        "heldout_new": heldout,
    }


def aligned_vector_residual(
    predicted: np.ndarray, target: np.ndarray
) -> np.ndarray:
    predicted = normalize_state(predicted)
    target = normalize_state(target)
    overlap = np.vdot(target, predicted)
    if abs(overlap) > 0.0:
        predicted = predicted * np.exp(-1j * np.angle(overlap))
    difference = predicted - target
    return np.concatenate((difference.real, difference.imag))


def residual_function(
    theta: np.ndarray,
    examples: Sequence[Example],
    predictor: Callable[[Sequence[tuple[float, int]], Sequence[float]],
                        np.ndarray],
) -> np.ndarray:
    blocks = [
        aligned_vector_residual(
            predictor(example.segments, theta), example.target_state
        )
        for example in examples
    ]
    return np.concatenate(blocks)


def deterministic_starts(count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = [
        np.array([1.05, -0.040]),
        np.array([1.40, +0.050]),
    ]
    while len(starts) < count:
        starts.append(rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS))
    return starts[:count]


def fit_model(
    model_name: str,
    predictor: Callable[[Sequence[tuple[float, int]], Sequence[float]],
                        np.ndarray],
    train_examples: Sequence[Example],
    starts: Sequence[np.ndarray],
    max_nfev: int,
) -> dict[str, Any]:
    records = []
    best = None
    started = time.perf_counter()
    for index, initial in enumerate(starts):
        fit = least_squares(
            residual_function,
            x0=np.asarray(initial, dtype=float),
            bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
            args=(train_examples, predictor),
            method="trf",
            x_scale="jac",
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
            max_nfev=max_nfev,
        )
        residual = residual_function(fit.x, train_examples, predictor)
        objective = float(np.dot(residual, residual))
        record = {
            "start_index": index,
            "initial_theta": initial.tolist(),
            "theta": fit.x.tolist(),
            "objective": objective,
            "success": bool(fit.success),
            "status": int(fit.status),
            "message": str(fit.message),
            "nfev": int(fit.nfev),
            "optimality": float(fit.optimality),
        }
        records.append(record)
        if best is None or objective < best["objective"]:
            best = record
    assert best is not None
    if not np.isfinite(best["objective"]):
        raise AssertionError(f"{model_name} optimizer returned non-finite loss.")
    return {
        "model": model_name,
        "best_theta": best["theta"],
        "best_objective": best["objective"],
        "best_success": best["success"],
        "best_status": best["status"],
        "best_message": best["message"],
        "elapsed_sec": time.perf_counter() - started,
        "starts": records,
    }


def evaluate_examples(
    model_name: str,
    predictor: Callable[[Sequence[tuple[float, int]], Sequence[float]],
                        np.ndarray],
    theta: Sequence[float],
    examples: Iterable[Example],
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        predicted = predictor(example.segments, theta)
        rows.append({
            "model": model_name,
            "example_id": example.example_id,
            "split": example.split,
            "orientation": example.orientation,
            "total_ns": example.total_ns,
            "center": example.center,
            "fraction_actual": example.fraction_actual,
            "gap": example.gap,
            "D_prediction_to_target": pure_trace_distance(
                predicted, example.target_state
            ),
            "TVD_prediction_to_target": tvd(
                predicted, example.target_state
            ),
        })
    return rows


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    d = np.array([row["D_prediction_to_target"] for row in rows])
    t = np.array([row["TVD_prediction_to_target"] for row in rows])
    return {
        "n": len(rows),
        "mean_D": float(d.mean()),
        "median_D": float(np.median(d)),
        "max_D": float(d.max()),
        "mean_TVD": float(t.mean()),
        "median_TVD": float(np.median(t)),
        "max_TVD": float(t.max()),
    }


def pair_contrast_rows(
    model_name: str,
    predictor: Callable[[Sequence[tuple[float, int]], Sequence[float]],
                        np.ndarray],
    theta: Sequence[float],
    specs: Sequence[Spec],
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        forward, reverse, f = schedules_from_spec(spec)
        target_f = evolve_path(forward, TRUE_THETA)
        target_r = evolve_path(reverse, TRUE_THETA)
        pred_f = predictor(forward, theta)
        pred_r = predictor(reverse, theta)
        target_d = pure_trace_distance(target_f, target_r)
        predicted_d = pure_trace_distance(pred_f, pred_r)
        target_tvd = tvd(target_f, target_r)
        predicted_tvd = tvd(pred_f, pred_r)
        rows.append({
            "model": model_name,
            "spec": spec.name,
            "split": split,
            "total_ns": spec.total_ns,
            "center": spec.center,
            "fraction_actual": f,
            "gap": spec.gap,
            "target_pair_D": target_d,
            "predicted_pair_D": predicted_d,
            "pair_D_abs_error": abs(predicted_d - target_d),
            "target_pair_TVD": target_tvd,
            "predicted_pair_TVD": predicted_tvd,
            "pair_TVD_abs_error": abs(predicted_tvd - target_tvd),
            "hidden_state_ratio_D_over_TVD": (
                target_d / target_tvd if target_tvd > 0.0 else math.inf
            ),
        })
    return rows


def shared_fixedH_bound_rows(
    specs: Sequence[Spec],
) -> list[dict[str, Any]]:
    """
    Evaluate the unrestricted common-endpoint obstruction for declared pairs.

    No learning model is called here.  The result bounds every representation
    that maps the matched pair to one shared fixed-H endpoint.
    """
    rows: list[dict[str, Any]] = []
    for spec in specs:
        forward, reverse, fraction = schedules_from_spec(spec)
        target_forward = evolve_path(forward, TRUE_THETA)
        target_reverse = evolve_path(reverse, TRUE_THETA)
        bound = shared_endpoint_minimax(target_forward, target_reverse)
        rows.append({
            "spec": spec.name,
            "total_ns": spec.total_ns,
            "center": spec.center,
            "fraction_actual": fraction,
            "gap": spec.gap,
            "target_pair_D": bound["target_pair_D"],
            "target_overlap_abs": bound["target_overlap_abs"],
            "target_overlap_abs_direct": bound[
                "target_overlap_abs_direct"
            ],
            "overlap_estimator_abs_difference": bound[
                "overlap_estimator_abs_difference"
            ],
            "target_projective_residual": bound[
                "target_projective_residual"
            ],
            "shared_fixedH_minimax_D_lower_bound": bound[
                "analytic_minimax_D_lower_bound"
            ],
            "shared_fixedH_minimum_mean_infidelity": bound[
                "analytic_minimum_mean_infidelity"
            ],
            "constructed_midpoint_D_to_forward": bound[
                "constructed_midpoint_D_to_first"
            ],
            "constructed_midpoint_D_to_reverse": bound[
                "constructed_midpoint_D_to_second"
            ],
            "bound_verification_error": bound[
                "bound_verification_error"
            ],
        })
    return rows


def gap_zero_control() -> dict[str, float]:
    spec = Spec("gap0", 8088, -0.31, 0.35, 0.0)
    forward, reverse, _ = schedules_from_spec(spec)
    a = evolve_path(forward, TRUE_THETA)
    b = evolve_path(reverse, TRUE_THETA)
    return {
        "D": pure_trace_distance(a, b),
        "TVD": tvd(a, b),
        "projective_residual": projective_residual(a, b),
    }


def ols_summary(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    design = np.column_stack((x_array, np.ones_like(x_array)))
    slope, intercept = np.linalg.lstsq(design, y_array, rcond=None)[0]
    predicted = slope * x_array + intercept
    residual = y_array - predicted
    ss_res = float(np.dot(residual, residual))
    centered = y_array - y_array.mean()
    ss_tot = float(np.dot(centered, centered))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0,
        "RMSE": float(math.sqrt(ss_res / len(y_array))),
    }


def run_n2_117_scan() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    durations = [
        clock_split(SCAN_TOTAL_NS, fraction)[0]
        for fraction in SCAN_FRACTIONS
    ]
    if len(set(durations)) != len(durations):
        raise AssertionError("117-point fraction grid has duplicate durations.")
    if set(durations) != {SCAN_TOTAL_NS - value for value in durations}:
        raise AssertionError("117-point fraction grid is not reflection closed.")

    rows: list[dict[str, Any]] = []
    point_index = 0
    for gap in SCAN_GAPS:
        for fraction in SCAN_FRACTIONS:
            point_index += 1
            spec = Spec(
                name=f"scan_{point_index:03d}",
                total_ns=SCAN_TOTAL_NS,
                center=SCAN_CENTER,
                fraction_requested=fraction,
                gap=gap,
            )
            forward, reverse, actual_fraction = schedules_from_spec(spec)
            state_forward = evolve_path(forward, TRUE_THETA)
            state_reverse = evolve_path(reverse, TRUE_THETA)
            distance = pure_trace_distance(state_forward, state_reverse)
            distribution_tvd = tvd(state_forward, state_reverse)
            shared_bound = shared_endpoint_minimax(
                state_forward, state_reverse
            )
            rows.append({
                "point_index": point_index,
                "gap": gap,
                "fraction_requested": fraction,
                "fraction_actual": actual_fraction,
                "duration1_ns": forward[0][1],
                "duration2_ns": forward[1][1],
                "delta1": forward[0][0],
                "delta2": forward[1][0],
                "weighted_average_detuning": (
                    forward[0][0] * forward[0][1]
                    + forward[1][0] * forward[1][1]
                ) / SCAN_TOTAL_NS,
                "pure_trace_distance": distance,
                "TVD_distribution": distribution_tvd,
                "projective_residual": projective_residual(
                    state_forward, state_reverse
                ),
                "shared_fixedH_minimax_D_lower_bound": shared_bound[
                    "analytic_minimax_D_lower_bound"
                ],
                "shared_fixedH_minimum_mean_infidelity": shared_bound[
                    "analytic_minimum_mean_infidelity"
                ],
                "shared_fixedH_bound_verification_error": shared_bound[
                    "bound_verification_error"
                ],
                "shared_fixedH_overlap_estimator_abs_difference": (
                    shared_bound["overlap_estimator_abs_difference"]
                ),
                "C_BCH_shape": gap * actual_fraction * (
                    1.0 - actual_fraction
                ),
                "C_linear_support": gap * min(
                    actual_fraction, 1.0 - actual_fraction
                ),
                "D_over_TVD": (
                    distance / distribution_tvd
                    if distribution_tvd > 1.0e-15 else math.nan
                ),
            })

    if len(rows) != SCAN_EXPECTED_POINTS:
        raise AssertionError(
            f"Expected {SCAN_EXPECTED_POINTS} scan rows, got {len(rows)}."
        )
    zero_rows = [row for row in rows if row["gap"] == 0.0]
    nonzero_rows = [row for row in rows if row["gap"] > 0.0]
    if len(zero_rows) != 13 or len(nonzero_rows) != 104:
        raise AssertionError("Unexpected zero/nonzero 117-point split.")

    floor_d = max(row["pure_trace_distance"] for row in zero_rows)
    floor_tvd = max(row["TVD_distribution"] for row in zero_rows)
    weakest_d = min(row["pure_trace_distance"] for row in nonzero_rows)
    weakest_tvd = min(row["TVD_distribution"] for row in nonzero_rows)

    fits = {}
    for proxy in ("C_BCH_shape", "C_linear_support"):
        fits[proxy] = ols_summary(
            [row[proxy] for row in nonzero_rows],
            [row["pure_trace_distance"] for row in nonzero_rows],
        )

    per_gap = []
    for gap in SCAN_GAPS[1:]:
        subset = [row for row in nonzero_rows if row["gap"] == gap]
        bch = ols_summary(
            [row["C_BCH_shape"] for row in subset],
            [row["pure_trace_distance"] for row in subset],
        )
        linear = ols_summary(
            [row["C_linear_support"] for row in subset],
            [row["pure_trace_distance"] for row in subset],
        )
        per_gap.append({
            "gap": gap,
            "C_BCH_RMSE": bch["RMSE"],
            "C_linear_support_RMSE": linear["RMSE"],
            "preferred_within_two_proxy_set": (
                "C_linear_support"
                if linear["RMSE"] < bch["RMSE"]
                else "C_BCH_shape"
            ),
        })

    finite_hidden_ratios = [
        row["D_over_TVD"] for row in nonzero_rows
        if np.isfinite(row["D_over_TVD"])
    ]
    incompatible_rows = [
        row for row in nonzero_rows
        if row["shared_fixedH_minimax_D_lower_bound"]
        >= MIN_RESOLVED_ORDER_D
    ]
    certificate = {
        "N": 2,
        "total_duration_ns": SCAN_TOTAL_NS,
        "weighted_average_detuning": SCAN_CENTER,
        "point_count": len(rows),
        "zero_gap_points": len(zero_rows),
        "nonzero_gap_points": len(nonzero_rows),
        "reflection_closed": True,
        "half_split_present": (SCAN_TOTAL_NS // 2) in durations,
        "max_zero_gap_D_floor": floor_d,
        "max_zero_gap_TVD_floor": floor_tvd,
        "weakest_nonzero_D": weakest_d,
        "weakest_nonzero_TVD": weakest_tvd,
        "weakest_D_to_floor_ratio": weakest_d / max(floor_d, 1.0e-16),
        "max_D": max(row["pure_trace_distance"] for row in nonzero_rows),
        "max_TVD": max(row["TVD_distribution"] for row in nonzero_rows),
        "max_finite_D_over_TVD": max(finite_hidden_ratios),
        "shared_fixedH_obstruction": {
            "representation": (
                "One unrestricted common output ray for both orderings, "
                "equivalent to the endpoint freedom of any shared fixed-H "
                "model for one fixed input and common duration."
            ),
            "incompatible_nonzero_points": len(incompatible_rows),
            "tested_nonzero_points": len(nonzero_rows),
            "incompatible_fraction": (
                len(incompatible_rows) / len(nonzero_rows)
            ),
            "minimum_minimax_D_lower_bound": min(
                row["shared_fixedH_minimax_D_lower_bound"]
                for row in nonzero_rows
            ),
            "median_minimax_D_lower_bound": float(np.median([
                row["shared_fixedH_minimax_D_lower_bound"]
                for row in nonzero_rows
            ])),
            "maximum_minimax_D_lower_bound": max(
                row["shared_fixedH_minimax_D_lower_bound"]
                for row in nonzero_rows
            ),
            "maximum_bound_verification_error": max(
                row["shared_fixedH_bound_verification_error"]
                for row in rows
            ),
            "maximum_overlap_estimator_abs_difference": max(
                row["shared_fixedH_overlap_estimator_abs_difference"]
                for row in rows
            ),
            "resolution_threshold": MIN_RESOLVED_ORDER_D,
        },
        "pooled_proxy_fits": fits,
        "per_gap_proxy_comparison": per_gap,
        "boundary": (
            "This fixed-(T, average-detuning) scan characterizes N=2 order "
            "response. It is not used to train the average-only learner."
        ),
    }
    return rows, certificate


def make_scan_plot(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    nonzero = [row for row in rows if row["gap"] > 0.0]
    norm = plt.Normalize(min(SCAN_GAPS[1:]), max(SCAN_GAPS[1:]))
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 3.9), constrained_layout=True
    )
    for gap in SCAN_GAPS[1:]:
        subset = sorted(
            (row for row in nonzero if row["gap"] == gap),
            key=lambda row: row["fraction_actual"],
        )
        color = cmap(norm(gap))
        axes[0].plot(
            [row["fraction_actual"] for row in subset],
            [row["pure_trace_distance"] for row in subset],
            "o-", ms=3.5, lw=1.2, color=color,
        )
        axes[1].plot(
            [row["fraction_actual"] for row in subset],
            [row["TVD_distribution"] for row in subset],
            "o-", ms=3.5, lw=1.2, color=color,
        )

    axes[0].set_title(r"(a) Full-state order signal")
    axes[0].set_xlabel(r"$f=t_1/T$")
    axes[0].set_ylabel(r"$D_{\rm pure}(\mathrm{forward},\mathrm{reverse})$")
    axes[1].set_title(r"(b) Computational-basis signal")
    axes[1].set_xlabel(r"$f=t_1/T$")
    axes[1].set_ylabel("TVD(forward, reverse)")
    scatter = axes[2].scatter(
        [row["TVD_distribution"] for row in nonzero],
        [row["pure_trace_distance"] for row in nonzero],
        c=[row["gap"] for row in nonzero],
        cmap=cmap,
        norm=norm,
        s=26,
        alpha=0.82,
    )
    axes[2].set_title("(c) State versus count visibility")
    axes[2].set_xlabel("computational-basis TVD")
    axes[2].set_ylabel(r"full-state $D_{\rm pure}$")
    for axis in axes:
        axis.grid(alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=axes[2], fraction=0.050, pad=0.035)
    colorbar.set_label(r"gap $g$ (rad/$\mu$s)")
    fig.suptitle("N=2 117-point pulse-order benchmark", fontsize=13)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty CSV.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(
    path: Path,
    evaluation_rows: Sequence[dict[str, Any]],
    contrast_rows: Sequence[dict[str, Any]],
) -> None:
    models = ("average_only_fixed_H", "path_aware_piecewise_H")
    colors = {
        "average_only_fixed_H": "#D55E00",
        "path_aware_piecewise_H": "#0072B2",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))

    # (a) Held-out orientation prediction errors.
    ax = axes[0]
    positions = np.arange(len(models))
    data = [
        [
            row["D_prediction_to_target"]
            for row in evaluation_rows
            if row["model"] == model
            and row["split"] == "orientation_reverse"
        ]
        for model in models
    ]
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True)
    for patch, model in zip(bp["boxes"], models):
        patch.set_facecolor(colors[model])
        patch.set_alpha(0.70)
    ax.set_xticks(positions, ["average-only", "path-aware"])
    ax.set_ylabel(r"$D_{\rm pure}$ to reverse target")
    ax.set_title("(a) Unseen orientation")
    ax.set_yscale("log")
    ax.set_ylim(1.0e-17, 1.0e-1)
    ax.grid(alpha=0.25, axis="y")

    # (b) Target versus predicted order contrast.
    ax = axes[1]
    maximum = 0.0
    for model in models:
        subset = [
            row for row in contrast_rows
            if row["model"] == model and row["split"] == "all_declared"
        ]
        x = np.array([row["target_pair_D"] for row in subset])
        y = np.array([row["predicted_pair_D"] for row in subset])
        maximum = max(maximum, float(x.max()), float(y.max()))
        ax.scatter(
            x, y, s=42, alpha=0.85, color=colors[model],
            label=("average-only" if model.startswith("average") else
                   "path-aware"),
        )
    ax.plot([0, maximum], [0, maximum], "--", color="black", lw=1)
    ax.set_xlabel(r"target forward--reverse $D_{\rm pure}$")
    ax.set_ylabel(r"predicted forward--reverse $D_{\rm pure}$")
    ax.set_title("(b) Order-contrast recovery")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    # (c) State-visible but count-hidden examples.
    ax = axes[2]
    targets = [
        row for row in contrast_rows
        if row["model"] == "path_aware_piecewise_H"
        and row["split"] == "all_declared"
    ]
    scatter = ax.scatter(
        [row["target_pair_TVD"] for row in targets],
        [row["target_pair_D"] for row in targets],
        c=[row["gap"] for row in targets],
        cmap="viridis", s=48, alpha=0.90,
    )
    ax.set_xlabel("computational-basis TVD")
    ax.set_ylabel(r"full-state $D_{\rm pure}$")
    ax.set_title("(c) Measurement visibility")
    ax.grid(alpha=0.25)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.050, pad=0.03)
    colorbar.set_label(r"gap $g$ (rad/$\mu$s)")

    fig.suptitle(
        "N=2 fixed-H versus path-aware held-out audit", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parameter_report(theta: Sequence[float]) -> dict[str, Any]:
    theta = np.asarray(theta, dtype=float)
    relative = np.abs(theta - TRUE_THETA) / np.maximum(
        np.abs(TRUE_THETA), 1.0e-12
    )
    return {
        "names": list(PARAMETER_NAMES),
        "true": TRUE_THETA.tolist(),
        "estimated": theta.tolist(),
        "absolute_error": np.abs(theta - TRUE_THETA).tolist(),
        "relative_error": relative.tolist(),
        "max_relative_error": float(relative.max()),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(
        args.output_dir
        or f"n2_fixedH_vs_pathaware_{timestamp}"
    )
    outdir.mkdir(parents=True, exist_ok=False)

    print("=" * 92)
    print("N=2 117-POINT + FIXED-H VS PATH-AWARE LEARNING AUDIT")
    print("=" * 92)
    print(f"output={outdir}")
    print(
        "train=forward-only | test=same-task reverse + wholly unseen schedules"
    )

    print("\n" + "=" * 92)
    print("A) INDEPENDENT 117-POINT PULSE-ORDER BENCHMARK")
    print("=" * 92)
    scan_rows, scan_certificate = run_n2_117_scan()
    print(
        f"points={scan_certificate['point_count']} "
        f"(nonzero={scan_certificate['nonzero_gap_points']}) | "
        f"max D={scan_certificate['max_D']:.6f} | "
        f"max TVD={scan_certificate['max_TVD']:.6f} | "
        f"weakest D/floor={scan_certificate['weakest_D_to_floor_ratio']:.3e}"
    )

    print("\n" + "=" * 92)
    print("B) UNRESTRICTED SHARED FIXED-H ENDPOINT OBSTRUCTION")
    print("=" * 92)
    shared_bound_rows = shared_fixedH_bound_rows(
        TRAIN_SPECS + HELDOUT_SPECS
    )
    maximum_bound_verification_error = max(
        row["bound_verification_error"] for row in shared_bound_rows
    )
    minimum_declared_pair_bound = min(
        row["shared_fixedH_minimax_D_lower_bound"]
        for row in shared_bound_rows
    )
    maximum_declared_pair_bound = max(
        row["shared_fixedH_minimax_D_lower_bound"]
        for row in shared_bound_rows
    )
    print(
        "analytic common-endpoint minimax D: "
        f"[{minimum_declared_pair_bound:.6e}, "
        f"{maximum_declared_pair_bound:.6e}] | "
        f"verification error={maximum_bound_verification_error:.3e}"
    )

    print("\n" + "=" * 92)
    print("C) FORWARD-TRAINED FIXED-H VS PATH-AWARE LEARNING TEST")
    print("=" * 92)
    dataset = build_dataset()
    starts = deterministic_starts(args.starts, args.seed)
    models = {
        "average_only_fixed_H": evolve_average,
        "path_aware_piecewise_H": evolve_path,
    }

    fits: dict[str, dict[str, Any]] = {}
    evaluation_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for model_name, predictor in models.items():
        fit = fit_model(
            model_name=model_name,
            predictor=predictor,
            train_examples=dataset["train_forward"],
            starts=starts,
            max_nfev=args.max_nfev,
        )
        fits[model_name] = fit
        theta = fit["best_theta"]
        for split, examples in dataset.items():
            evaluation_rows.extend(
                evaluate_examples(model_name, predictor, theta, examples)
            )
        contrast_rows.extend(
            pair_contrast_rows(
                model_name, predictor, theta,
                TRAIN_SPECS + HELDOUT_SPECS, "all_declared"
            )
        )
        print(
            f"[{model_name}] objective={fit['best_objective']:.3e} "
            f"theta={np.array(theta)} success={fit['best_success']}"
        )

    summaries: dict[str, dict[str, Any]] = {}
    for model_name in models:
        summaries[model_name] = {}
        for split in dataset:
            subset = [
                row for row in evaluation_rows
                if row["model"] == model_name and row["split"] == split
            ]
            summaries[model_name][split] = summarize_rows(subset)
        summaries[model_name]["parameter_recovery"] = parameter_report(
            fits[model_name]["best_theta"]
        )

    path_orientation_max = summaries["path_aware_piecewise_H"][
        "orientation_reverse"
    ]["max_D"]
    avg_orientation_median = summaries["average_only_fixed_H"][
        "orientation_reverse"
    ]["median_D"]
    path_orientation_median = summaries["path_aware_piecewise_H"][
        "orientation_reverse"
    ]["median_D"]
    advantage = avg_orientation_median / max(
        path_orientation_median, 1.0e-16
    )

    target_contrasts = [
        row for row in contrast_rows
        if row["model"] == "path_aware_piecewise_H"
    ]
    max_order_d = max(row["target_pair_D"] for row in target_contrasts)
    max_hidden_ratio = max(
        row["hidden_state_ratio_D_over_TVD"] for row in target_contrasts
        if math.isfinite(row["hidden_state_ratio_D_over_TVD"])
    )

    avg_contrasts = [
        row for row in contrast_rows
        if row["model"] == "average_only_fixed_H"
    ]
    path_contrasts = [
        row for row in contrast_rows
        if row["model"] == "path_aware_piecewise_H"
    ]
    avg_pair_d_max = max(row["predicted_pair_D"] for row in avg_contrasts)
    path_pair_error_max = max(row["pair_D_abs_error"] for row in path_contrasts)

    zero_control = gap_zero_control()
    if max(zero_control.values()) > 1.0e-12:
        raise AssertionError(f"Gap-zero control failed: {zero_control}")

    train_ids = {
        example.example_id for example in dataset["train_forward"]
    }
    test_ids = {
        example.example_id
        for split, examples in dataset.items()
        if split != "train_forward"
        for example in examples
    }
    learning_summaries = {
        (spec.total_ns, spec.center)
        for spec in TRAIN_SPECS + HELDOUT_SPECS
    }
    all_examples = [
        example for examples in dataset.values() for example in examples
    ]
    clock_and_summary_matching = all(
        (
            example.duration1_ns + example.duration2_ns
            == example.total_ns
            and example.duration1_ns % CLOCK_NS == 0
            and example.duration2_ns % CLOCK_NS == 0
            and abs(
                sum(
                    detuning * duration
                    for detuning, duration in example.segments
                ) / example.total_ns
                - example.center
            ) <= 1.0e-12
        )
        for example in all_examples
    )
    gates = {
        "scan_has_117_points": (
            scan_certificate["point_count"] == SCAN_EXPECTED_POINTS
        ),
        "scan_has_104_nonzero_gap_points": (
            scan_certificate["nonzero_gap_points"] == 104
        ),
        "scan_fraction_grid_reflection_closed": (
            scan_certificate["reflection_closed"]
        ),
        "scan_half_split_present": scan_certificate["half_split_present"],
        "shared_fixedH_bound_constructively_verified": (
            maximum_bound_verification_error <= 1.0e-12
        ),
        "scan_summary_excluded_from_learning_specs": (
            (SCAN_TOTAL_NS, SCAN_CENTER) not in learning_summaries
        ),
        "dataset_split_has_no_leakage": train_ids.isdisjoint(test_ids),
        "gap_zero_control": max(zero_control.values()) <= 1.0e-12,
        "all_fits_finite": all(
            np.isfinite(fit["best_objective"]) for fit in fits.values()
        ),
        "all_best_fits_successful": all(
            fit["best_success"] for fit in fits.values()
        ),
        "clock_and_summary_matching": clock_and_summary_matching,
    }
    if not all(gates.values()):
        raise AssertionError(f"Implementation gate failure: {gates}")

    scientific_tests = {
        "N2_117_order_signal_resolved": {
            "status": (
                "SUPPORTED"
                if scan_certificate["weakest_nonzero_D"]
                >= max(
                    MIN_RESOLVED_ORDER_D,
                    50.0 * scan_certificate["max_zero_gap_D_floor"],
                )
                else "NOT_RESOLVED"
            ),
            "weakest_nonzero_D": scan_certificate["weakest_nonzero_D"],
            "zero_gap_D_floor": scan_certificate["max_zero_gap_D_floor"],
            "threshold": max(
                MIN_RESOLVED_ORDER_D,
                50.0 * scan_certificate["max_zero_gap_D_floor"],
            ),
            "fail_fast": False,
        },
        "unrestricted_shared_fixedH_incompatible_on_scan": {
            "status": (
                "SUPPORTED"
                if scan_certificate["shared_fixedH_obstruction"][
                    "incompatible_nonzero_points"
                ] == scan_certificate["nonzero_gap_points"]
                else "PARTIALLY_SUPPORTED"
            ),
            **scan_certificate["shared_fixedH_obstruction"],
            "evidence_class": (
                "Architecture-independent analytic lower bound, verified by "
                "constructing the optimal projective midpoint."
            ),
            "fail_fast": False,
        },
        "minimal_N2_order_signal_resolved": {
            "status": (
                "SUPPORTED" if max_order_d >= MIN_RESOLVED_ORDER_D
                else "NOT_RESOLVED"
            ),
            "max_target_pair_D": max_order_d,
            "threshold": MIN_RESOLVED_ORDER_D,
            "fail_fast": False,
        },
        "average_only_erases_order_contrast": {
            "status": (
                "SUPPORTED" if avg_pair_d_max <= 1.0e-12
                else "NOT_SUPPORTED"
            ),
            "max_predicted_pair_D": avg_pair_d_max,
            "evidence_class": (
                "Structural consequence of the average-only representation; "
                "reported for auditability, not independent physical evidence."
            ),
            "fail_fast": False,
        },
        "path_model_recovers_heldout_orientation": {
            "status": (
                "SUPPORTED" if path_orientation_max <= PATH_MAX_TEST_D_TARGET
                else "NOT_SUPPORTED"
            ),
            "max_orientation_test_D": path_orientation_max,
            "target": PATH_MAX_TEST_D_TARGET,
            "fail_fast": False,
        },
        "path_model_recovers_physical_parameters": {
            "status": (
                "SUPPORTED"
                if summaries["path_aware_piecewise_H"][
                    "parameter_recovery"
                ]["max_relative_error"]
                <= PATH_PARAMETER_RELATIVE_ERROR_TARGET
                else "NOT_SUPPORTED"
            ),
            "max_relative_parameter_error": summaries[
                "path_aware_piecewise_H"
            ]["parameter_recovery"]["max_relative_error"],
            "target": PATH_PARAMETER_RELATIVE_ERROR_TARGET,
            "fail_fast": False,
        },
        "heldout_path_advantage": {
            "status": (
                "SUPPORTED" if advantage >= MIN_PATH_ADVANTAGE_FACTOR
                else "NOT_SUPPORTED"
            ),
            "median_orientation_D_advantage_factor": advantage,
            "threshold": MIN_PATH_ADVANTAGE_FACTOR,
            "fail_fast": False,
        },
        "fixed_H_failure_boundary_localized": {
            "status": (
                "SUPPORTED"
                if (
                    avg_pair_d_max <= 1.0e-12
                    and max_order_d >= MIN_RESOLVED_ORDER_D
                    and path_orientation_max <= PATH_MAX_TEST_D_TARGET
                )
                else "NOT_SUPPORTED"
            ),
            "boundary": (
                "Failure occurs when two schedules share the fixed-H input "
                "summary but have resolved distinct target rays. The "
                "summary-only model must identify them; the ordered model "
                "retains the missing variable."
            ),
            "average_model_max_predicted_order_D": avg_pair_d_max,
            "maximum_target_order_D": max_order_d,
            "path_model_max_heldout_orientation_D": path_orientation_max,
            "fail_fast": False,
        },
    }

    question_closure = {
        "Q1_independent_117_exact_state_scan": {
            "status": scientific_tests[
                "N2_117_order_signal_resolved"
            ]["status"],
            "evidence": "117 exact-state points; 104 nonzero-gap points.",
        },
        "Q2_minimal_interacting_N2_adaptation": {
            "status": scientific_tests[
                "minimal_N2_order_signal_resolved"
            ]["status"],
            "evidence": "Two interacting atoms in a four-dimensional Hilbert space.",
        },
        "Q3_shared_fixedH_distinguishability": {
            "status": scientific_tests[
                "unrestricted_shared_fixedH_incompatible_on_scan"
            ]["status"],
            "evidence": (
                "Exact minimax lower bound for any common endpoint, not only "
                "for the restricted average-only optimizer."
            ),
        },
        "Q4_practical_failure_boundary": {
            "status": scientific_tests[
                "fixed_H_failure_boundary_localized"
            ]["status"],
            "evidence": (
                "Forward-only training followed by unseen reverses and wholly "
                "unseen schedules; ordered model also recovers hidden parameters."
            ),
        },
    }

    source_path = script_path_if_available()
    result = {
        "status": (
            "VALID"
            if all(gates.values())
            else "INVALID"
        ),
        "scientific_status": (
            "PATH_INFORMATION_REQUIRED_IN_TESTED_N2_AUDIT"
            if all(
                test["status"] == "SUPPORTED"
                for test in scientific_tests.values()
            )
            else "MIXED_OR_NOT_RESOLVED"
        ),
        "scope": (
            "An independent fixed-summary 117-point N=2 order-response "
            "benchmark, followed by exact synthetic N=2 Rydberg-family "
            "calibration recovery. Forward-only training; same-task reverse "
            "and wholly new-schedule tests."
        ),
        "N2_117_scan": scan_certificate,
        "N2_117_scan_role": (
            "Independent physical-response benchmark only. Its 117 points are "
            "not training data for either learning model."
        ),
        "data_independence": {
            "historical_scripts_imported": False,
            "historical_results_loaded": False,
            "cached_targets_loaded": False,
            "targets_generated_fresh_in_current_run": True,
            "scan_used_for_learning_fit": False,
        },
        "shared_fixedH_pair_bounds": shared_bound_rows,
        "shared_fixedH_pair_bound_summary": {
            "pair_count": len(shared_bound_rows),
            "minimum_minimax_D_lower_bound": minimum_declared_pair_bound,
            "maximum_minimax_D_lower_bound": maximum_declared_pair_bound,
            "maximum_constructive_verification_error": (
                maximum_bound_verification_error
            ),
        },
        "question_closure": question_closure,
        "true_parameters": dict(zip(PARAMETER_NAMES, TRUE_THETA.tolist())),
        "fixed_interaction_rad_per_us": FIXED_INTERACTION,
        "train_specs": [asdict(x) for x in TRAIN_SPECS],
        "heldout_specs": [asdict(x) for x in HELDOUT_SPECS],
        "fits": fits,
        "summaries": summaries,
        "orientation_advantage_factor": advantage,
        "orientation_advantage_denominator_note": (
            "The path-aware denominator reaches the exact synthetic numerical "
            "floor because targets and the path-aware learner share the "
            "declared Hamiltonian family; interpret the factor as a model-"
            "adequacy diagnostic, not a hardware performance ratio."
        ),
        "max_target_order_D": max_order_d,
        "max_hidden_state_ratio_D_over_TVD": max_hidden_ratio,
        "average_model_max_predicted_order_D": avg_pair_d_max,
        "path_model_max_pair_D_error": path_pair_error_max,
        "gap_zero_control": zero_control,
        "implementation_gates": gates,
        "scientific_tests": scientific_tests,
        "claim_boundary": (
            "This is a noiseless exact synthetic two-atom audit. The "
            "117-point scan establishes order sensitivity over a declared "
            "fixed-summary grid; the separate learning test uses a declared "
            "two-parameter calibration family with fixed interaction. It is "
            "not QPU evidence, not "
            "a generic variational-circuit benchmark, and not proof that all "
            "Hamiltonian-learning models require this representation. It "
            "tests only whether the specified average-only compression loses "
            "held-out order information relative to the matched path-aware "
            "model."
        ),
        "provenance": {
            "script_version": SCRIPT_VERSION,
            "source_sha256": sha256_file(source_path),
            "source_hash_available": source_path is not None,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "seed": args.seed,
            "starts": args.starts,
            "max_nfev": args.max_nfev,
            "elapsed_sec": time.perf_counter() - started,
        },
    }

    write_csv(outdir / "n2_117_scan.csv", scan_rows)
    write_csv(
        outdir / "shared_fixedH_endpoint_bounds.csv",
        shared_bound_rows,
    )
    write_csv(outdir / "evaluation_rows.csv", evaluation_rows)
    write_csv(outdir / "pair_contrast_rows.csv", contrast_rows)
    (outdir / "certificate.json").write_text(
        json.dumps(jsonable(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    make_plot(
        outdir / "n2_fixedH_vs_pathaware.png",
        evaluation_rows,
        contrast_rows,
    )
    make_scan_plot(outdir / "n2_117_scan.png", scan_rows)

    print("\n" + "=" * 92)
    print("GLOBAL VERDICT")
    print("=" * 92)
    compact = {
        "status": result["status"],
        "scientific_status": result["scientific_status"],
        "N2_117_scan_points": scan_certificate["point_count"],
        "N2_117_nonzero_gap_points": scan_certificate[
            "nonzero_gap_points"
        ],
        "N2_117_max_D": scan_certificate["max_D"],
        "N2_117_max_TVD": scan_certificate["max_TVD"],
        "N2_117_weakest_D_to_floor_ratio": scan_certificate[
            "weakest_D_to_floor_ratio"
        ],
        "N2_117_shared_fixedH_incompatible_points": scan_certificate[
            "shared_fixedH_obstruction"
        ]["incompatible_nonzero_points"],
        "N2_117_shared_fixedH_tested_nonzero_points": scan_certificate[
            "shared_fixedH_obstruction"
        ]["tested_nonzero_points"],
        "shared_fixedH_declared_pair_minimax_D_range": [
            minimum_declared_pair_bound,
            maximum_declared_pair_bound,
        ],
        "question_closure": question_closure,
        "average_orientation_median_D": avg_orientation_median,
        "path_orientation_median_D": path_orientation_median,
        "path_orientation_max_D": path_orientation_max,
        "orientation_advantage_factor": advantage,
        "max_target_order_D": max_order_d,
        "max_hidden_state_ratio_D_over_TVD": max_hidden_ratio,
        "average_model_max_predicted_order_D": avg_pair_d_max,
        "path_model_max_pair_D_error": path_pair_error_max,
        "path_parameter_max_relative_error": summaries[
            "path_aware_piecewise_H"
        ]["parameter_recovery"]["max_relative_error"],
        "scientific_tests": scientific_tests,
        "claim_boundary": result["claim_boundary"],
    }
    print(json.dumps(jsonable(compact), indent=2, ensure_ascii=False))
    print(f"elapsed={time.perf_counter() - started:.2f}s")
    print(f"outputs={outdir}")


if __name__ == "__main__":
    main()
