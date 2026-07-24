#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N=2 117-point pulse-order scan plus fixed-H versus path-aware
Hamiltonian-learning audit.
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
from typing import Any, Sequence

import site
import sysconfig

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import expm
from scipy.optimize import least_squares


SCRIPT_VERSION = "3.4"
SEED = 20260725
CLOCK_NS = 4
SPACING_UM = 8.0

TRUE_OMEGA = 1.22
TRUE_DETUNING_OFFSET = 0.017
FIXED_INTERACTION = 20.70
TRUE_THETA = np.array([TRUE_OMEGA, TRUE_DETUNING_OFFSET], dtype=float)

PARAMETER_NAMES = ("Omega_rad_per_us", "detuning_offset_rad_per_us")
LOWER_BOUNDS = np.array([0.75, -0.12], dtype=float)
UPPER_BOUNDS = np.array([1.65, +0.12], dtype=float)

PATH_MAX_TEST_D_TARGET = 1.0e-6
PATH_PARAMETER_RELATIVE_ERROR_TARGET = 1.0e-4
MIN_RESOLVED_ORDER_D = 1.0e-4
MIN_PATH_ADVANTAGE_FACTOR = 10.0
COMPUTATIONAL_BASIS_VISIBILITY_THRESHOLD = 1.0e-4
FLOAT64_EPS = float(np.finfo(np.float64).eps)
FLOOR_EPSILON_UNITS = 8.0

SCAN_TOTAL_NS = 8088
SCAN_CENTER = -0.31
SCAN_GAPS = tuple(float(x) for x in np.linspace(0.0, 0.24, 9))
SCAN_FRACTIONS = tuple(float(x) for x in np.linspace(0.10, 0.90, 13))
SCAN_EXPECTED_POINTS = 117

MODULES_AT_IMPORT = frozenset(sys.modules)


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
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=2500)
    parser.add_argument("--output-dir", type=str, default=None)
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
    if isinstance(value, (list, tuple, frozenset, set)):
        return [jsonable(v) for v in value]
    return value


def float64_floor_report(value: float) -> dict[str, Any]:
    value = float(value)
    threshold = FLOOR_EPSILON_UNITS * FLOAT64_EPS
    floor_limited = abs(value) <= threshold
    return {
        "display": (
            (
                f"< {FLOOR_EPSILON_UNITS:g} eps "
                f"({threshold:.3e}, float64 floor)"
            )
            if floor_limited else f"{value:.6e}"
        ),
        "raw_value": value,
        "absolute_float64_epsilon_units": abs(value) / FLOAT64_EPS,
        "floor_classification_threshold_epsilon_units": FLOOR_EPSILON_UNITS,
        "floor_classification_threshold": threshold,
        "floor_limited": floor_limited,
    }


def normalize_state(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128).ravel()
    norm = float(np.linalg.norm(psi))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Invalid state norm.")
    return psi / norm


def projective_residual(psi: np.ndarray, phi: np.ndarray) -> float:
    psi = normalize_state(psi)
    phi = normalize_state(phi)
    overlap = np.vdot(psi, phi)
    if abs(overlap) > 0.0:
        phi = phi * np.exp(-1j * np.angle(overlap))
    return float(np.linalg.norm(psi - phi))


def pure_trace_distance(psi: np.ndarray, phi: np.ndarray) -> float:
    r = min(projective_residual(psi, phi), math.sqrt(2.0))
    return float(r * math.sqrt(max(0.0, 1.0 - 0.25 * r * r)))


def probabilities(psi: np.ndarray) -> np.ndarray:
    p = np.abs(normalize_state(psi)) ** 2
    return p / p.sum()


def tvd(psi: np.ndarray, phi: np.ndarray) -> float:
    return float(0.5 * np.abs(probabilities(psi) - probabilities(phi)).sum())


def shared_endpoint_minimax(psi, phi) -> dict[str, Any]:
    psi = normalize_state(psi)
    phi = normalize_state(phi)
    overlap = np.vdot(psi, phi)
    overlap_abs_direct = float(np.clip(abs(overlap), 0.0, 1.0))
    if abs(overlap) > 0.0:
        phi_aligned = phi * np.exp(-1j * np.angle(overlap))
    else:
        phi_aligned = phi
    pair_residual = min(float(np.linalg.norm(psi - phi_aligned)), math.sqrt(2.0))
    overlap_abs = float(np.clip(1.0 - 0.5 * pair_residual * pair_residual, 0.0, 1.0))
    analytic_bound = 0.5 * pair_residual
    mean_infidelity_bound = 0.25 * pair_residual * pair_residual
    midpoint = normalize_state(psi + phi_aligned)
    distance_to_first = pure_trace_distance(midpoint, psi)
    distance_to_second = pure_trace_distance(midpoint, phi)
    verification_error = max(
        abs(distance_to_first - analytic_bound),
        abs(distance_to_second - analytic_bound),
    )
    conservative_numeric_lower_bound = max(0.0, analytic_bound - verification_error)
    return {
        "target_overlap_abs": overlap_abs,
        "target_overlap_abs_direct": overlap_abs_direct,
        "overlap_estimator_abs_difference": abs(overlap_abs - overlap_abs_direct),
        "target_projective_residual": pair_residual,
        "target_pair_D": pure_trace_distance(psi, phi),
        "analytic_minimax_D_lower_bound": analytic_bound,
        "conservative_numeric_minimax_D_lower_bound": conservative_numeric_lower_bound,
        "analytic_minimum_mean_infidelity": mean_infidelity_bound,
        "constructed_midpoint_D_to_first": distance_to_first,
        "constructed_midpoint_D_to_second": distance_to_second,
        "bound_verification_error": verification_error,
        "midpoint_state": midpoint,
    }


def operators_n2():
    ident = np.eye(2, dtype=np.complex128)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    n = np.diag([0.0, 1.0]).astype(np.complex128)
    return (np.kron(x, ident) + np.kron(ident, x),
            np.kron(n, ident) + np.kron(ident, n),
            np.kron(n, n))


XSUM, NSUM, NN = operators_n2()
PSI0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)


def hamiltonian(detuning: float, theta: Sequence[float]) -> np.ndarray:
    omega, detuning_offset = np.asarray(theta, dtype=float)
    return (0.5 * omega * XSUM
            - (float(detuning) + detuning_offset) * NSUM
            + FIXED_INTERACTION * NN)


def evolve_path(segments, theta) -> np.ndarray:
    psi = PSI0.copy()
    for detuning, duration_ns in segments:
        psi = expm(-1j * hamiltonian(detuning, theta) * (duration_ns / 1000.0)) @ psi
    return normalize_state(psi)


def evolve_average(segments, theta) -> np.ndarray:
    total_ns = sum(duration for _, duration in segments)
    average = sum(d * t for d, t in segments) / total_ns
    return normalize_state(
        expm(-1j * hamiltonian(average, theta) * (total_ns / 1000.0)) @ PSI0)


def clock_split(total_ns: int, fraction: float) -> tuple[int, int]:
    if total_ns % CLOCK_NS != 0:
        raise ValueError(f"T={total_ns} ns is not clock aligned.")
    duration1 = int(round(total_ns * fraction / CLOCK_NS) * CLOCK_NS)
    duration1 = max(CLOCK_NS, min(total_ns - CLOCK_NS, duration1))
    duration2 = total_ns - duration1
    if duration1 % CLOCK_NS or duration2 % CLOCK_NS:
        raise AssertionError("Clock split failed.")
    return duration1, duration2


def schedules_from_spec(spec: Spec):
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


def make_example(spec, split, orientation, target_theta) -> Example:
    forward, reverse, f = schedules_from_spec(spec)
    segments = forward if orientation == "forward" else reverse
    return Example(
        example_id=f"{spec.name}_{orientation}", split=split,
        orientation=orientation, total_ns=spec.total_ns,
        duration1_ns=forward[0][1], duration2_ns=forward[1][1],
        fraction_actual=f, gap=spec.gap, center=spec.center,
        delta1=forward[0][0], delta2=forward[1][0], segments=segments,
        target_state=evolve_path(segments, target_theta))


def build_dataset() -> dict[str, list[Example]]:
    train = [make_example(s, "train_forward", "forward", TRUE_THETA) for s in TRAIN_SPECS]
    orientation = [make_example(s, "orientation_reverse", "reverse", TRUE_THETA) for s in TRAIN_SPECS]
    heldout: list[Example] = []
    for spec in HELDOUT_SPECS:
        heldout.append(make_example(spec, "heldout_new", "forward", TRUE_THETA))
        heldout.append(make_example(spec, "heldout_new", "reverse", TRUE_THETA))
    if {x.example_id for x in train} & {x.example_id for x in orientation + heldout}:
        raise AssertionError("Train/test leakage.")
    return {"train_forward": train, "orientation_reverse": orientation,
            "heldout_new": heldout}


def aligned_vector_residual(predicted, target) -> np.ndarray:
    predicted = normalize_state(predicted)
    target = normalize_state(target)
    overlap = np.vdot(target, predicted)
    if abs(overlap) > 0.0:
        predicted = predicted * np.exp(-1j * np.angle(overlap))
    difference = predicted - target
    return np.concatenate((difference.real, difference.imag))


def residual_function(theta, examples, predictor) -> np.ndarray:
    return np.concatenate([
        aligned_vector_residual(predictor(e.segments, theta), e.target_state)
        for e in examples])


def deterministic_starts(count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = [np.array([1.05, -0.040]), np.array([1.40, +0.050])]
    while len(starts) < count:
        starts.append(rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS))
    return starts[:count]


def fit_model(model_name, predictor, train_examples, starts, max_nfev) -> dict[str, Any]:
    records = []
    best = None
    started = time.perf_counter()
    for index, initial in enumerate(starts):
        fit = least_squares(
            residual_function, x0=np.asarray(initial, dtype=float),
            bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
            args=(train_examples, predictor), method="trf", x_scale="jac",
            ftol=1.0e-12, xtol=1.0e-12, gtol=1.0e-12, max_nfev=max_nfev)
        residual = residual_function(fit.x, train_examples, predictor)
        objective = float(np.dot(residual, residual))
        record = {
            "start_index": index, "initial_theta": initial.tolist(),
            "theta": fit.x.tolist(), "objective": objective,
            "success": bool(fit.success), "status": int(fit.status),
            "message": str(fit.message), "nfev": int(fit.nfev),
            "optimality": float(fit.optimality)}
        records.append(record)
        if best is None or objective < best["objective"]:
            best = record
    if best is None:
        raise AssertionError(f"{model_name}: no optimizer start produced a result.")
    if not np.isfinite(best["objective"]):
        raise AssertionError(f"{model_name} optimizer returned non-finite loss.")
    return {"model": model_name, "best_theta": best["theta"],
            "best_objective": best["objective"], "best_success": best["success"],
            "best_status": best["status"], "best_message": best["message"],
            "elapsed_sec": time.perf_counter() - started, "starts": records}


def evaluate_examples(model_name, predictor, theta, examples) -> list[dict[str, Any]]:
    rows = []
    for e in examples:
        predicted = predictor(e.segments, theta)
        rows.append({
            "model": model_name, "example_id": e.example_id, "split": e.split,
            "orientation": e.orientation, "total_ns": e.total_ns,
            "center": e.center, "fraction_actual": e.fraction_actual, "gap": e.gap,
            "D_prediction_to_target": pure_trace_distance(predicted, e.target_state),
            "TVD_prediction_to_target": tvd(predicted, e.target_state)})
    return rows


def summarize_rows(rows) -> dict[str, Any]:
    d = np.array([r["D_prediction_to_target"] for r in rows])
    t = np.array([r["TVD_prediction_to_target"] for r in rows])
    return {"n": len(rows), "mean_D": float(d.mean()), "median_D": float(np.median(d)),
            "max_D": float(d.max()), "mean_TVD": float(t.mean()),
            "median_TVD": float(np.median(t)), "max_TVD": float(t.max())}


def pair_contrast_rows(model_name, predictor, theta, specs, split) -> list[dict[str, Any]]:
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
            "model": model_name, "spec": spec.name, "split": split,
            "total_ns": spec.total_ns, "center": spec.center,
            "fraction_actual": f, "gap": spec.gap,
            "target_pair_D": target_d, "predicted_pair_D": predicted_d,
            "pair_D_abs_error": abs(predicted_d - target_d),
            "target_pair_TVD": target_tvd, "predicted_pair_TVD": predicted_tvd,
            "pair_TVD_abs_error": abs(predicted_tvd - target_tvd),
            "hidden_state_ratio_D_over_TVD": (
                target_d / target_tvd if target_tvd > 0.0 else math.inf)})
    return rows


def shared_fixedH_bound_rows(specs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        forward, reverse, fraction = schedules_from_spec(spec)
        bound = shared_endpoint_minimax(evolve_path(forward, TRUE_THETA),
                                        evolve_path(reverse, TRUE_THETA))
        rows.append({
            "spec": spec.name, "total_ns": spec.total_ns, "center": spec.center,
            "fraction_actual": fraction, "gap": spec.gap,
            "target_pair_D": bound["target_pair_D"],
            "target_overlap_abs": bound["target_overlap_abs"],
            "target_overlap_abs_direct": bound["target_overlap_abs_direct"],
            "overlap_estimator_abs_difference": bound["overlap_estimator_abs_difference"],
            "target_projective_residual": bound["target_projective_residual"],
            "shared_fixedH_minimax_D_lower_bound": bound["analytic_minimax_D_lower_bound"],
            "shared_fixedH_conservative_numeric_D_lower_bound":
                bound["conservative_numeric_minimax_D_lower_bound"],
            "shared_fixedH_minimum_mean_infidelity": bound["analytic_minimum_mean_infidelity"],
            "constructed_midpoint_D_to_forward": bound["constructed_midpoint_D_to_first"],
            "constructed_midpoint_D_to_reverse": bound["constructed_midpoint_D_to_second"],
            "bound_verification_error": bound["bound_verification_error"]})
    return rows


def gap_zero_control() -> dict[str, float]:
    spec = Spec("gap0", 8088, -0.31, 0.35, 0.0)
    forward, reverse, _ = schedules_from_spec(spec)
    a = evolve_path(forward, TRUE_THETA)
    b = evolve_path(reverse, TRUE_THETA)
    return {"D": pure_trace_distance(a, b), "TVD": tvd(a, b),
            "projective_residual": projective_residual(a, b)}


def ols_summary(x, y) -> dict[str, float]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    design = np.column_stack((x_array, np.ones_like(x_array)))
    slope, intercept = np.linalg.lstsq(design, y_array, rcond=None)[0]
    residual = y_array - (slope * x_array + intercept)
    ss_res = float(np.dot(residual, residual))
    centered = y_array - y_array.mean()
    ss_tot = float(np.dot(centered, centered))
    return {"slope": float(slope), "intercept": float(intercept),
            "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0,
            "RMSE": float(math.sqrt(ss_res / len(y_array)))}


def run_n2_117_scan():
    durations = [clock_split(SCAN_TOTAL_NS, f)[0] for f in SCAN_FRACTIONS]
    if len(set(durations)) != len(durations):
        raise AssertionError("117-point fraction grid has duplicate durations.")
    if set(durations) != {SCAN_TOTAL_NS - v for v in durations}:
        raise AssertionError("117-point fraction grid is not reflection closed.")

    rows: list[dict[str, Any]] = []
    point_index = 0
    for gap in SCAN_GAPS:
        for fraction in SCAN_FRACTIONS:
            point_index += 1
            spec = Spec(f"scan_{point_index:03d}", SCAN_TOTAL_NS, SCAN_CENTER,
                        fraction, gap)
            forward, reverse, actual_fraction = schedules_from_spec(spec)
            sf = evolve_path(forward, TRUE_THETA)
            sr = evolve_path(reverse, TRUE_THETA)
            distance = pure_trace_distance(sf, sr)
            distribution_tvd = tvd(sf, sr)
            shared_bound = shared_endpoint_minimax(sf, sr)
            rows.append({
                "point_index": point_index, "gap": gap,
                "fraction_requested": fraction, "fraction_actual": actual_fraction,
                "duration1_ns": forward[0][1], "duration2_ns": forward[1][1],
                "delta1": forward[0][0], "delta2": forward[1][0],
                "weighted_average_detuning": (
                    forward[0][0] * forward[0][1] + forward[1][0] * forward[1][1]
                ) / SCAN_TOTAL_NS,
                "pure_trace_distance": distance,
                "TVD_distribution": distribution_tvd,
                "projective_residual": projective_residual(sf, sr),
                "shared_fixedH_minimax_D_lower_bound":
                    shared_bound["analytic_minimax_D_lower_bound"],
                "shared_fixedH_conservative_numeric_D_lower_bound":
                    shared_bound["conservative_numeric_minimax_D_lower_bound"],
                "shared_fixedH_minimum_mean_infidelity":
                    shared_bound["analytic_minimum_mean_infidelity"],
                "shared_fixedH_bound_verification_error":
                    shared_bound["bound_verification_error"],
                "shared_fixedH_overlap_estimator_abs_difference":
                    shared_bound["overlap_estimator_abs_difference"],
                "C_BCH_shape": gap * actual_fraction * (1.0 - actual_fraction),
                "C_linear_support": gap * min(actual_fraction, 1.0 - actual_fraction),
                "D_over_TVD": (distance / distribution_tvd
                               if distribution_tvd > 1.0e-15 else math.nan)})

    if len(rows) != SCAN_EXPECTED_POINTS:
        raise AssertionError(f"Expected {SCAN_EXPECTED_POINTS} rows, got {len(rows)}.")
    zero_rows = [r for r in rows if r["gap"] == 0.0]
    nonzero_rows = [r for r in rows if r["gap"] > 0.0]
    if len(zero_rows) != 13 or len(nonzero_rows) != 104:
        raise AssertionError("Unexpected zero/nonzero 117-point split.")

    floor_d = max(r["pure_trace_distance"] for r in zero_rows)
    floor_tvd = max(r["TVD_distribution"] for r in zero_rows)
    weakest_d = min(r["pure_trace_distance"] for r in nonzero_rows)
    weakest_tvd = min(r["TVD_distribution"] for r in nonzero_rows)

    fits = {p: ols_summary([r[p] for r in nonzero_rows],
                           [r["pure_trace_distance"] for r in nonzero_rows])
            for p in ("C_BCH_shape", "C_linear_support")}

    per_gap = []
    for gap in SCAN_GAPS[1:]:
        subset = [r for r in nonzero_rows if r["gap"] == gap]
        bch = ols_summary([r["C_BCH_shape"] for r in subset],
                          [r["pure_trace_distance"] for r in subset])
        linear = ols_summary([r["C_linear_support"] for r in subset],
                             [r["pure_trace_distance"] for r in subset])
        per_gap.append({"gap": gap, "C_BCH_RMSE": bch["RMSE"],
                        "C_linear_support_RMSE": linear["RMSE"],
                        "preferred_within_two_proxy_set": (
                            "C_linear_support" if linear["RMSE"] < bch["RMSE"]
                            else "C_BCH_shape")})

    finite_hidden_ratios = [r["D_over_TVD"] for r in nonzero_rows
                            if np.isfinite(r["D_over_TVD"])]
    max_scan_d = max(r["pure_trace_distance"] for r in nonzero_rows)
    max_scan_tvd = max(r["TVD_distribution"] for r in nonzero_rows)
    incompatible_rows = [r for r in nonzero_rows
                         if r["shared_fixedH_conservative_numeric_D_lower_bound"]
                         >= MIN_RESOLVED_ORDER_D]
    certificate = {
        "N": 2, "total_duration_ns": SCAN_TOTAL_NS,
        "weighted_average_detuning": SCAN_CENTER, "point_count": len(rows),
        "zero_gap_points": len(zero_rows), "nonzero_gap_points": len(nonzero_rows),
        "reflection_closed": True,
        "half_split_present": (SCAN_TOTAL_NS // 2) in durations,
        "max_zero_gap_D_floor": floor_d, "max_zero_gap_TVD_floor": floor_tvd,
        "weakest_nonzero_D": weakest_d, "weakest_nonzero_TVD": weakest_tvd,
        "weakest_D_to_float64_floor_ratio_not_for_inference": (
            weakest_d / max(floor_d, 1.0e-16)),
        "zero_gap_D_floor_report": float64_floor_report(floor_d),
        "max_D": max_scan_d, "max_TVD": max_scan_tvd,
        "max_finite_D_over_TVD": max(finite_hidden_ratios),
        "witness_channel_visibility": {
            "full_state_signal_resolved": max_scan_d >= MIN_RESOLVED_ORDER_D,
            "computational_basis_visibility_suppressed": (
                max_scan_tvd < COMPUTATIONAL_BASIS_VISIBILITY_THRESHOLD),
            "computational_basis_visibility_threshold":
                COMPUTATIONAL_BASIS_VISIBILITY_THRESHOLD,
            "maximum_full_state_D": max_scan_d,
            "maximum_computational_basis_TVD": max_scan_tvd,
            "maximum_pointwise_D_over_TVD": max(finite_hidden_ratios),
            "interpretation": (
                "In this N=2 scan the resolved order information is "
                "predominantly phase/coherence encoded and is strongly "
                "suppressed in computational-basis probabilities."),
            "claim_role": "SCIENTIFIC_OUTCOME_NOT_IMPLEMENTATION_GATE"},
        "shared_fixedH_obstruction": {
            "representation": (
                "One unrestricted common output ray for both orderings, "
                "equivalent to the endpoint freedom of any shared fixed-H "
                "model for one fixed input and common duration."),
            "incompatible_nonzero_points": len(incompatible_rows),
            "tested_nonzero_points": len(nonzero_rows),
            "incompatible_fraction": len(incompatible_rows) / len(nonzero_rows),
            "minimum_minimax_D_lower_bound": min(
                r["shared_fixedH_minimax_D_lower_bound"] for r in nonzero_rows),
            "median_minimax_D_lower_bound": float(np.median(
                [r["shared_fixedH_minimax_D_lower_bound"] for r in nonzero_rows])),
            "maximum_minimax_D_lower_bound": max(
                r["shared_fixedH_minimax_D_lower_bound"] for r in nonzero_rows),
            "minimum_conservative_numeric_D_lower_bound": min(
                r["shared_fixedH_conservative_numeric_D_lower_bound"]
                for r in nonzero_rows),
            "maximum_conservative_numeric_D_lower_bound": max(
                r["shared_fixedH_conservative_numeric_D_lower_bound"]
                for r in nonzero_rows),
            "maximum_bound_verification_error": max(
                r["shared_fixedH_bound_verification_error"] for r in rows),
            "maximum_overlap_estimator_abs_difference": max(
                r["shared_fixedH_overlap_estimator_abs_difference"] for r in rows),
            "resolution_threshold": MIN_RESOLVED_ORDER_D},
        "pooled_proxy_fits": fits, "per_gap_proxy_comparison": per_gap,
        "boundary": (
            "This fixed-(T, average-detuning) scan characterizes N=2 order "
            "response. It is not used to train the average-only learner.")}
    return rows, certificate


def make_scan_plot(path: Path, rows) -> None:
    nonzero = [r for r in rows if r["gap"] > 0.0]
    norm = plt.Normalize(min(SCAN_GAPS[1:]), max(SCAN_GAPS[1:]))
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=True)
    for gap in SCAN_GAPS[1:]:
        subset = sorted((r for r in nonzero if r["gap"] == gap),
                        key=lambda r: r["fraction_actual"])
        color = cmap(norm(gap))
        axes[0].plot([r["fraction_actual"] for r in subset],
                     [r["pure_trace_distance"] for r in subset],
                     "o-", ms=3.5, lw=1.2, color=color)
        axes[1].plot([r["fraction_actual"] for r in subset],
                     [r["TVD_distribution"] for r in subset],
                     "o-", ms=3.5, lw=1.2, color=color)
    axes[0].set_title("(a) Full-state order signal")
    axes[0].set_xlabel(r"$f=t_1/T$")
    axes[0].set_ylabel(r"$D_{\rm pure}$(forward, reverse)")
    axes[1].set_title("(b) Computational-basis signal")
    axes[1].set_xlabel(r"$f=t_1/T$")
    axes[1].set_ylabel("TVD(forward, reverse)")
    scatter = axes[2].scatter([r["TVD_distribution"] for r in nonzero],
                              [r["pure_trace_distance"] for r in nonzero],
                              c=[r["gap"] for r in nonzero], cmap=cmap,
                              norm=norm, s=26, alpha=0.82)
    axes[2].set_title("(c) State versus count visibility")
    axes[2].set_xlabel("computational-basis TVD")
    axes[2].set_ylabel(r"full-state $D_{\rm pure}$")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[2], fraction=0.050, pad=0.035).set_label(
        r"gap $g$ (rad/$\mu$s)")
    fig.suptitle("N=2 117-point pulse-order benchmark", fontsize=13)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows) -> None:
    if not rows:
        raise ValueError("Cannot write empty CSV.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path: Path, evaluation_rows, contrast_rows) -> None:
    models = ("average_only_fixed_H", "path_aware_piecewise_H")
    colors = {"average_only_fixed_H": "#D55E00",
              "path_aware_piecewise_H": "#0072B2"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    ax = axes[0]
    positions = np.arange(len(models))
    data = [[r["D_prediction_to_target"] for r in evaluation_rows
             if r["model"] == m and r["split"] == "orientation_reverse"]
            for m in models]
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
    ax = axes[1]
    maximum = 0.0
    for model in models:
        subset = [r for r in contrast_rows
                  if r["model"] == model and r["split"] == "all_declared"]
        x = np.array([r["target_pair_D"] for r in subset])
        y = np.array([r["predicted_pair_D"] for r in subset])
        maximum = max(maximum, float(x.max()), float(y.max()))
        ax.scatter(x, y, s=42, alpha=0.85, color=colors[model],
                   label=("average-only" if model.startswith("average") else "path-aware"))
    ax.plot([0, maximum], [0, maximum], "--", color="black", lw=1)
    ax.set_xlabel(r"target forward--reverse $D_{\rm pure}$")
    ax.set_ylabel(r"predicted forward--reverse $D_{\rm pure}$")
    ax.set_title("(b) Order-contrast recovery")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax = axes[2]
    targets = [r for r in contrast_rows
               if r["model"] == "path_aware_piecewise_H" and r["split"] == "all_declared"]
    scatter = ax.scatter([r["target_pair_TVD"] for r in targets],
                         [r["target_pair_D"] for r in targets],
                         c=[r["gap"] for r in targets], cmap="viridis", s=48, alpha=0.90)
    ax.set_xlabel("computational-basis TVD")
    ax.set_ylabel(r"full-state $D_{\rm pure}$")
    ax.set_title("(c) Measurement visibility")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, fraction=0.050, pad=0.03).set_label(
        r"gap $g$ (rad/$\mu$s)")
    fig.suptitle("N=2 fixed-H versus path-aware held-out audit", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parameter_report(theta) -> dict[str, Any]:
    theta = np.asarray(theta, dtype=float)
    relative = np.abs(theta - TRUE_THETA) / np.maximum(np.abs(TRUE_THETA), 1.0e-12)
    return {"names": list(PARAMETER_NAMES), "true": TRUE_THETA.tolist(),
            "estimated": theta.tolist(),
            "absolute_error": np.abs(theta - TRUE_THETA).tolist(),
            "relative_error": relative.tolist(),
            "max_relative_error": float(relative.max())}


# ----------------------------------------------------------------------
# Data-independence witnesses (replaces hardcoded assertions)
# ----------------------------------------------------------------------
_INSTALL_PATH_MARKERS = (
    "site-packages", "dist-packages", ".venv", "venv",
    "node_modules", "__pypackages__",
)


def _interpreter_managed_roots() -> list[Path]:
    """Directories owned by the interpreter or by package installation."""
    roots: set[Path] = set()
    for key in ("purelib", "platlib", "stdlib", "platstdlib",
                "scripts", "data", "include"):
        try:
            value = sysconfig.get_path(key)
        except Exception:  # noqa: BLE001 - sysconfig layout varies
            continue
        if value:
            try:
                roots.add(Path(value).resolve())
            except (OSError, ValueError):
                pass
    try:
        for value in site.getsitepackages():
            roots.add(Path(value).resolve())
    except Exception:  # noqa: BLE001 - not present in all environments
        pass
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            roots.add(Path(user_site).resolve())
        elif user_site:
            roots.update(Path(value).resolve() for value in user_site)
    except Exception:  # noqa: BLE001
        pass
    for value in (sys.prefix, sys.base_prefix,
                  sys.exec_prefix, sys.base_exec_prefix):
        try:
            roots.add(Path(value).resolve())
        except (OSError, ValueError):
            pass
    return sorted(roots)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _looks_local(resolved: Path, candidate_dirs, managed_roots,
                 script: Path | None) -> bool:
    if script is not None and resolved == script:
        return False
    if not any(_is_under(resolved, d) for d in candidate_dirs):
        return False
    if any(part in _INSTALL_PATH_MARKERS for part in resolved.parts):
        return False
    if any(part.startswith("ipykernel_") for part in resolved.parts):
        return False
    return not any(_is_under(resolved, root) for root in managed_roots)


def local_module_imports() -> dict[str, list[str]]:
    """
    Modules that this script pulled in from the working directory.

    Witnesses the claim that no historical/companion script was imported,
    instead of asserting it as a constant.

    Two independent filters are required, because either one alone gives
    false positives:

      * Baseline. MODULES_AT_IMPORT is captured after the declared
        dependencies are imported, so anything already present is
        interpreter or dependency machinery rather than something this
        audit loaded.
      * Location. The module file must live under the working directory or
        beside this script, and must not sit inside any interpreter- or
        installer-owned directory.

    The location filter alone fails badly in notebooks: when the working
    directory is the home directory, user site-packages such as
    ~/.local/lib/pythonX/site-packages are nominally "under the working
    directory" and every installed dependency is misreported as a local
    companion script.

    The two categories are reported separately and only the first is gated,
    because the baseline filter has a real blind spot: a companion module
    imported in an EARLIER notebook cell is already in sys.modules when this
    script starts, so it lands in preexisting_before_this_run rather than in
    loaded_by_this_run. That list is reported for the reader to judge; it is
    not failed on, because a stale unrelated module in a long-lived kernel is
    not evidence that this audit consumed it.
    """
    candidate_dirs = {Path.cwd().resolve()}
    script = script_path_if_available()
    if script is not None:
        candidate_dirs.add(script.parent)
    managed_roots = _interpreter_managed_roots()
    loaded_by_this_run: list[str] = []
    preexisting: list[str] = []
    for name, module in list(sys.modules.items()):
        if name == "__main__":
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        try:
            resolved = Path(filename).resolve()
        except (OSError, ValueError):
            continue
        if not _looks_local(resolved, candidate_dirs, managed_roots, script):
            continue
        if name in MODULES_AT_IMPORT:
            preexisting.append(name)
        else:
            loaded_by_this_run.append(name)
    return {
        "loaded_by_this_run": sorted(loaded_by_this_run),
        "preexisting_before_this_run": sorted(preexisting),
    }


def target_regeneration_residual(dataset) -> float:
    """
    Recompute every target from the declared model and compare bitwise.

    Witnesses that targets were produced by evolve_path in this process
    rather than loaded from cache.
    """
    worst = 0.0
    for examples in dataset.values():
        for example in examples:
            again = evolve_path(example.segments, TRUE_THETA)
            worst = max(worst, float(np.max(np.abs(again - example.target_state))))
    return worst


def closure_entry(status: str, supported: str, unsupported: str) -> dict[str, str]:
    """Evidence text follows the computed status instead of being fixed."""
    return {
        "status": status,
        "evidence": supported if status == "SUPPORTED" else unsupported,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        base = Path(f"n2_fixedH_vs_pathaware_{timestamp}")
        outdir = base
        suffix = 1
        while outdir.exists():
            outdir = Path(f"{base}_{suffix:02d}")
            suffix += 1
        outdir.mkdir(parents=True, exist_ok=False)

    print("=" * 92)
    print("N=2 117-POINT + FIXED-H VS PATH-AWARE LEARNING AUDIT")
    print("=" * 92)
    print(f"output={outdir}")
    print("train=forward-only | test=same-task reverse + wholly unseen schedules")

    print("\n" + "=" * 92)
    print("A) INDEPENDENT 117-POINT PULSE-ORDER BENCHMARK")
    print("=" * 92)
    scan_rows, scan_certificate = run_n2_117_scan()
    print(f"points={scan_certificate['point_count']} "
          f"(nonzero={scan_certificate['nonzero_gap_points']}) | "
          f"max D={scan_certificate['max_D']:.6f} | "
          f"max TVD={scan_certificate['max_TVD']:.6f} | "
          f"weakest D={scan_certificate['weakest_nonzero_D']:.6e} | "
          "zero-gap D floor="
          f"{scan_certificate['zero_gap_D_floor_report']['display']}")

    print("\n" + "=" * 92)
    print("B) UNRESTRICTED SHARED FIXED-H ENDPOINT OBSTRUCTION")
    print("=" * 92)
    shared_bound_rows = shared_fixedH_bound_rows(TRAIN_SPECS + HELDOUT_SPECS)
    maximum_bound_verification_error = max(
        r["bound_verification_error"] for r in shared_bound_rows)
    minimum_declared_pair_bound = min(
        r["shared_fixedH_minimax_D_lower_bound"] for r in shared_bound_rows)
    maximum_declared_pair_bound = max(
        r["shared_fixedH_minimax_D_lower_bound"] for r in shared_bound_rows)
    minimum_declared_pair_conservative_bound = min(
        r["shared_fixedH_conservative_numeric_D_lower_bound"] for r in shared_bound_rows)
    maximum_declared_pair_conservative_bound = max(
        r["shared_fixedH_conservative_numeric_D_lower_bound"] for r in shared_bound_rows)
    scan_obstruction = scan_certificate["shared_fixedH_obstruction"]
    print("learning-pair analytic minimax D: "
          f"[{minimum_declared_pair_bound:.6e}, {maximum_declared_pair_bound:.6e}] | "
          "scan analytic minimax D: "
          f"[{scan_obstruction['minimum_minimax_D_lower_bound']:.6e}, "
          f"{scan_obstruction['maximum_minimax_D_lower_bound']:.6e}] | "
          "verification error="
          f"{float64_floor_report(maximum_bound_verification_error)['display']}")

    print("\n" + "=" * 92)
    print("C) HELD-OUT REPRESENTATION-ADEQUACY AUDIT")
    print("=" * 92)
    dataset = build_dataset()
    starts = deterministic_starts(args.starts, args.seed)
    models = {"average_only_fixed_H": evolve_average,
              "path_aware_piecewise_H": evolve_path}

    fits: dict[str, dict[str, Any]] = {}
    evaluation_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for model_name, predictor in models.items():
        fit = fit_model(model_name, predictor, dataset["train_forward"],
                        starts, args.max_nfev)
        fits[model_name] = fit
        theta = fit["best_theta"]
        for split, examples in dataset.items():
            evaluation_rows.extend(
                evaluate_examples(model_name, predictor, theta, examples))
        contrast_rows.extend(pair_contrast_rows(
            model_name, predictor, theta, TRAIN_SPECS + HELDOUT_SPECS, "all_declared"))
        print(f"[{model_name}] objective={fit['best_objective']:.3e} "
              f"theta={np.array(theta)} success={fit['best_success']}")

    summaries: dict[str, dict[str, Any]] = {}
    for model_name in models:
        summaries[model_name] = {}
        for split in dataset:
            subset = [r for r in evaluation_rows
                      if r["model"] == model_name and r["split"] == split]
            summaries[model_name][split] = summarize_rows(subset)
        summaries[model_name]["parameter_recovery"] = parameter_report(
            fits[model_name]["best_theta"])

    path_orientation_max = summaries["path_aware_piecewise_H"][
        "orientation_reverse"]["max_D"]
    avg_orientation_median = summaries["average_only_fixed_H"][
        "orientation_reverse"]["median_D"]
    path_orientation_median = summaries["path_aware_piecewise_H"][
        "orientation_reverse"]["median_D"]
    raw_floor_limited_advantage = avg_orientation_median / max(
        path_orientation_median, 1.0e-16)

    # ---- FIX A ------------------------------------------------------------
    # avg_median / PATH_MAX_TEST_D_TARGET is a lower bound on the error
    # reduction factor ONLY IF the path model actually meets that target.
    # The premise is now evaluated here and required by the status, instead of
    # living in a different test that this one never consults.
    path_within_target_premise = bool(
        path_orientation_max <= PATH_MAX_TEST_D_TARGET)
    heldout_error_reduction_lower_bound = (
        avg_orientation_median / PATH_MAX_TEST_D_TARGET)
    # -----------------------------------------------------------------------

    path_orientation_median_floor_report = float64_floor_report(path_orientation_median)
    path_orientation_max_floor_report = float64_floor_report(path_orientation_max)

    target_contrasts = [r for r in contrast_rows
                        if r["model"] == "path_aware_piecewise_H"]
    max_order_d = max(r["target_pair_D"] for r in target_contrasts)
    max_hidden_ratio = max(r["hidden_state_ratio_D_over_TVD"] for r in target_contrasts
                           if math.isfinite(r["hidden_state_ratio_D_over_TVD"]))

    avg_contrasts = [r for r in contrast_rows if r["model"] == "average_only_fixed_H"]
    path_contrasts = [r for r in contrast_rows if r["model"] == "path_aware_piecewise_H"]
    avg_pair_d_max = max(r["predicted_pair_D"] for r in avg_contrasts)
    path_pair_error_max = max(r["pair_D_abs_error"] for r in path_contrasts)
    path_pair_error_floor_report = float64_floor_report(path_pair_error_max)

    zero_control = gap_zero_control()
    if max(zero_control.values()) > 1.0e-12:
        raise AssertionError(f"Gap-zero control failed: {zero_control}")

    # ---- FIX C: witnessed data independence -------------------------------
    local_import_witness = local_module_imports()
    local_imports = local_import_witness["loaded_by_this_run"]
    preexisting_local_modules = local_import_witness["preexisting_before_this_run"]
    target_regeneration_max_difference = target_regeneration_residual(dataset)
    # -----------------------------------------------------------------------

    train_ids = {e.example_id for e in dataset["train_forward"]}
    test_ids = {e.example_id for split, examples in dataset.items()
                if split != "train_forward" for e in examples}
    learning_summaries = {(s.total_ns, s.center) for s in TRAIN_SPECS + HELDOUT_SPECS}
    all_examples = [e for examples in dataset.values() for e in examples]
    clock_and_summary_matching = all(
        (e.duration1_ns + e.duration2_ns == e.total_ns
         and e.duration1_ns % CLOCK_NS == 0
         and e.duration2_ns % CLOCK_NS == 0
         and abs(sum(d * t for d, t in e.segments) / e.total_ns - e.center) <= 1.0e-12)
        for e in all_examples)

    gates = {
        "scan_has_117_points": scan_certificate["point_count"] == SCAN_EXPECTED_POINTS,
        "scan_has_104_nonzero_gap_points": scan_certificate["nonzero_gap_points"] == 104,
        "scan_fraction_grid_reflection_closed": scan_certificate["reflection_closed"],
        "scan_half_split_present": scan_certificate["half_split_present"],
        "shared_fixedH_bound_constructively_verified": (
            maximum_bound_verification_error <= 1.0e-12),
        "scan_summary_excluded_from_learning_specs": (
            (SCAN_TOTAL_NS, SCAN_CENTER) not in learning_summaries),
        "dataset_split_has_no_leakage": train_ids.isdisjoint(test_ids),
        "gap_zero_control": max(zero_control.values()) <= 1.0e-12,
        "all_fits_finite": all(np.isfinite(f["best_objective"]) for f in fits.values()),
        "all_best_fits_successful": all(f["best_success"] for f in fits.values()),
        "clock_and_summary_matching": clock_and_summary_matching,
        "no_local_module_imports": len(local_imports) == 0,
        "targets_regenerate_bitwise": target_regeneration_max_difference == 0.0,
    }
    if not all(gates.values()):
        failed = sorted(name for name, ok in gates.items() if not ok)
        detail = {"failed_gates": failed, "all_gates": gates}
        if "no_local_module_imports" in failed:
            detail["local_modules_loaded_by_this_run"] = local_imports
            detail["local_modules_preexisting"] = preexisting_local_modules
            detail["working_directory"] = str(Path.cwd().resolve())
            detail["script_path"] = str(script_path_if_available())
        raise AssertionError(f"Implementation gate failure: {detail}")

    scientific_tests = {
        "N2_117_order_signal_resolved": {
            "status": ("SUPPORTED"
                       if scan_certificate["weakest_nonzero_D"]
                       >= max(MIN_RESOLVED_ORDER_D,
                              50.0 * scan_certificate["max_zero_gap_D_floor"])
                       else "NOT_RESOLVED"),
            "weakest_nonzero_D": scan_certificate["weakest_nonzero_D"],
            "zero_gap_D_floor": scan_certificate["max_zero_gap_D_floor"],
            "threshold": max(MIN_RESOLVED_ORDER_D,
                             50.0 * scan_certificate["max_zero_gap_D_floor"]),
            "counts_toward_scientific_status": True,
            "fail_fast": False},
        "unrestricted_shared_fixedH_incompatible_on_scan": {
            "status": ("SUPPORTED"
                       if scan_obstruction["incompatible_nonzero_points"]
                       == scan_certificate["nonzero_gap_points"]
                       else "PARTIALLY_SUPPORTED"),
            **scan_obstruction,
            "evidence_class": (
                "Architecture-independent analytic lower bound, verified by "
                "constructing the optimal projective midpoint."),
            "counts_toward_scientific_status": True,
            "fail_fast": False},
        "minimal_N2_order_signal_resolved": {
            "status": ("SUPPORTED" if max_order_d >= MIN_RESOLVED_ORDER_D
                       else "NOT_RESOLVED"),
            "max_target_pair_D": max_order_d,
            "threshold": MIN_RESOLVED_ORDER_D,
            "counts_toward_scientific_status": True,
            "fail_fast": False},
        # ---- FIX B --------------------------------------------------------
        # evolve_average maps a matched forward/reverse pair to one state by
        # construction, so this quantity cannot be anything but zero. It is
        # recorded as BY_CONSTRUCTION and excluded from the conjunction that
        # produces scientific_status, so that a tautology does not vote.
        "specified_summary_compression_erases_order_contrast": {
            "status": "BY_CONSTRUCTION",
            "construction_consistency_holds": bool(avg_pair_d_max <= 1.0e-12),
            "max_predicted_pair_D": avg_pair_d_max,
            "evidence_class": (
                "Structural consequence of the average-only representation. "
                "It cannot fail for any correct implementation, so it is "
                "reported for auditability only, is not independent physical "
                "evidence, and does not vote in scientific_status."),
            "counts_toward_scientific_status": False,
            "fail_fast": False},
        # -------------------------------------------------------------------
        "path_model_recovers_heldout_orientation": {
            "status": ("SUPPORTED" if path_orientation_max <= PATH_MAX_TEST_D_TARGET
                       else "NOT_SUPPORTED"),
            "max_orientation_test_D": path_orientation_max,
            "max_orientation_test_D_report": path_orientation_max_floor_report,
            "target": PATH_MAX_TEST_D_TARGET,
            "counts_toward_scientific_status": True,
            "fail_fast": False},
        "path_model_recovers_physical_parameters": {
            "status": ("SUPPORTED"
                       if summaries["path_aware_piecewise_H"]["parameter_recovery"][
                           "max_relative_error"] <= PATH_PARAMETER_RELATIVE_ERROR_TARGET
                       else "NOT_SUPPORTED"),
            "max_relative_parameter_error": summaries["path_aware_piecewise_H"][
                "parameter_recovery"]["max_relative_error"],
            "target": PATH_PARAMETER_RELATIVE_ERROR_TARGET,
            "counts_toward_scientific_status": True,
            "fail_fast": False},
        "heldout_path_error_reduction_lower_bound": {
            "status": ("SUPPORTED"
                       if (path_within_target_premise
                           and heldout_error_reduction_lower_bound
                           >= MIN_PATH_ADVANTAGE_FACTOR)
                       else "NOT_SUPPORTED"),
            "error_reduction_lower_bound": heldout_error_reduction_lower_bound,
            "premise_path_max_within_target": path_within_target_premise,
            "premise_statement": (
                "The quotient is a lower bound on the error reduction factor "
                "only while the path model's worst held-out orientation error "
                "stays within the predeclared denominator upper bound. That "
                "premise is evaluated here and required by this status."),
            "premise_observed_path_max_D": path_orientation_max,
            "numerator_average_only_median_D": avg_orientation_median,
            "denominator_upper_bound": PATH_MAX_TEST_D_TARGET,
            "raw_ratio_floor_limited": True,
            "raw_floor_limited_ratio_not_for_inference": raw_floor_limited_advantage,
            "raw_denominator_float64_epsilon_units": (
                path_orientation_median_floor_report["absolute_float64_epsilon_units"]),
            "threshold": MIN_PATH_ADVANTAGE_FACTOR,
            "counts_toward_scientific_status": True,
            "fail_fast": False},
        "specified_compression_failure_boundary_localized": {
            "status": ("SUPPORTED"
                       if (avg_pair_d_max <= 1.0e-12
                           and max_order_d >= MIN_RESOLVED_ORDER_D
                           and path_orientation_max <= PATH_MAX_TEST_D_TARGET)
                       else "NOT_SUPPORTED"),
            "boundary": (
                "Failure occurs when two schedules share the specified "
                "summary-only input but have resolved distinct target rays. "
                "That compression must identify them; the ordered model "
                "retains the omitted variable."),
            "partially_structural": True,
            "structural_component": (
                "The average_model_max_predicted_order_D condition is a "
                "construction consistency check and cannot fail; only the "
                "target-signal and path-recovery conditions are contingent."),
            "average_model_max_predicted_order_D": avg_pair_d_max,
            "maximum_target_order_D": max_order_d,
            "path_model_max_heldout_orientation_D": path_orientation_max,
            "counts_toward_scientific_status": True,
            "fail_fast": False},
    }

    voting_tests = {name: test for name, test in scientific_tests.items()
                    if test["counts_toward_scientific_status"]}
    all_voting_supported = all(t["status"] == "SUPPORTED" for t in voting_tests.values())

    question_closure = {
        "Q1a_independent_117_scan_design": {
            "status": "BY_CONSTRUCTION",
            "evidence": (
                f"The declared grid contains {len(SCAN_GAPS)} gaps x "
                f"{len(SCAN_FRACTIONS)} clock-aligned fractions and is "
                "excluded from learning.")},
        "Q1b_independent_117_order_signal": closure_entry(
            scientific_tests["N2_117_order_signal_resolved"]["status"],
            f"All {scan_certificate['nonzero_gap_points']} nonzero-gap "
            f"exact-state points exceed the declared resolution threshold "
            f"(weakest D = {scan_certificate['weakest_nonzero_D']:.6e}).",
            f"The weakest nonzero-gap point "
            f"(D = {scan_certificate['weakest_nonzero_D']:.6e}) does not "
            "exceed the declared resolution threshold."),
        "Q2a_minimal_interacting_N2_scope": {
            "status": "BY_CONSTRUCTION",
            "evidence": ("The declared model has two interacting atoms and a "
                         "four-dimensional Hilbert space.")},
        "Q2b_order_effect_resolved_in_N2": closure_entry(
            scientific_tests["minimal_N2_order_signal_resolved"]["status"],
            f"The N=2 target pairs have a resolved nonzero order signal "
            f"(max pair D = {max_order_d:.6e}).",
            f"The N=2 target pairs do not reach the declared resolution "
            f"threshold (max pair D = {max_order_d:.6e})."),
        "Q3_shared_fixedH_distinguishability": closure_entry(
            scientific_tests["unrestricted_shared_fixedH_incompatible_on_scan"]["status"],
            f"Exact minimax lower bound for any common endpoint on all "
            f"{scan_obstruction['tested_nonzero_points']} nonzero-gap points, "
            "not only for the restricted average-only optimizer.",
            f"Only {scan_obstruction['incompatible_nonzero_points']} of "
            f"{scan_obstruction['tested_nonzero_points']} nonzero-gap points "
            "clear the declared resolution threshold."),
        "Q4_practical_failure_boundary": closure_entry(
            scientific_tests["specified_compression_failure_boundary_localized"]["status"],
            "Forward-only training followed by unseen reverses and wholly "
            "unseen schedules; the ordered model also recovers the hidden "
            "parameters.",
            "The held-out audit did not localize the failure boundary under "
            "the declared thresholds."),
    }

    source_path = script_path_if_available()
    result = {
        "status": "VALID" if all(gates.values()) else "INVALID",
        "scientific_status": (
            "SPECIFIED_AVERAGE_ONLY_COMPRESSION_INSUFFICIENT_IN_TESTED_N2_AUDIT"
            if all_voting_supported else "MIXED_OR_NOT_RESOLVED"),
        "scientific_status_basis": {
            "voting_tests": sorted(voting_tests),
            "excluded_structural_tests": sorted(
                name for name in scientific_tests if name not in voting_tests),
            "rule": ("Only contingent tests vote. Tests that cannot fail for a "
                     "correct implementation are recorded as BY_CONSTRUCTION "
                     "and excluded."),
        },
        "scope": (
            "An independent fixed-summary 117-point N=2 order-response "
            "benchmark, followed by exact synthetic N=2 Rydberg-family "
            "calibration recovery. Forward-only training; same-task reverse "
            "and wholly new-schedule tests."),
        "N2_117_scan": scan_certificate,
        "N2_117_scan_role": (
            "Independent physical-response benchmark only. Its 117 points are "
            "not training data for either learning model."),
        "data_independence": {
            "evidence_class": (
                "Witnessed in this process where witnessable. "
                "local_module_imports and target_regeneration_max_difference "
                "are measured; the remaining entries are source-inspection "
                "statements by the author."),
            "measured_local_modules_loaded_by_this_run": local_imports,
            "measured_local_modules_preexisting_not_gated": (
                preexisting_local_modules),
            "local_import_witness_working_directory": str(Path.cwd().resolve()),
            "local_import_witness_script_path": (
                str(script_path_if_available())
                if script_path_if_available() is not None else None),
            "local_import_witness_definition": (
                "A module counts only if it is absent from the pre-run module "
                "baseline AND its file lies under the working directory or "
                "beside this script AND is outside every interpreter- or "
                "installer-owned directory. The baseline filter is what makes "
                "the witness valid in notebooks, where user site-packages sit "
                "under the working directory. Modules that were already "
                "loaded before this run are listed separately and are not "
                "gated, since the baseline filter cannot see them."),
            "measured_target_regeneration_max_difference": (
                target_regeneration_max_difference),
            "measured_scan_summary_in_learning_specs": (
                (SCAN_TOTAL_NS, SCAN_CENTER) in learning_summaries),
            "declared_historical_results_loaded": False,
            "declared_cached_targets_loaded": False,
        },
        "shared_fixedH_pair_bounds": shared_bound_rows,
        "shared_fixedH_pair_bound_summary": {
            "pair_count": len(shared_bound_rows),
            "learning_pair_analytic_minimax_D_range": [
                minimum_declared_pair_bound, maximum_declared_pair_bound],
            "learning_pair_conservative_numeric_minimax_D_range": [
                minimum_declared_pair_conservative_bound,
                maximum_declared_pair_conservative_bound],
            "maximum_constructive_verification_error": maximum_bound_verification_error},
        "minimax_range_separation": {
            "N2_117_scan_analytic_minimax_D_range": [
                scan_obstruction["minimum_minimax_D_lower_bound"],
                scan_obstruction["maximum_minimax_D_lower_bound"]],
            "N2_117_scan_conservative_numeric_minimax_D_range": [
                scan_obstruction["minimum_conservative_numeric_D_lower_bound"],
                scan_obstruction["maximum_conservative_numeric_D_lower_bound"]],
            "learning_pair_analytic_minimax_D_range": [
                minimum_declared_pair_bound, maximum_declared_pair_bound],
            "learning_pair_conservative_numeric_minimax_D_range": [
                minimum_declared_pair_conservative_bound,
                maximum_declared_pair_conservative_bound]},
        "question_closure": question_closure,
        "true_parameters": dict(zip(PARAMETER_NAMES, TRUE_THETA.tolist())),
        "fixed_interaction_rad_per_us": FIXED_INTERACTION,
        "train_specs": [asdict(x) for x in TRAIN_SPECS],
        "heldout_specs": [asdict(x) for x in HELDOUT_SPECS],
        "fits": fits,
        "summaries": summaries,
        "heldout_error_reduction": {
            "lower_bound": heldout_error_reduction_lower_bound,
            "premise_path_max_within_target": path_within_target_premise,
            "premise_observed_path_max_D": path_orientation_max,
            "numerator_average_only_median_D": avg_orientation_median,
            "denominator_upper_bound": PATH_MAX_TEST_D_TARGET,
            "raw_ratio_floor_limited": True,
            "raw_ratio_not_for_inference": raw_floor_limited_advantage,
            "raw_denominator_report": path_orientation_median_floor_report,
            "interpretation": (
                "Only the tolerance-based lower bound enters the scientific "
                "test, and only while its premise holds. The raw ratio divides "
                "by a float64-floor value and is retained solely for audit.")},
        "float64_floor_reports": {
            "path_orientation_median_D": path_orientation_median_floor_report,
            "path_orientation_max_D": path_orientation_max_floor_report,
            "path_model_max_pair_D_error": path_pair_error_floor_report},
        "witness_channel_visibility": scan_certificate["witness_channel_visibility"],
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
            "fixed-summary grid; the separate held-out representation-"
            "adequacy audit uses a declared two-parameter calibration family "
            "with fixed interaction. It is not QPU evidence, not a generic "
            "variational-circuit benchmark, and not proof that all "
            "Hamiltonian-learning models require this representation. It "
            "tests only whether the specified average-only compression loses "
            "held-out order information relative to the matched path-aware "
            "model."),
        "provenance": {
            "script_version": SCRIPT_VERSION,
            "source_sha256": sha256_file(source_path),
            "source_hash_available": source_path is not None,
            "assertions_enabled": __debug__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "seed": args.seed, "starts": args.starts, "max_nfev": args.max_nfev,
            "elapsed_sec": time.perf_counter() - started},
    }

    write_csv(outdir / "n2_117_scan.csv", scan_rows)
    write_csv(outdir / "shared_fixedH_endpoint_bounds.csv", shared_bound_rows)
    write_csv(outdir / "evaluation_rows.csv", evaluation_rows)
    write_csv(outdir / "pair_contrast_rows.csv", contrast_rows)
    (outdir / "certificate.json").write_text(
        json.dumps(jsonable(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    make_plot(outdir / "n2_fixedH_vs_pathaware.png", evaluation_rows, contrast_rows)
    make_scan_plot(outdir / "n2_117_scan.png", scan_rows)

    print("\n" + "=" * 92)
    print("GLOBAL VERDICT")
    print("=" * 92)
    compact = {
        "status": result["status"],
        "scientific_status": result["scientific_status"],
        "scientific_status_basis": result["scientific_status_basis"],
        "N2_117_scan_points": scan_certificate["point_count"],
        "N2_117_nonzero_gap_points": scan_certificate["nonzero_gap_points"],
        "N2_117_max_D": scan_certificate["max_D"],
        "N2_117_max_TVD": scan_certificate["max_TVD"],
        "N2_117_zero_gap_D_floor": scan_certificate["zero_gap_D_floor_report"],
        "N2_117_scan_analytic_minimax_D_range": [
            scan_obstruction["minimum_minimax_D_lower_bound"],
            scan_obstruction["maximum_minimax_D_lower_bound"]],
        "learning_pair_analytic_minimax_D_range": [
            minimum_declared_pair_bound, maximum_declared_pair_bound],
        "question_closure": question_closure,
        "data_independence": result["data_independence"],
        "average_orientation_median_D": avg_orientation_median,
        "path_orientation_median_D": path_orientation_median_floor_report,
        "path_orientation_max_D": path_orientation_max_floor_report,
        "heldout_error_reduction_lower_bound": {
            "value": heldout_error_reduction_lower_bound,
            "premise_path_max_within_target": path_within_target_premise,
            "denominator_upper_bound": PATH_MAX_TEST_D_TARGET,
            "raw_ratio_floor_limited": True},
        "max_target_order_D": max_order_d,
        "max_hidden_state_ratio_D_over_TVD": max_hidden_ratio,
        "witness_channel_visibility": scan_certificate["witness_channel_visibility"],
        "average_model_max_predicted_order_D": avg_pair_d_max,
        "path_model_max_pair_D_error": path_pair_error_floor_report,
        "path_parameter_max_relative_error": summaries["path_aware_piecewise_H"][
            "parameter_recovery"]["max_relative_error"],
        "scientific_test_statuses": {
            name: test["status"] for name, test in scientific_tests.items()},
        "claim_boundary": result["claim_boundary"]}
    print(json.dumps(jsonable(compact), indent=2, ensure_ascii=False))
    print(f"elapsed={time.perf_counter() - started:.2f}s")
    print(f"outputs={outdir}")


if __name__ == "__main__":
    main()
