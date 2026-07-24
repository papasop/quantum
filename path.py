#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2 FOUR-STEP PATH-RESOLVED NOISE CLOSURE

One minimal, falsifiable proposition:

    Full-unitary-equivalent controls can have different weak-noise channels,
    and their leading channel difference is predicted without target fitting
    from the ideal executed paths.

The four closed steps are

    1. SAME ENDPOINT:
       construct two transported six-segment controls and verify that each
       implements the same complete ideal two-qubit unitary as the reference,
       up to global phase;
    2. DIFFERENT PATH RESPONSE:
       compute the interaction-picture dissipative response
           K_z = integral U_z(t)^(-1) D U_z(t) dt
       by exact segmentwise Frechet derivatives;
    3. DIFFERENT NOISY CHANNEL:
       solve the complete piecewise-constant Lindblad channel at frozen
       dephasing rates;
    4. NO-FIT PREDICTION:
       compare the complete channel distance with
           gamma * ||U_0(T) (K_z-K_0)||_F / 16
       and test linear signal scaling plus quadratic prediction residual.

Model and endpoint-fiber lift are the physical-z part of M2:
exact two-atom Rydberg dynamics, six global piecewise-constant controls,
18 physical control coordinates, and a Euclidean minimum-norm endpoint lift.

This is a model-level numerical test.  It is not hardware evidence and it does
not imply that all quantum computation is geometric flow.

Colab / Jupyter:
    !pip install -q -U numpy scipy matplotlib
    # Paste this complete file into one cell, or save it and run:
    !python m2_four_step_path_noise_closure.py

The parser safely ignores Jupyter's ``-f kernel.json`` argument.
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
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import least_squares


VERSION = "M2-P4-v1"
C6 = 5_420_158.53  # rad um^6 / us
FLOOR = 1e-14


@dataclass(frozen=True)
class Cfg:
    # Exact M2 physical model.
    spacing_um: float = 6.0
    segments: int = 6
    segment_duration_us: float = 0.120

    # Endpoint-fiber construction.
    loop_epsilon: float = 0.040
    transport_step: float = 0.002
    control_fd: float = 0.002
    task_fd: float = 0.0005
    endpoint_infidelity_tol: float = 1e-11
    endpoint_residual_tol: float = 2e-9
    reachability_tol: float = 2e-4
    lift_tol: float = 1e-7
    path_rank_relative_cut: float = 1e-6
    path_spectral_gap_min: float = 1e4
    transport_convergence_tol: float = 0.02

    # Frozen local-dephasing scan, in 1/us.  gamma=0 is the numerical null.
    gammas: tuple[float, ...] = (
        0.0, 0.001875, 0.003750, 0.007500, 0.015000, 0.030000
    )

    # Independent verification of the exact Frechet derivative.  Negative
    # gamma is used only in this centered numerical derivative diagnostic.
    derivative_fd_gamma: float = 3e-4
    derivative_fd_relative_tol: float = 1e-6

    # Predeclared physics gates.
    response_to_endpoint_floor_min: float = 1e5
    signal_to_zero_noise_floor_min: float = 1e5
    signal_gamma_exponent_range: tuple[float, float] = (0.94, 1.06)
    residual_gamma_exponent_range: tuple[float, float] = (1.85, 2.15)
    maximum_prediction_relative_error: float = 0.03


def clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    den = max(0.5 * (np.linalg.norm(a) + np.linalg.norm(b)), FLOOR)
    return float(np.linalg.norm(a - b) / den)


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    keep = (x > 0.0) & (y > FLOOR)
    if np.count_nonzero(keep) < 3:
        return math.nan
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0])


class Model:
    """Exact two-atom, six-segment M2 Hamiltonian."""

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.d = 4
        self.liouville_d = self.d**2
        self.p = 3 * cfg.segments

        i2 = np.eye(2, dtype=complex)
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        n = np.array([[0, 0], [0, 1]], dtype=complex)

        def embed(a: np.ndarray, site: int) -> np.ndarray:
            return np.kron(a, i2) if site == 0 else np.kron(i2, a)

        self.xs = [embed(x, k) for k in range(2)]
        self.ys = [embed(y, k) for k in range(2)]
        self.ns = [embed(n, k) for k in range(2)]
        self.X = sum(self.xs)
        self.Y = sum(self.ys)
        self.N = sum(self.ns)
        self.V = (
            C6 / cfg.spacing_um**6 * (self.ns[0] @ self.ns[1])
        )

        twopi = 2.0 * np.pi
        self.omega0 = twopi * np.array(
            [2.0, 1.7, 2.3, 1.5, 2.1, 1.8]
        )
        self.delta0 = twopi * np.array(
            [-2.3, -1.2, 0.4, 1.4, 2.0, 0.8]
        )
        self.phase0 = np.array([0.0, 0.4, 1.1, 2.0, 2.7, -2.4])

        self.I = np.eye(self.d, dtype=complex)
        self.IL = np.eye(self.liouville_d, dtype=complex)
        self.z0 = np.zeros(self.p)
        self._unitary_cache: dict[tuple[float, ...], np.ndarray] = {}
        self.U0 = self.unitary(self.z0)

        # Unit-rate local dephasing generator.  The physical generator is
        # gamma*D, where gamma has units 1/us.
        self.D = np.zeros(
            (self.liouville_d, self.liouville_d), dtype=complex
        )
        for op in self.ns:
            ada = op.conj().T @ op
            self.D += np.kron(op.conj(), op)
            self.D -= 0.5 * np.kron(self.I, ada)
            self.D -= 0.5 * np.kron(ada.T, self.I)

    @staticmethod
    def key(z: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(z, dtype=float), 13))

    def H(self, z: np.ndarray, segment: int) -> np.ndarray:
        omega = self.omega0[segment] * (1.0 + z[3 * segment])
        if omega <= 0:
            raise ValueError("transport produced non-positive Omega")
        delta = self.delta0[segment] + 2.0 * np.pi * z[3 * segment + 1]
        phase = self.phase0[segment] + z[3 * segment + 2]
        return (
            0.5 * omega
            * (math.cos(phase) * self.X + math.sin(phase) * self.Y)
            - delta * self.N
            + self.V
        )

    def unitary(self, z: np.ndarray) -> np.ndarray:
        key = self.key(z)
        if key in self._unitary_cache:
            return self._unitary_cache[key].copy()
        u = self.I.copy()
        for segment in range(self.cfg.segments):
            u = (
                expm(
                    -1j
                    * self.H(z, segment)
                    * self.cfg.segment_duration_us
                )
                @ u
            )
        self._unitary_cache[key] = u.copy()
        return u

    def target(self, task: np.ndarray) -> np.ndarray:
        return (
            expm(-0.25j * (task[0] * self.X + task[1] * self.Y))
            @ self.U0
        )

    def endpoint_residual_vector(
        self, z: np.ndarray, task: np.ndarray
    ) -> np.ndarray:
        u = self.unitary(z)
        target = self.target(task)
        # Remove the best global phase before forming a Euclidean residual.
        u = u * np.exp(-1j * np.angle(np.vdot(target, u)))
        return np.r_[u.real.ravel(), u.imag.ravel()] - np.r_[
            target.real.ravel(), target.imag.ravel()
        ]

    def endpoint_infidelity(
        self, z: np.ndarray, task: np.ndarray
    ) -> float:
        overlap = np.trace(
            self.target(task).conj().T @ self.unitary(z)
        )
        fidelity = abs(overlap) ** 2 / self.d**2
        return float(max(0.0, 1.0 - min(1.0, fidelity.real)))

    def coherent_liouvillian(self, h: np.ndarray) -> np.ndarray:
        # Column-major vectorization:
        # vec(U rho U^dagger) = (U* kron U) vec(rho).
        return -1j * (
            np.kron(self.I, h) - np.kron(h.T, self.I)
        )

    def ideal_channel(self, z: np.ndarray) -> np.ndarray:
        u = self.unitary(z)
        return np.kron(u.conj(), u)

    def noisy_channel(self, z: np.ndarray, gamma: float) -> np.ndarray:
        channel = self.IL.copy()
        dt = self.cfg.segment_duration_us
        for segment in range(self.cfg.segments):
            generator = (
                self.coherent_liouvillian(self.H(z, segment))
                + gamma * self.D
            )
            channel = expm(generator * dt) @ channel
        return channel

    def ideal_response(
        self, z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (ideal channel S, dE/dgamma at zero, K=S^{-1}dE/dgamma).

        scipy.linalg.expm_frechet gives the exact derivative of every
        piecewise-constant segment exponential.  The product rule then gives
        the derivative of the complete channel without a time quadrature.
        """
        channel = self.IL.copy()
        derivative = np.zeros_like(channel)
        dt = self.cfg.segment_duration_us
        for segment in range(self.cfg.segments):
            a = self.coherent_liouvillian(self.H(z, segment)) * dt
            b = self.D * dt
            propagator, dpropagator = expm_frechet(
                a, b, compute_expm=True
            )
            derivative = dpropagator @ channel + propagator @ derivative
            channel = propagator @ channel
        response = np.linalg.solve(channel, derivative)
        return channel, derivative, response


def jacobian_control(
    model: Model, z: np.ndarray, task: np.ndarray, h: float
) -> np.ndarray:
    columns = []
    for k in range(model.p):
        dz = np.zeros(model.p)
        dz[k] = h
        columns.append(
            (
                model.endpoint_residual_vector(z + dz, task)
                - model.endpoint_residual_vector(z - dz, task)
            )
            / (2.0 * h)
        )
    return np.column_stack(columns)


def jacobian_task(
    model: Model, z: np.ndarray, task: np.ndarray, h: float
) -> np.ndarray:
    columns = []
    for k in range(2):
        ds = np.zeros(2)
        ds[k] = h
        columns.append(
            (
                model.endpoint_residual_vector(z, task + ds)
                - model.endpoint_residual_vector(z, task - ds)
            )
            / (2.0 * h)
        )
    return np.column_stack(columns)


def endpoint_geometry(
    q_h: np.ndarray, q_half: np.ndarray
) -> dict[str, Any]:
    uncertainty = np.linalg.norm(q_h - q_half, 2)

    def one(q: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        _, singular, vh = np.linalg.svd(q, full_matrices=True)
        cut = max(1e-7 * singular[0], 5.0 * uncertainty, FLOOR)
        rank = int(np.count_nonzero(singular > cut))
        vertical = vh[rank:].T
        projector = vertical @ vertical.T
        return rank, singular, 0.5 * (projector + projector.T)

    rank_h, singular_h, projector_h = one(q_h)
    rank_half, singular_half, projector_half = one(q_half)
    projector_change = float(
        np.linalg.norm(projector_h - projector_half, 2)
    )
    stable = bool(
        rank_h == rank_half
        and rank_half < q_h.shape[1]
        and projector_change < 0.02
    )
    return {
        "rank_h": rank_h,
        "rank_half": rank_half,
        "singular_values_h": singular_h,
        "singular_values_half": singular_half,
        "fiber_dimension": q_h.shape[1] - rank_half,
        "projector": projector_half,
        "projector_change": projector_change,
        "stable": stable,
    }


def rank_diagnostic(
    q: np.ndarray, expected_rank: int, relative_cut: float
) -> dict[str, Any]:
    singular = np.linalg.svd(q, compute_uv=False)
    cut = max(relative_cut * singular[0], FLOOR)
    numerical_rank = int(np.count_nonzero(singular > cut))
    retained = float(singular[expected_rank - 1])
    discarded = (
        float(singular[expected_rank])
        if expected_rank < len(singular)
        else 0.0
    )
    return {
        "numerical_rank": numerical_rank,
        "relative_cut_value": cut,
        "smallest_retained_singular_value": retained,
        "largest_discarded_singular_value": discarded,
        "retained_to_discarded_gap": float(
            retained / max(discarded, FLOOR)
        ),
    }


def lift_and_correct(
    cfg: Cfg,
    model: Model,
    z: np.ndarray,
    task: np.ndarray,
    d_task: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    q = jacobian_control(model, z, task, cfg.control_fd)
    b = jacobian_task(model, z, task, cfg.task_fd)
    rank_info = rank_diagnostic(
        q, rank, cfg.path_rank_relative_cut
    )

    u, _, _ = np.linalg.svd(q, full_matrices=True)
    image = u[:, :rank]
    reduced_q = image.T @ q
    reduced_b = image.T @ b
    reachability = np.linalg.norm(b - image @ reduced_b) / max(
        np.linalg.norm(b), FLOOR
    )

    rhs = -(reduced_b @ d_task)
    dz = reduced_q.T @ np.linalg.pinv(
        reduced_q @ reduced_q.T, rcond=1e-12
    ) @ rhs
    lift_error = np.linalg.norm(reduced_q @ dz - rhs) / max(
        np.linalg.norm(rhs), FLOOR
    )

    # The correction span is the Euclidean metric-normal space range(Q^T).
    normal, _ = np.linalg.qr(reduced_q.T)
    normal = normal[:, :rank]
    next_task = task + d_task
    predictor = z + dz
    fit = least_squares(
        lambda a: model.endpoint_residual_vector(
            predictor + normal @ a, next_task
        ),
        np.zeros(rank),
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=80,
    )
    next_z = predictor + normal @ fit.x
    return next_z, {
        "reachability": float(reachability),
        "lift_error": float(lift_error),
        "residual": float(
            np.linalg.norm(
                model.endpoint_residual_vector(next_z, next_task)
            )
        ),
        "infidelity": model.endpoint_infidelity(next_z, next_task),
        "rank": rank_info,
    }


def task_vertices(kind: str, epsilon: float) -> list[np.ndarray]:
    origin = np.zeros(2)
    x = np.array([epsilon, 0.0])
    y = np.array([0.0, epsilon])
    xy = np.array([epsilon, epsilon])
    if kind == "CW":
        return [origin, x, xy, y, origin]
    if kind == "CCW":
        return [origin, y, xy, x, origin]
    raise ValueError(f"unknown loop kind: {kind}")


def transport_loop(
    cfg: Cfg,
    model: Model,
    rank: int,
    kind: str,
    step: float,
) -> dict[str, Any]:
    z = model.z0.copy()
    task = np.zeros(2)
    worst = {
        "reachability": 0.0,
        "lift_error": 0.0,
        "residual": 0.0,
        "infidelity": 0.0,
    }
    rank_rows: list[dict[str, Any]] = []
    steps = 0

    vertices = task_vertices(kind, cfg.loop_epsilon)
    for start, end in zip(vertices[:-1], vertices[1:]):
        edge = end - start
        count = max(1, math.ceil(np.linalg.norm(edge) / step))
        d_task = edge / count
        for _ in range(count):
            z, diagnostics = lift_and_correct(
                cfg, model, z, task, d_task, rank
            )
            task += d_task
            for key in worst:
                worst[key] = max(worst[key], diagnostics[key])
            rank_rows.append(diagnostics["rank"])
            steps += 1

    final_q = jacobian_control(
        model, z, np.zeros(2), cfg.control_fd
    )
    rank_rows.append(
        rank_diagnostic(final_q, rank, cfg.path_rank_relative_cut)
    )

    rank_summary = {
        "audited_point_count": len(rank_rows),
        "observed_numerical_ranks": sorted(
            {int(row["numerical_rank"]) for row in rank_rows}
        ),
        "minimum_retained_singular_value": min(
            row["smallest_retained_singular_value"]
            for row in rank_rows
        ),
        "maximum_discarded_singular_value": max(
            row["largest_discarded_singular_value"]
            for row in rank_rows
        ),
        "minimum_retained_to_discarded_gap": min(
            row["retained_to_discarded_gap"] for row in rank_rows
        ),
    }
    rank_summary["stable"] = bool(
        rank_summary["observed_numerical_ranks"] == [rank]
        and rank_summary["minimum_retained_to_discarded_gap"]
        >= cfg.path_spectral_gap_min
    )

    residual = float(
        np.linalg.norm(
            model.endpoint_residual_vector(z, np.zeros(2))
        )
    )
    infidelity = model.endpoint_infidelity(z, np.zeros(2))
    numerical_pass = bool(
        residual <= cfg.endpoint_residual_tol
        and infidelity <= cfg.endpoint_infidelity_tol
        and worst["residual"] <= cfg.endpoint_residual_tol
        and worst["infidelity"] <= cfg.endpoint_infidelity_tol
        and worst["reachability"] <= cfg.reachability_tol
        and worst["lift_error"] <= cfg.lift_tol
        and rank_summary["stable"]
    )
    return {
        "kind": kind,
        "z": z,
        "steps": steps,
        "endpoint_residual": residual,
        "endpoint_infidelity": infidelity,
        "worst": worst,
        "path_rank": rank_summary,
        "numerical_pass": numerical_pass,
    }


def unitary_endpoint_metrics(
    model: Model, z: np.ndarray
) -> dict[str, float]:
    u0 = model.U0
    u = model.unitary(z)
    overlap = np.trace(u0.conj().T @ u)
    fidelity = float(abs(overlap) ** 2 / model.d**2)
    phase = float(np.angle(overlap))
    phase_aligned = u * np.exp(-1j * phase)
    return {
        "full_unitary_fidelity": fidelity,
        "full_unitary_infidelity": max(0.0, 1.0 - fidelity),
        "global_phase_rad": phase,
        "phase_aligned_frobenius_residual": float(
            np.linalg.norm(phase_aligned - u0)
        ),
    }


def derivative_difference_fd(
    cfg: Cfg,
    model: Model,
    z: np.ndarray,
) -> np.ndarray:
    h = cfg.derivative_fd_gamma
    positive = (
        model.noisy_channel(z, h)
        - model.noisy_channel(model.z0, h)
    )
    negative = (
        model.noisy_channel(z, -h)
        - model.noisy_channel(model.z0, -h)
    )
    return (positive - negative) / (2.0 * h)


def analyze_direction(
    cfg: Cfg,
    model: Model,
    run: dict[str, Any],
    base_response: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    z = run["z"]
    s0, derivative0, k0 = base_response
    s, derivative, k = model.ideal_response(z)

    delta_k = k - k0
    # The paper prediction uses the common reference endpoint.
    predicted_derivative = s0 @ delta_k
    # Product-rule derivative is an independent exact implementation check.
    direct_derivative = derivative - derivative0
    common_endpoint_relative_difference = relative_difference(
        predicted_derivative, direct_derivative
    )

    fd_derivative = derivative_difference_fd(cfg, model, z)
    derivative_fd_relative_error = relative_difference(
        fd_derivative, direct_derivative
    )

    response_norm = float(
        np.linalg.norm(delta_k) / math.sqrt(delta_k.size)
    )
    endpoint_channel_floor = float(
        np.linalg.norm(s - s0) / math.sqrt(s.size)
    )
    response_to_endpoint_floor = float(
        response_norm / max(endpoint_channel_floor, FLOOR)
    )

    base_channels = {
        gamma: model.noisy_channel(model.z0, gamma)
        for gamma in cfg.gammas
    }
    rows: list[dict[str, Any]] = []
    for gamma in cfg.gammas:
        exact = model.noisy_channel(z, gamma)
        actual = float(
            np.linalg.norm(exact - base_channels[gamma])
            / math.sqrt(exact.size)
        )
        predicted = float(
            gamma
            * np.linalg.norm(predicted_derivative)
            / math.sqrt(predicted_derivative.size)
        )
        absolute_residual = abs(actual - predicted)
        relative_error = (
            absolute_residual / actual if actual > FLOOR else math.nan
        )
        rows.append({
            "direction": run["kind"],
            "gamma_per_us": gamma,
            "actual_channel_distance": actual,
            "first_order_prediction": predicted,
            "absolute_prediction_residual": absolute_residual,
            "relative_prediction_error": relative_error,
        })

    gamma_values = np.array([row["gamma_per_us"] for row in rows])
    actual_values = np.array(
        [row["actual_channel_distance"] for row in rows]
    )
    residual_values = np.array(
        [row["absolute_prediction_residual"] for row in rows]
    )
    nonzero_rows = [row for row in rows if row["gamma_per_us"] > 0.0]
    zero_floor = rows[0]["actual_channel_distance"]
    weakest_signal = min(
        row["actual_channel_distance"] for row in nonzero_rows
    )
    maximum_relative_error = max(
        row["relative_prediction_error"] for row in nonzero_rows
    )

    signal_exponent = loglog_slope(gamma_values, actual_values)
    residual_exponent = loglog_slope(gamma_values, residual_values)
    signal_range = cfg.signal_gamma_exponent_range
    residual_range = cfg.residual_gamma_exponent_range
    endpoint = unitary_endpoint_metrics(model, z)

    gates = {
        "transport_numerical_validity": bool(run["numerical_pass"]),
        "same_full_unitary_endpoint": bool(
            endpoint["full_unitary_infidelity"]
            <= cfg.endpoint_infidelity_tol
        ),
        "path_response_above_endpoint_floor": bool(
            response_to_endpoint_floor
            >= cfg.response_to_endpoint_floor_min
        ),
        "frechet_derivative_verified": bool(
            derivative_fd_relative_error
            <= cfg.derivative_fd_relative_tol
        ),
        "common_endpoint_formula_verified": bool(
            common_endpoint_relative_difference <= 1e-8
        ),
        "noisy_signal_above_zero_noise_floor": bool(
            weakest_signal / max(zero_floor, FLOOR)
            >= cfg.signal_to_zero_noise_floor_min
        ),
        "linear_noise_scaling": bool(
            signal_range[0] <= signal_exponent <= signal_range[1]
        ),
        "quadratic_prediction_residual": bool(
            residual_range[0]
            <= residual_exponent
            <= residual_range[1]
        ),
        "first_order_prediction_accuracy": bool(
            maximum_relative_error
            <= cfg.maximum_prediction_relative_error
        ),
    }

    summary = {
        "direction": run["kind"],
        "endpoint": endpoint,
        "transport": {
            key: value for key, value in run.items() if key != "z"
        },
        "path_response": {
            "normalized_delta_K_frobenius": response_norm,
            "ideal_endpoint_channel_floor": endpoint_channel_floor,
            "response_to_endpoint_floor_ratio":
                response_to_endpoint_floor,
            "common_endpoint_vs_direct_derivative_relative_difference":
                common_endpoint_relative_difference,
            "frechet_vs_centered_fd_relative_difference":
                derivative_fd_relative_error,
        },
        "noise_scan": {
            "zero_noise_floor": zero_floor,
            "weakest_nonzero_signal": weakest_signal,
            "weakest_signal_to_zero_floor_ratio":
                weakest_signal / max(zero_floor, FLOOR),
            "signal_exponent_vs_gamma": signal_exponent,
            "prediction_residual_exponent_vs_gamma":
                residual_exponent,
            "maximum_prediction_relative_error":
                maximum_relative_error,
        },
        "gates": gates,
        "supported": bool(all(gates.values())),
    }
    return summary, rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot save an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(rows))


def save_controls(
    path: Path, model: Model, runs: dict[str, dict[str, Any]]
) -> None:
    rows = []
    controls = {"REFERENCE": model.z0}
    controls.update({kind: run["z"] for kind, run in runs.items()})
    for name, z in controls.items():
        for segment in range(model.cfg.segments):
            rows.append({
                "control": name,
                "segment_zero_based": segment,
                "omega_fractional_change": z[3 * segment],
                "detuning_addition_cycles_per_us": z[3 * segment + 1],
                "phase_addition_rad": z[3 * segment + 2],
                "omega_rad_per_us":
                    model.omega0[segment] * (1.0 + z[3 * segment]),
                "detuning_rad_per_us":
                    model.delta0[segment]
                    + 2.0 * np.pi * z[3 * segment + 1],
                "phase_rad":
                    model.phase0[segment] + z[3 * segment + 2],
                "duration_us": model.cfg.segment_duration_us,
            })
    save_csv(path, rows)


def save_plot(
    path: Path, rows: list[dict[str, Any]]
) -> str | None:
    # Avoid read-only HOME configuration warnings in managed notebooks.
    cache = Path("/tmp") / "m2_p4_matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for direction, marker in (("CW", "o"), ("CCW", "s")):
        selected = [
            row for row in rows
            if row["direction"] == direction
            and row["gamma_per_us"] > 0.0
        ]
        gamma = np.array([row["gamma_per_us"] for row in selected])
        actual = np.array(
            [row["actual_channel_distance"] for row in selected]
        )
        predicted = np.array(
            [row["first_order_prediction"] for row in selected]
        )
        residual = np.array(
            [row["absolute_prediction_residual"] for row in selected]
        )
        axes[0].loglog(
            gamma, actual, marker + "-", label=f"{direction} exact"
        )
        axes[0].loglog(
            gamma, predicted, marker + "--", label=f"{direction} first order"
        )
        axes[1].loglog(
            gamma, residual, marker + "-", label=direction
        )

    axes[0].set_xlabel(r"dephasing rate $\gamma$ ($\mu$s$^{-1}$)")
    axes[0].set_ylabel("normalized channel distance")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].set_xlabel(r"dephasing rate $\gamma$ ($\mu$s$^{-1}$)")
    axes[1].set_ylabel("absolute first-order residual")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path.name)


def latex_float(value: float) -> str:
    return f"{value:.8g}"


def save_latex_macros(
    path: Path, result: dict[str, Any]
) -> None:
    directions = {item["direction"]: item for item in result["directions"]}
    cw = directions["CW"]
    ccw = directions["CCW"]
    text = "\n".join([
        "% Auto-generated by m2_four_step_path_noise_closure.py",
        rf"\newcommand{{\MtwoPfourStatus}}{{{result['status'].replace('_', r'\_')}}}",
        rf"\newcommand{{\MtwoEndpointRank}}{{{result['endpoint_geometry']['rank_half']}}}",
        rf"\newcommand{{\MtwoFiberDimension}}{{{result['endpoint_geometry']['fiber_dimension']}}}",
        rf"\newcommand{{\MtwoCWEndpointInfidelity}}{{{latex_float(cw['endpoint']['full_unitary_infidelity'])}}}",
        rf"\newcommand{{\MtwoCCWEndpointInfidelity}}{{{latex_float(ccw['endpoint']['full_unitary_infidelity'])}}}",
        rf"\newcommand{{\MtwoCWSignalExponent}}{{{latex_float(cw['noise_scan']['signal_exponent_vs_gamma'])}}}",
        rf"\newcommand{{\MtwoCCWSignalExponent}}{{{latex_float(ccw['noise_scan']['signal_exponent_vs_gamma'])}}}",
        rf"\newcommand{{\MtwoCWResidualExponent}}{{{latex_float(cw['noise_scan']['prediction_residual_exponent_vs_gamma'])}}}",
        rf"\newcommand{{\MtwoCCWResidualExponent}}{{{latex_float(ccw['noise_scan']['prediction_residual_exponent_vs_gamma'])}}}",
        rf"\newcommand{{\MtwoCWMaxPredictionError}}{{{latex_float(cw['noise_scan']['maximum_prediction_relative_error'])}}}",
        rf"\newcommand{{\MtwoCCWMaxPredictionError}}{{{latex_float(ccw['noise_scan']['maximum_prediction_relative_error'])}}}",
        "",
    ])
    path.write_text(text, encoding="utf-8")


def pip_freeze() -> str:
    try:
        return subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
    except Exception as exc:
        return f"pip freeze unavailable: {type(exc).__name__}: {exc}\n"


def audit(cfg: Cfg) -> tuple[dict[str, Any], list[dict[str, Any]], dict]:
    model = Model(cfg)
    zero_task = np.zeros(2)
    geometry = endpoint_geometry(
        jacobian_control(model, model.z0, zero_task, cfg.control_fd),
        jacobian_control(model, model.z0, zero_task, cfg.control_fd / 2.0),
    )
    if not geometry["stable"]:
        raise AssertionError("endpoint rank/fiber projector is unstable")
    rank = int(geometry["rank_half"])

    runs = {
        kind: transport_loop(
            cfg, model, rank, kind, cfg.transport_step
        )
        for kind in ("CW", "CCW")
    }
    # A single step-halving audit checks that the transported physical control
    # and the resulting path response are not transport-discretization artifacts.
    fine_cw = transport_loop(
        cfg, model, rank, "CW", cfg.transport_step / 2.0
    )

    base_response = model.ideal_response(model.z0)
    direction_results = []
    all_rows: list[dict[str, Any]] = []
    for kind in ("CW", "CCW"):
        direction, rows = analyze_direction(
            cfg, model, runs[kind], base_response
        )
        direction_results.append(direction)
        all_rows.extend(rows)

    coarse_response = model.ideal_response(runs["CW"]["z"])[2]
    fine_response = model.ideal_response(fine_cw["z"])[2]
    transport_control_convergence = relative_difference(
        runs["CW"]["z"], fine_cw["z"]
    )
    transport_response_convergence = relative_difference(
        coarse_response - base_response[2],
        fine_response - base_response[2],
    )
    convergence_gate = bool(
        fine_cw["numerical_pass"]
        and transport_control_convergence
        <= cfg.transport_convergence_tol
        and transport_response_convergence
        <= cfg.transport_convergence_tol
    )

    geometry_public = {
        key: value for key, value in geometry.items() if key != "projector"
    }
    all_direction_gates = bool(
        all(result["supported"] for result in direction_results)
    )
    numerical_pass = bool(
        geometry["stable"]
        and all(run["numerical_pass"] for run in runs.values())
        and convergence_gate
    )
    supported = bool(numerical_pass and all_direction_gates)
    status = (
        "NUMERICAL_FAIL_NO_PHYSICAL_INTERPRETATION"
        if not numerical_pass
        else "FOUR_STEP_PATH_RESOLVED_NOISE_CLOSURE_SUPPORTED"
        if supported
        else "PATH_RESOLVED_FIRST_ORDER_PREDICTION_NOT_SUPPORTED"
    )

    result = {
        "version": VERSION,
        "status": status,
        "physical_support": supported,
        "proposition": (
            "For the tested exact two-atom Rydberg controls, distinct "
            "implementations with the same complete ideal unitary endpoint "
            "have different weak-dephasing channels, and the leading channel "
            "difference is predicted without target fitting by the "
            "interaction-picture dissipative response of the ideal paths."
        ),
        "four_step_closure": {
            "step_1": "same complete ideal unitary endpoint",
            "step_2": "different ideal-path dissipative response K",
            "step_3": "different complete Lindblad channel",
            "step_4": (
                "no-fit first-order prediction with linear signal and "
                "quadratic residual"
            ),
        },
        "claim_boundary": (
            "Applies only to this exact two-atom, six-segment Rydberg model, "
            "the predeclared Euclidean endpoint connection, local dephasing, "
            "and the tested weak-noise range.  It is not hardware evidence, "
            "does not select a unique natural metric, and does not prove that "
            "all quantum computation is geometric flow."
        ),
        "configuration": asdict(cfg),
        "endpoint_geometry": geometry_public,
        "transport_step_halving": {
            "coarse_step": cfg.transport_step,
            "fine_step": cfg.transport_step / 2.0,
            "control_relative_difference": transport_control_convergence,
            "path_response_relative_difference":
                transport_response_convergence,
            "fine_transport_numerical_pass": fine_cw["numerical_pass"],
            "gate": convergence_gate,
        },
        "directions": direction_results,
        "global_gates": {
            "endpoint_geometry_stable": geometry["stable"],
            "both_transports_numerically_valid":
                all(run["numerical_pass"] for run in runs.values()),
            "transport_step_halving_converged": convergence_gate,
            "both_directions_close_all_four_steps": all_direction_gates,
        },
    }
    return result, all_rows, runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")

    raw = sys.argv[1:]
    cleaned: list[str] = []
    ignored: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == "-f" and i + 1 < len(raw):
            ignored.extend(raw[i:i + 2])
            i += 2
        elif raw[i].startswith("-f="):
            ignored.append(raw[i])
            i += 1
        else:
            cleaned.append(raw[i])
            i += 1
    if ignored:
        print(f"[notebook] ignored kernel arguments: {ignored}")
    return parser.parse_args(cleaned)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = Path(
        args.output_dir
        or f"m2_four_step_closure_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output.mkdir(parents=True, exist_ok=False)

    script_value = globals().get("__file__")
    script_path = (
        Path(script_value).resolve()
        if script_value and Path(script_value).is_file()
        else None
    )
    provenance = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "scipy": package_version("scipy"),
        "matplotlib": package_version("matplotlib"),
        "script_path": str(script_path) if script_path else None,
        "script_sha256": sha256(script_path) if script_path else None,
    }
    summary: dict[str, Any] = {
        "version": VERSION,
        "status": "RUNNING",
        "provenance": provenance,
    }
    save_json(output / "summary.json", summary)

    print("\n" + "=" * 96)
    print("M2 FOUR-STEP PATH-RESOLVED NOISE CLOSURE")
    print("=" * 96)
    try:
        cfg = Cfg()
        result, rows, runs = audit(cfg)
        save_json(output / "certificate.json", result)
        save_csv(output / "noise_prediction.csv", rows)

        # Recreate the model only for an explicit, human-readable control table.
        save_controls(output / "controls.csv", Model(cfg), runs)
        figure = save_plot(output / "path_noise_prediction.png", rows)
        save_latex_macros(output / "results_macros.tex", result)
        (output / "pip_freeze.txt").write_text(
            pip_freeze(), encoding="utf-8"
        )

        summary.update({
            "status": "COMPLETE",
            "scientific_status": result["status"],
            "physical_support": result["physical_support"],
            "outputs": {
                "certificate": "certificate.json",
                "noise_prediction": "noise_prediction.csv",
                "controls": "controls.csv",
                "figure": figure,
                "latex_macros": "results_macros.tex",
                "pip_freeze": "pip_freeze.txt",
            },
        })

        concise = {
            "status": result["status"],
            "physical_support": result["physical_support"],
            "endpoint_geometry": {
                "rank": result["endpoint_geometry"]["rank_half"],
                "fiber_dimension":
                    result["endpoint_geometry"]["fiber_dimension"],
                "stable": result["endpoint_geometry"]["stable"],
            },
            "transport_step_halving":
                result["transport_step_halving"],
            "directions": [
                {
                    "direction": item["direction"],
                    "endpoint_infidelity":
                        item["endpoint"]["full_unitary_infidelity"],
                    "normalized_delta_K":
                        item["path_response"][
                            "normalized_delta_K_frobenius"
                        ],
                    "signal_exponent":
                        item["noise_scan"]["signal_exponent_vs_gamma"],
                    "residual_exponent":
                        item["noise_scan"][
                            "prediction_residual_exponent_vs_gamma"
                        ],
                    "maximum_prediction_relative_error":
                        item["noise_scan"][
                            "maximum_prediction_relative_error"
                        ],
                    "gates": item["gates"],
                }
                for item in result["directions"]
            ],
            "global_gates": result["global_gates"],
            "claim_boundary": result["claim_boundary"],
        }
        print(json.dumps(clean(concise), indent=2, ensure_ascii=False))

        if not result["global_gates"]["both_transports_numerically_valid"]:
            raise AssertionError(
                "transport numerical gates failed; do not interpret physics"
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

