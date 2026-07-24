#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1.1 coordinate-covariant closed-loop test of one proposition:

    The task returns to its origin, but its implementation retains
    orientation-dependent, area-scaled geometric memory that is invariant
    under a non-orthogonal linear reparameterization of physical controls.

Model: exact two-atom Rydberg dynamics, six piecewise-constant controls.
Connection: Euclidean minimum-norm horizontal lift of the full-unitary
endpoint constraint.  Readout: complete Lindblad dephasing channel.

The same physical connection is integrated in two charts.  If y=Rz and the
original metric is Mz=I, the transformed metric is

    My = R^{-T} Mz R^{-1}.

After mapping y back to physical z, the holonomy and complete noisy channel
must agree with the original chart.  A negative control intentionally keeps
My=I; it must disagree, showing that covariance is not automatic.

Colab:
    !pip install -q -U numpy scipy
    !python geometric_memory_covariant_m1_1.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares


VERSION = "M1.1"
C6 = 5_420_158.53                  # rad um^6 / us
GAMMA = 0.030                       # local dephasing rate, 1/us
FLOOR = 1e-13


@dataclass(frozen=True)
class Cfg:
    spacing: float = 6.0
    segments: int = 6
    dt: float = 0.120
    fd: float = 0.002
    task_fd: float = 0.0005
    eps: tuple[float, ...] = (0.005, 0.010, 0.020, 0.040)
    step: float = 0.002
    endpoint_infidelity_tol: float = 1e-11
    endpoint_residual_tol: float = 2e-9
    reachability_tol: float = 2e-4
    lift_tol: float = 1e-7
    convergence_tol: float = 0.10
    fiber_normal_fraction_tol: float = 0.02
    exponent_range: tuple[float, float] = (1.65, 2.35)
    orientation_cosine_max: float = -0.85
    orientation_oddness_tol: float = 0.35
    signal_to_floor_min: float = 50.0
    collapsed_ratio_max: float = 0.10
    chart_seed: int = 20260724
    chart_condition_number: float = 25.0
    covariance_control_tol: float = 0.015
    covariance_channel_tol: float = 0.015
    covariance_exponent_tol: float = 0.03
    covariance_orientation_tol: float = 0.03
    wrong_metric_min_difference: float = 0.10


def clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def save_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.real(np.vdot(a, b)) / den) if den > FLOOR else math.nan


def slope(x: np.ndarray, y: np.ndarray) -> float:
    keep = (x > 0) & (y > FLOOR)
    if keep.sum() < 2:
        return math.nan
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0])


class Model:
    def __init__(self, c: Cfg):
        self.c = c
        self.d = 4
        self.p = 3 * c.segments
        i = np.eye(2, dtype=complex)
        x = np.array([[0, 1], [1, 0]], complex)
        y = np.array([[0, -1j], [1j, 0]], complex)
        n = np.array([[0, 0], [0, 1]], complex)
        emb = lambda a, k: np.kron(a, i) if k == 0 else np.kron(i, a)
        self.xs = [emb(x, k) for k in range(2)]
        self.ys = [emb(y, k) for k in range(2)]
        self.ns = [emb(n, k) for k in range(2)]
        self.X, self.Y, self.N = map(sum, (self.xs, self.ys, self.ns))
        self.V = C6 / c.spacing**6 * (self.ns[0] @ self.ns[1])
        m = 2 * np.pi
        self.o0 = m * np.array([2.0, 1.7, 2.3, 1.5, 2.1, 1.8])
        self.d0 = m * np.array([-2.3, -1.2, 0.4, 1.4, 2.0, 0.8])
        self.ph0 = np.array([0.0, 0.4, 1.1, 2.0, 2.7, -2.4])
        self.I = np.eye(self.d, dtype=complex)
        self.z0 = np.zeros(self.p)
        self.ucache: dict[tuple[float, ...], np.ndarray] = {}
        self.ccache: dict[tuple[float, ...], np.ndarray] = {}
        self.calls = {"unitary": 0, "channel": 0}
        self.U0 = self.unitary(self.z0)

    @staticmethod
    def key(z: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(z), 13))

    def H(self, z: np.ndarray, j: int) -> np.ndarray:
        omega = self.o0[j] * (1 + z[3*j])
        if omega <= 0:
            raise ValueError("transport produced non-positive Omega")
        delta = self.d0[j] + 2*np.pi*z[3*j+1]
        phase = self.ph0[j] + z[3*j+2]
        return (
            0.5*omega*(math.cos(phase)*self.X + math.sin(phase)*self.Y)
            - delta*self.N + self.V
        )

    def unitary(self, z: np.ndarray) -> np.ndarray:
        k = self.key(z)
        if k in self.ucache:
            return self.ucache[k].copy()
        u = self.I.copy()
        for j in range(self.c.segments):
            u = expm(-1j*self.H(z, j)*self.c.dt) @ u
        self.calls["unitary"] += 1
        self.ucache[k] = u.copy()
        return u

    def target(self, s: np.ndarray) -> np.ndarray:
        return expm(-0.25j*(s[0]*self.X + s[1]*self.Y)) @ self.U0

    def residual(self, z: np.ndarray, s: np.ndarray) -> np.ndarray:
        u, target = self.unitary(z), self.target(s)
        u *= np.exp(-1j*np.angle(np.vdot(target, u)))
        return np.r_[u.real.ravel(), u.imag.ravel()] - np.r_[
            target.real.ravel(), target.imag.ravel()
        ]

    def infidelity(self, z: np.ndarray, s: np.ndarray) -> float:
        overlap = np.trace(self.target(s).conj().T @ self.unitary(z))
        f = abs(overlap)**2 / self.d**2
        return float(max(0, 1-min(1, f.real)))

    def L(self, h: np.ndarray) -> np.ndarray:
        out = -1j*(np.kron(self.I, h) - np.kron(h.T, self.I))
        for n in self.ns:
            a = math.sqrt(GAMMA)*n
            ada = a.conj().T @ a
            out += np.kron(a.conj(), a)
            out -= 0.5*np.kron(self.I, ada)
            out -= 0.5*np.kron(ada.T, self.I)
        return out

    def channel(self, z: np.ndarray) -> np.ndarray:
        k = self.key(z)
        if k in self.ccache:
            return self.ccache[k].copy()
        e = np.eye(self.d**2, dtype=complex)
        for j in range(self.c.segments):
            e = expm(self.L(self.H(z, j))*self.c.dt) @ e
        self.calls["channel"] += 1
        self.ccache[k] = e.copy()
        return e


class Chart:
    """Coordinates q with physical controls z=Aq and metric dq^T M dq."""

    def __init__(
        self,
        model: Model,
        name: str,
        physical_from_chart: np.ndarray,
        metric: np.ndarray,
    ):
        self.model = model
        self.name = name
        self.A = np.asarray(physical_from_chart, float)
        self.M = sym(np.asarray(metric, float))
        self.Minv = np.linalg.inv(self.M)
        self.q0 = np.zeros(model.p)

    def physical(self, q: np.ndarray) -> np.ndarray:
        return self.A @ np.asarray(q, float)

    def residual(self, q: np.ndarray, s: np.ndarray) -> np.ndarray:
        return self.model.residual(self.physical(q), s)

    def infidelity(self, q: np.ndarray, s: np.ndarray) -> float:
        return self.model.infidelity(self.physical(q), s)

    def channel(self, q: np.ndarray) -> np.ndarray:
        return self.model.channel(self.physical(q))

    def coordinate_steps(self, physical_h: float) -> np.ndarray:
        # Every chart-axis finite difference has the same physical norm.
        return physical_h / np.linalg.norm(self.A, axis=0)


def jac_q(chart: Chart, q: np.ndarray, s: np.ndarray, h: float) -> np.ndarray:
    cols, steps = [], chart.coordinate_steps(h)
    for k in range(chart.model.p):
        dq = np.zeros(chart.model.p); dq[k] = steps[k]
        cols.append(
            (chart.residual(q+dq, s)-chart.residual(q-dq, s))/(2*steps[k])
        )
    return np.column_stack(cols)


def jac_s(chart: Chart, q: np.ndarray, s: np.ndarray, h: float) -> np.ndarray:
    cols = []
    for k in range(2):
        ds = np.zeros(2); ds[k] = h
        cols.append((chart.residual(q, s+ds)-chart.residual(q, s-ds))/(2*h))
    return np.column_stack(cols)


def geometry(q1: np.ndarray, q2: np.ndarray) -> dict[str, Any]:
    uncertainty = np.linalg.norm(q1-q2, 2)
    def one(q: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        _, sv, vh = np.linalg.svd(q, full_matrices=True)
        rank = int((sv > max(1e-7*sv[0], 5*uncertainty, FLOOR)).sum())
        v = vh[rank:].T
        return rank, sv, sym(v @ v.T)
    r1, sv1, p1 = one(q1)
    r2, sv2, p2 = one(q2)
    change = float(np.linalg.norm(p1-p2, 2))
    return {
        "rank_h": r1, "rank_half": r2, "singular_values": sv2,
        "fiber_dimension": q1.shape[1]-r2, "projector": p2,
        "projector_change": change,
        "stable": r1 == r2 and r2 < q1.shape[1] and change < 0.02,
    }


def lift_and_correct(
    c: Cfg, chart: Chart, q0: np.ndarray, s: np.ndarray,
    ds: np.ndarray, rank: int,
) -> tuple[np.ndarray, dict[str, float]]:
    q, b = jac_q(chart, q0, s, c.fd), jac_s(chart, q0, s, c.task_fd)
    u, _, _ = np.linalg.svd(q, full_matrices=True)
    image = u[:, :rank]
    qr, br = image.T @ q, image.T @ b
    reach = np.linalg.norm(b-image@br) / max(np.linalg.norm(b), FLOOR)
    rhs = -(br @ ds)
    dq = chart.Minv @ qr.T @ np.linalg.pinv(
        sym(qr @ chart.Minv @ qr.T), rcond=1e-12
    ) @ rhs
    lift_error = np.linalg.norm(qr@dq-rhs) / max(np.linalg.norm(rhs), FLOOR)
    # Metric-normal correction span M^{-1}Q^T transforms covariantly.
    normal_raw = chart.Minv @ qr.T
    normal, _ = np.linalg.qr(normal_raw)
    normal = normal[:, :rank]
    s2, predictor = s+ds, q0+dq
    fit = least_squares(
        lambda a: chart.residual(predictor+normal@a, s2),
        np.zeros(rank), ftol=1e-12, xtol=1e-12, gtol=1e-12,
        max_nfev=80,
    )
    q2 = predictor + normal@fit.x
    return q2, {
        "reachability": float(reach),
        "lift_error": float(lift_error),
        "residual": float(np.linalg.norm(chart.residual(q2, s2))),
        "infidelity": chart.infidelity(q2, s2),
    }


def vertices(kind: str, e: float) -> list[np.ndarray]:
    o, x, y, xy = (
        np.zeros(2), np.array([e, 0.]), np.array([0., e]), np.array([e, e])
    )
    return {
        "CW": [o, x, xy, y, o],
        "CCW": [o, y, xy, x, o],
        "ZERO_AREA": [o, x, o, y, o],
    }[kind]


def loop(
    c: Cfg, chart: Chart, rank: int, kind: str, e: float, step: float,
) -> dict:
    q, s = chart.q0.copy(), np.zeros(2)
    worst = {"reachability": 0., "lift_error": 0., "residual": 0., "infidelity": 0.}
    nsteps = 0
    path = vertices(kind, e)
    for a, b in zip(path[:-1], path[1:]):
        edge = b-a
        n = max(1, math.ceil(np.linalg.norm(edge)/step))
        ds = edge/n
        for _ in range(n):
            q, d = lift_and_correct(c, chart, q, s, ds, rank)
            s += ds
            for k in worst:
                worst[k] = max(worst[k], d[k])
            nsteps += 1
    residual = float(np.linalg.norm(chart.residual(q, np.zeros(2))))
    infid = chart.infidelity(q, np.zeros(2))
    ok = (
        residual <= c.endpoint_residual_tol
        and infid <= c.endpoint_infidelity_tol
        and worst["residual"] <= c.endpoint_residual_tol
        and worst["infidelity"] <= c.endpoint_infidelity_tol
        and worst["reachability"] <= c.reachability_tol
        and worst["lift_error"] <= c.lift_tol
    )
    return {
        "q": q, "z": chart.physical(q), "channel": chart.channel(q),
        "steps": nsteps, "worst": worst,
        "endpoint_residual": residual, "endpoint_infidelity": infid,
        "numerical_pass": bool(ok),
    }


def chart_suite(
    c: Cfg, chart: Chart, rank: int, physical_projector: np.ndarray,
    base_channel: np.ndarray,
) -> dict[str, Any]:
    g = geometry(
        jac_q(chart, chart.q0, np.zeros(2), c.fd),
        jac_q(chart, chart.q0, np.zeros(2), c.fd/2),
    )
    runs, table = {}, []
    for e in c.eps:
        for kind in ("CW", "CCW", "ZERO_AREA"):
            r = loop(c, chart, rank, kind, e, c.step)
            dz = r["z"]-chart.model.z0
            r["v"] = physical_projector@dz
            r["vnorm"] = float(np.linalg.norm(r["v"]))
            r["normal_fraction"] = float(
                np.linalg.norm(dz-r["v"]) / max(np.linalg.norm(dz), FLOOR)
            )
            r["dc"] = r["channel"]-base_channel
            r["cdist"] = float(np.linalg.norm(r["dc"])/math.sqrt(r["dc"].size))
            runs[e, kind] = r
            table.append({
                "epsilon": e, "kind": kind, "steps": r["steps"],
                "endpoint_infidelity": r["endpoint_infidelity"],
                "endpoint_residual": r["endpoint_residual"],
                "vertical_control_norm": r["vnorm"],
                "fiber_normal_fraction": r["normal_fraction"],
                "channel_distance": r["cdist"],
                "numerical_pass": r["numerical_pass"],
            })

    e = np.asarray(c.eps)
    cw_v = np.array([runs[x, "CW"]["vnorm"] for x in c.eps])
    cw_c = np.array([runs[x, "CW"]["cdist"] for x in c.eps])
    z_v = np.array([runs[x, "ZERO_AREA"]["vnorm"] for x in c.eps])
    z_c = np.array([runs[x, "ZERO_AREA"]["cdist"] for x in c.eps])
    orient = []
    for x in c.eps:
        a, b = runs[x, "CW"], runs[x, "CCW"]
        sv = max(0.5*(np.linalg.norm(a["v"])+np.linalg.norm(b["v"])), FLOOR)
        sc = max(0.5*(np.linalg.norm(a["dc"])+np.linalg.norm(b["dc"])), FLOOR)
        orient.append({
            "epsilon": x,
            "control_cosine": cosine(a["v"], b["v"]),
            "control_oddness": float(np.linalg.norm(a["v"]+b["v"])/sv),
            "channel_cosine": cosine(a["dc"], b["dc"]),
            "channel_oddness": float(np.linalg.norm(a["dc"]+b["dc"])/sc),
        })

    largest = c.eps[-1]
    coarse = runs[largest, "CW"]
    fine = loop(c, chart, rank, "CW", largest, c.step/2)
    fine_v = physical_projector@(fine["z"]-chart.model.z0)
    fine_dc = fine["channel"]-base_channel
    conv_v = np.linalg.norm(coarse["v"]-fine_v) / max(
        0.5*(np.linalg.norm(coarse["v"])+np.linalg.norm(fine_v)), FLOOR
    )
    conv_c = np.linalg.norm(coarse["dc"]-fine_dc) / max(
        0.5*(np.linalg.norm(coarse["dc"])+np.linalg.norm(fine_dc)), FLOOR
    )
    vmax, cmax = cw_v[-1], cw_c[-1]
    vfloor, cfloor = max(z_v.max(), FLOOR), max(z_c.max(), FLOOR)
    measurements = {
        "control_exponent_vs_epsilon": slope(e, cw_v),
        "channel_exponent_vs_epsilon": slope(e, cw_c),
        "largest_orientation": orient[-1],
        "control_signal_to_zero_area_floor": float(vmax/vfloor),
        "channel_signal_to_zero_area_floor": float(cmax/cfloor),
        "zero_area_to_rectangle_control_ratio": float(z_v[-1]/max(vmax, FLOOR)),
        "zero_area_to_rectangle_channel_ratio": float(z_c[-1]/max(cmax, FLOOR)),
        "step_halving_control_difference": float(conv_v),
        "step_halving_channel_difference": float(conv_c),
        "maximum_fiber_normal_fraction": max(
            r["normal_fraction"] for (x, k), r in runs.items() if k != "ZERO_AREA"
        ),
    }
    lo, hi = c.exponent_range
    o = orient[-1]
    numerical = (
        all(r["numerical_pass"] for r in runs.values())
        and fine["numerical_pass"]
        and conv_v <= c.convergence_tol and conv_c <= c.convergence_tol
    )
    gates = {
        "numerical_closure_and_convergence": numerical,
        "fiber_membership": measurements["maximum_fiber_normal_fraction"]
            <= c.fiber_normal_fraction_tol,
        "control_area_scaling": lo <= measurements["control_exponent_vs_epsilon"] <= hi,
        "channel_area_scaling": lo <= measurements["channel_exponent_vs_epsilon"] <= hi,
        "orientation_reversal": o["control_cosine"] <= c.orientation_cosine_max
            and o["control_oddness"] <= c.orientation_oddness_tol,
        "control_signal_above_floor":
            measurements["control_signal_to_zero_area_floor"] >= c.signal_to_floor_min,
        "channel_signal_above_floor":
            measurements["channel_signal_to_zero_area_floor"] >= c.signal_to_floor_min,
        "zero_area_control_small":
            measurements["zero_area_to_rectangle_control_ratio"] <= c.collapsed_ratio_max,
        "zero_area_channel_small":
            measurements["zero_area_to_rectangle_channel_ratio"] <= c.collapsed_ratio_max,
    }
    return {
        "chart": chart.name,
        "endpoint_geometry": {k: v for k, v in g.items() if k != "projector"},
        "measurements": measurements,
        "gates": gates,
        "orientation_by_epsilon": orient,
        "loop_table": table,
        "_runs": runs,
    }


def random_chart_transform(c: Cfg, p: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(c.chart_seed)
    left, _ = np.linalg.qr(rng.normal(size=(p, p)))
    right, _ = np.linalg.qr(rng.normal(size=(p, p)))
    singular = np.geomspace(
        1/math.sqrt(c.chart_condition_number),
        math.sqrt(c.chart_condition_number),
        p,
    )
    return left @ np.diag(singular) @ right.T, singular


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.linalg.norm(a-b)
        / max(0.5*(np.linalg.norm(a)+np.linalg.norm(b)), FLOOR)
    )


def audit(c: Cfg) -> dict[str, Any]:
    m = Model(c)
    identity = np.eye(m.p)
    base_chart = Chart(m, "physical_z", identity, identity)
    base_geometry = geometry(
        jac_q(base_chart, base_chart.q0, np.zeros(2), c.fd),
        jac_q(base_chart, base_chart.q0, np.zeros(2), c.fd/2),
    )
    if not base_geometry["stable"]:
        raise AssertionError("endpoint rank/fiber projector is unstable")
    rank = base_geometry["rank_half"]
    physical_projector = base_geometry["projector"]
    base_channel = m.channel(m.z0)

    # y=Rz, hence z=A y with A=R^{-1}; My=A^T A preserves dz^T dz.
    R, singular = random_chart_transform(c, m.p)
    A = np.linalg.inv(R)
    transformed = Chart(m, "covariant_y", A, A.T @ A)
    wrong = Chart(m, "wrong_metric_y", A, identity)

    original = chart_suite(
        c, base_chart, rank, physical_projector, base_channel
    )
    covariant = chart_suite(
        c, transformed, rank, physical_projector, base_channel
    )

    comparisons = []
    for e in c.eps:
        for kind in ("CW", "CCW", "ZERO_AREA"):
            a = original["_runs"][e, kind]
            b = covariant["_runs"][e, kind]
            comparisons.append({
                "epsilon": e,
                "kind": kind,
                "physical_control_relative_difference": relative(a["v"], b["v"]),
                "channel_relative_difference": relative(a["dc"], b["dc"]),
                "endpoint_control_relative_difference": relative(a["z"], b["z"]),
            })

    max_control = max(
        row["physical_control_relative_difference"] for row in comparisons
    )
    max_channel = max(row["channel_relative_difference"] for row in comparisons)
    exponent_difference = max(
        abs(
            original["measurements"]["control_exponent_vs_epsilon"]
            - covariant["measurements"]["control_exponent_vs_epsilon"]
        ),
        abs(
            original["measurements"]["channel_exponent_vs_epsilon"]
            - covariant["measurements"]["channel_exponent_vs_epsilon"]
        ),
    )
    orientation_difference = max(
        abs(
            original["measurements"]["largest_orientation"][key]
            - covariant["measurements"]["largest_orientation"][key]
        )
        for key in (
            "control_cosine", "control_oddness",
            "channel_cosine", "channel_oddness",
        )
    )

    largest = c.eps[-1]
    wrong_result = loop(c, wrong, rank, "CW", largest, c.step)
    wrong_v = physical_projector @ (wrong_result["z"]-m.z0)
    wrong_dc = wrong_result["channel"]-base_channel
    reference = original["_runs"][largest, "CW"]
    wrong_control_difference = relative(wrong_v, reference["v"])
    wrong_channel_difference = relative(wrong_dc, reference["dc"])

    numerical = bool(
        original["gates"]["numerical_closure_and_convergence"]
        and covariant["gates"]["numerical_closure_and_convergence"]
        and wrong_result["numerical_pass"]
    )
    gates = {
        "numerical_validity": numerical,
        "original_M1_pass": bool(all(original["gates"].values())),
        "transformed_M1_pass": bool(all(covariant["gates"].values())),
        "transformed_endpoint_rank_stable": bool(
            covariant["endpoint_geometry"]["stable"]
            and covariant["endpoint_geometry"]["rank_half"] == rank
        ),
        "physical_control_covariance": max_control <= c.covariance_control_tol,
        "complete_channel_covariance": max_channel <= c.covariance_channel_tol,
        "scaling_exponent_covariance":
            exponent_difference <= c.covariance_exponent_tol,
        "orientation_covariance":
            orientation_difference <= c.covariance_orientation_tol,
        "wrong_metric_negative_control": bool(
            wrong_result["numerical_pass"]
            and wrong_control_difference >= c.wrong_metric_min_difference
        ),
    }
    supported = bool(all(gates.values()))
    status = (
        "NUMERICAL_FAIL_NO_PHYSICAL_INTERPRETATION" if not numerical else
        "COORDINATE_COVARIANT_GEOMETRIC_MEMORY_SUPPORTED" if supported else
        "COORDINATE_COVARIANCE_NOT_SUPPORTED"
    )
    for suite in (original, covariant):
        suite.pop("_runs")
    return {
        "status": status,
        "physical_support": supported,
        "proposition": (
            "The task returns to its origin while the implementation retains "
            "area-scaled geometric memory invariant under a non-orthogonal "
            "control reparameterization when the connection metric is "
            "transformed covariantly."
        ),
        "claim_boundary": (
            "Applies only to this exact two-atom model, the predeclared "
            "physical Euclidean control metric, and linear chart changes. "
            "It does not prove a uniquely selected natural metric, nonlinear "
            "coordinate covariance, hardware behavior, or universal manifold "
            "computation."
        ),
        "coordinate_transform": {
            "definition": "y=Rz; z=A y; A=R^{-1}; My=A^T A",
            "seed": c.chart_seed,
            "requested_condition_number": c.chart_condition_number,
            "actual_condition_number_R": float(np.linalg.cond(R)),
            "condition_number_My": float(np.linalg.cond(transformed.M)),
            "singular_values_R": singular,
        },
        "original_chart": original,
        "covariant_chart": covariant,
        "covariance_measurements": {
            "max_physical_control_relative_difference": max_control,
            "max_complete_channel_relative_difference": max_channel,
            "max_scaling_exponent_difference": exponent_difference,
            "max_orientation_statistic_difference": orientation_difference,
            "wrong_metric_control_relative_difference": wrong_control_difference,
            "wrong_metric_channel_relative_difference": wrong_channel_difference,
            "wrong_metric_endpoint_infidelity":
                wrong_result["endpoint_infidelity"],
        },
        "covariance_by_loop": comparisons,
        "gates": gates,
        "solver_calls": m.calls,
    }


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir")
    raw, cleaned, ignored, i = sys.argv[1:], [], [], 0
    while i < len(raw):
        if raw[i] == "-f" and i+1 < len(raw):
            ignored += raw[i:i+2]; i += 2
        elif raw[i].startswith("-f="):
            ignored.append(raw[i]); i += 1
        else:
            cleaned.append(raw[i]); i += 1
    if ignored:
        print(f"[notebook] ignored kernel arguments: {ignored}")
    return p.parse_args(cleaned)


def main() -> None:
    a, started = args(), time.perf_counter()
    out = Path(a.output_dir or f"geometric_memory_m1_1_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=False)
    script = globals().get("__file__")
    summary = {
        "version": VERSION, "status": "RUNNING",
        "script_sha256": sha256(Path(script)) if script and Path(script).is_file() else None,
    }
    save_json(out/"summary.json", summary)
    print("\n"+"="*92)
    print("M1.1 COORDINATE-COVARIANT GEOMETRIC-MEMORY AUDIT")
    print("="*92)
    try:
        result = audit(Cfg())
        save_json(out/"certificate.json", result)
        summary.update({"status": "COMPLETE", "scientific_interpretation": result})
        print(json.dumps(clean({
            "status": result["status"],
            "physical_support": result["physical_support"],
            "coordinate_transform": result["coordinate_transform"],
            "original_M1_measurements":
                result["original_chart"]["measurements"],
            "covariance_measurements": result["covariance_measurements"],
            "gates": result["gates"],
            "claim_boundary": result["claim_boundary"],
        }), indent=2, ensure_ascii=False))
        if not result["gates"]["numerical_validity"]:
            raise AssertionError("numerical gates failed; do not interpret physics")
    except Exception as exc:
        summary.update({
            "status": "FAIL", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        })
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter()-started
        save_json(out/"summary.json", summary)
        print(f"elapsed={summary['elapsed_seconds']:.2f}s")
        print(f"outputs={out}")


if __name__ == "__main__":
    main()
