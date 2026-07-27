#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2 FOUR-STEP PATH-RESOLVED NOISE CLOSURE  (M2-P4-v1.4.1)

One minimal, falsifiable proposition:

    Full-unitary-equivalent controls can have different weak-noise channels,
    and their leading channel difference is predicted without target fitting
    from the ideal executed paths.

The four closed steps are

    1. SAME ENDPOINT, DIFFERENT CONTROL:
       construct a transported six-segment control, verify that it implements
       the same complete ideal two-qubit unitary as the reference up to global
       phase, AND verify that it is a different control (gate G1b: v1.1 had no
       such gate, so a zero-holonomy loop would have satisfied "step 1" with
       z = z0 and only failed later, indirectly);
    2. DIFFERENT PATH RESPONSE:
       compute the interaction-picture dissipative response
           K_z = integral U_z(t)^(-1) D U_z(t) dt
       by exact segmentwise Frechet derivatives;
    3. DIFFERENT NOISY CHANNEL:
       solve the complete piecewise-constant Lindblad channel at frozen
       dephasing rates;
    4. NO-FIT PREDICTION:
       compare the exact channel difference with gamma * S_0 (K_z - K_0), at
       the OPERATOR level as well as at the norm level, and test linear signal
       scaling plus quadratic prediction residual.

WHAT CHANGED IN v1.2, AND WHY
-----------------------------
Each item below is a defect of v1.1, not a preference.

(a) NaN was silently discarded. maximum_prediction_relative_error was computed
    with max() over rows whose relative error is NaN whenever the exact channel
    distance falls to the FLOOR. Python's max() drops NaN because every
    comparison with it is False, so an undefined entry could leave the gate
    passing on the remaining rows. The scan now requires every nonzero-gamma
    relative error to be finite (gate G4d) before any accuracy gate is read.

(b) The zero-noise floor was read positionally as rows[0]. That is only the
    gamma = 0 row if cfg.gammas happens to start with 0.0. Cfg now validates
    the grid (sorted, unique, nonnegative, exactly one zero) and the floor is
    located by value.

(c) Step 1 claimed "distinct implementations" but nothing tested distinctness.
    G1b now gates ||z - z0|| directly.

(d) THE TWO DIRECTIONS ARE NOT INDEPENDENT EVIDENCE. Measured on the frozen
    configuration: ||z_CW + z_CCW|| / ||z_CW|| = 4.3e-2 and
    ||dK_CW + dK_CCW|| / ||dK_CW|| = 3.6e-2. CCW is the same loop traversed
    backwards, so it lands at approximately the opposite holonomy, and every
    gate in this script depends only on norms, which are invariant under the
    sign flip. That is why v1.1's CW and CCW exponents agreed to six digits.
    "both_directions_close_all_four_steps" therefore reads as two confirmations
    but is close to one. D2 reports the antipodality explicitly so a reader
    cannot mistake it for replication.

(e) The first-order test compared NORMS only:
    | ||dE(gamma)|| - gamma*||S_0 dK|| |. Two channels with equal norms and
    different directions pass that. The proposition is an operator statement,
    so G4b now tests ||dE(gamma) - gamma*S_0*dK|| directly. On the frozen
    configuration the two residuals agree to about 5%, so this does not change
    the conclusion; it changes what the conclusion is entitled to say.

(f) The rank gate was tunable by the analyst. The discarded singular values of
    the finite-difference Jacobian are central-difference truncation error, so
    they scale as control_fd^2: measured 3.636e-7, 9.090e-8, 2.273e-8, 5.681e-9
    at control_fd = 2e-3, 1e-3, 5e-4, 2.5e-4, i.e. a factor of 4 per halving.
    "retained/discarded >= 1e4" can therefore be met by shrinking the step,
    which says nothing about the map. G2b replaces the tunable statement with
    the scale-free one: halving control_fd must divide the largest discarded
    singular value by ~4. If a discarded direction were real rather than
    truncation, that ratio would be ~1.
    (The transported point itself is insensitive to control_fd -- measured
    relative change 0.0000 across a factor of 8 -- because the least-squares
    correction re-solves the endpoint exactly at every step. Reported, not
    assumed.)

(g) The LaTeX macros emitted eight significant digits for quantities the
    script's own step-halving audit shows are converged to about 1%.

(g2) v1.2 fixed (g) with ONE global budget and thereby broke it the other way.
    Measured per quantity, the drifts are not comparable at all:
    normalized_delta_K 6.0e-3 (2 digits), signal_exponent 6.1e-8 (7 digits),
    operator_residual_exponent 1.5e-8 (7 digits), max operator prediction error
    3.0e-6 (5 digits). Under one global budget of two digits the fitted
    exponents printed as 1.0 and 2.0 -- exactly their predicted integers -- so
    the rounding rule manufactured perfect agreement and hid that the signal
    exponent is 0.9965298, systematically below one because of the O(gamma^2)
    term. v1.3 analyses the fine-step landing point in full and gives every
    quoted scalar its own budget. Quantities at the double-precision floor (the
    endpoint infidelity is ~1e-16, and the fine run returned exactly 0.0) are
    marked FLOOR_LIMITED and quoted as upper bounds, since their step-halving
    "drift" of 100% is meaningless.

(h) A non-positive Omega raised inside the least-squares callback and aborted
    the run with a bare traceback. Invalid controls are now penalized inside
    the solver and counted, and the count is gated.

(i) The unitary cache key rounded z to 13 decimals, so two controls closer than
    5e-14 aliased to the same cached unitary. The key is now the exact bytes.

WHAT v1.4 ADDS, AND THE MEASUREMENT THAT FORCED ITS DESIGN
----------------------------------------------------------
v1.3 reported that CW and CCW are the same loop reversed, hence correlated, and
asserted that "a loop of a different shape or scale in the task plane" would be
an independent test. That assertion was WRONG, and the first v1.4 build failed
its own new gate proving it: a triangle at a different scale in the (X, Y) plane
landed with control overlap +0.99985 against the square, and the ratio of the
two landing-point norms was 1.1311 against an enclosed-area ratio of 1.125.

For a small loop the holonomy is the curvature two-form contracted with the
enclosed area, so the DIRECTION of the landing point in control space is fixed
by the plane; only the magnitude tracks the area. No reshaping or rescaling
within one task plane can reach a different point of the fiber.

v1.4 therefore makes the task space three dimensional, with generators X, Y and
N, and puts the second loop in the (X, N) plane. The reachability of all three
generators is checked by the same lift diagnostic as before. A same-plane
reshaped loop is still transported as a NEGATIVE CONTROL, so the certificate
contains the evidence for why the plane had to change rather than the shape.

v1.4.1 adds a cap on the quoted precision. Two v1.4 runs of identical source on
different platforms produced identical VALUES but different BUDGETS for the ALT
residual exponents: their -log10(drift) landed at 7.999 and 7.663, and the drift
is a difference of two nearly equal numbers, so a BLAS-level factor-of-two change
moves it across the decade boundary. A platform-dependent precision claim is not
defensible in a paper, so budgets are capped at six significant figures and every
budget whose -log10(drift) sits within 0.3 of a decade boundary is flagged
marginal_budget in the certificate. The flag is reported, not acted on: forcing
the lower count would reduce ||dK|| to one significant figure for no gain.

The collinearity gate is applied to the CONTROL overlap. The response overlap
stays high (about 0.94) even for clearly distinct landing points, because K is
approximately linear in z near the reference and its derivative is dominated by
one direction; that is reported, not gated.

Model and endpoint-fiber lift are the physical-z part of M2: exact two-atom
Rydberg dynamics, six global piecewise-constant controls, 18 physical control
coordinates, and a Euclidean minimum-norm endpoint lift.

This is a model-level numerical test. It is not hardware evidence and it does
not imply that all quantum computation is geometric flow.

Colab / Jupyter:
    # Save this complete file as path1.41.py and run:
    !python path1.41.py

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


VERSION = "M2-P4-v1.4.1"
C6 = 5_420_158.53  # rad um^6 / us
FLOOR = 1e-14
INVALID_CONTROL_PENALTY = 1e3

# Below this, a reported scalar is at the double-precision floor of the
# construction and must be quoted as an upper bound, never as a value.
NUMERICAL_FLOOR = 1e-13

# Every scalar the paper is allowed to quote, with its LaTeX macro stem. Each
# one gets its OWN digit budget from the transport step-halving audit.
QUOTED_QUANTITIES = (
    ("path_response.normalized_delta_K_frobenius", "DeltaK", "convergence"),
    ("noise_scan.signal_exponent_vs_gamma", "SignalExponent", "convergence"),
    ("noise_scan.norm_residual_exponent_vs_gamma", "NormResidualExponent",
     "convergence"),
    ("noise_scan.operator_residual_exponent_vs_gamma",
     "OperatorResidualExponent", "convergence"),
    ("noise_scan.maximum_norm_relative_prediction_error",
     "MaxNormPredictionError", "convergence"),
    ("noise_scan.maximum_operator_relative_prediction_error",
     "MaxOperatorPredictionError", "convergence"),
    ("endpoint.full_unitary_infidelity", "EndpointInfidelity", "floor"),
    ("transport.control_separation_from_reference", "ControlSeparation",
     "convergence"),
)


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

    # Step 1 distinctness. Without this, a zero-holonomy loop would satisfy
    # "same endpoint" trivially with z = z0.
    minimum_control_separation: float = 1e-6

    # Second, non-collinear landing-point construction. CW and CCW are opposite
    # orientations of the same square loop in the (X, Y) task plane and are
    # therefore strongly correlated. ALT is a square loop in the distinct
    # (X, N) task plane and at a different scale. Its landing point is not
    # constrained by the same-plane small-loop area law to lie on the same
    # control-space ray as CW.
    #
    # The task space is three dimensional, with generators X, Y and N.
    # The primary CW/CCW loop lies in the (X, Y) coordinate plane; ALT lies
    # in the (X, N) coordinate plane.
    loop_plane: tuple = (0, 1)
    alt_loop_plane: tuple = (0, 2)
    alt_loop_epsilon: float = 0.120
    # ALT needs its own transport step. Measured: the ABSOLUTE discretization
    # error of a landing point scales as the loop scale epsilon, while the
    # holonomy itself scales as the enclosed area, i.e. epsilon^2. The RELATIVE
    # step-halving drift therefore scales as step/epsilon, and a loop in a
    # different plane does not inherit CW's step.
    alt_transport_step: float = 0.0005
    # Negative control: a reshaped loop of similar enclosed area IN THE SAME
    # PLANE, transported only to demonstrate that reshaping cannot produce a
    # different landing point.
    same_plane_control_epsilon: float = 0.060
    # Cap on the quoted precision of any scalar. The step-halving drift is a
    # difference of two nearly equal numbers, so it is itself reproducible only
    # to a factor of a few across BLAS implementations. Two v1.4 runs of the
    # same source on different platforms produced identical values but DIFFERENT
    # budgets for the ALT residual exponents, because their -log10(drift) landed
    # at 7.999 and 7.663, i.e. on and just under a decade boundary. Capping the
    # budget removes that platform dependence, and no quantity in this
    # construction is worth more than six significant figures in a paper.
    maximum_quoted_digits: int = 6
    # A budget is flagged marginal when -log10(drift) sits within this fraction
    # of a decade boundary, i.e. when a factor-2 change in the drift would move
    # it. Reported, not acted on.
    decade_margin_warning: float = 0.30

    # |cos| between two landing points, applied to the CONTROL vectors. This does
    # NOT establish independence; it rules out the degenerate case where the
    # "second" loop lands on the same ray as the first and re-tests nothing.
    landing_point_collinearity_max: float = 0.9

    # G2b: the discarded singular values must behave as central-difference
    # truncation error, i.e. shrink by ~4 when control_fd is halved.
    fd_truncation_ratio_range: tuple = (3.0, 5.0)

    # Frozen local-dephasing scan, in 1/us. Exactly one entry must be 0.0.
    gammas: tuple = (0.0, 0.001875, 0.003750, 0.007500, 0.015000, 0.030000)

    # Independent verification of the exact Frechet derivative. Negative gamma
    # is used only in this centered numerical derivative diagnostic.
    derivative_fd_gamma: float = 3e-4
    derivative_fd_relative_tol: float = 1e-6

    # Predeclared physics gates.
    response_to_endpoint_floor_min: float = 1e5
    signal_to_zero_noise_floor_min: float = 1e5
    signal_gamma_exponent_range: tuple = (0.94, 1.06)
    residual_gamma_exponent_range: tuple = (1.85, 2.15)
    maximum_prediction_relative_error: float = 0.03

    def __post_init__(self) -> None:
        g = np.asarray(self.gammas, dtype=float)
        if g.size < 4:
            raise ValueError("gammas needs at least four points")
        if not np.all(np.isfinite(g)) or np.any(g < 0.0):
            raise ValueError("gammas must be finite and nonnegative")
        if np.unique(g).size != g.size:
            raise ValueError("gammas contains duplicates")
        if not np.all(np.diff(g) > 0.0):
            raise ValueError("gammas must be strictly increasing")
        if int(np.count_nonzero(g == 0.0)) != 1:
            raise ValueError("gammas must contain exactly one zero entry")


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


def finite_max(values) -> tuple[float, bool]:
    """max() that refuses to hide NaN.

    v1.1 used the builtin max on a list that could contain NaN. Every
    comparison with NaN is False, so max silently returns the largest of the
    remaining entries and an undefined row leaves no trace. Here the caller
    gets both the maximum and whether the input was fully defined.
    """
    array = np.asarray(list(values), dtype=float)
    all_finite = bool(array.size > 0 and np.all(np.isfinite(array)))
    if not all_finite:
        return math.nan, False
    return float(array.max()), True


def justified_digits(relative_drift: float) -> int:
    """Significant figures a convergence-limited quantity may be quoted to."""
    if not math.isfinite(relative_drift) or relative_drift <= 0.0:
        return 8
    return int(max(1, min(8, math.floor(-math.log10(relative_drift)))))


def round_significant(value: float, digits: int) -> float:
    if not math.isfinite(value) or value == 0.0:
        return value
    exponent = math.floor(math.log10(abs(value)))
    return float(round(value, -(exponent - digits + 1)))


def nested_get(mapping: dict, dotted: str) -> float:
    node: Any = mapping
    for key in dotted.split("."):
        node = node[key]
    return float(node)


def format_quoted(value: float, digits: int) -> str:
    if not math.isfinite(value):
        return "undefined"
    if value == 0.0:
        return "0"
    return f"{round_significant(value, digits):.{max(digits - 1, 0)}e}"


def quantity_convergence(
    coarse: dict,
    fine: dict,
    maximum_digits: int,
    decade_margin_warning: float,
    floor: float = NUMERICAL_FLOOR,
) -> dict[str, Any]:
    """Per-quantity digit budget from the transport step-halving audit.

    v1.2 derived ONE global significant-figure count from the drift of the
    control vector and applied it to everything. That was wrong in both
    directions at once. Measured here:

        normalized_delta_K            drift 6.0e-3  ->  2 digits
        signal_exponent               drift 6.1e-8  ->  7 digits
        operator_residual_exponent    drift 1.5e-8  ->  7 digits
        max_operator_prediction_error drift 3.0e-6  ->  5 digits

    Under the global rule the exponents were printed as 1.0e+00 and 2.0e+00,
    i.e. exactly the predicted integers. That erased the fact that the signal
    exponent is 0.9965298, systematically below one because of the O(gamma^2)
    term -- the rounding manufactured perfect agreement with the prediction.
    A fitted slope is far better converged than the landing point it was fitted
    at, and each quantity now carries its own budget.

    Quantities sitting at the double-precision floor (the endpoint infidelity
    is ~1e-16, and the fine run returned exactly 0.0) are not
    convergence-limited at all. Their drift is meaningless, so they are
    classified FLOOR_LIMITED and quoted only as an upper bound.
    """
    out: dict[str, Any] = {}
    for dotted, macro, kind in QUOTED_QUANTITIES:
        a = nested_get(coarse, dotted)
        b = nested_get(fine, dotted)
        magnitude = max(abs(a), abs(b))
        if kind == "floor" or magnitude <= floor:
            out[dotted] = {
                "macro_stem": macro,
                "coarse_step_value": a,
                "fine_step_value": b,
                "regime": "FLOOR_LIMITED",
                "numerical_floor": floor,
                "digits": None,
                "quoted": f"< {floor:.0e}",
                "note": "at the double-precision floor of the construction; an "
                "upper bound is the only honest quotation",
            }
            continue
        drift = abs(a - b) / max(abs(a), FLOOR)
        raw_digits = justified_digits(drift)
        digits = int(min(raw_digits, maximum_digits))
        decade_position = (
            -math.log10(drift) if drift > 0 and math.isfinite(drift) else float("inf")
        )
        margin = (
            decade_position - math.floor(decade_position)
            if math.isfinite(decade_position)
            else float("inf")
        )
        out[dotted] = {
            "macro_stem": macro,
            "coarse_step_value": a,
            "fine_step_value": b,
            "regime": "CONVERGENCE_LIMITED",
            "relative_drift": drift,
            "unconstrained_digits": raw_digits,
            "digits": digits,
            "capped_by_maximum": bool(raw_digits > maximum_digits),
            "decade_margin": margin,
            "marginal_budget": bool(
                raw_digits <= maximum_digits and margin < decade_margin_warning
            ),
            "quoted": format_quoted(a, digits),
        }
    return out


class Model:
    """Exact two-atom, six-segment M2 Hamiltonian."""

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.d = 4
        self.liouville_d = self.d**2
        self.p = 3 * cfg.segments
        self.invalid_control_evaluations = 0

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
        self.V = C6 / cfg.spacing_um**6 * (self.ns[0] @ self.ns[1])

        twopi = 2.0 * np.pi
        self.omega0 = twopi * np.array([2.0, 1.7, 2.3, 1.5, 2.1, 1.8])
        self.delta0 = twopi * np.array([-2.3, -1.2, 0.4, 1.4, 2.0, 0.8])
        self.phase0 = np.array([0.0, 0.4, 1.1, 2.0, 2.7, -2.4])

        # Three task generators. v1.4 needed a task direction outside the (X, Y)
        # plane: see the module docstring on the small-loop area law.
        self.task_generators = [self.X, self.Y, self.N]
        self.task_dimension = len(self.task_generators)

        self.I = np.eye(self.d, dtype=complex)
        self.IL = np.eye(self.liouville_d, dtype=complex)
        self.z0 = np.zeros(self.p)
        self._unitary_cache: dict[bytes, np.ndarray] = {}
        self.U0 = self.unitary(self.z0)

        # Unit-rate local dephasing generator. The physical generator is
        # gamma*D, where gamma has units 1/us. Column-major vectorization.
        self.D = np.zeros((self.liouville_d, self.liouville_d), dtype=complex)
        for op in self.ns:
            ada = op.conj().T @ op
            self.D += np.kron(op.conj(), op)
            self.D -= 0.5 * np.kron(self.I, ada)
            self.D -= 0.5 * np.kron(ada.T, self.I)

    @staticmethod
    def key(z: np.ndarray) -> bytes:
        # Exact bytes, not a rounded tuple: v1.1 rounded to 13 decimals, so two
        # controls closer than 5e-14 shared a cache entry and one silently
        # received the other's unitary.
        return np.ascontiguousarray(np.asarray(z, dtype=float)).tobytes()

    def valid(self, z: np.ndarray) -> bool:
        z = np.asarray(z, dtype=float)
        if not np.all(np.isfinite(z)):
            return False
        return bool(np.all(self.omega0 * (1.0 + z[0::3]) > 0.0))

    def H(self, z: np.ndarray, segment: int) -> np.ndarray:
        omega = self.omega0[segment] * (1.0 + z[3 * segment])
        if omega <= 0:
            raise ValueError("transport produced non-positive Omega")
        delta = self.delta0[segment] + 2.0 * np.pi * z[3 * segment + 1]
        phase = self.phase0[segment] + z[3 * segment + 2]
        return (
            0.5 * omega * (math.cos(phase) * self.X + math.sin(phase) * self.Y)
            - delta * self.N
            + self.V
        )

    def unitary(self, z: np.ndarray) -> np.ndarray:
        key = self.key(z)
        cached = self._unitary_cache.get(key)
        if cached is not None:
            return cached.copy()
        u = self.I.copy()
        for segment in range(self.cfg.segments):
            u = expm(-1j * self.H(z, segment) * self.cfg.segment_duration_us) @ u
        self._unitary_cache[key] = u.copy()
        return u

    def target(self, task: np.ndarray) -> np.ndarray:
        generator = sum(
            float(t) * g for t, g in zip(np.asarray(task, float), self.task_generators)
        )
        return expm(-0.25j * generator) @ self.U0

    def endpoint_residual_vector(self, z: np.ndarray, task: np.ndarray) -> np.ndarray:
        u = self.unitary(z)
        target = self.target(task)
        # Remove the best global phase before forming a Euclidean residual.
        u = u * np.exp(-1j * np.angle(np.vdot(target, u)))
        return np.r_[u.real.ravel(), u.imag.ravel()] - np.r_[
            target.real.ravel(), target.imag.ravel()
        ]

    def penalized_residual_vector(
        self, z: np.ndarray, task: np.ndarray
    ) -> np.ndarray:
        """Residual for the solver, finite even on an invalid control.

        v1.1 let Model.H raise inside the least-squares callback, which aborted
        the whole run with a traceback instead of recording a rejected step.
        """
        if not self.valid(z):
            self.invalid_control_evaluations += 1
            return np.full(2 * self.d * self.d, INVALID_CONTROL_PENALTY)
        return self.endpoint_residual_vector(z, task)

    def endpoint_infidelity(self, z: np.ndarray, task: np.ndarray) -> float:
        overlap = np.trace(self.target(task).conj().T @ self.unitary(z))
        fidelity = abs(overlap) ** 2 / self.d**2
        return float(max(0.0, 1.0 - min(1.0, fidelity.real)))

    def coherent_liouvillian(self, h: np.ndarray) -> np.ndarray:
        # Column-major: vec(U rho U^dagger) = (U* kron U) vec(rho).
        return -1j * (np.kron(self.I, h) - np.kron(h.T, self.I))

    def ideal_channel(self, z: np.ndarray) -> np.ndarray:
        u = self.unitary(z)
        return np.kron(u.conj(), u)

    def noisy_channel(self, z: np.ndarray, gamma: float) -> np.ndarray:
        channel = self.IL.copy()
        dt = self.cfg.segment_duration_us
        for segment in range(self.cfg.segments):
            generator = self.coherent_liouvillian(self.H(z, segment)) + gamma * self.D
            channel = expm(generator * dt) @ channel
        return channel

    def ideal_response(self, z: np.ndarray):
        """Return (ideal channel S, dE/dgamma at zero, K = S^{-1} dE/dgamma).

        scipy.linalg.expm_frechet gives the exact derivative of every
        piecewise-constant segment exponential; the product rule then gives the
        derivative of the complete channel without a time quadrature.
        """
        channel = self.IL.copy()
        derivative = np.zeros_like(channel)
        dt = self.cfg.segment_duration_us
        for segment in range(self.cfg.segments):
            a = self.coherent_liouvillian(self.H(z, segment)) * dt
            b = self.D * dt
            propagator, dpropagator = expm_frechet(a, b, compute_expm=True)
            derivative = dpropagator @ channel + propagator @ derivative
            channel = propagator @ channel
        response = np.linalg.solve(channel, derivative)
        return channel, derivative, response


def jacobian_control(model: Model, z: np.ndarray, task: np.ndarray, h: float) -> np.ndarray:
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


def jacobian_task(model: Model, z: np.ndarray, task: np.ndarray, h: float) -> np.ndarray:
    columns = []
    for k in range(model.task_dimension):
        ds = np.zeros(model.task_dimension)
        ds[k] = h
        columns.append(
            (
                model.endpoint_residual_vector(z, task + ds)
                - model.endpoint_residual_vector(z, task - ds)
            )
            / (2.0 * h)
        )
    return np.column_stack(columns)


def endpoint_geometry(q_h: np.ndarray, q_half: np.ndarray) -> dict[str, Any]:
    uncertainty = np.linalg.norm(q_h - q_half, 2)

    def one(q: np.ndarray):
        _, singular, vh = np.linalg.svd(q, full_matrices=True)
        cut = max(1e-7 * singular[0], 5.0 * uncertainty, FLOOR)
        rank = int(np.count_nonzero(singular > cut))
        vertical = vh[rank:].T
        projector = vertical @ vertical.T
        return rank, singular, 0.5 * (projector + projector.T)

    rank_h, singular_h, projector_h = one(q_h)
    rank_half, singular_half, projector_half = one(q_half)
    projector_change = float(np.linalg.norm(projector_h - projector_half, 2))
    stable = bool(
        rank_h == rank_half and rank_half < q_h.shape[1] and projector_change < 0.02
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


def fd_truncation_audit(
    cfg: Cfg, rank: int, jacobians: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Test that the discarded singular directions are truncation, not signal.

    A central difference has error O(h^2), so halving control_fd must divide the
    largest discarded singular value by about four. A genuinely small but real
    singular direction would give a ratio near one. This replaces v1.1's
    retained/discarded gap, which the analyst could satisfy simply by choosing a
    smaller control_fd.
    """
    # The two Jacobians were already built for endpoint_geometry; rebuilding
    # them here cost 36 extra unitary evaluations for nothing.
    values = {}
    for label, h in (("h", cfg.control_fd), ("h_half", cfg.control_fd / 2.0)):
        singular = np.linalg.svd(jacobians[label], compute_uv=False)
        values[label] = {
            "control_fd": float(h),
            "smallest_retained_singular_value": float(singular[rank - 1]),
            "largest_discarded_singular_value": float(singular[rank])
            if rank < singular.size
            else 0.0,
        }
    coarse = values["h"]["largest_discarded_singular_value"]
    fine = values["h_half"]["largest_discarded_singular_value"]
    ratio = float(coarse / max(fine, FLOOR))
    low, high = cfg.fd_truncation_ratio_range
    return {
        **values,
        "discarded_ratio_h_over_h_half": ratio,
        "expected_ratio_for_central_difference": 4.0,
        "accepted_ratio_range": [low, high],
        "pass": bool(low <= ratio <= high),
        "note": "A ratio near 4 means the discarded directions are "
        "central-difference truncation error and the fiber dimension is real. "
        "A ratio near 1 would mean a genuine small singular direction was being "
        "discarded.",
    }


def rank_diagnostic(q: np.ndarray, expected_rank: int, relative_cut: float) -> dict[str, Any]:
    singular = np.linalg.svd(q, compute_uv=False)
    cut = max(relative_cut * singular[0], FLOOR)
    numerical_rank = int(np.count_nonzero(singular > cut))
    retained = float(singular[expected_rank - 1])
    discarded = float(singular[expected_rank]) if expected_rank < len(singular) else 0.0
    return {
        "numerical_rank": numerical_rank,
        "relative_cut_value": cut,
        "smallest_retained_singular_value": retained,
        "largest_discarded_singular_value": discarded,
        "retained_to_discarded_gap": float(retained / max(discarded, FLOOR)),
    }


def lift_and_correct(cfg: Cfg, model: Model, z, task, d_task, rank):
    q = jacobian_control(model, z, task, cfg.control_fd)
    b = jacobian_task(model, z, task, cfg.task_fd)
    rank_info = rank_diagnostic(q, rank, cfg.path_rank_relative_cut)

    u, _, _ = np.linalg.svd(q, full_matrices=True)
    image = u[:, :rank]
    reduced_q = image.T @ q
    reduced_b = image.T @ b
    reachability = np.linalg.norm(b - image @ reduced_b) / max(np.linalg.norm(b), FLOOR)

    rhs = -(reduced_b @ d_task)
    dz = reduced_q.T @ np.linalg.pinv(reduced_q @ reduced_q.T, rcond=1e-12) @ rhs
    lift_error = np.linalg.norm(reduced_q @ dz - rhs) / max(np.linalg.norm(rhs), FLOOR)

    # The correction span is the Euclidean metric-normal space range(Q^T).
    normal, _ = np.linalg.qr(reduced_q.T)
    normal = normal[:, :rank]
    next_task = task + d_task
    predictor = z + dz
    fit = least_squares(
        lambda a: model.penalized_residual_vector(predictor + normal @ a, next_task),
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
            np.linalg.norm(model.endpoint_residual_vector(next_z, next_task))
        ),
        "infidelity": model.endpoint_infidelity(next_z, next_task),
        "rank": rank_info,
    }


def task_vertices(
    kind: str, epsilon: float, plane: tuple, dimension: int
) -> list[np.ndarray]:
    """Closed loop in one coordinate plane of the task space."""
    first, second = plane
    if first == second or not {first, second} <= set(range(dimension)):
        raise ValueError(f"invalid task plane {plane} for dimension {dimension}")

    def point(a: float, b: float) -> np.ndarray:
        v = np.zeros(dimension)
        v[first] = a
        v[second] = b
        return v

    origin = point(0.0, 0.0)
    if kind == "CW":
        return [origin, point(epsilon, 0.0), point(epsilon, epsilon),
                point(0.0, epsilon), origin]
    if kind == "CCW":
        return [origin, point(0.0, epsilon), point(epsilon, epsilon),
                point(epsilon, 0.0), origin]
    if kind == "TRI":
        # Reshaped loop, same plane. Used only as a negative control: the
        # small-loop holonomy is the curvature contracted with the enclosed
        # area, so its DIRECTION does not depend on the shape.
        return [origin, point(epsilon, 0.0), point(0.5 * epsilon, epsilon), origin]
    raise ValueError(f"unknown loop kind: {kind}")


def transport_loop(
    cfg: Cfg,
    model: Model,
    rank: int,
    kind: str,
    step: float,
    epsilon: float,
    plane: tuple,
):
    invalid_before = model.invalid_control_evaluations
    z = model.z0.copy()
    task = np.zeros(model.task_dimension)
    worst = {"reachability": 0.0, "lift_error": 0.0, "residual": 0.0, "infidelity": 0.0}
    rank_rows: list[dict[str, Any]] = []
    steps = 0

    vertices = task_vertices(kind, epsilon, plane, model.task_dimension)
    for start, end in zip(vertices[:-1], vertices[1:]):
        edge = end - start
        count = max(1, math.ceil(np.linalg.norm(edge) / step))
        d_task = edge / count
        for _ in range(count):
            z, diagnostics = lift_and_correct(cfg, model, z, task, d_task, rank)
            task = task + d_task
            for key in worst:
                worst[key] = max(worst[key], diagnostics[key])
            rank_rows.append(diagnostics["rank"])
            steps += 1

    zero_task = np.zeros(model.task_dimension)
    final_q = jacobian_control(model, z, zero_task, cfg.control_fd)
    rank_rows.append(rank_diagnostic(final_q, rank, cfg.path_rank_relative_cut))

    rank_summary = {
        "audited_point_count": len(rank_rows),
        "observed_numerical_ranks": sorted(
            {int(row["numerical_rank"]) for row in rank_rows}
        ),
        "minimum_retained_singular_value": min(
            row["smallest_retained_singular_value"] for row in rank_rows
        ),
        "maximum_discarded_singular_value": max(
            row["largest_discarded_singular_value"] for row in rank_rows
        ),
        "minimum_retained_to_discarded_gap": min(
            row["retained_to_discarded_gap"] for row in rank_rows
        ),
        "control_fd": float(cfg.control_fd),
        "note": "the gap depends on control_fd because the discarded values are "
        "truncation error; see the scale-free test in fd_truncation_audit",
    }
    rank_summary["stable"] = bool(
        rank_summary["observed_numerical_ranks"] == [rank]
        and rank_summary["minimum_retained_to_discarded_gap"] >= cfg.path_spectral_gap_min
    )

    residual = float(np.linalg.norm(model.endpoint_residual_vector(z, zero_task)))
    infidelity = model.endpoint_infidelity(z, zero_task)
    control_separation = float(np.linalg.norm(z - model.z0))
    invalid = int(model.invalid_control_evaluations - invalid_before)
    numerical_pass = bool(
        residual <= cfg.endpoint_residual_tol
        and infidelity <= cfg.endpoint_infidelity_tol
        and worst["residual"] <= cfg.endpoint_residual_tol
        and worst["infidelity"] <= cfg.endpoint_infidelity_tol
        and worst["reachability"] <= cfg.reachability_tol
        and worst["lift_error"] <= cfg.lift_tol
        and rank_summary["stable"]
        and invalid == 0
    )
    return {
        "kind": kind,
        "loop_epsilon": float(epsilon),
        "loop_plane": [int(plane[0]), int(plane[1])],
        "z": z,
        "steps": steps,
        "endpoint_residual": residual,
        "endpoint_infidelity": infidelity,
        "control_separation_from_reference": control_separation,
        "invalid_control_evaluations": invalid,
        "worst": worst,
        "path_rank": rank_summary,
        "numerical_pass": numerical_pass,
    }


def unitary_endpoint_metrics(model: Model, z: np.ndarray) -> dict[str, float]:
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
        "phase_aligned_frobenius_residual": float(np.linalg.norm(phase_aligned - u0)),
    }


def derivative_difference_fd(cfg: Cfg, model: Model, z: np.ndarray) -> np.ndarray:
    h = cfg.derivative_fd_gamma
    positive = model.noisy_channel(z, h) - model.noisy_channel(model.z0, h)
    negative = model.noisy_channel(z, -h) - model.noisy_channel(model.z0, -h)
    return (positive - negative) / (2.0 * h)


def analyze_direction(cfg: Cfg, model: Model, run: dict[str, Any], base_response):
    z = run["z"]
    s0, derivative0, k0 = base_response
    s, derivative, k = model.ideal_response(z)
    scale = math.sqrt(s.size)  # = 16 for a 16x16 Liouville matrix

    delta_k = k - k0
    # The paper prediction uses the common reference endpoint.
    predicted_derivative = s0 @ delta_k
    # Product-rule derivative is an independent exact implementation check.
    direct_derivative = derivative - derivative0
    common_endpoint_relative_difference = relative_difference(
        predicted_derivative, direct_derivative
    )

    fd_derivative = derivative_difference_fd(cfg, model, z)
    derivative_fd_relative_error = relative_difference(fd_derivative, direct_derivative)

    response_norm = float(np.linalg.norm(delta_k) / scale)
    endpoint_channel_floor = float(np.linalg.norm(s - s0) / scale)
    response_to_endpoint_floor = float(response_norm / max(endpoint_channel_floor, FLOOR))

    base_channels = {g: model.noisy_channel(model.z0, g) for g in cfg.gammas}
    rows: list[dict[str, Any]] = []
    for gamma in cfg.gammas:
        exact_difference = model.noisy_channel(z, gamma) - base_channels[gamma]
        prediction_operator = gamma * predicted_derivative
        actual = float(np.linalg.norm(exact_difference) / scale)
        predicted = float(np.linalg.norm(prediction_operator) / scale)
        norm_residual = abs(actual - predicted)
        # Operator-level residual. Strictly stronger than the norm comparison:
        # equal norms in different directions cannot pass this one.
        operator_residual = float(
            np.linalg.norm(exact_difference - prediction_operator) / scale
        )
        rows.append(
            {
                "direction": run["kind"],
                "gamma_per_us": gamma,
                "actual_channel_distance": actual,
                "first_order_prediction": predicted,
                "absolute_prediction_residual": norm_residual,
                "relative_prediction_error": (
                    norm_residual / actual if actual > FLOOR else math.nan
                ),
                "operator_prediction_residual": operator_residual,
                "operator_relative_prediction_error": (
                    operator_residual / actual if actual > FLOOR else math.nan
                ),
            }
        )

    gamma_values = np.array([row["gamma_per_us"] for row in rows])
    actual_values = np.array([row["actual_channel_distance"] for row in rows])
    norm_residual_values = np.array(
        [row["absolute_prediction_residual"] for row in rows]
    )
    operator_residual_values = np.array(
        [row["operator_prediction_residual"] for row in rows]
    )
    nonzero_rows = [row for row in rows if row["gamma_per_us"] > 0.0]

    # Located by value, not by position: v1.1 read rows[0] and would have taken
    # a nonzero gamma as the floor had the grid been reordered.
    zero_rows = [row for row in rows if row["gamma_per_us"] == 0.0]
    if len(zero_rows) != 1:
        raise AssertionError("the gamma grid must contain exactly one zero point")
    zero_floor = zero_rows[0]["actual_channel_distance"]

    weakest_signal = min(row["actual_channel_distance"] for row in nonzero_rows)
    maximum_relative_error, norm_errors_defined = finite_max(
        row["relative_prediction_error"] for row in nonzero_rows
    )
    maximum_operator_relative_error, operator_errors_defined = finite_max(
        row["operator_relative_prediction_error"] for row in nonzero_rows
    )
    errors_defined = bool(norm_errors_defined and operator_errors_defined)

    signal_exponent = loglog_slope(gamma_values, actual_values)
    norm_residual_exponent = loglog_slope(gamma_values, norm_residual_values)
    operator_residual_exponent = loglog_slope(gamma_values, operator_residual_values)
    signal_range = cfg.signal_gamma_exponent_range
    residual_range = cfg.residual_gamma_exponent_range
    endpoint = unitary_endpoint_metrics(model, z)

    gates = {
        "G1a_transport_numerical_validity": bool(run["numerical_pass"]),
        "G1_same_full_unitary_endpoint": bool(
            endpoint["full_unitary_infidelity"] <= cfg.endpoint_infidelity_tol
        ),
        "G1b_control_distinct_from_reference": bool(
            run["control_separation_from_reference"] >= cfg.minimum_control_separation
        ),
        "G2_path_response_above_endpoint_floor": bool(
            response_to_endpoint_floor >= cfg.response_to_endpoint_floor_min
        ),
        "G2c_frechet_derivative_verified": bool(
            derivative_fd_relative_error <= cfg.derivative_fd_relative_tol
        ),
        "G2d_common_endpoint_formula_verified": bool(
            common_endpoint_relative_difference <= 1e-8
        ),
        "G3_noisy_signal_above_zero_noise_floor": bool(
            weakest_signal / max(zero_floor, FLOOR) >= cfg.signal_to_zero_noise_floor_min
        ),
        "G4d_prediction_errors_all_defined": errors_defined,
        "G4_linear_noise_scaling": bool(
            signal_range[0] <= signal_exponent <= signal_range[1]
        ),
        "G4a_quadratic_norm_residual": bool(
            residual_range[0] <= norm_residual_exponent <= residual_range[1]
        ),
        "G4b_quadratic_operator_residual": bool(
            residual_range[0] <= operator_residual_exponent <= residual_range[1]
        ),
        "G4c_first_order_prediction_accuracy": bool(
            errors_defined
            and maximum_relative_error <= cfg.maximum_prediction_relative_error
            and maximum_operator_relative_error
            <= cfg.maximum_prediction_relative_error
        ),
    }

    summary = {
        "direction": run["kind"],
        "endpoint": endpoint,
        "transport": {key: value for key, value in run.items() if key != "z"},
        "path_response": {
            "normalized_delta_K_frobenius": response_norm,
            "ideal_endpoint_channel_floor": endpoint_channel_floor,
            "response_to_endpoint_floor_ratio": response_to_endpoint_floor,
            "common_endpoint_vs_direct_derivative_relative_difference": (
                common_endpoint_relative_difference
            ),
            "frechet_vs_centered_fd_relative_difference": derivative_fd_relative_error,
        },
        "noise_scan": {
            "zero_noise_floor": zero_floor,
            "weakest_nonzero_signal": weakest_signal,
            "weakest_signal_to_zero_floor_ratio": weakest_signal / max(zero_floor, FLOOR),
            "signal_exponent_vs_gamma": signal_exponent,
            "norm_residual_exponent_vs_gamma": norm_residual_exponent,
            "operator_residual_exponent_vs_gamma": operator_residual_exponent,
            "maximum_norm_relative_prediction_error": maximum_relative_error,
            "maximum_operator_relative_prediction_error": (
                maximum_operator_relative_error
            ),
            "all_relative_errors_defined": errors_defined,
        },
        "gates": gates,
        "supported": bool(all(gates.values())),
    }
    return summary, rows


def normalized_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """cos angle between two landing points or two responses.

    Scale free, so it compares landing points of different norms -- which the
    sum/difference ratios used for the CW/CCW pair cannot do, because a loop of
    a different size produces a holonomy of a different magnitude and the ratio
    is then dominated by that mismatch rather than by direction.
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= FLOOR or nb <= FLOOR:
        return math.nan
    return float(np.real(np.vdot(a, b)) / (na * nb))


def landing_point_relations(
    model: Model, runs: dict[str, Any], base_response, cfg: Cfg
) -> dict[str, Any]:
    """Pairwise relations among every transported landing point.

    Reported, not gated as "independence": there is no threshold at which two
    landing points become independent. What IS gated, elsewhere, is the
    degenerate case -- a second loop whose landing point lies on the same ray as
    the first re-tests nothing.
    """
    k0 = base_response[2]
    names = list(runs)
    z = {name: runs[name]["z"] for name in names}
    dk = {name: model.ideal_response(runs[name]["z"])[2] - k0 for name in names}

    pairs: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a}_vs_{b}"
            pairs[key] = {
                "control_overlap": normalized_overlap(z[a], z[b]),
                "response_overlap": normalized_overlap(dk[a], dk[b]),
                "control_norm_ratio": float(
                    np.linalg.norm(z[b]) / max(np.linalg.norm(z[a]), FLOOR)
                ),
                "control_sum_over_norm": float(
                    np.linalg.norm(z[a] + z[b]) / max(np.linalg.norm(z[a]), FLOOR)
                ),
                "control_difference_over_norm": float(
                    np.linalg.norm(z[a] - z[b]) / max(np.linalg.norm(z[a]), FLOOR)
                ),
                "response_sum_over_norm": float(
                    np.linalg.norm(dk[a] + dk[b]) / max(np.linalg.norm(dk[a]), FLOOR)
                ),
            }

    reversed_pair = pairs["CW_vs_CCW"]
    reshaped_pair = pairs.get("CW_vs_SAME_PLANE_RESHAPED", {})
    second_pair = pairs.get("CW_vs_ALT", {})
    finding = (
        "MEASURED IN THIS RUN. (1) Reversed loop, CW vs CCW: control overlap "
        f"{reversed_pair['control_overlap']:+.4f}, response overlap "
        f"{reversed_pair['response_overlap']:+.4f}. Reversing orientation gives "
        "the antipodal landing point, and every gate here reads norms, which are "
        "invariant under dK -> -dK; that is why the two agree to six digits in "
        "the fitted exponents."
    )
    if reshaped_pair:
        finding += (
            " (2) Reshaped loop in the SAME task plane, CW vs "
            f"SAME_PLANE_RESHAPED: control overlap "
            f"{reshaped_pair['control_overlap']:+.4f}, control norm ratio "
            f"{reshaped_pair['control_norm_ratio']:.4f}. Reshaping does not help "
            "either: for a small loop the holonomy is the curvature contracted "
            "with the enclosed area, so the DIRECTION in control space is fixed "
            "by the plane and only the magnitude tracks the area."
        )
    if second_pair:
        finding += (
            " (3) Second loop in a DIFFERENT task plane, CW vs ALT: control "
            f"overlap {second_pair['control_overlap']:+.4f}, response overlap "
            f"{second_pair['response_overlap']:+.4f}. This is the one comparison "
            "that reaches a genuinely different point of the fiber, so the "
            "first-order prediction is exercised at two unrelated landing points."
        )
    return {
        "status": "REPORTED",
        "pairs": pairs,
        "finding": finding,
        "how_to_read_the_numbers": (
            "control_overlap and response_overlap are cosines: +1 means the two "
            "landing points lie on the same ray, -1 means antipodal, and a value "
            "away from both means a genuinely different point of the fiber was "
            "reached. The sum and difference ratios are meaningful only when the "
            "two norms match, i.e. for the CW/CCW pair; for loops of different "
            "plane or scale, read the cosines."
        ),
        "why_the_response_overlap_stays_high": (
            "Near the reference, K is approximately linear in z and its "
            "derivative is dominated by one direction, so dK inherits a large "
            "overlap even when the landing points themselves are clearly "
            "distinct. The collinearity gate is therefore applied to the control "
            "overlap, which is what 'a different point of the fiber' means, and "
            "the response overlap is reported."
        ),
        "why_it_matters": (
            "Every gate in this script reads norms, and ||dK|| and ||dE|| are "
            "invariant under dK -> -dK. A sign-flipped landing point reproduces "
            "every gate value. Neither reversing nor reshaping a loop in one "
            "plane escapes this; changing the task plane does. The ALT loop "
            "exists so that at least one comparison is not a consequence of the "
            "construction."
        ),
        "caveat": (
            "Two non-collinear landing points are not statistically independent "
            "samples of anything: they are two deterministic points of one "
            "fiber, chosen by the analyst. The claim supported is that the "
            "first-order prediction holds at more than one place on the fiber, "
            "not that it was sampled at random."
        ),
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot save an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(rows))


def save_controls(path: Path, model: Model, runs: dict[str, dict[str, Any]]) -> None:
    rows = []
    controls = {"REFERENCE": model.z0}
    controls.update({kind: run["z"] for kind, run in runs.items()})
    for name, z in controls.items():
        for segment in range(model.cfg.segments):
            rows.append(
                {
                    "control": name,
                    "segment_zero_based": segment,
                    "omega_fractional_change": z[3 * segment],
                    "detuning_addition_cycles_per_us": z[3 * segment + 1],
                    "phase_addition_rad": z[3 * segment + 2],
                    "omega_rad_per_us": model.omega0[segment] * (1.0 + z[3 * segment]),
                    "detuning_rad_per_us": model.delta0[segment]
                    + 2.0 * np.pi * z[3 * segment + 1],
                    "phase_rad": model.phase0[segment] + z[3 * segment + 2],
                    "duration_us": model.cfg.segment_duration_us,
                }
            )
    save_csv(path, rows)


def save_plot(path: Path, rows: list[dict[str, Any]]) -> str | None:
    cache = Path("/tmp") / "m2_p4_matplotlib_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for direction, marker in (("CW", "o"), ("CCW", "s")):
        selected = [
            row
            for row in rows
            if row["direction"] == direction and row["gamma_per_us"] > 0.0
        ]
        gamma = np.array([row["gamma_per_us"] for row in selected])
        actual = np.array([row["actual_channel_distance"] for row in selected])
        predicted = np.array([row["first_order_prediction"] for row in selected])
        norm_residual = np.array(
            [row["absolute_prediction_residual"] for row in selected]
        )
        operator_residual = np.array(
            [row["operator_prediction_residual"] for row in selected]
        )
        axes[0].loglog(gamma, actual, marker + "-", label=f"{direction} exact")
        axes[0].loglog(gamma, predicted, marker + "--", label=f"{direction} first order")
        axes[1].loglog(gamma, norm_residual, marker + "-", label=f"{direction} norm")
        axes[1].loglog(
            gamma, operator_residual, marker + ":", label=f"{direction} operator"
        )

    axes[0].set_xlabel(r"dephasing rate $\gamma$ ($\mu$s$^{-1}$)")
    axes[0].set_ylabel("normalized channel distance")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].set_xlabel(r"dephasing rate $\gamma$ ($\mu$s$^{-1}$)")
    axes[1].set_ylabel("first-order residual")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path.name)


def save_latex_macros(path: Path, result: dict[str, Any]) -> None:
    """Emit macros at the digits each quantity individually earned.

    v1.1 printed eight significant digits for everything. v1.2 over-corrected
    with one global budget of two digits, which rounded the fitted exponents to
    1.0 and 2.0 -- exactly the predicted integers -- and so manufactured perfect
    agreement out of a rounding rule. Budgets are now per quantity, and a
    floor-limited quantity is emitted as an inequality rather than a value.
    """
    convergence = result["justified_significant_figures_per_quantity"]["CW"]
    quoted = result["quoted_values"]
    status_tex = result["status"].replace("_", r"\_")
    lines = [
        "% Auto-generated by path1.41.py (M2-P4-v1.4.1)",
        "% Each numeric macro carries the significant figures justified by the",
        "% transport step-halving drift OF THAT QUANTITY; floor-limited",
        "% quantities are emitted as upper bounds.",
        rf"\newcommand{{\MtwoPfourStatus}}{{{status_tex}}}",
        rf"\newcommand{{\MtwoEndpointRank}}{{{result['endpoint_geometry']['rank_half']}}}",
        rf"\newcommand{{\MtwoFiberDimension}}{{{result['endpoint_geometry']['fiber_dimension']}}}",
    ]
    for direction in quoted:
        for dotted, macro, _kind in QUOTED_QUANTITIES:
            record = quoted[direction][dotted]
            text = record["quoted"]
            if record["regime"] == "FLOOR_LIMITED":
                # An inequality, not a number: the fine-step run returned
                # exactly 0.0 for the endpoint infidelity. Rendered as a power
                # of ten, since "$<1e-13$" typesets as italic "1e-13".
                exponent = int(round(math.log10(NUMERICAL_FLOOR)))
                text = rf"$<10^{{{exponent}}}$"
            lines.append(
                rf"\newcommand{{\Mtwo{direction}{macro}}}{{{text}}}"
            )
            digits = record["digits"]
            if digits is not None:
                lines.append(
                    rf"\newcommand{{\Mtwo{direction}{macro}Digits}}{{{digits}}}"
                )
    pairs = result["diagnostics"]["D2_landing_point_relations"]["pairs"]
    for key, macro in (("CW_vs_CCW", "ReversedLoop"), ("CW_vs_ALT", "SecondLoop")):
        if key not in pairs:
            continue
        lines.append(
            rf"\newcommand{{\Mtwo{macro}ControlOverlap}}"
            rf"{{{format_quoted(pairs[key]['control_overlap'], 4)}}}"
        )
        lines.append(
            rf"\newcommand{{\Mtwo{macro}ResponseOverlap}}"
            rf"{{{format_quoted(pairs[key]['response_overlap'], 4)}}}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


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


def audit(cfg: Cfg):
    model = Model(cfg)
    zero_task = np.zeros(model.task_dimension)
    jacobians = {
        "h": jacobian_control(model, model.z0, zero_task, cfg.control_fd),
        "h_half": jacobian_control(model, model.z0, zero_task, cfg.control_fd / 2.0),
    }
    geometry = endpoint_geometry(jacobians["h"], jacobians["h_half"])
    if not geometry["stable"]:
        raise AssertionError("endpoint rank/fiber projector is unstable")
    rank = int(geometry["rank_half"])
    truncation = fd_truncation_audit(cfg, rank, jacobians)

    # Three landing points: a square loop in the (X, Y) task plane traversed in
    # both orientations, plus a second square loop in the distinct (X, N) task
    # plane and at a different scale. The triangular loop evaluated below is a
    # same-plane negative control and is not one of the three main landing points.
    loops = {
        "CW": ("CW", cfg.loop_epsilon, tuple(cfg.loop_plane), cfg.transport_step),
        "CCW": ("CCW", cfg.loop_epsilon, tuple(cfg.loop_plane), cfg.transport_step),
        "ALT": (
            "CW",
            cfg.alt_loop_epsilon,
            tuple(cfg.alt_loop_plane),
            cfg.alt_transport_step,
        ),
    }
    runs = {
        name: transport_loop(cfg, model, rank, kind, step, epsilon, plane)
        for name, (kind, epsilon, plane, step) in loops.items()
    }
    # Negative control, not part of the four-step claim: a reshaped loop in the
    # SAME plane as CW. Its landing point is expected to be collinear with CW's,
    # which is the measured reason ALT had to change plane rather than shape.
    same_plane_control = transport_loop(
        cfg,
        model,
        rank,
        "TRI",
        cfg.transport_step,
        cfg.same_plane_control_epsilon,
        tuple(cfg.loop_plane),
    )
    # Each landing point gets its OWN step-halving audit. Inheriting CW's budget
    # for ALT would be unjustified: ALT is a different loop with its own
    # discretization error, and the near-antipodality that licenses sharing a
    # budget between CW and CCW does not relate ALT to either.
    fine_runs = {
        name: transport_loop(cfg, model, rank, kind, step / 2.0, epsilon, plane)
        for name, (kind, epsilon, plane, step) in loops.items()
        if name != "CCW"
    }

    base_response = model.ideal_response(model.z0)
    direction_results = []
    all_rows: list[dict[str, Any]] = []
    summaries = {}
    for name in loops:
        summary, rows = analyze_direction(cfg, model, runs[name], base_response)
        summary["direction"] = name
        for row in rows:
            row["direction"] = name
        summaries[name] = summary
        direction_results.append(summary)
        all_rows.extend(rows)

    step_halving = {}
    for name, fine in fine_runs.items():
        coarse_response = model.ideal_response(runs[name]["z"])[2]
        fine_response = model.ideal_response(fine["z"])[2]
        control_drift = relative_difference(runs[name]["z"], fine["z"])
        response_drift = relative_difference(
            coarse_response - base_response[2], fine_response - base_response[2]
        )
        coarse_step = loops[name][3]
        step_halving[name] = {
            "coarse_step": coarse_step,
            "fine_step": coarse_step / 2.0,
            "loop_epsilon": runs[name]["loop_epsilon"],
            "control_relative_difference": control_drift,
            "path_response_relative_difference": response_drift,
            "fine_transport_numerical_pass": fine["numerical_pass"],
            "gate": bool(
                fine["numerical_pass"]
                and control_drift <= cfg.transport_convergence_tol
                and response_drift <= cfg.transport_convergence_tol
            ),
        }
    convergence_gate = bool(all(v["gate"] for v in step_halving.values()))
    worst_drift = max(
        max(v["control_relative_difference"], v["path_response_relative_difference"])
        for v in step_halving.values()
    )

    # Per-quantity digit budgets, measured separately for each audited landing
    # point. CCW inherits CW's budget, and that inheritance is stated: D2 shows
    # the two are near-antipodal points of the same loop.
    convergence = {}
    for name, fine in fine_runs.items():
        fine_summary, _ = analyze_direction(cfg, model, fine, base_response)
        convergence[name] = quantity_convergence(
            summaries[name],
            fine_summary,
            cfg.maximum_quoted_digits,
            cfg.decade_margin_warning,
        )
    convergence["CCW"] = convergence["CW"]

    quoted = {}
    for name in loops:
        budget = convergence[name]
        quoted[name] = {
            dotted: {
                "value": nested_get(summaries[name], dotted),
                "quoted": (
                    budget[dotted]["quoted"]
                    if budget[dotted]["regime"] == "FLOOR_LIMITED"
                    else format_quoted(
                        nested_get(summaries[name], dotted), budget[dotted]["digits"]
                    )
                ),
                "digits": budget[dotted]["digits"],
                "regime": budget[dotted]["regime"],
                "budget_measured_on": "CW" if name == "CCW" else name,
            }
            for dotted, _macro, _kind in QUOTED_QUANTITIES
        }

    relations = landing_point_relations(
        model, {**runs, "SAME_PLANE_RESHAPED": same_plane_control}, base_response, cfg
    )
    second_loop = relations["pairs"]["CW_vs_ALT"]
    # Gated on the CONTROL overlap: the question is whether the second loop
    # reached a different POINT of the fiber. The response overlap is reported
    # rather than gated, because dK is approximately linear in z near z0 and is
    # dominated by one direction, so it stays high even for clearly distinct
    # landing points.
    collinearity = abs(second_loop["control_overlap"])
    second_loop_gate = {
        "control_overlap": second_loop["control_overlap"],
        "response_overlap": second_loop["response_overlap"],
        "absolute_control_overlap": collinearity,
        "threshold": cfg.landing_point_collinearity_max,
        "pass": bool(
            math.isfinite(collinearity)
            and collinearity <= cfg.landing_point_collinearity_max
        ),
        "note": "Rules out the degenerate case in which the second loop lands on "
        "the same ray as the first and re-tests nothing. Passing does NOT "
        "establish statistical independence; see the caveat in D2.",
    }

    geometry_public = {k: v for k, v in geometry.items() if k != "projector"}
    all_direction_gates = bool(all(r["supported"] for r in direction_results))
    numerical_pass = bool(
        geometry["stable"]
        and truncation["pass"]
        and all(run["numerical_pass"] for run in runs.values())
        and convergence_gate
        and second_loop_gate["pass"]
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
            "For the tested exact two-atom Rydberg controls, a distinct "
            "implementation with the same complete ideal unitary endpoint has a "
            "different weak-dephasing channel, and the leading channel "
            "difference is predicted without target fitting by the "
            "interaction-picture dissipative response of the ideal paths."
        ),
        "four_step_closure": {
            "step_1": "same complete ideal unitary endpoint, different control",
            "step_2": "different ideal-path dissipative response K",
            "step_3": "different complete Lindblad channel",
            "step_4": (
                "no-fit first-order prediction with linear signal and quadratic "
                "residual, tested at the operator level as well as in norm"
            ),
        },
        "claim_boundary": (
            "Applies only to this exact two-atom, six-segment Rydberg model, the "
            "predeclared Euclidean endpoint connection, local dephasing, and the "
            "tested weak-noise range. The transported control is a POINT of the "
            "endpoint fiber; the loop is how that point was constructed, and the "
            "point moves with the transport step, so every scalar is quoted only "
            "to the digits its own step-halving drift justifies (see "
            "justified_significant_figures_per_quantity and quoted_values), and "
            "quantities at the double-precision floor are quoted as upper bounds "
            "rather than values. CW and CCW are near-antipodal landing points of "
            "the same loop and are not independent confirmations of each other; "
            "the ALT loop is a second, non-collinear landing point, so the "
            "first-order prediction is exercised at two unrelated points of the "
            "fiber, though all of them are deterministic points chosen by the "
            "analyst rather than random samples; see D2_landing_point_relations. "
            "It is not hardware evidence, does not "
            "select a unique natural metric, and does not prove that all quantum "
            "computation is geometric flow."
        ),
        "configuration": asdict(cfg),
        "endpoint_geometry": geometry_public,
        "fd_truncation_audit": truncation,
        "loops": {
            name: {
                "kind": kind,
                "epsilon": epsilon,
                "plane": [int(plane[0]), int(plane[1])],
                "transport_step": step,
            }
            for name, (kind, epsilon, plane, step) in loops.items()
        },
        "task_generators": ["X", "Y", "N"],
        "transport_step_halving": step_halving,
        "second_loop_not_collinear": second_loop_gate,
        "justified_significant_figures_per_quantity": convergence,
        "quoted_values": quoted,
        "worst_landing_point_relative_drift": worst_drift,
        "quotation_rule": (
            "Each scalar is compared against its own value at half the transport "
            "step; digits = floor(-log10(relative drift)). Quantities at the "
            "double-precision floor are marked FLOOR_LIMITED and quoted only as "
            "an upper bound. The budget is measured on the CW landing point and "
            "applied to CCW, which D2 shows is the near-antipodal landing point "
            "of the same construction."
        ),
        "diagnostics": {
            "D2_landing_point_relations": relations,
        },
        "directions": direction_results,
        "global_gates": {
            "endpoint_geometry_stable": geometry["stable"],
            "fd_discarded_values_are_truncation": truncation["pass"],
            "all_transports_numerically_valid": all(
                run["numerical_pass"] for run in runs.values()
            ),
            "all_step_halving_audits_converged": convergence_gate,
            "second_loop_not_collinear_with_first": second_loop_gate["pass"],
            "all_landing_points_close_all_four_steps": all_direction_gates,
        },
    }
    return result, all_rows, runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2 four-step path-noise closure")
    parser.add_argument("--output-dir")

    raw = sys.argv[1:]
    cleaned: list[str] = []
    ignored: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == "-f" and i + 1 < len(raw):
            ignored.extend(raw[i : i + 2])
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
        args.output_dir or f"m2_four_step_closure_{time.strftime('%Y%m%d_%H%M%S')}"
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
    print(f"M2 FOUR-STEP PATH-RESOLVED NOISE CLOSURE  ({VERSION})")
    print("=" * 96)
    try:
        cfg = Cfg()
        result, rows, runs = audit(cfg)
        save_json(output / "certificate.json", result)
        save_csv(output / "noise_prediction.csv", rows)
        save_controls(output / "controls.csv", Model(cfg), runs)
        figure = save_plot(output / "path_noise_prediction.png", rows)
        save_latex_macros(output / "results_macros.tex", result)
        (output / "pip_freeze.txt").write_text(pip_freeze(), encoding="utf-8")

        summary.update(
            {
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
            }
        )

        concise = {
            "status": result["status"],
            "physical_support": result["physical_support"],
            "worst_landing_point_relative_drift": result[
                "worst_landing_point_relative_drift"
            ],
            "endpoint_geometry": {
                "rank": result["endpoint_geometry"]["rank_half"],
                "fiber_dimension": result["endpoint_geometry"]["fiber_dimension"],
                "stable": result["endpoint_geometry"]["stable"],
            },
            "fd_truncation_audit": {
                k: result["fd_truncation_audit"][k]
                for k in (
                    "discarded_ratio_h_over_h_half",
                    "accepted_ratio_range",
                    "pass",
                )
            },
            "loops": result["loops"],
            "transport_step_halving": result["transport_step_halving"],
            "second_loop_not_collinear": result["second_loop_not_collinear"],
            "diagnostics": result["diagnostics"],
            "quoted_values": {
                direction: {
                    dotted.split(".")[-1]: record["quoted"]
                    for dotted, record in fields.items()
                }
                for direction, fields in result["quoted_values"].items()
            },
            "directions": [
                {
                    "direction": item["direction"],
                    "gates": item["gates"],
                }
                for item in result["directions"]
            ],
            "global_gates": result["global_gates"],
            "claim_boundary": result["claim_boundary"],
        }
        print(json.dumps(clean(concise), indent=2, ensure_ascii=False))

        # v1.1 raised only on the transport gate, after everything was written,
        # so a failed geometry, truncation or convergence gate exited zero.
        failed = [
            name for name, ok in result["global_gates"].items() if not ok
        ]
        if failed:
            raise AssertionError(
                "global gates failed; do not interpret physics: " + ", ".join(failed)
            )
    except Exception as exc:
        summary.update(
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        save_json(output / "summary.json", summary)
        print(f"elapsed={summary['elapsed_seconds']:.2f}s")
        print(f"outputs={output}")


if __name__ == "__main__":
    main()
