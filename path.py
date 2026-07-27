#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EP-OBS-4SET-v1.2 -- amplified four-setting path-susceptibility search.

This file contains ONE experiment. The Stage-2 gap-revalidation entry point, the
Stage-3 dephasing entry point, and the v1.0 four-setting entry point have been
removed rather than carried along dead: in v1.1 all four wrote certificates
carrying the same VERSION string, which made archived certificates mutually
indistinguishable. Every certificate written here additionally carries
cert["experiment"] = "amplified_four_setting".

WHAT IS MEASURED
----------------
Two controls z_CW and z_CCW are transported around opposite orientations of a
closed task-space loop and then projected onto the endpoint fiber of the
MODULATED AnalogDevice waveform, so that

    U_mod(z_CW) = U_mod(z_CCW) = U_mod(0)

to within the declared endpoint tolerance. Both are therefore full-unitary
equivalent to the reference. A segment-local coherent detuning probe +/-delta is
then applied identically to each, and the four-arm antisymmetric witness

    S(delta) = 1/2 [ ( W_CCW(+d) - W_CCW(-d) ) - ( W_CW(+d) - W_CW(-d) ) ]

is evaluated on native computational-basis counts (rr, rg, gr, gg) with one
common outcome weight vector w.

WHAT THE WITNESS DOES AND DOES NOT ESTABLISH  (corrected from v1.1)
-------------------------------------------------------------------
S is EXACTLY odd in delta by construction:

    S(d) = odd[f_CCW](d) - odd[f_CW](d),

so the delta = 0 baseline cancels identically inside each bracket, whether or
not the two controls are endpoint equivalent. v1.1's docstring attributed the
baseline-free property to the endpoint match; that attribution was wrong.

Consequences, stated plainly:

  * S != 0 on its own witnesses NOTHING about the endpoint fiber. Two entirely
    inequivalent controls also give S != 0.
  * The scientific content is the CONJUNCTION of the endpoint gates (G2, G2b),
    the zero-probe gate (G3), and a nonzero, linear-in-delta S on a held-out
    grid (G5). The endpoint gates are what license the interpretation; the
    antisymmetry is what removes the baseline.
  * Because S is exactly odd, the leading contamination is O(delta^3), not
    O(delta^2). G11 fits S = a*delta + c*delta^3 on the held-out grid and
    reports c/a and the delta at which the cubic term reaches 1%.

The measured object is a difference between two POINTS of the fiber over
U_mod(0). "Path-conditioned" is a label for how those points were reached, not
for a measured quantity: the transport is the construction, the landing point is
the object. The landing point moves with the transport step h_s, so the frozen
(eps, segment, w, chi) are h_s-dependent design choices. G8 quantifies this by
repeating the selected configuration at h_s/2 and reporting the drift in chi.
Quote chi as a property of THIS landing point.

DESIGN / TEST SEPARATION
------------------------
Selection data (eps grid, probe segment, outcome weights, linear coefficient
chi) is frozen using only --selection-deltas-mhz. The disjoint
--test-deltas-mhz grid is touched exactly once, after freezing. The h_s
sensitivity audit (G8) also uses selection deltas only, so it does not
contaminate the held-out grid.

Everything is deterministic at the model level, so there is no overfitting to
noise in selection. The same procedure applied to real counts WOULD be biased,
and the optimal weights are optimal for simulated probabilities only.

SHOT FEASIBILITY  (corrected from v1.1)
---------------------------------------
shots_5sigma scales as delta^-2, so min-over-the-grid always lands on the
largest delta, which is also the delta with the worst linearity. v1.1 therefore
asserted "linear" and "feasible" at two different points of the same grid. Here
feasibility is evaluated on the LINEAR SUBSET only: the held-out deltas that
individually satisfy --heldout-relative-error-tol. Both numbers are reported.

Similarly, the standardized-gain target is no longer a hardcoded constant. Since
shots = 25 / (score^2 * delta^2), the self-consistent requirement at the largest
held-out delta is score >= 5 / (delta * sqrt(N_threshold)); --target-
standardized-gain defaults to that derived value.

Run:

    python ep_obs_four_setting_amplified_v1_2.py

Local Pulser only. No cloud credentials are requested and no job is submitted.

Author: script prepared for Y. Y. N. Li.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "ep_obs_matplotlib")
)

# ----------------------------------------------------------------------------
# 0. PROVENANCE
# ----------------------------------------------------------------------------

VERSION = "EP-OBS-4SET-v1.2"
EXPERIMENT = "amplified_four_setting"

_SOURCE_PATH_OVERRIDE = None


def self_sha256() -> str:
    """SHA-256 of the executed source, or an explicit 'unavailable'."""
    path = _SOURCE_PATH_OVERRIDE
    if path is None:
        try:
            path = os.path.abspath(__file__)
        except NameError:
            return "unavailable (no source path exposed)"
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        return "unavailable (no source path exposed)"


def package_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "scipy", "pulser", "pulser_simulation", "qutip", "matplotlib"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "not installed"
    return out


# ----------------------------------------------------------------------------
# 1. MODEL  (two atoms, ground-rydberg, Pulser convention)
# ----------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
NN = np.array([[1, 0], [0, 0]], dtype=complex)  # n = |r><r| in (|r>,|g>) ordering

X_OP = np.kron(SX, I2) + np.kron(I2, SX)
Y_OP = np.kron(SY, I2) + np.kron(I2, SY)
N_OP = np.kron(NN, I2) + np.kron(I2, NN)
NN_OP = np.kron(NN, NN)

DIM = 4

# Pulser computational basis order: |rr>, |rg>, |gr>, |gg>
IDX_RR, IDX_RG, IDX_GR, IDX_GG = 0, 1, 2, 3
OUTCOME_ORDER = ["rr", "rg", "gr", "gg"]

ATOM_SEP = 6.0  # um

# Fallback only. It equals MockDevice.interaction_coeff (rydberg_level 70) and is
# used ONLY for the pre-override MockDevice convention probe. AnalogDevice
# (rydberg_level 60) has a different coefficient and overrides this at runtime;
# see stage3_configure_analog_interaction and gate G0c.
C6 = 5420158.53
U_INT = C6 / ATOM_SEP**6

TAU = 0.120  # us, per drive segment
NSEG = 6

# gap_tau reproduces the idle AnalogDevice inserts at a phase jump
# (2 x phase_jump_time). In this experiment it affects only the M2 SEED: the
# propagated evolution is always the modulated waveform actually sampled from
# the compiled sequence. G12 audits the executed gap against this seed model.
MODEL = {"gap_tau": 0.0, "omega_scale": 1.0, "phase_sign": 1.0}

OMEGA0 = 2 * np.pi * np.array([2.0, 1.7, 2.3, 1.5, 2.1, 1.8])
DELTA0 = 2 * np.pi * np.array([-2.3, -1.2, 0.4, 1.4, 2.0, 0.8])
PHI0 = np.array([0.0, 0.4, 1.1, 2.0, 2.7, -2.4])


def controls_from_z(z: np.ndarray, phase_sign: "float | None" = None):
    """z in R^18 -> (Omega_j, Delta_j, phi_j). z[:,1] is in MHz."""
    if phase_sign is None:
        phase_sign = MODEL["phase_sign"]
    z = np.asarray(z, dtype=float).reshape(NSEG, 3)
    omega = MODEL["omega_scale"] * OMEGA0 * (1.0 + z[:, 0])
    delta = DELTA0 + 2 * np.pi * z[:, 1]
    phi = phase_sign * (PHI0 + z[:, 2])
    return omega, delta, phi


def hamiltonian(omega: float, delta: float, phi: float) -> np.ndarray:
    """Pulser-convention two-atom Rydberg Hamiltonian. Reads U_INT at call time."""
    return (
        0.5 * omega * (np.cos(phi) * X_OP - np.sin(phi) * Y_OP)
        - delta * N_OP
        + U_INT * NN_OP
    )


def prop_from_H(H: np.ndarray, tau: float) -> np.ndarray:
    w, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * w * tau)) @ V.conj().T


@dataclass
class Segment:
    omega: float
    delta: float
    phi: float
    tau: float


def segments_of(z, phase_sign=None):
    """Seed schedule: six drive segments with optional idle gaps between them."""
    omega, delta, phi = controls_from_z(z, phase_sign)
    gap = MODEL["gap_tau"]
    segs = []
    for j in range(NSEG):
        if gap > 0 and j > 0:
            segs.append(Segment(0.0, 0.0, 0.0, gap))
        segs.append(Segment(omega[j], delta[j], phi[j], TAU))
    return segs


def unitary_of_z(z: np.ndarray, phase_sign=None) -> np.ndarray:
    U = np.eye(DIM, dtype=complex)
    for sg in segments_of(z, phase_sign):
        U = prop_from_H(hamiltonian(sg.omega, sg.delta, sg.phi), sg.tau) @ U
    return U


OBSERVABLES = {
    "P_gg": np.diag([0.0, 0.0, 0.0, 1.0]),
    "P_rr": np.diag([1.0, 0.0, 0.0, 0.0]),
    "P_one": np.diag([0.0, 1.0, 1.0, 0.0]),
    "N_ryd": np.diag([2.0, 1.0, 1.0, 0.0]),
    "ZZ": np.diag([1.0, -1.0, -1.0, 1.0]),
}


# ----------------------------------------------------------------------------
# 2. SEED TRANSPORT (M2 lift on the analytic gap-aware schedule)
# ----------------------------------------------------------------------------


def u_target(s: np.ndarray, U0: np.ndarray) -> np.ndarray:
    from scipy.linalg import expm

    gen = -0.25j * (s[0] * X_OP + s[1] * Y_OP)
    return expm(gen) @ U0


def residual(z, s, U0, phase_sign=None) -> np.ndarray:
    Uz = unitary_of_z(z, phase_sign)
    Ut = u_target(np.asarray(s, float), U0)
    theta = np.angle(np.trace(Ut.conj().T @ Uz))
    R = np.exp(-1j * theta) * Uz - Ut
    return np.concatenate([R.real.ravel(), R.imag.ravel()])


def endpoint_infidelity(z, U0, phase_sign=None) -> float:
    Uz = unitary_of_z(z, phase_sign)
    return float(abs(1.0 - abs(np.trace(U0.conj().T @ Uz)) ** 2 / DIM**2))


def jacobians(z, s, U0, h=1e-6, phase_sign=None):
    z = np.asarray(z, float)
    s = np.asarray(s, float)
    Q = np.empty((32, 18))
    for i in range(18):
        e = np.zeros(18)
        e[i] = h
        Q[:, i] = (
            residual(z + e, s, U0, phase_sign) - residual(z - e, s, U0, phase_sign)
        ) / (2 * h)
    B = np.empty((32, 2))
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        B[:, i] = (
            residual(z, s + e, U0, phase_sign) - residual(z, s - e, U0, phase_sign)
        ) / (2 * h)
    return Q, B


def minnorm_solve(Q, rhs, rcond=1e-6):
    U, sv, Vt = np.linalg.svd(Q, full_matrices=False)
    keep = sv > rcond * sv[0]
    inv = np.zeros_like(sv)
    inv[keep] = 1.0 / sv[keep]
    dz = Vt.T @ (inv * (U.conj().T @ rhs))
    res = float(np.linalg.norm(Q @ dz - rhs))
    smin_keep = float(sv[keep].min()) if keep.any() else 0.0
    smax_drop = float(sv[~keep].max()) if (~keep).any() else 0.0
    return dz, res, int(keep.sum()), smin_keep, smax_drop


def correct_endpoint(z, s, U0, rcond=1e-6, iters=3, phase_sign=None):
    z = np.asarray(z, float).copy()
    for _ in range(iters):
        r = residual(z, s, U0, phase_sign)
        if np.linalg.norm(r) < 1e-13:
            break
        Q, _ = jacobians(z, s, U0, phase_sign=phase_sign)
        dz, _, _, _, _ = minnorm_solve(Q, -r, rcond)
        z = z + dz
    return z, float(np.linalg.norm(residual(z, s, U0, phase_sign)))


def m2_transport(U0, vertices, h_s, rcond=1e-6, phase_sign=None):
    z = np.zeros(18)
    s = np.array(vertices[0], float)
    audit = {
        "n_steps": 0,
        "ranks": [],
        "smin_keep": [],
        "smax_drop": [],
        "lift_res": [],
        "corr_res": [],
    }
    effective = []
    for a, b in zip(vertices[:-1], vertices[1:]):
        seg = np.array(b, float) - np.array(a, float)
        L = float(np.linalg.norm(seg))
        nst = max(1, int(round(L / h_s)))
        effective.append(float(L / nst))
        ds = seg / nst
        for _ in range(nst):
            Q, B = jacobians(z, s, U0, phase_sign=phase_sign)
            dz, lres, rank, smin, smax = minnorm_solve(Q, -(B @ ds), rcond)
            z = z + dz
            s = s + ds
            z, cres = correct_endpoint(z, s, U0, rcond, iters=2, phase_sign=phase_sign)
            audit["n_steps"] += 1
            audit["ranks"].append(rank)
            audit["smin_keep"].append(smin)
            audit["smax_drop"].append(smax)
            audit["lift_res"].append(lres)
            audit["corr_res"].append(cres)
    z, cres = correct_endpoint(z, s, U0, rcond, iters=6, phase_sign=phase_sign)
    audit["effective_h_s"] = float(np.mean(effective))
    audit["effective_h_s_spread"] = float(np.max(effective) - np.min(effective))
    audit["nominal_h_s"] = float(h_s)
    audit["reach_res"] = float(np.linalg.norm(s - np.array(vertices[-1], float)))
    audit["final_res"] = cres
    audit["rank_set"] = sorted(set(audit["ranks"]))
    audit["min_smin_keep"] = float(np.min(audit["smin_keep"]))
    audit["max_smax_drop"] = float(np.max(audit["smax_drop"]))
    audit["max_lift_res"] = float(np.max(audit["lift_res"]))
    audit["max_corr_res"] = float(np.max(audit["corr_res"]))
    for k in ("ranks", "smin_keep", "smax_drop", "lift_res", "corr_res"):
        audit.pop(k)
    return z, audit


def loop_vertices(eps, direction):
    if direction == "CW":
        return [(0, 0), (eps, 0), (eps, eps), (0, eps), (0, 0)]
    return [(0, 0), (0, eps), (eps, eps), (eps, 0), (0, 0)]


def loglog_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = (x > 0) & (y > 0)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


# ----------------------------------------------------------------------------
# 3. PULSER COMPILATION
# ----------------------------------------------------------------------------


def build_pulser_sequence(segs, device_name="mock"):
    from pulser import Register, Sequence, Pulse
    from pulser.devices import MockDevice, AnalogDevice

    dev = MockDevice if device_name == "mock" else AnalogDevice
    reg = Register.from_coordinates([[0.0, 0.0], [ATOM_SEP, 0.0]], prefix="q")
    seq = Sequence(reg, dev)
    seq.declare_channel("ryd", "rydberg_global")
    for sg in segs:
        dur_ns = int(round(sg.tau * 1000))
        if sg.omega == 0.0 and sg.delta == 0.0:
            if device_name == "mock":
                seq.delay(dur_ns, "ryd")
            # On AnalogDevice the idle is inserted automatically at a phase jump;
            # adding it explicitly would double-count it. G12 audits that the
            # executed idle actually matches MODEL['gap_tau'].
            continue
        seq.add(
            Pulse.ConstantPulse(
                dur_ns, float(sg.omega), float(sg.delta), float(sg.phi) % (2 * np.pi)
            ),
            "ryd",
        )
    return seq


def compiled_blocks(seq, modulation=False):
    from pulser.sampler import sample

    ch = sample(seq, modulation=modulation).channel_samples["ryd"]
    amp, det, ph = np.asarray(ch.amp), np.asarray(ch.det), np.asarray(ch.phase)
    key = np.stack([amp, det, ph], axis=1)
    starts = [0] + list(np.where(np.any(np.diff(key, axis=0) != 0, axis=1))[0] + 1)
    starts.append(len(amp))
    return [
        Segment(amp[a], det[a], ph[a], (b - a) * 1e-3)
        for a, b in zip(starts[:-1], starts[1:])
    ]


def unitary_of_blocks(blocks) -> np.ndarray:
    U = np.eye(DIM, dtype=complex)
    for sg in blocks:
        U = prop_from_H(hamiltonian(sg.omega, sg.delta, sg.phi), sg.tau) @ U
    return U


def split_blocks(blocks, factor: int):
    """Subdivide every block. Same physics, different floating-point path.

    Used only to bound the numerical noise floor of the witness (G10).
    """
    if factor <= 1:
        return list(blocks)
    out = []
    for sg in blocks:
        for _ in range(factor):
            out.append(Segment(sg.omega, sg.delta, sg.phi, sg.tau / factor))
    return out


def pulser_hamiltonian_probe(device_name="mock"):
    """Assert hamiltonian() equals Pulser's H on the NAMED device.

    v1.1 hardcoded MockDevice and ran this BEFORE overriding C6 with
    AnalogDevice's value, so the gate certified a Hamiltonian that was not the
    one subsequently propagated. The device is now a parameter and the probe is
    run again after the override (gate G0c).

    Amplitude is kept below AnalogDevice's max_amp = 2*pi*2.0 rad/us so the same
    probe compiles on both devices.
    """
    from pulser_simulation import QutipEmulator

    seg = Segment(2 * np.pi * 1.84, 2 * np.pi * (-2.3), 0.4, 0.120)
    seq = build_pulser_sequence([seg], device_name)
    sim = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
    H_pulser = np.asarray(sim.get_hamiltonian(60).full())
    H_ours = hamiltonian(seg.omega, seg.delta, seg.phi)
    err = float(np.max(np.abs(H_pulser - H_ours)))
    init = np.asarray(sim.initial_state.full()).ravel()
    gg_ok = bool(abs(abs(init[IDX_GG]) - 1.0) < 1e-12)
    return err, gg_ok


def stage3_configure_analog_interaction():
    """Adopt the interaction coefficient declared by the active AnalogDevice.

    NOTE: this mutates module globals. Nothing in this file runs on MockDevice
    physics after this point except the already-completed G0a probe.
    """
    from pulser.devices import AnalogDevice

    global C6, U_INT
    fallback = float(C6)
    device_c6 = float(AnalogDevice.interaction_coeff)
    C6 = device_c6
    U_INT = C6 / ATOM_SEP**6
    return {
        "fallback_C6": fallback,
        "AnalogDevice_C6": device_c6,
        "absolute_difference_before_override": abs(device_c6 - fallback),
        "U_int_at_declared_separation": float(U_INT),
        "source": "pulser.devices.AnalogDevice.interaction_coeff",
    }


def analog_device_limits() -> dict:
    from pulser.devices import AnalogDevice

    ch = AnalogDevice.channels["rydberg_global"]
    return {
        "rydberg_level": int(AnalogDevice.rydberg_level),
        "interaction_coeff": float(AnalogDevice.interaction_coeff),
        "max_amp_rad_per_us": float(ch.max_amp),
        "max_abs_detuning_rad_per_us": float(ch.max_abs_detuning),
        "phase_jump_time_ns": float(ch.phase_jump_time),
        "clock_period_ns": float(ch.clock_period),
        "min_atom_distance_um": float(AnalogDevice.min_atom_distance),
        "max_sequence_duration_ns": (
            None
            if AnalogDevice.max_sequence_duration is None
            else int(AnalogDevice.max_sequence_duration)
        ),
    }


def executed_gap_audit(z) -> dict:
    """Compare the idle AnalogDevice actually inserts with MODEL['gap_tau'].

    build_pulser_sequence relies on the device inserting 2 x phase_jump_time at
    every phase jump. That assumption fails silently if two consecutive segments
    share a phase modulo 2*pi, so the minimum phase separation is reported too.
    This is informational: the propagated evolution is the sampled waveform, so
    a mismatch degrades only the quality of the M2 seed.
    """
    seq = build_pulser_sequence(segments_of(z), "analog")
    blocks = compiled_blocks(seq, modulation=False)
    idle = [b.tau for b in blocks if b.omega == 0.0 and b.delta == 0.0]
    n_drive = len([b for b in blocks if b.omega != 0.0])
    inferred = float(sum(idle) / max(n_drive - 1, 1))
    _, _, phi = controls_from_z(z)
    ph = np.mod(np.asarray(phi, float), 2 * np.pi)
    diffs = np.abs(np.diff(ph))
    min_phase_sep = float(np.min(np.minimum(diffs, 2 * np.pi - diffs)))
    return {
        "inferred_idle_per_junction_us": inferred,
        "seed_model_gap_us": float(MODEL["gap_tau"]),
        "absolute_difference_us": abs(inferred - MODEL["gap_tau"]),
        "minimum_consecutive_phase_separation_rad": min_phase_sep,
        "all_consecutive_phases_distinct": bool(min_phase_sep > 1e-9),
    }


# ----------------------------------------------------------------------------
# 4. MODULATED ENDPOINT RE-LIFT
# ----------------------------------------------------------------------------


def stage3_modulated_blocks(z: np.ndarray):
    seq = build_pulser_sequence(segments_of(z), "analog")
    return compiled_blocks(seq, modulation=True)


def stage3_modulated_unitary(z: np.ndarray) -> np.ndarray:
    return unitary_of_blocks(stage3_modulated_blocks(z))


def stage3_phase_aligned_residual(U: np.ndarray, Ut: np.ndarray) -> np.ndarray:
    phase = np.angle(np.trace(Ut.conj().T @ U))
    R = np.exp(-1j * phase) * U - Ut
    return np.concatenate([R.real.ravel(), R.imag.ravel()])


def stage3_endpoint_infidelity(U: np.ndarray, Ut: np.ndarray) -> float:
    return float(abs(1.0 - abs(np.trace(Ut.conj().T @ U)) ** 2 / DIM**2))


def stage3_modulated_jacobian(z: np.ndarray, U_target: np.ndarray, h: float):
    z = np.asarray(z, float)
    Q = np.empty((2 * DIM * DIM, z.size), dtype=float)
    for j in range(z.size):
        e = np.zeros_like(z)
        e[j] = h
        rp = stage3_phase_aligned_residual(stage3_modulated_unitary(z + e), U_target)
        rm = stage3_phase_aligned_residual(stage3_modulated_unitary(z - e), U_target)
        Q[:, j] = (rp - rm) / (2.0 * h)
    return Q


def stage3_modulated_relift(
    z_seed,
    U_target,
    *,
    fd_step: float,
    max_iters: int,
    residual_tol: float,
    rcond: float,
    log,
):
    """Project a gap-aware seed onto the modulated endpoint fiber.

    v1.1's line search broke on the first alpha with rnt < rn, which made the
    "if not accepted and best[0] < rn" fallback unreachable: if nothing was
    accepted then every rnt >= rn, hence best[0] >= rn. The advertised safety
    net did not exist. Here every alpha is evaluated and the best one is taken,
    so the fallback is real. Cost is ~5 endpoint evaluations per iteration
    against ~36 for the Jacobian, i.e. about +10%.
    """
    z = np.asarray(z_seed, float).copy()
    history = []
    rank_set = set()
    termination = "iterations_exhausted"
    for iteration in range(max_iters):
        U = stage3_modulated_unitary(z)
        r = stage3_phase_aligned_residual(U, U_target)
        rn = float(np.linalg.norm(r))
        eu = stage3_endpoint_infidelity(U, U_target)
        if rn <= residual_tol:
            termination = "residual_tolerance_met"
            history.append(
                {
                    "iteration": iteration,
                    "residual_norm": rn,
                    "endpoint_infidelity": eu,
                    "accepted_alpha": None,
                    "status": "TOLERANCE_MET",
                }
            )
            break

        Q = stage3_modulated_jacobian(z, U_target, fd_step)
        Us, sv, Vt = np.linalg.svd(Q, full_matrices=False)
        keep = sv > rcond * sv[0]
        rank = int(np.count_nonzero(keep))
        rank_set.add(rank)
        inv = np.zeros_like(sv)
        inv[keep] = 1.0 / sv[keep]
        dz = Vt.T @ (inv * (Us.T @ (-r)))

        dz_norm = float(np.linalg.norm(dz))
        if dz_norm > 0.20:
            dz *= 0.20 / dz_norm
            dz_norm = 0.20

        best = None
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = z + alpha * dz
            try:
                U_trial = stage3_modulated_unitary(trial)
            except Exception:
                continue
            rnt = float(
                np.linalg.norm(stage3_phase_aligned_residual(U_trial, U_target))
            )
            if best is None or rnt < best[0]:
                best = (rnt, alpha, trial, U_trial)

        if best is None or best[0] >= rn:
            termination = "line_search_stalled"
            history.append(
                {
                    "iteration": iteration,
                    "residual_norm": rn,
                    "endpoint_infidelity": eu,
                    "jacobian_rank": rank,
                    "step_norm": dz_norm,
                    "accepted_alpha": None,
                    "best_trial_residual_norm": (
                        None if best is None else float(best[0])
                    ),
                    "status": "LINE_SEARCH_STALLED",
                }
            )
            break

        rnt, alpha, z, U_trial = best
        eut = stage3_endpoint_infidelity(U_trial, U_target)
        history.append(
            {
                "iteration": iteration,
                "residual_norm": rn,
                "endpoint_infidelity": eu,
                "jacobian_rank": rank,
                "minimum_kept_singular_value": float(sv[keep].min()),
                "maximum_dropped_singular_value": (
                    float(sv[~keep].max()) if np.any(~keep) else 0.0
                ),
                "step_norm": dz_norm,
                "accepted_alpha": float(alpha),
                "accepted_residual_norm": float(rnt),
                "accepted_endpoint_infidelity": eut,
            }
        )
        log(
            f"      iter={iteration:02d} rank={rank:2d} "
            f"|r|={rn:.3e}->{rnt:.3e} epsU={eu:.3e}->{eut:.3e} alpha={alpha:g}"
        )

    U_final = stage3_modulated_unitary(z)
    r_final = stage3_phase_aligned_residual(U_final, U_target)
    return z, U_final, {
        "history": history,
        "termination": termination,
        "rank_set": sorted(rank_set),
        "final_residual_norm": float(np.linalg.norm(r_final)),
        "final_endpoint_infidelity": stage3_endpoint_infidelity(U_final, U_target),
        "correction_norm": float(np.linalg.norm(z - np.asarray(z_seed, float))),
        "seed_norm": float(np.linalg.norm(z_seed)),
        "final_norm": float(np.linalg.norm(z)),
    }


# ----------------------------------------------------------------------------
# 5. FOUR-SETTING PROBE
# ----------------------------------------------------------------------------


def _four_parse_positive_csv(text: str, name: str) -> list:
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated float grid") from exc
    if not values or any((not np.isfinite(x)) or x <= 0.0 for x in values):
        raise ValueError(f"{name} must contain finite positive values")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values")
    return sorted(values)


def _four_parse_segments(text: str) -> list:
    if text.strip().lower() == "all":
        return list(range(NSEG))
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(
            "--probe-segments must be 'all' or a 1-based integer list"
        ) from exc
    if not values or any(index < 1 or index > NSEG for index in values):
        raise ValueError(f"probe segment indices must lie in 1..{NSEG}")
    if len(set(values)) != len(values):
        raise ValueError("--probe-segments contains duplicates")
    return [index - 1 for index in values]


def four_apply_detuning_probe(z, segment_index: int, delta_mhz: float) -> np.ndarray:
    """Coherent detuning offset on one drive segment.

    z[j,1] is in MHz because controls_from_z maps it through
    Delta_j = Delta0_j + 2*pi*z[j,1]. The probe leaves every phase untouched, so
    the compiled block structure and the executed duration are unchanged.
    """
    out = np.asarray(z, dtype=float).reshape(NSEG, 3).copy()
    out[segment_index, 1] += float(delta_mhz)
    return out.reshape(-1)


def four_native_record(z, block_split: int = 1) -> dict:
    blocks = stage3_modulated_blocks(z)
    U = unitary_of_blocks(split_blocks(blocks, block_split))
    psi0 = np.zeros(DIM, dtype=complex)
    psi0[IDX_GG] = 1.0
    psi = U @ psi0
    probabilities = np.maximum(np.real(psi.conj() * psi), 0.0)
    probabilities /= probabilities.sum()
    means = {}
    for name, observable in OBSERVABLES.items():
        outcomes = np.real(np.diag(observable))
        means[name] = float(probabilities @ outcomes)
    return {
        "blocks": blocks,
        "unitary": U,
        "probabilities": probabilities,
        "means": means,
        "n_blocks": len(blocks),
        "duration_us": float(sum(block.tau for block in blocks)),
    }


def four_evaluate_settings(z_cw, z_ccw, segment_index, delta_mhz, block_split=1):
    return {
        "CW_plus": four_native_record(
            four_apply_detuning_probe(z_cw, segment_index, +delta_mhz), block_split
        ),
        "CW_minus": four_native_record(
            four_apply_detuning_probe(z_cw, segment_index, -delta_mhz), block_split
        ),
        "CCW_plus": four_native_record(
            four_apply_detuning_probe(z_ccw, segment_index, +delta_mhz), block_split
        ),
        "CCW_minus": four_native_record(
            four_apply_detuning_probe(z_ccw, segment_index, -delta_mhz), block_split
        ),
    }


ARM_COEFFICIENTS = {
    "CCW_plus": +0.5,
    "CCW_minus": -0.5,
    "CW_plus": -0.5,
    "CW_minus": +0.5,
}


def amplified_contrast_statistics(records: dict) -> dict:
    """Native four-outcome contrast vector and its four-arm covariance.

        S = w . contrast ,   Var[S_hat] = (w . Cov . w) / shots_per_arm .
    """
    contrast = np.zeros(DIM, dtype=float)
    covariance = np.zeros((DIM, DIM), dtype=float)
    probabilities = {}
    for arm, coefficient in ARM_COEFFICIENTS.items():
        p = np.asarray(records[arm]["probabilities"], dtype=float)
        probabilities[arm] = p
        contrast += coefficient * p
        covariance += (coefficient * coefficient) * (np.diag(p) - np.outer(p, p))
    return {
        "contrast": contrast,
        "covariance": covariance,
        "probabilities": probabilities,
        "minimum_outcome_probability": float(
            min(float(p.min()) for p in probabilities.values())
        ),
    }


def amplified_optimal_weights(
    statistics: list, deltas_mhz: list, ridge: float = 0.0, rcond: float = 1e-12
) -> dict:
    """Optimal native score frozen from selection data only.

    Conditioning caveat, now reported rather than hidden: the multinomial
    covariance has an eigenvalue of order p_k for each outcome k. With
    rcond = 1e-12 relative to a largest eigenvalue of order 0.25, a direction
    with p_k ~ 1e-6 is RETAINED and inverted, giving w_k ~ 1e6 and, after the
    max-norm rescale, a score that is effectively "count outcome k only". That
    is optimal under the noiseless multinomial model but places the entire shot
    budget on a rare outcome, where hardware SPAM and detection infidelity
    dominate. Use --covariance-ridge to regularize; G9 reports the spectrum,
    the retained rank, the weight concentration, and the minimum outcome
    probability so the caveat is visible in the certificate.

    A constant shift of all weights changes no contrast and carries no variance;
    the pseudoinverse already returns a vector orthogonal to that gauge
    direction, so the explicit mean subtraction below is a no-op kept only for
    numerical hygiene.
    """
    slope_vectors = np.stack(
        [stat["contrast"] / delta for stat, delta in zip(statistics, deltas_mhz)]
    )
    covariance_mean = np.mean(
        np.stack([stat["covariance"] for stat in statistics]), axis=0
    )
    if ridge > 0.0:
        covariance_mean = covariance_mean + ridge * (
            float(np.trace(covariance_mean)) / DIM
        ) * np.eye(DIM)
    slope_vector = np.mean(slope_vectors, axis=0)
    eigenvalues = np.linalg.eigvalsh(covariance_mean)
    covariance_pinv = np.linalg.pinv(covariance_mean, rcond=rcond)
    weights = covariance_pinv @ slope_vector
    weights = weights - np.mean(weights)
    scale = float(np.max(np.abs(weights)))
    if not np.isfinite(scale) or scale <= 1e-14:
        raise RuntimeError("optimal native score is numerically degenerate")
    weights = weights / scale
    retained_rank = int(
        np.count_nonzero(eigenvalues > rcond * max(float(eigenvalues.max()), 1e-300))
    )
    return {
        "weights": weights,
        "selection_mean_slope_vector": slope_vector,
        "selection_mean_covariance": covariance_mean,
        "covariance_eigenvalues": eigenvalues,
        "retained_rank": retained_rank,
        "weight_concentration": float(
            np.max(np.abs(weights)) / max(float(np.sum(np.abs(weights))), 1e-300)
        ),
        "selection_generalized_score_bound": float(
            np.sqrt(max(float(slope_vector @ covariance_pinv @ slope_vector), 0.0))
        ),
    }


def amplified_weighted_witness(statistics: dict, weights: np.ndarray) -> dict:
    signal = float(weights @ statistics["contrast"])
    variance_numerator = float(max(weights @ statistics["covariance"] @ weights, 0.0))
    shots = (
        float(25.0 * variance_numerator / (signal * signal))
        if abs(signal) > 1e-300
        else float("inf")
    )
    return {
        "signal": signal,
        "variance_numerator_per_arm": variance_numerator,
        "shots_per_arm_5sigma": shots,
        "standardized_score_per_mhz": None,
        "arm_weighted_means": {
            arm: float(weights @ probability)
            for arm, probability in statistics["probabilities"].items()
        },
    }


def four_prepare_modulated_paths(args, eps: float, h_s: float, log) -> dict:
    """Reproduce and re-lift the CW/CCW pair at one loop scale.

    v1.1 mutated args.eps inside the selection loop and read it here, which also
    corrupted the certificate's recorded arguments. eps and h_s are now explicit.
    """
    U0_gap = unitary_of_z(np.zeros(18))
    seeds = {}
    log(f"  [eps={eps:g}, h_s={h_s:g}] STEP 1  gap-aware M2 seeds")
    for direction in ("CW", "CCW"):
        started = time.time()
        z_seed, audit = m2_transport(
            U0_gap, loop_vertices(eps, direction), h_s, rcond=args.rcond
        )
        seeds[direction] = {
            "z": z_seed,
            "audit": audit,
            "gap_endpoint_infidelity": endpoint_infidelity(z_seed, U0_gap),
        }
        log(
            f"    {direction}: steps={audit['n_steps']} ranks={audit['rank_set']} "
            f"gap-epsU={seeds[direction]['gap_endpoint_infidelity']:.3e} "
            f"|z|={np.linalg.norm(z_seed):.3e} ({time.time()-started:.1f}s)"
        )

    log("  STEP 2  projection onto the modulated full-unitary fiber")
    z0 = np.zeros(18)
    U0_mod = stage3_modulated_unitary(z0)
    corrected = {}
    for direction in ("CW", "CCW"):
        z_seed = seeds[direction]["z"]
        seed_error = stage3_endpoint_infidelity(
            stage3_modulated_unitary(z_seed), U0_mod
        )
        log(f"    {direction}: seed modulated epsU={seed_error:.3e}")
        z, U_solver, audit = stage3_modulated_relift(
            z_seed,
            U0_mod,
            fd_step=args.fd_step,
            max_iters=args.relift_iters,
            residual_tol=args.endpoint_residual_tol,
            rcond=args.rcond,
            log=log,
        )
        record = four_native_record(z)
        endpoint_error = stage3_endpoint_infidelity(record["unitary"], U0_mod)
        residual_norm = float(
            np.linalg.norm(stage3_phase_aligned_residual(record["unitary"], U0_mod))
        )
        corrected[direction] = {
            "z": z,
            "record": record,
            "audit": audit,
            "seed_modulated_endpoint_infidelity": seed_error,
            "endpoint_infidelity": endpoint_error,
            "endpoint_residual_norm": residual_norm,
            "solver_resampling_error": float(
                np.max(np.abs(record["unitary"] - U_solver))
            ),
        }
        log(
            f"    final epsU={endpoint_error:.3e} |r|={residual_norm:.3e} "
            f"|dz|={audit['correction_norm']:.3e} |z|={audit['final_norm']:.3e} "
            f"term={audit['termination']}"
        )

    # Distinctness. The CW-vs-CCW PROCESS infidelity is ~0 by design (both equal
    # U0_mod), so it is a consistency check, NOT a separation measure. v1.1 had
    # no separation gate at all: a re-lift that collapsed both seeds onto the
    # same fiber point would have passed every endpoint gate with S == 0.
    pair_separation = float(
        np.linalg.norm(corrected["CW"]["z"] - corrected["CCW"]["z"])
    )
    pair_unitary_infidelity = stage3_endpoint_infidelity(
        corrected["CW"]["record"]["unitary"], corrected["CCW"]["record"]["unitary"]
    )
    pair_probability_tvd = float(
        0.5
        * np.sum(
            np.abs(
                corrected["CW"]["record"]["probabilities"]
                - corrected["CCW"]["record"]["probabilities"]
            )
        )
    )
    durations = {d: corrected[d]["record"]["duration_us"] for d in corrected}
    endpoint_pass = all(
        corrected[d]["endpoint_infidelity"] <= args.endpoint_infidelity_tol
        and corrected[d]["endpoint_residual_norm"] <= args.endpoint_residual_tol
        and corrected[d]["solver_resampling_error"] <= 1e-12
        for d in corrected
    )
    duration_pass = abs(durations["CW"] - durations["CCW"]) <= 1e-9
    separation_pass = pair_separation >= args.pair_separation_tol
    path_nontrivial = all(
        corrected[d]["audit"]["final_norm"] > 1e-5 for d in corrected
    )
    log(
        f"    pair: ||z_CW - z_CCW||={pair_separation:.3e} "
        f"(tol {args.pair_separation_tol:.1e}) "
        f"process-infid={pair_unitary_infidelity:.3e} (expected ~0) "
        f"TVD={pair_probability_tvd:.3e}"
    )
    return {
        "eps": float(eps),
        "h_s": float(h_s),
        "reference_z": z0,
        "reference_unitary": U0_mod,
        "seeds": seeds,
        "corrected": corrected,
        "pair_separation": pair_separation,
        "pair_unitary_infidelity": pair_unitary_infidelity,
        "pair_probability_tvd": pair_probability_tvd,
        "durations_us": durations,
        "endpoint_pass": bool(endpoint_pass),
        "duration_pass": bool(duration_pass),
        "separation_pass": bool(separation_pass),
        "path_nontrivial": bool(path_nontrivial),
    }


def evaluate_candidate(
    paths, segment_index, deltas, ridge, log=None, weights=None
):
    """Selection-side evaluation of one (paths, segment) candidate."""
    z_cw = paths["corrected"]["CW"]["z"]
    z_ccw = paths["corrected"]["CCW"]["z"]
    statistics = [
        amplified_contrast_statistics(
            four_evaluate_settings(z_cw, z_ccw, segment_index, delta)
        )
        for delta in deltas
    ]
    optimal = None
    if weights is None:
        optimal = amplified_optimal_weights(statistics, deltas, ridge=ridge)
        weights = optimal["weights"]
    witnesses = [amplified_weighted_witness(stat, weights) for stat in statistics]
    slopes = np.asarray(
        [w["signal"] / d for w, d in zip(witnesses, deltas)], float
    )
    scores = np.asarray(
        [
            abs(w["signal"] / d)
            / np.sqrt(max(w["variance_numerator_per_arm"], 1e-300))
            for w, d in zip(witnesses, deltas)
        ],
        float,
    )
    return {
        "statistics": statistics,
        "optimal": optimal,
        "weights": np.asarray(weights, float),
        "witnesses": witnesses,
        "median_slope_per_mhz": float(np.median(slopes)),
        "median_standardized_score": float(np.median(scores)),
        "relative_slope_spread": float(
            np.ptp(slopes) / max(abs(float(np.median(slopes))), 1e-300)
        ),
        "best_selection_shots_per_arm": float(
            min(w["shots_per_arm_5sigma"] for w in witnesses)
        ),
        "minimum_outcome_probability": float(
            min(stat["minimum_outcome_probability"] for stat in statistics)
        ),
    }


def cubic_fit(deltas, signals) -> dict:
    """Informational fit S = a*delta + c*delta^3 on the held-out grid.

    S is exactly odd, so this is the correct leading correction; v1.1's residual
    bookkeeping implicitly assumed an O(delta^2) term that cannot exist.
    """
    d = np.asarray(deltas, float)
    s = np.asarray(signals, float)
    if d.size < 2:
        return {"a": float("nan"), "c": float("nan")}
    A = np.stack([d, d**3], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, s, rcond=None)
    a, c = float(coeffs[0]), float(coeffs[1])
    ratio = abs(c / a) if abs(a) > 1e-300 else float("inf")
    return {
        "linear_coefficient_a": a,
        "cubic_coefficient_c": c,
        "abs_c_over_a": ratio,
        "delta_at_one_percent_cubic": (
            float(np.sqrt(0.01 / ratio)) if ratio > 0 else float("inf")
        ),
        "max_abs_relative_cubic_on_grid": (
            float(ratio * float(np.max(d)) ** 2) if np.isfinite(ratio) else float("inf")
        ),
    }


def _in_notebook() -> bool:
    if "ipykernel" in sys.modules or "google.colab" in sys.modules:
        return True
    return any("kernel-" in a and a.endswith(".json") for a in sys.argv[1:])


# ----------------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------------

BLOCKING_GATES = (
    "G0a_mock_hamiltonian_convention",
    "G0b_analogdevice_interaction_loaded",
    "G0c_analogdevice_hamiltonian_verified",
    "G1_selection_search_completed",
    "G2_selected_modulated_endpoint",
    "G2b_path_pair_separation",
    "G3_zero_probe_weighted_output_match",
    "G4_heldout_program_acceptance_and_duration",
    "G5_heldout_linear_path_susceptibility",
)
SHOT_GATE = "G7_shot_feasibility_on_linear_subset"


def amplified_main(argv=None):
    # allow_abbrev=False: in v1.1 "--eps 0.12" was silently absorbed by
    # --eps-grid through argparse prefix matching, collapsing the selection grid
    # to a single point with no warning. Unknown arguments are now a hard error
    # rather than a [note], so flags carried over from an earlier entry point
    # cannot silently change the experiment.
    ap = argparse.ArgumentParser(
        prog="ep_obs_four_setting_amplified_v1_2.py", allow_abbrev=False
    )
    ap.add_argument(
        "--quick",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use h_s=0.008 for each M2 seed transport (default: enabled)",
    )
    ap.add_argument(
        "--eps-grid",
        default="0.04,0.08,0.12,0.16,0.18",
        help="selection-only M2 loop scales; exactly one is frozen",
    )
    ap.add_argument("--hs", type=float, default=0.002)
    ap.add_argument("--gap", type=float, default=0.340)
    ap.add_argument("--omega-scale", type=float, default=0.80)
    ap.add_argument("--phase-sign", type=float, default=1.0, choices=[1.0, -1.0])
    ap.add_argument("--fd-step", type=float, default=2e-5)
    ap.add_argument("--relift-iters", type=int, default=8)
    ap.add_argument("--endpoint-residual-tol", type=float, default=2e-9)
    ap.add_argument("--endpoint-infidelity-tol", type=float, default=1e-11)
    ap.add_argument("--rcond", type=float, default=1e-6)
    ap.add_argument(
        "--pair-separation-tol",
        type=float,
        default=1e-6,
        help="minimum ||z_CW - z_CCW||; guards against a collapsed re-lift",
    )
    ap.add_argument("--selection-deltas-mhz", default="0.005,0.010")
    ap.add_argument("--test-deltas-mhz", default="0.020,0.040,0.080,0.120,0.200")
    ap.add_argument("--probe-segments", default="all")
    ap.add_argument("--shot-threshold", type=float, default=1e7)
    ap.add_argument(
        "--target-standardized-gain",
        type=float,
        default=None,
        help="default: derived from --shot-threshold and the largest held-out "
        "delta, since shots = 25/(score^2 delta^2)",
    )
    ap.add_argument("--heldout-relative-error-tol", type=float, default=0.10)
    ap.add_argument(
        "--cubic-contamination-tol",
        type=float,
        default=0.02,
        help="G11 threshold on |c/a| * delta_max^2, the cubic-over-linear ratio "
        "at the largest held-out delta",
    )
    ap.add_argument(
        "--min-outcome-probability",
        type=float,
        default=0.0,
        help="if > 0, turns the score-conditioning report into a gate requiring "
        "every native outcome probability to stay above this floor",
    )
    ap.add_argument(
        "--covariance-ridge",
        type=float,
        default=0.0,
        help="relative ridge added to the selection covariance before inversion; "
        "0.0 reproduces the unregularized optimal score",
    )
    ap.add_argument(
        "--hs-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="repeat the frozen configuration at h_s/2 and report the drift in "
        "chi (selection deltas only; the held-out grid is untouched)",
    )
    ap.add_argument("--numerical-floor-split", type=int, default=2)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--source-path", default=None)

    if argv is None:
        argv = [] if _in_notebook() else sys.argv[1:]
    args, unknown = ap.parse_known_args(list(argv))
    if unknown:
        ap.error("unrecognized arguments: " + " ".join(unknown))

    eps_grid = _four_parse_positive_csv(args.eps_grid, "--eps-grid")
    selection_deltas = _four_parse_positive_csv(
        args.selection_deltas_mhz, "--selection-deltas-mhz"
    )
    test_deltas = _four_parse_positive_csv(args.test_deltas_mhz, "--test-deltas-mhz")
    overlap = sorted(set(selection_deltas).intersection(test_deltas))
    if overlap:
        raise ValueError(
            "selection/test delta grids must be disjoint; overlap="
            + ",".join(map(str, overlap))
        )
    probe_segments = _four_parse_segments(args.probe_segments)
    if args.outdir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.outdir = f"ep_obs_four_setting_amplified_v1_2_{stamp}"

    global _SOURCE_PATH_OVERRIDE
    if args.source_path:
        _SOURCE_PATH_OVERRIDE = args.source_path
    MODEL["gap_tau"] = float(args.gap)
    MODEL["omega_scale"] = float(args.omega_scale)
    MODEL["phase_sign"] = float(args.phase_sign)
    os.makedirs(args.outdir, exist_ok=True)
    logfile = open(
        os.path.join(args.outdir, "four_setting_v1_2_run.log"), "w", encoding="utf-8"
    )

    def log(message=""):
        print(message)
        logfile.write(str(message) + "\n")
        logfile.flush()

    h_s = 0.008 if args.quick else args.hs

    cert = {
        "version": VERSION,
        "experiment": EXPERIMENT,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_sha256": self_sha256(),
        "packages": package_versions(),
        # dict(...) snapshot: v1.1 stored vars(args) itself, which is the live
        # Namespace.__dict__, so later mutation rewrote the recorded arguments.
        "args": dict(vars(args)),
        "effective_h_s_nominal": float(h_s),
        "eps_selection_grid": eps_grid,
        "selection_deltas_mhz": selection_deltas,
        "heldout_test_deltas_mhz": test_deltas,
        "probe_segments_1based": [index + 1 for index in probe_segments],
        "native_outcome_order": OUTCOME_ORDER,
        "cloud_access": "none",
        "gates": {},
        "diagnostics": {},
    }

    def add_gate(name, payload, blocking):
        """A gate has a genuine pass/fail decision against a declared threshold."""
        payload = dict(payload)
        if "pass" not in payload:
            raise RuntimeError(f"gate {name} has no pass/fail decision")
        payload["blocking"] = bool(blocking)
        cert["gates"][name] = payload
        return payload

    def add_diagnostic(name, payload):
        """A diagnostic REPORTS numbers. It has no pass/fail and never claims one.

        v1.2-rc1 wrote '"pass": True' for the conditioning report and the cubic
        fit. Neither is a test, so a hardcoded PASS was a claim the code had not
        earned; a reader could not distinguish 'checked and passed' from 'not
        checked'. They are recorded here instead.
        """
        payload = dict(payload)
        payload.pop("pass", None)
        payload["status"] = "REPORTED"
        cert["diagnostics"][name] = payload
        return payload

    def write_cert():
        with open(
            os.path.join(args.outdir, "four_setting_v1_2_certificate.json"),
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(cert, fh, indent=2, default=float)

    log("=" * 92)
    log(f"{VERSION}  AMPLIFIED FOUR-SETTING SEARCH  ({EXPERIMENT})")
    log("=" * 92)
    log(f"UTC={cert['utc']}  h_s={h_s:g}  gap={MODEL['gap_tau']:g} us")
    log(
        f"eps grid={eps_grid} | selection deltas={selection_deltas} MHz | "
        f"held-out deltas={test_deltas} MHz | cloud access=none"
    )
    for key, value in cert["packages"].items():
        log(f"  {key:20s} {value}")

    try:
        from pulser.devices import AnalogDevice

        _ = AnalogDevice.channels["rydberg_global"]
    except Exception as exc:
        log(f"\nFATAL: Pulser AnalogDevice unavailable: {type(exc).__name__}: {exc}")
        cert["scientific_status"] = "PULSER_ANALOGDEVICE_UNAVAILABLE"
        write_cert()
        logfile.close()
        return 1

    cert["device_limits"] = analog_device_limits()
    log(
        "  AnalogDevice: max_amp={max_amp_rad_per_us:.6f} rad/us, "
        "phase_jump_time={phase_jump_time_ns:.0f} ns, "
        "clock={clock_period_ns:.0f} ns, rydberg_level={rydberg_level}".format(
            **cert["device_limits"]
        )
    )

    # ---- G0a: convention probe on MockDevice, with the fallback coefficient --
    mock_error, gg_ok = pulser_hamiltonian_probe("mock")
    add_gate(
        "G0a_mock_hamiltonian_convention",
        {
            "device": "MockDevice",
            "hamiltonian_max_abs_error": mock_error,
            "initial_state_is_gg": gg_ok,
            "interaction_coefficient_used": float(C6),
            "threshold": 1e-9,
            "pass": bool(mock_error < 1e-9 and gg_ok),
            "scope": "drive sign/basis convention only; NOT the coefficient used "
            "downstream",
        },
        blocking=True,
    )
    log(f"\n[G0a] MockDevice convention: max|dH|={mock_error:.3e} init=|gg>:{gg_ok}")

    # ---- G0b/G0c: adopt AnalogDevice's coefficient, then RE-VERIFY -----------
    interaction_audit = stage3_configure_analog_interaction()
    add_gate(
        "G0b_analogdevice_interaction_loaded",
        {
            **interaction_audit,
            "pass": bool(
                np.isfinite(interaction_audit["AnalogDevice_C6"])
                and interaction_audit["AnalogDevice_C6"] > 0.0
            ),
        },
        blocking=True,
    )
    log(
        f"[G0b] AnalogDevice C6={interaction_audit['AnalogDevice_C6']:.8g} "
        f"(fallback {interaction_audit['fallback_C6']:.8g}, "
        f"U_int {interaction_audit['U_int_at_declared_separation']:.6f} rad/us)"
    )
    analog_error, analog_gg_ok = pulser_hamiltonian_probe("analog")
    add_gate(
        "G0c_analogdevice_hamiltonian_verified",
        {
            "device": "AnalogDevice",
            "hamiltonian_max_abs_error": analog_error,
            "initial_state_is_gg": analog_gg_ok,
            "interaction_coefficient_used": float(C6),
            "threshold": 1e-9,
            "pass": bool(analog_error < 1e-9 and analog_gg_ok),
            "scope": "post-override Hamiltonian, i.e. the one actually propagated",
        },
        blocking=True,
    )
    log(
        f"[G0c] AnalogDevice Hamiltonian (post-override): "
        f"max|dH|={analog_error:.3e} init=|gg>:{analog_gg_ok}"
    )
    foundation_pass = all(
        cert["gates"][n]["pass"]
        for n in (
            "G0a_mock_hamiltonian_convention",
            "G0b_analogdevice_interaction_loaded",
            "G0c_analogdevice_hamiltonian_verified",
        )
    )
    if not foundation_pass:
        cert["scientific_status"] = "DEVICE_HAMILTONIAN_FOUNDATION_NOT_RESOLVED"
        cert["next_step"] = (
            "Resolve the Pulser basis/sign convention or the AnalogDevice "
            "interaction metadata before constructing any endpoint-equivalent pair."
        )
        write_cert()
        log(f"\nGLOBAL VERDICT: {cert['scientific_status']}")
        logfile.close()
        return 2

    # G12 is evaluated after freezing, on the controls actually used, not on the
    # reference alone.

    # ---------------- SELECTION PHASE ---------------------------------------
    log("\n" + "=" * 92)
    log("SELECTION PHASE  eps + segment + native outcome score")
    log("=" * 92)
    search_rows = []
    eps_audits = []
    best_bundle = None
    attempted = 0
    completed = 0
    started = time.time()
    for eps in eps_grid:
        log("\n" + "#" * 92)
        log(f"EPSILON CANDIDATE {eps:g}  (selection data only)")
        log("#" * 92)
        try:
            paths = four_prepare_modulated_paths(args, eps, h_s, log)
        except Exception as exc:
            eps_audits.append(
                {
                    "eps": eps,
                    "status": "PATH_PREPARATION_REJECTED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            log(f"  eps={eps:g} rejected: {type(exc).__name__}: {exc}")
            continue
        path_gate = bool(
            paths["endpoint_pass"]
            and paths["duration_pass"]
            and paths["separation_pass"]
            and paths["path_nontrivial"]
        )
        eps_audits.append(
            {
                "eps": eps,
                "status": "PATH_ACCEPTED" if path_gate else "PATH_GATE_FAILED",
                "endpoint_pass": paths["endpoint_pass"],
                "duration_pass": paths["duration_pass"],
                "separation_pass": paths["separation_pass"],
                "path_nontrivial": paths["path_nontrivial"],
                "pair_separation": paths["pair_separation"],
                "CW_norm": paths["corrected"]["CW"]["audit"]["final_norm"],
                "CCW_norm": paths["corrected"]["CCW"]["audit"]["final_norm"],
                "CW_endpoint_infidelity": paths["corrected"]["CW"][
                    "endpoint_infidelity"
                ],
                "CCW_endpoint_infidelity": paths["corrected"]["CCW"][
                    "endpoint_infidelity"
                ],
            }
        )
        if not path_gate:
            log(f"  eps={eps:g}: path gate failed; excluded from selection")
            continue

        for segment_index in probe_segments:
            attempted += 1
            try:
                candidate = evaluate_candidate(
                    paths, segment_index, selection_deltas, args.covariance_ridge
                )
            except Exception as exc:
                log(
                    f"  eps={eps:g} segment={segment_index+1} rejected: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            completed += 1
            weights = candidate["weights"]
            row = {
                "eps": eps,
                "segment_1based": segment_index + 1,
                "pair_separation": paths["pair_separation"],
                "median_slope_per_mhz": candidate["median_slope_per_mhz"],
                "median_standardized_score": candidate["median_standardized_score"],
                "relative_slope_spread": candidate["relative_slope_spread"],
                "best_selection_shots_per_arm": candidate[
                    "best_selection_shots_per_arm"
                ],
                "minimum_outcome_probability": candidate[
                    "minimum_outcome_probability"
                ],
                "retained_rank": candidate["optimal"]["retained_rank"],
                "weight_concentration": candidate["optimal"]["weight_concentration"],
                "weight_rr": float(weights[0]),
                "weight_rg": float(weights[1]),
                "weight_gr": float(weights[2]),
                "weight_gg": float(weights[3]),
            }
            search_rows.append(row)
            log(
                f"  segment={segment_index+1} score={row['median_standardized_score']:.6e} "
                f"chi={row['median_slope_per_mhz']:+.6e}/MHz "
                f"shots={row['best_selection_shots_per_arm']:.3e} "
                f"spread={row['relative_slope_spread']:.2%} "
                f"p_min={row['minimum_outcome_probability']:.2e}"
            )
            if (
                best_bundle is None
                or row["median_standardized_score"]
                > best_bundle["row"]["median_standardized_score"]
            ):
                best_bundle = {
                    "row": row,
                    "paths": paths,
                    "candidate": candidate,
                    "weights": np.array(weights, copy=True),
                }

    add_gate(
        "G1_selection_search_completed",
        {
            "epsilon_candidates": len(eps_grid),
            "surviving_epsilon_candidates": len({r["eps"] for r in search_rows}),
            "attempted_candidates": attempted,
            "completed_candidates": completed,
            "pass": bool(best_bundle is not None),
        },
        blocking=True,
    )
    if best_bundle is None:
        cert["epsilon_audits"] = eps_audits
        cert["scientific_status"] = "NO_SELECTION_CANDIDATE_SURVIVED"
        cert["next_step"] = (
            "No (eps, segment) candidate passed the endpoint, separation and "
            "duration gates. Inspect epsilon_audits before touching the held-out "
            "grid."
        )
        write_cert()
        log(f"\nGLOBAL VERDICT: {cert['scientific_status']}")
        logfile.close()
        return 2

    baseline_rows = [r for r in search_rows if abs(r["eps"] - eps_grid[0]) <= 1e-12]
    baseline_score = (
        max(r["median_standardized_score"] for r in baseline_rows)
        if baseline_rows
        else None
    )
    selected_row = best_bundle["row"]
    selected_eps = float(selected_row["eps"])
    selected_segment = int(selected_row["segment_1based"]) - 1
    selected_weights = np.asarray(best_bundle["weights"], float)
    frozen_slope = float(selected_row["median_slope_per_mhz"])
    selected_paths = best_bundle["paths"]
    selected_optimal = best_bundle["candidate"]["optimal"]

    standardized_gain = (
        float(selected_row["median_standardized_score"] / baseline_score)
        if baseline_score
        else None
    )
    # Self-consistent target: shots = 25/(score^2 delta^2) <= N  =>
    # score >= 5/(delta sqrt(N)), evaluated at the largest held-out delta.
    max_test_delta = float(max(test_deltas))
    required_score = 5.0 / (max_test_delta * np.sqrt(args.shot_threshold))
    derived_target_gain = (
        float(required_score / baseline_score) if baseline_score else None
    )
    target_gain = (
        float(args.target_standardized_gain)
        if args.target_standardized_gain is not None
        else derived_target_gain
    )

    log("\n" + "=" * 92)
    log("FROZEN SELECTION")
    log("=" * 92)
    log(
        f"eps={selected_eps:g} | segment={selected_segment+1} | "
        f"chi={frozen_slope:+.7e} /MHz"
    )
    log("weights (rr,rg,gr,gg)=" + np.array2string(selected_weights, precision=7))
    log(
        f"covariance eigenvalues="
        + np.array2string(selected_optimal["covariance_eigenvalues"], precision=3)
        + f" retained_rank={selected_optimal['retained_rank']}"
    )
    log(
        f"standardized score={selected_row['median_standardized_score']:.6e}"
        + (
            f" | gain vs eps={eps_grid[0]:g}: {standardized_gain:.3f}x "
            f"(required {target_gain:.3f}x)"
            if standardized_gain is not None and target_gain is not None
            else " | baseline unavailable"
        )
    )
    log(f"selection elapsed={time.time()-started:.1f}s")

    cert["epsilon_audits"] = eps_audits
    cert["selection"] = {
        "rule": (
            "maximize the median selection-grid standardized slope after "
            "analytically optimizing one common diagonal outcome score"
        ),
        "selected_eps": selected_eps,
        "selected_segment_1based": selected_segment + 1,
        "outcome_weights_rr_rg_gr_gg": selected_weights.tolist(),
        "frozen_linear_coefficient_per_mhz": frozen_slope,
        "selected_row": selected_row,
        "baseline_eps": eps_grid[0],
        "baseline_standardized_score": baseline_score,
        "standardized_gain_vs_baseline": standardized_gain,
        "required_standardized_score_at_max_test_delta": required_score,
        "derived_target_gain": derived_target_gain,
    }
    add_gate(
        "G2_selected_modulated_endpoint",
        {
            "selected_eps": selected_eps,
            "CW_endpoint_infidelity": selected_paths["corrected"]["CW"][
                "endpoint_infidelity"
            ],
            "CCW_endpoint_infidelity": selected_paths["corrected"]["CCW"][
                "endpoint_infidelity"
            ],
            "endpoint_infidelity_threshold": args.endpoint_infidelity_tol,
            "durations_us": selected_paths["durations_us"],
            "pass": bool(
                selected_paths["endpoint_pass"] and selected_paths["duration_pass"]
            ),
        },
        blocking=True,
    )
    add_gate(
        "G2b_path_pair_separation",
        {
            "pair_separation": selected_paths["pair_separation"],
            "threshold": args.pair_separation_tol,
            "CW_vs_CCW_process_infidelity": selected_paths["pair_unitary_infidelity"],
            "CW_vs_CCW_population_TVD": selected_paths["pair_probability_tvd"],
            "pass": bool(selected_paths["separation_pass"]),
            "note": "process infidelity is ~0 by design (both equal U_mod(0)); it "
            "is a consistency check, not a separation measure",
        },
        blocking=True,
    )
    add_diagnostic(
        "D1_score_conditioning",
        {
            "covariance_eigenvalues": selected_optimal["covariance_eigenvalues"].tolist(),
            "retained_rank": selected_optimal["retained_rank"],
            "weight_concentration_max_over_l1": selected_optimal[
                "weight_concentration"
            ],
            "minimum_outcome_probability": selected_row["minimum_outcome_probability"],
            "covariance_ridge": args.covariance_ridge,
            "note": "There is no universal threshold on conditioning, so this "
            "REPORTS rather than tests. A weight concentration near 1 combined "
            "with a small minimum outcome probability means the shot budget rests "
            "on a rare outcome and assumes perfect readout. Set "
            "--min-outcome-probability to turn the readable half of this into an "
            "actual gate.",
        },
    )
    if args.min_outcome_probability > 0.0:
        add_gate(
            "G9_minimum_outcome_probability",
            {
                "minimum_outcome_probability": selected_row[
                    "minimum_outcome_probability"
                ],
                "threshold": args.min_outcome_probability,
                "pass": bool(
                    selected_row["minimum_outcome_probability"]
                    >= args.min_outcome_probability
                ),
            },
            blocking=False,
        )

    with open(
        os.path.join(args.outdir, "four_setting_v1_2_selection.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(search_rows[0].keys()))
        writer.writeheader()
        writer.writerows(search_rows)

    z_cw = selected_paths["corrected"]["CW"]["z"]
    z_ccw = selected_paths["corrected"]["CCW"]["z"]

    # ---------------- G12: executed gap, on the controls actually used -------
    # Auditing only the reference would leave the two frozen controls unchecked:
    # they carry nonzero phase offsets z[:,2], so their phase table, and hence
    # whether AnalogDevice inserts an idle at every junction, differs from the
    # reference. The detuning probe leaves phases untouched, so auditing the
    # unprobed controls covers all four arms.
    gap_audits = {
        "reference": executed_gap_audit(selected_paths["reference_z"]),
        "CW": executed_gap_audit(z_cw),
        "CCW": executed_gap_audit(z_ccw),
    }
    add_gate(
        "G12_executed_gap_matches_seed_model",
        {
            **gap_audits,
            "absolute_difference_threshold_us": 1e-6,
            "pass": bool(
                all(
                    audit["absolute_difference_us"] <= 1e-6
                    and audit["all_consecutive_phases_distinct"]
                    for audit in gap_audits.values()
                )
            ),
            "note": "Informational seed-model audit; the sampled-waveform "
            "propagation remains authoritative, so a failure degrades only the "
            "quality of the M2 seed.",
        },
        blocking=False,
    )
    for tag, audit in gap_audits.items():
        log(
            f"[G12] {tag:9s} executed idle/junction="
            f"{audit['inferred_idle_per_junction_us']*1000:.3f} ns vs seed model "
            f"{MODEL['gap_tau']*1000:.0f} ns | min phase separation="
            f"{audit['minimum_consecutive_phase_separation_rad']:.4f} rad | "
            f"distinct={audit['all_consecutive_phases_distinct']}"
        )

    # ---------------- numerical noise floor (selection deltas only) ---------
    floor_delta = float(min(selection_deltas))
    split = max(1, int(args.numerical_floor_split))
    stat_1 = amplified_contrast_statistics(
        four_evaluate_settings(z_cw, z_ccw, selected_segment, floor_delta, 1)
    )
    stat_n = amplified_contrast_statistics(
        four_evaluate_settings(z_cw, z_ccw, selected_segment, floor_delta, split)
    )
    s_1 = float(selected_weights @ stat_1["contrast"])
    s_n = float(selected_weights @ stat_n["contrast"])
    floor_rel = abs(s_n - s_1) / max(abs(s_1), 1e-300)
    add_gate(
        "G10_numerical_floor",
        {
            "delta_mhz": floor_delta,
            "block_split_factor": split,
            "S_single": s_1,
            "S_split": s_n,
            "absolute_difference": abs(s_n - s_1),
            "relative_difference": floor_rel,
            "pass": bool(floor_rel <= 1e-3),
            "note": "same physics, different floating-point path; bounds the "
            "propagation error of an O(1e-8) difference of O(1) populations",
        },
        blocking=False,
    )
    log(
        f"\n[G10] numerical floor at delta={floor_delta:g} MHz: "
        f"S={s_1:+.6e} vs split-{split} {s_n:+.6e} "
        f"(relative {floor_rel:.2e})"
    )

    # ---------------- h_s sensitivity (selection deltas only) ----------------
    if args.hs_audit:
        log("\n" + "=" * 92)
        log("G8  h_s SENSITIVITY OF THE FROZEN DESIGN  (selection deltas only)")
        log("=" * 92)
        try:
            paths_half = four_prepare_modulated_paths(args, selected_eps, h_s / 2, log)
            candidate_half = evaluate_candidate(
                paths_half,
                selected_segment,
                selection_deltas,
                args.covariance_ridge,
                weights=selected_weights,
            )
            chi_half = candidate_half["median_slope_per_mhz"]
            drift = abs(chi_half - frozen_slope) / max(abs(frozen_slope), 1e-300)
            add_gate(
                "G8_hs_sensitivity",
                {
                    "h_s": float(h_s),
                    "h_s_halved": float(h_s / 2),
                    "chi_at_h_s": frozen_slope,
                    "chi_at_h_s_over_2": chi_half,
                    "relative_drift": float(drift),
                    "pair_separation_at_h_s": selected_paths["pair_separation"],
                    "pair_separation_at_h_s_over_2": paths_half["pair_separation"],
                    "pass": bool(drift <= 0.10),
                    "note": "informational. The transport landing point moves with "
                    "h_s, so chi is a property of THIS landing point. A large "
                    "drift does not invalidate the witness; it bounds how many "
                    "digits of chi may be quoted.",
                },
                blocking=False,
            )
            log(
                f"  chi(h_s={h_s:g})={frozen_slope:+.6e}  "
                f"chi(h_s/2)={chi_half:+.6e}  relative drift={drift:.2%}"
            )
            digits = (
                max(1, int(np.floor(-np.log10(max(drift, 1e-16)))))
                if drift > 0
                else 6
            )
            log(f"  -> quote chi to at most {digits} significant figure(s)")
        except Exception as exc:
            add_gate(
                "G8_hs_sensitivity",
                {
                    "status": "NOT_RUN",
                    "error": f"{type(exc).__name__}: {exc}",
                    "pass": False,
                },
                blocking=False,
            )
            log(f"  h_s audit failed: {type(exc).__name__}: {exc}")
    else:
        add_gate(
            "G8_hs_sensitivity",
            {
                "status": "NOT_RUN",
                "pass": False,
                "note": "disabled with --no-hs-audit; chi is then an unaudited "
                "function of the transport step",
            },
            blocking=False,
        )

    # ---------------- HELD-OUT TEST -----------------------------------------
    log("\n" + "=" * 92)
    log("HELD-OUT TEST  frozen eps, path pair, segment, and outcome score")
    log("=" * 92)
    log(
        f"  {'delta MHz':>10s} {'S exact':>13s} {'prediction':>13s} "
        f"{'rel.err':>9s} {'shots/arm':>13s}  linear?"
    )
    heldout_rows = []
    heldout_device_pass = True
    for delta_mhz in test_deltas:
        try:
            records = four_evaluate_settings(z_cw, z_ccw, selected_segment, delta_mhz)
        except Exception as exc:
            heldout_device_pass = False
            log(f"  {delta_mhz:10.6f} REJECTED: {type(exc).__name__}: {exc}")
            continue
        statistics = amplified_contrast_statistics(records)
        witness = amplified_weighted_witness(statistics, selected_weights)
        prediction = delta_mhz * frozen_slope
        residual_value = abs(witness["signal"] - prediction)
        relative_error = residual_value / max(abs(witness["signal"]), 1e-300)
        in_linear = bool(relative_error <= args.heldout_relative_error_tol)
        duration_spread = max(
            r["duration_us"] for r in records.values()
        ) - min(r["duration_us"] for r in records.values())
        row = {
            "delta_mhz": delta_mhz,
            "signal": witness["signal"],
            "prediction": prediction,
            "absolute_residual": residual_value,
            "relative_error": relative_error,
            "within_linearity_tolerance": in_linear,
            "shots_per_arm_5sigma": witness["shots_per_arm_5sigma"],
            "variance_numerator_per_arm": witness["variance_numerator_per_arm"],
            "minimum_outcome_probability": statistics["minimum_outcome_probability"],
            "duration_spread_us": duration_spread,
        }
        heldout_rows.append(row)
        log(
            f"  {delta_mhz:10.6f} {witness['signal']:+13.6e} {prediction:+13.6e} "
            f"{relative_error:9.3%} {witness['shots_per_arm_5sigma']:13.3e}  "
            f"{'yes' if in_linear else 'NO'}"
        )

    if heldout_rows:
        delta_array = np.asarray([r["delta_mhz"] for r in heldout_rows], float)
        signal_array = np.asarray([r["signal"] for r in heldout_rows], float)
        signal_exponent = loglog_slope(delta_array, np.abs(signal_array))
        maximum_relative_error = float(max(r["relative_error"] for r in heldout_rows))
        maximum_signal = float(max(abs(r["signal"]) for r in heldout_rows))
        duration_pass = all(r["duration_spread_us"] <= 1e-9 for r in heldout_rows)
        linear_rows = [r for r in heldout_rows if r["within_linearity_tolerance"]]
        # Feasibility on the LINEAR SUBSET. shots ~ delta^-2, so an unrestricted
        # min always lands on the largest delta, which is also the worst point
        # for linearity: v1.1 asserted the two properties at different deltas.
        best_shots_linear = (
            float(min(r["shots_per_arm_5sigma"] for r in linear_rows))
            if linear_rows
            else float("inf")
        )
        best_shots_unrestricted = float(
            min(r["shots_per_arm_5sigma"] for r in heldout_rows)
        )
        cubic = cubic_fit(delta_array, signal_array)
    else:
        signal_exponent = float("nan")
        maximum_relative_error = float("inf")
        maximum_signal = 0.0
        duration_pass = False
        linear_rows = []
        best_shots_linear = float("inf")
        best_shots_unrestricted = float("inf")
        cubic = cubic_fit([], [])

    baseline_probability_difference = (
        selected_paths["corrected"]["CCW"]["record"]["probabilities"]
        - selected_paths["corrected"]["CW"]["record"]["probabilities"]
    )
    baseline_weighted_difference = float(
        abs(selected_weights @ baseline_probability_difference)
    )

    add_gate(
        "G3_zero_probe_weighted_output_match",
        {
            "absolute_weighted_difference": baseline_weighted_difference,
            "threshold": 1e-9,
            "pass": bool(baseline_weighted_difference <= 1e-9),
            "note": "independent re-lift check. It is NOT what makes S baseline "
            "free: S is exactly odd in delta, so f(0) cancels identically.",
        },
        blocking=True,
    )
    add_gate(
        "G4_heldout_program_acceptance_and_duration",
        {
            "completed_points": len(heldout_rows),
            "declared_points": len(test_deltas),
            "equal_duration": duration_pass,
            "pass": bool(
                heldout_device_pass
                and len(heldout_rows) == len(test_deltas)
                and duration_pass
            ),
            "note": "the four arms differ only in detuning, so equal duration is "
            "near-tautological; it is a compilation check, not evidence",
        },
        blocking=True,
    )
    add_gate(
        "G5_heldout_linear_path_susceptibility",
        {
            "signal_exponent": signal_exponent,
            "accepted_exponent_interval": [0.90, 1.10],
            "maximum_relative_error": maximum_relative_error,
            "relative_error_threshold": args.heldout_relative_error_tol,
            "points_within_linearity_tolerance": len(linear_rows),
            "pass": bool(
                0.90 <= signal_exponent <= 1.10
                and len(linear_rows) > 0
                and maximum_relative_error <= args.heldout_relative_error_tol
            ),
        },
        blocking=True,
    )
    # A real threshold, not an automatic PASS. "S is odd, therefore the leading
    # correction is cubic" is a statement about the FORM of the correction; it
    # says nothing about its SIZE, which is exactly what has to be bounded for
    # the frozen linear coefficient to be extrapolated from the selection grid
    # to the held-out grid. The tested quantity is
    #   |c delta_max^3| / |a delta_max| = |c/a| delta_max^2 .
    cubic_contamination = (
        float(cubic.get("max_abs_relative_cubic_on_grid", float("inf")))
        if heldout_rows
        else float("inf")
    )
    add_gate(
        "G11_heldout_cubic_contamination",
        {
            **cubic,
            "max_test_delta_mhz": max_test_delta,
            "cubic_over_linear_at_max_delta": cubic_contamination,
            "threshold": args.cubic_contamination_tol,
            "pass": bool(
                np.isfinite(cubic_contamination)
                and cubic_contamination <= args.cubic_contamination_tol
            ),
            "note": "S is exactly odd, so the leading correction is O(delta^3); "
            "this bounds its magnitude at the largest held-out delta.",
        },
        blocking=False,
    )
    add_gate(
        "G6_standardized_gain",
        {
            "gain_vs_baseline": standardized_gain,
            "target": target_gain,
            "target_source": (
                "explicit --target-standardized-gain"
                if args.target_standardized_gain is not None
                else "derived from --shot-threshold and max held-out delta"
            ),
            "pass": bool(
                standardized_gain is not None
                and target_gain is not None
                and standardized_gain >= target_gain
            ),
        },
        blocking=False,
    )
    add_gate(
        SHOT_GATE,
        {
            "best_shots_per_arm_5sigma_on_linear_subset": best_shots_linear,
            "best_shots_per_arm_5sigma_unrestricted": best_shots_unrestricted,
            "linear_subset_deltas_mhz": [r["delta_mhz"] for r in linear_rows],
            "threshold": args.shot_threshold,
            "four_independent_arms_per_delta": True,
            "pass": bool(best_shots_linear <= args.shot_threshold),
        },
        blocking=True,
    )

    cert["heldout_test"] = {
        "completed_points": len(heldout_rows),
        "declared_points": len(test_deltas),
        "signal_exponent": signal_exponent,
        "maximum_relative_error": maximum_relative_error,
        "maximum_absolute_signal": maximum_signal,
        "best_shots_per_arm_5sigma_on_linear_subset": best_shots_linear,
        "best_shots_per_arm_5sigma_unrestricted": best_shots_unrestricted,
        "cubic_structure": cubic,
    }

    if heldout_rows:
        with open(
            os.path.join(args.outdir, "four_setting_v1_2_heldout.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=list(heldout_rows[0].keys()))
            writer.writeheader()
            writer.writerows(heldout_rows)

    with open(
        os.path.join(args.outdir, "four_setting_v1_2_controls.json"),
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            {
                "reference": selected_paths["reference_z"].tolist(),
                "CW_modulated_relift": z_cw.tolist(),
                "CCW_modulated_relift": z_ccw.tolist(),
                "selected_eps": selected_eps,
                "effective_h_s": float(h_s),
                "selected_probe_segment_1based": selected_segment + 1,
                "outcome_order": OUTCOME_ORDER,
                "frozen_outcome_weights": selected_weights.tolist(),
                "frozen_linear_coefficient_per_mhz": frozen_slope,
                "selection_deltas_mhz": selection_deltas,
                "heldout_test_deltas_mhz": test_deltas,
            },
            fh,
            indent=2,
        )

    cert["path_relift_audit"] = {
        d: {
            "seed": selected_paths["seeds"][d]["audit"],
            "modulated_relift": selected_paths["corrected"][d]["audit"],
        }
        for d in ("CW", "CCW")
    }

    # BLOCKING_GATES deliberately excludes SHOT_GATE: structural support and
    # practical feasibility are separate claims and are reported separately.
    missing = [n for n in BLOCKING_GATES if n not in cert["gates"]]
    if missing:
        raise RuntimeError(f"blocking gates never evaluated: {missing}")
    structural_ok = all(cert["gates"][n]["pass"] for n in BLOCKING_GATES)
    shot_ok = cert["gates"][SHOT_GATE]["pass"]
    if structural_ok and shot_ok:
        status = "AMPLIFIED_FOUR_SETTING_WITNESS_SHOT_FEASIBLE"
        next_step = (
            "Freeze one held-out delta inside the linear subset and reproduce its "
            "four programs with an independent Pulser propagation engine before "
            "any cloud submission."
        )
    elif structural_ok:
        status = "AMPLIFIED_FOUR_SETTING_WITNESS_SUPPORTED_SIGNAL_TOO_SMALL"
        next_step = (
            "The predeclared amplification search stayed below the shot threshold "
            "on the linear subset. Do not submit; optimize inside the modulated "
            "endpoint fiber rather than expanding eps further."
        )
    else:
        status = "AMPLIFIED_FOUR_SETTING_WITNESS_NOT_CLOSED"
        next_step = (
            "Inspect the first failed blocking gate; no path-conditioned "
            "derivative claim follows from the held-out split."
        )
    cert["scientific_status"] = status
    cert["next_step"] = next_step
    cert["claim_boundary"] = (
        "Two-atom local Pulser AnalogDevice modulation model, exact sampled "
        "coherent propagation on the 1 ns grid, native |gg> preparation and "
        "complete native population outcomes, one segment-local coherent "
        "detuning probe, an eps/segment/score selection grid, and a strictly "
        "disjoint held-out probe grid. S is exactly odd in delta, so a nonzero S "
        "alone witnesses nothing about the endpoint fiber; the claim is the "
        "conjunction of the endpoint gates, the separation gate and the held-out "
        "linearity gate. chi is a property of the transport landing point at the "
        "recorded h_s, not of the loop; see G8. The optimal outcome score is "
        "classical post-processing of native counts optimized against NOISELESS "
        "simulated probabilities, and its shot budget assumes perfect readout; "
        "see G9. No calibrated FRESNEL waveform, cloud-emulator evidence, or QPU "
        "evidence."
    )
    write_cert()

    log("\n" + "=" * 92)
    log("GLOBAL VERDICT")
    log("=" * 92)
    log("  blocking gates:")
    for name, gate in cert["gates"].items():
        if gate.get("blocking"):
            log(f"    {'PASS' if gate['pass'] else 'FAIL':5s} {name}")
    log("  informational gates (real thresholds; do not affect scientific_status):")
    for name, gate in cert["gates"].items():
        if not gate.get("blocking"):
            log(f"    {'PASS' if gate['pass'] else 'FAIL':5s} {name}")
    log("  diagnostics (reported, no pass/fail claimed):")
    for name in cert["diagnostics"]:
        log(f"    REPORT {name}")
    log(f"\nscientific_status={status}")
    log(
        f"selected eps={selected_eps:g} | segment={selected_segment+1} | "
        f"h_s={h_s:g} | gain="
        f"{'N/A' if standardized_gain is None else f'{standardized_gain:.3f}x'}"
    )
    log(f"best held-out shots/arm on linear subset={best_shots_linear:.3e}")
    log(f"next={next_step}")
    log(
        "written: four_setting_v1_2_certificate.json, "
        "four_setting_v1_2_controls.json, four_setting_v1_2_selection.csv, "
        "four_setting_v1_2_heldout.csv, four_setting_v1_2_run.log"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
        eps_values = sorted({r["eps"] for r in search_rows})
        best_by_eps = [
            max(
                r["median_standardized_score"]
                for r in search_rows
                if r["eps"] == value
            )
            for value in eps_values
        ]
        axes[0].plot(eps_values, best_by_eps, "o-")
        axes[0].set_xlabel(r"M2 loop scale $\epsilon$")
        axes[0].set_ylabel("best selection standardized slope")
        axes[0].set_title("selection-only amplification")
        axes[0].grid(alpha=0.3)
        if heldout_rows:
            deltas = np.asarray([r["delta_mhz"] for r in heldout_rows], float)
            signals = np.asarray([r["signal"] for r in heldout_rows], float)
            mask = np.asarray(
                [r["within_linearity_tolerance"] for r in heldout_rows], bool
            )
            axes[1].plot(deltas, signals, "o-", label="held-out exact")
            if mask.any():
                axes[1].plot(
                    deltas[mask],
                    signals[mask],
                    "o",
                    ms=10,
                    mfc="none",
                    label="within linearity tol",
                )
            axes[1].plot(
                deltas, deltas * frozen_slope, "--", label="frozen linear prediction"
            )
            axes[1].legend(fontsize=8)
        axes[1].set_xlabel(r"$|\delta|$ [MHz]")
        axes[1].set_ylabel(r"$S_w(\delta)$")
        axes[1].set_title("frozen four-setting witness")
        axes[1].grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(
            os.path.join(args.outdir, "four_setting_v1_2_amplification.png"), dpi=170
        )
        log("written: four_setting_v1_2_amplification.png")
    except Exception as exc:
        log(f"(figure skipped: {type(exc).__name__}: {exc})")

    logfile.close()
    return 0


def run_amplified(**kwargs):
    """Notebook entry point.

    With allow_abbrev=False and strict unknown-argument handling, a typo or a
    flag carried over from an older entry point now raises instead of silently
    changing the experiment.
    """
    argv = []
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            argv.append(flag if value else "--no-" + key.replace("_", "-"))
        else:
            argv.extend([flag, str(value)])
    return amplified_main(argv)


if __name__ == "__main__":
    exit_code = amplified_main()
    if not _in_notebook():
        raise SystemExit(exit_code)


