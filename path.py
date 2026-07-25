#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EP-OBS-4SET-v1.1 -- amplified four-setting path-susceptibility search.

Version 1.1 adds two predeclared amplification mechanisms without touching the
held-out test grid:

  1. an epsilon sweep that enlarges the M2 endpoint-fiber loop and selects one
     device-accepted, modulation-aware, full-unitary-equivalent path pair using
     selection data only;
  2. the statistically optimal diagonal score over the complete native outcome
     alphabet (rr, rg, gr, gg), rather than choosing among five hand-written
     observables.

The epsilon, probe segment, outcome weights, and linear coefficient are frozen
before the disjoint held-out delta grid is evaluated.

This standalone file contains the complete Stage-3 modulation-aware endpoint
re-lift and adds a separate four-setting experiment:

    CW(+delta), CW(-delta), CCW(+delta), CCW(-delta).

For every candidate drive segment it evaluates a native outcome score

    S(delta) = 1/2 * [
        W_CCW(+delta) - W_CCW(-delta)
      - W_CW (+delta) + W_CW (-delta)
    ].

At delta=0 the two paths implement the same complete modulated unitary.  A
nonzero linear coefficient of S(delta) therefore measures a path-conditioned
coherent susceptibility, not a baseline endpoint difference. Selection deltas
choose one segment and one common score over (rr,rg,gr,gg); disjoint held-out
deltas then test the frozen choice and determine the actual shot budget.

This is deliberately NOT the Stage-3 uniform-dephasing experiment.  Its slope
must be computed afresh and the old 10^10--10^11 shot estimate is never reused.

PURPOSE
-------
The manuscript "Ideal Executed Paths Predict Weak-Noise Differences between
Full-Unitary-Equivalent Rydberg Controls" establishes a *model-level* result:
two controls z and z0 with the same complete ideal unitary have different
interaction-picture dissipative response operators K_z, and the Frobenius channel
distance D_E(gamma) = ||E_z - E_z0||_F / 16 is predicted parameter-free by
gamma * ||K_z - K_0||_F / 16.

D_E is not directly available from one native preparation/readout setting.
This script moves to an *observable-level* quantity that a neutral-atom
experiment can in principle estimate:

    DeltaP(gamma) = Tr[ M ( E_z(T;gamma) - E_z0(T;gamma) )(rho) ]
                  = gamma * chi_{M,rho}[DeltaK] + O(gamma^2),

    chi_{M,rho}[DeltaK] = Tr[ M * unvec( (G_z - G_0) vec(rho) ) ],
    G_z = d/dgamma E_z(T;gamma) |_{gamma=0}  (= U_z(T) K_z).

CONSTRAINTS TAKEN SERIOUSLY (this is the whole point of the script)
-------------------------------------------------------------------
  * rho is NOT optimized over all states. The native initialization used here
    is |gg>, so rho = |gg><gg| is hard-wired. Preparation pulses could enlarge
    this family, but they are outside this audit.
  * M is NOT an arbitrary Hermitian operator. Readout projects each atom onto
    {|g>, |r>}. Admissible M are therefore diagonal in the computational basis.
  * Stage 3 deliberately appends NO basis-rotation pulse. It first asks whether
    the modulated-path signal is visible in native computational-basis
    observables. This avoids introducing another path-dependent phase jump.
  * A shot budget is computed from the exact binomial/multinomial variance.
    If the required shot count is absurd, the script says so. It does not hide it.

STAGE-3 TEST CHAIN
------------------
  (1) reproduce the gap-aware M2 seed paths;
  (2) read AnalogDevice's interaction coefficient at runtime;
  (3) locally re-lift both paths on modulation=True waveforms;
  (4) independently resample the final controls and recertify the endpoint;
  (5) compute dE/dgamma directly on the modulated 1 ns waveform;
  (6) test native observables in a declared weak-noise window.

HAMILTONIAN CONVENTION
----------------------
Not assumed. Extracted from Pulser at runtime and asserted (Gate H0). Pulser uses
the ordered eigenbasis (|r>, |g>) and

    H = (Omega/2) ( cos(phi) X - sin(phi) Y ) - Delta * N + (C6/a^6) n1 n2 ,

i.e. the manuscript's +sin(phi) Y corresponds to phi -> -phi here.

That relabeling is NOT a symmetry of the numerical phase table. For one
Hermitian segment,

    H(-phi) = H(phi)^* = H(phi)^T,
    U_j(-phi) = U_j(phi)^T.

For a multi-segment chronological product, however, transposition reverses the
matrix-product order. Applying -phi to every segment without reversing the
schedule therefore does not generally give U(phi)^T or U(phi)^*. Gate H1 checks
the Hamiltonian and single-segment transpose identities and records the expected
multi-segment gap. The fiber existence, rank, and first-order law are convention
independent; numerical values of ||K_z-K_0||, chi and DeltaP are not. Select the
convention with --phase-sign; it is applied globally through MODEL["phase_sign"].

STAGE-3 PURPOSE AND USAGE
-------------------------
Stage 2 established that the 340 ns phase-jump idle evolution can be modelled
exactly, but Pulser's 8 MHz modulation breaks the endpoint equivalence. Stage 3
therefore treats the MODULATED AnalogDevice sampled waveform as the endpoint
map itself:

    1. reproduce the two gap-aware M2 seed paths;
    2. project each seed locally onto the modulated full-unitary endpoint fiber;
    3. verify the corrected paths remain distinct and device accepted;
    4. compute the dephasing response directly on the modulated 1 ns waveform;
    5. select only native computational-basis observables (no readout pulse);
    6. separate a declared weak-noise window from the gamma=0.03 stress point.

Run:

    python ep_obs_stage3_modulated_relift_v1_1.py

Defaults are frozen from Stage 2: omega_scale=0.80, gap_tau=0.340 us, quick M2
seed transport, local Pulser only. The script never requests cloud credentials
and never submits a job. A successful local re-lift is not calibrated FRESNEL
or QPU evidence.

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
from scipy.linalg import expm, expm_frechet

# Avoid notebook noise when the default matplotlib config directory is
# read-only. This is set before package_versions() imports matplotlib.
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "ep_obs_matplotlib")
)

# ----------------------------------------------------------------------------
# 0. PROVENANCE
# ----------------------------------------------------------------------------

VERSION = "EP-OBS-4SET-v1.1"


_SOURCE_PATH_OVERRIDE = None


def self_sha256() -> str:
    """SHA-256 of the executed source.

    A notebook cell does not expose __file__, so the digest is unavailable unless
    the caller points at the archived file explicitly (--source-path, or
    run(source_path=...)). Reporting 'unavailable' is deliberate: a digest of a
    file that may differ from what was actually executed is worse than none.
    """
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
N_LOC = [np.kron(NN, I2), np.kron(I2, NN)]

DIM = 4
DIM2 = DIM * DIM

# computational basis order (Pulser): |rr>, |rg>, |gr>, |gg>
IDX_RR, IDX_RG, IDX_GR, IDX_GG = 0, 1, 2, 3
RHO0 = np.zeros((DIM, DIM), dtype=complex)
RHO0[IDX_GG, IDX_GG] = 1.0  # native initialization used in this audit

ATOM_SEP = 6.0  # um
C6 = 5420158.53  # fallback only; Stage 3 replaces this from AnalogDevice
U_INT = C6 / ATOM_SEP**6

TAU = 0.120  # us, per segment
NSEG = 6
T_TOTAL = NSEG * TAU

# Schedule model. GAP_TAU > 0 reproduces what a real device actually executes:
# AnalogDevice inserts 2 x phase_jump_time (2 x 170 ns) of idle between pulses
# whose phase differs, during which Omega = Delta = 0 and only the C6 interaction
# acts. The M2 lift is performed on THIS schedule, so endpoint equivalence holds
# for the executed evolution rather than for an idealized gap-free description.
MODEL = {"gap_tau": 0.0, "omega_scale": 1.0, "phase_sign": 1.0}

# reference control z0  (Table I of the manuscript)
OMEGA0 = 2 * np.pi * np.array([2.0, 1.7, 2.3, 1.5, 2.1, 1.8])
DELTA0 = 2 * np.pi * np.array([-2.3, -1.2, 0.4, 1.4, 2.0, 0.8])
PHI0 = np.array([0.0, 0.4, 1.1, 2.0, 2.7, -2.4])


def controls_from_z(z: np.ndarray, phase_sign: "float | None" = None):
    """z in R^18 -> (Omega_j, Delta_j, phi_j) arrays, manuscript parametrization.

    phase_sign=None means "use the globally selected convention", so --phase-sign
    reaches every downstream computation through this single point.
    """
    if phase_sign is None:
        phase_sign = MODEL["phase_sign"]
    z = np.asarray(z, dtype=float).reshape(NSEG, 3)
    omega = MODEL["omega_scale"] * OMEGA0 * (1.0 + z[:, 0])
    delta = DELTA0 + 2 * np.pi * z[:, 1]
    phi = phase_sign * (PHI0 + z[:, 2])
    return omega, delta, phi


def hamiltonian(omega: float, delta: float, phi: float) -> np.ndarray:
    """Pulser-convention two-atom Rydberg Hamiltonian (verified by Gate H0)."""
    return (
        0.5 * omega * (np.cos(phi) * X_OP - np.sin(phi) * Y_OP)
        - delta * N_OP
        + U_INT * NN_OP
    )


def prop_from_H(H: np.ndarray, tau: float) -> np.ndarray:
    """exp(-i H tau) via Hermitian eigendecomposition (fast + unitary to 1e-16)."""
    w, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * w * tau)) @ V.conj().T


@dataclass
class Segment:
    omega: float
    delta: float
    phi: float
    tau: float


def segments_of(z, phase_sign=None, readout: "Segment | None" = None):
    """Full executed schedule: six drive segments, optional idle gaps, optional
    readout-rotation segment appended identically for every control."""
    omega, delta, phi = controls_from_z(z, phase_sign)
    gap = MODEL["gap_tau"]
    segs = []
    for j in range(NSEG):
        if gap > 0 and j > 0:
            segs.append(Segment(0.0, 0.0, 0.0, gap))
        segs.append(Segment(omega[j], delta[j], phi[j], TAU))
    if readout is not None:
        if gap > 0:
            segs.append(Segment(0.0, 0.0, 0.0, gap))
        segs.append(readout)
    return segs


def unitary_of_z(z: np.ndarray, phase_sign=None) -> np.ndarray:
    """Ideal endpoint of the SCHEDULE (gaps included if MODEL['gap_tau'] > 0)."""
    U = np.eye(DIM, dtype=complex)
    for sg in segments_of(z, phase_sign):
        U = prop_from_H(hamiltonian(sg.omega, sg.delta, sg.phi), sg.tau) @ U
    return U


# --- Liouville space (column stacking: vec(A X B) = kron(B.T, A) vec(X)) -----

EYE_D = np.eye(DIM, dtype=complex)


def vec(rho: np.ndarray) -> np.ndarray:
    return rho.reshape(-1, order="F")


def unvec(v: np.ndarray) -> np.ndarray:
    return v.reshape(DIM, DIM, order="F")


def sup_hamiltonian(H: np.ndarray) -> np.ndarray:
    return -1j * (np.kron(EYE_D, H) - np.kron(H.T, EYE_D))


def sup_unitary(U: np.ndarray) -> np.ndarray:
    return np.kron(U.conj(), U)


def build_dissipator() -> np.ndarray:
    """D[rho] = sum_k ( n_k rho n_k - 1/2 {n_k, rho} ),  occupation dephasing.

    Matches Pulser eff_noise with operators n_k and rate gamma exactly
    (collapse ops sqrt(gamma) n_k applied locally on each atom).
    """
    D = np.zeros((DIM2, DIM2), dtype=complex)
    for n in N_LOC:
        D += np.kron(n.T, n) - 0.5 * np.kron(EYE_D, n) - 0.5 * np.kron(n.T, EYE_D)
    return D


D_SUP = build_dissipator()


# ----------------------------------------------------------------------------
# 2. ENDPOINT RESIDUAL, JACOBIANS, M2 LIFT
# ----------------------------------------------------------------------------


def u_target(s: np.ndarray, U0: np.ndarray) -> np.ndarray:
    gen = -0.25j * (s[0] * X_OP + s[1] * Y_OP)
    return expm(gen) @ U0


def residual(z: np.ndarray, s: np.ndarray, U0: np.ndarray, phase_sign=None) -> np.ndarray:
    """Phase-aligned real residual r(z,s) in R^32."""
    Uz = unitary_of_z(z, phase_sign)
    Ut = u_target(np.asarray(s, float), U0)
    c = np.trace(Ut.conj().T @ Uz)
    theta = np.angle(c)
    R = np.exp(-1j * theta) * Uz - Ut
    return np.concatenate([R.real.ravel(), R.imag.ravel()])


def endpoint_infidelity(z: np.ndarray, U0: np.ndarray, phase_sign=None) -> float:
    Uz = unitary_of_z(z, phase_sign)
    return float(abs(1.0 - abs(np.trace(U0.conj().T @ Uz)) ** 2 / DIM**2))


def jacobians(z, s, U0, h=1e-6, phase_sign=None):
    """Central-difference Q = dr/dz (32x18), B = dr/ds (32x2)."""
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
    """Minimum-norm least-squares solution of Q dz = rhs, via reduced SVD."""
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
    """Gauss-Newton correction restricted to range(Q^T) (min-norm LS step)."""
    z = np.asarray(z, float).copy()
    for _ in range(iters):
        r = residual(z, s, U0, phase_sign)
        if np.linalg.norm(r) < 1e-13:
            break
        Q, _ = jacobians(z, s, U0, phase_sign=phase_sign)
        dz, _, _, _, _ = minnorm_solve(Q, -r, rcond)
        z = z + dz
    return z, float(np.linalg.norm(residual(z, s, U0, phase_sign)))


def m2_transport(U0, vertices, h_s, rcond=1e-6, phase_sign=None, verbose=False):
    """Lift a closed task-space loop; return final z and a numerical audit."""
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
    for a, b in zip(vertices[:-1], vertices[1:]):
        seg = np.array(b, float) - np.array(a, float)
        L = float(np.linalg.norm(seg))
        nst = max(1, int(round(L / h_s)))
        audit.setdefault("effective_h_s", []).append(float(L / nst))
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
    s_final = s
    z, cres = correct_endpoint(z, s_final, U0, rcond, iters=6, phase_sign=phase_sign)
    eff = audit.pop("effective_h_s", [h_s])
    audit["effective_h_s"] = float(np.mean(eff))
    audit["effective_h_s_spread"] = float(np.max(eff) - np.min(eff))
    audit["nominal_h_s"] = float(h_s)
    audit["reach_res"] = float(np.linalg.norm(s_final - np.array(vertices[-1], float)))
    audit["final_res"] = cres
    audit["rank_set"] = sorted(set(audit["ranks"]))
    audit["min_smin_keep"] = float(np.min(audit["smin_keep"]))
    audit["max_smax_drop"] = float(np.max(audit["smax_drop"]))
    audit["max_lift_res"] = float(np.max(audit["lift_res"]))
    audit["max_corr_res"] = float(np.max(audit["corr_res"]))
    for k in ("ranks", "smin_keep", "smax_drop", "lift_res", "corr_res"):
        audit.pop(k)
    return z, audit


# ----------------------------------------------------------------------------
# 3. CHANNELS, RESPONSE OPERATORS
# ----------------------------------------------------------------------------


def channel(segs, gamma: float) -> np.ndarray:
    """Exact Lindblad channel as a 16x16 Liouville matrix."""
    E = np.eye(DIM2, dtype=complex)
    for sg in segs:
        L0 = sup_hamiltonian(hamiltonian(sg.omega, sg.delta, sg.phi))
        E = expm((L0 + gamma * D_SUP) * sg.tau) @ E
    return E


def channel_derivative(segs) -> np.ndarray:
    """G = d/dgamma E(gamma) at gamma = 0, via the Frechet derivative of expm."""
    props = []
    fre = []
    for sg in segs:
        A = sup_hamiltonian(hamiltonian(sg.omega, sg.delta, sg.phi)) * sg.tau
        E_dir = D_SUP * sg.tau
        P, Lf = expm_frechet(A, E_dir, compute_expm=True)
        props.append(P)
        fre.append(Lf)
    n = len(segs)
    G = np.zeros((DIM2, DIM2), dtype=complex)
    for j in range(n):
        left = np.eye(DIM2, dtype=complex)
        for k in range(n - 1, j, -1):
            left = left @ props[k]
        right = np.eye(DIM2, dtype=complex)
        for k in range(j - 1, -1, -1):
            right = right @ props[k]
        G += left @ fre[j] @ right
    return G


def ideal_channel(segs) -> np.ndarray:
    E = np.eye(DIM2, dtype=complex)
    for sg in segs:
        E = sup_unitary(prop_from_H(hamiltonian(sg.omega, sg.delta, sg.phi), sg.tau)) @ E
    return E


# ----------------------------------------------------------------------------
# 4. OBSERVABLES  (diagonal in the computational basis = actually measurable)
# ----------------------------------------------------------------------------

OBSERVABLES = {
    "P_gg": np.diag([0.0, 0.0, 0.0, 1.0]),
    "P_rr": np.diag([1.0, 0.0, 0.0, 0.0]),
    "P_one": np.diag([0.0, 1.0, 1.0, 0.0]),  # exactly one Rydberg
    "N_ryd": np.diag([2.0, 1.0, 1.0, 0.0]),  # total Rydberg number
    "ZZ": np.diag([1.0, -1.0, -1.0, 1.0]),  # parity
}


def probs_of(E: np.ndarray) -> np.ndarray:
    """Diagonal populations after the channel, starting from |gg>."""
    rho = unvec(E @ vec(RHO0))
    p = np.real(np.diag(rho))
    return p


def observable_value(M: np.ndarray, E: np.ndarray) -> float:
    return float(np.real(np.trace(M @ unvec(E @ vec(RHO0)))))


def observable_variance(M: np.ndarray, E: np.ndarray) -> float:
    """Exact single-shot variance of the measured random variable M (diagonal)."""
    p = probs_of(E)
    m = np.real(np.diag(M))
    mean = float(p @ m)
    return float(p @ (m**2) - mean**2)


# ----------------------------------------------------------------------------
# 5. PULSER COMPILATION + RESAMPLING GATE
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
            # on a real device the delay is inserted automatically by the phase
            # jump; adding it again would double-count it
            continue
        seq.add(
            Pulse.ConstantPulse(
                dur_ns, float(sg.omega), float(sg.delta), float(sg.phi) % (2 * np.pi)
            ),
            "ryd",
        )
    return seq


def unitary_from_sampled_program(seq) -> np.ndarray:
    """Rebuild U from MockDevice's unmodulated 1 ns sampled program.

    This verifies the Pulser translation of the programmed square pulses. It is
    not an AnalogDevice compilation or a finite-bandwidth waveform validation.
    """
    from pulser.sampler import sample

    ch = sample(seq).channel_samples["ryd"]
    amp, det, ph = np.asarray(ch.amp), np.asarray(ch.det), np.asarray(ch.phase)
    dt = 1e-3  # 1 ns in us
    U = np.eye(DIM, dtype=complex)
    # group consecutive identical (amp, det, phase) samples for speed
    key = np.stack([amp, det, ph], axis=1)
    starts = [0] + list(np.where(np.any(np.diff(key, axis=0) != 0, axis=1))[0] + 1)
    starts.append(len(amp))
    for a, b in zip(starts[:-1], starts[1:]):
        H = hamiltonian(amp[a], det[a], ph[a])
        U = prop_from_H(H, (b - a) * dt) @ U
    return U


def pulser_hamiltonian_probe():
    """Gate H0: assert our hamiltonian() equals Pulser's, no convention guessing."""
    from pulser_simulation import QutipEmulator

    seg = Segment(2 * np.pi * 2.0, 2 * np.pi * (-2.3), 0.4, 0.120)
    seq = build_pulser_sequence([seg], "mock")
    sim = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
    H_pulser = np.asarray(sim.get_hamiltonian(60).full())
    H_ours = hamiltonian(seg.omega, seg.delta, seg.phi)
    err = float(np.max(np.abs(H_pulser - H_ours)))
    init = np.asarray(sim.initial_state.full()).ravel()
    gg_ok = bool(abs(abs(init[IDX_GG]) - 1.0) < 1e-12)
    return err, gg_ok


def _pulser_result_probabilities(result) -> np.ndarray:
    """Extract final populations from legacy or current Qutip result objects."""
    if hasattr(result, "states"):
        state = result.states[-1]
    elif hasattr(result, "get_final_state"):
        state = result.get_final_state()
    elif hasattr(result, "final_state"):
        state = result.final_state
    else:
        raise TypeError("Unsupported Pulser/Qutip result object: no final state.")
    arr = np.asarray(state.full() if hasattr(state, "full") else state)
    if arr.ndim == 1 or 1 in arr.shape:
        return np.abs(arr.reshape(-1)) ** 2
    return np.real(np.diag(arr))


def pulser_validation(segs, gamma_list=(0.0, 0.030)):
    """Three separately labelled checks against Pulser.

    L1  internal H(t) against Pulser's sampled H(t)
    L2  complete ideal superoperator from Pulser's sampled H(t) against the
        internal complete ideal superoperator (plus a non-voting |gg> check)
    L3  QutipEmulator.run() convergence audit against exact Liouville
        propagation, using the current NoiseModel API.

    L3 is diagnostic. A mismatch is labelled NOT_RESOLVED unless tightening the
    ODE tolerances and max_step demonstrably converges below the declared signal
    resolution. No intrinsic "systematic error" is assigned without that audit.
    """
    from pulser.noise_model import NoiseModel
    from pulser_simulation import QutipEmulator

    out = {}
    seq = build_pulser_sequence(segs, "mock")
    sim = QutipEmulator.from_sequence(
        seq, sampling_rate=1.0, evaluation_times="Minimal"
    )
    n_ns = int(round(sum(sg.tau for sg in segs) * 1000))

    # L1 and L2 share one pass: compare H at every 1 ns sample and propagate it.
    edges = np.cumsum([sg.tau for sg in segs])
    H_ours = [hamiltonian(sg.omega, sg.delta, sg.phi) for sg in segs]
    U = np.eye(DIM, dtype=complex)
    l1 = 0.0
    for k in range(n_ns):
        Hp = np.asarray(sim.get_hamiltonian(k).full())
        j = int(np.searchsorted(edges, k * 1e-3, side="right"))
        j = min(j, len(segs) - 1)
        l1 = max(l1, float(np.max(np.abs(Hp - H_ours[j]))))
        U = prop_from_H(Hp, 1e-3) @ U
    out["L1_hamiltonian_max_abs_err"] = float(l1)
    out["L1_samples_checked"] = int(n_ns)

    E_pulser_h = sup_unitary(U)
    E_internal = ideal_channel(segs)
    out["L2_complete_ideal_superoperator_max_abs_err"] = float(
        np.max(np.abs(E_pulser_h - E_internal))
    )
    p_exact_pulser_h = np.abs(
        U @ np.array([0, 0, 0, 1], dtype=complex)
    ) ** 2
    p_internal = probs_of(channel(segs, 0.0))
    out["L2_gg_population_max_abs_err_nonvoting"] = float(
        np.max(np.abs(p_exact_pulser_h - p_internal))
    )

    # L3: progressively stricter QuTiP profiles. max_step is in microseconds.
    profiles = {
        "default": {},
        "tight": {
            "atol": 1e-10,
            "rtol": 1e-10,
            "max_step": 1e-3,
            "nsteps": 200000,
        },
        "tighter": {
            "atol": 1e-12,
            "rtol": 1e-12,
            "max_step": 2.5e-4,
            "nsteps": 500000,
        },
    }
    l3 = {"profiles": {}, "successive_profile_population_changes": {}}
    profile_probabilities = {}
    for profile, options in profiles.items():
        l3["profiles"][profile] = {}
        profile_probabilities[profile] = {}
        for g in gamma_list:
            noise_model = None
            if g > 0:
                noise_model = NoiseModel(
                    eff_noise_opers=(NN,),
                    eff_noise_rates=(float(g),),
                )
            emulator = QutipEmulator.from_sequence(
                seq,
                sampling_rate=1.0,
                evaluation_times="Minimal",
                noise_model=noise_model,
            )
            result = emulator.run(progress_bar=False, **options)
            pv = _pulser_result_probabilities(result)
            profile_probabilities[profile][g] = pv
            l3["profiles"][profile][f"gamma_{g}"] = {
                "max_population_abs_error_vs_exact": float(
                    np.max(np.abs(pv - probs_of(channel(segs, g))))
                ),
                "populations": pv.tolist(),
            }

    names = list(profiles)
    for previous, current in zip(names[:-1], names[1:]):
        key = f"{previous}_to_{current}"
        l3["successive_profile_population_changes"][key] = {
            f"gamma_{g}": float(
                np.max(
                    np.abs(
                        profile_probabilities[current][g]
                        - profile_probabilities[previous][g]
                    )
                )
            )
            for g in gamma_list
        }
    out["L3_qutip_convergence_audit"] = l3
    return out


def compiled_blocks(seq, modulation=False):
    """Return the schedule Pulser would actually play, as (amp, det, phase, tau)."""
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


def analog_schedule_report(z_map, U0, observable_name, readout, log):
    """Audit Pulser's declared AnalogDevice and modulation model.

    Two distinct effects are separated:
      (a) phase-jump idle gaps  -> deterministic, modelled exactly by MODEL['gap_tau']
      (b) finite modulation bandwidth (8 MHz) -> smears the 120 ns segment edges,
          is NOT in the model, and breaks the exact endpoint equivalence.

    Effect (b) produces a *coherent* difference between the two controls that is
    present already at gamma = 0. If it exceeds the path-induced weak-noise split,
    the observable-level program is confounded under this Pulser device model.
    This is not a calibrated FRESNEL waveform or a QPU execution.
    """
    out = {}
    try:
        seqs = {
            tag: build_pulser_sequence(segments_of(z, readout=readout), "analog")
            for tag, z in z_map.items()
        }
    except Exception as exc:
        from pulser.devices import AnalogDevice

        lim = AnalogDevice.channels["rydberg_global"].max_amp
        om = max(sg.omega for z in z_map.values()
                 for sg in segments_of(z, readout=readout))
        log(f"  AnalogDevice rejected the sequence: {type(exc).__name__}: {exc}")
        log(f"  max Omega in the control = {om:.3f} rad/us, device limit = "
            f"{lim:.3f} rad/us  ->  re-run with --omega-scale "
            f"{0.98*lim/om:.3f}")
        return {"compilable": False, "error": f"{type(exc).__name__}: {exc}",
                "max_omega": float(om), "device_max_amp": float(lim),
                "suggested_omega_scale": float(0.98 * lim / om)}

    out["compilable"] = True
    M = OBSERVABLES[observable_name]
    U_unmod, U_mod = {}, {}
    for tag, seq in seqs.items():
        blocks_u = compiled_blocks(seq, modulation=False)
        blocks_m = compiled_blocks(seq, modulation=True)
        U_unmod[tag] = unitary_of_blocks(blocks_u)
        U_mod[tag] = unitary_of_blocks(blocks_m)
        out[f"{tag}_n_blocks_unmod"] = len(blocks_u)
        out[f"{tag}_duration_us"] = float(sum(b.tau for b in blocks_u))

    # (a) infer the idle gap the device actually inserts, then compare
    blocks_ref = compiled_blocks(seqs["z0"], modulation=False)
    idle = [b.tau for b in blocks_ref if b.omega == 0.0 and b.delta == 0.0]
    inferred_gap = float(sum(idle) / max(len([b for b in blocks_ref
                                              if b.omega != 0.0]) - 1, 1))
    out["inferred_gap_us"] = inferred_gap
    out["model_gap_us"] = MODEL["gap_tau"]
    if abs(inferred_gap - MODEL["gap_tau"]) > 1e-9:
        log(f"  !! device inserts {inferred_gap*1000:.0f} ns of idle between drive "
            f"segments; current model uses {MODEL['gap_tau']*1000:.0f} ns.")
        log(f"     re-run with --gap {inferred_gap:.3f} to lift on the executed "
            f"schedule.")
    model_err = {
        tag: float(np.max(np.abs(U_unmod[tag] - unitary_of_blocks(
            segments_of(z, readout=readout)))))
        for tag, z in z_map.items()
    }
    out["gap_model_vs_pulser_maxabs"] = model_err
    out["gap_model_ok"] = bool(max(model_err.values()) < 1e-9)
    log(f"  executed duration on AnalogDevice: "
        f"{out['z0_duration_us']:.3f} us  (gap-aware analytic model: "
        f"{sum(s.tau for s in segments_of(np.zeros(18), readout=readout)):.3f} us)")
    log(f"  gap model reproduces Pulser schedule: max|dU| = "
        f"{max(model_err.values()):.3e}")

    # (b) modulation damage, measured on the observable itself, at gamma = 0
    def p_of(U):
        rho = U @ RHO0 @ U.conj().T
        return float(np.real(np.trace(M @ rho)))

    for tag in z_map:
        out[f"{tag}_eps_U_mod_vs_unmod"] = float(
            abs(1 - abs(np.trace(U_unmod[tag].conj().T @ U_mod[tag])) ** 2 / DIM**2)
        )
    ref = "z0"
    for tag in z_map:
        if tag == ref:
            continue
        out[f"{tag}_eps_U_unmod_vs_z0"] = float(
            abs(1 - abs(np.trace(U_unmod[ref].conj().T @ U_unmod[tag])) ** 2 / DIM**2)
        )
        out[f"{tag}_eps_U_mod_vs_z0"] = float(
            abs(1 - abs(np.trace(U_mod[ref].conj().T @ U_mod[tag])) ** 2 / DIM**2)
        )
        out[f"{tag}_coherent_dP_from_modulation"] = abs(
            p_of(U_mod[tag]) - p_of(U_mod[ref])
        )
        log(f"  {tag}: eps_U(unmod vs z0) = {out[f'{tag}_eps_U_unmod_vs_z0']:.3e}  ->  "
            f"eps_U(modulated vs z0) = {out[f'{tag}_eps_U_mod_vs_z0']:.3e}")
        log(f"      coherent |dP| injected by modulation at gamma=0: "
            f"{out[f'{tag}_coherent_dP_from_modulation']:.3e}")
    return out


def analog_device_max_amp_mhz() -> "float | None":
    """Read the active Pulser AnalogDevice limit instead of hard-coding it."""
    try:
        from pulser.devices import AnalogDevice

        return float(
            AnalogDevice.channels["rydberg_global"].max_amp / (2 * np.pi)
        )
    except Exception:
        return None


def analog_compilability_gate(results, design, log):
    """Validate every frozen control/reference/readout pair on AnalogDevice.

    Acceptance means only that Pulser's declared AnalogDevice constraints accept
    the programmed sequence. It does not establish a calibrated FRESNEL waveform,
    post-modulation endpoint equality, cloud execution, or QPU evidence.
    """
    cases = {}
    all_ok = True
    for direction in design:
        readout = Segment(
            design[direction]["readout_omega"],
            0.0,
            design[direction]["readout_phi"],
            design[direction]["readout_tau"],
        )
        for role, z in (
            ("reference", np.zeros(18)),
            ("candidate", results[direction]["z"]),
        ):
            name = f"{direction}_{role}"
            try:
                seq = build_pulser_sequence(
                    segments_of(z, readout=readout), "analog"
                )
                cases[name] = {
                    "accepted": True,
                    "duration_ns": int(seq.get_duration()),
                }
            except Exception as exc:
                all_ok = False
                cases[name] = {
                    "accepted": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
    limit = analog_device_max_amp_mhz()
    out = {
        "cases": cases,
        "device_max_amp_over_2pi_mhz": limit,
        "pass": bool(all_ok),
        "status": "SUPPORTED" if all_ok else "REJECTED",
        "note": "Pulser AnalogDevice acceptance only; modulated endpoint matching "
                "is a separate test.",
    }
    log(
        "  AnalogDevice program acceptance: "
        f"{out['status']} ({sum(v['accepted'] for v in cases.values())}/"
        f"{len(cases)} cases)"
    )
    for name, rec in cases.items():
        if not rec["accepted"]:
            log(f"    {name}: {rec['error']}")
    return out


# ----------------------------------------------------------------------------
# 6. MAIN EXPERIMENT
# ----------------------------------------------------------------------------

GAMMAS = np.array([0.0, 0.001875, 0.003750, 0.007500, 0.015000, 0.030000])


def loop_vertices(eps, direction):
    if direction == "CW":
        return [(0, 0), (eps, 0), (eps, eps), (0, eps), (0, 0)]
    return [(0, 0), (0, eps), (eps, eps), (eps, 0), (0, 0)]


def scan_readout(dG, n_theta=41, n_phi=49, omega_max=2 * np.pi * 2.5, tau_ro=0.100,
                 quick=False):
    """Search the appended readout pulse (Omega, phi) and the observable M
    maximizing the predicted first-order slope |chi|.

    The readout pulse is a genuine extra segment: interaction and noise act during
    it. Because E_z(0) = E_0(0), the readout only multiplies the derivative from
    the left, so dG_total = P_ro(0) dG exactly.
    """
    if quick:
        n_theta, n_phi = 15, 17
    best = None
    rows = []
    omegas = np.linspace(0.0, omega_max, n_theta)
    phis = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    # the executed tail is (idle gap, if the schedule has one) then the readout
    # pulse; omitting the gap here would corrupt the predicted slope
    gap = MODEL["gap_tau"]
    P_gap = (
        sup_unitary(prop_from_H(hamiltonian(0.0, 0.0, 0.0), gap))
        if gap > 0
        else np.eye(DIM2, dtype=complex)
    )
    for om in omegas:
        for ph in phis:
            P_ro = sup_unitary(
                prop_from_H(hamiltonian(om, 0.0, ph), tau_ro)
            ) @ P_gap
            drho = unvec(P_ro @ dG @ vec(RHO0))
            for name, M in OBSERVABLES.items():
                chi = float(np.real(np.trace(M @ drho)))
                rows.append((om, ph, name, chi))
                if best is None or abs(chi) > abs(best[3]):
                    best = (om, ph, name, chi)
            if om == 0.0:
                break  # phase is irrelevant when Omega = 0
    # unconstrained bound for rho = |gg>: max over ALL POVM-like M with ||M||<=1
    drho_no_ro = unvec(P_gap @ dG @ vec(RHO0))
    bound = float(np.sum(np.abs(np.linalg.eigvalsh((drho_no_ro + drho_no_ro.conj().T) / 2))))
    return best, rows, bound


def loglog_slope(x, y):
    m = (x > 0) & (y > 0)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


def run_direction(direction, eps, h_s, U0, args, log):
    t0 = time.time()
    log(f"\n--- lift {direction}, eps = {eps:g}, h_s = {h_s:g} ---")
    z, audit = m2_transport(U0, loop_vertices(eps, direction), h_s)
    eu = endpoint_infidelity(z, U0)
    log(f"    steps={audit['n_steps']}  rank(s)={audit['rank_set']}  "
        f"eps_U={eu:.3e}  |r|={audit['final_res']:.3e}  ({time.time()-t0:.1f}s)")
    return z, audit, eu


def _in_notebook() -> bool:
    """True inside Jupyter / Colab, where sys.argv carries the kernel's -f flag."""
    if "ipykernel" in sys.modules or "google.colab" in sys.modules:
        return True
    return any("kernel-" in a and a.endswith(".json") for a in sys.argv[1:])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ep_obs_stage2_gap_revalidated.py")
    ap.add_argument(
        "--quick",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the coarse transport/readout screen (default: enabled)",
    )
    ap.add_argument("--eps", type=float, default=0.040)
    ap.add_argument("--hs", type=float, default=0.002)
    ap.add_argument("--eps-sweep", action="store_true")
    ap.add_argument("--step-halving", action="store_true",
                    help="audit the h_s dependence of ||K_z-K_0||_F: it is NOT a "
                         "converged number, and this reports how far from "
                         "converged it is")
    ap.add_argument("--no-pulser-check", action="store_true")
    ap.add_argument(
        "--outdir",
        default=None,
        help="output directory; default is a timestamped stage-2 directory",
    )
    ap.add_argument("--gamma-max-extra", type=float, default=0.0,
                    help="if >0, append extra gammas up to this value")
    ap.add_argument("--gap", type=float, default=0.340,
                    help="idle gap in us inserted between drive segments "
                         "(default: 0.340, frozen from Stage 1)")
    ap.add_argument("--omega-scale", type=float, default=0.80,
                    help="global rescale of the reference Rabi table; device "
                         "limits are read from AnalogDevice at runtime")
    ap.add_argument("--phase-sign", type=float, default=1.0, choices=[1.0, -1.0],
                    help="+1 = Pulser convention (default), -1 = manuscript's "
                         "+sin(phi) Y convention applied to the same phase table")
    ap.add_argument("--source-path", default=None,
                    help="path to the archived source, for the provenance digest "
                         "when running from a notebook cell (no __file__)")
    ap.add_argument(
        "--analog-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="audit Pulser's AnalogDevice modulation model (default: enabled)",
    )
    if argv is None:
        argv = [] if _in_notebook() else sys.argv[1:]
    # parse_known_args so that a host environment injecting its own flags
    # (Colab's "-f kernel-*.json") cannot kill the run
    args, ignored = ap.parse_known_args(list(argv))
    if ignored:
        print(f"[note] ignoring arguments not belonging to this script: {ignored}")
    if args.outdir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.outdir = f"ep_obs_stage2_gap_revalidated_{stamp}"

    global _SOURCE_PATH_OVERRIDE
    if args.source_path:
        _SOURCE_PATH_OVERRIDE = args.source_path
    MODEL["gap_tau"] = float(args.gap)
    MODEL["omega_scale"] = float(args.omega_scale)
    MODEL["phase_sign"] = float(args.phase_sign)
    os.makedirs(args.outdir, exist_ok=True)
    logfile = open(os.path.join(args.outdir, "ep_obs_run.log"), "w")

    def log(msg=""):
        print(msg)
        logfile.write(str(msg) + "\n")
        logfile.flush()

    h_s = 0.008 if args.quick else args.hs
    cert = {
        "version": VERSION,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_sha256": self_sha256(),
        "packages": package_versions(),
        "args": vars(args),
        "model": {
            "atoms": 2, "separation_um": ATOM_SEP, "C6": C6, "U_int": U_INT,
            "tau_us": TAU, "n_segments": NSEG, "T_us": T_TOTAL,
            "gap_tau_us": MODEL["gap_tau"], "omega_scale": MODEL["omega_scale"],
            "phase_sign": MODEL["phase_sign"],
            "schedule_duration_us": float(
                sum(s_.tau for s_ in segments_of(np.zeros(18)))),
            "dissipator": "occupation dephasing, sum_k D[n_k]",
            "input_state": "|gg><gg| (native initialization used in this audit)",
        },
        "gates": {},
    }

    log("=" * 78)
    log(f"{VERSION}  observable-level executed-path diagnostic (PASQAL / Pulser local)")
    log("=" * 78)
    log(f"UTC          : {cert['utc']}")
    log(f"source sha256: {cert['source_sha256']}")
    for k, v in cert["packages"].items():
        log(f"  {k:20s} {v}")

    # ---------------- Gate H0 : Hamiltonian convention -----------------------
    if not args.no_pulser_check:
        err, gg_ok = pulser_hamiltonian_probe()
        cert["gates"]["H0_hamiltonian_matches_pulser"] = {
            "max_abs_err": err, "init_state_is_gg": gg_ok, "pass": bool(err < 1e-9 and gg_ok)
        }
        log(f"\n[H0] internal H == Pulser H : max|dH| = {err:.3e}   init=|gg>: {gg_ok}")

    U0 = unitary_of_z(np.zeros(18))

    # ---------------- Gate H1 : phase-convention bookkeeping -----------------
    # For a single Hermitian segment:
    #   H(-phi) = H(phi)^* = H(phi)^T,
    #   exp[-i H(-phi)t] = exp[-i H(phi)t]^T.
    # For a chronological product, transposition reverses matrix-product order.
    # Negating every phase without reversing the schedule therefore does not
    # generally transpose the complete time-ordered unitary.
    h_conj_err = float(
        np.max(np.abs(hamiltonian(1.3, -0.7, -0.9) - hamiltonian(1.3, -0.7, 0.9).conj()))
    )
    h_transpose_err = float(
        np.max(np.abs(hamiltonian(1.3, -0.7, -0.9) - hamiltonian(1.3, -0.7, 0.9).T))
    )
    Uj_plus = prop_from_H(hamiltonian(1.3, -0.7, 0.9), 0.12)
    Uj_minus = prop_from_H(hamiltonian(1.3, -0.7, -0.9), 0.12)
    single_segment_transpose_err = float(np.max(np.abs(Uj_minus - Uj_plus.T)))
    U_plus = unitary_of_z(np.zeros(18), phase_sign=+1.0)
    U_minus = unitary_of_z(np.zeros(18), phase_sign=-1.0)
    multisegment_transpose_gap = float(np.max(np.abs(U_minus - U_plus.T)))
    cert["gates"]["H1_phase_convention_bookkeeping"] = {
        "H_minus_phi_equals_conj_H": h_conj_err,
        "H_minus_phi_equals_transpose_H": h_transpose_err,
        "single_segment_U_minus_phi_equals_transpose_U": single_segment_transpose_err,
        "multisegment_same_order_transpose_gap": multisegment_transpose_gap,
        "pass": bool(
            h_conj_err < 1e-14
            and h_transpose_err < 1e-14
            and single_segment_transpose_err < 1e-14
        ),
        "note": "The gate checks the Hamiltonian and single-segment identities. "
                "The complete same-order product is not expected to transpose, "
                "because transposition reverses matrix-product order.",
    }
    log(f"[H1] H(-phi) == H(phi)* == H(phi)^T : "
        f"max errors={h_conj_err:.3e}/{h_transpose_err:.3e}")
    log(f"     one-segment |U(-phi)-U(phi)^T|={single_segment_transpose_err:.3e}")
    log(f"     same-order multi-segment transpose gap="
        f"{multisegment_transpose_gap:.3e} (expected nonzero)")

    # ---------------- 1. freeze two equivalent controls ----------------------
    log("\n" + "-" * 78)
    log("STEP 1  freeze two full-unitary-equivalent controls (M2 lift)")
    log("-" * 78)
    results = {}
    for direction in ("CW", "CCW"):
        z, audit, eu = run_direction(direction, args.eps, h_s, U0, args, log)
        results[direction] = {"z": z, "audit": audit, "eps_U": eu}

    cert["gates"]["G1_endpoint_infidelity"] = {
        d: results[d]["eps_U"] for d in results
    }
    cert["gates"]["G1_endpoint_infidelity"]["threshold"] = 1e-11
    cert["gates"]["G1_endpoint_infidelity"]["pass"] = bool(
        all(results[d]["eps_U"] <= 1e-11 for d in results)
    )
    cert["gates"]["G2_residual_norm"] = {
        d: results[d]["audit"]["final_res"] for d in results
    }
    cert["gates"]["G2_residual_norm"]["threshold"] = 2e-9
    cert["gates"]["G2_residual_norm"]["pass"] = bool(
        all(results[d]["audit"]["final_res"] <= 2e-9 for d in results)
    )
    cert["gates"]["G3_rank_audit"] = {
        d: {
            "ranks": results[d]["audit"]["rank_set"],
            "min_kept_sv": results[d]["audit"]["min_smin_keep"],
            "max_dropped_sv": results[d]["audit"]["max_smax_drop"],
            "gap": results[d]["audit"]["min_smin_keep"]
            / max(results[d]["audit"]["max_smax_drop"], 1e-300),
        }
        for d in results
    }
    cert["gates"]["G3_rank_audit"]["pass"] = bool(
        all(
            results[d]["audit"]["rank_set"] == [8]
            and results[d]["audit"]["min_smin_keep"]
            / max(results[d]["audit"]["max_smax_drop"], 1e-300)
            >= 1e4
            for d in results
        )
    )
    cert["gates"]["G4_distinct_controls"] = {
        d: float(np.linalg.norm(results[d]["z"])) for d in results
    }
    cert["gates"]["G4_distinct_controls"]["pass"] = bool(
        all(np.linalg.norm(results[d]["z"]) > 1e-6 for d in results)
    )

    # ---------------- G14 : transport convergence audit ----------------------
    if args.step_halving:
        log("\n" + "-" * 78)
        log("G14  step-halving audit of the transport, both directions")
        log("     ||K_z-K_0||_F is a property of the particular fiber point the")
        log("     transport lands on. That point moves with h_s. The physics claim")
        log("     (a distinct control with an identical endpoint exists, and its")
        log("     first-order response predicts the split) holds at every h_s; the")
        log("     numerical VALUE of ||dK|| does not converge quickly.")
        log("-" * 78)
        segs_ref = segments_of(np.zeros(18))
        K0_ref = np.linalg.solve(ideal_channel(segs_ref),
                                 channel_derivative(segs_ref))
        LIFT_RES_MAX, LIFT_EPSU_MAX = 2e-9, 1e-11
        g14 = {"lift_residual_threshold": LIFT_RES_MAX,
               "endpoint_infidelity_threshold": LIFT_EPSU_MAX,
               "directions": {}}
        for direction in ("CW", "CCW"):
            hist, rejected = {}, []
            for hs_a in (h_s, h_s / 2, h_s / 4):
                za, auda = m2_transport(U0, loop_vertices(args.eps, direction), hs_a)
                eua = endpoint_infidelity(za, U0)
                sg = segments_of(za)
                dKa = np.linalg.solve(ideal_channel(sg),
                                      channel_derivative(sg)) - K0_ref
                rec = {
                    "nominal_h_s": float(hs_a),
                    "effective_h_s": auda["effective_h_s"],
                    "effective_h_s_spread": auda["effective_h_s_spread"],
                    "n_steps": auda["n_steps"],
                    "lift_final_res": auda["final_res"],
                    "max_lift_res": auda["max_lift_res"],
                    "eps_U": eua,
                    "RK": float(np.linalg.norm(dKa, "fro") / 16.0),
                    "z_norm": float(np.linalg.norm(za)),
                }
                # a silently failed lift must never enter the extrapolation
                rec["usable"] = bool(auda["final_res"] <= LIFT_RES_MAX
                                     and eua <= LIFT_EPSU_MAX)
                if rec["usable"]:
                    # keyed by the NOMINAL step, which is distinct by construction.
                    # Keying by the effective step would silently drop a grid
                    # whenever round(L/h_s) quantization makes two grids coincide.
                    hist[hs_a] = (za, rec["RK"], rec)
                else:
                    rejected.append(rec)
                    log(f"  [{direction}] h_s={hs_a:.5f} REJECTED: "
                        f"|r|={auda['final_res']:.2e}, eps_U={eua:.2e}")
                log(f"  [{direction}] h_s nominal {hs_a:.5f} / effective "
                    f"{auda['effective_h_s']:.5f} (spread "
                    f"{auda['effective_h_s_spread']:.1e}, {auda['n_steps']} steps)"
                    f"   ||dK||_F/16 = {rec['RK']:.7e}   |r| = "
                    f"{auda['final_res']:.2e}   eps_U = {eua:.2e}")
            entry = {"grids": [hist[k][2] for k in sorted(hist, reverse=True)],
                     "rejected": rejected}
            ks = sorted(hist, reverse=True)          # nominal, coarsest first
            eff = {k: hist[k][2]["effective_h_s"] for k in ks}
            # round(L/h_s) quantization can make two nominal grids share the same
            # effective step; that breaks the refinement assumption entirely
            n_distinct = len({round(v, 15) for v in eff.values()})
            entry["effective_h_s"] = {f"nominal={k}": eff[k] for k in ks}
            entry["distinct_effective_grids"] = n_distinct
            if n_distinct < len(ks):
                log(f"  [{direction}] !! quantization collapsed {len(ks)} nominal "
                    f"grids onto {n_distinct} effective ones; refinement is not "
                    f"actually happening. Reduce h_s or increase eps.")
            deltas = []
            for a, b in zip(ks[:-1], ks[1:]):
                dz = float(np.linalg.norm(hist[b][0] - hist[a][0])
                           / np.linalg.norm(hist[b][0]))
                dk = float(abs(hist[b][1] - hist[a][1]) / hist[b][1])
                ea, eb = eff[a], eff[b]
                deltas.append({"from_effective_h_s": ea, "to_effective_h_s": eb,
                               "refinement_ratio": float(ea / eb) if eb > 0 else
                               float("inf"),
                               "delta_z": dz, "delta_K": dk})
                log(f"  [{direction}] {ea:.5f} -> {eb:.5f} "
                    f"(ratio {ea/eb:.4f}): delta_z = {dz:.4f}   "
                    f"delta_K = {dk:.4f}")
            entry["deltas"] = deltas
            if len(ks) >= 2 and n_distinct >= 2:
                # first-order Richardson on the ACTUAL effective steps:
                #   f0 = f(h2) + (f(h2) - f(h1)) * h2 / (h1 - h2)
                # reduces to 2 f(h2) - f(h1) only when h1 = 2 h2 exactly
                h1, h2 = eff[ks[-2]], eff[ks[-1]]
                f1, f2 = hist[ks[-2]][1], hist[ks[-1]][1]
                if h1 - h2 <= 1e-12 * max(h1, 1e-30):
                    rich = float("nan")
                    log(f"  [{direction}] two finest effective steps coincide; "
                        f"Richardson extrapolation skipped")
                else:
                    rich = f2 + (f2 - f1) * h2 / (h1 - h2)
                entry["richardson_h_to_0"] = float(rich)
                entry["richardson_used"] = {"h1": h1, "h2": h2}
                log(f"  [{direction}] first-order extrapolate h_s -> 0 : "
                    f"||dK||_F/16 ~ {rich:.4e}")
                log(f"  [{direction}] -> quote at most 2 significant figures "
                    f"({f2:.2e}); further digits are an artefact of h_s.")
            ratios_ok = all(1.5 <= d["refinement_ratio"] <= 3.0 for d in deltas)
            entry["refinement_ratios_ok"] = bool(ratios_ok)
            entry["pass"] = bool(
                not rejected
                and len(ks) == 3
                and n_distinct == 3
                and ratios_ok
                and np.isfinite(entry.get("richardson_h_to_0", np.nan))
                and all(d["delta_z"] <= 0.02 and d["delta_K"] <= 0.02
                        for d in deltas)
            )
            g14["directions"][direction] = entry
        g14["pass"] = bool(all(v["pass"] for v in g14["directions"].values()))
        g14["note"] = ("passing does NOT mean ||dK|| is converged; convergence is "
                       "first order and the value still drifts in the third "
                       "significant figure. Grids whose lift did not meet the "
                       "residual and endpoint thresholds are rejected and excluded "
                       "from the extrapolation.")
        cert["gates"]["G14_transport_step_halving"] = g14

    # ---------------- 2. verify MockDevice sampled-program translation -------
    log("\n" + "-" * 78)
    log("STEP 2  verify MockDevice's unmodulated sampled-program translation")
    log("        (this is not an AnalogDevice compilation)")
    log("-" * 78)
    if not args.no_pulser_check:
        translation = {}
        for tag, z in [("z0", np.zeros(18))] + [(d, results[d]["z"]) for d in results]:
            segs = segments_of(z)
            seq = build_pulser_sequence(segs, "mock")
            Uc = unitary_from_sampled_program(seq)
            Ua = unitary_of_z(z)
            translation[tag] = {
                "sampled_vs_analytic_maxabs": float(np.max(np.abs(Uc - Ua))),
                "sampled_eps_U_vs_U0": float(
                    abs(1.0 - abs(np.trace(U0.conj().T @ Uc)) ** 2 / DIM**2)
                ),
            }
            log(f"  {tag:4s}  ||U_sampled - U_analytic||_max = "
                f"{translation[tag]['sampled_vs_analytic_maxabs']:.3e}   "
                f"eps_U(sampled vs U0) = "
                f"{translation[tag]['sampled_eps_U_vs_U0']:.3e}")
        cert["gates"]["G5_mockdevice_unmodulated_translation"] = translation
        cert["gates"]["G5_mockdevice_unmodulated_translation"]["pass"] = bool(
            all(v["sampled_eps_U_vs_U0"] <= 1e-11 for v in translation.values())
        )

    # ---------------- 3-5. response operators + observable design ------------
    log("\n" + "-" * 78)
    log("STEP 3  path-resolved response and observable design")
    log("-" * 78)
    segs0 = segments_of(np.zeros(18))
    G0 = channel_derivative(segs0)
    U0_sup = ideal_channel(segs0)
    K0 = np.linalg.solve(U0_sup, G0)

    design = {}
    for d in results:
        segs = segments_of(results[d]["z"])
        Gz = channel_derivative(segs)
        dG = Gz - G0
        Kz = np.linalg.solve(ideal_channel(segs), Gz)
        dK = Kz - K0
        RK = float(np.linalg.norm(dK, "fro") / 16.0)
        # consistency: dG should equal U0_sup @ dK
        num = float(np.linalg.norm(dG - U0_sup @ dK, "fro"))
        den = 0.5 * (np.linalg.norm(dG, "fro") + np.linalg.norm(U0_sup @ dK, "fro"))
        delta_common = float(num / max(den, 1e-300))
        best, rows, bound = scan_readout(dG, quick=args.quick)
        om_ro, ph_ro, mname, chi = best
        design[d] = {
            "RK": RK,
            "delta_common": delta_common,
            "readout_omega": float(om_ro),
            "readout_phi": float(ph_ro),
            "readout_tau": 0.100,
            "observable": mname,
            "chi": float(chi),
            "chi_bound_trace_norm": bound,
            "dG": dG,
        }
        log(f"  {d:4s}  ||K_z-K_0||_F/16 = {RK:.7e}   delta_common = {delta_common:.2e}")
        log(f"        best readout: Omega/2pi = {om_ro/(2*np.pi):.4f} MHz, "
            f"phi = {ph_ro:.4f} rad, tau = 100 ns")
        log(f"        best observable: {mname:6s}  chi = {chi:+.7e}  "
            f"(bound over all M with ||M||<=1: {bound:.3e})")

    cert["gates"]["G6_delta_common"] = {
        d: design[d]["delta_common"] for d in design
    }
    cert["gates"]["G6_delta_common"]["threshold"] = 1e-8
    cert["gates"]["G6_delta_common"]["pass"] = bool(
        all(design[d]["delta_common"] <= 1e-8 for d in design)
    )

    # AnalogDevice acceptance is a real operational gate, not an informational
    # print. It is evaluated after readout design so every complete program is
    # checked, including the appended readout pulse.
    if not args.no_pulser_check:
        log("\n  DEVICE GATE")
        cert["gates"]["G13_analogdevice_program_compilable"] = (
            analog_compilability_gate(results, design, log)
        )
    else:
        cert["gates"]["G13_analogdevice_program_compilable"] = {
            "pass": False,
            "status": "NOT_RUN",
            "note": "Pulser checks disabled; no device claim is available.",
        }

    # ---------------- 6. gamma sweep, exact Lindblad -------------------------
    log("\n" + "-" * 78)
    log("STEP 4-6  gamma sweep, exact Lindblad, parameter-free slope prediction")
    log("-" * 78)
    gammas = GAMMAS.copy()
    if args.gamma_max_extra > 0:
        extra = np.geomspace(GAMMAS[-1] * 2, args.gamma_max_extra, 4)
        gammas = np.concatenate([gammas, extra])

    csv_rows = []
    for d in design:
        ro = Segment(design[d]["readout_omega"], 0.0, design[d]["readout_phi"],
                     design[d]["readout_tau"])
        segs_z = segments_of(results[d]["z"], readout=ro)
        segs_0 = segments_of(np.zeros(18), readout=ro)
        M = OBSERVABLES[design[d]["observable"]]
        chi = design[d]["chi"]
        dP, Pz, P0, DE = [], [], [], []
        for g in gammas:
            Ez = channel(segs_z, g)
            E0 = channel(segs_0, g)
            pz = observable_value(M, Ez)
            p0 = observable_value(M, E0)
            Pz.append(pz)
            P0.append(p0)
            dP.append(pz - p0)
            DE.append(float(np.linalg.norm(Ez - E0, "fro") / 16.0))
        dP = np.array(dP)
        DE = np.array(DE)
        pred = chi * gammas
        resid = np.abs(dP - pred)
        rel = np.where(np.abs(dP) > 0, resid / np.maximum(np.abs(dP), 1e-300), np.nan)
        p_signal = loglog_slope(gammas, np.abs(dP))
        p_resid = loglog_slope(gammas, resid)
        max_rel = float(np.nanmax(rel[gammas > 0]))
        floor = float(abs(dP[0]))
        design[d].update(
            dict(gammas=gammas, dP=dP, DE=DE, pred=pred, resid=resid,
                 Pz=np.array(Pz), P0=np.array(P0),
                 p_signal=p_signal, p_resid=p_resid, max_rel=max_rel, floor=floor)
        )
        log(f"\n  === {d} ===  observable {design[d]['observable']}, "
            f"predicted slope chi = {chi:+.7e}")
        log(f"   {'gamma':>10s} {'P_z':>14s} {'P_z0':>14s} {'DeltaP':>13s} "
            f"{'gamma*chi':>13s} {'residual':>11s} {'rel.err':>9s}")
        for i, g in enumerate(gammas):
            log(f"   {g:10.6f} {Pz[i]:14.10f} {P0[i]:14.10f} {dP[i]:+13.6e} "
                f"{pred[i]:+13.6e} {resid[i]:11.4e} "
                f"{'' if g == 0 else f'{rel[i]:9.3%}'}")
            csv_rows.append(dict(direction=d, gamma=g, P_z=Pz[i], P_z0=P0[i],
                                 dP=dP[i], pred=pred[i], residual=resid[i],
                                 D_E=DE[i], observable=design[d]["observable"]))
        log(f"   zero-noise floor |DeltaP(0)| = {floor:.3e}")
        log(f"   log-log exponent of |DeltaP| : {p_signal:.6f}   (target 1)")
        log(f"   log-log exponent of residual : {p_resid:.6f}   (target 2)")
        log(f"   max relative error           : {max_rel:.4%}")

    cert["gates"]["G7_signal_exponent"] = {d: design[d]["p_signal"] for d in design}
    cert["gates"]["G7_signal_exponent"]["window"] = [0.94, 1.06]
    cert["gates"]["G7_signal_exponent"]["pass"] = bool(
        all(0.94 <= design[d]["p_signal"] <= 1.06 for d in design)
    )
    cert["gates"]["G8_residual_exponent"] = {d: design[d]["p_resid"] for d in design}
    cert["gates"]["G8_residual_exponent"]["window"] = [1.85, 2.15]
    cert["gates"]["G8_residual_exponent"]["pass"] = bool(
        all(1.85 <= design[d]["p_resid"] <= 2.15 for d in design)
    )
    cert["gates"]["G9_max_relative_error"] = {d: design[d]["max_rel"] for d in design}
    cert["gates"]["G9_max_relative_error"]["threshold"] = 0.03
    cert["gates"]["G9_max_relative_error"]["pass"] = bool(
        all(design[d]["max_rel"] <= 0.03 for d in design)
    )
    cert["gates"]["G10_signal_over_zero_noise_floor"] = {
        d: float(abs(design[d]["dP"][1]) / max(design[d]["floor"], 1e-300))
        for d in design
    }
    cert["gates"]["G10_signal_over_zero_noise_floor"]["threshold"] = 1e5
    cert["gates"]["G10_signal_over_zero_noise_floor"]["pass"] = bool(
        all(
            abs(design[d]["dP"][1]) / max(design[d]["floor"], 1e-300) >= 1e5
            for d in design
        )
    )

    # ---------------- Pulser emulator cross-check ----------------------------
    if not args.no_pulser_check:
        log("\n" + "-" * 78)
        log("VALIDATION AGAINST PULSER  (three independent levels)")
        log("-" * 78)
        d0 = list(design)[0]
        ro = Segment(design[d0]["readout_omega"], 0.0, design[d0]["readout_phi"],
                     design[d0]["readout_tau"])
        val = pulser_validation(segments_of(results[d0]["z"], readout=ro))
        log(f"  L1  our H(t) vs Pulser H(t)            : {val['L1_hamiltonian_max_abs_err']:.3e}")
        log("  L2  complete ideal superoperator from Pulser H(t): "
            f"{val['L2_complete_ideal_superoperator_max_abs_err']:.3e}")
        log("      non-voting |gg> population check             : "
            f"{val['L2_gg_population_max_abs_err_nonvoting']:.3e}")
        l3audit = val["L3_qutip_convergence_audit"]
        for profile, gdata in l3audit["profiles"].items():
            for gamma_key, rec in gdata.items():
                log(f"  L3  Qutip {profile:7s}, {gamma_key:14s}: "
                    f"{rec['max_population_abs_error_vs_exact']:.3e}")
        for transition, gdata in (
            l3audit["successive_profile_population_changes"].items()
        ):
            for gamma_key, value in gdata.items():
                log(f"      convergence {transition:18s}, {gamma_key:14s}: "
                    f"{value:.3e}")
        sig = abs(design[d0]["dP"][-1])
        tight_profile = l3audit["profiles"]["tighter"]
        worst_l3 = max(
            rec["max_population_abs_error_vs_exact"]
            for rec in tight_profile.values()
        )
        last_change = max(
            l3audit["successive_profile_population_changes"][
                "tight_to_tighter"
            ].values()
        )
        qutip_resolved = bool(
            worst_l3 < 0.1 * sig and last_change < 0.1 * sig
        )
        log(f"\n  path signal to be resolved             : {sig:.3e}")
        log(f"  tightest Qutip-vs-exact discrepancy    : {worst_l3:.3e}"
            f"   ({worst_l3/max(sig,1e-300):.1f}x the signal)")
        log(f"  tight-to-tighter population change     : {last_change:.3e}")
        log("  -> Qutip solver cross-check status      : "
            f"{'RESOLVED' if qutip_resolved else 'NOT_RESOLVED'}")
        log("     The exact Liouville engine remains the model result; no intrinsic")
        log("     Qutip systematic-error mechanism is asserted by this audit.")
        cert["gates"]["G11a_model_matches_pulser"] = {
            "L1": val["L1_hamiltonian_max_abs_err"],
            "L2_complete_ideal_superoperator": (
                val["L2_complete_ideal_superoperator_max_abs_err"]
            ),
            "L2_gg_population_nonvoting": (
                val["L2_gg_population_max_abs_err_nonvoting"]
            ),
            "threshold": 1e-10,
            "pass": bool(val["L1_hamiltonian_max_abs_err"] < 1e-10
                         and val["L2_complete_ideal_superoperator_max_abs_err"]
                         < 1e-10),
        }
        cert["gates"]["G11b_qutip_solver_resolution"] = {
            "tightest_profile_worst_abs_err": worst_l3,
            "tight_to_tighter_population_change": last_change,
            "signal": sig,
            "ratio_err_over_signal": float(worst_l3 / max(sig, 1e-300)),
            "resolution_fraction": 0.1,
            "status": "RESOLVED" if qutip_resolved else "NOT_RESOLVED",
            "pass": qutip_resolved,
            "profiles": l3audit,
            "note": "A failure means this solver cross-check does not resolve "
                    "the path signal under the tested numerical profiles. It "
                    "does not invalidate the exact Liouville calculation and "
                    "does not identify an intrinsic Pulser/Qutip error source.",
        }

    # ---------------- Pulser AnalogDevice model report -----------------------
    if args.analog_report and not args.no_pulser_check:
        log("\n" + "-" * 78)
        log("PULSER DEVICE-MODEL AUDIT  AnalogDevice acceptance and modulation")
        log("  (a) phase-jump idle gaps   (deterministic, modellable)")
        log("  (b) 8 MHz modulation       (breaks the endpoint equivalence)")
        log("-" * 78)
        d0 = list(design)[0]
        ro = Segment(design[d0]["readout_omega"], 0.0, design[d0]["readout_phi"],
                     design[d0]["readout_tau"])
        zmap = {"z0": np.zeros(18)}
        zmap.update({d: results[d]["z"] for d in results})
        rep = analog_schedule_report(zmap, U0, design[d0]["observable"], ro, log)
        cert["analog_report"] = rep
        cert["gates"]["G13a_gap_model_reproduces_unmodulated_schedule"] = {
            "maximum_unitary_max_abs_error": (
                max(rep.get("gap_model_vs_pulser_maxabs", {}).values())
                if rep.get("gap_model_vs_pulser_maxabs")
                else None
            ),
            "model_gap_us": rep.get("model_gap_us"),
            "inferred_gap_us": rep.get("inferred_gap_us"),
            "threshold": 1e-9,
            "status": (
                "SUPPORTED"
                if rep.get("gap_model_ok", False)
                else ("NOT_SUPPORTED" if rep.get("compilable") else "NOT_TESTED")
            ),
            "pass": bool(rep.get("gap_model_ok", False)),
            "note": "This compares the gap-aware analytic schedule with "
                    "AnalogDevice's unmodulated sampled schedule. It is not a "
                    "finite-bandwidth endpoint test.",
        }
        if rep.get("compilable"):
            mod_endpoint_values = [
                rep[f"{direction}_eps_U_mod_vs_z0"]
                for direction in results
            ]
            max_mod_endpoint = max(mod_endpoint_values)
            cert["gates"]["G13b_modulated_endpoint_match"] = {
                "maximum_endpoint_infidelity": max_mod_endpoint,
                "threshold": 1e-11,
                "status": (
                    "SUPPORTED"
                    if max_mod_endpoint <= 1e-11
                    else "NOT_SUPPORTED"
                ),
                "pass": bool(max_mod_endpoint <= 1e-11),
                "note": "Evaluated on Pulser's modulation model, not on a "
                        "calibrated FRESNEL waveform.",
            }
            sig = abs(design[d0]["dP"][-1])
            coh = rep.get(f"{d0}_coherent_dP_from_modulation", float("inf"))
            log(f"\n  path signal at gamma={design[d0]['gammas'][-1]:g}: |dP| = {sig:.3e}")
            log(f"  modulation artefact at gamma=0 : |dP| = {coh:.3e}")
            log(f"  signal / artefact = {sig / max(coh, 1e-300):.3e}")
            cert["gates"]["G13c_signal_over_modulation_artefact"] = {
                "signal": sig, "artefact": coh,
                "ratio": float(sig / max(coh, 1e-300)),
                "threshold": 10.0,
                "pass": bool(sig / max(coh, 1e-300) >= 10.0),
                "note": "a FAIL means the control must be re-lifted on the "
                        "modulated schedule before any hardware run",
            }
        else:
            cert["gates"]["G13b_modulated_endpoint_match"] = {
                "pass": False,
                "status": "NOT_TESTED",
                "note": "AnalogDevice rejected at least one program before a "
                        "modulated endpoint comparison could be made.",
            }
    elif not args.no_pulser_check:
        cert["gates"]["G13a_gap_model_reproduces_unmodulated_schedule"] = {
            "pass": False,
            "status": "NOT_RUN",
            "note": "Run with --analog-report for the executed-schedule audit.",
        }
        cert["gates"]["G13b_modulated_endpoint_match"] = {
            "pass": False,
            "status": "NOT_RUN",
            "note": "Run with --analog-report for the Pulser modulation audit.",
        }

    # ---------------- 7. shot budget (the reality check) ---------------------
    log("\n" + "-" * 78)
    log("FEASIBILITY  shots required to resolve the split at 5 sigma")
    log("-" * 78)
    feas = {}
    for d in design:
        ro = Segment(design[d]["readout_omega"], 0.0, design[d]["readout_phi"],
                     design[d]["readout_tau"])
        segs_z = segments_of(results[d]["z"], readout=ro)
        segs_0 = segments_of(np.zeros(18), readout=ro)
        M = OBSERVABLES[design[d]["observable"]]
        g = design[d]["gammas"][-1]
        Ez, E0 = channel(segs_z, g), channel(segs_0, g)
        vz, v0 = observable_variance(M, Ez), observable_variance(M, E0)
        dp = abs(observable_value(M, Ez) - observable_value(M, E0))
        n5 = 25.0 * (vz + v0) / max(dp, 1e-300) ** 2
        feas[d] = {"gamma": float(g), "abs_dP": dp, "var_z": vz, "var_0": v0,
                   "shots_per_arm_5sigma": n5}
        log(f"  {d}: at gamma = {g:g} 1/us, |DeltaP| = {dp:.3e}, "
            f"Var = {vz:.4f}/{v0:.4f}")
        log(f"      shots per arm for 5 sigma: {n5:.3e}")
    cert["feasibility"] = feas
    worst = max(feas[d]["shots_per_arm_5sigma"] for d in feas)
    cert["gates"]["G12_shot_budget_practical"] = {
        "max_shots_per_arm_5sigma": worst,
        "threshold": 1e8,
        "pass": bool(worst <= 1e8),
        "note": "informational; a FAIL means the eps=0.04 fiber loop is too small "
                "for an observable-level experiment, not that the physics is wrong",
    }
    log(f"\n  [G12] worst-case shots per arm = {worst:.3e}  "
        f"({'FEASIBLE' if worst <= 1e8 else 'NOT FEASIBLE at this loop scale'})")

    # ---------------- 8. optional design sweep over loop scale ---------------
    if args.eps_sweep:
        log("\n" + "-" * 78)
        log("DESIGN SWEEP  loop scale eps -> response, slope, required shots")
        log("-" * 78)
        sweep = []
        gam_probe = (0.03, 0.10, 0.30)
        log(f"  columns: eps | eps_U | ||dK||_F/16 | M | chi | then per gamma in "
            f"{gam_probe}: |dP|, shots(5sigma), linearity error")
        for eps in (0.04, 0.08, 0.16, 0.32, 0.64, 1.00):
            hs = max(4 * eps / 120.0, 0.002)
            try:
                z, audit = m2_transport(U0, loop_vertices(eps, "CW"), hs)
            except Exception as exc:
                log(f"  eps={eps:5.2f}: lift failed ({exc})")
                continue
            eu = endpoint_infidelity(z, U0)
            om, dl, ph = controls_from_z(z)
            segs = segments_of(z)
            Gz_ = channel_derivative(segs)
            dG = Gz_ - G0
            dK = np.linalg.solve(ideal_channel(segs), Gz_) - K0
            RK = float(np.linalg.norm(dK, "fro") / 16.0)
            best, _, bound = scan_readout(dG, quick=True)
            chi = best[3]
            ro = Segment(best[0], 0.0, best[1], 0.100)
            M = OBSERVABLES[best[2]]
            segs_z = segments_of(z, readout=ro)
            segs_0 = segments_of(np.zeros(18), readout=ro)
            per_gamma = {}
            for g in gam_probe:
                Ez, E0e = channel(segs_z, g), channel(segs_0, g)
                dp = observable_value(M, Ez) - observable_value(M, E0e)
                n5 = 25.0 * (observable_variance(M, Ez) + observable_variance(M, E0e)) \
                    / max(abs(dp), 1e-300) ** 2
                lin = abs(abs(dp) - abs(chi * g)) / max(abs(dp), 1e-300)
                per_gamma[f"gamma_{g}"] = {"abs_dP": abs(dp),
                                           "shots_5sigma": n5,
                                           "linearity_rel_err": lin}
            rec = dict(eps=eps, h_s=hs, eps_U=eu, RK=RK, observable=best[2],
                       chi=float(chi), per_gamma=per_gamma,
                       max_Omega_over_2pi=float(np.max(om) / (2 * np.pi)),
                       min_Omega_over_2pi=float(np.min(om) / (2 * np.pi)),
                       z_norm=float(np.linalg.norm(z)))
            sweep.append(rec)
            tail = "  ".join(
                f"[g={g}: |dP|={per_gamma[f'gamma_{g}']['abs_dP']:.2e} "
                f"N={per_gamma[f'gamma_{g}']['shots_5sigma']:.1e} "
                f"lin={per_gamma[f'gamma_{g}']['linearity_rel_err']:.1%}]"
                for g in gam_probe)
            log(f"  eps={eps:5.2f} eU={eu:7.1e} ||dK||/16={RK:8.2e} {best[2]:5s} "
                f"chi={chi:+9.2e}  {tail}")
            device_limit_mhz = analog_device_max_amp_mhz()
            limit_text = (
                f"{device_limit_mhz:.2f} MHz"
                if device_limit_mhz is not None
                else "unavailable"
            )
            log(f"        Omega/2pi in [{np.min(om)/(2*np.pi):.2f},"
                f"{np.max(om)/(2*np.pi):.2f}] MHz  "
                f"(runtime AnalogDevice limit {limit_text})")
        # design recommendation
        DEV_LIMIT_MHZ = analog_device_max_amp_mhz()
        ok = [] if DEV_LIMIT_MHZ is None else [
            r for r in sweep
            if r["max_Omega_over_2pi"] <= DEV_LIMIT_MHZ
            and min(r["per_gamma"][f"gamma_{g}"]["shots_5sigma"]
                    for g in gam_probe) <= 1e7
        ]
        over = [] if DEV_LIMIT_MHZ is None else [
            r for r in sweep
            if r["max_Omega_over_2pi"] > DEV_LIMIT_MHZ
            and min(r["per_gamma"][f"gamma_{g}"]["shots_5sigma"]
                    for g in gam_probe) <= 1e7
        ]
        log("")
        if DEV_LIMIT_MHZ is None:
            log("  RECOMMENDATION: AnalogDevice amplitude limits were unavailable "
                "at runtime, so no device-feasibility recommendation is issued.")
        elif ok:
            b = min(ok, key=lambda r: min(
                r["per_gamma"][f"gamma_{g}"]["shots_5sigma"] for g in gam_probe))
            log(f"  RECOMMENDATION: eps >= {b['eps']:g} with gamma >= 0.1 1/us brings "
                f"the split within ~1e7 shots per arm, with Omega inside the "
                f"AnalogDevice limit.")
        elif over:
            b = min(over, key=lambda r: r["eps"])
            log(f"  RECOMMENDATION: the shot budget only becomes reasonable at "
                f"eps >= {b['eps']:g} (>=1e7 -> "
                f"{min(b['per_gamma'][f'gamma_{g}']['shots_5sigma'] for g in gam_probe):.1e}"
                f" shots), but that lift drives Omega/2pi to "
                f"{b['max_Omega_over_2pi']:.2f} MHz, above the {DEV_LIMIT_MHZ:.2f} MHz "
                f"AnalogDevice limit. Rescale the reference table first "
                f"(--omega-scale ~0.8) and re-run the sweep; the fiber structure is "
                f"unchanged by a uniform rescale of Omega only up to the fixed C6 "
                f"term, so the sweep must actually be redone, not extrapolated.")
        else:
            log("  RECOMMENDATION: no (eps, gamma) in the scanned box reaches 1e7 "
                "shots per arm. The observable-level experiment is not feasible "
                "for this reference control without a larger fiber excursion or a "
                "stronger controllable noise lever.")
        log("  Note: at gamma >= 0.1 the O(gamma^2) term is no longer negligible; "
            "the parameter-free LINEAR prediction must then be replaced by a "
            "second-order one, or gamma extrapolated to zero from the scan.")
        cert["eps_sweep"] = sweep

    # ---------------- outputs ------------------------------------------------
    with open(os.path.join(args.outdir, "ep_obs_gamma_scan.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    ctrl = {"z0": np.zeros(18).tolist()}
    for d in results:
        ctrl[d] = results[d]["z"].tolist()
    with open(os.path.join(args.outdir, "ep_obs_controls.json"), "w") as fh:
        json.dump(ctrl, fh, indent=2)

    for d in design:
        for k in ("dG",):
            design[d].pop(k, None)
        for k in ("gammas", "dP", "DE", "pred", "resid", "Pz", "P0"):
            design[d][k] = np.asarray(design[d][k]).tolist()
    cert["design"] = design
    cert["lift_audit"] = {d: results[d]["audit"] for d in results}

    def gate_pass(name):
        gate = cert["gates"].get(name)
        return bool(isinstance(gate, dict) and gate.get("pass", False))

    model_gate_names = [
        "H1_phase_convention_bookkeeping",
        "G1_endpoint_infidelity",
        "G2_residual_norm",
        "G3_rank_audit",
        "G4_distinct_controls",
        "G6_delta_common",
        "G7_signal_exponent",
        "G8_residual_exponent",
        "G9_max_relative_error",
        "G10_signal_over_zero_noise_floor",
    ]
    if "G14_transport_step_halving" in cert["gates"]:
        model_gate_names.append("G14_transport_step_halving")
    model_supported = all(gate_pass(name) for name in model_gate_names)

    translation_gate_names = [
        "H0_hamiltonian_matches_pulser",
        "G5_mockdevice_unmodulated_translation",
        "G11a_model_matches_pulser",
    ]
    translation_ran = all(
        name in cert["gates"] for name in translation_gate_names
    )
    translation_supported = bool(
        translation_ran
        and all(gate_pass(name) for name in translation_gate_names)
    )
    qutip_status = cert["gates"].get(
        "G11b_qutip_solver_resolution", {}
    ).get("status", "NOT_RUN")
    analog_program_status = cert["gates"].get(
        "G13_analogdevice_program_compilable", {}
    ).get("status", "NOT_RUN")
    gap_model_status = cert["gates"].get(
        "G13a_gap_model_reproduces_unmodulated_schedule", {}
    ).get("status", "NOT_RUN")
    modulated_endpoint_status = cert["gates"].get(
        "G13b_modulated_endpoint_match", {}
    ).get("status", "NOT_RUN")
    shot_status = (
        "SUPPORTED"
        if gate_pass("G12_shot_budget_practical")
        else "NOT_SUPPORTED"
    )
    experiment_ready = bool(
        model_supported
        and translation_supported
        and qutip_status == "RESOLVED"
        and analog_program_status == "SUPPORTED"
        and gap_model_status == "SUPPORTED"
        and modulated_endpoint_status == "SUPPORTED"
        and gate_pass("G13c_signal_over_modulation_artefact")
        and shot_status == "SUPPORTED"
    )
    cert["status_summary"] = {
        "model_observable_bridge": (
            "SUPPORTED" if model_supported else "NOT_SUPPORTED"
        ),
        "mock_waveform_translation": (
            "SUPPORTED"
            if translation_supported
            else ("NOT_SUPPORTED" if translation_ran else "NOT_RUN")
        ),
        "qutip_solver_resolution": qutip_status,
        "analogdevice_program_compilation": analog_program_status,
        "executed_schedule_gap_match": gap_model_status,
        "modulated_endpoint_match": modulated_endpoint_status,
        "shot_feasibility": shot_status,
        "overall_operational_status": (
            "EXPERIMENT_READY" if experiment_ready else "EXPERIMENT_NOT_READY"
        ),
    }
    if analog_program_status == "NOT_RUN":
        stage2_status = "DEVICE_AUDIT_NOT_RUN"
        stage2_next = (
            "Run with Pulser checks enabled; the local-only result cannot decide "
            "the executed-schedule or modulation branch."
        )
    elif analog_program_status != "SUPPORTED":
        stage2_status = "DEVICE_PROGRAM_NOT_ACCEPTED"
        stage2_next = (
            "Adjust the amplitude/timing parameters before any modulation-aware "
            "lift."
        )
    elif gap_model_status != "SUPPORTED":
        stage2_status = "EXECUTED_GAP_MODEL_NOT_REPRODUCED"
        stage2_next = (
            "Inspect the sampled phase-jump blocks; do not attribute the "
            "remaining mismatch to finite-bandwidth modulation yet."
        )
    elif modulated_endpoint_status != "SUPPORTED":
        stage2_status = "GAP_REVALIDATED_MODULATION_REMAINS"
        stage2_next = (
            "Re-lift the endpoint-equivalent pair directly on Pulser's "
            "modulated sampled waveform; do not submit this pair to cloud/QPU."
        )
    else:
        stage2_status = "MODULATED_ENDPOINT_EQUIVALENCE_SUPPORTED"
        stage2_next = (
            "Proceed to a full-precision lift and observable/shot optimization "
            "before considering cloud execution."
        )
    cert["stage2_decision"] = {
        "status": stage2_status,
        "next_step": stage2_next,
    }
    cert["scientific_status"] = (
        "MODEL_OBSERVABLE_BRIDGE_SUPPORTED"
        if model_supported
        else "MODEL_OBSERVABLE_BRIDGE_NOT_SUPPORTED"
    )
    cert["claim_boundary"] = (
        "The exact two-atom coherent/Lindblad model and the declared native "
        "|gg> plus global-readout family are tested. MockDevice sampled-program "
        "translation, Qutip solver resolution, AnalogDevice program acceptance, "
        "modulated endpoint survival, and shot feasibility are reported as "
        "separate evidence layers. No cloud-emulator or QPU evidence is claimed."
    )
    with open(os.path.join(args.outdir, "ep_obs_certificate.json"), "w") as fh:
        json.dump(cert, fh, indent=2, default=float)

    log("\n" + "=" * 78)
    log("GATE SUMMARY")
    log("=" * 78)
    for name, val in cert["gates"].items():
        if isinstance(val, dict) and "pass" in val:
            declared = val.get("status")
            label = (
                declared
                if declared in {
                    "NOT_RUN", "NOT_TESTED", "NOT_RESOLVED", "REJECTED"
                }
                else ("PASS" if val["pass"] else "FAIL")
            )
            log(f"  {label:12s}  {name}")
    log("\nSTATUS SUMMARY")
    for name, status in cert["status_summary"].items():
        log(f"  {name:34s}: {status}")
    log("\nSTAGE-2 DECISION")
    log(f"  status                            : {stage2_status}")
    log(f"  next                              : {stage2_next}")
    log("\nwritten: ep_obs_certificate.json, ep_obs_gamma_scan.csv, "
        "ep_obs_controls.json, ep_obs_run.log")

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for d, c in zip(design, ("tab:blue", "tab:orange")):
            g = np.array(design[d]["gammas"])
            dP = np.abs(np.array(design[d]["dP"]))
            pr = np.abs(np.array(design[d]["pred"]))
            rs = np.array(design[d]["resid"])
            m = g > 0
            ax[0].loglog(g[m], dP[m], "o-", color=c,
                         label=f"{d} exact |$\\Delta P$| ({design[d]['observable']})")
            ax[0].loglog(g[m], pr[m], "--", color=c, label=f"{d} $\\gamma\\chi$ (no fit)")
            ax[1].loglog(g[m], rs[m], "s-", color=c,
                         label=f"{d} residual, slope={design[d]['p_resid']:.3f}")
        ax[0].set_xlabel(r"$\gamma$  [$\mu s^{-1}$]")
        ax[0].set_ylabel(r"$|\Delta P|$")
        ax[0].set_title("observable-level split, parameter-free prediction")
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=.3, which="both")
        ax[1].set_xlabel(r"$\gamma$  [$\mu s^{-1}$]")
        ax[1].set_ylabel(r"$|\Delta P - \gamma\chi|$")
        ax[1].set_title("residual (expected slope 2)")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "ep_obs_prediction.png"), dpi=160)
        log("written: ep_obs_prediction.png")
    except Exception as exc:
        log(f"(figure skipped: {exc})")

    logfile.close()


def run(**kwargs):
    """Notebook entry point.

        import ep_obs_stage3_modulated_relift_v1_1 as ep
        ep.run(quick=True, outdir="out")

    Booleans map to flags, everything else to "--key value". Underscores in the
    keyword become dashes, so eps_sweep -> --eps-sweep.
    """
    argv = []
    for key, val in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        else:
            argv += [flag, str(val)]
    return stage3_main(argv)


def stage3_modulated_blocks(z: np.ndarray):
    """AnalogDevice's actually sampled, modulated drive waveform for z."""
    seq = build_pulser_sequence(segments_of(z), "analog")
    return compiled_blocks(seq, modulation=True)


def stage3_configure_analog_interaction():
    """Use the interaction coefficient declared by the active AnalogDevice.

    The earlier model-level scripts used MockDevice's coefficient. Stage 3
    samples AnalogDevice waveforms, so silently retaining a different C6 would
    mix two device models in one evolution.
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


def stage3_modulated_unitary(z: np.ndarray) -> np.ndarray:
    return unitary_of_blocks(stage3_modulated_blocks(z))


def stage3_phase_aligned_residual(U: np.ndarray, Ut: np.ndarray) -> np.ndarray:
    """Real residual after removing the physically irrelevant global phase."""
    overlap = np.trace(Ut.conj().T @ U)
    phase = np.angle(overlap)
    R = np.exp(-1j * phase) * U - Ut
    return np.concatenate([R.real.ravel(), R.imag.ravel()])


def stage3_endpoint_infidelity(U: np.ndarray, Ut: np.ndarray) -> float:
    return float(abs(1.0 - abs(np.trace(Ut.conj().T @ U)) ** 2 / DIM**2))


def stage3_modulated_jacobian(
    z: np.ndarray,
    U_target: np.ndarray,
    h: float,
):
    """Central-difference Jacobian of the modulated full-unitary endpoint."""
    z = np.asarray(z, float)
    Q = np.empty((2 * DIM * DIM, z.size), dtype=float)
    for j in range(z.size):
        e = np.zeros_like(z)
        e[j] = h
        rp = stage3_phase_aligned_residual(
            stage3_modulated_unitary(z + e), U_target
        )
        rm = stage3_phase_aligned_residual(
            stage3_modulated_unitary(z - e), U_target
        )
        Q[:, j] = (rp - rm) / (2.0 * h)
    return Q


def stage3_modulated_relift(
    z_seed: np.ndarray,
    U_target: np.ndarray,
    *,
    fd_step: float,
    max_iters: int,
    residual_tol: float,
    rcond: float,
    log,
):
    """Project a nontrivial gap-aware seed onto the modulated endpoint fiber.

    The correction is local and minimum norm. A backtracking line search keeps
    every accepted step close to the seed and prevents a failed Newton step from
    being silently accepted.
    """
    z = np.asarray(z_seed, float).copy()
    history = []
    rank_set = set()
    for iteration in range(max_iters):
        U = stage3_modulated_unitary(z)
        r = stage3_phase_aligned_residual(U, U_target)
        rn = float(np.linalg.norm(r))
        eu = stage3_endpoint_infidelity(U, U_target)
        if rn <= residual_tol:
            history.append(
                {
                    "iteration": iteration,
                    "residual_norm": rn,
                    "endpoint_infidelity": eu,
                    "accepted_alpha": 0.0,
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

        # The endpoint defect is small; a large correction indicates a local
        # Jacobian failure and must be damped rather than trusted.
        dz_norm = float(np.linalg.norm(dz))
        if dz_norm > 0.20:
            dz *= 0.20 / dz_norm
            dz_norm = 0.20

        accepted = False
        best = None
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial = z + alpha * dz
            try:
                Ut = stage3_modulated_unitary(trial)
            except Exception:
                continue
            rt = stage3_phase_aligned_residual(Ut, U_target)
            rnt = float(np.linalg.norm(rt))
            if best is None or rnt < best[0]:
                best = (rnt, alpha, trial, Ut)
            if rnt < rn:
                accepted = True
                break
        if not accepted and best is not None and best[0] < rn:
            rnt, alpha, trial, Ut = best
            accepted = True
        if not accepted:
            history.append(
                {
                    "iteration": iteration,
                    "residual_norm": rn,
                    "endpoint_infidelity": eu,
                    "jacobian_rank": rank,
                    "step_norm": dz_norm,
                    "accepted_alpha": None,
                    "status": "LINE_SEARCH_STALLED",
                }
            )
            break

        z = trial
        eut = stage3_endpoint_infidelity(Ut, U_target)
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
            f"|r|={rn:.3e}->{rnt:.3e} epsU={eu:.3e}->{eut:.3e} "
            f"alpha={alpha:g}"
        )

    U_final = stage3_modulated_unitary(z)
    r_final = stage3_phase_aligned_residual(U_final, U_target)
    return z, U_final, {
        "history": history,
        "rank_set": sorted(rank_set),
        "final_residual_norm": float(np.linalg.norm(r_final)),
        "final_endpoint_infidelity": stage3_endpoint_infidelity(
            U_final, U_target
        ),
        "correction_norm": float(np.linalg.norm(z - z_seed)),
        "seed_norm": float(np.linalg.norm(z_seed)),
        "final_norm": float(np.linalg.norm(z)),
    }


def stage3_channel_and_derivative(blocks):
    """O(n) exact ideal channel and dE/dgamma at gamma=0.

    This recurrence replaces the O(n^2) prefix/suffix construction, which is
    impractical for a continuously modulated 1 ns waveform.
    """
    E = np.eye(DIM2, dtype=complex)
    G = np.zeros((DIM2, DIM2), dtype=complex)
    for sg in blocks:
        A = sup_hamiltonian(
            hamiltonian(sg.omega, sg.delta, sg.phi)
        ) * sg.tau
        P, Lf = expm_frechet(A, D_SUP * sg.tau, compute_expm=True)
        G = P @ G + Lf @ E
        E = P @ E
    return E, G


def stage3_channel(blocks, gamma: float):
    E = np.eye(DIM2, dtype=complex)
    for sg in blocks:
        L0 = sup_hamiltonian(
            hamiltonian(sg.omega, sg.delta, sg.phi)
        )
        E = expm((L0 + gamma * D_SUP) * sg.tau) @ E
    return E


def stage3_best_native_observable(dG: np.ndarray):
    """Select a computational-basis observable without an appended pulse."""
    drho = unvec(dG @ vec(RHO0))
    candidates = []
    for name, M in OBSERVABLES.items():
        chi = float(np.real(np.trace(M @ drho)))
        candidates.append((name, chi))
    name, chi = max(candidates, key=lambda item: abs(item[1]))
    herm = (drho + drho.conj().T) / 2
    all_observable_bound = float(np.sum(np.abs(np.linalg.eigvalsh(herm))))
    return name, chi, candidates, all_observable_bound


def stage3_shots_5sigma(M, Ez, E0, dP):
    if abs(dP) < 1e-300:
        return float("inf")
    vz = observable_variance(M, Ez)
    v0 = observable_variance(M, E0)
    return float(25.0 * (vz + v0) / (dP * dP))


def stage3_main(argv=None):
    ap = argparse.ArgumentParser(prog="ep_obs_stage3_modulated_relift_v1_1.py")
    ap.add_argument(
        "--quick",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="coarse M2 seed transport (default: enabled)",
    )
    ap.add_argument("--eps", type=float, default=0.040)
    ap.add_argument("--hs", type=float, default=0.002)
    ap.add_argument("--gap", type=float, default=0.340)
    ap.add_argument("--omega-scale", type=float, default=0.80)
    ap.add_argument("--phase-sign", type=float, default=1.0, choices=[1.0, -1.0])
    ap.add_argument("--fd-step", type=float, default=2e-5)
    ap.add_argument("--relift-iters", type=int, default=6)
    ap.add_argument("--endpoint-residual-tol", type=float, default=2e-9)
    ap.add_argument("--endpoint-infidelity-tol", type=float, default=1e-11)
    ap.add_argument("--rcond", type=float, default=1e-6)
    ap.add_argument("--shot-threshold", type=float, default=1e7)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--source-path", default=None)
    if argv is None:
        argv = [] if _in_notebook() else sys.argv[1:]
    args, ignored = ap.parse_known_args(list(argv))
    if ignored:
        print(f"[note] ignoring arguments not belonging to this script: {ignored}")
    if args.outdir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.outdir = f"ep_obs_stage3_modulated_relift_{stamp}"

    global _SOURCE_PATH_OVERRIDE
    if args.source_path:
        _SOURCE_PATH_OVERRIDE = args.source_path
    MODEL["gap_tau"] = float(args.gap)
    MODEL["omega_scale"] = float(args.omega_scale)
    MODEL["phase_sign"] = float(args.phase_sign)
    os.makedirs(args.outdir, exist_ok=True)
    logfile = open(os.path.join(args.outdir, "ep_obs_stage3_run.log"), "w")

    def log(msg=""):
        print(msg)
        logfile.write(str(msg) + "\n")
        logfile.flush()

    h_s = 0.008 if args.quick else args.hs
    cert = {
        "version": VERSION,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_sha256": self_sha256(),
        "packages": package_versions(),
        "args": vars(args),
        "model": {
            "atoms": 2,
            "gap_tau_us": MODEL["gap_tau"],
            "omega_scale": MODEL["omega_scale"],
            "phase_sign": MODEL["phase_sign"],
            "device_model": "Pulser AnalogDevice",
            "waveform": "sample(..., modulation=True), 1 ns grid",
            "noise": "local occupation dephasing",
            "readout": "native computational basis; no appended pulse",
        },
        "gates": {},
    }

    log("=" * 88)
    log(f"{VERSION}  MODULATION-AWARE ENDPOINT RE-LIFT")
    log("=" * 88)
    log(f"UTC={cert['utc']}")
    log(
        f"omega_scale={MODEL['omega_scale']:.3f} | gap={MODEL['gap_tau']:.3f} us "
        f"| fd_step={args.fd_step:.1e} | cloud access=none"
    )
    for key, value in cert["packages"].items():
        log(f"  {key:20s} {value}")

    try:
        from pulser.devices import AnalogDevice
        _ = AnalogDevice.channels["rydberg_global"]
    except Exception as exc:
        log(f"\nFATAL: Pulser AnalogDevice unavailable: {type(exc).__name__}: {exc}")
        cert["scientific_status"] = "PULSER_ANALOGDEVICE_UNAVAILABLE"
        with open(
            os.path.join(args.outdir, "ep_obs_stage3_certificate.json"), "w"
        ) as fh:
            json.dump(cert, fh, indent=2, default=float)
        logfile.close()
        return 1

    h0_err, gg_ok = pulser_hamiltonian_probe()
    cert["gates"]["H0_hamiltonian_matches_pulser"] = {
        "max_abs_error": h0_err,
        "initial_state_is_gg": gg_ok,
        "threshold": 1e-9,
        "pass": bool(h0_err < 1e-9 and gg_ok),
        "scope": "drive-sign/basis convention probe on MockDevice",
    }
    log(
        f"\n[H0] drive convention == Pulser/MockDevice: max|dH|={h0_err:.3e} "
        f"init=|gg>:{gg_ok}"
    )
    interaction_audit = stage3_configure_analog_interaction()
    cert["analog_interaction_audit"] = interaction_audit
    cert["model"]["C6"] = interaction_audit["AnalogDevice_C6"]
    cert["model"]["U_int"] = interaction_audit[
        "U_int_at_declared_separation"
    ]
    cert["gates"]["H0b_analogdevice_interaction_loaded"] = {
        **interaction_audit,
        "pass": bool(
            np.isfinite(interaction_audit["AnalogDevice_C6"])
            and interaction_audit["AnalogDevice_C6"] > 0.0
        ),
    }
    log(
        "     AnalogDevice C6 loaded at runtime: "
        f"{interaction_audit['AnalogDevice_C6']:.8g} "
        f"(fallback difference "
        f"{interaction_audit['absolute_difference_before_override']:.3e})"
    )
    foundation_pass = bool(
        cert["gates"]["H0_hamiltonian_matches_pulser"]["pass"]
        and cert["gates"]["H0b_analogdevice_interaction_loaded"]["pass"]
    )
    if not foundation_pass:
        cert["scientific_status"] = "DEVICE_HAMILTONIAN_FOUNDATION_NOT_RESOLVED"
        cert["next_step"] = (
            "Resolve the Pulser basis/sign or AnalogDevice interaction metadata "
            "before constructing endpoint-equivalent paths."
        )
        with open(
            os.path.join(args.outdir, "ep_obs_stage3_certificate.json"), "w"
        ) as fh:
            json.dump(cert, fh, indent=2, default=float)
        log(f"\nGLOBAL VERDICT: {cert['scientific_status']}")
        logfile.close()
        return 2

    # 1. Reproduce the Stage-2 nontrivial seeds on the gap-aware schedule.
    log("\n" + "-" * 88)
    log("STEP 1  reproduce gap-aware M2 seed paths")
    log("-" * 88)
    U0_gap = unitary_of_z(np.zeros(18))
    seeds = {}
    for direction in ("CW", "CCW"):
        t0 = time.time()
        z_seed, audit = m2_transport(
            U0_gap,
            loop_vertices(args.eps, direction),
            h_s,
            rcond=args.rcond,
        )
        gap_eu = endpoint_infidelity(z_seed, U0_gap)
        seeds[direction] = {
            "z": z_seed,
            "audit": audit,
            "gap_endpoint_infidelity": gap_eu,
        }
        log(
            f"  {direction}: steps={audit['n_steps']} ranks={audit['rank_set']} "
            f"gap-epsU={gap_eu:.3e} |z|={np.linalg.norm(z_seed):.3e} "
            f"({time.time()-t0:.1f}s)"
        )

    # 2. Compute the modulated target and project each seed onto its endpoint.
    log("\n" + "-" * 88)
    log("STEP 2  local projection onto the modulated endpoint fiber")
    log("-" * 88)
    z0 = np.zeros(18)
    U0_mod = stage3_modulated_unitary(z0)
    ref_blocks = stage3_modulated_blocks(z0)
    corrected = {}
    for direction in ("CW", "CCW"):
        z_seed = seeds[direction]["z"]
        U_seed = stage3_modulated_unitary(z_seed)
        seed_eu = stage3_endpoint_infidelity(U_seed, U0_mod)
        log(f"\n  {direction}: seed modulated epsU={seed_eu:.3e}")
        z, U, audit = stage3_modulated_relift(
            z_seed,
            U0_mod,
            fd_step=args.fd_step,
            max_iters=args.relift_iters,
            residual_tol=args.endpoint_residual_tol,
            rcond=args.rcond,
            log=log,
        )
        blocks = stage3_modulated_blocks(z)
        U_resampled = unitary_of_blocks(blocks)
        resampling_unitary_error = float(np.max(np.abs(U_resampled - U)))
        resampled_endpoint_infidelity = stage3_endpoint_infidelity(
            U_resampled, U0_mod
        )
        resampled_residual_norm = float(
            np.linalg.norm(
                stage3_phase_aligned_residual(U_resampled, U0_mod)
            )
        )
        audit["independent_resampling_unitary_max_error"] = (
            resampling_unitary_error
        )
        audit["resampled_endpoint_infidelity"] = (
            resampled_endpoint_infidelity
        )
        audit["resampled_endpoint_residual_norm"] = resampled_residual_norm
        corrected[direction] = {
            "z": z,
            "U": U_resampled,
            "blocks": blocks,
            "audit": audit,
            "seed_modulated_endpoint_infidelity": seed_eu,
        }
        log(
            f"    final: epsU={audit['final_endpoint_infidelity']:.3e} "
            f"|r|={audit['final_residual_norm']:.3e} "
            f"|dz|={audit['correction_norm']:.3e} "
            f"|z|={audit['final_norm']:.3e} blocks={len(blocks)}\n"
            f"           independent resample: epsU="
            f"{resampled_endpoint_infidelity:.3e} "
            f"|r|={resampled_residual_norm:.3e} "
            f"max|dU|={resampling_unitary_error:.3e}"
        )

    endpoint_pass = all(
        max(
            corrected[d]["audit"]["final_endpoint_infidelity"],
            corrected[d]["audit"]["resampled_endpoint_infidelity"],
        )
        <= args.endpoint_infidelity_tol
        and max(
            corrected[d]["audit"]["final_residual_norm"],
            corrected[d]["audit"]["resampled_endpoint_residual_norm"],
        )
        <= args.endpoint_residual_tol
        for d in corrected
    )
    resampling_pass = all(
        corrected[d]["audit"]["independent_resampling_unitary_max_error"]
        <= 1e-12
        for d in corrected
    )
    rank_pass = all(
        corrected[d]["audit"]["rank_set"] in ([8], [])
        for d in corrected
    )
    distinct_pass = all(
        corrected[d]["audit"]["final_norm"] > 1e-5
        and corrected[d]["audit"]["correction_norm"]
        < 0.5 * corrected[d]["audit"]["seed_norm"]
        for d in corrected
    )
    reference_duration = float(sum(sg.tau for sg in ref_blocks))
    corrected_durations = {
        d: float(sum(sg.tau for sg in corrected[d]["blocks"]))
        for d in corrected
    }
    duration_pass = all(
        abs(corrected_durations[d] - reference_duration) <= 1e-9
        for d in corrected
    )
    cert["gates"]["G1_modulated_endpoint_match"] = {
        d: {
            "solver_endpoint_infidelity": corrected[d]["audit"][
                "final_endpoint_infidelity"
            ],
            "resampled_endpoint_infidelity": corrected[d]["audit"][
                "resampled_endpoint_infidelity"
            ],
            "solver_residual_norm": corrected[d]["audit"][
                "final_residual_norm"
            ],
            "resampled_residual_norm": corrected[d]["audit"][
                "resampled_endpoint_residual_norm"
            ],
        }
        for d in corrected
    }
    cert["gates"]["G1_modulated_endpoint_match"].update(
        {
            "threshold": args.endpoint_infidelity_tol,
            "residual_threshold": args.endpoint_residual_tol,
            "pass": endpoint_pass,
        }
    )
    cert["gates"]["G1b_modulated_resampling_deterministic"] = {
        d: corrected[d]["audit"][
            "independent_resampling_unitary_max_error"
        ]
        for d in corrected
    }
    cert["gates"]["G1b_modulated_resampling_deterministic"].update(
        {"threshold": 1e-12, "pass": resampling_pass}
    )
    cert["gates"]["G2_modulated_endpoint_rank"] = {
        d: corrected[d]["audit"]["rank_set"] for d in corrected
    }
    cert["gates"]["G2_modulated_endpoint_rank"]["pass"] = rank_pass
    cert["gates"]["G3_nontrivial_path_survives_projection"] = {
        d: {
            "seed_norm": corrected[d]["audit"]["seed_norm"],
            "correction_norm": corrected[d]["audit"]["correction_norm"],
            "final_norm": corrected[d]["audit"]["final_norm"],
        }
        for d in corrected
    }
    cert["gates"]["G3_nontrivial_path_survives_projection"][
        "pass"
    ] = distinct_pass
    cert["gates"]["G3b_modulated_duration_match"] = {
        "reference_duration_us": reference_duration,
        "candidate_duration_us": corrected_durations,
        "threshold_us": 1e-9,
        "pass": duration_pass,
    }
    log(
        f"  duration reference/CW/CCW = {reference_duration:.6f}/"
        f"{corrected_durations['CW']:.6f}/"
        f"{corrected_durations['CCW']:.6f} us"
    )

    if not (endpoint_pass and resampling_pass):
        cert["scientific_status"] = "MODULATED_ENDPOINT_RELIFT_NOT_RESOLVED"
        cert["next_step"] = (
            "If the endpoint gate failed, increase --relift-iters or adjust "
            "--fd-step. If only the resampling gate failed, inspect Pulser "
            "sampling determinism. Do not evaluate the noise-response claim."
        )
        cert["seeds"] = {
            d: {
                "gap_endpoint_infidelity": seeds[d][
                    "gap_endpoint_infidelity"
                ],
                "audit": seeds[d]["audit"],
            }
            for d in seeds
        }
        cert["relift"] = {d: corrected[d]["audit"] for d in corrected}
        with open(
            os.path.join(args.outdir, "ep_obs_stage3_certificate.json"), "w"
        ) as fh:
            json.dump(cert, fh, indent=2, default=float)
        log("\nGLOBAL VERDICT: MODULATED_ENDPOINT_RELIFT_NOT_RESOLVED")
        log(f"next: {cert['next_step']}")
        logfile.close()
        return 2

    # 3. Compute exact response directly on the modulated waveform.
    log("\n" + "-" * 88)
    log("STEP 3  native-observable response on the modulated waveform")
    log("-" * 88)
    t0 = time.time()
    E0_ideal, G0 = stage3_channel_and_derivative(ref_blocks)
    log(
        f"  reference blocks={len(ref_blocks)} "
        f"ideal-unitary/channel consistency="
        f"{np.max(np.abs(E0_ideal-sup_unitary(U0_mod))):.3e}"
    )
    response = {}
    for direction in ("CW", "CCW"):
        Ez_ideal, Gz = stage3_channel_and_derivative(
            corrected[direction]["blocks"]
        )
        dG = Gz - G0
        obs_name, chi, candidates, bound = stage3_best_native_observable(dG)
        response[direction] = {
            "E_ideal": Ez_ideal,
            "G": Gz,
            "dG": dG,
            "observable": obs_name,
            "chi": chi,
            "all_native_chi": {
                name: value for name, value in candidates
            },
            "unrestricted_observable_bound": bound,
            "normalized_delta_G": float(np.linalg.norm(dG, "fro") / 16.0),
            "ideal_channel_endpoint_error": float(
                np.max(np.abs(Ez_ideal - E0_ideal))
            ),
        }
        log(
            f"  {direction}: ||dG||F/16="
            f"{response[direction]['normalized_delta_G']:.3e} "
            f"native={obs_name} chi={chi:+.3e} "
            f"all-M bound={bound:.3e}"
        )
    log(f"  derivative propagation elapsed={time.time()-t0:.1f}s")

    response_nonzero = all(
        abs(response[d]["chi"]) > 1e-10 for d in response
    )
    cert["gates"]["G4_modulated_path_response_nonzero"] = {
        d: {
            "normalized_delta_G": response[d]["normalized_delta_G"],
            "native_observable": response[d]["observable"],
            "chi": response[d]["chi"],
        }
        for d in response
    }
    cert["gates"]["G4_modulated_path_response_nonzero"][
        "threshold_abs_chi"
    ] = 1e-10
    cert["gates"]["G4_modulated_path_response_nonzero"][
        "pass"
    ] = response_nonzero

    # 4. Declared weak window and one separate stress point.
    log("\n" + "-" * 88)
    log("STEP 4  weak-noise prediction and separate stress point")
    log("-" * 88)
    gammas = np.array(
        [0.0, 0.001875, 0.003750, 0.007500, 0.015000, 0.030000]
    )
    weak_max = 0.015
    ref_channels = {0.0: E0_ideal}
    for gamma in gammas[1:]:
        ref_channels[float(gamma)] = stage3_channel(ref_blocks, float(gamma))

    csv_rows = []
    for direction in ("CW", "CCW"):
        obs_name = response[direction]["observable"]
        M = OBSERVABLES[obs_name]
        chi = response[direction]["chi"]
        dP, pred, residuals, shots = [], [], [], []
        channels_z = {0.0: response[direction]["E_ideal"]}
        for gamma in gammas[1:]:
            channels_z[float(gamma)] = stage3_channel(
                corrected[direction]["blocks"], float(gamma)
            )
        log(f"\n  === {direction} / {obs_name} / chi={chi:+.7e} ===")
        log(
            f"  {'gamma':>9s} {'DeltaP':>13s} {'gamma*chi':>13s} "
            f"{'residual':>12s} {'rel.err':>9s} {'shots/arm':>13s}"
        )
        for gamma in gammas:
            g = float(gamma)
            Ez = channels_z[g]
            E0 = ref_channels[g]
            delta_p = observable_value(M, Ez) - observable_value(M, E0)
            prediction = g * chi
            residual_value = abs(delta_p - prediction)
            rel = (
                residual_value / abs(delta_p)
                if abs(delta_p) > 1e-300 and g > 0
                else float("nan")
            )
            nshots = (
                stage3_shots_5sigma(M, Ez, E0, delta_p)
                if g > 0
                else float("inf")
            )
            dP.append(delta_p)
            pred.append(prediction)
            residuals.append(residual_value)
            shots.append(nshots)
            csv_rows.append(
                {
                    "direction": direction,
                    "observable": obs_name,
                    "gamma": g,
                    "DeltaP": delta_p,
                    "prediction": prediction,
                    "residual": residual_value,
                    "relative_error": rel,
                    "shots_per_arm_5sigma": nshots,
                    "regime": (
                        "zero"
                        if g == 0
                        else ("weak" if g <= weak_max else "stress")
                    ),
                }
            )
            log(
                f"  {g:9.6f} {delta_p:+13.6e} {prediction:+13.6e} "
                f"{residual_value:12.4e} "
                f"{'' if g == 0 else f'{rel:9.3%}'} "
                f"{'' if g == 0 else f'{nshots:13.3e}'}"
            )
        dP = np.asarray(dP)
        pred = np.asarray(pred)
        residuals = np.asarray(residuals)
        shots = np.asarray(shots)
        weak = (gammas > 0) & (gammas <= weak_max)
        weak_rel = residuals[weak] / np.maximum(np.abs(dP[weak]), 1e-300)
        response[direction].update(
            {
                "gammas": gammas.tolist(),
                "DeltaP": dP.tolist(),
                "prediction": pred.tolist(),
                "residual": residuals.tolist(),
                "shots_per_arm_5sigma": [
                    None if not np.isfinite(value) else float(value)
                    for value in shots
                ],
                "weak_signal_exponent": loglog_slope(
                    gammas[weak], np.abs(dP[weak])
                ),
                "weak_residual_exponent": loglog_slope(
                    gammas[weak], residuals[weak]
                ),
                "weak_max_relative_error": float(np.max(weak_rel)),
                "zero_noise_floor": float(abs(dP[0])),
                "stress_relative_error": float(
                    residuals[-1] / max(abs(dP[-1]), 1e-300)
                ),
                "weak_best_shots": float(np.min(shots[weak])),
                "stress_shots": float(shots[-1]),
            }
        )
        log(
            f"    weak exponents signal/residual="
            f"{response[direction]['weak_signal_exponent']:.4f}/"
            f"{response[direction]['weak_residual_exponent']:.4f}; "
            f"weak max rel.err="
            f"{response[direction]['weak_max_relative_error']:.2%}"
        )

    weak_law_pass = all(
        0.94 <= response[d]["weak_signal_exponent"] <= 1.06
        and 1.80 <= response[d]["weak_residual_exponent"] <= 2.20
        and response[d]["weak_max_relative_error"] <= 0.03
        and response[d]["zero_noise_floor"] <= 1e-9
        for d in response
    )
    shot_pass = all(
        response[d]["weak_best_shots"] <= args.shot_threshold
        for d in response
    )
    cert["gates"]["G5_weak_noise_parameter_free_prediction"] = {
        d: {
            "signal_exponent": response[d]["weak_signal_exponent"],
            "residual_exponent": response[d]["weak_residual_exponent"],
            "maximum_relative_error": response[d][
                "weak_max_relative_error"
            ],
            "zero_noise_floor": response[d]["zero_noise_floor"],
        }
        for d in response
    }
    cert["gates"]["G5_weak_noise_parameter_free_prediction"].update(
        {
            "weak_gamma_max": weak_max,
            "maximum_relative_error_threshold": 0.03,
            "pass": weak_law_pass,
        }
    )
    cert["gates"]["G6_native_observable_shot_feasibility"] = {
        d: {
            "weak_best_shots_per_arm": response[d]["weak_best_shots"],
            "stress_shots_per_arm": response[d]["stress_shots"],
        }
        for d in response
    }
    cert["gates"]["G6_native_observable_shot_feasibility"].update(
        {"threshold": args.shot_threshold, "pass": shot_pass}
    )

    # Outputs and decision.
    with open(
        os.path.join(args.outdir, "ep_obs_stage3_gamma_scan.csv"),
        "w",
        newline="",
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    controls = {"reference": z0.tolist()}
    controls.update(
        {
            d: {
                "seed": seeds[d]["z"].tolist(),
                "modulated_relift": corrected[d]["z"].tolist(),
            }
            for d in corrected
        }
    )
    with open(
        os.path.join(args.outdir, "ep_obs_stage3_controls.json"), "w"
    ) as fh:
        json.dump(controls, fh, indent=2)

    cert["seed_audit"] = {
        d: {
            "gap_endpoint_infidelity": seeds[d]["gap_endpoint_infidelity"],
            "audit": seeds[d]["audit"],
        }
        for d in seeds
    }
    cert["relift_audit"] = {d: corrected[d]["audit"] for d in corrected}
    cert["response"] = {
        d: {
            key: value
            for key, value in response[d].items()
            if key not in {"E_ideal", "G", "dG"}
        }
        for d in response
    }
    if (
        foundation_pass
        and endpoint_pass
        and resampling_pass
        and duration_pass
        and response_nonzero
        and weak_law_pass
    ):
        if shot_pass:
            status = "MODULATED_NATIVE_OBSERVABLE_BRIDGE_FEASIBLE_IN_LOCAL_MODEL"
            next_step = (
                "Freeze the corrected controls and run a Pulser/Qutip numerical "
                "cross-check before any cloud submission."
            )
        else:
            status = "MODULATED_ENDPOINT_AND_RESPONSE_SUPPORTED_SIGNAL_TOO_SMALL"
            next_step = (
                "Keep these paths as a structural certificate; next optimize a "
                "device-compatible basis rotation or a larger endpoint-fiber "
                "excursion before cloud/QPU use."
            )
    else:
        status = "MODULATED_PATH_RESPONSE_NOT_CLOSED"
        next_step = (
            "Inspect the failed gate; do not interpret a finite-gamma split as "
            "path-resolved noise evidence."
        )
    cert["scientific_status"] = status
    cert["next_step"] = next_step
    cert["claim_boundary"] = (
        "Local Pulser AnalogDevice modulation model, exact 1 ns sampled "
        "waveform propagation, two atoms, local occupation dephasing, native "
        "computational-basis observables, and the declared gamma grid only. "
        "No calibrated FRESNEL waveform, cloud emulator, or QPU evidence."
    )
    with open(
        os.path.join(args.outdir, "ep_obs_stage3_certificate.json"), "w"
    ) as fh:
        json.dump(cert, fh, indent=2, default=float)

    log("\n" + "=" * 88)
    log("GLOBAL VERDICT")
    log("=" * 88)
    for name, gate in cert["gates"].items():
        if isinstance(gate, dict) and "pass" in gate:
            log(f"  {'PASS' if gate['pass'] else 'FAIL':5s} {name}")
    log(f"\nscientific_status={status}")
    log(f"next={next_step}")
    log(
        "written: ep_obs_stage3_certificate.json, "
        "ep_obs_stage3_controls.json, ep_obs_stage3_gamma_scan.csv, "
        "ep_obs_stage3_run.log"
    )
    logfile.close()
    return 0


def _four_parse_positive_csv(text: str, name: str) -> list[float]:
    """Parse a comma-separated, strictly positive, duplicate-free float grid."""
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated float grid") from exc
    if not values or any((not np.isfinite(x)) or x <= 0.0 for x in values):
        raise ValueError(f"{name} must contain finite positive values")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values")
    return sorted(values)


def _four_parse_segments(text: str) -> list[int]:
    """Return zero-based segment indices from 'all' or a 1-based CSV list."""
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


def four_apply_detuning_probe(
    z: np.ndarray, segment_index: int, delta_mhz: float
) -> np.ndarray:
    """Apply a coherent detuning offset to one drive segment.

    The control coordinate z[j,1] is measured in MHz because controls_from_z()
    maps it to an angular detuning through Delta_j = Delta0_j + 2*pi*z[j,1].
    """
    out = np.asarray(z, dtype=float).reshape(NSEG, 3).copy()
    out[segment_index, 1] += float(delta_mhz)
    return out.reshape(-1)


def four_native_record(z: np.ndarray) -> dict:
    """Compile a probed control and return all native observable statistics."""
    blocks = stage3_modulated_blocks(z)
    U = unitary_of_blocks(blocks)
    psi0 = np.zeros(DIM, dtype=complex)
    psi0[IDX_GG] = 1.0
    psi = U @ psi0
    probabilities = np.maximum(np.real(psi.conj() * psi), 0.0)
    probabilities /= probabilities.sum()
    means = {}
    variances = {}
    for name, observable in OBSERVABLES.items():
        outcomes = np.real(np.diag(observable))
        mean = float(probabilities @ outcomes)
        variance = float(
            max(probabilities @ (outcomes * outcomes) - mean * mean, 0.0)
        )
        means[name] = mean
        variances[name] = variance
    return {
        "blocks": blocks,
        "unitary": U,
        "probabilities": probabilities,
        "means": means,
        "variances": variances,
        "duration_us": float(sum(block.tau for block in blocks)),
    }


def four_evaluate_settings(
    z_cw: np.ndarray,
    z_ccw: np.ndarray,
    segment_index: int,
    delta_mhz: float,
) -> dict:
    """Evaluate the four independent arms for one segment and |delta|."""
    records = {
        "CW_plus": four_native_record(
            four_apply_detuning_probe(z_cw, segment_index, +delta_mhz)
        ),
        "CW_minus": four_native_record(
            four_apply_detuning_probe(z_cw, segment_index, -delta_mhz)
        ),
        "CCW_plus": four_native_record(
            four_apply_detuning_probe(z_ccw, segment_index, +delta_mhz)
        ),
        "CCW_minus": four_native_record(
            four_apply_detuning_probe(z_ccw, segment_index, -delta_mhz)
        ),
    }
    return records


def four_witness(records: dict, observable_name: str) -> dict:
    """Compute the antisymmetric four-arm witness and exact shot variance."""
    means = {
        arm: record["means"][observable_name]
        for arm, record in records.items()
    }
    variances = {
        arm: record["variances"][observable_name]
        for arm, record in records.items()
    }
    ccw_odd = means["CCW_plus"] - means["CCW_minus"]
    cw_odd = means["CW_plus"] - means["CW_minus"]
    signal = 0.5 * (ccw_odd - cw_odd)
    # Each arm receives N statistically independent shots:
    # Var[S_hat] = sum_arm Var[M_arm] / (4 N).
    variance_numerator = 0.25 * sum(variances.values())
    shots_5sigma = (
        float(25.0 * variance_numerator / (signal * signal))
        if abs(signal) > 1e-300
        else float("inf")
    )
    return {
        "signal": float(signal),
        "ccw_odd_difference": float(ccw_odd),
        "cw_odd_difference": float(cw_odd),
        "means": means,
        "variances": variances,
        "variance_numerator_per_arm": float(variance_numerator),
        "shots_per_arm_5sigma": shots_5sigma,
    }


def four_prepare_modulated_paths(args, log) -> dict:
    """Reproduce and re-lift CW/CCW paths on AnalogDevice modulation."""
    h_s = 0.008 if args.quick else args.hs
    U0_gap = unitary_of_z(np.zeros(18))
    seeds = {}
    log("\n" + "-" * 88)
    log("STEP 1  reproduce gap-aware M2 seeds")
    log("-" * 88)
    for direction in ("CW", "CCW"):
        started = time.time()
        z_seed, audit = m2_transport(
            U0_gap,
            loop_vertices(args.eps, direction),
            h_s,
            rcond=args.rcond,
        )
        seeds[direction] = {
            "z": z_seed,
            "audit": audit,
            "gap_endpoint_infidelity": endpoint_infidelity(z_seed, U0_gap),
        }
        log(
            f"  {direction}: steps={audit['n_steps']} "
            f"ranks={audit['rank_set']} "
            f"gap-epsU={seeds[direction]['gap_endpoint_infidelity']:.3e} "
            f"|z|={np.linalg.norm(z_seed):.3e} "
            f"({time.time()-started:.1f}s)"
        )

    log("\n" + "-" * 88)
    log("STEP 2  project both seeds onto the modulated full-unitary fiber")
    log("-" * 88)
    z0 = np.zeros(18)
    U0_mod = stage3_modulated_unitary(z0)
    corrected = {}
    for direction in ("CW", "CCW"):
        z_seed = seeds[direction]["z"]
        seed_error = stage3_endpoint_infidelity(
            stage3_modulated_unitary(z_seed), U0_mod
        )
        log(f"\n  {direction}: seed modulated epsU={seed_error:.3e}")
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
            np.linalg.norm(
                stage3_phase_aligned_residual(record["unitary"], U0_mod)
            )
        )
        solver_resampling_error = float(
            np.max(np.abs(record["unitary"] - U_solver))
        )
        corrected[direction] = {
            "z": z,
            "record": record,
            "audit": audit,
            "endpoint_infidelity": endpoint_error,
            "endpoint_residual_norm": residual_norm,
            "solver_resampling_error": solver_resampling_error,
        }
        log(
            f"    final epsU={endpoint_error:.3e} |r|={residual_norm:.3e} "
            f"|dz|={audit['correction_norm']:.3e} |z|={audit['final_norm']:.3e} "
            f"max|U_resampled-U_solver|={solver_resampling_error:.3e}"
        )

    pair_unitary_error = stage3_endpoint_infidelity(
        corrected["CW"]["record"]["unitary"],
        corrected["CCW"]["record"]["unitary"],
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
    durations = {
        direction: corrected[direction]["record"]["duration_us"]
        for direction in corrected
    }
    endpoint_pass = all(
        corrected[direction]["endpoint_infidelity"]
        <= args.endpoint_infidelity_tol
        and corrected[direction]["endpoint_residual_norm"]
        <= args.endpoint_residual_tol
        and corrected[direction]["solver_resampling_error"] <= 1e-12
        for direction in corrected
    )
    duration_pass = abs(durations["CW"] - durations["CCW"]) <= 1e-9
    log(
        f"\n  CW-vs-CCW baseline: process infidelity={pair_unitary_error:.3e} "
        f"population TVD={pair_probability_tvd:.3e}"
    )
    log(
        f"  modulated durations CW/CCW="
        f"{durations['CW']:.6f}/{durations['CCW']:.6f} us"
    )
    return {
        "reference_z": z0,
        "reference_unitary": U0_mod,
        "seeds": seeds,
        "corrected": corrected,
        "pair_unitary_infidelity": pair_unitary_error,
        "pair_probability_tvd": pair_probability_tvd,
        "durations_us": durations,
        "endpoint_pass": bool(endpoint_pass),
        "duration_pass": bool(duration_pass),
    }


def four_setting_main(argv=None):
    ap = argparse.ArgumentParser(prog="ep_obs_four_setting_probe_v1_0.py")
    ap.add_argument(
        "--quick",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use h_s=0.008 for the M2 seed transport (default: enabled)",
    )
    ap.add_argument("--eps", type=float, default=0.040)
    ap.add_argument("--hs", type=float, default=0.002)
    ap.add_argument("--gap", type=float, default=0.340)
    ap.add_argument("--omega-scale", type=float, default=0.80)
    ap.add_argument("--phase-sign", type=float, default=1.0, choices=[1.0, -1.0])
    ap.add_argument("--fd-step", type=float, default=2e-5)
    ap.add_argument("--relift-iters", type=int, default=6)
    ap.add_argument("--endpoint-residual-tol", type=float, default=2e-9)
    ap.add_argument("--endpoint-infidelity-tol", type=float, default=1e-11)
    ap.add_argument("--rcond", type=float, default=1e-6)
    ap.add_argument(
        "--selection-deltas-mhz",
        default="0.005,0.010",
        help="positive |delta| values used only to select segment/observable",
    )
    ap.add_argument(
        "--test-deltas-mhz",
        default="0.020,0.040,0.080,0.120,0.200",
        help="disjoint held-out positive |delta| values",
    )
    ap.add_argument(
        "--probe-segments",
        default="all",
        help="'all' or a 1-based comma-separated subset of the six segments",
    )
    ap.add_argument(
        "--observable",
        default="auto",
        choices=["auto"] + list(OBSERVABLES.keys()),
        help="freeze one native observable or select it on the selection grid",
    )
    ap.add_argument("--shot-threshold", type=float, default=1e7)
    ap.add_argument("--heldout-relative-error-tol", type=float, default=0.10)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--source-path", default=None)
    if argv is None:
        argv = [] if _in_notebook() else sys.argv[1:]
    args, ignored = ap.parse_known_args(list(argv))
    if ignored:
        print(f"[note] ignoring arguments not belonging to this script: {ignored}")

    selection_deltas = _four_parse_positive_csv(
        args.selection_deltas_mhz, "--selection-deltas-mhz"
    )
    test_deltas = _four_parse_positive_csv(
        args.test_deltas_mhz, "--test-deltas-mhz"
    )
    overlap = sorted(set(selection_deltas).intersection(test_deltas))
    if overlap:
        raise ValueError(
            "selection and test delta grids must be disjoint; overlap="
            + ",".join(map(str, overlap))
        )
    probe_segments = _four_parse_segments(args.probe_segments)
    if args.outdir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.outdir = f"ep_obs_four_setting_probe_{stamp}"

    global _SOURCE_PATH_OVERRIDE
    if args.source_path:
        _SOURCE_PATH_OVERRIDE = args.source_path
    MODEL["gap_tau"] = float(args.gap)
    MODEL["omega_scale"] = float(args.omega_scale)
    MODEL["phase_sign"] = float(args.phase_sign)
    os.makedirs(args.outdir, exist_ok=True)
    logfile = open(
        os.path.join(args.outdir, "four_setting_run.log"),
        "w",
        encoding="utf-8",
    )

    def log(message=""):
        print(message)
        logfile.write(str(message) + "\n")
        logfile.flush()

    cert = {
        "version": VERSION,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_sha256": self_sha256(),
        "packages": package_versions(),
        "args": vars(args),
        "selection_deltas_mhz": selection_deltas,
        "heldout_test_deltas_mhz": test_deltas,
        "probe_segments_1based": [index + 1 for index in probe_segments],
        "cloud_access": "none",
        "gates": {},
    }
    log("=" * 88)
    log(f"{VERSION}  FOUR-SETTING COHERENT PATH-SUSCEPTIBILITY WITNESS")
    log("=" * 88)
    log(f"UTC={cert['utc']}")
    log(
        "witness=1/2[(CCW+ - CCW-) - (CW+ - CW-)] | "
        "initial=|gg> | readout=native computational basis"
    )
    log(
        f"selection deltas MHz={selection_deltas} | "
        f"held-out deltas MHz={test_deltas} | cloud access=none"
    )
    for key, value in cert["packages"].items():
        log(f"  {key:20s} {value}")

    try:
        from pulser.devices import AnalogDevice

        _ = AnalogDevice.channels["rydberg_global"]
    except Exception as exc:
        log(f"\nFATAL: Pulser AnalogDevice unavailable: {type(exc).__name__}: {exc}")
        cert["scientific_status"] = "PULSER_ANALOGDEVICE_UNAVAILABLE"
        with open(
            os.path.join(args.outdir, "four_setting_certificate.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(cert, file, indent=2, default=float)
        logfile.close()
        return 1

    h0_error, gg_ok = pulser_hamiltonian_probe()
    interaction_audit = stage3_configure_analog_interaction()
    cert["device_foundation"] = {
        "hamiltonian_max_abs_error": h0_error,
        "initial_state_is_gg": gg_ok,
        "interaction": interaction_audit,
    }
    foundation_pass = bool(
        h0_error < 1e-9
        and gg_ok
        and interaction_audit["AnalogDevice_C6"] > 0.0
    )
    cert["gates"]["G0_device_hamiltonian_foundation"] = {
        **cert["device_foundation"],
        "pass": foundation_pass,
    }
    log(
        f"\n[H0] internal drive == Pulser: max|dH|={h0_error:.3e} "
        f"init=|gg>:{gg_ok}"
    )
    if not foundation_pass:
        cert["scientific_status"] = "DEVICE_HAMILTONIAN_FOUNDATION_NOT_RESOLVED"
        with open(
            os.path.join(args.outdir, "four_setting_certificate.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(cert, file, indent=2, default=float)
        logfile.close()
        return 2

    paths = four_prepare_modulated_paths(args, log)
    endpoint_gate = bool(paths["endpoint_pass"] and paths["duration_pass"])
    cert["gates"]["G1_modulated_full_unitary_endpoint_match"] = {
        "CW": {
            "endpoint_infidelity": paths["corrected"]["CW"][
                "endpoint_infidelity"
            ],
            "residual_norm": paths["corrected"]["CW"][
                "endpoint_residual_norm"
            ],
        },
        "CCW": {
            "endpoint_infidelity": paths["corrected"]["CCW"][
                "endpoint_infidelity"
            ],
            "residual_norm": paths["corrected"]["CCW"][
                "endpoint_residual_norm"
            ],
        },
        "CW_vs_CCW_process_infidelity": paths["pair_unitary_infidelity"],
        "CW_vs_CCW_population_TVD": paths["pair_probability_tvd"],
        "durations_us": paths["durations_us"],
        "pass": endpoint_gate,
    }
    if not endpoint_gate:
        cert["scientific_status"] = "MODULATED_ENDPOINT_RELIFT_NOT_RESOLVED"
        cert["next_step"] = (
            "Resolve the full-unitary endpoint gate before evaluating a "
            "path-conditioned derivative."
        )
        with open(
            os.path.join(args.outdir, "four_setting_certificate.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(cert, file, indent=2, default=float)
        log(f"\nGLOBAL VERDICT: {cert['scientific_status']}")
        logfile.close()
        return 2

    z_cw = paths["corrected"]["CW"]["z"]
    z_ccw = paths["corrected"]["CCW"]["z"]

    # Selection uses only the declared selection deltas.
    log("\n" + "-" * 88)
    log("STEP 3  selection-only scan over segment and native observable")
    log("-" * 88)
    selection_rows = []
    selection_groups = {}
    allowed_observables = (
        list(OBSERVABLES.keys())
        if args.observable == "auto"
        else [args.observable]
    )
    selection_runtime_start = time.time()
    selection_device_pass = True
    for segment_index in probe_segments:
        for delta_mhz in selection_deltas:
            try:
                records = four_evaluate_settings(
                    z_cw, z_ccw, segment_index, delta_mhz
                )
            except Exception as exc:
                selection_device_pass = False
                log(
                    f"  segment={segment_index+1} delta={delta_mhz:g} MHz "
                    f"REJECTED: {type(exc).__name__}: {exc}"
                )
                continue
            duration_spread = max(
                record["duration_us"] for record in records.values()
            ) - min(record["duration_us"] for record in records.values())
            for observable_name in allowed_observables:
                witness = four_witness(records, observable_name)
                slope = witness["signal"] / delta_mhz
                standard_score = abs(slope) / np.sqrt(
                    max(witness["variance_numerator_per_arm"], 1e-300)
                )
                row = {
                    "regime": "selection",
                    "segment_1based": segment_index + 1,
                    "observable": observable_name,
                    "delta_mhz": delta_mhz,
                    "signal": witness["signal"],
                    "slope_per_mhz": slope,
                    "standardized_slope_per_sqrt_shot_per_mhz": standard_score,
                    "shots_per_arm_5sigma": witness["shots_per_arm_5sigma"],
                    "duration_spread_us": duration_spread,
                    "prediction": "",
                    "absolute_residual": "",
                    "relative_error": "",
                    "signal_regime": "selection_only",
                }
                selection_rows.append(row)
                selection_groups.setdefault(
                    (segment_index, observable_name), []
                ).append(row)

    if not selection_groups:
        cert["scientific_status"] = "FOUR_SETTING_PROGRAMS_REJECTED"
        cert["gates"]["G2_selection_program_acceptance"] = {
            "pass": False
        }
        with open(
            os.path.join(args.outdir, "four_setting_certificate.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(cert, file, indent=2, default=float)
        logfile.close()
        return 2

    group_scores = []
    for (segment_index, observable_name), rows in selection_groups.items():
        slopes = np.asarray([row["slope_per_mhz"] for row in rows], float)
        scores = np.asarray(
            [
                row["standardized_slope_per_sqrt_shot_per_mhz"]
                for row in rows
            ],
            float,
        )
        relative_slope_spread = float(
            np.ptp(slopes) / max(abs(np.median(slopes)), 1e-300)
        )
        group_scores.append(
            {
                "segment_index": segment_index,
                "observable": observable_name,
                "median_slope_per_mhz": float(np.median(slopes)),
                "median_standardized_score": float(np.median(scores)),
                "selection_relative_slope_spread": relative_slope_spread,
                "best_selection_shots_per_arm": float(
                    min(row["shots_per_arm_5sigma"] for row in rows)
                ),
            }
        )
    selected = max(
        group_scores, key=lambda row: row["median_standardized_score"]
    )
    selected_segment = int(selected["segment_index"])
    selected_observable = str(selected["observable"])
    selected_slope = float(selected["median_slope_per_mhz"])
    log(
        f"  selected segment={selected_segment+1} "
        f"observable={selected_observable} "
        f"chi_probe={selected_slope:+.7e} per MHz "
        f"selection slope spread={selected['selection_relative_slope_spread']:.2%}"
    )
    log(
        f"  selection scan elapsed="
        f"{time.time()-selection_runtime_start:.1f}s"
    )
    cert["selection"] = {
        "rule": (
            "maximize median |S/delta| divided by sqrt of the exact four-arm "
            "single-shot variance numerator"
        ),
        "selected_segment_1based": selected_segment + 1,
        "selected_observable": selected_observable,
        "frozen_linear_coefficient_per_mhz": selected_slope,
        "selected_group": selected,
        "all_group_scores": group_scores,
    }
    cert["gates"]["G2_selection_program_acceptance"] = {
        "attempted_program_sets": len(probe_segments)
        * len(selection_deltas),
        "completed_program_sets": len(
            {
                (row["segment_1based"], row["delta_mhz"])
                for row in selection_rows
            }
        ),
        "pass": selection_device_pass,
    }

    # Held-out test: segment, observable, and linear coefficient are frozen.
    log("\n" + "-" * 88)
    log("STEP 4  disjoint held-out delta test of the frozen four settings")
    log("-" * 88)
    log(
        f"  frozen segment={selected_segment+1}, "
        f"observable={selected_observable}, "
        f"prediction S(delta)=delta*({selected_slope:+.7e})"
    )
    log(
        f"  {'delta MHz':>10s} {'S exact':>13s} {'prediction':>13s} "
        f"{'rel.err':>9s} {'shots/arm':>13s}"
    )
    test_rows = []
    test_witnesses = []
    test_device_pass = True
    for delta_mhz in test_deltas:
        try:
            records = four_evaluate_settings(
                z_cw, z_ccw, selected_segment, delta_mhz
            )
        except Exception as exc:
            test_device_pass = False
            log(
                f"  {delta_mhz:10.6f} REJECTED: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        witness = four_witness(records, selected_observable)
        prediction = delta_mhz * selected_slope
        residual = abs(witness["signal"] - prediction)
        relative_error = residual / max(abs(witness["signal"]), 1e-300)
        duration_spread = max(
            record["duration_us"] for record in records.values()
        ) - min(record["duration_us"] for record in records.values())
        row = {
            "regime": "heldout_test",
            "segment_1based": selected_segment + 1,
            "observable": selected_observable,
            "delta_mhz": delta_mhz,
            "signal": witness["signal"],
            "slope_per_mhz": witness["signal"] / delta_mhz,
            "standardized_slope_per_sqrt_shot_per_mhz": (
                abs(witness["signal"] / delta_mhz)
                / np.sqrt(
                    max(witness["variance_numerator_per_arm"], 1e-300)
                )
            ),
            "shots_per_arm_5sigma": witness["shots_per_arm_5sigma"],
            "duration_spread_us": duration_spread,
            "prediction": prediction,
            "absolute_residual": residual,
            "relative_error": relative_error,
            "signal_regime": "heldout_only",
        }
        test_rows.append(row)
        test_witnesses.append(witness)
        log(
            f"  {delta_mhz:10.6f} {witness['signal']:+13.6e} "
            f"{prediction:+13.6e} {relative_error:9.3%} "
            f"{witness['shots_per_arm_5sigma']:13.3e}"
        )

    all_rows = selection_rows + test_rows
    with open(
        os.path.join(args.outdir, "four_setting_scan.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    controls = {
        "reference": paths["reference_z"].tolist(),
        "CW_modulated_relift": z_cw.tolist(),
        "CCW_modulated_relift": z_ccw.tolist(),
        "selected_probe_segment_1based": selected_segment + 1,
        "selected_observable": selected_observable,
        "selection_deltas_mhz": selection_deltas,
        "heldout_test_deltas_mhz": test_deltas,
        "four_setting_definition": {
            "CW_plus": "+delta on selected detuning coordinate",
            "CW_minus": "-delta on selected detuning coordinate",
            "CCW_plus": "+delta on selected detuning coordinate",
            "CCW_minus": "-delta on selected detuning coordinate",
        },
    }
    with open(
        os.path.join(args.outdir, "four_setting_controls.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(controls, file, indent=2)

    if test_rows:
        test_delta_array = np.asarray(
            [row["delta_mhz"] for row in test_rows], float
        )
        test_signal_array = np.asarray(
            [abs(row["signal"]) for row in test_rows], float
        )
        heldout_signal_exponent = loglog_slope(
            test_delta_array, test_signal_array
        )
        heldout_max_relative_error = float(
            max(row["relative_error"] for row in test_rows)
        )
        best_test_shots = float(
            min(row["shots_per_arm_5sigma"] for row in test_rows)
        )
        maximum_test_signal = float(
            max(abs(row["signal"]) for row in test_rows)
        )
        heldout_linear_pass = bool(
            0.90 <= heldout_signal_exponent <= 1.10
            and heldout_max_relative_error
            <= args.heldout_relative_error_tol
        )
        shot_pass = bool(best_test_shots <= args.shot_threshold)
        duration_pass = all(
            row["duration_spread_us"] <= 1e-9 for row in test_rows
        )
    else:
        heldout_signal_exponent = float("nan")
        heldout_max_relative_error = float("inf")
        best_test_shots = float("inf")
        maximum_test_signal = 0.0
        heldout_linear_pass = False
        shot_pass = False
        duration_pass = False

    baseline_observable_difference = abs(
        paths["corrected"]["CCW"]["record"]["means"][selected_observable]
        - paths["corrected"]["CW"]["record"]["means"][selected_observable]
    )
    baseline_pass = bool(baseline_observable_difference <= 1e-9)
    cert["heldout_test"] = {
        "completed_points": len(test_rows),
        "declared_points": len(test_deltas),
        "signal_exponent": heldout_signal_exponent,
        "maximum_relative_error_to_frozen_linear_prediction": (
            heldout_max_relative_error
        ),
        "maximum_absolute_signal": maximum_test_signal,
        "best_shots_per_arm_5sigma": best_test_shots,
    }
    cert["gates"]["G3_zero_probe_outputs_match"] = {
        "observable": selected_observable,
        "absolute_CW_CCW_difference": baseline_observable_difference,
        "threshold": 1e-9,
        "pass": baseline_pass,
    }
    cert["gates"]["G4_heldout_program_acceptance_and_duration"] = {
        "completed_points": len(test_rows),
        "declared_points": len(test_deltas),
        "equal_duration": duration_pass,
        "pass": bool(
            test_device_pass
            and len(test_rows) == len(test_deltas)
            and duration_pass
        ),
    }
    cert["gates"]["G5_heldout_linear_path_susceptibility"] = {
        "signal_exponent": heldout_signal_exponent,
        "accepted_exponent_interval": [0.90, 1.10],
        "maximum_relative_error": heldout_max_relative_error,
        "relative_error_threshold": args.heldout_relative_error_tol,
        "pass": heldout_linear_pass,
    }
    cert["gates"]["G6_four_setting_shot_feasibility"] = {
        "best_shots_per_arm_5sigma": best_test_shots,
        "threshold": args.shot_threshold,
        "four_independent_arms_per_delta": True,
        "pass": shot_pass,
    }

    scientific_support = bool(
        foundation_pass
        and endpoint_gate
        and cert["gates"]["G2_selection_program_acceptance"]["pass"]
        and baseline_pass
        and cert["gates"]["G4_heldout_program_acceptance_and_duration"][
            "pass"
        ]
        and heldout_linear_pass
    )
    if scientific_support and shot_pass:
        status = "FOUR_SETTING_PATH_SUSCEPTIBILITY_SUPPORTED_AND_SHOT_FEASIBLE"
        next_step = (
            "Freeze the four schedules at one held-out delta and perform an "
            "independent Pulser solver cross-check before any cloud submission."
        )
    elif scientific_support:
        status = "FOUR_SETTING_PATH_SUSCEPTIBILITY_SUPPORTED_SIGNAL_TOO_SMALL"
        next_step = (
            "The coherent witness is structurally supported but not practical "
            "at the declared shot threshold; enlarge the endpoint-fiber "
            "excursion or optimize the native standardized response."
        )
    else:
        status = "FOUR_SETTING_PATH_SUSCEPTIBILITY_NOT_CLOSED"
        next_step = (
            "Inspect the first failed gate; do not interpret the observed split "
            "as a validated path-conditioned derivative."
        )
    cert["scientific_status"] = status
    cert["next_step"] = next_step
    cert["claim_boundary"] = (
        "Two-atom local Pulser AnalogDevice modulation model, exact sampled "
        "coherent propagation, native |gg> preparation and computational-basis "
        "observables, one identical segment-local detuning probe, and the "
        "declared selection/test delta grids only. This measures a coherent "
        "path-conditioned susceptibility, not natural dephasing, calibrated "
        "FRESNEL noise, cloud-emulator evidence, or QPU evidence."
    )
    cert["path_relift_audit"] = {
        direction: {
            "seed": paths["seeds"][direction]["audit"],
            "modulated_relift": paths["corrected"][direction]["audit"],
        }
        for direction in ("CW", "CCW")
    }
    with open(
        os.path.join(args.outdir, "four_setting_certificate.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(cert, file, indent=2, default=float)

    log("\n" + "=" * 88)
    log("GLOBAL VERDICT")
    log("=" * 88)
    for gate_name, gate in cert["gates"].items():
        log(f"  {'PASS' if gate['pass'] else 'FAIL':5s} {gate_name}")
    log(f"\nscientific_status={status}")
    log(f"best held-out shots/arm={best_test_shots:.3e}")
    log(f"next={next_step}")
    log(
        "written: four_setting_certificate.json, four_setting_controls.json, "
        "four_setting_scan.csv, four_setting_run.log"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        deltas = np.asarray([row["delta_mhz"] for row in test_rows], float)
        signals = np.asarray([row["signal"] for row in test_rows], float)
        predictions = deltas * selected_slope
        shots = np.asarray(
            [row["shots_per_arm_5sigma"] for row in test_rows], float
        )
        axes[0].plot(deltas, signals, "o-", label="held-out exact")
        axes[0].plot(deltas, predictions, "--", label="frozen linear prediction")
        axes[0].set_xlabel(r"$|\delta|$ [MHz]")
        axes[0].set_ylabel(r"$S(\delta)$")
        axes[0].set_title(
            f"segment {selected_segment+1}, {selected_observable}"
        )
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[1].semilogy(deltas, shots, "o-")
        axes[1].axhline(
            args.shot_threshold,
            color="tab:red",
            linestyle="--",
            label="declared threshold",
        )
        axes[1].set_xlabel(r"$|\delta|$ [MHz]")
        axes[1].set_ylabel("5-sigma shots per arm")
        axes[1].grid(alpha=0.3, which="both")
        axes[1].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(
            os.path.join(args.outdir, "four_setting_prediction.png"),
            dpi=170,
        )
        log("written: four_setting_prediction.png")
    except Exception as exc:
        log(f"(figure skipped: {type(exc).__name__}: {exc})")

    logfile.close()
    return 0


def run_four_setting(**kwargs):
    """Notebook entry point: import this file, then call run_four_setting()."""
    argv = []
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            elif key == "quick":
                argv.append("--no-quick")
        else:
            argv.extend([flag, str(value)])
    return four_setting_main(argv)


def amplified_contrast_statistics(records: dict) -> dict:
    """Native four-outcome contrast vector and its four-arm covariance.

    Basis order is Pulser's (rr, rg, gr, gg).  For an outcome score vector w,

        S = w.T @ contrast
        Var[S_hat] = (w.T @ covariance @ w) / shots_per_arm.

    The 1/2 coefficients match the declared four-setting witness.
    """
    coefficients = {
        "CCW_plus": +0.5,
        "CCW_minus": -0.5,
        "CW_plus": -0.5,
        "CW_minus": +0.5,
    }
    contrast = np.zeros(DIM, dtype=float)
    covariance = np.zeros((DIM, DIM), dtype=float)
    probabilities = {}
    for arm, coefficient in coefficients.items():
        p = np.asarray(records[arm]["probabilities"], dtype=float)
        probabilities[arm] = p
        contrast += coefficient * p
        covariance += (coefficient * coefficient) * (
            np.diag(p) - np.outer(p, p)
        )
    return {
        "contrast": contrast,
        "covariance": covariance,
        "probabilities": probabilities,
    }


def amplified_optimal_weights(
    statistics: list[dict], deltas_mhz: list[float]
) -> dict:
    """Optimal native score frozen from selection data only.

    A constant shift of every outcome weight changes no contrast and has zero
    multinomial variance.  The pseudoinverse handles this gauge direction.
    """
    slope_vectors = np.stack(
        [
            stat["contrast"] / delta
            for stat, delta in zip(statistics, deltas_mhz)
        ]
    )
    covariance_mean = np.mean(
        np.stack([stat["covariance"] for stat in statistics]), axis=0
    )
    slope_vector = np.mean(slope_vectors, axis=0)
    covariance_pinv = np.linalg.pinv(covariance_mean, rcond=1e-12)
    weights = covariance_pinv @ slope_vector
    weights -= np.mean(weights)
    scale = float(np.max(np.abs(weights)))
    if not np.isfinite(scale) or scale <= 1e-14:
        raise RuntimeError("optimal native score is numerically degenerate")
    weights /= scale
    standardized_bound = float(
        np.sqrt(
            max(
                np.real(slope_vector @ covariance_pinv @ slope_vector),
                0.0,
            )
        )
    )
    return {
        "weights": weights,
        "selection_mean_slope_vector": slope_vector,
        "selection_mean_covariance": covariance_mean,
        "selection_generalized_score_bound": standardized_bound,
    }


def amplified_weighted_witness(statistics: dict, weights: np.ndarray) -> dict:
    signal = float(weights @ statistics["contrast"])
    variance_numerator = float(
        max(weights @ statistics["covariance"] @ weights, 0.0)
    )
    shots = (
        float(25.0 * variance_numerator / (signal * signal))
        if abs(signal) > 1e-300
        else float("inf")
    )
    arm_means = {
        arm: float(weights @ probability)
        for arm, probability in statistics["probabilities"].items()
    }
    return {
        "signal": signal,
        "variance_numerator_per_arm": variance_numerator,
        "shots_per_arm_5sigma": shots,
        "arm_weighted_means": arm_means,
    }


def amplified_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="ep_obs_four_setting_amplified_v1_1.py"
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
        "--selection-deltas-mhz",
        default="0.005,0.010",
        help="selection-only probe magnitudes",
    )
    ap.add_argument(
        "--test-deltas-mhz",
        default="0.020,0.040,0.080,0.120,0.200",
        help="disjoint held-out probe magnitudes",
    )
    ap.add_argument(
        "--probe-segments",
        default="all",
        help="'all' or a 1-based comma-separated subset",
    )
    ap.add_argument("--shot-threshold", type=float, default=1e7)
    ap.add_argument("--target-standardized-gain", type=float, default=17.5)
    ap.add_argument("--heldout-relative-error-tol", type=float, default=0.10)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--source-path", default=None)
    if argv is None:
        argv = [] if _in_notebook() else sys.argv[1:]
    args, ignored = ap.parse_known_args(list(argv))
    if ignored:
        print(f"[note] ignoring arguments not belonging to this script: {ignored}")

    eps_grid = _four_parse_positive_csv(args.eps_grid, "--eps-grid")
    selection_deltas = _four_parse_positive_csv(
        args.selection_deltas_mhz, "--selection-deltas-mhz"
    )
    test_deltas = _four_parse_positive_csv(
        args.test_deltas_mhz, "--test-deltas-mhz"
    )
    overlap = sorted(set(selection_deltas).intersection(test_deltas))
    if overlap:
        raise ValueError(
            "selection/test delta grids must be disjoint; overlap="
            + ",".join(map(str, overlap))
        )
    probe_segments = _four_parse_segments(args.probe_segments)
    if args.outdir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.outdir = f"ep_obs_four_setting_amplified_{stamp}"

    global _SOURCE_PATH_OVERRIDE
    if args.source_path:
        _SOURCE_PATH_OVERRIDE = args.source_path
    MODEL["gap_tau"] = float(args.gap)
    MODEL["omega_scale"] = float(args.omega_scale)
    MODEL["phase_sign"] = float(args.phase_sign)
    os.makedirs(args.outdir, exist_ok=True)
    logfile = open(
        os.path.join(args.outdir, "four_setting_v1_1_run.log"),
        "w",
        encoding="utf-8",
    )

    def log(message=""):
        print(message)
        logfile.write(str(message) + "\n")
        logfile.flush()

    cert = {
        "version": VERSION,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_sha256": self_sha256(),
        "packages": package_versions(),
        "args": vars(args),
        "eps_selection_grid": eps_grid,
        "selection_deltas_mhz": selection_deltas,
        "heldout_test_deltas_mhz": test_deltas,
        "probe_segments_1based": [index + 1 for index in probe_segments],
        "native_outcome_order": ["rr", "rg", "gr", "gg"],
        "cloud_access": "none",
        "gates": {},
    }
    log("=" * 92)
    log(f"{VERSION}  AMPLIFIED FOUR-SETTING SEARCH")
    log("=" * 92)
    log(f"UTC={cert['utc']}")
    log(
        "selection searches epsilon + segment + optimal native outcome score; "
        "held-out deltas are touched only after freezing"
    )
    log(
        f"epsilon grid={eps_grid} | selection deltas={selection_deltas} MHz | "
        f"held-out deltas={test_deltas} MHz"
    )
    for key, value in cert["packages"].items():
        log(f"  {key:20s} {value}")

    try:
        from pulser.devices import AnalogDevice

        _ = AnalogDevice.channels["rydberg_global"]
    except Exception as exc:
        log(f"\nFATAL: Pulser AnalogDevice unavailable: {type(exc).__name__}: {exc}")
        cert["scientific_status"] = "PULSER_ANALOGDEVICE_UNAVAILABLE"
        with open(
            os.path.join(args.outdir, "four_setting_v1_1_certificate.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(cert, file, indent=2, default=float)
        logfile.close()
        return 1

    h0_error, gg_ok = pulser_hamiltonian_probe()
    interaction_audit = stage3_configure_analog_interaction()
    foundation_pass = bool(
        h0_error < 1e-9
        and gg_ok
        and interaction_audit["AnalogDevice_C6"] > 0.0
    )
    cert["gates"]["G0_device_hamiltonian_foundation"] = {
        "hamiltonian_max_abs_error": h0_error,
        "initial_state_is_gg": gg_ok,
        "interaction": interaction_audit,
        "pass": foundation_pass,
    }
    log(
        f"\n[H0] internal drive == Pulser: max|dH|={h0_error:.3e} "
        f"init=|gg>:{gg_ok}"
    )
    if not foundation_pass:
        cert["scientific_status"] = "DEVICE_HAMILTONIAN_FOUNDATION_NOT_RESOLVED"
        with open(
            os.path.join(args.outdir, "four_setting_v1_1_certificate.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(cert, file, indent=2, default=float)
        logfile.close()
        return 2

    log("\n" + "=" * 92)
    log("SELECTION PHASE  epsilon + segment + native score")
    log("=" * 92)
    search_rows = []
    eps_audits = []
    best_bundle = None
    attempted_program_sets = 0
    completed_program_sets = 0
    search_started = time.time()
    for eps in eps_grid:
        log("\n" + "#" * 92)
        log(f"EPSILON CANDIDATE {eps:g}  (selection data only)")
        log("#" * 92)
        args.eps = float(eps)
        try:
            paths = four_prepare_modulated_paths(args, log)
        except Exception as exc:
            eps_audits.append(
                {
                    "eps": eps,
                    "status": "PATH_PREPARATION_REJECTED",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            log(f"  epsilon={eps:g} rejected: {type(exc).__name__}: {exc}")
            continue
        path_nontrivial = all(
            paths["corrected"][direction]["audit"]["final_norm"] > 1e-5
            for direction in ("CW", "CCW")
        )
        path_gate = bool(
            paths["endpoint_pass"]
            and paths["duration_pass"]
            and path_nontrivial
        )
        eps_audit = {
            "eps": eps,
            "status": "PATH_ACCEPTED" if path_gate else "PATH_GATE_FAILED",
            "endpoint_pass": paths["endpoint_pass"],
            "duration_pass": paths["duration_pass"],
            "path_nontrivial": path_nontrivial,
            "CW_norm": paths["corrected"]["CW"]["audit"]["final_norm"],
            "CCW_norm": paths["corrected"]["CCW"]["audit"]["final_norm"],
            "CW_endpoint_infidelity": paths["corrected"]["CW"][
                "endpoint_infidelity"
            ],
            "CCW_endpoint_infidelity": paths["corrected"]["CCW"][
                "endpoint_infidelity"
            ],
            "CW_vs_CCW_process_infidelity": paths[
                "pair_unitary_infidelity"
            ],
        }
        eps_audits.append(eps_audit)
        if not path_gate:
            log(f"  epsilon={eps:g}: path gate failed; excluded from selection")
            continue

        z_cw = paths["corrected"]["CW"]["z"]
        z_ccw = paths["corrected"]["CCW"]["z"]
        for segment_index in probe_segments:
            statistics_list = []
            segment_rejected = False
            for delta_mhz in selection_deltas:
                attempted_program_sets += 1
                try:
                    records = four_evaluate_settings(
                        z_cw, z_ccw, segment_index, delta_mhz
                    )
                except Exception as exc:
                    segment_rejected = True
                    log(
                        f"  eps={eps:g} segment={segment_index+1} "
                        f"delta={delta_mhz:g} REJECTED: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    break
                completed_program_sets += 1
                statistics_list.append(
                    amplified_contrast_statistics(records)
                )
            if segment_rejected or len(statistics_list) != len(
                selection_deltas
            ):
                continue
            try:
                optimal = amplified_optimal_weights(
                    statistics_list, selection_deltas
                )
            except Exception as exc:
                log(
                    f"  eps={eps:g} segment={segment_index+1} score rejected: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            weights = optimal["weights"]
            witnesses = [
                amplified_weighted_witness(stat, weights)
                for stat in statistics_list
            ]
            slopes = np.asarray(
                [
                    witness["signal"] / delta
                    for witness, delta in zip(witnesses, selection_deltas)
                ],
                float,
            )
            standardized = np.asarray(
                [
                    abs(witness["signal"] / delta)
                    / np.sqrt(
                        max(
                            witness["variance_numerator_per_arm"],
                            1e-300,
                        )
                    )
                    for witness, delta in zip(
                        witnesses, selection_deltas
                    )
                ],
                float,
            )
            median_slope = float(np.median(slopes))
            median_score = float(np.median(standardized))
            relative_slope_spread = float(
                np.ptp(slopes) / max(abs(median_slope), 1e-300)
            )
            best_selection_shots = float(
                min(
                    witness["shots_per_arm_5sigma"]
                    for witness in witnesses
                )
            )
            row = {
                "eps": eps,
                "segment_1based": segment_index + 1,
                "CW_path_norm": eps_audit["CW_norm"],
                "CCW_path_norm": eps_audit["CCW_norm"],
                "median_slope_per_mhz": median_slope,
                "median_standardized_score": median_score,
                "relative_slope_spread": relative_slope_spread,
                "best_selection_shots_per_arm": best_selection_shots,
                "weight_rr": float(weights[0]),
                "weight_rg": float(weights[1]),
                "weight_gr": float(weights[2]),
                "weight_gg": float(weights[3]),
            }
            search_rows.append(row)
            log(
                f"  segment={segment_index+1} score={median_score:.6e} "
                f"chi={median_slope:+.6e}/MHz "
                f"shots={best_selection_shots:.3e} "
                f"spread={relative_slope_spread:.2%}"
            )
            if (
                best_bundle is None
                or median_score
                > best_bundle["row"]["median_standardized_score"]
            ):
                best_bundle = {
                    "row": row,
                    "paths": paths,
                    "weights": weights.copy(),
                    "selection_optimal": optimal,
                    "selection_witnesses": witnesses,
                }

    if best_bundle is None:
        cert["scientific_status"] = "NO_SELECTION_CANDIDATE_SURVIVED"
        cert["epsilon_audits"] = eps_audits
        cert["gates"]["G1_selection_search_completed"] = {
            "attempted_program_sets": attempted_program_sets,
            "completed_program_sets": completed_program_sets,
            "pass": False,
        }
        with open(
            os.path.join(args.outdir, "four_setting_v1_1_certificate.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(cert, file, indent=2, default=float)
        logfile.close()
        return 2

    baseline_rows = [
        row for row in search_rows if abs(row["eps"] - 0.04) <= 1e-12
    ]
    baseline_score = (
        max(row["median_standardized_score"] for row in baseline_rows)
        if baseline_rows
        else None
    )
    selected_row = best_bundle["row"]
    selected_eps = float(selected_row["eps"])
    selected_segment = int(selected_row["segment_1based"]) - 1
    selected_weights = np.asarray(best_bundle["weights"], float)
    frozen_slope = float(selected_row["median_slope_per_mhz"])
    standardized_gain = (
        float(
            selected_row["median_standardized_score"] / baseline_score
        )
        if baseline_score is not None and baseline_score > 0
        else None
    )
    log("\n" + "=" * 92)
    log("FROZEN SELECTION")
    log("=" * 92)
    log(
        f"epsilon={selected_eps:g} | segment={selected_segment+1} | "
        f"chi={frozen_slope:+.7e}/MHz"
    )
    log(
        "weights (rr,rg,gr,gg)="
        + np.array2string(selected_weights, precision=7)
    )
    log(
        f"standardized score={selected_row['median_standardized_score']:.6e}"
        + (
            f" | gain vs eps=0.04={standardized_gain:.3f}x"
            if standardized_gain is not None
            else " | eps=0.04 baseline unavailable"
        )
    )
    log(f"selection elapsed={time.time()-search_started:.1f}s")

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
        "baseline_eps_0p04_standardized_score": baseline_score,
        "standardized_gain_vs_eps_0p04": standardized_gain,
    }
    cert["gates"]["G1_selection_search_completed"] = {
        "epsilon_candidates": len(eps_grid),
        "surviving_epsilon_candidates": len(
            {row["eps"] for row in search_rows}
        ),
        "attempted_program_sets": attempted_program_sets,
        "completed_program_sets": completed_program_sets,
        "pass": True,
    }

    with open(
        os.path.join(args.outdir, "four_setting_v1_1_selection.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(search_rows[0].keys()))
        writer.writeheader()
        writer.writerows(search_rows)

    # The test grid is touched only here, after every design choice is frozen.
    log("\n" + "=" * 92)
    log("HELD-OUT TEST  frozen epsilon, path pair, segment, and outcome score")
    log("=" * 92)
    log(
        f"  {'delta MHz':>10s} {'S exact':>13s} {'prediction':>13s} "
        f"{'rel.err':>9s} {'shots/arm':>13s}"
    )
    selected_paths = best_bundle["paths"]
    z_cw = selected_paths["corrected"]["CW"]["z"]
    z_ccw = selected_paths["corrected"]["CCW"]["z"]
    heldout_rows = []
    heldout_device_pass = True
    for delta_mhz in test_deltas:
        try:
            records = four_evaluate_settings(
                z_cw, z_ccw, selected_segment, delta_mhz
            )
        except Exception as exc:
            heldout_device_pass = False
            log(
                f"  {delta_mhz:10.6f} REJECTED: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        statistics = amplified_contrast_statistics(records)
        witness = amplified_weighted_witness(
            statistics, selected_weights
        )
        prediction = delta_mhz * frozen_slope
        residual = abs(witness["signal"] - prediction)
        relative_error = residual / max(abs(witness["signal"]), 1e-300)
        duration_spread = max(
            record["duration_us"] for record in records.values()
        ) - min(record["duration_us"] for record in records.values())
        row = {
            "delta_mhz": delta_mhz,
            "signal": witness["signal"],
            "prediction": prediction,
            "absolute_residual": residual,
            "relative_error": relative_error,
            "shots_per_arm_5sigma": witness["shots_per_arm_5sigma"],
            "variance_numerator_per_arm": witness[
                "variance_numerator_per_arm"
            ],
            "duration_spread_us": duration_spread,
        }
        heldout_rows.append(row)
        log(
            f"  {delta_mhz:10.6f} {witness['signal']:+13.6e} "
            f"{prediction:+13.6e} {relative_error:9.3%} "
            f"{witness['shots_per_arm_5sigma']:13.3e}"
        )

    if heldout_rows:
        delta_array = np.asarray(
            [row["delta_mhz"] for row in heldout_rows], float
        )
        signal_array = np.asarray(
            [abs(row["signal"]) for row in heldout_rows], float
        )
        signal_exponent = loglog_slope(delta_array, signal_array)
        maximum_relative_error = float(
            max(row["relative_error"] for row in heldout_rows)
        )
        best_shots = float(
            min(row["shots_per_arm_5sigma"] for row in heldout_rows)
        )
        maximum_signal = float(
            max(abs(row["signal"]) for row in heldout_rows)
        )
        duration_pass = all(
            row["duration_spread_us"] <= 1e-9 for row in heldout_rows
        )
    else:
        signal_exponent = float("nan")
        maximum_relative_error = float("inf")
        best_shots = float("inf")
        maximum_signal = 0.0
        duration_pass = False

    baseline_probability_difference = (
        selected_paths["corrected"]["CCW"]["record"]["probabilities"]
        - selected_paths["corrected"]["CW"]["record"]["probabilities"]
    )
    baseline_weighted_difference = float(
        abs(selected_weights @ baseline_probability_difference)
    )
    baseline_pass = baseline_weighted_difference <= 1e-9
    heldout_complete = bool(
        heldout_device_pass
        and len(heldout_rows) == len(test_deltas)
        and duration_pass
    )
    linear_pass = bool(
        0.90 <= signal_exponent <= 1.10
        and maximum_relative_error <= args.heldout_relative_error_tol
    )
    shot_pass = bool(best_shots <= args.shot_threshold)
    gain_pass = bool(
        standardized_gain is not None
        and standardized_gain >= args.target_standardized_gain
    )
    cert["heldout_test"] = {
        "completed_points": len(heldout_rows),
        "declared_points": len(test_deltas),
        "signal_exponent": signal_exponent,
        "maximum_relative_error": maximum_relative_error,
        "maximum_absolute_signal": maximum_signal,
        "best_shots_per_arm_5sigma": best_shots,
    }
    cert["gates"]["G2_selected_modulated_full_unitary_endpoint"] = {
        "selected_eps": selected_eps,
        "CW_endpoint_infidelity": selected_paths["corrected"]["CW"][
            "endpoint_infidelity"
        ],
        "CCW_endpoint_infidelity": selected_paths["corrected"]["CCW"][
            "endpoint_infidelity"
        ],
        "CW_vs_CCW_process_infidelity": selected_paths[
            "pair_unitary_infidelity"
        ],
        "pass": bool(
            selected_paths["endpoint_pass"]
            and selected_paths["duration_pass"]
        ),
    }
    cert["gates"]["G3_zero_probe_weighted_output_match"] = {
        "absolute_weighted_difference": baseline_weighted_difference,
        "threshold": 1e-9,
        "pass": baseline_pass,
    }
    cert["gates"]["G4_heldout_program_acceptance_and_duration"] = {
        "completed_points": len(heldout_rows),
        "declared_points": len(test_deltas),
        "equal_duration": duration_pass,
        "pass": heldout_complete,
    }
    cert["gates"]["G5_heldout_linear_path_susceptibility"] = {
        "signal_exponent": signal_exponent,
        "accepted_exponent_interval": [0.90, 1.10],
        "maximum_relative_error": maximum_relative_error,
        "relative_error_threshold": args.heldout_relative_error_tol,
        "pass": linear_pass,
    }
    cert["gates"]["G6_target_standardized_gain"] = {
        "gain_vs_eps_0p04": standardized_gain,
        "target": args.target_standardized_gain,
        "pass": gain_pass,
    }
    cert["gates"]["G7_four_setting_shot_feasibility"] = {
        "best_shots_per_arm_5sigma": best_shots,
        "threshold": args.shot_threshold,
        "four_independent_arms_per_delta": True,
        "pass": shot_pass,
    }

    with open(
        os.path.join(args.outdir, "four_setting_v1_1_heldout.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        heldout_fields = [
            "delta_mhz",
            "signal",
            "prediction",
            "absolute_residual",
            "relative_error",
            "shots_per_arm_5sigma",
            "variance_numerator_per_arm",
            "duration_spread_us",
        ]
        writer = csv.DictWriter(file, fieldnames=heldout_fields)
        writer.writeheader()
        writer.writerows(heldout_rows)

    controls = {
        "reference": selected_paths["reference_z"].tolist(),
        "CW_modulated_relift": z_cw.tolist(),
        "CCW_modulated_relift": z_ccw.tolist(),
        "selected_eps": selected_eps,
        "selected_probe_segment_1based": selected_segment + 1,
        "outcome_order": ["rr", "rg", "gr", "gg"],
        "frozen_outcome_weights": selected_weights.tolist(),
        "selection_deltas_mhz": selection_deltas,
        "heldout_test_deltas_mhz": test_deltas,
    }
    with open(
        os.path.join(args.outdir, "four_setting_v1_1_controls.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(controls, file, indent=2)

    scientific_support = bool(
        foundation_pass
        and cert["gates"]["G2_selected_modulated_full_unitary_endpoint"][
            "pass"
        ]
        and baseline_pass
        and heldout_complete
        and linear_pass
    )
    if scientific_support and shot_pass:
        status = "AMPLIFIED_FOUR_SETTING_WITNESS_SHOT_FEASIBLE"
        next_step = (
            "Freeze one held-out delta and independently reproduce its four "
            "programs with a second Pulser propagation engine before cloud use."
        )
    elif scientific_support:
        status = "AMPLIFIED_FOUR_SETTING_WITNESS_SUPPORTED_SIGNAL_TOO_SMALL"
        next_step = (
            "The predeclared amplification search remained below the shot "
            "threshold. Do not submit; optimize directly inside the modulated "
            "endpoint fiber rather than expanding epsilon further."
        )
    else:
        status = "AMPLIFIED_FOUR_SETTING_WITNESS_NOT_CLOSED"
        next_step = (
            "Inspect the first failed gate; no path-conditioned derivative "
            "claim should be made from the held-out split."
        )
    cert["scientific_status"] = status
    cert["next_step"] = next_step
    cert["claim_boundary"] = (
        "Two-atom local Pulser AnalogDevice modulation model, exact sampled "
        "coherent propagation, native |gg> preparation and complete native "
        "population outcomes, an epsilon/segment/score selection grid, and a "
        "strictly disjoint held-out probe grid. The optimal score is classical "
        "post-processing of native counts, not tomography. No calibrated "
        "FRESNEL noise, cloud-emulator evidence, or QPU evidence."
    )
    with open(
        os.path.join(args.outdir, "four_setting_v1_1_certificate.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(cert, file, indent=2, default=float)

    log("\n" + "=" * 92)
    log("GLOBAL VERDICT")
    log("=" * 92)
    for gate_name, gate in cert["gates"].items():
        log(f"  {'PASS' if gate['pass'] else 'FAIL':5s} {gate_name}")
    log(f"\nscientific_status={status}")
    log(
        f"selected epsilon={selected_eps:g} | standardized gain="
        f"{standardized_gain if standardized_gain is not None else 'N/A'}"
    )
    log(f"best held-out shots/arm={best_shots:.3e}")
    log(f"next={next_step}")
    log(
        "written: four_setting_v1_1_certificate.json, "
        "four_setting_v1_1_controls.json, "
        "four_setting_v1_1_selection.csv, "
        "four_setting_v1_1_heldout.csv, four_setting_v1_1_run.log"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
        eps_values = sorted({row["eps"] for row in search_rows})
        best_by_eps = [
            max(
                row["median_standardized_score"]
                for row in search_rows
                if row["eps"] == eps
            )
            for eps in eps_values
        ]
        axes[0].plot(eps_values, best_by_eps, "o-")
        axes[0].set_xlabel(r"M2 loop scale $\epsilon$")
        axes[0].set_ylabel("best selection standardized slope")
        axes[0].set_title("selection-only amplification")
        axes[0].grid(alpha=0.3)

        if heldout_rows:
            deltas = np.asarray(
                [row["delta_mhz"] for row in heldout_rows], float
            )
            signals = np.asarray(
                [row["signal"] for row in heldout_rows], float
            )
            axes[1].plot(deltas, signals, "o-", label="held-out exact")
            axes[1].plot(
                deltas,
                deltas * frozen_slope,
                "--",
                label="frozen linear prediction",
            )
        axes[1].set_xlabel(r"$|\delta|$ [MHz]")
        axes[1].set_ylabel(r"$S_w(\delta)$")
        axes[1].set_title("frozen four-setting witness")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(
            os.path.join(
                args.outdir, "four_setting_v1_1_amplification.png"
            ),
            dpi=170,
        )
        log("written: four_setting_v1_1_amplification.png")
    except Exception as exc:
        log(f"(figure skipped: {type(exc).__name__}: {exc})")

    logfile.close()
    return 0


def run_amplified(**kwargs):
    """Notebook entry point for the v1.1 amplified search."""
    argv = []
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            elif key == "quick":
                argv.append("--no-quick")
        else:
            argv.extend([flag, str(value)])
    return amplified_main(argv)


if __name__ == "__main__":
    exit_code = amplified_main()
    if not _in_notebook():
        raise SystemExit(exit_code)


