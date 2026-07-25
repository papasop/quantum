#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EP-OBS-v1.0  --  Observable-level executed-path diagnostic on a PASQAL/Pulser local emulator.

PURPOSE
-------
The manuscript "Ideal Executed Paths Predict Weak-Noise Differences between
Full-Unitary-Equivalent Rydberg Controls" establishes a *model-level* result:
two controls z and z0 with the same complete ideal unitary have different
interaction-picture dissipative response operators K_z, and the Frobenius channel
distance D_E(gamma) = ||E_z - E_z0||_F / 16 is predicted parameter-free by
gamma * ||K_z - K_0||_F / 16.

D_E is not measurable. This script closes the remaining step listed in the
manuscript's Discussion by moving to an *observable-level* quantity that a
neutral-atom machine can actually produce:

    DeltaP(gamma) = Tr[ M ( E_z(T;gamma) - E_z0(T;gamma) )(rho) ]
                  = gamma * chi_{M,rho}[DeltaK] + O(gamma^2),

    chi_{M,rho}[DeltaK] = Tr[ M * unvec( (G_z - G_0) vec(rho) ) ],
    G_z = d/dgamma E_z(T;gamma) |_{gamma=0}  (= U_z(T) K_z).

CONSTRAINTS TAKEN SERIOUSLY (this is the whole point of the script)
-------------------------------------------------------------------
  * rho is NOT optimized over all states. On an analog neutral-atom device the
    only preparable input is |gg>. rho = |gg><gg| is hard-wired.
  * M is NOT an arbitrary Hermitian operator. Readout projects each atom onto
    {|g>, |r>}. Admissible M are therefore diagonal in the computational basis.
  * Basis rotation is obtained by APPENDING A REAL PULSE (a 7th segment, global,
    identical for both controls), simulated with the full interaction Hamiltonian
    AND with the same noise. Nothing is applied "by hand" as a perfect rotation.
  * A shot budget is computed from the exact binomial/multinomial variance.
    If the required shot count is absurd, the script says so. It does not hide it.

SIX STEPS REQUESTED
-------------------
  (1) freeze two full-unitary-equivalent controls              -> Sec. 3 (M2 lift)
  (2) re-verify the ideal endpoint AFTER compilation           -> Sec. 4 (resample gate)
  (3) choose rho and M maximizing |DeltaP|                      -> Sec. 6 (readout scan)
  (4) sweep the controllable noise strength gamma               -> Sec. 7
  (5) predict the slope in advance, no fitting                  -> Sec. 6/7
  (6) observe the linear split                                  -> Sec. 7 + Sec. 9 (feasibility)

HAMILTONIAN CONVENTION
----------------------
Not assumed. Extracted from Pulser at runtime and asserted (Gate H0). Pulser uses
the ordered eigenbasis (|r>, |g>) and

    H = (Omega/2) ( cos(phi) X - sin(phi) Y ) - Delta * N + (C6/a^6) n1 n2 ,

i.e. the manuscript's +sin(phi) Y corresponds to phi -> -phi here.

That relabeling is NOT a symmetry of the result. It is true that
H(-phi) = conj(H(phi)) elementwise (X, N, n1n2 real; Y imaginary), and Gate H1
verifies exactly that. It is false that this makes U(-phi) a simple function of
U(phi): conjugating each factor transposes it, and transposition reverses the
order of a time-ordered product. So the same numerical phase table read under the
two conventions gives genuinely different unitaries and different numerical values
of ||K_z - K_0||, chi and DeltaP. What is convention independent is the structure
(existence of the fiber, its rank, the exponents, the first-order law), not the
numbers. Gate H1 reports |U(-phi) - conj(U(phi))| explicitly BECAUSE it does not
vanish. Select the convention with --phase-sign; it is applied globally through
MODEL["phase_sign"].

USAGE
-----
    python pasqal_local_observable_path_split.py                 # full run
    python pasqal_local_observable_path_split.py --quick         # coarse, ~1 min
    python pasqal_local_observable_path_split.py --eps-sweep     # add design sweep

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
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
from scipy.linalg import expm, expm_frechet

# ----------------------------------------------------------------------------
# 0. PROVENANCE
# ----------------------------------------------------------------------------

VERSION = "EP-OBS-v1.0"


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
RHO0[IDX_GG, IDX_GG] = 1.0  # the only preparable input state

ATOM_SEP = 6.0  # um
C6 = 5420158.53  # rad * um^6 / us   (== pulser MockDevice.interaction_coeff)
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


def unitary_from_compiled_samples(seq) -> np.ndarray:
    """Rebuild U from the waveform Pulser actually schedules (1 ns resolution).

    This is the post-compilation re-verification: it does not trust the analytic
    six-segment description, it reads back what the device would play.
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


def pulser_validation(segs, gamma_list=(0.0, 0.030)):
    """Three separate checks against Pulser, reported separately.

    L1  our hamiltonian() == Pulser's get_hamiltonian(t) at every sample
    L2  exact product of Pulser's OWN H(t) == our Liouville channel
    L3  Pulser's QutipEmulator.run() == our Liouville channel

    L1 and L2 validate the model and the propagator. L3 is a check ON PULSER'S
    SOLVER: it uses an interpolated representation of the sampled coefficient
    arrays and therefore carries an O(1e-3) systematic error on 120 ns square
    pulses. That is orders of magnitude above the path signal, so run() is NOT
    used as the numerical engine here. This is recorded rather than hidden.
    """
    import qutip
    from pulser_simulation import QutipEmulator, SimConfig

    out = {}
    seq = build_pulser_sequence(segs, "mock")
    sim = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
    n_ns = int(round(sum(sg.tau for sg in segs) * 1000))

    # L1 and L2 share one pass: L1 compares H at EVERY sample, L2 propagates it
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
    p_exact_pulserH = np.abs(U @ np.array([0, 0, 0, 1], dtype=complex)) ** 2
    p_internal = probs_of(channel(segs, 0.0))
    out["L2_exact_propagation_max_abs_err"] = float(
        np.max(np.abs(p_exact_pulserH - p_internal))
    )

    # L3
    l3 = {}
    for g in gamma_list:
        if g == 0.0:
            res = sim.run(progress_bar=False)
            pv = np.abs(np.asarray(res.states[-1].full()).ravel()) ** 2
        else:
            cfg = SimConfig(
                noise=("eff_noise",),
                eff_noise_opers=[qutip.Qobj(NN)],
                eff_noise_rates=[float(g)],
            )
            s2 = QutipEmulator.from_sequence(seq, sampling_rate=1.0, config=cfg)
            res = s2.run(progress_bar=False)
            pv = np.real(np.diag(np.asarray(res.states[-1].full())))
        l3[f"gamma_{g}"] = float(np.max(np.abs(pv - probs_of(channel(segs, g)))))
    out["L3_qutip_run_max_abs_err"] = l3
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
    """What a real AnalogDevice would execute, and what that does to the result.

    Two distinct effects are separated:
      (a) phase-jump idle gaps  -> deterministic, modelled exactly by MODEL['gap_tau']
      (b) finite modulation bandwidth (8 MHz) -> smears the 120 ns segment edges,
          is NOT in the model, and breaks the exact endpoint equivalence.

    Effect (b) produces a *coherent* difference between the two controls that is
    present already at gamma = 0. If it exceeds the path-induced weak-noise split,
    the observable-level experiment cannot be run on that device as specified.
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
        f"{out['z0_duration_us']:.3f} us  (gap-free model: "
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
    ap = argparse.ArgumentParser(prog="pasqal_local_observable_path_split.py")
    ap.add_argument("--quick", action="store_true", help="coarse transport + coarse scan")
    ap.add_argument("--eps", type=float, default=0.040)
    ap.add_argument("--hs", type=float, default=0.002)
    ap.add_argument("--eps-sweep", action="store_true")
    ap.add_argument("--step-halving", action="store_true",
                    help="audit the h_s dependence of ||K_z-K_0||_F: it is NOT a "
                         "converged number, and this reports how far from "
                         "converged it is")
    ap.add_argument("--no-pulser-check", action="store_true")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--gamma-max-extra", type=float, default=0.0,
                    help="if >0, append extra gammas up to this value")
    ap.add_argument("--gap", type=float, default=0.0,
                    help="idle gap in us inserted between drive segments; use "
                         "0.340 to model AnalogDevice phase-jump delays")
    ap.add_argument("--omega-scale", type=float, default=1.0,
                    help="global rescale of the reference Rabi table; 0.8 brings "
                         "it inside AnalogDevice's 2 MHz amplitude limit")
    ap.add_argument("--phase-sign", type=float, default=1.0, choices=[1.0, -1.0],
                    help="+1 = Pulser convention (default), -1 = manuscript's "
                         "+sin(phi) Y convention applied to the same phase table")
    ap.add_argument("--source-path", default=None,
                    help="path to the archived source, for the provenance digest "
                         "when running from a notebook cell (no __file__)")
    ap.add_argument("--analog-report", action="store_true",
                    help="compile onto AnalogDevice and quantify modulation damage")
    if argv is None:
        argv = [] if _in_notebook() else sys.argv[1:]
    # parse_known_args so that a host environment injecting its own flags
    # (Colab's "-f kernel-*.json") cannot kill the run
    args, ignored = ap.parse_known_args(list(argv))
    if ignored:
        print(f"[note] ignoring arguments not belonging to this script: {ignored}")

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
            "input_state": "|gg><gg| (only preparable state)",
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
    # TRUE statement, verified here: H(-phi) = conj(H(phi)) elementwise.
    # FALSE statement, explicitly NOT claimed: that this makes U(-phi) a simple
    # function of U(phi). It does not - conjugating each factor transposes it,
    # and transposition reverses the order of the time-ordered product. So the
    # manuscript's +sin(phi) Y convention and Pulser's -sin(phi) Y convention,
    # applied to the SAME numerical phase table, give genuinely different
    # unitaries and different numerical values of ||K_z - K_0||. The structure
    # of the result (fiber existence, rank, exponents, first-order law) is
    # convention independent; the numbers are not. Use --phase-sign -1 to run
    # the manuscript convention.
    h_conj_err = float(
        np.max(np.abs(hamiltonian(1.3, -0.7, -0.9) - hamiltonian(1.3, -0.7, 0.9).conj()))
    )
    U_plus = unitary_of_z(np.zeros(18), phase_sign=+1.0)
    U_minus = unitary_of_z(np.zeros(18), phase_sign=-1.0)
    u_flip_gap = float(np.max(np.abs(U_minus - U_plus.conj())))
    cert["gates"]["H1_phase_convention_bookkeeping"] = {
        "H_minus_phi_equals_conj_H": h_conj_err,
        "U_flip_vs_conj_U_gap": u_flip_gap,
        "pass": bool(h_conj_err < 1e-14),
        "note": "gate is on the Hamiltonian identity only; the unitary gap is "
                "reported because it is NOT expected to vanish",
    }
    log(f"[H1] H(-phi) == conj(H(phi)) : max|dH| = {h_conj_err:.3e}")
    log(f"     (for the record, |U(-phi) - conj(U(phi))| = {u_flip_gap:.3e}, "
        f"nonzero by construction - the ordered product does not conjugate)")

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

    # ---------------- 2. re-verify AFTER compilation -------------------------
    log("\n" + "-" * 78)
    log("STEP 2  compile to Pulser and re-verify the ideal endpoint from the")
    log("        waveform actually scheduled (not from the analytic segments)")
    log("-" * 78)
    if not args.no_pulser_check:
        comp = {}
        for tag, z in [("z0", np.zeros(18))] + [(d, results[d]["z"]) for d in results]:
            segs = segments_of(z)
            seq = build_pulser_sequence(segs, "mock")
            Uc = unitary_from_compiled_samples(seq)
            Ua = unitary_of_z(z)
            comp[tag] = {
                "compiled_vs_analytic_maxabs": float(np.max(np.abs(Uc - Ua))),
                "compiled_eps_U_vs_U0": float(
                    abs(1.0 - abs(np.trace(U0.conj().T @ Uc)) ** 2 / DIM**2)
                ),
            }
            log(f"  {tag:4s}  ||U_compiled - U_analytic||_max = "
                f"{comp[tag]['compiled_vs_analytic_maxabs']:.3e}   "
                f"eps_U(compiled vs U0) = {comp[tag]['compiled_eps_U_vs_U0']:.3e}")
        cert["gates"]["G5_post_compilation_endpoint"] = comp
        cert["gates"]["G5_post_compilation_endpoint"]["pass"] = bool(
            all(v["compiled_eps_U_vs_U0"] <= 1e-11 for v in comp.values())
        )
        # AnalogDevice feasibility (honest report, not a pass/fail of the physics)
        try:
            build_pulser_sequence(segments_of(np.zeros(18)), "analog")
            analog_msg = "AnalogDevice accepted the reference sequence"
            analog_ok = True
        except Exception as exc:
            analog_msg = f"AnalogDevice REJECTED: {type(exc).__name__}: {exc}"
            analog_ok = False
        log(f"  [device] {analog_msg}")
        cert["gates"]["G5b_analog_device_compilable"] = {
            "ok": analog_ok, "message": analog_msg
        }

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
        log(f"  L2  exact product of Pulser's own H(t) : {val['L2_exact_propagation_max_abs_err']:.3e}")
        for k, v in val["L3_qutip_run_max_abs_err"].items():
            log(f"  L3  QutipEmulator.run(), {k:14s}: {v:.3e}")
        sig = abs(design[d0]["dP"][-1])
        worst_l3 = max(val["L3_qutip_run_max_abs_err"].values())
        log(f"\n  path signal to be resolved             : {sig:.3e}")
        log(f"  QutipEmulator.run() systematic error   : {worst_l3:.3e}"
            f"   ({worst_l3/max(sig,1e-300):.1f}x the signal)")
        log("  -> the emulator's ODE path is unusable here; all numbers above")
        log("     come from exact Liouville propagation, validated by L1 and L2.")
        cert["gates"]["G11a_model_matches_pulser"] = {
            "L1": val["L1_hamiltonian_max_abs_err"],
            "L2": val["L2_exact_propagation_max_abs_err"],
            "threshold": 1e-10,
            "pass": bool(val["L1_hamiltonian_max_abs_err"] < 1e-10
                         and val["L2_exact_propagation_max_abs_err"] < 1e-10),
        }
        cert["gates"]["G11b_qutip_run_resolves_signal"] = {
            "worst_abs_err": worst_l3,
            "signal": sig,
            "ratio_err_over_signal": float(worst_l3 / max(sig, 1e-300)),
            "pass": bool(worst_l3 < 0.1 * sig),
            "note": "expected FAIL: QutipEmulator.run() interpolates the sampled "
                    "waveform and carries ~1e-3 systematic error on square pulses. "
                    "Recorded as a documented limitation, not used as the engine.",
        }

    # ---------------- AnalogDevice reality report ----------------------------
    if args.analog_report and not args.no_pulser_check:
        log("\n" + "-" * 78)
        log("HARDWARE REALITY  compile onto AnalogDevice; separate the two effects")
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
        if rep.get("compilable"):
            sig = abs(design[d0]["dP"][-1])
            coh = rep.get(f"{d0}_coherent_dP_from_modulation", float("inf"))
            log(f"\n  path signal at gamma={design[d0]['gammas'][-1]:g}: |dP| = {sig:.3e}")
            log(f"  modulation artefact at gamma=0 : |dP| = {coh:.3e}")
            log(f"  signal / artefact = {sig / max(coh, 1e-300):.3e}")
            cert["gates"]["G13_signal_over_modulation_artefact"] = {
                "signal": sig, "artefact": coh,
                "ratio": float(sig / max(coh, 1e-300)),
                "threshold": 10.0,
                "pass": bool(sig / max(coh, 1e-300) >= 10.0),
                "note": "a FAIL means the control must be re-lifted on the "
                        "modulated schedule before any hardware run",
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
            log(f"        Omega/2pi in [{np.min(om)/(2*np.pi):.2f},"
                f"{np.max(om)/(2*np.pi):.2f}] MHz  (AnalogDevice limit 2.00 MHz)")
        # design recommendation
        DEV_LIMIT_MHZ = 2.0  # AnalogDevice rydberg_global max_amp / 2pi
        ok = [r for r in sweep
              if r["max_Omega_over_2pi"] <= DEV_LIMIT_MHZ
              and min(r["per_gamma"][f"gamma_{g}"]["shots_5sigma"]
                      for g in gam_probe) <= 1e7]
        over = [r for r in sweep
                if r["max_Omega_over_2pi"] > DEV_LIMIT_MHZ
                and min(r["per_gamma"][f"gamma_{g}"]["shots_5sigma"]
                        for g in gam_probe) <= 1e7]
        log("")
        if ok:
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

    all_pass = all(
        v.get("pass", True) for v in cert["gates"].values() if isinstance(v, dict)
    )
    cert["ALL_GATES_PASS"] = bool(all_pass)
    with open(os.path.join(args.outdir, "ep_obs_certificate.json"), "w") as fh:
        json.dump(cert, fh, indent=2, default=float)

    log("\n" + "=" * 78)
    log("GATE SUMMARY")
    log("=" * 78)
    for name, val in cert["gates"].items():
        if isinstance(val, dict) and "pass" in val:
            log(f"  {'PASS' if val['pass'] else 'FAIL'}  {name}")
    log(f"\n  ALL GATES PASS: {all_pass}")
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

        import pasqal_local_observable_path_split as ep
        ep.run(quick=True, analog_report=True, outdir="out")

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
    return main(argv)


if __name__ == "__main__":
    main()

