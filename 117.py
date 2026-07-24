# -*- coding: utf-8 -*-
"""
fixedH_path_area_v7_fixed.py

Fail-fast, reproducible validation for Y.Y.N. Li's two-segment neutral-atom
pulse path-ordering work.

Scientific scope
----------------
- Exact local Pulser/Qutip statevector simulation, not a QPU result.
- Two-segment forward/reverse schedules have exactly matched total duration,
  integrated Rabi area, and weighted-average detuning.
- C_nc_simple = |Delta_2-Delta_1| f(1-f) is used only because total time,
  geometry, N, and Omega are fixed. The full BCH proxy differs by a constant.
- Correlations describe a deterministic parameter scan. No iid significance
  p-values are reported.
- The three-segment extension is a matched-control Magnus diagnostic, not a
  claimed counterexample to the two-segment law.

Major v7 repairs
----------------
- Removes the old subset-column KeyError and records OLS slope/intercept/R^2.
- Makes every scientific gate fail-fast; the program cannot print PASS/DONE
  after a failed core module.
- Replaces the invalid three-segment permutations (which had unequal weighted
  averages) with all six equal-duration permutations and an explicit signed
  second-order Magnus coefficient.
- Renames and documents the Pulser basis conversion; audits it at several
  Pulser-vs-expm points with a hard infidelity threshold.
- Records the actual pulse area Phi = integral Omega dt.
- Reports both all-117 and nonzero-gap-only correlations and avoids the word
  "strictly monotonic" unless an explicit monotonicity gate is satisfied.

Colab install:
    !pip install -q -U pulser==1.8.0 pulser-simulation==1.8.0 pandas numpy matplotlib scipy
Run:
    python fixedH_path_area_v7_fixed.py
"""

import itertools
import math
import json
import time
import sys
import platform
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
# ---------- scipy ----------
try:
    from scipy.linalg import expm
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    expm = None

# ---------- Pulser ----------
from pulser import Pulse, Sequence, Register
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import ConstantWaveform
from pulser_simulation import QutipEmulator

# ---------- constants ----------
CLOCK_NS = 4
MIN_DURATION_NS = 16
MAX_DURATION_NS = 10000
OMEGA = 1.22
SPACING_UM = 8.0
AVG_DETUNING = -0.31
CROSS_VALIDATION_MAX_INFIDELITY = 1.0e-5
ZERO_GAP_ABS_TOL = 5.0e-12
SCHEDULE_MATCH_TOL = 1.0e-12

# ---------- output directory with timestamp ----------
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUTDIR = Path(f"fixedH_path_area_v7_fixed_{TIMESTAMP}")
OUTDIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# JSON sanitizer (handles NaN/Inf)
# ----------------------------------------------------------------------
def json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def package_version(distribution_name):
    """Return installed distribution version without making metadata nonfatal."""
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "unknown"


# ----------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------
def normalize_state(psi):
    psi = np.asarray(psi, dtype=np.complex128).ravel()
    nrm = np.linalg.norm(psi)
    if nrm <= 0:
        raise ValueError("Zero statevector norm.")
    return psi / nrm


def normalize_prob(p):
    p = np.asarray(p, dtype=float).ravel()
    s = np.sum(p)
    if s <= 0:
        raise ValueError("Non-positive probability sum.")
    return p / s


def probs_from_state(psi):
    return normalize_prob(np.abs(normalize_state(psi)) ** 2)


def state_overlap(psi, phi):
    return complex(np.vdot(normalize_state(psi), normalize_state(phi)))


def state_fidelity(psi, phi):
    return float(abs(state_overlap(psi, phi)) ** 2)


def pure_trace_distance(psi, phi):
    return float(math.sqrt(max(0.0, 1.0 - state_fidelity(psi, phi))))


def fubini_study_angle(psi, phi):
    s = abs(state_overlap(psi, phi))
    return float(math.acos(max(0.0, min(1.0, s))))


def tvd_prob(p, q):
    p = normalize_prob(p)
    q = normalize_prob(q)
    return float(0.5 * np.sum(np.abs(p - q)))


def bhattacharyya_coeff(p, q):
    p = normalize_prob(p)
    q = normalize_prob(q)
    return float(np.sum(np.sqrt(p * q)))


def phase_gap_metric(psi, phi):
    p = probs_from_state(psi)
    q = probs_from_state(phi)
    bc = bhattacharyya_coeff(p, q)
    overlap_abs = abs(state_overlap(psi, phi))
    return float(bc - overlap_abs), float(bc), float(overlap_abs)


def pair_metrics(psi_a, psi_b, pair_label):
    p = probs_from_state(psi_a)
    q = probs_from_state(psi_b)
    phase_gap, bc, overlap_abs = phase_gap_metric(psi_a, psi_b)
    return {
        "pair": pair_label,
        "fidelity": state_fidelity(psi_a, psi_b),
        "overlap_abs": overlap_abs,
        "pure_trace_distance": pure_trace_distance(psi_a, psi_b),
        "fubini_study_angle_rad": fubini_study_angle(psi_a, psi_b),
        "TVD_distribution": tvd_prob(p, q),
        "phase_gap_BC_minus_overlap": phase_gap,
        "classical_BC": bc,
    }


# ----------------------------------------------------------------------
# Exact scipy expm reference solver (independent of Pulser ODE)
# ----------------------------------------------------------------------
def exact_hamiltonian_for_segments(n, segments, omega=OMEGA, coords=None):
    """
    Build H(Δt) for a piecewise constant two-segment pulse.
    Returns list of (H, dt) where dt in μs.
    """
    if coords is None:
        coords = nominal_coords(n)
    dim = 2 ** n
    C6 = DigitalAnalogDevice.interaction_coeff  # in MHz * μm^6
    # number operator per qubit
    nq = [np.zeros(dim, dtype=float) for _ in range(n)]
    for idx in range(dim):
        b = format(idx, f"0{n}b")
        for q, ch in enumerate(b):
            if ch == '1':
                nq[q][idx] = 1.0
    # interaction energy (diagonal)
    Hint_diag = np.zeros(dim, dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            v = C6 / (r ** 6)
            Hint_diag += v * (nq[i] * nq[j])
    # H_X (off-diagonal)
    HX = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(dim):
        for q in range(n):
            j = idx ^ (1 << (n-1-q))
            HX[j, idx] += omega / 2.0
    # Nop
    Nop = np.sum(nq, axis=0)
    def H_for_det(det):
        return HX + np.diag(-det * Nop + Hint_diag)   # det sign: Pulser convention -Δ·n
    Hs = []
    for det, dur_ns in segments:
        dt = dur_ns / 1000.0   # μs
        Hs.append((H_for_det(det), dt))
    return Hs


def exact_state_from_segments(n, segments, omega=OMEGA, coords=None):
    """Exact unitary evolution via scipy.linalg.expm (piecewise constant)."""
    Hs = exact_hamiltonian_for_segments(n, segments, omega, coords)
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = 1.0
    for H, dt in Hs:
        psi = expm(-1j * H * dt) @ psi
    return normalize_state(psi)


def cross_validate_pulser_vs_exact(n, segments, omega=OMEGA, coords=None):
    """Return fidelity between Pulser and independent scipy expm evolution."""
    psi_pulser, _ = final_statevector_from_segments(n, segments, omega, coords)
    psi_exact = exact_state_from_segments(n, segments, omega, coords)
    fidelity = state_fidelity(psi_pulser, psi_exact)
    return fidelity, psi_pulser, psi_exact


# ----------------------------------------------------------------------
# Pulser backend (kept for backward compatibility)
# ----------------------------------------------------------------------
def nominal_coords(n, spacing_um=SPACING_UM):
    return np.array([[(i - (n - 1) / 2) * spacing_um, 0.0] for i in range(n)], dtype=float)


def make_register(n, coords=None):
    if coords is None:
        coords = nominal_coords(n)
    return Register({f"q{i}": np.array(coords[i], dtype=float) for i in range(n)})


def add_constant_pulse(seq, omega, detuning, duration_ns, phase=0.0):
    duration_ns = int(duration_ns)
    if duration_ns < MIN_DURATION_NS:
        raise ValueError(
            f"duration_ns={duration_ns} is below device minimum {MIN_DURATION_NS} ns"
        )
    if duration_ns % CLOCK_NS != 0:
        raise ValueError(f"duration_ns={duration_ns} not aligned to {CLOCK_NS} ns")
    omega_wf = ConstantWaveform(duration_ns, float(omega))
    det_wf = ConstantWaveform(duration_ns, float(detuning))
    seq.add(Pulse(omega_wf, det_wf, phase), "rydberg_global")


def build_sequence_explicit(n, segments, omega=OMEGA, coords=None):
    reg = make_register(n, coords=coords)
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    for detuning, duration_ns in segments:
        add_constant_pulse(seq, omega, detuning, int(duration_ns))
    return seq


def get_final_state_array(result):
    """
    Try multiple accessors; record which one succeeded.
    """
    attempts = []
    if hasattr(result, "get_final_state"):
        attempts.append(("get_final_state", lambda: result.get_final_state()))
        attempts.append(("get_final_state(reduce)", lambda: result.get_final_state(reduce_to_basis="ground-rydberg")))
        attempts.append(("get_final_state(ignore_phase)", lambda: result.get_final_state(ignore_global_phase=True)))
    if hasattr(result, "states") and len(result.states) > 0:
        attempts.append(("states[-1]", lambda: result.states[-1]))
    if hasattr(result, "_states") and len(result._states) > 0:
        attempts.append(("_states[-1]", lambda: result._states[-1]))
    last_err = None
    for name, call in attempts:
        try:
            state = call()
            if hasattr(state, "full"):
                arr = np.asarray(state.full()).ravel()
            else:
                arr = np.asarray(state).ravel()
            return arr, name
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Could not access final state. Last error: {repr(last_err)}")


def final_statevector_from_segments(n, segments, omega=OMEGA, coords=None):
    seq = build_sequence_explicit(n, segments, omega=omega, coords=coords)
    sim = QutipEmulator.from_sequence(seq)
    result = sim.run()
    arr, branch = get_final_state_array(result)
    arr = normalize_state(arr)
    if len(arr) != 2 ** n:
        raise RuntimeError(f"Final state dimension {len(arr)} != 2^N={2**n}")
    arr_standard = pulser_to_standard_basis(arr, n)
    return arr_standard, branch


def pulser_to_standard_basis(psi, n):
    """
    Convert Pulser's local |r>,|g> index convention to the standard bit labels
    used by exact_hamiltonian_for_segments, where 0=|g> and 1=|r>.

    This is a bitwise-complement permutation, not a bit-order reversal. Applying
    it to both members of a pair preserves overlap and full-distribution TVD.
    In Pulser-vs-expm validation it is a required basis conversion and is
    therefore audited explicitly at several parameter points.
    """
    psi = normalize_state(psi)
    dim = 2 ** n
    if len(psi) != dim:
        raise ValueError(f"len(psi)={len(psi)} != 2^N={dim}")
    out = np.zeros_like(psi)
    for i in range(dim):
        b = format(i, f"0{n}b")
        fb = "".join("1" if c == "0" else "0" for c in b)
        j = int(fb, 2)
        out[j] = psi[i]
    return normalize_state(out)


def duration_from_loop_ns(n, omega=OMEGA, loop=2.22):
    t_us = 2 * math.pi * loop / (math.sqrt(n) * omega)
    t_ns = int(round(1000 * t_us))
    t_ns = int(round(t_ns / CLOCK_NS) * CLOCK_NS)
    if t_ns < MIN_DURATION_NS or t_ns > MAX_DURATION_NS:
        raise ValueError(f"Computed duration {t_ns} ns outside allowed range [{MIN_DURATION_NS}, {MAX_DURATION_NS}]")
    return t_ns


def split_duration_clock(total_ns, frac):
    total_ns = int(round(total_ns / CLOCK_NS) * CLOCK_NS)
    d1 = int(round(total_ns * frac / CLOCK_NS) * CLOCK_NS)
    d1 = max(MIN_DURATION_NS, min(total_ns - MIN_DURATION_NS, d1))
    d2 = total_ns - d1
    if d2 < MIN_DURATION_NS:
        raise ValueError("Second segment is below the minimum duration.")
    return int(d1), int(d2), int(total_ns)


def weighted_avg(det1, det2, d1_ns, d2_ns):
    return float((det1 * d1_ns + det2 * d2_ns) / (d1_ns + d2_ns))


def integrated_rabi_area(segments, omega=OMEGA):
    """Dimensionless pulse area Phi = integral Omega dt, with dt in microseconds."""
    return float(omega * sum(dur_ns for _, dur_ns in segments) / 1000.0)


def commutator_norm_proxy(n):
    dim = 2 ** n
    HX = np.zeros((dim, dim), dtype=np.complex128)
    Nop = np.zeros((dim, dim), dtype=np.complex128)
    for x in range(dim):
        weight = bin(x).count("1")
        Nop[x, x] = weight
        for q in range(n):
            y = x ^ (1 << q)
            HX[y, x] += OMEGA / 2.0
    comm = HX @ Nop - Nop @ HX
    return float(np.linalg.norm(comm, ord=2))


# ----------------------------------------------------------------------
# N=2 fixed-H test
# ----------------------------------------------------------------------
def best_shared_output_certificate(psi_f, psi_r):
    ov = state_overlap(psi_f, psi_r)
    s = abs(ov)
    phi_best = normalize_state(psi_f + np.exp(-1j * np.angle(ov)) * psi_r)
    return {
        "target_overlap_abs": float(s),
        "target_fidelity": float(s**2),
        "target_pure_trace_distance": float(math.sqrt(max(0.0, 1.0 - s**2))),
        "shared_best_total_infidelity_loss_analytic": float(1.0 - s),
        "shared_best_fidelity_each_analytic": float((1.0 + s) / 2.0),
        "shared_best_fidelity_to_forward": state_fidelity(psi_f, phi_best),
        "shared_best_fidelity_to_reverse": state_fidelity(psi_r, phi_best),
        "interpretation": (
            "For a single shared Hamiltonian (time-independent or time-dependent) starting from the same ψ0 "
            "and same total time, the output is a single state. It cannot match two different target states. "
            "The residual 1-s is optimizer-independent. This is a statement about single-output maps, not "
            "a Hamiltonian-learning certificate. The TVD between targets is only ~1e-5, so the difference is "
            "phase-dominated and would require ~1e10 shots to resolve in population."
        ),
    }


def run_n2_fixedH():
    print("\n" + "="*80 + "\nA) N=2 CLOCK4 TARGETS + SHARED FIXED-H CERTIFICATE\n" + "="*80)
    T_ns = 2832 + 5256
    avg_det = weighted_avg(-0.38, -0.25, 2832, 5256)
    forward_segments = [(-0.38, 2832), (-0.25, 5256)]
    reverse_segments = [(-0.25, 5256), (-0.38, 2832)]
    avg_segments = [(avg_det, T_ns)]
    # base is defined but not used for path-ordering; we'll keep but note
    psi_f, _ = final_statevector_from_segments(2, forward_segments)
    psi_r, _ = final_statevector_from_segments(2, reverse_segments)
    psi_avg, _ = final_statevector_from_segments(2, avg_segments)
    # optional base: skip to keep clean
    cert = best_shared_output_certificate(psi_f, psi_r)
    cert["N"] = 2
    cert["T_ns"] = T_ns
    cert["forward_segments"] = forward_segments
    cert["reverse_segments"] = reverse_segments
    cert["avg_segments"] = avg_segments
    # We also compute pair metrics
    pairs = [
        ("forward_vs_reverse", psi_f, psi_r),
        ("forward_vs_avg", psi_f, psi_avg),
        ("reverse_vs_avg", psi_r, psi_avg),
    ]
    pair_rows = []
    for label, a, b in pairs:
        m = pair_metrics(a, b, label)
        pair_rows.append(m)
    pair_df = pd.DataFrame(pair_rows)
    print(pair_df.to_string(index=False))
    # Save
    pair_csv = OUTDIR / "n2_pair_metrics_clock4.csv"
    cert_json = OUTDIR / "n2_fixedH_certificate.json"
    pair_df.to_csv(pair_csv, index=False)
    with open(cert_json, "w") as f:
        json.dump(json_sanitize(cert), f, indent=2, allow_nan=False)
    print(f"Saved: {pair_csv}, {cert_json}")
    return psi_f, psi_r, psi_avg, cert, pair_df


# ----------------------------------------------------------------------
# Matched three-segment Magnus diagnostic
# ----------------------------------------------------------------------
def signed_second_order_detuning_area(segments):
    """
    Coefficient multiplying [H_X,N] in the pairwise second-order Magnus sum,
    up to the conventional overall factor/sign:

        A2 = sum_{i>j} (Delta_i - Delta_j) t_i t_j.

    Times are expressed in microseconds, so A2 has the corresponding squared
    time factor. Its sign changes under exact schedule reversal.
    """
    area = 0.0
    for i in range(len(segments)):
        delta_i, duration_i_ns = segments[i]
        for j in range(i):
            delta_j, duration_j_ns = segments[j]
            area += (
                (delta_i - delta_j)
                * (duration_i_ns / 1000.0)
                * (duration_j_ns / 1000.0)
            )
    return float(area)


def run_3segment_magnus_diagnostic(
    n=8, omega=OMEGA, avg_det=AVG_DETUNING, nominal_total_ns=None
):
    """
    Compare all six temporal permutations of three detunings.

    Every segment has exactly the same clock-aligned duration, so every
    permutation has identical total duration, integrated Rabi area, weighted
    average detuning, and detuning multiset. The only changed variable is order.
    States are compared with the constant-average schedule to test how much the
    signed second-order Magnus coefficient organizes the response. This is a
    diagnostic beyond the paper's two-segment claim, not a proof of a universal
    three-segment law.
    """
    if nominal_total_ns is None:
        nominal_total_ns = duration_from_loop_ns(n)
    duration_each_ns = int(
        round((nominal_total_ns / 3.0) / CLOCK_NS) * CLOCK_NS
    )
    if duration_each_ns < MIN_DURATION_NS:
        raise ValueError("Clock-aligned three-segment duration is too short.")

    total_ns = 3 * duration_each_ns
    detunings = (avg_det - 0.12, avg_det, avg_det + 0.12)
    constant_segments = [(avg_det, total_ns)]
    psi_constant, constant_branch = final_statevector_from_segments(
        n, constant_segments, omega=omega
    )

    reference_total = total_ns
    reference_area = integrated_rabi_area(constant_segments, omega)
    rows = []
    for permutation_index, permutation in enumerate(
        itertools.permutations(detunings), start=1
    ):
        segments = [(delta, duration_each_ns) for delta in permutation]
        total = sum(duration for _, duration in segments)
        avg = sum(delta * duration for delta, duration in segments) / total
        area = integrated_rabi_area(segments, omega)
        assert total == reference_total
        assert abs(avg - avg_det) <= SCHEDULE_MATCH_TOL
        assert abs(area - reference_area) <= SCHEDULE_MATCH_TOL

        psi, branch = final_statevector_from_segments(n, segments, omega=omega)
        metrics = pair_metrics(
            psi, psi_constant, f"permutation_{permutation_index}_vs_constant"
        )
        a2 = signed_second_order_detuning_area(segments)
        row = {
            "permutation_index": permutation_index,
            "detuning_order": "|".join(f"{x:.8f}" for x in permutation),
            "duration_each_ns": duration_each_ns,
            "total_duration_ns": total,
            "pulse_area": area,
            "weighted_avg_detuning": avg,
            "A2_signed": a2,
            "A2_abs": abs(a2),
            "pulser_state_branch": branch,
            "constant_state_branch": constant_branch,
        }
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("permutation_index")
    max_avg_error = float(np.max(np.abs(df["weighted_avg_detuning"] - avg_det)))
    max_area_error = float(np.max(np.abs(df["pulse_area"] - reference_area)))
    if max_avg_error > SCHEDULE_MATCH_TOL or max_area_error > SCHEDULE_MATCH_TOL:
        raise AssertionError("Three-segment integrated-content matching failed.")

    df.to_csv(OUTDIR / "three_segment_magnus_diagnostic.csv", index=False)
    print("\nTHREE-SEGMENT MATCHED MAGNUS DIAGNOSTIC")
    print(
        df[
            [
                "permutation_index",
                "detuning_order",
                "A2_signed",
                "pure_trace_distance",
                "TVD_distribution",
            ]
        ].to_string(index=False)
    )
    print(
        f"Matched total={total_ns} ns, pulse_area={reference_area:.9f}, "
        f"max_avg_error={max_avg_error:.3e}"
    )
    return df


# ----------------------------------------------------------------------
# 117-point scan
# ----------------------------------------------------------------------
def spearman_corr_tiesafe(x, y):
    """Tie-aware Spearman rank correlation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if SCIPY_AVAILABLE:
        rho, _ = spearmanr(x, y)
        return float(rho)
    raise RuntimeError("SciPy is required for tie-aware Spearman correlation.")


def ols_fit_with_intercept(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r2)


def fit_summary_for_witness(sub_df, xcol, ycol, subset_name):
    # sub_df is already a subset with all rows, we don't need to filter by subset column
    x = sub_df[xcol].values.astype(float)
    y = sub_df[ycol].values.astype(float)
    rho = spearman_corr_tiesafe(x, y)
    slope, intercept, r2 = ols_fit_with_intercept(x, y)
    return {
        "subset": subset_name,
        "x": xcol,
        "y": ycol,
        "n": int(len(sub_df)),
        "spearman_rho": rho,
        "ols_slope": slope,
        "ols_intercept": intercept,
        "ols_R2": r2,
    }


def run_117_scan():
    print("\n" + "="*80 + "\nB) 117-POINT STATEVECTOR PATH-AREA SCAN\n" + "="*80)
    n = 8
    total_ns = duration_from_loop_ns(n)
    comm_norm = commutator_norm_proxy(n)
    print(f"N={n}, total_ns={total_ns}, comm_norm={comm_norm:.6f}")
    print("Grid: 9 gaps in [0, 0.24], 13 fractions in [0.10, 0.90]")
    # generate points
    rows = []
    idx = 0
    for gap in np.linspace(0.0, 0.24, 9):
        for frac_nom in np.linspace(0.10, 0.90, 13):
            idx += 1
            d1, d2, T = split_duration_clock(total_ns, frac_nom)
            f_actual = d1 / T
            delta1 = -0.31 - (1.0 - f_actual) * gap
            delta2 = -0.31 + f_actual * gap
            forward_segments = [(delta1, d1), (delta2, d2)]
            reverse_segments = [(delta2, d2), (delta1, d1)]
            # Hard assert schedule matching
            fwd_total = sum(d for _, d in forward_segments)
            rev_total = sum(d for _, d in reverse_segments)
            fwd_area = integrated_rabi_area(forward_segments)
            rev_area = integrated_rabi_area(reverse_segments)
            fwd_avg = weighted_avg(forward_segments[0][0], forward_segments[1][0],
                                   forward_segments[0][1], forward_segments[1][1])
            rev_avg = weighted_avg(reverse_segments[0][0], reverse_segments[1][0],
                                   reverse_segments[0][1], reverse_segments[1][1])
            assert fwd_total == rev_total, "Total duration mismatch"
            assert abs(fwd_area - rev_area) <= SCHEDULE_MATCH_TOL, "Area mismatch"
            assert abs(fwd_avg - rev_avg) <= SCHEDULE_MATCH_TOL, "Avg detuning mismatch"
            assert abs(fwd_avg - AVG_DETUNING) <= SCHEDULE_MATCH_TOL
            # compute states
            psi_f, branch_f = final_statevector_from_segments(n, forward_segments)
            psi_r, branch_r = final_statevector_from_segments(n, reverse_segments)
            m = pair_metrics(psi_f, psi_r, "forward_vs_reverse")
            Cnc_simple = abs(delta2 - delta1) * f_actual * (1.0 - f_actual)
            row = {
                "idx": idx,
                "N": n,
                "gap": float(gap),
                "frac_nominal": float(frac_nom),
                "frac_actual": f_actual,
                "delta1": delta1,
                "delta2": delta2,
                "duration1_ns": d1,
                "duration2_ns": d2,
                "T_ns": T,
                "pulse_area": fwd_area,
                "weighted_avg_detuning": weighted_avg(delta1, delta2, d1, d2),
                "C_nc_simple": Cnc_simple,
                "commutator_norm_proxy": comm_norm,
                "pulser_branch_forward": branch_f,
                "pulser_branch_reverse": branch_r,
            }
            row.update(m)
            rows.append(row)
            print(f"[{idx:03d}/117] gap={gap:.4f} f={f_actual:.4f} "
                  f"D={m['pure_trace_distance']:.6f} "
                  f"Γ={m['phase_gap_BC_minus_overlap']:.6f} "
                  f"TVD={m['TVD_distribution']:.6f}")
    df = pd.DataFrame(rows)

    zero = df[df["gap"] == 0.0]
    zero_maxima = {
        witness: float(np.max(np.abs(zero[witness].to_numpy(dtype=float))))
        for witness in [
            "pure_trace_distance",
            "phase_gap_BC_minus_overlap",
            "TVD_distribution",
        ]
    }
    if any(value > ZERO_GAP_ABS_TOL for value in zero_maxima.values()):
        raise AssertionError(f"Zero-gap witness gate failed: {zero_maxima}")

    df.to_csv(OUTDIR / "scan117_statevector_metrics.csv", index=False)

    # compute correlations
    summaries = []
    for subset_name in ["all_117_including_zero_gap", "nonzero_gap_only"]:
        if subset_name == "all_117_including_zero_gap":
            sub_df = df.copy()
        else:
            sub_df = df[df["gap"] > 0].copy()
        # Only use C_nc_simple (C_nc is proportional)
        for ycol in ["pure_trace_distance", "phase_gap_BC_minus_overlap", "TVD_distribution"]:
            s = fit_summary_for_witness(sub_df, "C_nc_simple", ycol, subset_name)
            summaries.append(s)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTDIR / "scan117_correlation_summary.csv", index=False)

    # generate certificate
    cert = {
        "experiment": "117-point statevector path-area scan",
        "N": n,
        "points": len(df),
        "grid": {"gaps": [float(x) for x in np.linspace(0.0,0.24,9)],
                 "fracs": [float(x) for x in np.linspace(0.10,0.90,13)]},
        "zero_gap_control_points": int((df["gap"]==0).sum()),
        "nonzero_gap_points": int((df["gap"]>0).sum()),
        "total_duration_ns": total_ns,
        "Omega": OMEGA,
        "pulse_area": float(df["pulse_area"].iloc[0]),
        "avg_detuning_target": AVG_DETUNING,
        "commutator_norm_proxy": comm_norm,
        "zero_gap_max_abs_witnesses": zero_maxima,
        "correlations": summary_df.to_dict(orient="records"),
        "note": "Only C_nc_simple is used; C_nc is proportional with constant factor T^2/1e6 * comm_norm.",
        "interpretation_boundary": (
            "These are strong rank/linear associations on a deterministic grid, "
            "not proof of strict monotonicity and not iid significance tests."
        ),
    }
    with open(OUTDIR / "scan117_certificate.json", "w") as f:
        json.dump(json_sanitize(cert), f, indent=2, allow_nan=False)
    print("\nCORRELATION SUMMARY (Spearman ρ)")
    print(
        summary_df[
            [
                "subset",
                "y",
                "n",
                "spearman_rho",
                "ols_slope",
                "ols_intercept",
                "ols_R2",
            ]
        ].to_string(index=False)
    )
    print("Zero-gap max absolute witnesses:", zero_maxima)
    print("\nSaved outputs under", OUTDIR)
    return df, summary_df, cert


# ----------------------------------------------------------------------
# Cross-validation with expm
# ----------------------------------------------------------------------
def run_cross_validation():
    print("\n" + "="*80 + "\nC) MULTI-POINT CROSS-VALIDATION: Pulser vs expm\n" + "="*80)
    test_specs = [
        (2, 0.35, 0.13),
        (5, 0.50, 0.18),
        (8, 0.20, 0.12),
        (8, 0.50, 0.12),
        (8, 0.80, 0.24),
    ]
    rows = []
    for n, fraction, gap in test_specs:
        total_ns = duration_from_loop_ns(n)
        d1, d2, _ = split_duration_clock(total_ns, fraction)
        f_actual = d1 / total_ns
        delta1 = AVG_DETUNING - (1.0 - f_actual) * gap
        delta2 = AVG_DETUNING + f_actual * gap
        segments = [(delta1, d1), (delta2, d2)]
        psi_p, branch = final_statevector_from_segments(n, segments)
        psi_e = exact_state_from_segments(n, segments)
        fidelity = state_fidelity(psi_p, psi_e)
        infidelity = 1.0 - fidelity
        rows.append(
            {
                "N": n,
                "frac_actual": f_actual,
                "gap": gap,
                "segments": repr(segments),
                "fidelity": fidelity,
                "infidelity": infidelity,
                "pulser_branch": branch,
            }
        )
        print(
            f"N={n} f={f_actual:.6f} gap={gap:.3f} "
            f"fidelity={fidelity:.12f} infidelity={infidelity:.3e}"
        )

    cv_df = pd.DataFrame(rows)
    max_infidelity = float(cv_df["infidelity"].max())
    if max_infidelity > CROSS_VALIDATION_MAX_INFIDELITY:
        raise AssertionError(
            f"Pulser/expm cross-validation failed: max infidelity "
            f"{max_infidelity:.3e} > {CROSS_VALIDATION_MAX_INFIDELITY:.3e}"
        )

    probe = np.arange(16, dtype=np.complex128) + 1j * np.arange(16)[::-1]
    probe = normalize_state(probe)
    roundtrip = pulser_to_standard_basis(
        pulser_to_standard_basis(probe, 4), 4
    )
    basis_roundtrip_error = float(np.linalg.norm(probe - roundtrip))
    if basis_roundtrip_error > 1.0e-14:
        raise AssertionError("Pulser basis conversion is not involutive.")

    cv_df.to_csv(OUTDIR / "cross_validation_points.csv", index=False)
    cv_cert = {
        "points": cv_df.to_dict(orient="records"),
        "max_infidelity": max_infidelity,
        "max_allowed_infidelity": CROSS_VALIDATION_MAX_INFIDELITY,
        "basis_roundtrip_error": basis_roundtrip_error,
        "versions": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": package_version("scipy"),
            "pulser": package_version("pulser"),
            "pulser_simulation": package_version("pulser-simulation"),
        },
    }
    with open(OUTDIR / "cross_validation.json", "w") as f:
        json.dump(json_sanitize(cv_cert), f, indent=2, allow_nan=False)
    print(
        f"CROSS-VALIDATION PASS: max infidelity={max_infidelity:.3e}; "
        f"basis roundtrip={basis_roundtrip_error:.3e}"
    )
    return cv_df, cv_cert


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("\n" + "="*80 + "\nFIXED-H + 117-POINT PATH-AREA VALIDATION v7\n" + "="*80)
    print("OUTDIR:", OUTDIR)
    print("This script includes:\n"
          " - N=2 fixed-H certificate (clarified)\n"
          " - 117-point scan with only C_nc_simple\n"
          " - matched 3-segment Magnus diagnostic\n"
          " - multi-point fail-fast expm cross-validation\n"
          " - Hard assertions on schedule matching\n"
          " - Version tracking\n"
          " - Tie-aware correlation\n")
    t0 = time.time()

    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is required; install the declared dependencies.")

    # Every stage is fail-fast. No broad exception handler is used here.
    cv_df, cv_cert = run_cross_validation()
    run_n2_fixedH()
    df3 = run_3segment_magnus_diagnostic()
    df, summary_df, cert = run_117_scan()

    run_summary = {
        "status": "PASS",
        "cross_validation_max_infidelity": cv_cert["max_infidelity"],
        "three_segment_rows": int(len(df3)),
        "scan_points": int(len(df)),
        "output_directory": str(OUTDIR),
    }
    with open(OUTDIR / "run_summary.json", "w") as f:
        json.dump(json_sanitize(run_summary), f, indent=2, allow_nan=False)

    print("\n" + "="*80 + "\nALL SCIENTIFIC GATES PASS\n" + "="*80)
    print(f"Elapsed: {time.time()-t0:.2f} s")
    print(f"Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()
