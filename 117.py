# -*- coding: utf-8 -*-
"""
fixedH_path_order_response_v9.py

Fail-fast response-geometry validation for Y.Y.N. Li's two-segment
neutral-atom pulse-ordering work.

Scientific scope
----------------
- Independent dense scipy evolution is the primary scan backend; local
  Pulser/Qutip statevectors are used for sampled cross-validation.
- Two-segment forward/reverse schedules have exactly matched total duration,
  integrated Rabi area, and weighted-average detuning.
- Two competing fixed-T schedule-shape proxies are compared rather than
  presupposing a path-area law:
    C_BCH = |g| f(1-f)
    C_lin = |g| min(f,1-f)
  With only one T, the scan cannot determine T versus T^2 scaling.
- Correlations describe a deterministic parameter scan. No iid significance
  p-values are reported.
- Signed-gap symmetry, f-reflection even/odd response, zero-gap numerical
  floors, and equal-A2 three-segment degeneracies are reported explicitly.

Major v8 repairs
----------------
- Moves the zero-gap gate to infidelity/TVD space and adds a weakest-signal to
  numerical-floor ratio.
- Uses scipy evolution for the full scan and adds the weakest g=0.03 point to
  Pulser cross-validation.
- Compares BCH and linear-response-support proxies with residual tables.
- Adds the exact signed-gap schedule identity and numerical witness gate:
    D(1-f,+g) = D(f,-g).
- Decomposes positive-gap witnesses into f-reflection even and odd parts.
- Turns equal-A2 three-segment pairs into a decisive insufficiency diagnostic
  and measures one-pulse versus three-identical-pulse segmentation artifacts.
- Records script SHA256, pip freeze, per-point timing, and installed versions.

Major v9 extension
------------------
- Adds the missing exact-reversal direction: fixed detuning multiset,
  fixed total duration, fixed weighted-average detuning, and variable
  three-segment time partitions.
- Scans three reversal-pair topologies at every partition and compares the two
  topologies that have exactly identical |A2| but different temporal placement.
- Separates "A2 is a strong organizer" from the stricter and independently
  testable statement "A2 is sufficient for reversal pairs."

Colab install:
    !pip install -q -U pulser==1.8.0 pulser-simulation==1.8.0 pandas numpy matplotlib scipy
Run:
    python fixedH_path_order_response_v9.py
"""

import hashlib
import itertools
import math
import json
import subprocess
import time
import sys
import platform
from importlib import metadata
from functools import lru_cache
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
ZERO_GAP_INFIDELITY_TOL = 1.0e-10
ZERO_GAP_TVD_TOL = 1.0e-7
MIN_SIGNAL_TO_FLOOR = 50.0
SYMMETRY_ABS_TOL = 2.0e-10
SCHEDULE_MATCH_TOL = 1.0e-12

# ---------- output directory with timestamp ----------
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUTDIR = Path(f"fixedH_path_order_response_v9_{TIMESTAMP}")
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
    # Pulser convention: angular-frequency interaction coefficient,
    # in rad * micrometer^6 / microsecond.
    C6 = DigitalAnalogDevice.interaction_coeff
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
    if coords is None:
        return exact_state_from_segments_cached(
            n,
            tuple((float(det), int(dur)) for det, dur in segments),
            float(omega),
        ).copy()
    Hs = exact_hamiltonian_for_segments(n, segments, omega, coords)
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = 1.0
    for H, dt in Hs:
        psi = expm(-1j * H * dt) @ psi
    return normalize_state(psi)


@lru_cache(maxsize=None)
def default_geometry_propagator(n, detuning, duration_ns, omega):
    """Cached dense propagator for the declared default geometry."""
    H, dt = exact_hamiltonian_for_segments(
        n, [(detuning, duration_ns)], omega=omega, coords=None
    )[0]
    return expm(-1j * H * dt)


@lru_cache(maxsize=None)
def exact_state_from_segments_cached(n, segments_tuple, omega):
    """Cache schedule states and their reused segment propagators."""
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = 1.0
    for detuning, duration_ns in segments_tuple:
        propagator = default_geometry_propagator(
            n, detuning, duration_ns, omega
        )
        psi = propagator @ psi
    normalized = normalize_state(psi)
    normalized.setflags(write=False)
    return normalized


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
    if duration_ns > MAX_DURATION_NS:
        raise ValueError(
            f"duration_ns={duration_ns} exceeds maximum {MAX_DURATION_NS} ns"
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


def make_two_segment_schedules(
    total_ns, duration1_ns, signed_gap, center_detuning=AVG_DETUNING
):
    """
    Construct matched forward/reverse schedules from an actual clock duration.

    signed_gap = Delta_2 - Delta_1 may be positive or negative. Both schedules
    have weighted-average detuning center_detuning exactly.
    """
    duration1_ns = int(duration1_ns)
    duration2_ns = int(total_ns - duration1_ns)
    if min(duration1_ns, duration2_ns) < MIN_DURATION_NS:
        raise ValueError("Two-segment schedule contains an undersized segment.")
    f_actual = duration1_ns / total_ns
    delta1 = center_detuning - (1.0 - f_actual) * signed_gap
    delta2 = center_detuning + f_actual * signed_gap
    forward = [(delta1, duration1_ns), (delta2, duration2_ns)]
    reverse = [(delta2, duration2_ns), (delta1, duration1_ns)]
    return forward, reverse, f_actual


def fixed_commutator_norm(n):
    """Fixed metadata value ||[H_X,N]|| for the declared N and Omega."""
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
            "a Hamiltonian-learning certificate."
        ),
    }


def run_n2_fixedH():
    print("\n" + "="*80 + "\nA) N=2 CLOCK4 TARGETS + SHARED FIXED-H CERTIFICATE\n" + "="*80)
    T_ns = 2832 + 5256
    avg_det = weighted_avg(-0.38, -0.25, 2832, 5256)
    forward_segments = [(-0.38, 2832), (-0.25, 5256)]
    reverse_segments = [(-0.25, 5256), (-0.38, 2832)]
    avg_segments = [(avg_det, T_ns)]
    psi_f, _ = final_statevector_from_segments(2, forward_segments)
    psi_r, _ = final_statevector_from_segments(2, reverse_segments)
    psi_avg, _ = final_statevector_from_segments(2, avg_segments)
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
    fr_tvd = float(
        pair_df.loc[
            pair_df["pair"] == "forward_vs_reverse", "TVD_distribution"
        ].iloc[0]
    )
    cert["forward_reverse_TVD"] = fr_tvd
    cert["rough_population_shot_scale_1_over_TVD2"] = (
        float(1.0 / fr_tvd**2) if fr_tvd > 0 else None
    )
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
    n=8,
    omega=OMEGA,
    avg_det=AVG_DETUNING,
    nominal_total_ns=None,
    numerical_floor_D=0.0,
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
    constant_single = [(avg_det, total_ns)]
    constant_three = [(avg_det, duration_each_ns)] * 3
    psi_constant = exact_state_from_segments(n, constant_single, omega=omega)
    psi_constant_three = exact_state_from_segments(
        n, constant_three, omega=omega
    )
    segmentation_exact = pair_metrics(
        psi_constant,
        psi_constant_three,
        "constant_one_pulse_vs_three_identical_expm",
    )

    psi_constant_pulser, branch_single = final_statevector_from_segments(
        n, constant_single, omega=omega
    )
    psi_constant_three_pulser, branch_three = final_statevector_from_segments(
        n, constant_three, omega=omega
    )
    segmentation_pulser = pair_metrics(
        psi_constant_pulser,
        psi_constant_three_pulser,
        "constant_one_pulse_vs_three_identical_pulser",
    )

    reference_total = total_ns
    reference_area = integrated_rabi_area(constant_single, omega)
    rows = []
    permutation_states = {}
    low_detuning, middle_detuning, high_detuning = detunings
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

        psi = exact_state_from_segments(n, segments, omega=omega)
        permutation_states[permutation_index] = psi
        metrics = pair_metrics(
            psi, psi_constant, f"permutation_{permutation_index}_vs_constant"
        )
        a2 = signed_second_order_detuning_area(segments)
        row = {
            "permutation_index": permutation_index,
            "detuning_order": "|".join(f"{x:.8f}" for x in permutation),
            "first_segment_role": (
                "low"
                if permutation[0] == low_detuning
                else "mid"
                if permutation[0] == middle_detuning
                else "high"
            ),
            "duration_each_ns": duration_each_ns,
            "total_duration_ns": total,
            "pulse_area": area,
            "weighted_avg_detuning": avg,
            "A2_signed": a2,
            "A2_abs": abs(a2),
        }
        row.update(metrics)
        row["rough_shot_scale_1_over_TVD2"] = (
            float(1.0 / metrics["TVD_distribution"] ** 2)
            if metrics["TVD_distribution"] > 0
            else None
        )
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("permutation_index")
    max_avg_error = float(np.max(np.abs(df["weighted_avg_detuning"] - avg_det)))
    max_area_error = float(np.max(np.abs(df["pulse_area"] - reference_area)))
    if max_avg_error > SCHEDULE_MATCH_TOL or max_area_error > SCHEDULE_MATCH_TOL:
        raise AssertionError("Three-segment integrated-content matching failed.")

    # For equal durations, A2 reduces exactly to 2*tau^2*(Delta_3-Delta_1).
    tau_us = duration_each_ns / 1000.0
    expected_a2 = []
    for order in df["detuning_order"]:
        values = [float(x) for x in order.split("|")]
        expected_a2.append(2.0 * tau_us**2 * (values[2] - values[0]))
    a2_closed_form_error = float(
        np.max(np.abs(df["A2_signed"].to_numpy() - np.asarray(expected_a2)))
    )
    if a2_closed_form_error > 1.0e-12:
        raise AssertionError("Equal-duration A2 closed-form identity failed.")

    def witness_at(index, column="pure_trace_distance"):
        return float(
            df.loc[df["permutation_index"] == index, column].iloc[0]
        )

    equal_a2_differences = {
        "D_perm2_minus_perm3_abs": abs(witness_at(2) - witness_at(3)),
        "D_perm4_minus_perm5_abs": abs(witness_at(4) - witness_at(5)),
        "TVD_perm2_minus_perm3_abs": abs(
            witness_at(2, "TVD_distribution")
            - witness_at(3, "TVD_distribution")
        ),
        "TVD_perm4_minus_perm5_abs": abs(
            witness_at(4, "TVD_distribution")
            - witness_at(5, "TVD_distribution")
        ),
    }
    decision_threshold_D = max(
        MIN_SIGNAL_TO_FLOOR * numerical_floor_D,
        MIN_SIGNAL_TO_FLOOR
        * float(segmentation_exact["pure_trace_distance"]),
        1.0e-10,
    )
    decision_threshold_TVD = max(
        MIN_SIGNAL_TO_FLOOR
        * float(segmentation_exact["TVD_distribution"]),
        1.0e-10,
    )
    max_equal_a2_D_difference = max(
        equal_a2_differences["D_perm2_minus_perm3_abs"],
        equal_a2_differences["D_perm4_minus_perm5_abs"],
    )
    max_equal_a2_TVD_difference = max(
        equal_a2_differences["TVD_perm2_minus_perm3_abs"],
        equal_a2_differences["TVD_perm4_minus_perm5_abs"],
    )
    tvd_non_reversal_insufficiency_resolved = (
        max_equal_a2_TVD_difference > decision_threshold_TVD
    )
    d_non_reversal_insufficiency_resolved = (
        max_equal_a2_D_difference > decision_threshold_D
    )

    first_segment_summary = (
        df.groupby("first_segment_role", sort=False)
        .agg(
            count=("permutation_index", "count"),
            mean_D=("pure_trace_distance", "mean"),
            min_D=("pure_trace_distance", "min"),
            max_D=("pure_trace_distance", "max"),
            mean_TVD=("TVD_distribution", "mean"),
            min_TVD=("TVD_distribution", "min"),
            max_TVD=("TVD_distribution", "max"),
            mean_rough_shots=("rough_shot_scale_1_over_TVD2", "mean"),
        )
        .reset_index()
    )
    mid_mean_tvd = float(
        first_segment_summary.loc[
            first_segment_summary["first_segment_role"] == "mid", "mean_TVD"
        ].iloc[0]
    )
    nonmid_mean_tvd = float(
        df.loc[df["first_segment_role"] != "mid", "TVD_distribution"].mean()
    )
    first_segment_suppression_ratio = nonmid_mean_tvd / mid_mean_tvd

    reversal_rows = []
    for left_index, right_index in [(1, 6), (2, 4), (3, 5)]:
        reversal_metrics = pair_metrics(
            permutation_states[left_index],
            permutation_states[right_index],
            f"permutation_{left_index}_vs_reverse_{right_index}",
        )
        reversal_metrics["permutation_left"] = left_index
        reversal_metrics["permutation_right"] = right_index
        left_a2 = abs(
            float(
                df.loc[
                    df["permutation_index"] == left_index, "A2_signed"
                ].iloc[0]
            )
        )
        right_a2 = abs(
            float(
                df.loc[
                    df["permutation_index"] == right_index, "A2_signed"
                ].iloc[0]
            )
        )
        if abs(left_a2 - right_a2) > 1.0e-12:
            raise AssertionError("Reversal pair does not have matched |A2|.")
        reversal_metrics["A2_abs_each_schedule"] = left_a2
        reversal_metrics["D_over_A2_abs"] = (
            reversal_metrics["pure_trace_distance"] / left_a2
        )
        reversal_metrics["TVD_over_A2_abs"] = (
            reversal_metrics["TVD_distribution"] / left_a2
        )
        reversal_metrics["rough_shot_scale_1_over_TVD2"] = (
            float(1.0 / reversal_metrics["TVD_distribution"] ** 2)
            if reversal_metrics["TVD_distribution"] > 0
            else None
        )
        reversal_rows.append(reversal_metrics)
    reversal_df = pd.DataFrame(reversal_rows)

    def relative_span(values):
        values = np.asarray(values, dtype=float)
        return float((np.max(values) - np.min(values)) / np.mean(values))

    reversal_pair_scaling = {
        "D_over_abs_A2_mean": float(reversal_df["D_over_A2_abs"].mean()),
        "D_over_abs_A2_relative_span": relative_span(
            reversal_df["D_over_A2_abs"]
        ),
        "TVD_over_abs_A2_mean": float(
            reversal_df["TVD_over_A2_abs"].mean()
        ),
        "TVD_over_abs_A2_relative_span": relative_span(
            reversal_df["TVD_over_A2_abs"]
        ),
        "interpretation": (
            "Across these three exact reversal pairs, |A2| remains a strong "
            "organizer. This is a three-pair consistency result, not a "
            "universal scaling proof."
        ),
    }

    non_reversal_verdict = (
        "A2_insufficient_for_non_reversal_comparisons"
        if tvd_non_reversal_insufficiency_resolved
        else "non_reversal_A2_insufficiency_not_resolved_at_current_precision"
    )

    diagnostic = {
        "backend_for_permutations": "scipy_expm",
        "matched_total_duration_ns": total_ns,
        "matched_pulse_area": reference_area,
        "max_weighted_average_error": max_avg_error,
        "A2_closed_form_max_error": a2_closed_form_error,
        "segmentation_artifact_expm": segmentation_exact,
        "segmentation_artifact_pulser": segmentation_pulser,
        "pulser_segmentation_branches": {
            "single": branch_single,
            "three": branch_three,
        },
        "equal_A2_differences": equal_a2_differences,
        "D_decision_threshold": decision_threshold_D,
        "TVD_decision_threshold": decision_threshold_TVD,
        "primary_TVD_non_reversal_verdict": non_reversal_verdict,
        "secondary_D_non_reversal_verdict": (
            "A2_insufficient_for_non_reversal_comparisons"
            if d_non_reversal_insufficiency_resolved
            else "non_reversal_A2_insufficiency_not_resolved_at_current_precision"
        ),
        "first_segment_group_summary": first_segment_summary.to_dict(
            orient="records"
        ),
        "nonmid_to_mid_mean_TVD_ratio": first_segment_suppression_ratio,
        "reversal_pair_metrics": reversal_df.to_dict(orient="records"),
        "reversal_pair_A2_scaling": reversal_pair_scaling,
        "interpretation_boundary": (
            "The A2 insufficiency verdict applies only to non-reversal path "
            "comparisons such as permutation 2 versus 3. Exact reversal pairs "
            "remain well organized by |A2| in this diagnostic. TVD is the "
            "primary population-level discriminator; D is secondary. Rough "
            "1/TVD^2 values are scale indicators, not powered sample sizes."
        ),
    }

    df.to_csv(OUTDIR / "three_segment_magnus_diagnostic.csv", index=False)
    first_segment_summary.to_csv(
        OUTDIR / "three_segment_first_segment_groups.csv", index=False
    )
    reversal_df.to_csv(
        OUTDIR / "three_segment_reversal_pairs.csv", index=False
    )
    with open(OUTDIR / "three_segment_magnus_verdict.json", "w") as f:
        json.dump(json_sanitize(diagnostic), f, indent=2, allow_nan=False)
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
    print("Segmentation artifact D (expm/Pulser):", 
          f"{segmentation_exact['pure_trace_distance']:.3e}/"
          f"{segmentation_pulser['pure_trace_distance']:.3e}")
    print("Equal-A2 differences:", equal_a2_differences)
    print("\nFIRST-SEGMENT GROUP SUMMARY")
    print(first_segment_summary.to_string(index=False))
    print(
        "Non-mid / mid mean TVD ratio:",
        f"{first_segment_suppression_ratio:.6f}",
    )
    print("\nREVERSAL-PAIR METRICS")
    print(
        reversal_df[
            [
                "permutation_left",
                "permutation_right",
                "A2_abs_each_schedule",
                "pure_trace_distance",
                "D_over_A2_abs",
                "TVD_distribution",
                "TVD_over_A2_abs",
                "rough_shot_scale_1_over_TVD2",
            ]
        ].to_string(index=False)
    )
    print("REVERSAL-PAIR A2 SCALING:", reversal_pair_scaling)
    print("PRIMARY TVD NON-REVERSAL VERDICT:", non_reversal_verdict)
    print(
        "SECONDARY D NON-REVERSAL VERDICT:",
        diagnostic["secondary_D_non_reversal_verdict"],
    )
    return df, diagnostic


# ----------------------------------------------------------------------
# Fixed-multiset, variable-time-partition exact-reversal audit
# ----------------------------------------------------------------------
def run_variable_partition_reversal_audit(
    n=8,
    omega=OMEGA,
    avg_det=AVG_DETUNING,
    detuning_half_gap=0.12,
    nominal_total_ns=None,
    numerical_floor=None,
):
    """
    Test the missing A2 direction using exact reversal pairs.

    Controls held fixed across the audit:
      - detuning multiset {c-h, c, c+h},
      - total duration,
      - integrated Rabi area,
      - weighted-average detuning c,
      - geometry, N, and Omega.

    The low and high detunings each receive duration a, while the middle
    detuning receives T-2a. Varying a therefore changes only the time partition.
    At every partition we evaluate three independent exact-reversal topologies:

      P1: low, mid, high  <-> high, mid, low
      P2: low, high, mid <-> mid, high, low
      P3: mid, low, high <-> high, low, mid

    P2 and P3 have exactly the same |A2| = 2*h*a^2 for every partition. Their
    witness difference is therefore a direct sufficiency test for scalar A2
    within the exact-reversal class.
    """
    if numerical_floor is None:
        numerical_floor = {"max_D": 0.0, "max_TVD": 0.0}
    if nominal_total_ns is None:
        nominal_total_ns = duration_from_loop_ns(n)

    # Use the nearest clock-aligned total divisible into three equal segments,
    # so endpoint_fraction=1/3 reproduces the equal-duration v8 diagnostic.
    equal_duration_ns = int(
        round((nominal_total_ns / 3.0) / CLOCK_NS) * CLOCK_NS
    )
    total_ns = 3 * equal_duration_ns
    pulse_area = omega * total_ns / 1000.0

    endpoint_fraction_nominals = [
        0.05,
        0.08,
        0.12,
        0.16,
        0.20,
        0.25,
        1.0 / 3.0,
        0.38,
        0.42,
        0.46,
        0.48,
    ]
    endpoint_durations = []
    for fraction in endpoint_fraction_nominals:
        duration = int(round((fraction * total_ns) / CLOCK_NS) * CLOCK_NS)
        duration = max(
            MIN_DURATION_NS,
            min((total_ns - MIN_DURATION_NS) // 2, duration),
        )
        duration = int(round(duration / CLOCK_NS) * CLOCK_NS)
        if duration not in endpoint_durations:
            endpoint_durations.append(duration)

    detuning_by_role = {
        "low": avg_det - detuning_half_gap,
        "mid": avg_det,
        "high": avg_det + detuning_half_gap,
    }
    topologies = [
        ("P1_sweep", ("low", "mid", "high")),
        ("P2_edge_first", ("low", "high", "mid")),
        ("P3_mid_first", ("mid", "low", "high")),
    ]

    rows = []
    for partition_index, endpoint_duration_ns in enumerate(
        endpoint_durations, start=1
    ):
        point_start = time.perf_counter()
        middle_duration_ns = total_ns - 2 * endpoint_duration_ns
        if middle_duration_ns < MIN_DURATION_NS:
            raise AssertionError("Variable-partition middle segment is too short.")
        duration_by_role = {
            "low": endpoint_duration_ns,
            "mid": middle_duration_ns,
            "high": endpoint_duration_ns,
        }
        endpoint_fraction_actual = endpoint_duration_ns / total_ns

        for topology_name, role_order in topologies:
            reverse_role_order = tuple(reversed(role_order))
            forward_segments = [
                (detuning_by_role[role], duration_by_role[role])
                for role in role_order
            ]
            reverse_segments = list(reversed(forward_segments))
            independently_built_reverse = [
                (detuning_by_role[role], duration_by_role[role])
                for role in reverse_role_order
            ]
            if reverse_segments != independently_built_reverse:
                raise AssertionError("Exact reversal construction failed.")

            forward_total = sum(duration for _, duration in forward_segments)
            reverse_total = sum(duration for _, duration in reverse_segments)
            forward_average = sum(
                detuning * duration for detuning, duration in forward_segments
            ) / forward_total
            reverse_average = sum(
                detuning * duration for detuning, duration in reverse_segments
            ) / reverse_total
            forward_area = integrated_rabi_area(forward_segments, omega)
            reverse_area = integrated_rabi_area(reverse_segments, omega)
            if forward_total != total_ns or reverse_total != total_ns:
                raise AssertionError("Variable-partition total duration mismatch.")
            if abs(forward_average - avg_det) > SCHEDULE_MATCH_TOL:
                raise AssertionError("Variable-partition average detuning mismatch.")
            if abs(reverse_average - avg_det) > SCHEDULE_MATCH_TOL:
                raise AssertionError("Reversal average detuning mismatch.")
            if abs(forward_area - reverse_area) > SCHEDULE_MATCH_TOL:
                raise AssertionError("Variable-partition pulse area mismatch.")

            a2_forward = signed_second_order_detuning_area(forward_segments)
            a2_reverse = signed_second_order_detuning_area(reverse_segments)
            if abs(a2_forward + a2_reverse) > 1.0e-12:
                raise AssertionError("A2 did not change sign under exact reversal.")

            a_us = endpoint_duration_ns / 1000.0
            total_us = total_ns / 1000.0
            if topology_name == "P1_sweep":
                expected_a2_abs = (
                    2.0
                    * detuning_half_gap
                    * a_us
                    * (total_us - a_us)
                )
            else:
                expected_a2_abs = (
                    2.0 * detuning_half_gap * a_us**2
                )
            if abs(abs(a2_forward) - expected_a2_abs) > 1.0e-12:
                raise AssertionError("Variable-partition A2 closed form failed.")

            psi_forward = exact_state_from_segments(n, forward_segments, omega)
            psi_reverse = exact_state_from_segments(n, reverse_segments, omega)
            metrics = pair_metrics(
                psi_forward,
                psi_reverse,
                f"{topology_name}_forward_vs_exact_reverse",
            )
            row = {
                "partition_index": partition_index,
                "endpoint_fraction_actual": endpoint_fraction_actual,
                "endpoint_duration_ns": endpoint_duration_ns,
                "middle_duration_ns": middle_duration_ns,
                "total_duration_ns": total_ns,
                "pulse_area": pulse_area,
                "weighted_avg_detuning": forward_average,
                "topology": topology_name,
                "forward_role_order": "|".join(role_order),
                "reverse_role_order": "|".join(reverse_role_order),
                "A2_signed_forward": a2_forward,
                "A2_signed_reverse": a2_reverse,
                "A2_abs_each_schedule": abs(a2_forward),
                "D_over_A2_abs": (
                    metrics["pure_trace_distance"] / abs(a2_forward)
                    if abs(a2_forward) > 0
                    else None
                ),
                "TVD_over_A2_abs": (
                    metrics["TVD_distribution"] / abs(a2_forward)
                    if abs(a2_forward) > 0
                    else None
                ),
                "rough_shot_scale_1_over_TVD2": (
                    1.0 / metrics["TVD_distribution"] ** 2
                    if metrics["TVD_distribution"] > 0
                    else None
                ),
            }
            row.update(metrics)
            row["elapsed_seconds"] = time.perf_counter() - point_start
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(
        OUTDIR / "variable_partition_reversal_pairs.csv", index=False
    )

    # P2 and P3 are matched-A2 exact reversal pairs at every partition.
    matched_rows = []
    for partition_index, group in df.groupby("partition_index", sort=True):
        p2 = group[group["topology"] == "P2_edge_first"].iloc[0]
        p3 = group[group["topology"] == "P3_mid_first"].iloc[0]
        a2_error = abs(
            float(p2["A2_abs_each_schedule"])
            - float(p3["A2_abs_each_schedule"])
        )
        if a2_error > 1.0e-12:
            raise AssertionError("P2/P3 matched-A2 identity failed.")
        mean_D = 0.5 * (
            float(p2["pure_trace_distance"])
            + float(p3["pure_trace_distance"])
        )
        mean_TVD = 0.5 * (
            float(p2["TVD_distribution"])
            + float(p3["TVD_distribution"])
        )
        matched_rows.append(
            {
                "partition_index": int(partition_index),
                "endpoint_fraction_actual": float(
                    p2["endpoint_fraction_actual"]
                ),
                "endpoint_duration_ns": int(p2["endpoint_duration_ns"]),
                "middle_duration_ns": int(p2["middle_duration_ns"]),
                "A2_abs_matched": float(p2["A2_abs_each_schedule"]),
                "A2_abs_error": a2_error,
                "P2_D": float(p2["pure_trace_distance"]),
                "P3_D": float(p3["pure_trace_distance"]),
                "D_abs_split": abs(
                    float(p2["pure_trace_distance"])
                    - float(p3["pure_trace_distance"])
                ),
                "D_relative_split": (
                    abs(
                        float(p2["pure_trace_distance"])
                        - float(p3["pure_trace_distance"])
                    )
                    / mean_D
                    if mean_D > 0
                    else 0.0
                ),
                "P2_TVD": float(p2["TVD_distribution"]),
                "P3_TVD": float(p3["TVD_distribution"]),
                "TVD_abs_split": abs(
                    float(p2["TVD_distribution"])
                    - float(p3["TVD_distribution"])
                ),
                "TVD_relative_split": (
                    abs(
                        float(p2["TVD_distribution"])
                        - float(p3["TVD_distribution"])
                    )
                    / mean_TVD
                    if mean_TVD > 0
                    else 0.0
                ),
            }
        )
    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_csv(
        OUTDIR / "variable_partition_matched_A2_splits.csv", index=False
    )

    # Quantify organizer quality globally and within each topology.
    fit_rows = []
    for subset_name, subset in [
        ("all_reversal_pairs", df),
        *[
            (topology_name, df[df["topology"] == topology_name])
            for topology_name, _ in topologies
        ],
    ]:
        for witness in ["pure_trace_distance", "TVD_distribution"]:
            summary = fit_summary_for_witness(
                subset,
                "A2_abs_each_schedule",
                witness,
                subset_name,
            )
            fit_rows.append(summary)
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(
        OUTDIR / "variable_partition_A2_fit_summary.csv", index=False
    )

    D_threshold = max(
        MIN_SIGNAL_TO_FLOOR * float(numerical_floor.get("max_D", 0.0)),
        1.0e-10,
    )
    TVD_threshold = max(
        MIN_SIGNAL_TO_FLOOR * float(numerical_floor.get("max_TVD", 0.0)),
        1.0e-10,
    )
    max_D_split = float(matched_df["D_abs_split"].max())
    max_TVD_split = float(matched_df["TVD_abs_split"].max())
    TVD_sufficiency_rejected = max_TVD_split > TVD_threshold
    D_sufficiency_rejected = max_D_split > D_threshold
    primary_verdict = (
        "A2_strong_but_not_sufficient_for_exact_reversal_pairs"
        if TVD_sufficiency_rejected
        else "A2_sufficiency_not_rejected_for_exact_reversal_pairs"
    )

    all_tvd_fit = fit_df[
        (fit_df["subset"] == "all_reversal_pairs")
        & (fit_df["y"] == "TVD_distribution")
    ].iloc[0]
    all_D_fit = fit_df[
        (fit_df["subset"] == "all_reversal_pairs")
        & (fit_df["y"] == "pure_trace_distance")
    ].iloc[0]
    diagnostic = {
        "experiment": (
            "fixed detuning multiset, fixed average, variable time partition, "
            "exact reversal pairs"
        ),
        "backend": "scipy_expm",
        "N": n,
        "total_duration_ns": total_ns,
        "pulse_area": pulse_area,
        "avg_detuning": avg_det,
        "detuning_multiset": list(detuning_by_role.values()),
        "partitions": int(len(endpoint_durations)),
        "reversal_pairs": int(len(df)),
        "matched_A2_P2_P3_pairs": int(len(matched_df)),
        "D_floor_threshold": D_threshold,
        "TVD_floor_threshold": TVD_threshold,
        "max_matched_A2_D_split": max_D_split,
        "max_matched_A2_TVD_split": max_TVD_split,
        "max_matched_A2_D_relative_split": float(
            matched_df["D_relative_split"].max()
        ),
        "max_matched_A2_TVD_relative_split": float(
            matched_df["TVD_relative_split"].max()
        ),
        "all_pair_D_A2_R2": float(all_D_fit["ols_R2"]),
        "all_pair_TVD_A2_R2": float(all_tvd_fit["ols_R2"]),
        "primary_TVD_verdict": primary_verdict,
        "secondary_D_verdict": (
            "A2_strong_but_not_sufficient_for_exact_reversal_pairs"
            if D_sufficiency_rejected
            else "A2_sufficiency_not_rejected_for_exact_reversal_pairs"
        ),
        "interpretation_boundary": (
            "A high A2 correlation can coexist with resolved matched-A2 splits. "
            "Organizer quality and scalar sufficiency are distinct claims. "
            "This audit fixes the detuning multiset, total time, pulse area, "
            "weighted-average detuning, geometry, N, and Omega."
        ),
    }
    with open(
        OUTDIR / "variable_partition_reversal_verdict.json", "w"
    ) as f:
        json.dump(json_sanitize(diagnostic), f, indent=2, allow_nan=False)

    print("\n" + "=" * 80)
    print("D) FIXED-MULTISET VARIABLE-PARTITION REVERSAL AUDIT")
    print("=" * 80)
    print(
        df[
            [
                "partition_index",
                "endpoint_fraction_actual",
                "topology",
                "A2_abs_each_schedule",
                "pure_trace_distance",
                "TVD_distribution",
                "D_over_A2_abs",
                "TVD_over_A2_abs",
            ]
        ].to_string(index=False)
    )
    print("\nMATCHED-A2 P2/P3 SPLITS")
    print(matched_df.to_string(index=False))
    print("\nA2 FIT SUMMARY")
    print(
        fit_df[
            [
                "subset",
                "y",
                "n",
                "spearman_rho",
                "ols_slope",
                "ols_intercept",
                "ols_R2",
                "rmse",
            ]
        ].to_string(index=False)
    )
    print("PRIMARY TVD REVERSAL VERDICT:", primary_verdict)
    print("SECONDARY D REVERSAL VERDICT:", diagnostic["secondary_D_verdict"])
    return df, matched_df, fit_df, diagnostic


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
    x = sub_df[xcol].values.astype(float)
    y = sub_df[ycol].values.astype(float)
    rho = spearman_corr_tiesafe(x, y)
    slope, intercept, r2 = ols_fit_with_intercept(x, y)
    prediction = slope * x + intercept
    return {
        "subset": subset_name,
        "x": xcol,
        "y": ycol,
        "n": int(len(sub_df)),
        "spearman_rho": rho,
        "ols_slope": slope,
        "ols_intercept": intercept,
        "ols_R2": r2,
        "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "mae": float(np.mean(np.abs(y - prediction))),
    }


def run_117_scan():
    print("\n" + "="*80 + "\nB) 117-POINT SIGNED-RESPONSE SCAN\n" + "="*80)
    n = 8
    total_ns = duration_from_loop_ns(n)
    comm_norm = fixed_commutator_norm(n)
    print(f"N={n}, total_ns={total_ns}, fixed_comm_norm={comm_norm:.6f}")
    print("Grid: 9 gaps in [0, 0.24], 13 fractions in [0.10, 0.90]")

    rows = []
    idx = 0
    for gap in np.linspace(0.0, 0.24, 9):
        for frac_nom in np.linspace(0.10, 0.90, 13):
            point_start = time.perf_counter()
            idx += 1
            d1, d2, T = split_duration_clock(total_ns, frac_nom)
            forward_segments, reverse_segments, f_actual = (
                make_two_segment_schedules(T, d1, gap, AVG_DETUNING)
            )
            delta1 = forward_segments[0][0]
            delta2 = forward_segments[1][0]
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

            psi_f = exact_state_from_segments(n, forward_segments)
            psi_r = exact_state_from_segments(n, reverse_segments)
            m = pair_metrics(psi_f, psi_r, "forward_vs_reverse")
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
                "C_BCH_shape": abs(gap) * f_actual * (1.0 - f_actual),
                "C_linear_support": abs(gap) * min(f_actual, 1.0 - f_actual),
                "backend": "scipy_expm",
            }
            row.update(m)
            row["rough_shot_scale_1_over_TVD2"] = (
                float(1.0 / m["TVD_distribution"] ** 2)
                if m["TVD_distribution"] > 0
                else None
            )
            row["elapsed_seconds"] = time.perf_counter() - point_start
            rows.append(row)
            print(f"[{idx:03d}/117] gap={gap:.4f} f={f_actual:.4f} "
                  f"D={m['pure_trace_distance']:.6f} "
                  f"Γ={m['phase_gap_BC_minus_overlap']:.6f} "
                  f"TVD={m['TVD_distribution']:.6f}")
    df = pd.DataFrame(rows)

    zero = df[df["gap"] == 0.0]
    numerical_floor = {
        "max_abs_infidelity": float(
            np.max(np.abs(1.0 - zero["fidelity"].to_numpy(dtype=float)))
        ),
        "max_D": float(np.max(np.abs(zero["pure_trace_distance"]))),
        "max_TVD": float(np.max(np.abs(zero["TVD_distribution"]))),
        "max_phase_gap": float(
            np.max(np.abs(zero["phase_gap_BC_minus_overlap"]))
        ),
    }
    if numerical_floor["max_abs_infidelity"] > ZERO_GAP_INFIDELITY_TOL:
        raise AssertionError(f"Zero-gap infidelity gate failed: {numerical_floor}")
    if numerical_floor["max_TVD"] > ZERO_GAP_TVD_TOL:
        raise AssertionError(f"Zero-gap TVD gate failed: {numerical_floor}")

    nonzero = df[df["gap"] > 0.0].copy()
    nonzero["D_over_gap"] = (
        nonzero["pure_trace_distance"] / nonzero["gap"]
    )
    gap_linearity_rows = []
    for frac_actual, group in nonzero.groupby("frac_actual", sort=True):
        ordered = group.sort_values("gap")
        ratios = ordered["D_over_gap"].to_numpy(dtype=float)
        gap_linearity_rows.append(
            {
                "frac_actual": float(frac_actual),
                "n_gaps": int(len(ordered)),
                "mean_D_over_gap": float(np.mean(ratios)),
                "min_D_over_gap": float(np.min(ratios)),
                "max_D_over_gap": float(np.max(ratios)),
                "relative_span": float(
                    (np.max(ratios) - np.min(ratios)) / np.mean(ratios)
                ),
                "endpoint_relative_change": float(
                    (ratios[-1] - ratios[0]) / ratios[0]
                ),
            }
        )
    gap_linearity_df = pd.DataFrame(gap_linearity_rows)
    gap_linearity_df.to_csv(
        OUTDIR / "D_over_gap_linearity_by_fraction.csv", index=False
    )
    min_signal_D = float(nonzero["pure_trace_distance"].min())
    effective_D_floor = max(
        numerical_floor["max_D"],
        math.sqrt(numerical_floor["max_abs_infidelity"]),
        1.0e-15,
    )
    signal_to_floor = min_signal_D / effective_D_floor
    if signal_to_floor < MIN_SIGNAL_TO_FLOOR:
        raise AssertionError(
            f"Weakest signal is too close to numerical floor: "
            f"ratio={signal_to_floor:.3e}, floor={numerical_floor}"
        )

    df.to_csv(OUTDIR / "scan117_statevector_metrics.csv", index=False)

    proxies = ["C_BCH_shape", "C_linear_support"]
    witnesses = [
        "pure_trace_distance",
        "phase_gap_BC_minus_overlap",
        "TVD_distribution",
    ]
    summaries = []
    residual_rows = []
    subset_map = {
        "nonzero_gap_primary": nonzero,
        "all_117_sensitivity_only": df,
    }
    for subset_name, sub_df in subset_map.items():
        for proxy in proxies:
            for witness in witnesses:
                summary = fit_summary_for_witness(
                    sub_df, proxy, witness, subset_name
                )
                summaries.append(summary)
                prediction = (
                    summary["ols_slope"] * sub_df[proxy].to_numpy(dtype=float)
                    + summary["ols_intercept"]
                )
                for (_, source_row), pred in zip(sub_df.iterrows(), prediction):
                    residual_rows.append(
                        {
                            "subset": subset_name,
                            "proxy": proxy,
                            "witness": witness,
                            "idx": int(source_row["idx"]),
                            "gap": float(source_row["gap"]),
                            "frac_actual": float(source_row["frac_actual"]),
                            "observed": float(source_row[witness]),
                            "predicted": float(pred),
                            "residual": float(source_row[witness] - pred),
                        }
                    )
    summary_df = pd.DataFrame(summaries)
    residual_df = pd.DataFrame(residual_rows)
    summary_df.to_csv(OUTDIR / "scan117_correlation_summary.csv", index=False)
    residual_df.to_csv(OUTDIR / "scan117_fit_residuals.csv", index=False)

    lookup = {
        (round(float(row.gap), 12), int(row.duration1_ns)): row
        for row in nonzero.itertuples(index=False)
    }

    # Positive-gap f-reflection decomposition.
    even_odd_rows = []
    for row in nonzero.itertuples(index=False):
        if row.duration1_ns > row.duration2_ns:
            continue
        reflected = lookup[(round(float(row.gap), 12), int(row.duration2_ns))]
        out = {
            "gap": float(row.gap),
            "f_left": float(row.frac_actual),
            "f_right": float(reflected.frac_actual),
        }
        for witness in witnesses:
            left = float(getattr(row, witness))
            right = float(getattr(reflected, witness))
            even = 0.5 * (left + right)
            odd = 0.5 * (left - right)
            out[f"{witness}_even"] = even
            out[f"{witness}_odd"] = odd
            out[f"{witness}_odd_fraction"] = (
                abs(odd) / abs(even) if abs(even) > 0 else 0.0
            )
        even_odd_rows.append(out)
    even_odd_df = pd.DataFrame(even_odd_rows)
    even_odd_df.to_csv(OUTDIR / "f_reflection_even_odd.csv", index=False)

    # Exact signed-gap schedule identity and numerical witness gate.
    symmetry_rows = []
    max_schedule_parameter_error = 0.0
    for row in nonzero.itertuples(index=False):
        f_pos_reflected, r_pos_reflected, _ = make_two_segment_schedules(
            total_ns, int(row.duration2_ns), float(row.gap), AVG_DETUNING
        )
        f_negative, r_negative, _ = make_two_segment_schedules(
            total_ns, int(row.duration1_ns), -float(row.gap), AVG_DETUNING
        )

        def schedule_parameter_error(schedule_a, schedule_b):
            error = 0.0
            for (delta_a, duration_a), (delta_b, duration_b) in zip(
                schedule_a, schedule_b
            ):
                error = max(
                    error,
                    abs(delta_a - delta_b),
                    float(abs(duration_a - duration_b)),
                )
            return error

        schedule_error = max(
            schedule_parameter_error(f_pos_reflected, r_negative),
            schedule_parameter_error(r_pos_reflected, f_negative),
        )
        max_schedule_parameter_error = max(
            max_schedule_parameter_error, schedule_error
        )
        psi_fn = exact_state_from_segments(n, f_negative)
        psi_rn = exact_state_from_segments(n, r_negative)
        negative_metrics = pair_metrics(
            psi_fn, psi_rn, "forward_vs_reverse_negative_gap"
        )
        reflected_row = lookup[
            (round(float(row.gap), 12), int(row.duration2_ns))
        ]
        symmetry_rows.append(
            {
                "gap_positive": float(row.gap),
                "f_original": float(row.frac_actual),
                "f_reflected": float(reflected_row.frac_actual),
                "D_negative_gap": negative_metrics["pure_trace_distance"],
                "D_reflected_positive_gap": float(
                    reflected_row.pure_trace_distance
                ),
                "D_abs_error": abs(
                    negative_metrics["pure_trace_distance"]
                    - float(reflected_row.pure_trace_distance)
                ),
                "TVD_negative_gap": negative_metrics["TVD_distribution"],
                "TVD_reflected_positive_gap": float(
                    reflected_row.TVD_distribution
                ),
                "TVD_abs_error": abs(
                    negative_metrics["TVD_distribution"]
                    - float(reflected_row.TVD_distribution)
                ),
                "schedule_parameter_error": schedule_error,
            }
        )
    symmetry_df = pd.DataFrame(symmetry_rows)
    symmetry_df.to_csv(OUTDIR / "signed_gap_symmetry_audit.csv", index=False)
    symmetry_max_D_error = float(symmetry_df["D_abs_error"].max())
    symmetry_max_TVD_error = float(symmetry_df["TVD_abs_error"].max())
    symmetry_threshold = max(
        SYMMETRY_ABS_TOL, MIN_SIGNAL_TO_FLOOR * effective_D_floor
    )
    symmetry_tvd_threshold = max(
        SYMMETRY_ABS_TOL,
        MIN_SIGNAL_TO_FLOOR * numerical_floor["max_TVD"],
    )
    if max_schedule_parameter_error > SCHEDULE_MATCH_TOL:
        raise AssertionError(
            f"Signed-gap schedule identity failed: {max_schedule_parameter_error}"
        )
    if symmetry_max_D_error > symmetry_threshold:
        raise AssertionError(
            f"Signed-gap D identity failed: {symmetry_max_D_error} > "
            f"{symmetry_threshold}"
        )
    if symmetry_max_TVD_error > symmetry_tvd_threshold:
        raise AssertionError(
            f"Signed-gap TVD identity failed: {symmetry_max_TVD_error} > "
            f"{symmetry_tvd_threshold}"
        )

    primary = summary_df[
        summary_df["subset"] == "nonzero_gap_primary"
    ].copy()
    d_models = primary[primary["y"] == "pure_trace_distance"].set_index("x")
    proxy_verdict = {
        "D_R2_BCH": float(d_models.loc["C_BCH_shape", "ols_R2"]),
        "D_R2_linear_support": float(
            d_models.loc["C_linear_support", "ols_R2"]
        ),
        "D_RMSE_BCH": float(d_models.loc["C_BCH_shape", "rmse"]),
        "D_RMSE_linear_support": float(
            d_models.loc["C_linear_support", "rmse"]
        ),
        "preferred_on_this_fixed_T_grid": (
            "C_linear_support"
            if d_models.loc["C_linear_support", "rmse"]
            < d_models.loc["C_BCH_shape", "rmse"]
            else "C_BCH_shape"
        ),
        "boundary": (
            "Fixed T cannot determine T versus T^2 scaling; this comparison "
            "selects schedule-shape proxies only."
        ),
    }

    cert = {
        "experiment": "117-point two-proxy signed-response scan",
        "primary_backend": "scipy_expm",
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
        "fixed_commutator_norm_metadata": comm_norm,
        "zero_gap_numerical_floor": numerical_floor,
        "weakest_nonzero_D": min_signal_D,
        "weakest_D_to_floor_ratio": signal_to_floor,
        "rough_shot_scale_from_N8_TVD": {
            "minimum_best_case": float(
                nonzero["rough_shot_scale_1_over_TVD2"].min()
            ),
            "maximum_weakest_case": float(
                nonzero["rough_shot_scale_1_over_TVD2"].max()
            ),
            "boundary": (
                "1/TVD^2 is a rough scale only, not a powered hypothesis-test "
                "sample-size calculation."
            ),
        },
        "D_over_gap_linearity_by_fraction": gap_linearity_df.to_dict(
            orient="records"
        ),
        "proxy_comparison": proxy_verdict,
        "signed_gap_symmetry": {
            "identity": "D(1-f,+g) = D(f,-g)",
            "max_schedule_parameter_error": max_schedule_parameter_error,
            "max_D_error": symmetry_max_D_error,
            "max_TVD_error": symmetry_max_TVD_error,
            "D_threshold": symmetry_threshold,
            "TVD_threshold": symmetry_tvd_threshold,
        },
        "max_positive_gap_D_odd_fraction": float(
            even_odd_df["pure_trace_distance_odd_fraction"].max()
        ),
        "fits": summary_df.to_dict(orient="records"),
        "interpretation_boundary": (
            "Nonzero-gap points are primary. The 13 repeated zero-gap controls "
            "are numerical-floor controls and sensitivity fits only. No iid "
            "significance p-values or strict monotonicity claims are made."
        ),
    }
    with open(OUTDIR / "scan117_certificate.json", "w") as f:
        json.dump(json_sanitize(cert), f, indent=2, allow_nan=False)
    print("\nTWO-PROXY FIT SUMMARY")
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
                "rmse",
            ]
        ].to_string(index=False)
    )
    print("Zero-gap numerical floor:", numerical_floor)
    print("Weakest D / effective floor:", f"{signal_to_floor:.3e}")
    print("Proxy verdict:", proxy_verdict)
    print("\nD/g LINEARITY BY FRACTION")
    print(gap_linearity_df.to_string(index=False))
    print(
        "Signed-gap symmetry max D/TVD errors:",
        f"{symmetry_max_D_error:.3e}/{symmetry_max_TVD_error:.3e}",
    )
    print("\nSaved outputs under", OUTDIR)
    return df, summary_df, residual_df, even_odd_df, symmetry_df, cert


# ----------------------------------------------------------------------
# Cross-validation with expm
# ----------------------------------------------------------------------
def run_cross_validation():
    print("\n" + "="*80 + "\nC) MULTI-POINT CROSS-VALIDATION: Pulser vs expm\n" + "="*80)
    test_specs = [
        (2, 0.35, 0.13),
        (5, 0.50, 0.18),
        (8, 0.10, 0.03),
        (8, 0.20, 0.12),
        (8, 0.50, 0.12),
        (8, 0.80, 0.24),
    ]
    rows = []
    for n, fraction, gap in test_specs:
        total_ns = duration_from_loop_ns(n)
        d1, d2, _ = split_duration_clock(total_ns, fraction)
        forward, reverse, f_actual = make_two_segment_schedules(
            total_ns, d1, gap, AVG_DETUNING
        )
        for order_name, segments in [("forward", forward), ("reverse", reverse)]:
            psi_p, branch = final_statevector_from_segments(n, segments)
            psi_e = exact_state_from_segments(n, segments)
            fidelity = state_fidelity(psi_p, psi_e)
            infidelity = abs(1.0 - fidelity)
            rows.append(
                {
                    "N": n,
                    "frac_actual": f_actual,
                    "gap": gap,
                    "order": order_name,
                    "segments": repr(segments),
                    "fidelity": fidelity,
                    "infidelity": infidelity,
                    "pulser_branch": branch,
                }
            )
            print(
                f"N={n} f={f_actual:.6f} gap={gap:.3f} {order_name} "
                f"fidelity={fidelity:.12f} infidelity={infidelity:.3e}"
            )

    cv_df = pd.DataFrame(rows)
    max_infidelity = float(cv_df["infidelity"].max())
    if max_infidelity > CROSS_VALIDATION_MAX_INFIDELITY:
        raise AssertionError(
            f"Pulser/expm cross-validation failed: max infidelity "
            f"{max_infidelity:.3e} > {CROSS_VALIDATION_MAX_INFIDELITY:.3e}"
        )

    cv_df.to_csv(OUTDIR / "cross_validation_points.csv", index=False)
    cv_cert = {
        "points": cv_df.to_dict(orient="records"),
        "max_infidelity": max_infidelity,
        "max_allowed_infidelity": CROSS_VALIDATION_MAX_INFIDELITY,
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
        f"CROSS-VALIDATION PASS: max infidelity={max_infidelity:.3e}"
    )
    return cv_df, cv_cert


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------
def write_provenance():
    if "__file__" in globals():
        source_path = Path(__file__).resolve()
        source_bytes = source_path.read_bytes()
        source_label = str(source_path)
        source_kind = "python_file"
    else:
        # Jupyter/Colab copy-paste execution has no __file__. IPython keeps the
        # submitted cell text in its input history, which is the correct source
        # artifact to hash in that execution mode.
        source_path = None
        source_bytes = None
        source_label = "<interactive_notebook_cell>"
        source_kind = "interactive_notebook_cell"
        try:
            ipython = get_ipython()  # noqa: F821 - defined by IPython
            input_history = ipython.user_ns.get("In", [])
            if input_history:
                source_bytes = input_history[-1].encode("utf-8")
        except (NameError, AttributeError, IndexError, UnicodeEncodeError):
            source_bytes = None

    source_sha256 = (
        hashlib.sha256(source_bytes).hexdigest()
        if source_bytes is not None
        else None
    )
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (OUTDIR / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    provenance = {
        "source_label": source_label,
        "source_kind": source_kind,
        "source_sha256": source_sha256,
        "source_hash_available": source_sha256 is not None,
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": sys.version,
        "versions": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scipy": package_version("scipy"),
            "pulser": package_version("pulser"),
            "pulser_simulation": package_version("pulser-simulation"),
            "qutip": package_version("qutip"),
        },
    }
    with open(OUTDIR / "provenance.json", "w") as f:
        json.dump(json_sanitize(provenance), f, indent=2, allow_nan=False)
    return provenance


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("\n" + "="*80 + "\nFIXED-H SIGNED-RESPONSE VALIDATION v9\n" + "="*80)
    print("OUTDIR:", OUTDIR)
    print("This script includes:\n"
          " - N=2 fixed-H certificate (clarified)\n"
          " - expm-primary 117-point two-proxy scan\n"
          " - signed-gap identity and f-reflection even/odd split\n"
          " - equal-A2 three-segment insufficiency diagnostic\n"
          " - fixed-multiset variable-partition exact-reversal audit\n"
          " - 12-case forward/reverse Pulser cross-validation\n"
          " - fail-fast numerical-floor and schedule gates\n"
          " - SHA256, pip-freeze, versions, and per-point timing\n")
    t0 = time.time()

    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is required; install the declared dependencies.")

    provenance = write_provenance()
    cv_df, cv_cert = run_cross_validation()
    _, _, _, n2_cert, _ = run_n2_fixedH()
    (
        df,
        summary_df,
        residual_df,
        even_odd_df,
        symmetry_df,
        cert,
    ) = run_117_scan()
    df3, three_segment = run_3segment_magnus_diagnostic(
        numerical_floor_D=cert["zero_gap_numerical_floor"]["max_D"]
    )
    (
        variable_partition_df,
        variable_partition_matched_df,
        variable_partition_fit_df,
        variable_partition,
    ) = run_variable_partition_reversal_audit(
        numerical_floor=cert["zero_gap_numerical_floor"]
    )

    run_summary = {
        "status": "PASS",
        "source_sha256": provenance["source_sha256"],
        "source_kind": provenance["source_kind"],
        "primary_backend": "scipy_expm",
        "cross_validation_max_infidelity": cv_cert["max_infidelity"],
        "three_segment_rows": int(len(df3)),
        "three_segment_primary_TVD_non_reversal_verdict": three_segment[
            "primary_TVD_non_reversal_verdict"
        ],
        "three_segment_secondary_D_non_reversal_verdict": three_segment[
            "secondary_D_non_reversal_verdict"
        ],
        "three_segment_reversal_pair_A2_scaling": three_segment[
            "reversal_pair_A2_scaling"
        ],
        "three_segment_nonmid_to_mid_mean_TVD_ratio": three_segment[
            "nonmid_to_mid_mean_TVD_ratio"
        ],
        "variable_partition_reversal_rows": int(len(variable_partition_df)),
        "variable_partition_matched_A2_rows": int(
            len(variable_partition_matched_df)
        ),
        "variable_partition_primary_TVD_verdict": variable_partition[
            "primary_TVD_verdict"
        ],
        "variable_partition_secondary_D_verdict": variable_partition[
            "secondary_D_verdict"
        ],
        "variable_partition_all_pair_D_A2_R2": variable_partition[
            "all_pair_D_A2_R2"
        ],
        "variable_partition_all_pair_TVD_A2_R2": variable_partition[
            "all_pair_TVD_A2_R2"
        ],
        "variable_partition_max_matched_A2_D_split": variable_partition[
            "max_matched_A2_D_split"
        ],
        "variable_partition_max_matched_A2_TVD_split": variable_partition[
            "max_matched_A2_TVD_split"
        ],
        "scan_points": int(len(df)),
        "primary_proxy_verdict": cert["proxy_comparison"],
        "signed_gap_symmetry": cert["signed_gap_symmetry"],
        "max_positive_gap_D_odd_fraction": cert[
            "max_positive_gap_D_odd_fraction"
        ],
        "population_visibility_scale": {
            "N2_forward_reverse_TVD": n2_cert["forward_reverse_TVD"],
            "N2_rough_1_over_TVD2": n2_cert[
                "rough_population_shot_scale_1_over_TVD2"
            ],
            "N8_best_rough_1_over_TVD2": cert[
                "rough_shot_scale_from_N8_TVD"
            ]["minimum_best_case"],
            "boundary": (
                "These are rough 1/TVD^2 scales, not powered sample-size "
                "requirements."
            ),
        },
        "output_directory": str(OUTDIR),
    }
    with open(OUTDIR / "run_summary.json", "w") as f:
        json.dump(json_sanitize(run_summary), f, indent=2, allow_nan=False)

    print("\n" + "="*80 + "\nALL NUMERICAL AND STRUCTURAL GATES PASS\n" + "="*80)
    print(f"Elapsed: {time.time()-t0:.2f} s")
    print(f"Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()
