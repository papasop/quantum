# -*- coding: utf-8 -*-
"""
paper117_path_order_validation.py  (v3)

Frozen paper-only validation for Y.Y.N. Li's neutral-atom pulse-ordering
manuscript.

Scientific scope
----------------
- Independent dense scipy evolution is the primary scan backend; local
  Pulser/Qutip statevectors are used for sampled cross-validation.
- Two-segment forward/reverse schedules have exactly matched total duration,
  integrated Rabi area, and weighted-average detuning.
- A small SPECIFIED set of physically motivated fixed-T schedule-shape
  proxies is compared rather than presupposing a path-area law:
      C_BCH = |g| f(1-f)          second-order Magnus shape
      C_lin = |g| min(f,1-f)      support of the differential schedule
  This is an exclusion test, not a search over arbitrary functions. The set is
  specified, NOT preregistered: C_lin was introduced after early data
  analysis. Only the 117-point grid is frozen. The
  reference shape C_sin2 = |g| sin^2(pi f) is computed and reported but is
  EXCLUDED from the verdict. It is an exploratory smooth symmetric reference
  shape used only to diagnose structured residuals; it is not derived here and
  carries no theoretical status. With one T the scan cannot determine T versus
  T^2 scaling.
- The proxy comparison is primarily done PER GAP. A pooled fit across gaps
  mixes the f-shape question with the residual g-dependence of D and can
  invert the ranking; the pooled fit is reported only as a sensitivity, and a
  disagreement between the two raises an explicit warning.
- Correlations describe a deterministic parameter scan. No iid significance
  p-values are reported.
- The signed-gap relation is an algebraic schedule identity, not an
  independent numerical witness.
- The one-pulse versus three-identical-pulse Pulser comparison is an
  implementation sanity check, not independent physical evidence: both
  schedules define the same sampled control waveform.

Numerical discipline (v3)
-------------------------
- The pure-state trace distance is computed from the phase-aligned projective
  residual r via D = r*sqrt(1 - r^2/4), which is algebraically identical to
  sqrt(1-F) but free of the catastrophic cancellation in 1-F. Both estimators
  are computed and their disagreement is recorded and gated.
- PAPER_TOTAL_NS must be divisible by 24 so that BOTH the equal-duration
  three-segment split AND the clock-aligned f=1/2 two-segment split exist.
  The f -> 1-f reflection closure of the duration grid is asserted explicitly
  instead of being relied upon as a rounding accident.
- Cross-validation is reported at the WITNESS level (D, TVD) in addition to
  the state level, because the paper's claims are about the witnesses.
- Perturbative-validity metadata max|A2|*||[H_X,N]|| is recorded so that the
  domain of validity of C_BCH is explicit.

Colab install:
    !pip install -q -U pulser==1.8.0 pulser-simulation==1.8.0 pandas numpy scipy
Run:
    python paper117_path_order_validation.py
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
from pulser.sampler import sample as sample_sequence
from pulser_simulation import QutipEmulator

# ---------- constants ----------
CLOCK_NS = 4
MIN_DURATION_NS = 16
MAX_DURATION_NS = 10000
OMEGA = 1.22
SPACING_UM = 8.0
AVG_DETUNING = -0.31

# Frozen common duration for every N=8 paper diagnostic.
#
# Divisibility requirement: PAPER_TOTAL_NS % 24 == 0.
#   * % 12 == 0 (= 3*CLOCK_NS) gives a clock-aligned equal three-way split.
#   * %  8 == 0 gives a clock-aligned f = 1/2 two-way split, which is what
#     makes the fraction grid closed under f -> 1-f.
# 4044 satisfies only the first and silently drops the f=1/2 reflection pair;
# 4032 and 4056 satisfy both. 4032 is used here.
PAPER_TOTAL_NS = 4032
DURATION_QUANTUM_NS = 24

# When True, every N uses a duration snapped to DURATION_QUANTUM_NS, so the
# script has ONE duration convention. When False, only N=8 is frozen and other
# N keep the raw loop-motivated duration. The snap shift is always reported by
# duration_convention_report(); set to False to reproduce the older mixed
# convention exactly.
UNIFY_DURATION_CONVENTION = True

FRACTION_GRID = np.linspace(0.10, 0.90, 13)
GAP_GRID = np.linspace(0.0, 0.24, 9)

CROSS_VALIDATION_MAX_INFIDELITY = 1.0e-5
# Witness-level backend agreement. The dominant contribution is Pulser's
# finite waveform sampling systematic, which does NOT shrink with ODE
# tolerance, so these thresholds are documented systematic bounds, not
# convergence claims.
#
# A point passes if EITHER criterion holds. The absolute criterion exists
# because a witness can be numerically degenerate: at N=2 the forward/reverse
# TVD is ~6e-6, i.e. below the backends' own absolute agreement, so its
# relative error is uninformative rather than alarming. The relative criterion
# is what actually constrains the manuscript, whose witnesses are O(1e-2).
WITNESS_CROSS_VALIDATION_MAX_ABS_ERROR = 5.0e-4
WITNESS_CROSS_VALIDATION_MAX_REL_ERROR = 5.0e-2
ZERO_GAP_INFIDELITY_TOL = 1.0e-10
ZERO_GAP_TVD_TOL = 1.0e-7
MIN_SIGNAL_TO_FLOOR = 50.0
SCHEDULE_MATCH_TOL = 1.0e-12
# Agreement required between the cancellation-free D estimator and the naive
# sqrt(1-F) estimator, on points where the naive one is not floor-limited.
D_ESTIMATOR_MAX_REL_DISAGREEMENT = 1.0e-9
# The differential-schedule support quantities are exact constructions, so
# they are gated at floating-point level rather than merely recorded.
DIFFERENTIAL_SUPPORT_TOL = 1.0e-12

GRID_CLOSURE_NOTE = (
    "The f-reflection decomposition requires the clock-aligned duration grid "
    "to be closed under d1 -> total-d1. This is asserted, not assumed."
)

if PAPER_TOTAL_NS % DURATION_QUANTUM_NS != 0:
    raise RuntimeError(
        f"PAPER_TOTAL_NS={PAPER_TOTAL_NS} must be divisible by "
        f"{DURATION_QUANTUM_NS} ns so that both the equal three-segment split "
        f"and the clock-aligned f=1/2 split exist."
    )
if not (MIN_DURATION_NS <= PAPER_TOTAL_NS <= MAX_DURATION_NS):
    raise RuntimeError("PAPER_TOTAL_NS is outside the supported duration range.")

# ---------- output directory with timestamp ----------
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUTDIR = Path(f"paper117_path_order_{TIMESTAMP}")
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
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return None
        return obj
    return obj


def package_version(distribution_name):
    """Return installed distribution version without making metadata fatal."""
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


def projective_residual(psi, phi):
    """
    r = min_theta || psi - e^{i theta} phi ||, for normalized pure states.

    Exact relations, with s = |<psi|phi>|:
        r^2 = 2(1 - s),  so  s = 1 - r^2/2  and  r in [0, sqrt(2)].
    Unlike 1 - F = 1 - s^2, this quantity is evaluated without cancellation:
    it is a direct vector norm and is accurate to relative machine precision
    all the way down to r ~ 1e-16.
    """
    psi = normalize_state(psi)
    phi = normalize_state(phi)
    overlap = np.vdot(phi, psi)
    magnitude = abs(overlap)
    phase = overlap / magnitude if magnitude > 0.0 else 1.0 + 0.0j
    return float(np.linalg.norm(psi - phase * phi))


def pure_trace_distance(psi, phi):
    """
    Pure-state trace distance D = sqrt(1 - F), evaluated stably.

    Using s = 1 - r^2/2:
        D^2 = 1 - s^2 = (1-s)(1+s) = (r^2/2)(2 - r^2/2) = r^2 (1 - r^2/4)
        D   = r sqrt(1 - r^2/4)
    which is monotone on r in [0, sqrt(2)] and has no cancellation.
    """
    r = projective_residual(psi, phi)
    r = min(r, math.sqrt(2.0))
    return float(r * math.sqrt(max(0.0, 1.0 - 0.25 * r * r)))


def pure_trace_distance_naive(psi, phi):
    """Legacy sqrt(1-F) estimator, kept only to audit the stable one."""
    fidelity = float(np.clip(state_fidelity(psi, phi), 0.0, 1.0))
    return float(math.sqrt(1.0 - fidelity))


def fidelity_roundoff_scale(psi, phi):
    """
    Cancellation scale of the naive estimator: sqrt(|1-F|) with raw F.

    This is a numerical diagnostic of the LEGACY estimator, not a physical
    distance. It is retained so that the improvement from the stable formula
    is quantified rather than asserted.
    """
    return float(math.sqrt(abs(1.0 - state_fidelity(psi, phi))))


def fubini_study_angle(psi, phi):
    """theta = 2 arcsin(r/2), the cancellation-free form of arccos(s)."""
    r = projective_residual(psi, phi)
    return float(2.0 * math.asin(min(1.0, r / 2.0)))


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
    stable_d = pure_trace_distance(psi_a, psi_b)
    naive_d = pure_trace_distance_naive(psi_a, psi_b)
    return {
        "pair": pair_label,
        "fidelity": state_fidelity(psi_a, psi_b),
        "overlap_abs": overlap_abs,
        "pure_trace_distance": stable_d,
        "pure_trace_distance_naive": naive_d,
        "D_estimator_abs_difference": abs(stable_d - naive_d),
        "projective_residual": projective_residual(psi_a, psi_b),
        "fidelity_roundoff_scale": fidelity_roundoff_scale(psi_a, psi_b),
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
    Build H(dt) for a piecewise constant pulse train.
    Returns list of (H, dt) with dt in microseconds.
    """
    if coords is None:
        coords = nominal_coords(n)
    dim = 2 ** n
    # Pulser convention: angular-frequency interaction coefficient,
    # in rad * micrometer^6 / microsecond.
    C6 = DigitalAnalogDevice.interaction_coeff
    nq = [np.zeros(dim, dtype=float) for _ in range(n)]
    for idx in range(dim):
        b = format(idx, f"0{n}b")
        for q, ch in enumerate(b):
            if ch == "1":
                nq[q][idx] = 1.0
    Hint_diag = np.zeros(dim, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
            Hint_diag += (C6 / (r ** 6)) * (nq[i] * nq[j])
    HX = np.zeros((dim, dim), dtype=np.complex128)
    for idx in range(dim):
        for q in range(n):
            HX[idx ^ (1 << (n - 1 - q)), idx] += omega / 2.0
    Nop = np.sum(nq, axis=0)

    def H_for_det(det):
        # det sign: Pulser convention -Delta * n
        return HX + np.diag(-det * Nop + Hint_diag)

    return [(H_for_det(det), dur_ns / 1000.0) for det, dur_ns in segments]


def exact_state_from_segments(n, segments, omega=OMEGA, coords=None):
    """Exact unitary evolution via scipy.linalg.expm (piecewise constant)."""
    if coords is None:
        return exact_state_from_segments_cached(
            n,
            tuple((float(det), int(dur)) for det, dur in segments),
            float(omega),
        ).copy()
    Hs = exact_hamiltonian_for_segments(n, segments, omega, coords)
    psi = np.zeros(2 ** n, dtype=np.complex128)
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
    psi = np.zeros(2 ** n, dtype=np.complex128)
    psi[0] = 1.0
    for detuning, duration_ns in segments_tuple:
        psi = default_geometry_propagator(n, detuning, duration_ns, omega) @ psi
    normalized = normalize_state(psi)
    normalized.setflags(write=False)
    return normalized


def propagator_cache_report():
    info = default_geometry_propagator.cache_info()
    return {
        "hits": int(info.hits),
        "misses": int(info.misses),
        "entries": int(info.currsize),
        "approx_megabytes": float(info.currsize * (2 ** 8) ** 2 * 16 / 1.0e6),
    }


# ----------------------------------------------------------------------
# Pulser backend
# ----------------------------------------------------------------------
def nominal_coords(n, spacing_um=SPACING_UM):
    return np.array(
        [[(i - (n - 1) / 2) * spacing_um, 0.0] for i in range(n)], dtype=float
    )


def make_register(n, coords=None):
    if coords is None:
        coords = nominal_coords(n)
    return Register(
        {f"q{i}": np.array(coords[i], dtype=float) for i in range(n)}
    )


def add_constant_pulse(seq, omega, detuning, duration_ns, phase=0.0):
    duration_ns = int(duration_ns)
    if duration_ns < MIN_DURATION_NS:
        raise ValueError(
            f"duration_ns={duration_ns} is below minimum {MIN_DURATION_NS} ns"
        )
    if duration_ns > MAX_DURATION_NS:
        raise ValueError(
            f"duration_ns={duration_ns} exceeds maximum {MAX_DURATION_NS} ns"
        )
    if duration_ns % CLOCK_NS != 0:
        raise ValueError(
            f"duration_ns={duration_ns} not aligned to {CLOCK_NS} ns"
        )
    seq.add(
        Pulse(
            ConstantWaveform(duration_ns, float(omega)),
            ConstantWaveform(duration_ns, float(detuning)),
            phase,
        ),
        "rydberg_global",
    )


def build_sequence_explicit(n, segments, omega=OMEGA, coords=None):
    seq = Sequence(make_register(n, coords=coords), DigitalAnalogDevice)
    seq.declare_channel("rydberg_global", "rydberg_global")
    for detuning, duration_ns in segments:
        add_constant_pulse(seq, omega, detuning, int(duration_ns))
    return seq


def compare_sampled_waveforms(n, segments_a, segments_b, omega=OMEGA):
    """
    Compare the sampled control waveforms of two schedules element by element.

    This turns the claim "these two schedules define the same input" from an
    inference about Pulser's sampler into a measured statement.
    """
    samples = []
    for segments in (segments_a, segments_b):
        sequence = build_sequence_explicit(n, segments, omega=omega)
        channel = list(
            sample_sequence(sequence).channel_samples.values()
        )[0]
        samples.append(
            (
                np.asarray(channel.amp, dtype=float),
                np.asarray(channel.det, dtype=float),
                np.asarray(channel.phase, dtype=float),
            )
        )
    (amp_a, det_a, phase_a), (amp_b, det_b, phase_b) = samples
    same_length = (
        amp_a.shape == amp_b.shape
        and det_a.shape == det_b.shape
        and phase_a.shape == phase_b.shape
    )
    if not same_length:
        return {
            "waveforms_identical": False,
            "same_sample_count": False,
            "sample_counts": [int(amp_a.size), int(amp_b.size)],
        }
    return {
        "waveforms_identical": bool(
            np.array_equal(amp_a, amp_b)
            and np.array_equal(det_a, det_b)
            and np.array_equal(phase_a, phase_b)
        ),
        "same_sample_count": True,
        "sample_count": int(amp_a.size),
        "max_abs_amplitude_difference": float(np.max(np.abs(amp_a - amp_b))),
        "max_abs_detuning_difference": float(np.max(np.abs(det_a - det_b))),
        "max_abs_phase_difference": float(np.max(np.abs(phase_a - phase_b))),
    }


def get_final_state_array(result):
    """Try multiple accessors; record which one succeeded."""
    attempts = []
    if hasattr(result, "get_final_state"):
        attempts.append(
            ("get_final_state", lambda: result.get_final_state())
        )
        attempts.append(
            (
                "get_final_state(reduce)",
                lambda: result.get_final_state(
                    reduce_to_basis="ground-rydberg"
                ),
            )
        )
        attempts.append(
            (
                "get_final_state(ignore_phase)",
                lambda: result.get_final_state(ignore_global_phase=True),
            )
        )
    if hasattr(result, "states") and len(result.states) > 0:
        attempts.append(("states[-1]", lambda: result.states[-1]))
    if hasattr(result, "_states") and len(result._states) > 0:
        attempts.append(("_states[-1]", lambda: result._states[-1]))
    last_err = None
    for name, call in attempts:
        try:
            state = call()
            arr = (
                np.asarray(state.full()).ravel()
                if hasattr(state, "full")
                else np.asarray(state).ravel()
            )
            return arr, name
        except Exception as exc:  # noqa: BLE001 - accessor probing
            last_err = exc
    raise RuntimeError(
        f"Could not access final state. Last error: {repr(last_err)}"
    )


def final_statevector_from_segments(n, segments, omega=OMEGA, coords=None):
    seq = build_sequence_explicit(n, segments, omega=omega, coords=coords)
    result = QutipEmulator.from_sequence(seq).run()
    arr, branch = get_final_state_array(result)
    arr = normalize_state(arr)
    if len(arr) != 2 ** n:
        raise RuntimeError(f"Final state dimension {len(arr)} != 2^N={2**n}")
    return pulser_to_standard_basis(arr, n), branch


def pulser_to_standard_basis(psi, n):
    """
    Convert Pulser's local |r>,|g> index convention to the standard bit labels
    used by exact_hamiltonian_for_segments, where 0=|g> and 1=|r>.

    This is a bitwise-complement permutation, not a bit-order reversal.
    """
    psi = normalize_state(psi)
    dim = 2 ** n
    if len(psi) != dim:
        raise ValueError(f"len(psi)={len(psi)} != 2^N={dim}")
    out = np.zeros_like(psi)
    for i in range(dim):
        flipped = "".join(
            "1" if c == "0" else "0" for c in format(i, f"0{n}b")
        )
        out[int(flipped, 2)] = psi[i]
    return normalize_state(out)


def duration_from_loop_ns(n, omega=OMEGA, loop=2.22):
    """Loop-motivated duration, before the divisibility snap."""
    t_us = 2 * math.pi * loop / (math.sqrt(n) * omega)
    t_ns = int(round(1000 * t_us))
    t_ns = int(round(t_ns / CLOCK_NS) * CLOCK_NS)
    if t_ns < MIN_DURATION_NS or t_ns > MAX_DURATION_NS:
        raise ValueError(
            f"Computed duration {t_ns} ns outside allowed range "
            f"[{MIN_DURATION_NS}, {MAX_DURATION_NS}]"
        )
    return t_ns


def paper_total_ns(n):
    """
    Single duration convention for the whole script.

    N=8 uses the frozen PAPER_TOTAL_NS. Every other N snaps the loop-motivated
    duration to the nearest multiple of DURATION_QUANTUM_NS, so the same
    three-segment and half-split guarantees hold there too. The snap shift is
    reported by duration_convention_report().
    """
    if n == 8:
        return PAPER_TOTAL_NS
    raw = duration_from_loop_ns(n)
    if not UNIFY_DURATION_CONVENTION:
        return raw
    snapped = int(
        round(raw / DURATION_QUANTUM_NS) * DURATION_QUANTUM_NS
    )
    if snapped < MIN_DURATION_NS or snapped > MAX_DURATION_NS:
        raise ValueError(f"Snapped duration {snapped} ns outside range.")
    return snapped


def duration_convention_report(n_values):
    rows = []
    for n in n_values:
        raw = duration_from_loop_ns(n)
        used = paper_total_ns(n)
        rows.append(
            {
                "N": int(n),
                "loop_motivated_ns": int(raw),
                "used_ns": int(used),
                "snap_shift_ns": int(used - raw),
                "divisible_by_quantum": bool(
                    used % DURATION_QUANTUM_NS == 0
                ),
            }
        )
    return rows


def split_duration_clock(total_ns, frac):
    total_ns = int(round(total_ns / CLOCK_NS) * CLOCK_NS)
    d1 = int(round(total_ns * frac / CLOCK_NS) * CLOCK_NS)
    d1 = max(MIN_DURATION_NS, min(total_ns - MIN_DURATION_NS, d1))
    d2 = total_ns - d1
    if d2 < MIN_DURATION_NS:
        raise ValueError("Second segment is below the minimum duration.")
    return int(d1), int(d2), int(total_ns)


def duration_grid(total_ns, fractions=FRACTION_GRID):
    """Clock-aligned first-segment durations for the declared fraction grid."""
    return [split_duration_clock(total_ns, f)[0] for f in fractions]


def assert_reflection_closed_grid(total_ns, fractions=FRACTION_GRID):
    """
    Fail fast if the clock-aligned grid is not closed under d1 -> total-d1.

    Without this, the f-reflection decomposition either silently drops rows
    (when the orphan lands on d1 > d2) or raises KeyError (when it does not).
    Both outcomes have been observed for nearby total durations.
    """
    d1_values = duration_grid(total_ns, fractions)
    d1_set = set(d1_values)
    reflected_set = {total_ns - d for d in d1_set}
    if d1_set != reflected_set:
        orphans = sorted(d1_set.symmetric_difference(reflected_set))
        raise AssertionError(
            f"Fraction grid at total={total_ns} ns is not closed under "
            f"f -> 1-f. Orphan durations: {orphans}. {GRID_CLOSURE_NOTE}"
        )
    if len(d1_set) != len(d1_values):
        raise AssertionError(
            f"Fraction grid at total={total_ns} ns has duplicate durations."
        )
    return {
        "total_ns": int(total_ns),
        "durations": [int(x) for x in d1_values],
        "reflection_closed": True,
        "half_split_present": bool((total_ns // 2) in d1_set),
        "reflection_pair_count": int(
            sum(1 for d in d1_values if d <= total_ns - d)
        ),
    }


def weighted_avg(det1, det2, d1_ns, d2_ns):
    return float((det1 * d1_ns + det2 * d2_ns) / (d1_ns + d2_ns))


def integrated_rabi_area(segments, omega=OMEGA):
    """Dimensionless pulse area Phi = int Omega dt, with dt in microseconds."""
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


def differential_schedule_support_us(total_ns, duration1_ns):
    """
    Predicted support length of the forward-minus-reverse detuning difference.

    With f = d1/T and m = min(f, 1-f) the construction gives

        Delta_f(t) - Delta_r(t) = -g  on [0, mT]
                                =  0  on [mT, (1-m)T]
                                = +g  on [(1-m)T, T]      (for f < 1/2)

    so the differential schedule is supported on a set of total length 2*m*T.
    This is a property of make_two_segment_schedules, and it is NOT asserted
    on the strength of this docstring: measure_differential_support() below
    evaluates the actual schedules, the scan records the residuals, and the
    validation gate differential_support_identity_pass requires all three of
    them to sit below DIFFERENTIAL_SUPPORT_TOL.
    """
    f = duration1_ns / total_ns
    return float(2.0 * min(f, 1.0 - f) * total_ns / 1000.0)


def piecewise_value(schedule, t_ns):
    """Value of a piecewise-constant schedule at time t_ns."""
    elapsed = 0
    for value, duration_ns in schedule:
        elapsed += duration_ns
        if t_ns < elapsed:
            return float(value)
    return float(schedule[-1][0])


def measure_differential_support(forward, reverse, total_ns, signed_gap):
    """
    Directly measure the forward-minus-reverse detuning difference.

    Returns the measured support length, the measured magnitude on the
    support, the measured maximum magnitude off the support, and the error
    against the predicted 2*min(f,1-f)*T. Every number is read off the actual
    schedule objects; nothing is taken from the docstring above.
    """
    boundaries = sorted(
        {0, total_ns}
        | {forward[0][1], reverse[0][1]}
    )
    support_ns = 0
    on_support_magnitudes = []
    off_support_magnitudes = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right <= left:
            continue
        probe = (left + right) // 2
        difference = piecewise_value(forward, probe) - piecewise_value(
            reverse, probe
        )
        if abs(difference) > 0.0:
            support_ns += right - left
            on_support_magnitudes.append(abs(difference))
        else:
            off_support_magnitudes.append(abs(difference))
    predicted_us = differential_schedule_support_us(total_ns, forward[0][1])
    return {
        "measured_support_us": float(support_ns / 1000.0),
        "predicted_support_us": predicted_us,
        "support_abs_error_us": float(
            abs(support_ns / 1000.0 - predicted_us)
        ),
        "measured_on_support_magnitude": (
            float(max(on_support_magnitudes)) if on_support_magnitudes else 0.0
        ),
        "on_support_magnitude_error": (
            float(max(abs(x - abs(signed_gap)) for x in on_support_magnitudes))
            if on_support_magnitudes
            else float(abs(signed_gap))
        ),
        "measured_off_support_magnitude": (
            float(max(off_support_magnitudes))
            if off_support_magnitudes
            else 0.0
        ),
    }


def fixed_commutator_norm(n):
    """Fixed metadata value ||[H_X,N]|| for the declared N and Omega."""
    dim = 2 ** n
    HX = np.zeros((dim, dim), dtype=np.complex128)
    Nop = np.zeros((dim, dim), dtype=np.complex128)
    for x in range(dim):
        Nop[x, x] = bin(x).count("1")
        for q in range(n):
            HX[x ^ (1 << q), x] += OMEGA / 2.0
    return float(np.linalg.norm(HX @ Nop - Nop @ HX, ord=2))


# ----------------------------------------------------------------------
# Matched three-segment Magnus diagnostic
# ----------------------------------------------------------------------
def signed_second_order_detuning_area(segments):
    """
    Coefficient multiplying [H_X,N] in the pairwise second-order Magnus sum,
    up to the conventional overall factor/sign:

        A2 = sum_{i>j} (Delta_i - Delta_j) t_i t_j.

    Times are in microseconds. Its sign changes under schedule reversal.
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
    numerical_floor_TVD=0.0,
):
    """
    Compare all six temporal permutations of three detunings.

    Every segment has exactly the same clock-aligned duration, so every
    permutation has identical total duration, integrated Rabi area, weighted
    average detuning, and detuning multiset. The only changed variable is
    order. This is a diagnostic beyond the paper's two-segment claim, not a
    proof of a universal three-segment law.
    """
    if nominal_total_ns is None:
        nominal_total_ns = paper_total_ns(n)
    nominal_total_ns = int(nominal_total_ns)
    if nominal_total_ns % (3 * CLOCK_NS) != 0:
        raise ValueError(
            f"Three-segment total {nominal_total_ns} ns must be divisible by "
            f"3*CLOCK_NS={3 * CLOCK_NS} ns. Use paper_total_ns(N)."
        )
    duration_each_ns = nominal_total_ns // 3
    if duration_each_ns < MIN_DURATION_NS:
        raise ValueError("Clock-aligned three-segment duration is too short.")
    total_ns = 3 * duration_each_ns
    if total_ns != nominal_total_ns:
        raise AssertionError("Three-segment diagnostic changed total duration.")

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
    # The stable estimator makes the projective residual the binding scale,
    # so the expm segmentation floor is used directly rather than being
    # max()-ed against the cancellation-inflated sqrt(|1-F|) value.
    segmentation_expm_state_floor = float(
        segmentation_exact["projective_residual"]
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
    waveform_comparison = compare_sampled_waveforms(
        n, constant_single, constant_three, omega=omega
    )
    pulser_segmentation_check = {
        "structurally_non_independent": bool(
            waveform_comparison["waveforms_identical"]
        ),
        "sampled_waveform_comparison": waveform_comparison,
        "single_branch": branch_single,
        "three_branch": branch_three,
        "interpretation": (
            "Implementation sanity check only, and carries no gate. The "
            "sampled control waveforms of the one-pulse and three-pulse "
            "constant schedules are compared directly above; when they are "
            "identical, Pulser agreement between the two is a statement about "
            "the sampler, not independent physical evidence."
        ),
    }

    reference_total = total_ns
    reference_area = integrated_rabi_area(constant_single, omega)
    rows = []
    low_detuning, middle_detuning, high_detuning = detunings
    for permutation_index, permutation in enumerate(
        itertools.permutations(detunings), start=1
    ):
        segments = [(delta, duration_each_ns) for delta in permutation]
        total = sum(duration for _, duration in segments)
        avg = sum(delta * duration for delta, duration in segments) / total
        area = integrated_rabi_area(segments, omega)
        if total != reference_total:
            raise AssertionError("Permutation changed total duration.")
        if abs(avg - avg_det) > SCHEDULE_MATCH_TOL:
            raise AssertionError("Permutation changed weighted-average detuning.")
        if abs(area - reference_area) > SCHEDULE_MATCH_TOL:
            raise AssertionError("Permutation changed integrated Rabi area.")

        psi = exact_state_from_segments(n, segments, omega=omega)
        metrics = pair_metrics(
            psi, psi_constant, f"permutation_{permutation_index}_vs_constant"
        )
        row = {
            "permutation_index": permutation_index,
            "detuning_order": "|".join(repr(float(x)) for x in permutation),
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
            "A2_signed": signed_second_order_detuning_area(segments),
        }
        row["A2_abs"] = abs(row["A2_signed"])
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
    # repr() round-trips the float exactly, so this identity is tested at
    # machine precision instead of at the precision of a %.8f string.
    tau_us = duration_each_ns / 1000.0
    expected_a2 = [
        2.0 * tau_us ** 2 * (float(o.split("|")[2]) - float(o.split("|")[0]))
        for o in df["detuning_order"]
    ]
    a2_closed_form_error = float(
        np.max(np.abs(df["A2_signed"].to_numpy() - np.asarray(expected_a2)))
    )
    if a2_closed_form_error > 1.0e-12:
        raise AssertionError("Equal-duration A2 closed-form identity failed.")

    def witness_at(index, column="pure_trace_distance"):
        return float(df.loc[df["permutation_index"] == index, column].iloc[0])

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
        MIN_SIGNAL_TO_FLOOR * float(numerical_floor_D),
        MIN_SIGNAL_TO_FLOOR * segmentation_expm_state_floor,
        1.0e-12,
    )
    decision_threshold_TVD = max(
        MIN_SIGNAL_TO_FLOOR * float(numerical_floor_TVD),
        MIN_SIGNAL_TO_FLOOR * float(segmentation_exact["TVD_distribution"]),
        1.0e-12,
    )
    max_equal_a2_D_difference = max(
        equal_a2_differences["D_perm2_minus_perm3_abs"],
        equal_a2_differences["D_perm4_minus_perm5_abs"],
    )
    max_equal_a2_TVD_difference = max(
        equal_a2_differences["TVD_perm2_minus_perm3_abs"],
        equal_a2_differences["TVD_perm4_minus_perm5_abs"],
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

    diagnostic = {
        "backend_for_permutations": "scipy_expm",
        "matched_total_duration_ns": total_ns,
        "segment_duration_ns": duration_each_ns,
        "matched_pulse_area": reference_area,
        "max_weighted_average_error": max_avg_error,
        "A2_closed_form_max_error": a2_closed_form_error,
        "segmentation_artifact_expm": segmentation_exact,
        "segmentation_expm_state_floor": segmentation_expm_state_floor,
        "segmentation_artifact_pulser": segmentation_pulser,
        "pulser_segmentation_check": pulser_segmentation_check,
        "input_numerical_floors": {
            "zero_gap_D": float(numerical_floor_D),
            "zero_gap_TVD": float(numerical_floor_TVD),
        },
        "equal_A2_differences": equal_a2_differences,
        "max_equal_A2_D_difference": max_equal_a2_D_difference,
        "max_equal_A2_TVD_difference": max_equal_a2_TVD_difference,
        "D_decision_threshold": decision_threshold_D,
        "TVD_decision_threshold": decision_threshold_TVD,
        "TVD_resolved_above_threshold": bool(
            max_equal_a2_TVD_difference > decision_threshold_TVD
        ),
        "D_resolved_above_threshold": bool(
            max_equal_a2_D_difference > decision_threshold_D
        ),
        "first_segment_group_summary": first_segment_summary.to_dict(
            orient="records"
        ),
        "nonmid_to_mid_mean_TVD_ratio": nonmid_mean_tvd / mid_mean_tvd,
        "interpretation_boundary": (
            "Applies only to non-reversal path comparisons such as "
            "permutation 2 versus 3. Exact reversal-pair scaling and variable "
            "partitions are outside this script. TVD is the primary "
            "population-level discriminator; D is secondary. Rough 1/TVD^2 "
            "values are scale indicators, not powered sample sizes."
        ),
    }

    df.to_csv(OUTDIR / "three_segment_magnus_diagnostic.csv", index=False)
    first_segment_summary.to_csv(
        OUTDIR / "three_segment_first_segment_groups.csv", index=False
    )
    with open(OUTDIR / "three_segment_magnus_verdict.json", "w") as handle:
        json.dump(json_sanitize(diagnostic), handle, indent=2, allow_nan=False)

    print("\nTHREE-SEGMENT MATCHED MAGNUS DIAGNOSTIC")
    print(
        df[
            [
                "permutation_index",
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
    print(
        "expm segmentation state floor:",
        f"{segmentation_expm_state_floor:.3e}",
        "| Pulser segmentation check is structurally non-independent",
    )
    print("Equal-A2 differences:", equal_a2_differences)
    print(
        "Thresholds  D:",
        f"{decision_threshold_D:.3e}",
        " TVD:",
        f"{decision_threshold_TVD:.3e}",
    )
    print("\nFIRST-SEGMENT GROUP SUMMARY")
    print(first_segment_summary.to_string(index=False))
    return df, diagnostic


# ----------------------------------------------------------------------
# 117-point scan
# ----------------------------------------------------------------------
def spearman_corr_tiesafe(x, y):
    """Tie-aware Spearman rank correlation."""
    if not SCIPY_AVAILABLE:
        raise RuntimeError("SciPy is required for tie-aware Spearman.")
    rho, _ = spearmanr(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return float(rho)


def ols_fit_with_intercept(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    prediction = slope * x + intercept
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(slope), float(intercept), float(1.0 - ss_res / ss_tot)


def fit_summary_for_witness(sub_df, xcol, ycol, subset_name):
    x = sub_df[xcol].values.astype(float)
    y = sub_df[ycol].values.astype(float)
    slope, intercept, r2 = ols_fit_with_intercept(x, y)
    prediction = slope * x + intercept
    return {
        "subset": subset_name,
        "x": xcol,
        "y": ycol,
        "n": int(len(sub_df)),
        "spearman_rho": spearman_corr_tiesafe(x, y),
        "ols_slope": slope,
        "ols_intercept": intercept,
        "ols_R2": r2,
        "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "mae": float(np.mean(np.abs(y - prediction))),
    }


# The manuscript's comparison is over a small SPECIFIED mechanistic proxy set.
# Keeping the set small and fixed is what keeps this an exclusion test rather
# than a function search.
#
# Naming note: the 117-point duration/gap grid is FROZEN (it is fixed before
# the states are evolved and is asserted for reflection closure). The proxy
# set is only SPECIFIED, not preregistered: C_lin was introduced after early
# data analysis, so no claim of preregistration is made anywhere.
SPECIFIED_PROXY_COLUMNS = ["C_BCH_shape", "C_linear_support"]

# Reference shapes are recorded but NEVER enter the preference verdict. They
# exist to answer one question: is the specified-set winner's residual unstructured,
# or is there an obvious shape it is missing?
#
# sin^2(pi f) is an exploratory smooth symmetric reference shape used only to
# diagnose structured residuals. It is not derived here and does not enter the
# specified proxy comparison.
REFERENCE_SHAPE_COLUMNS = ["C_sin2_shape"]
ALL_SHAPE_COLUMNS = SPECIFIED_PROXY_COLUMNS + REFERENCE_SHAPE_COLUMNS
PROXY_COLUMNS = SPECIFIED_PROXY_COLUMNS
WITNESS_COLUMNS = [
    "pure_trace_distance",
    "phase_gap_BC_minus_overlap",
    "TVD_distribution",
]


def run_117_scan():
    print("\n" + "=" * 80 + "\nB) 117-POINT SIGNED-RESPONSE SCAN\n" + "=" * 80)
    n = 8
    total_ns = paper_total_ns(n)
    grid_report = assert_reflection_closed_grid(total_ns)
    comm_norm = fixed_commutator_norm(n)
    total_us = total_ns / 1000.0
    max_a2 = float(np.max(GAP_GRID)) * total_us ** 2 * 0.25
    perturbative_validity = {
        "max_abs_A2_us2_rad": max_a2,
        "commutator_norm_rad_per_us": comm_norm,
        "max_A2_times_commutator_norm": max_a2 * comm_norm,
        "second_order_magnus_truncation_valid": bool(
            max_a2 * comm_norm < 1.0
        ),
        "note": (
            "C_BCH is the second-order Magnus shape. When "
            "max|A2|*||[H_X,N]|| is not small compared with 1, the truncation "
            "that produces C_BCH is outside its domain of validity on this "
            "grid, and its poor fit is expected rather than informative."
        ),
    }
    print(
        f"N={n}, total_ns={total_ns}, comm_norm={comm_norm:.6f}, "
        f"max|A2|*||[H_X,N]||={max_a2 * comm_norm:.4f}"
    )
    print(
        f"Grid: {len(GAP_GRID)} gaps x {len(FRACTION_GRID)} fractions, "
        f"reflection-closed={grid_report['reflection_closed']}, "
        f"half-split present={grid_report['half_split_present']}"
    )

    rows = []
    idx = 0
    for gap in GAP_GRID:
        for frac_nom in FRACTION_GRID:
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
            fwd_avg = weighted_avg(delta1, delta2, d1, d2)
            rev_avg = weighted_avg(delta2, delta1, d2, d1)
            if fwd_total != rev_total:
                raise AssertionError("Total duration mismatch.")
            if abs(fwd_area - rev_area) > SCHEDULE_MATCH_TOL:
                raise AssertionError("Integrated Rabi area mismatch.")
            if abs(fwd_avg - rev_avg) > SCHEDULE_MATCH_TOL:
                raise AssertionError("Weighted-average detuning mismatch.")
            if abs(fwd_avg - AVG_DETUNING) > SCHEDULE_MATCH_TOL:
                raise AssertionError("Weighted average off target.")

            support_measurement = measure_differential_support(
                forward_segments, reverse_segments, T, gap
            )
            psi_f = exact_state_from_segments(n, forward_segments)
            psi_r = exact_state_from_segments(n, reverse_segments)
            metrics = pair_metrics(psi_f, psi_r, "forward_vs_reverse")
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
                "weighted_avg_detuning": fwd_avg,
                "differential_support_us": support_measurement[
                    "measured_support_us"
                ],
                "differential_support_predicted_us": support_measurement[
                    "predicted_support_us"
                ],
                "differential_support_abs_error_us": support_measurement[
                    "support_abs_error_us"
                ],
                "differential_on_support_magnitude_error": (
                    support_measurement["on_support_magnitude_error"]
                ),
                "differential_off_support_magnitude": support_measurement[
                    "measured_off_support_magnitude"
                ],
                "C_BCH_shape": abs(gap) * f_actual * (1.0 - f_actual),
                "C_linear_support": abs(gap) * min(f_actual, 1.0 - f_actual),
                "C_sin2_shape": abs(gap) * math.sin(math.pi * f_actual) ** 2,
                "backend": "scipy_expm",
            }
            row.update(metrics)
            row["rough_shot_scale_1_over_TVD2"] = (
                float(1.0 / metrics["TVD_distribution"] ** 2)
                if metrics["TVD_distribution"] > 0
                else None
            )
            row["elapsed_seconds"] = time.perf_counter() - point_start
            rows.append(row)
            print(
                f"[{idx:03d}/{len(GAP_GRID) * len(FRACTION_GRID)}] "
                f"gap={gap:.4f} f={f_actual:.4f} "
                f"D={metrics['pure_trace_distance']:.6f} "
                f"TVD={metrics['TVD_distribution']:.6f}"
            )
    df = pd.DataFrame(rows)

    zero = df[df["gap"] == 0.0]
    nonzero = df[df["gap"] > 0.0].copy()

    numerical_floor = {
        "max_abs_infidelity": float(
            np.max(np.abs(1.0 - zero["fidelity"].to_numpy(dtype=float)))
        ),
        "max_D_stable": float(np.max(zero["pure_trace_distance"])),
        "max_D_naive_legacy": float(np.max(zero["pure_trace_distance_naive"])),
        "max_projective_residual": float(np.max(zero["projective_residual"])),
        "max_fidelity_roundoff_scale_legacy": float(
            np.max(zero["fidelity_roundoff_scale"])
        ),
        "max_TVD": float(np.max(zero["TVD_distribution"])),
        "max_phase_gap": float(
            np.max(np.abs(zero["phase_gap_BC_minus_overlap"]))
        ),
        "note": (
            "max_D_stable is the floor of the estimator actually used for the "
            "reported witness. The legacy sqrt(1-F) values are retained only "
            "to quantify how much the old estimator was inflated by "
            "cancellation."
        ),
    }
    if numerical_floor["max_abs_infidelity"] > ZERO_GAP_INFIDELITY_TOL:
        raise AssertionError(f"Zero-gap infidelity gate failed: {numerical_floor}")
    if numerical_floor["max_TVD"] > ZERO_GAP_TVD_TOL:
        raise AssertionError(f"Zero-gap TVD gate failed: {numerical_floor}")

    # Cross-check the two D estimators where the naive one is not floor
    # limited, i.e. on the nonzero-gap physics points.
    estimator_rel_disagreement = float(
        np.max(
            np.abs(
                nonzero["pure_trace_distance"].to_numpy(dtype=float)
                - nonzero["pure_trace_distance_naive"].to_numpy(dtype=float)
            )
            / nonzero["pure_trace_distance"].to_numpy(dtype=float)
        )
    )
    if estimator_rel_disagreement > D_ESTIMATOR_MAX_REL_DISAGREEMENT:
        raise AssertionError(
            f"Stable and naive D estimators disagree by "
            f"{estimator_rel_disagreement:.3e} on physics points."
        )

    nonzero["D_over_gap"] = nonzero["pure_trace_distance"] / nonzero["gap"]
    gap_linearity_rows = []
    for frac_actual, group in nonzero.groupby("frac_actual", sort=True):
        ratios = group.sort_values("gap")["D_over_gap"].to_numpy(dtype=float)
        gap_linearity_rows.append(
            {
                "frac_actual": float(frac_actual),
                "n_gaps": int(len(ratios)),
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
        numerical_floor["max_D_stable"],
        numerical_floor["max_projective_residual"],
        1.0e-16,
    )
    signal_to_floor = min_signal_D / effective_D_floor
    legacy_effective_D_floor = max(
        numerical_floor["max_D_naive_legacy"],
        numerical_floor["max_fidelity_roundoff_scale_legacy"],
        1.0e-16,
    )
    if signal_to_floor < MIN_SIGNAL_TO_FLOOR:
        raise AssertionError(
            f"Weakest signal is too close to numerical floor: "
            f"ratio={signal_to_floor:.3e}, floor={numerical_floor}"
        )

    df.to_csv(OUTDIR / "scan117_statevector_metrics.csv", index=False)

    summaries = []
    residual_rows = []
    subset_map = {
        "nonzero_gap_primary": nonzero,
        "all_points_sensitivity_only": df,
    }
    for subset_name, sub_df in subset_map.items():
        for proxy in ALL_SHAPE_COLUMNS:
            for witness in WITNESS_COLUMNS:
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

    # f-reflection decomposition. Closure was asserted above, so every kept
    # row is guaranteed to have a partner and the expected count is fixed.
    lookup = {
        (round(float(row.gap), 12), int(row.duration1_ns)): row
        for row in nonzero.itertuples(index=False)
    }
    expected_even_odd_rows = grid_report["reflection_pair_count"] * int(
        (GAP_GRID > 0).sum()
    )
    even_odd_rows = []
    for row in nonzero.itertuples(index=False):
        if row.duration1_ns > row.duration2_ns:
            continue
        key = (round(float(row.gap), 12), int(row.duration2_ns))
        if key not in lookup:
            raise AssertionError(
                f"Reflection partner missing for {key}. {GRID_CLOSURE_NOTE}"
            )
        reflected = lookup[key]
        out = {
            "gap": float(row.gap),
            "f_left": float(row.frac_actual),
            "f_right": float(reflected.frac_actual),
            "is_self_reflected": bool(
                int(row.duration1_ns) == int(row.duration2_ns)
            ),
        }
        for witness in WITNESS_COLUMNS:
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
    if len(even_odd_df) != expected_even_odd_rows:
        raise AssertionError(
            f"f-reflection decomposition produced {len(even_odd_df)} rows, "
            f"expected {expected_even_odd_rows}."
        )

    # Signed-gap relation: algebraic schedule relabelling only.
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
        symmetry_rows.append(
            {
                "gap_positive": float(row.gap),
                "f_original": float(row.frac_actual),
                "f_reflected": float(1.0 - row.frac_actual),
                "forward_reflected_equals_reverse_negative_exactly": (
                    f_pos_reflected == r_negative
                ),
                "reverse_reflected_equals_forward_negative_exactly": (
                    r_pos_reflected == f_negative
                ),
                "schedule_parameter_error": schedule_error,
            }
        )
    symmetry_df = pd.DataFrame(symmetry_rows)
    symmetry_df.to_csv(OUTDIR / "signed_gap_symmetry_audit.csv", index=False)
    exact_schedule_identity_fraction = float(
        (
            symmetry_df["forward_reflected_equals_reverse_negative_exactly"]
            & symmetry_df["reverse_reflected_equals_forward_negative_exactly"]
        ).mean()
    )
    if max_schedule_parameter_error > SCHEDULE_MATCH_TOL:
        raise AssertionError(
            f"Signed-gap schedule identity failed: "
            f"{max_schedule_parameter_error}"
        )

    primary = summary_df[summary_df["subset"] == "nonzero_gap_primary"]
    d_models = primary[primary["y"] == "pure_trace_distance"].set_index("x")
    pooled_ranked = (
        d_models.loc[SPECIFIED_PROXY_COLUMNS, "rmse"].sort_values()
    )

    # A pooled fit across all gaps mixes the schedule-shape question (the
    # f-dependence) with the residual g-dependence of D. Each gap on its own
    # is a clean fixed-g test of the f-shape, so the per-gap comparison is
    # primary and the pooled fit is reported only as a sensitivity.
    per_gap_rows = []
    for gap_value, group in nonzero.groupby("gap", sort=True):
        entry = {"gap": float(gap_value), "n": int(len(group))}
        for proxy in ALL_SHAPE_COLUMNS:
            fit = fit_summary_for_witness(
                group, proxy, "pure_trace_distance", f"gap={gap_value:.4f}"
            )
            entry[f"rmse_{proxy}"] = fit["rmse"]
            entry[f"R2_{proxy}"] = fit["ols_R2"]
        entry["winner"] = min(
            SPECIFIED_PROXY_COLUMNS, key=lambda p: entry[f"rmse_{p}"]
        )
        ordered = sorted(
            SPECIFIED_PROXY_COLUMNS, key=lambda p: entry[f"rmse_{p}"]
        )
        entry["winner_to_runner_up_rmse_ratio"] = float(
            entry[f"rmse_{ordered[0]}"] / entry[f"rmse_{ordered[1]}"]
        )
        # How much of the specified-set winner's residual is explained by a shape
        # outside the specified set. A ratio far below 1 means the winner's
        # residual is structured, not noise.
        best_reference = min(
            REFERENCE_SHAPE_COLUMNS, key=lambda p: entry[f"rmse_{p}"]
        )
        entry["best_reference_shape"] = best_reference
        entry["reference_to_specified_winner_rmse_ratio"] = float(
            entry[f"rmse_{best_reference}"] / entry[f"rmse_{entry['winner']}"]
        )
        per_gap_rows.append(entry)
    per_gap_df = pd.DataFrame(per_gap_rows)
    per_gap_df.to_csv(
        OUTDIR / "proxy_comparison_by_gap.csv", index=False
    )
    per_gap_winners = list(per_gap_df["winner"])
    unanimous = len(set(per_gap_winners)) == 1
    per_gap_preferred = per_gap_winners[0] if unanimous else None
    pooled_preferred = str(pooled_ranked.index[0])
    rankings_agree = bool(unanimous and per_gap_preferred == pooled_preferred)

    reference_ratio = float(
        per_gap_df["reference_to_specified_winner_rmse_ratio"].max()
    )
    specified_winner_max_rmse = float(
        per_gap_df[[f"rmse_{p}" for p in SPECIFIED_PROXY_COLUMNS]]
        .min(axis=1)
        .max()
    )
    residual_structure_diagnostic = {
        "verdict_role": "NOT_PART_OF_VERDICT",
        "reference_shapes": list(REFERENCE_SHAPE_COLUMNS),
        "worst_case_reference_to_winner_rmse_ratio": reference_ratio,
        "specified_winner_worst_per_gap_rmse": specified_winner_max_rmse,
        "specified_winner_worst_rmse_as_fraction_of_D_range": float(
            specified_winner_max_rmse
            / float(nonzero["pure_trace_distance"].max())
        ),
        "residual_is_structured": bool(reference_ratio < 0.5),
        "reference_shape_status": (
            "Exploratory smooth symmetric reference shape used only to "
            "diagnose structured residuals. It is not derived here, has no "
            "theoretical status in this manuscript, and does not enter the "
            "specified proxy comparison."
        ),
        "interpretation": (
            "A reference shape outside the specified set reduces the per-gap "
            "RMSE by the stated factor. When that factor is well below one, "
            "the specified-set winner's residual is structured rather than noise, "
            "so the comparison should be reported as excluding the losing "
            "specified proxy as a sufficient global finite-time organizer, "
            "and NOT as support for the winner being the correct functional "
            "form."
        ),
    }

    proxy_verdict = {
        "specified_proxy_set": list(SPECIFIED_PROXY_COLUMNS),
        "reference_shapes_excluded_from_verdict": list(
            REFERENCE_SHAPE_COLUMNS
        ),
        "primary_test": "per_gap_fixed_g_f_shape",
        "per_gap": {
            "rows": per_gap_df.to_dict(orient="records"),
            "winners": per_gap_winners,
            "unanimous": bool(unanimous),
            "preferred_within_specified_set": per_gap_preferred,
            "median_winner_to_runner_up_rmse_ratio": float(
                per_gap_df["winner_to_runner_up_rmse_ratio"].median()
            ),
        },
        "pooled_sensitivity": {
            "D_R2": {
                p: float(d_models.loc[p, "ols_R2"])
                for p in SPECIFIED_PROXY_COLUMNS
            },
            "D_RMSE": {
                p: float(d_models.loc[p, "rmse"])
                for p in SPECIFIED_PROXY_COLUMNS
            },
            "rmse_ranking_best_first": list(pooled_ranked.index),
            "preferred_within_specified_set": pooled_preferred,
            "caveat": (
                "Pooling all gaps into one fit with a single slope and "
                "intercept lets the residual g-dependence of D contaminate a "
                "question that is purely about the f-shape. This ranking is "
                "not the primary result."
            ),
        },
        "per_gap_and_pooled_rankings_agree": rankings_agree,
        "ranking_stability_warning": (
            None
            if rankings_agree
            else (
                "The per-gap and pooled comparisons select DIFFERENT proxies. "
                "The pooled ranking is an artefact of mixing the f-shape "
                "question with the g-dependence and must not be quoted as a "
                "schedule-shape result."
            )
        ),
        "residual_structure_diagnostic": residual_structure_diagnostic,
        "boundary": (
            "Only a small specified mechanistic proxy set is compared; this "
            "is not a search over arbitrary functions, and the set is "
            "specified rather than preregistered. The result excludes the "
            "losing specified proxy as a sufficient global finite-time "
            "organizer on this grid. It does not falsify its possible "
            "small-partition asymptotic role or the Magnus expansion, whose "
            "second-order truncation is not uniformly controlled across this "
            "grid: see perturbative_validity. It is also not evidence that "
            "the winner is the correct functional form: see "
            "residual_structure_diagnostic. Fixed T cannot determine T "
            "versus T^2 scaling."
        ),
        "result_statement": (
            "C_BCH = |g| f(1-f) is insufficient as a global finite-time "
            "organizing proxy on the tested grid."
        ),
        "set_provenance": (
            "Specified mechanistic proxy set, not a preregistration. The "
            "117-point grid is frozen; the proxy set is not."
        ),
    }

    cert = {
        "experiment": "fixed-T multi-proxy signed-response scan",
        "primary_backend": "scipy_expm",
        "N": n,
        "points": int(len(df)),
        "grid": {
            "gaps": [float(x) for x in GAP_GRID],
            "fracs": [float(x) for x in FRACTION_GRID],
            "closure": grid_report,
        },
        "zero_gap_control_points": int((df["gap"] == 0).sum()),
        "nonzero_gap_points": int((df["gap"] > 0).sum()),
        "total_duration_ns": total_ns,
        "Omega": OMEGA,
        "pulse_area": float(df["pulse_area"].iloc[0]),
        "avg_detuning_target": AVG_DETUNING,
        "perturbative_validity": perturbative_validity,
        "differential_support_identity": {
            "claim": (
                "Delta_f - Delta_r is +/-|g| on two end windows and exactly "
                "zero in between, with total support 2*min(f,1-f)*T."
            ),
            "evidence_class": (
                "Measured on every nonzero-gap schedule in this run, not "
                "quoted from a derivation."
            ),
            "max_support_abs_error_us": float(
                nonzero["differential_support_abs_error_us"].max()
            ),
            "max_on_support_magnitude_error": float(
                nonzero["differential_on_support_magnitude_error"].max()
            ),
            "max_off_support_magnitude": float(
                nonzero["differential_off_support_magnitude"].max()
            ),
            "tolerance": DIFFERENTIAL_SUPPORT_TOL,
            "passes": bool(
                max(
                    float(nonzero["differential_support_abs_error_us"].max()),
                    float(
                        nonzero[
                            "differential_on_support_magnitude_error"
                        ].max()
                    ),
                    float(
                        nonzero["differential_off_support_magnitude"].max()
                    ),
                )
                <= DIFFERENTIAL_SUPPORT_TOL
            ),
        },
        "zero_gap_numerical_floor": numerical_floor,
        "effective_D_floor": effective_D_floor,
        "legacy_effective_D_floor": legacy_effective_D_floor,
        "floor_improvement_factor": float(
            legacy_effective_D_floor / effective_D_floor
        ),
        "D_estimator_max_relative_disagreement": estimator_rel_disagreement,
        "weakest_nonzero_D": min_signal_D,
        "weakest_D_to_floor_ratio": signal_to_floor,
        "rough_shot_scale": {
            "minimum_best_case": float(
                nonzero["rough_shot_scale_1_over_TVD2"].min()
            ),
            "maximum_weakest_case": float(
                nonzero["rough_shot_scale_1_over_TVD2"].max()
            ),
            "boundary": (
                "1/TVD^2 is a rough scale only, not a powered hypothesis-test "
                "sample-size calculation. It, not the float64 floor, is what "
                "bounds experimental resolvability."
            ),
        },
        "D_over_gap_linearity_by_fraction": gap_linearity_df.to_dict(
            orient="records"
        ),
        "proxy_comparison": proxy_verdict,
        "signed_gap_schedule_identity": {
            "schedule_identity": "S_f(1-f,+g)=S_r(f,-g), S_r(1-f,+g)=S_f(f,-g)",
            "implied_witness_identity": "D(1-f,+g)=D(f,-g)",
            "max_schedule_parameter_error": max_schedule_parameter_error,
            "schedule_match_tolerance": SCHEDULE_MATCH_TOL,
            "exact_tuple_identity_fraction": exact_schedule_identity_fraction,
            "evidence_class": (
                "Algebraic schedule identity and software regression check. "
                "The implied D/TVD equality cannot fail independently of the "
                "schedule check and is therefore not reported as a witness."
            ),
        },
        "f_reflection": {
            "rows": int(len(even_odd_df)),
            "expected_rows": int(expected_even_odd_rows),
            "self_reflected_rows": int(
                even_odd_df["is_self_reflected"].sum()
            ),
            "max_D_odd_fraction": float(
                even_odd_df["pure_trace_distance_odd_fraction"].max()
            ),
        },
        "fits": summary_df.to_dict(orient="records"),
        "interpretation_boundary": (
            "Nonzero-gap points are primary. The repeated zero-gap controls "
            "are numerical-floor controls and sensitivity fits only. No iid "
            "significance p-values or strict monotonicity claims are made."
        ),
    }
    with open(OUTDIR / "scan117_certificate.json", "w") as handle:
        json.dump(json_sanitize(cert), handle, indent=2, allow_nan=False)

    print("\nPROXY COMPARISON, PRIMARY: per-gap fixed-g f-shape RMSE")
    print(
        per_gap_df[
            ["gap"]
            + [f"rmse_{p}" for p in ALL_SHAPE_COLUMNS]
            + ["winner", "reference_to_specified_winner_rmse_ratio"]
        ].to_string(index=False)
    )
    print(
        "per-gap specified-set winner:",
        per_gap_preferred if unanimous else f"not unanimous {per_gap_winners}",
    )
    print(
        "residual structure: best reference shape reaches",
        f"{reference_ratio:.3g}x the specified-set winner's per-gap RMSE",
        "-> structured" if residual_structure_diagnostic[
            "residual_is_structured"
        ] else "-> not obviously structured",
    )
    print("\nPOOLED SENSITIVITY (not primary). Column x is the proxy.")
    print(
        d_models.loc[
            SPECIFIED_PROXY_COLUMNS, ["ols_R2", "rmse", "spearman_rho"]
        ]
        .sort_values("rmse")
        .reset_index()
        .to_string(index=False)
    )
    print("pooled specified-set winner:", pooled_preferred)
    if not rankings_agree:
        print(
            "RANKING WARNING:",
            proxy_verdict["ranking_stability_warning"],
        )
    print("\nALL SHAPE x WITNESS FITS (nonzero-gap subset)")
    print(
        summary_df[summary_df["subset"] == "nonzero_gap_primary"][
            ["x", "y", "n", "spearman_rho", "ols_slope", "ols_intercept",
             "ols_R2", "rmse"]
        ].to_string(index=False)
    )
    print("\nZero-gap floor (stable):", f"{effective_D_floor:.3e}")
    print(
        "Zero-gap floor (legacy sqrt(1-F)):",
        f"{legacy_effective_D_floor:.3e}",
        f"-> improvement x{cert['floor_improvement_factor']:.3g}",
    )
    print("Weakest D / effective floor:", f"{signal_to_floor:.3e}")
    print(
        "D estimator max relative disagreement:",
        f"{estimator_rel_disagreement:.3e}",
    )
    print(
        "f-reflection rows:",
        len(even_odd_df),
        "expected:",
        expected_even_odd_rows,
    )
    print("\nSaved outputs under", OUTDIR)
    return df, summary_df, residual_df, even_odd_df, symmetry_df, cert


# ----------------------------------------------------------------------
# Cross-validation with Pulser: state level AND witness level
# ----------------------------------------------------------------------
def run_cross_validation():
    print(
        "\n" + "=" * 80
        + "\nC) CROSS-VALIDATION: Pulser vs expm (state and witness level)\n"
        + "=" * 80
    )
    test_specs = [
        (2, 0.35, 0.13),
        (5, 0.50, 0.18),
        (8, 0.10, 0.03),
        (8, 0.20, 0.12),
        (8, 0.50, 0.12),
        (8, 0.80, 0.24),
    ]
    state_rows = []
    witness_rows = []
    for n, fraction, gap in test_specs:
        total_ns = paper_total_ns(n)
        d1, _, _ = split_duration_clock(total_ns, fraction)
        forward, reverse, f_actual = make_two_segment_schedules(
            total_ns, d1, gap, AVG_DETUNING
        )
        pulser_states = {}
        expm_states = {}
        for order_name, segments in [("forward", forward), ("reverse", reverse)]:
            psi_p, branch = final_statevector_from_segments(n, segments)
            psi_e = exact_state_from_segments(n, segments)
            pulser_states[order_name] = psi_p
            expm_states[order_name] = psi_e
            fidelity = state_fidelity(psi_p, psi_e)
            state_rows.append(
                {
                    "N": n,
                    "frac_actual": f_actual,
                    "gap": gap,
                    "order": order_name,
                    "segments": repr(segments),
                    "fidelity": fidelity,
                    "infidelity": abs(1.0 - fidelity),
                    "projective_residual": projective_residual(psi_p, psi_e),
                    "pulser_branch": branch,
                }
            )
        expm_metrics = pair_metrics(
            expm_states["forward"], expm_states["reverse"], "expm"
        )
        pulser_metrics = pair_metrics(
            pulser_states["forward"], pulser_states["reverse"], "pulser"
        )
        d_abs = abs(
            pulser_metrics["pure_trace_distance"]
            - expm_metrics["pure_trace_distance"]
        )
        tvd_abs = abs(
            pulser_metrics["TVD_distribution"]
            - expm_metrics["TVD_distribution"]
        )
        d_rel = d_abs / expm_metrics["pure_trace_distance"]
        tvd_rel = tvd_abs / expm_metrics["TVD_distribution"]
        d_ok = (
            d_abs <= WITNESS_CROSS_VALIDATION_MAX_ABS_ERROR
            or d_rel <= WITNESS_CROSS_VALIDATION_MAX_REL_ERROR
        )
        tvd_ok = (
            tvd_abs <= WITNESS_CROSS_VALIDATION_MAX_ABS_ERROR
            or tvd_rel <= WITNESS_CROSS_VALIDATION_MAX_REL_ERROR
        )
        witness_rows.append(
            {
                "N": n,
                "frac_actual": f_actual,
                "gap": gap,
                "D_expm": expm_metrics["pure_trace_distance"],
                "D_pulser": pulser_metrics["pure_trace_distance"],
                "D_abs_error": d_abs,
                "D_relative_error": d_rel,
                "D_pass": bool(d_ok),
                "D_relative_criterion_uninformative": bool(
                    d_rel > WITNESS_CROSS_VALIDATION_MAX_REL_ERROR
                ),
                "TVD_expm": expm_metrics["TVD_distribution"],
                "TVD_pulser": pulser_metrics["TVD_distribution"],
                "TVD_abs_error": tvd_abs,
                "TVD_relative_error": tvd_rel,
                "TVD_pass": bool(tvd_ok),
                "TVD_relative_criterion_uninformative": bool(
                    tvd_rel > WITNESS_CROSS_VALIDATION_MAX_REL_ERROR
                ),
            }
        )
        print(
            f"N={n} f={f_actual:.6f} gap={gap:.3f}  "
            f"D {expm_metrics['pure_trace_distance']:.9f} vs "
            f"{pulser_metrics['pure_trace_distance']:.9f} "
            f"(abs {d_abs:.2e}, rel {d_rel:.2e})  "
            f"TVD abs {tvd_abs:.2e}, rel {tvd_rel:.2e}"
        )

    state_df = pd.DataFrame(state_rows)
    witness_df = pd.DataFrame(witness_rows)
    max_infidelity = float(state_df["infidelity"].max())
    max_witness_abs = float(
        max(witness_df["D_abs_error"].max(), witness_df["TVD_abs_error"].max())
    )
    max_witness_rel = float(
        max(
            witness_df["D_relative_error"].max(),
            witness_df["TVD_relative_error"].max(),
        )
    )
    all_witness_points_pass = bool(
        witness_df["D_pass"].all() and witness_df["TVD_pass"].all()
    )
    # The witness magnitude above which the two backends agree to 1 percent,
    # implied directly by the measured absolute agreement. This is the number
    # the manuscript should quote when claiming backend independence.
    witness_scale_for_one_percent = float(max_witness_abs / 0.01)
    # Rows whose witness is itself below the measured absolute agreement.
    degenerate_rows = []
    nondegenerate_relative_errors = []
    for record in witness_df.to_dict(orient="records"):
        for witness_name in ("D", "TVD"):
            relative_error = record[f"{witness_name}_relative_error"]
            if record[f"{witness_name}_expm"] < max_witness_abs:
                degenerate_rows.append(
                    {
                        "N": record["N"],
                        "gap": record["gap"],
                        "witness": witness_name,
                        "expm_value": record[f"{witness_name}_expm"],
                        "absolute_agreement_scale": max_witness_abs,
                        "relative_error": relative_error,
                    }
                )
            else:
                nondegenerate_relative_errors.append(float(relative_error))
    max_nondegenerate_relative_error = (
        float(max(nondegenerate_relative_errors))
        if nondegenerate_relative_errors
        else float("nan")
    )

    if max_infidelity > CROSS_VALIDATION_MAX_INFIDELITY:
        raise AssertionError(
            f"State-level cross-validation failed: max infidelity "
            f"{max_infidelity:.3e} > {CROSS_VALIDATION_MAX_INFIDELITY:.3e}"
        )
    if not all_witness_points_pass:
        failures = witness_df[~(witness_df["D_pass"] & witness_df["TVD_pass"])]
        raise AssertionError(
            "Witness-level cross-validation failed on:\n"
            + failures.to_string(index=False)
        )

    state_df.to_csv(OUTDIR / "cross_validation_states.csv", index=False)
    witness_df.to_csv(OUTDIR / "cross_validation_witnesses.csv", index=False)
    cv_cert = {
        "state_level": {
            "points": state_df.to_dict(orient="records"),
            "max_infidelity": max_infidelity,
            "max_allowed_infidelity": CROSS_VALIDATION_MAX_INFIDELITY,
        },
        "witness_level": {
            "points": witness_df.to_dict(orient="records"),
            "criterion": (
                "A point passes if its absolute OR relative backend "
                "disagreement is within tolerance."
            ),
            "max_absolute_error": max_witness_abs,
            "max_allowed_absolute_error": (
                WITNESS_CROSS_VALIDATION_MAX_ABS_ERROR
            ),
            "max_relative_error_all_rows": max_witness_rel,
            "max_nondegenerate_relative_error": (
                max_nondegenerate_relative_error
            ),
            "degenerate_witness_count": len(degenerate_rows),
            "relative_error_reading_note": (
                "max_relative_error_all_rows is inflated by witnesses that "
                "are themselves smaller than the measured absolute backend "
                "agreement; it must not be read as a backend disagreement. "
                "max_nondegenerate_relative_error is the figure that "
                "characterises Pulser-versus-exponentiation agreement on the "
                "witnesses the manuscript actually uses."
            ),
            "max_allowed_relative_error": (
                WITNESS_CROSS_VALIDATION_MAX_REL_ERROR
            ),
            "all_points_pass": all_witness_points_pass,
            "witness_magnitude_for_one_percent_agreement": (
                witness_scale_for_one_percent
            ),
            "degenerate_witness_rows": degenerate_rows,
            "degenerate_witness_note": (
                "A row is listed above when its witness value is smaller than "
                "the largest absolute backend disagreement measured in this "
                "run, so that its relative error is a ratio of two "
                "noise-scale quantities and carries no information. The "
                "listed values are measured, not assumed."
            ),
            "why_this_matters": (
                "The manuscript's claims are stated in terms of D and TVD, "
                "not in terms of state fidelity. State-level infidelity of "
                "order 1e-7 corresponds to witness-level agreement of only a "
                "few significant figures, so the witness-level number is the "
                "one that bounds the reported quantities."
            ),
            "known_systematic": (
                "The remaining Pulser-versus-exponentiation discrepancy is "
                "recorded as a backend-dependent numerical systematic. Its "
                "detailed dependence on solver tolerances and waveform "
                "sampling is not established by this script."
            ),
        },
        "versions": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": package_version("scipy"),
            "pulser": package_version("pulser"),
            "pulser_simulation": package_version("pulser-simulation"),
        },
    }
    with open(OUTDIR / "cross_validation.json", "w") as handle:
        json.dump(json_sanitize(cv_cert), handle, indent=2, allow_nan=False)
    print(
        f"CROSS-VALIDATION PASS: state max infidelity={max_infidelity:.3e}, "
        f"witness max abs error={max_witness_abs:.3e}, "
        f"1%-agreement witness scale={witness_scale_for_one_percent:.3e}"
    )
    print(
        f"  max relative error over all rows = {max_witness_rel:.4%} "
        f"(includes {len(degenerate_rows)} degenerate witness"
        f"{'' if len(degenerate_rows) == 1 else 's'})"
    )
    print(
        "  max relative error over nondegenerate witnesses = "
        f"{max_nondegenerate_relative_error:.4%}"
    )
    return state_df, witness_df, cv_cert


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
    freeze = None
    freeze_error = None
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
    ) as exc:
        freeze_error = repr(exc)
    if freeze is not None:
        (OUTDIR / "pip_freeze.txt").write_text(freeze, encoding="utf-8")

    provenance = {
        "source_label": source_label,
        "source_kind": source_kind,
        "source_sha256": source_sha256,
        "source_hash_available": source_sha256 is not None,
        "pip_freeze_captured": freeze is not None,
        "pip_freeze_error": freeze_error,
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": sys.version,
        "assertions_enabled": __debug__,
        "duration_convention": duration_convention_report([2, 5, 8]),
        "versions": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scipy": package_version("scipy"),
            "pulser": package_version("pulser"),
            "pulser_simulation": package_version("pulser-simulation"),
            "qutip": package_version("qutip"),
        },
    }
    with open(OUTDIR / "provenance.json", "w") as handle:
        json.dump(json_sanitize(provenance), handle, indent=2, allow_nan=False)
    return provenance


# ----------------------------------------------------------------------
# Frozen paper-only main
# ----------------------------------------------------------------------
def main():
    print(
        "\n" + "=" * 88
        + "\nPAPER117 NEUTRAL-ATOM PULSE-ORDER VALIDATION (v3)\n"
        + "=" * 88
    )
    print("output:", OUTDIR)
    print(
        "scope: fixed-T multi-proxy scan + f-reflection decomposition + "
        "algebraic signed-gap schedule check + six equal-duration "
        "permutations + state- and witness-level Pulser cross-validation"
    )
    started = time.perf_counter()

    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "SciPy is required. Install scipy, numpy, pandas, pulser, "
            "and pulser-simulation."
        )

    provenance = write_provenance()
    _, _, cv_cert = run_cross_validation()
    (
        scan_df,
        _,
        _,
        even_odd_df,
        symmetry_df,
        scan_cert,
    ) = run_117_scan()
    permutation_df, three_segment = run_3segment_magnus_diagnostic(
        nominal_total_ns=scan_cert["total_duration_ns"],
        numerical_floor_D=scan_cert["effective_D_floor"],
        numerical_floor_TVD=scan_cert["zero_gap_numerical_floor"]["max_TVD"],
    )

    segmentation_expm = three_segment["segmentation_artifact_expm"]
    equal_a2 = three_segment["equal_A2_differences"]
    expected_points = len(GAP_GRID) * len(FRACTION_GRID)
    expected_nonzero = int((GAP_GRID > 0).sum()) * len(FRACTION_GRID)
    elapsed = time.perf_counter() - started

    # Numerical validity only. These may fail-fast; they encode no physics
    # conclusion.
    validation_gates = {
        "scan_point_count": len(scan_df) == expected_points,
        "scan_nonzero_gap_count": (
            int((scan_df["gap"] > 0).sum()) == expected_nonzero
        ),
        "grid_reflection_closed": bool(
            scan_cert["grid"]["closure"]["reflection_closed"]
        ),
        "grid_half_split_present": bool(
            scan_cert["grid"]["closure"]["half_split_present"]
        ),
        "f_reflection_row_count": (
            scan_cert["f_reflection"]["rows"]
            == scan_cert["f_reflection"]["expected_rows"]
        ),
        "common_N8_total_duration": (
            scan_cert["total_duration_ns"]
            == three_segment["matched_total_duration_ns"]
            == PAPER_TOTAL_NS
        ),
        "signed_gap_schedule_identity_pass": (
            scan_cert["signed_gap_schedule_identity"][
                "max_schedule_parameter_error"
            ]
            <= SCHEDULE_MATCH_TOL
        ),
        "differential_support_identity_pass": bool(
            scan_cert["differential_support_identity"]["passes"]
        ),
        "D_estimators_agree": (
            scan_cert["D_estimator_max_relative_disagreement"]
            <= D_ESTIMATOR_MAX_REL_DISAGREEMENT
        ),
        "weakest_signal_above_floor": (
            scan_cert["weakest_D_to_floor_ratio"] >= MIN_SIGNAL_TO_FLOOR
        ),
        "pulser_state_cross_validation": (
            cv_cert["state_level"]["max_infidelity"]
            <= CROSS_VALIDATION_MAX_INFIDELITY
        ),
        "pulser_witness_cross_validation": bool(
            cv_cert["witness_level"]["all_points_pass"]
        ),
        "six_permutations_present": len(permutation_df) == 6,
        "expm_segmentation_floor_below_signal": bool(
            three_segment["segmentation_expm_state_floor"]
            < float(min(permutation_df["pure_trace_distance"])) / 1.0e5
            and segmentation_expm["TVD_distribution"]
            < min(
                equal_a2["TVD_perm2_minus_perm3_abs"],
                equal_a2["TVD_perm4_minus_perm5_abs"],
            )
            / 1.0e5
        ),
    }

    # Physics findings. Recorded, never fail-fast: a negative result is a
    # result, not a broken run.
    scientific_tests = {
        "equal_A2_TVD_insufficiency": {
            "status": (
                "SUPPORTED"
                if three_segment["TVD_resolved_above_threshold"]
                else "NOT_RESOLVED"
            ),
            "max_observed_difference": three_segment[
                "max_equal_A2_TVD_difference"
            ],
            "decision_threshold": three_segment["TVD_decision_threshold"],
            "fail_fast": False,
        },
        "equal_A2_D_insufficiency": {
            "status": (
                "SUPPORTED"
                if three_segment["D_resolved_above_threshold"]
                else "NOT_RESOLVED"
            ),
            "max_observed_difference": three_segment[
                "max_equal_A2_D_difference"
            ],
            "decision_threshold": three_segment["D_decision_threshold"],
            "fail_fast": False,
        },
        "proxy_selection_within_specified_set": {
            "status": "RECORDED",
            "specified_proxy_set": scan_cert["proxy_comparison"][
                "specified_proxy_set"
            ],
            "primary_per_gap_preferred": scan_cert["proxy_comparison"][
                "per_gap"
            ]["preferred_within_specified_set"],
            "per_gap_unanimous": scan_cert["proxy_comparison"]["per_gap"][
                "unanimous"
            ],
            "pooled_preferred": scan_cert["proxy_comparison"][
                "pooled_sensitivity"
            ]["preferred_within_specified_set"],
            "rankings_agree": scan_cert["proxy_comparison"][
                "per_gap_and_pooled_rankings_agree"
            ],
            "ranking_stability_warning": scan_cert["proxy_comparison"][
                "ranking_stability_warning"
            ],
            "residual_structure_diagnostic": scan_cert["proxy_comparison"][
                "residual_structure_diagnostic"
            ],
            "boundary": scan_cert["proxy_comparison"]["boundary"],
            "fail_fast": False,
        },
        "second_order_magnus_domain_of_validity": {
            "status": "RECORDED",
            "detail": scan_cert["perturbative_validity"],
            "fail_fast": False,
        },
    }

    implementation_checks = {
        "pulser_constant_segmentation": three_segment[
            "pulser_segmentation_check"
        ],
        "metrics": three_segment["segmentation_artifact_pulser"],
        "propagator_cache": propagator_cache_report(),
    }

    status = "VALID" if all(validation_gates.values()) else "INVALID"
    summary = {
        "status": status,
        "paper_scope": (
            "Fixed-T multi-proxy comparison over a declared proxy set and a "
            "six-permutation equal-A2 diagnostic. No N=2 shared-generator or "
            "variable-partition claim is included."
        ),
        "primary_backend": "scipy_expm",
        "total_duration_ns": scan_cert["total_duration_ns"],
        "scan_points": int(len(scan_df)),
        "nonzero_gap_points": int((scan_df["gap"] > 0).sum()),
        "even_odd_rows": int(len(even_odd_df)),
        "signed_gap_identity_rows": int(len(symmetry_df)),
        "permutation_rows": int(len(permutation_df)),
        "proxy_comparison": scan_cert["proxy_comparison"],
        "perturbative_validity": scan_cert["perturbative_validity"],
        "zero_gap_numerical_floor": scan_cert["zero_gap_numerical_floor"],
        "effective_D_floor": scan_cert["effective_D_floor"],
        "legacy_effective_D_floor": scan_cert["legacy_effective_D_floor"],
        "floor_improvement_factor": scan_cert["floor_improvement_factor"],
        "weakest_D_to_floor_ratio": scan_cert["weakest_D_to_floor_ratio"],
        "cross_validation_state_max_infidelity": cv_cert["state_level"][
            "max_infidelity"
        ],
        "cross_validation_witness_max_absolute_error": cv_cert[
            "witness_level"
        ]["max_absolute_error"],
        "cross_validation_witness_max_relative_error_all_rows": cv_cert[
            "witness_level"
        ]["max_relative_error_all_rows"],
        "cross_validation_witness_max_nondegenerate_relative_error": cv_cert[
            "witness_level"
        ]["max_nondegenerate_relative_error"],
        "witness_magnitude_for_one_percent_backend_agreement": cv_cert[
            "witness_level"
        ]["witness_magnitude_for_one_percent_agreement"],
        "equal_A2_differences": equal_a2,
        "validation_gates": validation_gates,
        "scientific_tests": scientific_tests,
        "implementation_checks": implementation_checks,
        "source_sha256": provenance["source_sha256"],
        "assertions_enabled": __debug__,
        "elapsed_sec": elapsed,
        "output_directory": str(OUTDIR),
        "claim_boundary": (
            "The scan compares schedule-shape proxies at one fixed total "
            "duration. It does not determine T versus T^2 scaling, validate "
            "QPU hardware, establish a universal path-area law, or test a "
            "Hamiltonian-learning model."
        ),
    }
    with open(OUTDIR / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(json_sanitize(summary), handle, indent=2, allow_nan=False)

    print("\n" + "=" * 88)
    print("GLOBAL VERDICT")
    print("=" * 88)
    print(json.dumps(json_sanitize(summary), indent=2, allow_nan=False))
    if status != "VALID":
        failed = [k for k, v in validation_gates.items() if not v]
        raise AssertionError(f"Paper117 validation gates failed: {failed}")
    print(f"elapsed={elapsed:.2f}s")
    print(f"outputs={OUTDIR}")


if __name__ == "__main__":
    main()
