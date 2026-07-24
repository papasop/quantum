#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASQAL Cloud v1.1 — Random-Phase Area-Matched Manifold Test

Prerequisite
------------
You already reached:

    PASQAL_CLOUD_RESOURCE_MATCHED_PHASE_GEOMETRY_GAIN20_EMU = SUCCESS

Why v1.1?
---------
v1.0 compared:
    geoflow_bb1 vs flat_area_matched

A reviewer may say:
    "flat_area_matched is a weak / straw-man control."

v1.1 adds a random-phase area-matched ensemble:
    - same pulse areas as BB1/geoflow
    - same total duration
    - same stress scales
    - random phases, with first phase fixed to 0 as gauge

This tests whether the BB1/geoflow path is merely better than a bad flat
control, or whether it sits in a high-robustness region of the phase-path
manifold.

Protocols
---------
1. flat_area_matched
   areas: [pi, pi, 2pi, pi] * scale
   phases: [0, 0, 0, 0]

2. geoflow_bb1
   areas: [pi, pi, 2pi, pi] * scale
   phases: [0, phi, 3phi, phi], phi=acos(-1/4)

3. random_phase_k
   areas: [pi, pi, 2pi, pi] * scale
   phases: [0, r1, r2, r3], r_i uniform in [0, 2pi)
   The same random phase vector is reused across all stress scales.

Default stress scales
---------------------
    0.6, 0.8, 1.2, 1.4
Scale 1.0 is omitted by default because it is usually a trivial non-stress point.

Paper config (IMPORTANT -- reproduces the reported v2 headline)
---------------------------------------------------------------
The DEFAULT scales above are NOT the scales used for the paper's "v2 full
measured" run. The paper reports the stress window {0.6, 0.8, 1.0, 1.2} at 500
trajectories. Running this driver with defaults gives a DIFFERENT (and more
conservative) random-comparator distribution. To reproduce the paper's v2
numbers (flat 0.5000, bb1 0.9930, random median 0.5158, random best 0.6725):

    python k1_pasqal_cloud_random_phase_manifold_v11.py \
        --scales 0.6 0.8 1.0 1.2 --runs 500 --seed 11

then cross-check the resulting raw_results.csv with:

    python pasqal_v2_reproduce_summary.py --raw raw_results.csv

flat and bb1 are stress-window invariant (flat=0.5000, bb1~=0.99 on both grids);
the random-phase median/best are stress-window sensitive.

Default cost
------------
    (2 + n_random) * n_scales jobs
Default n_random=6 and n_scales=4 => 32 cloud jobs.

Run in Colab
------------
!python pasqal_cloud_random_phase_manifold_v11.py \
  --prompt \
  --sequence-device FRESNEL_CAN1 \
  --device-type EMU_FREE \
  --scales 0.6 0.8 1.2 1.4 \
  --n-random 6 \
  --runs 100 \
  --seed 11

Lower quota
-----------
!python pasqal_cloud_random_phase_manifold_v11.py \
  --prompt \
  --sequence-device FRESNEL_CAN1 \
  --device-type EMU_FREE \
  --scales 0.8 1.2 \
  --n-random 4 \
  --runs 50 \
  --seed 11

Interpretation
--------------
If success:
    PASQAL_CLOUD_RANDOM_PHASE_MANIFOLD_EMU = SUCCESS

Safe claim:
    On PASQAL Cloud EMU, the BB1-type landmark is not only better than a
    flat phase control; it lies in a high-robustness region relative to an
    area-and-duration-matched random-phase ensemble, and/or random search can
    discover comparable high-robustness paths.

Boundary:
---------
This is PASQAL Cloud EMU, not physical FRESNEL QPU.
This is a phase-manifold ablation / search-control experiment, not a claim
that BB1 itself is newly invented.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import getpass
import importlib
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict

import numpy as np

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning"
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.filterwarnings("ignore", message=r".*datetime\.datetime\.utcnow.*")


# =============================================================================
# Utilities
# =============================================================================

@contextlib.contextmanager
def quiet_stderr(enabled: bool = True):
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        yield


def pip_install() -> None:
    print("[INSTALL] Installing pulser / pulser-pasqal / pasqal-cloud ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "pulser", "pulser-pasqal", "pasqal-cloud",
    ])
    importlib.invalidate_caches()
    print("[INSTALL] done")


def import_or_install(name: str, auto_install: bool):
    try:
        with quiet_stderr(True):
            return importlib.import_module(name)
    except ModuleNotFoundError:
        if not auto_install:
            raise
        pip_install()
        with quiet_stderr(True):
            return importlib.import_module(name)


def redact(x: Any) -> Any:
    if not isinstance(x, str):
        return x
    if "@" in x and "." in x:
        a, b = x.split("@", 1)
        return a[:2] + "***@" + b
    if len(x) >= 16:
        return x[:8] + "***"
    return x


def safe_repr(x: Any, maxlen: int = 6000) -> str:
    try:
        s = repr(x)
    except BaseException as e:
        s = f"<repr failed: {type(e).__name__}: {e}>"
    if len(s) > maxlen:
        return s[:maxlen] + f"... <truncated {len(s)-maxlen} chars>"
    return s


def json_sanitize(x: Any, depth: int = 0):
    if depth > 6:
        return safe_repr(x, 1200)
    if x is None or isinstance(x, (bool, int, float, str)):
        return redact(x)
    if isinstance(x, dict):
        return {str(k): json_sanitize(v, depth + 1) for k, v in list(x.items())[:300]}
    if isinstance(x, (list, tuple, set)):
        return [json_sanitize(v, depth + 1) for v in list(x)[:300]]

    for method in ("model_dump", "dict", "to_dict", "asdict"):
        if hasattr(x, method):
            try:
                return json_sanitize(getattr(x, method)(), depth + 1)
            except BaseException:
                pass

    small = {}
    for attr in (
        "batch_id", "_batch_id", "job_ids", "_job_ids", "job_id", "id",
        "status", "results", "_results", "jobs", "available", "device_type",
        "backend", "data", "name", "type", "errors", "counter", "sampling_dist"
    ):
        if hasattr(x, attr):
            try:
                small[attr] = json_sanitize(getattr(x, attr), depth + 1)
            except BaseException as e:
                small[attr] = f"<error {type(e).__name__}: {str(e)[:300]}>"
    if small:
        return small

    return safe_repr(x, 2000)


def prompt_credentials(force_prompt: bool) -> Tuple[str, str, Optional[str]]:
    email = os.environ.get("PASQAL_EMAIL", "").strip()
    password = os.environ.get("PASQAL_PASSWORD", "")
    project_id = os.environ.get("PASQAL_PROJECT_ID", "").strip() or None

    if force_prompt or not email:
        email = input("PASQAL email: ").strip()
    if force_prompt or not password:
        password = getpass.getpass("PASQAL password: ")
    if not project_id:
        raw = input("PASQAL project_id / workspace_id (optional, press Enter to skip): ").strip()
        project_id = raw or None

    if not email:
        raise RuntimeError("Empty PASQAL email.")
    if not password:
        raise RuntimeError("Empty PASQAL password.")
    return email, password, project_id


# =============================================================================
# PASQAL / Pulser
# =============================================================================

def connect_cloud(email: str, password: str, project_id: Optional[str], auto_install: bool):
    pulser_pasqal = import_or_install("pulser_pasqal", auto_install=auto_install)
    PasqalCloud = getattr(pulser_pasqal, "PasqalCloud")

    attempts = []
    if project_id:
        attempts.append(("username_password_project_id", {
            "username": email, "password": password, "project_id": project_id,
        }))
    attempts.append(("username_password", {"username": email, "password": password}))

    errors = []
    for label, kwargs in attempts:
        try:
            with quiet_stderr(True):
                cloud = PasqalCloud(**kwargs)
            return pulser_pasqal, cloud, label
        except BaseException as e:
            errors.append(f"{label}: {type(e).__name__}: {str(e)[:500]}")
    raise RuntimeError("Could not connect:\n" + "\n".join(errors))


def pulser_classes():
    pulser = importlib.import_module("pulser")
    Register = getattr(pulser, "Register", None) or getattr(importlib.import_module("pulser.register"), "Register")
    Sequence = getattr(pulser, "Sequence", None) or getattr(importlib.import_module("pulser.sequence"), "Sequence")
    Pulse = getattr(pulser, "Pulse", None) or getattr(importlib.import_module("pulser.pulse"), "Pulse")
    ConstantWaveform = getattr(importlib.import_module("pulser.waveforms"), "ConstantWaveform")
    return Register, Sequence, Pulse, ConstantWaveform


def coerce_device_type(value: Optional[str]):
    if not value:
        return None, None
    try:
        pasqal_cloud = importlib.import_module("pasqal_cloud")
        DT = getattr(pasqal_cloud, "DeviceTypeName", None)
        if DT is not None and hasattr(DT, "__members__"):
            for k, v in DT.__members__.items():
                vv = getattr(v, "value", str(v))
                if value == k or value == vv or value.lower() == k.lower() or value.lower() == str(vv).lower():
                    return v, f"DeviceTypeName.{k}"
    except BaseException:
        pass
    return value, "string_fallback"


def quantize_duration_ns(x: float, step: int, min_duration: int, max_duration: int) -> int:
    q = int(round(float(x) / step) * step)
    q = max(int(min_duration), q)
    q = min(int(max_duration), q)
    return int(q)


# =============================================================================
# Phase manifold
# =============================================================================

def bb1_phi_for_pi() -> float:
    return math.acos(-0.25)


def generate_phase_paths(n_random: int, seed: int) -> Dict[str, List[float]]:
    rng = random.Random(int(seed))
    phi = bb1_phi_for_pi()

    paths: Dict[str, List[float]] = {
        "flat_area_matched": [0.0, 0.0, 0.0, 0.0],
        "geoflow_bb1": [0.0, phi, (3.0 * phi) % (2.0 * math.pi), phi],
    }

    for i in range(int(n_random)):
        paths[f"random_phase_{i:03d}"] = [
            0.0,
            rng.random() * 2.0 * math.pi,
            rng.random() * 2.0 * math.pi,
            rng.random() * 2.0 * math.pi,
        ]

    return paths


def pulse_path_from_phases(phases: List[float], scale: float) -> List[Tuple[float, float]]:
    s = float(scale)
    if len(phases) != 4:
        raise ValueError("Need exactly four phases.")
    return [
        (s * math.pi, float(phases[0])),
        (s * math.pi, float(phases[1])),
        (s * 2.0 * math.pi, float(phases[2])),
        (s * math.pi, float(phases[3])),
    ]


def expected_p_excited_unitary(pulses: List[Tuple[float, float]]) -> float:
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    eye = np.eye(2, dtype=complex)

    def R(theta: float, phase: float):
        n = math.cos(phase) * sx + math.sin(phase) * sy
        return math.cos(theta / 2.0) * eye - 1j * math.sin(theta / 2.0) * n

    U = eye.copy()
    for area, phase in pulses:
        U = R(float(area), float(phase)) @ U

    psi0 = np.array([1.0, 0.0], dtype=complex)
    psi = U @ psi0
    return float(abs(psi[1]) ** 2)


def build_phase_path_sequence(
    device: Any,
    path_name: str,
    phases: List[float],
    scale: float,
    target_duration_ns: int,
    duration_step_ns: int,
    min_duration_ns: int,
    max_total_duration_ns: int,
):
    Register, Sequence, Pulse, ConstantWaveform = pulser_classes()

    pulses = pulse_path_from_phases(phases, scale)
    total_area_abs = sum(abs(a) for a, _ in pulses)
    amp = total_area_abs / (float(target_duration_ns) / 1000.0)

    reg = Register({"q0": (0.0, 0.0)})
    seq = Sequence(reg, device)
    seq.declare_channel("ch0", "rydberg_global")

    built_pulses = []
    total_duration = 0

    for area, phase in pulses:
        area_abs = abs(float(area))
        duration_ns = quantize_duration_ns(
            area_abs / amp * 1000.0,
            step=duration_step_ns,
            min_duration=min_duration_ns,
            max_duration=max_total_duration_ns,
        )
        local_amp = amp

        if total_duration + duration_ns > max_total_duration_ns:
            raise RuntimeError(
                f"Sequence duration would exceed {max_total_duration_ns} ns: "
                f"{total_duration}+{duration_ns}. Reduce target_duration_ns or scales."
            )

        local_phase = float(phase) if area >= 0 else (float(phase) + math.pi) % (2 * math.pi)

        if hasattr(Pulse, "ConstantPulse"):
            pulse = Pulse.ConstantPulse(int(duration_ns), float(local_amp), 0.0, float(local_phase))
        else:
            amp_wf = ConstantWaveform(int(duration_ns), float(local_amp))
            det_wf = ConstantWaveform(int(duration_ns), 0.0)
            pulse = Pulse(amp_wf, det_wf, float(local_phase))

        seq.add(pulse, "ch0")
        total_duration += duration_ns

        built_pulses.append({
            "area_rad": float(area),
            "phase_rad": float(local_phase),
            "duration_ns": int(duration_ns),
            "amp_rad_per_us": float(local_amp),
            "approx_area_rad": float(local_amp) * int(duration_ns) / 1000.0 * (1.0 if area >= 0 else -1.0),
        })

    try:
        seq.measure("ground-rydberg")
    except BaseException:
        try:
            seq.measure()
        except BaseException:
            pass

    expected = expected_p_excited_unitary(pulses)

    meta = {
        "path_name": path_name,
        "path_class": (
            "flat_control" if path_name == "flat_area_matched"
            else "geoflow_landmark" if path_name == "geoflow_bb1"
            else "random_phase"
        ),
        "scale": float(scale),
        "phases_rad": [float(x) for x in phases],
        "n_pulses": len(pulses),
        "target_duration_ns": int(target_duration_ns),
        "total_duration_ns": int(total_duration),
        "total_area_abs_rad": float(total_area_abs),
        "amp_rad_per_us_global": float(amp),
        "pulses": built_pulses,
        "expected_p_excited": expected,
    }
    return seq, meta


# =============================================================================
# Result parser
# =============================================================================

def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def looks_like_count_dict(d: Dict[Any, Any]) -> bool:
    if not d:
        return False
    if not all(isinstance(k, str) for k in d.keys()):
        return False
    if not all(is_number(v) for v in d.values()):
        return False
    keys = list(d.keys())
    bit_like = all(re.fullmatch(r"[01]+", k) is not None for k in keys)
    basis_like = all(k.lower() in {
        "0", "1", "g", "r", "ground", "rydberg",
        "|0>", "|1>", "|g>", "|r>", "0_state", "1_state"
    } for k in keys)
    return bit_like or basis_like


def recursive_find_count_dicts(x: Any, path: str = "$") -> List[Tuple[str, Dict[str, float]]]:
    found: List[Tuple[str, Dict[str, float]]] = []
    if isinstance(x, dict):
        if looks_like_count_dict(x):
            found.append((path, {str(k): float(v) for k, v in x.items()}))
        for k, v in x.items():
            found.extend(recursive_find_count_dicts(v, f"{path}.{k}"))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            found.extend(recursive_find_count_dicts(v, f"{path}[{i}]"))
    elif isinstance(x, str):
        for m in re.finditer(r"\{[^{}]{1,500}\}", x):
            snippet = m.group(0)
            if ("'0'" in snippet or '"0"' in snippet or "'1'" in snippet or '"1"' in snippet):
                try:
                    import ast
                    d = ast.literal_eval(snippet)
                    if isinstance(d, dict) and looks_like_count_dict(d):
                        found.append((path + ".repr_dict", {str(k): float(v) for k, v in d.items()}))
                except Exception:
                    pass
    return found


def pick_best_counts(cands: List[Tuple[str, Dict[str, float]]]) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    if not cands:
        return None, None

    def score(item: Tuple[str, Dict[str, float]]) -> Tuple[float, int]:
        path, d = item
        total = sum(d.values())
        nkeys = len(d)
        s = 0.0
        low = path.lower()
        for word in ["count", "counter", "sampling", "result", "dist", "prob"]:
            if word in low:
                s += 10
        if total > 0:
            s += math.log1p(total)
        s += nkeys
        return s, nkeys

    return max(cands, key=score)


def classify_excited_probability(counts: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        return {
            "total": 0.0,
            "p_ground_like": float("nan"),
            "p_excited_like": float("nan"),
            "p_other": float("nan"),
        }

    ground = 0.0
    excited = 0.0
    other = 0.0

    for k, v in counts.items():
        kl = str(k).lower().strip()

        if kl in {"0", "g", "ground", "|0>", "|g>", "0_state"}:
            ground += v
        elif kl in {"1", "r", "rydberg", "|1>", "|r>", "1_state"}:
            excited += v
        elif re.fullmatch(r"[01]+", kl):
            if "1" in kl:
                excited += v
            else:
                ground += v
        else:
            other += v

    return {
        "total": total,
        "p_ground_like": ground / total,
        "p_excited_like": excited / total,
        "p_other": other / total,
    }


def parse_counts_from_result(got: Any) -> Dict[str, Any]:
    root = json_sanitize(got)
    cands = recursive_find_count_dicts(root)
    path, counts = pick_best_counts(cands)

    if counts is None:
        return {
            "counts_found": False,
            "counts_path": "",
            "counts_json": "",
            "total": float("nan"),
            "p_ground_like": float("nan"),
            "p_excited_like": float("nan"),
            "p_other": float("nan"),
            "sanitized_result": root,
        }

    probs = classify_excited_probability(counts)
    return {
        "counts_found": True,
        "counts_path": path,
        "counts_json": json.dumps(counts, sort_keys=True),
        **probs,
        "sanitized_result": root,
    }


# =============================================================================
# Submit
# =============================================================================

def raw_attr(obj: Any, names: List[str]):
    for name in names:
        if hasattr(obj, name):
            try:
                val = getattr(obj, name)
                if isinstance(val, str):
                    return val
            except BaseException:
                pass
    return None


def extract_raw_batch_id(remote: Any) -> Optional[str]:
    direct = raw_attr(remote, ["batch_id", "_batch_id", "id"])
    if direct:
        return direct

    for method in ("model_dump", "dict", "to_dict", "asdict"):
        if hasattr(remote, method):
            try:
                d = getattr(remote, method)()
                if isinstance(d, dict):
                    for k in ("batch_id", "_batch_id", "id"):
                        v = d.get(k)
                        if isinstance(v, str):
                            return v
            except BaseException:
                pass
    return None


def classify_error(e: BaseException) -> str:
    msg = str(e)
    if "sequence's duration exceeded" in msg or "duration can be at most" in msg:
        return "SEQUENCE_OR_PULSE_DURATION_TOO_LONG"
    if "amplitude" in msg.lower() or "Rabi" in msg:
        return "AMPLITUDE_CONSTRAINT"
    if "CB1109" in msg or "closed batch should have at least one job" in msg:
        return "NO_JOB_PARAMS_CB1109"
    if "CB1107" in msg or "cannot create a batch with this Device type" in msg:
        return "PROJECT_DEVICE_TYPE_NOT_ALLOWED_CB1107"
    if "403" in msg or "Forbidden" in msg:
        return "FORBIDDEN_PERMISSION"
    if "422" in msg and "valid uuid" in msg:
        return "BAD_BATCH_ID_USED"
    if "400" in msg or "Bad request" in msg:
        return "BAD_REQUEST"
    if "validation" in msg.lower():
        return "VALIDATION_ERROR"
    return type(e).__name__


def submit_one(
    cloud: Any,
    seq: Any,
    dev_type: Any,
    runs: int,
    wait: bool,
    open_batch: bool,
):
    kwargs = {
        "wait": bool(wait),
        "open": bool(open_batch),
        "job_params": [{"runs": int(runs)}],
    }
    if dev_type is not None:
        kwargs["device_type"] = dev_type

    with quiet_stderr(True):
        remote = cloud.submit(seq, **kwargs)

    raw_batch_id = extract_raw_batch_id(remote)
    parsed = {
        "counts_found": False,
        "counts_path": "",
        "counts_json": "",
        "total": float("nan"),
        "p_ground_like": float("nan"),
        "p_excited_like": float("nan"),
        "p_other": float("nan"),
    }
    get_results_ok = None
    get_results_error = None

    if raw_batch_id:
        try:
            with quiet_stderr(True):
                got = cloud.get_results(raw_batch_id)
            get_results_ok = True
            parsed = parse_counts_from_result(got)
        except BaseException as e:
            get_results_ok = False
            get_results_error = f"{type(e).__name__}: {str(e)[:2000]}"

    return {
        "submit_ok": True,
        "remote_type": str(type(remote)),
        "remote_info": json_sanitize(remote),
        "batch_id_redacted": redact(raw_batch_id) if raw_batch_id else None,
        "get_results_ok": get_results_ok,
        "get_results_error": get_results_error,
        **parsed,
    }


# =============================================================================
# Analysis
# =============================================================================

def aggregate_by_path(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    usable = [
        r for r in rows
        if r.get("counts_found") and r.get("p_excited_like") == r.get("p_excited_like")
    ]

    by_path: Dict[str, List[Dict[str, Any]]] = {}
    for r in usable:
        by_path.setdefault(str(r["path_name"]), []).append(r)

    agg: Dict[str, Dict[str, Any]] = {}
    for name, rs in by_path.items():
        rs_sorted = sorted(rs, key=lambda x: float(x["scale"]))
        measured = [float(r["p_excited_like"]) for r in rs_sorted]
        expected = [float(r["expected_p_excited"]) for r in rs_sorted]
        agg[name] = {
            "path_name": name,
            "path_class": rs_sorted[0].get("path_class"),
            "phases_rad": rs_sorted[0].get("phases_rad"),
            "n_scales": len(rs_sorted),
            "scales": [float(r["scale"]) for r in rs_sorted],
            "mean_measured": float(sum(measured) / len(measured)),
            "min_measured": float(min(measured)),
            "max_measured": float(max(measured)),
            "mean_expected": float(sum(expected) / len(expected)),
            "values_measured": measured,
            "values_expected": expected,
        }
    return agg


def percentile_rank(value: float, sample: List[float]) -> float:
    if not sample:
        return float("nan")
    return sum(1 for x in sample if x <= value) / len(sample)


def analyze_manifold(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg = aggregate_by_path(rows)
    flat = agg.get("flat_area_matched")
    bb1 = agg.get("geoflow_bb1")
    random_paths = [v for k, v in agg.items() if k.startswith("random_phase_")]

    if flat is None or bb1 is None or not random_paths:
        return {
            "pass_random_phase_manifold": False,
            "reason": "missing flat, bb1, or random paths",
            "path_aggregates": agg,
        }

    random_means = sorted(float(p["mean_measured"]) for p in random_paths)
    random_median = float(np.median(random_means))
    random_mean = float(sum(random_means) / len(random_means))
    random_best = float(max(random_means))
    random_worst = float(min(random_means))
    random_p75 = float(np.percentile(random_means, 75))
    random_p90 = float(np.percentile(random_means, 90))

    best_random = max(random_paths, key=lambda p: float(p["mean_measured"]))
    bb1_mean = float(bb1["mean_measured"])
    flat_mean = float(flat["mean_measured"])

    bb1_vs_random_median_abs = bb1_mean - random_median
    bb1_vs_random_median_rel = bb1_vs_random_median_abs / max(random_median, 1e-12)
    bb1_vs_flat_abs = bb1_mean - flat_mean
    bb1_vs_flat_rel = bb1_vs_flat_abs / max(flat_mean, 1e-12)
    best_random_vs_flat_abs = random_best - flat_mean
    best_random_vs_flat_rel = best_random_vs_flat_abs / max(flat_mean, 1e-12)

    bb1_rank = percentile_rank(bb1_mean, random_means)

    pass_bb1_beats_random_median = bool(
        bb1_vs_random_median_abs >= 0.10 and bb1_vs_random_median_rel >= 0.20
    )
    pass_bb1_high_percentile = bool(bb1_rank >= 0.75)
    pass_random_search_discovers_high = bool(
        random_best >= 0.90 and best_random_vs_flat_abs >= 0.20
    )
    pass_bb1_close_to_best = bool(bb1_mean >= 0.95 * max(random_best, 1e-12))

    # Main success criterion: either BB1 is high vs random median, or random search
    # reveals a comparable high-robustness region. This prevents overclaiming uniqueness.
    pass_random_phase_manifold = bool(
        (pass_bb1_beats_random_median and pass_bb1_high_percentile)
        or (pass_random_search_discovers_high and pass_bb1_close_to_best)
    )

    sorted_random = sorted(random_paths, key=lambda p: float(p["mean_measured"]), reverse=True)

    return {
        "pass_random_phase_manifold": pass_random_phase_manifold,
        "n_random_paths": len(random_paths),
        "flat_mean": flat_mean,
        "bb1_mean": bb1_mean,
        "random_mean": random_mean,
        "random_median": random_median,
        "random_best": random_best,
        "random_worst": random_worst,
        "random_p75": random_p75,
        "random_p90": random_p90,
        "bb1_percentile_rank_vs_random": bb1_rank,
        "bb1_vs_random_median_abs": bb1_vs_random_median_abs,
        "bb1_vs_random_median_rel": bb1_vs_random_median_rel,
        "bb1_vs_flat_abs": bb1_vs_flat_abs,
        "bb1_vs_flat_rel": bb1_vs_flat_rel,
        "best_random_name": best_random["path_name"],
        "best_random_mean": random_best,
        "best_random_phases_rad": best_random.get("phases_rad"),
        "best_random_vs_flat_abs": best_random_vs_flat_abs,
        "best_random_vs_flat_rel": best_random_vs_flat_rel,
        "bb1_close_to_best": pass_bb1_close_to_best,
        "pass_bb1_beats_random_median": pass_bb1_beats_random_median,
        "pass_bb1_high_percentile": pass_bb1_high_percentile,
        "pass_random_search_discovers_high": pass_random_search_discovers_high,
        "interpretation": (
            "If pass_bb1_beats_random_median is true, BB1/geoflow is not merely "
            "better than flat; it is high relative to random area-matched phase paths. "
            "If pass_random_search_discovers_high is true, random phase search already "
            "finds high-robustness regions, supporting discoverability on the manifold."
        ),
        "top_random_paths": sorted_random[:5],
        "path_aggregates": agg,
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    base = [
        "path_name", "path_class", "scale", "p_excited_like", "expected_p_excited",
        "abs_error_vs_expected", "p_ground_like", "p_other", "total",
        "counts_found", "counts_path", "submit_ok", "get_results_ok",
        "batch_id_redacted", "n_pulses", "total_duration_ns",
        "total_area_abs_rad", "amp_rad_per_us_global",
        "phases_rad", "counts_json",
    ]
    extra = [
        k for r in rows for k in r.keys()
        if k not in base and k not in {"sanitized_result", "remote_info", "pulses"}
    ]
    keys = base + sorted(set(extra))

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            rr = {k: r.get(k) for k in keys}
            w.writerow(rr)


def write_aggregate_csv(analysis: Dict[str, Any], path: Path) -> None:
    agg = analysis.get("path_aggregates", {})
    keys = [
        "path_name", "path_class", "mean_measured", "min_measured", "max_measured",
        "mean_expected", "n_scales", "scales", "phases_rad",
        "values_measured", "values_expected",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for name, a in sorted(agg.items()):
            w.writerow({k: a.get(k) for k in keys})


def make_plot(analysis: Dict[str, Any], outdir: Path) -> List[Path]:
    paths: List[Path] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return paths

    agg = analysis.get("path_aggregates", {})
    if not agg:
        return paths

    # Plot 1: path means
    names = []
    vals = []
    classes = []
    for name, a in sorted(agg.items(), key=lambda kv: float(kv[1].get("mean_measured", -1)), reverse=True):
        names.append(name)
        vals.append(float(a.get("mean_measured", float("nan"))))
        classes.append(a.get("path_class"))

    plt.figure(figsize=(max(8, len(names) * 0.45), 5))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=75, ha="right")
    plt.ylabel("Mean P(excited-like) over stress scales")
    plt.title("PASQAL Cloud EMU v1.1 random-phase manifold")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p1 = outdir / "random_phase_path_means_v11.png"
    plt.savefig(p1, dpi=170)
    plt.close()
    paths.append(p1)

    # Plot 2: scale curves for flat, bb1, best random
    plt.figure()
    selected = ["flat_area_matched", "geoflow_bb1"]
    best_name = analysis.get("best_random_name")
    if best_name:
        selected.append(best_name)

    for name in selected:
        a = agg.get(name)
        if not a:
            continue
        plt.plot(a["scales"], a["values_measured"], marker="o", label=name)

    plt.xlabel("pulse-area scale stress s")
    plt.ylabel("P(excited-like / Rydberg-like)")
    plt.title("Selected phase paths across stress scales")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p2 = outdir / "random_phase_selected_curves_v11.png"
    plt.savefig(p2, dpi=170)
    plt.close()
    paths.append(p2)

    return paths


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="pasqal_cloud_random_phase_manifold_v11")
    p.add_argument("--prompt", action="store_true")
    p.add_argument("--auto-install", action="store_true", default=True)

    p.add_argument("--sequence-device", default="FRESNEL_CAN1")
    p.add_argument("--device-type", default="EMU_FREE")
    p.add_argument("--scales", nargs="+", type=float, default=[0.6, 0.8, 1.2, 1.4])
    p.add_argument("--n-random", type=int, default=6)
    p.add_argument("--seed", type=int, default=11)

    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--target-duration-ns", type=int, default=5000)
    p.add_argument("--duration-step-ns", type=int, default=4)
    p.add_argument("--min-duration-ns", type=int, default=100)
    p.add_argument("--max-total-duration-ns", type=int, default=6000)

    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--open-batch", action="store_true")
    p.add_argument("--skip-confirm", action="store_true")
    p.add_argument("--list-only", action="store_true")

    args, unknown = p.parse_known_args()
    if unknown:
        print("[Colab/Jupyter notice] Ignored kernel arguments:", unknown)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    phase_paths = generate_phase_paths(args.n_random, args.seed)
    path_names = list(phase_paths.keys())

    print("=" * 100)
    print("PASQAL Cloud v1.1 — Random-Phase Area-Matched Manifold Test")
    print("=" * 100)
    print("Purpose: defend v1.0 against the 'flat control is straw-man' objection.")
    print("Main comparison: geoflow_bb1 vs random-phase area/duration-matched ensemble.")
    print("Default target: PASQAL Cloud EMU_FREE, not physical FRESNEL QPU.")
    print("Password hidden and not saved.")

    email, password, project_id = prompt_credentials(force_prompt=args.prompt)
    pulser_pasqal, cloud, constructor_used = connect_cloud(email, password, project_id, args.auto_install)

    print("\n[CONNECTED]")
    print("pulser_pasqal_version =", getattr(pulser_pasqal, "__version__", None))
    print("constructor_used =", constructor_used)
    print("email =", redact(email))
    print("project_id =", redact(project_id) if project_id else None)

    print("\n[FETCH DEVICES]")
    devices = cloud.fetch_available_devices()
    print("available device keys =", list(devices.keys()))

    if args.sequence_device not in devices:
        print(f"[ERROR] sequence-device {args.sequence_device!r} not available.")
        raise SystemExit(2)

    seq_device = devices[args.sequence_device]
    dev_type, dev_type_how = coerce_device_type(args.device_type)

    print("\n[TEST CONFIG]")
    print("sequence_device =", args.sequence_device)
    print("submit_device_type =", args.device_type, "via", dev_type_how)
    print("scales =", args.scales)
    print("n_random =", args.n_random)
    print("seed =", args.seed)
    print("runs =", args.runs)
    print("target_duration_ns =", args.target_duration_ns)
    print("max_total_duration_ns =", args.max_total_duration_ns)
    print("n_jobs =", len(path_names) * len(args.scales))
    print("BB1 phi =", bb1_phi_for_pi())
    if str(args.device_type).startswith("EMU_"):
        print("Readout: EMU_* success is cloud emulator evidence, not physical QPU evidence.")

    print("\n[PHASE PATHS]")
    for name in path_names:
        print(f" {name:22s}", [round(x, 6) for x in phase_paths[name]])

    built = []
    print("\n[BUILD SEQUENCES]")
    for scale in args.scales:
        for name in path_names:
            phases = phase_paths[name]
            try:
                seq, meta = build_phase_path_sequence(
                    seq_device,
                    path_name=name,
                    phases=phases,
                    scale=float(scale),
                    target_duration_ns=args.target_duration_ns,
                    duration_step_ns=args.duration_step_ns,
                    min_duration_ns=args.min_duration_ns,
                    max_total_duration_ns=args.max_total_duration_ns,
                )
                seq = cloud.update_sequence_device(seq)
                built.append({
                    "path_name": name,
                    "scale": float(scale),
                    "sequence": seq,
                    "meta": meta,
                    "build_ok": True,
                })
                print(
                    f" scale={float(scale):.3f} path={name:22s} "
                    f"build_ok=True expected={meta['expected_p_excited']:.4f} "
                    f"duration={meta['total_duration_ns']} area={meta['total_area_abs_rad']:.3f} "
                    f"amp={meta['amp_rad_per_us_global']:.3f}"
                )
            except BaseException as e:
                built.append({
                    "path_name": name,
                    "scale": float(scale),
                    "sequence": None,
                    "meta": {},
                    "build_ok": False,
                    "failure_classification": classify_error(e),
                    "error": f"{type(e).__name__}: {str(e)[:2000]}",
                })
                print(
                    f" scale={float(scale):.3f} path={name:22s} "
                    f"build_ok=False {classify_error(e)} {str(e)[:600]}"
                )

    build_failures = [b for b in built if not b["build_ok"]]

    report: Dict[str, Any] = {
        "schema": "pasqal_cloud_random_phase_manifold_v11",
        "connected": True,
        "pulser_pasqal_version": getattr(pulser_pasqal, "__version__", None),
        "constructor_used": constructor_used,
        "email_redacted": redact(email),
        "project_id_redacted": redact(project_id) if project_id else None,
        "available_devices": list(devices.keys()),
        "sequence_device": args.sequence_device,
        "submit_device_type": args.device_type,
        "submit_device_type_coerced": safe_repr(dev_type, 1000),
        "scales": [float(x) for x in args.scales],
        "n_random": int(args.n_random),
        "seed": int(args.seed),
        "phase_paths": phase_paths,
        "runs": int(args.runs),
        "target_duration_ns": int(args.target_duration_ns),
        "max_total_duration_ns": int(args.max_total_duration_ns),
        "built_meta": [{k: v for k, v in b.items() if k != "sequence"} for b in built],
        "submitted": False,
        "rows": [],
        "analysis": {},
        "claim_boundary": (
            "PASQAL Cloud EMU random-phase area-matched manifold test. "
            "Not physical QPU evidence. BB1 is treated as a known robustness landmark, "
            "not as a newly invented pulse formula."
        ),
    }

    if args.list_only or build_failures:
        path = outdir / "random_phase_manifold_build_status_v11.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        if build_failures:
            print("\n[BUILD FAILED] no submit; fix build errors first.")
        else:
            print("\n[LIST ONLY] no submit")
        print("[FILES]")
        print(path)
        return

    print("\n[CONFIRM SUBMIT]")
    print(f"This will submit {len(built)} PASQAL Cloud EMU job(s), each with runs={args.runs}.")
    print("This may consume PASQAL quota/credits.")
    print("Main goal: compare geoflow_bb1 to area/duration-matched random phase paths.")
    if not args.skip_confirm:
        txt = input("Type exactly SUBMIT to send the v1.1 random-phase manifold test: ").strip()
        if txt != "SUBMIT":
            report["reason"] = "user did not type SUBMIT"
            path = outdir / "random_phase_manifold_status_v11.json"
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            print("[ABORTED] no jobs submitted.")
            print("[FILES]")
            print(path)
            return

    print("\n[SUBMITTING RANDOM-PHASE MANIFOLD TEST]")
    report["submitted"] = True
    rows = []

    for item in built:
        name = item["path_name"]
        scale = item["scale"]
        seq = item["sequence"]
        meta = item["meta"]

        print(f"\n--- scale={scale:.3f} path={name} ---")
        row = {
            "path_name": name,
            "scale": scale,
            **meta,
            "submit_ok": False,
        }

        try:
            out = submit_one(
                cloud=cloud,
                seq=seq,
                dev_type=dev_type,
                runs=args.runs,
                wait=not args.no_wait,
                open_batch=args.open_batch,
            )
            row.update(out)
            row["abs_error_vs_expected"] = (
                abs(float(row["p_excited_like"]) - float(row["expected_p_excited"]))
                if row.get("counts_found") else float("nan")
            )
            print("submit: ok")
            print("batch_id =", out.get("batch_id_redacted"))
            print("get_results_ok =", out.get("get_results_ok"))
            print("P_excited =", out.get("p_excited_like"), "expected =", meta.get("expected_p_excited"))
        except BaseException as e:
            row.update({
                "submit_ok": False,
                "failure_classification": classify_error(e),
                "error_type": type(e).__name__,
                "error": str(e)[:6000],
                "traceback_tail": traceback.format_exc()[-5000:],
            })
            print("submit: failed")
            print("classification =", classify_error(e))
            print(type(e).__name__, str(e)[:2500])

        rows.append(row)

    analysis = analyze_manifold(rows)

    report["rows"] = rows
    report["analysis"] = analysis
    report["all_submit_ok"] = all(r.get("submit_ok") for r in rows)
    report["pass_random_phase_manifold"] = bool(analysis.get("pass_random_phase_manifold"))

    json_path = outdir / "random_phase_manifold_status_v11.json"
    csv_path = outdir / "random_phase_manifold_results_v11.csv"
    agg_csv_path = outdir / "random_phase_path_aggregates_v11.csv"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_csv(rows, csv_path)
    write_aggregate_csv(analysis, agg_csv_path)
    plot_paths = make_plot(analysis, outdir)

    print("\n[STATUS SUMMARY]")
    print(json.dumps({
        "submitted": report.get("submitted"),
        "all_submit_ok": report.get("all_submit_ok"),
        "n_jobs": len(rows),
        "n_submit_ok": sum(1 for r in rows if r.get("submit_ok")),
        "n_counts_found": sum(1 for r in rows if r.get("counts_found")),
        "pass_random_phase_manifold": analysis.get("pass_random_phase_manifold"),
        "flat_mean": analysis.get("flat_mean"),
        "bb1_mean": analysis.get("bb1_mean"),
        "random_median": analysis.get("random_median"),
        "random_mean": analysis.get("random_mean"),
        "random_best": analysis.get("random_best"),
        "random_p75": analysis.get("random_p75"),
        "random_p90": analysis.get("random_p90"),
        "bb1_percentile_rank_vs_random": analysis.get("bb1_percentile_rank_vs_random"),
        "bb1_vs_random_median_abs": analysis.get("bb1_vs_random_median_abs"),
        "bb1_vs_random_median_rel": analysis.get("bb1_vs_random_median_rel"),
        "pass_bb1_beats_random_median": analysis.get("pass_bb1_beats_random_median"),
        "pass_bb1_high_percentile": analysis.get("pass_bb1_high_percentile"),
        "pass_random_search_discovers_high": analysis.get("pass_random_search_discovers_high"),
        "best_random_name": analysis.get("best_random_name"),
        "best_random_mean": analysis.get("best_random_mean"),
        "bb1_close_to_best": analysis.get("bb1_close_to_best"),
    }, indent=2, ensure_ascii=False))

    print("\n[PATH AGGREGATES]")
    for name, a in sorted(analysis.get("path_aggregates", {}).items(), key=lambda kv: float(kv[1]["mean_measured"]), reverse=True):
        print(
            f"{name:22s} class={a.get('path_class'):16s} "
            f"mean={float(a['mean_measured']):.4f} "
            f"min={float(a['min_measured']):.4f} "
            f"max={float(a['max_measured']):.4f} "
            f"phases={[round(float(x), 4) for x in a.get('phases_rad', [])]}"
        )

    print("\n[FILES]")
    print(json_path)
    print(csv_path)
    print(agg_csv_path)
    for pth in plot_paths:
        print(pth)

    print("\n[HONEST READOUT]")
    if analysis.get("pass_random_phase_manifold"):
        print("PASQAL_CLOUD_RANDOM_PHASE_MANIFOLD_EMU = SUCCESS")
        if analysis.get("pass_bb1_beats_random_median"):
            print("BB1/geoflow is high relative to the random phase median, not just better than flat control.")
        if analysis.get("pass_random_search_discovers_high"):
            print("Random phase search also discovers high-robustness regions, supporting manifold discoverability.")
        print("Boundary: this is PASQAL Cloud EMU, not physical FRESNEL QPU.")
        print("BB1 is a known landmark, not claimed as a newly invented composite pulse.")
    else:
        print("PASQAL_CLOUD_RANDOM_PHASE_MANIFOLD_EMU = PARTIAL/NOT YET")
        print("Inspect path aggregates. The result may still support v1.0, but avoid strong search/discoverability claims.")


if __name__ == "__main__":
    main()
