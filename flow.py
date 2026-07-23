#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Standalone Pasqal geometric-flow 5→7→9 atom scaling validation."""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASQAL / PULSER — GEOMETRIC FLOW MINIMAL VALIDATION v1

Question tested
---------------
In a genuine interacting Rydberg many-body simulation, can a locally identified
FULL coupled response operator:

  1) be recovered stably from local pulse perturbations,
  2) predict unseen perturbations better than diagonal/scalar/rotated controls,
  3) improve local optimization through regularized A^{-1} preconditioning?

This is a local Pulser + QuTiP emulator experiment. It is not a QPU claim and
not a universal compiler test.

Install in Colab
----------------
!pip -q install pulser pulser-simulation qutip scipy pandas matplotlib
!python pasqal_geometric_flow_minimal.py

Fast debug
----------
!python pasqal_geometric_flow_minimal.py --preset debug

Paper-style local screen
------------------------
!python pasqal_geometric_flow_minimal.py --preset screen

Outputs
-------
pasqal_geometric_flow_minimal_results/
    summary.json
    train_data.csv
    heldout_data.csv
    recovery_splits.csv
    optimization_trace.csv
    heldout_prediction.png
    operator_heatmap.png
    optimization_calls.png
"""


import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

try:
    import pulser
    from pulser import Pulse, Register, Sequence
    from pulser.devices import MockDevice
    from pulser_simulation import QutipEmulator
except Exception as exc:
    raise RuntimeError(
        "\nPulser scientific stack could not be imported.\n"
        "On a fresh Colab runtime, run:\n"
        "  %pip uninstall -y pulser pulser-core pulser-simulation qutip scipy numpy\n"
        "  %pip install --no-cache-dir --upgrade --force-reinstall "
        "\"pulser==1.8.0\" pandas matplotlib\n"
        "Then restart the runtime once before running this file.\n"
        f"\nOriginal import error: {type(exc).__name__}: {exc}"
    ) from exc


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class Config:
    seed: int = 20260724
    preset: str = "screen"
    n_atoms: int = 5
    spacing_um: float = 6.0

    duration_1_ns: int = 420
    duration_2_ns: int = 520
    omega_1: float = 2.0 * 2.0 * np.pi
    omega_2: float = 1.65 * 2.0 * np.pi
    delta_1: float = -2.3 * 2.0 * np.pi
    delta_2: float = 1.15 * 2.0 * np.pi

    train_points: int = 40
    heldout_points: int = 24
    perturbation_radius: float = 0.055
    heldout_radius_min: float = 0.018
    heldout_radius_max: float = 0.058
    ridge: float = 1.0e-10
    rotated_control_angle: float = 0.61

    sampling_rate: float = 0.08

    split_recoveries: int = 12
    split_fraction: float = 0.72

    optimization_starts: int = 2
    optimization_maxiter: int = 7
    optimization_start_radius: float = 0.075
    finite_difference_h: float = 0.004
    preconditioner_floor_fraction: float = 0.08
    max_step_norm: float = 0.030

    output_dir: str = "pasqal_geometric_flow_minimal_results"


def preset_config(name: str, seed: int, output_dir: str) -> Config:
    if name == "debug":
        return Config(
            seed=seed,
            preset=name,
            n_atoms=4,
            train_points=22,
            heldout_points=12,
            split_recoveries=5,
            optimization_starts=1,
            optimization_maxiter=4,
            sampling_rate=0.06,
            output_dir=output_dir,
        )
    if name == "screen":
        return Config(seed=seed, preset=name, output_dir=output_dir)
    if name == "formal":
        return Config(
            seed=seed,
            preset=name,
            n_atoms=6,
            train_points=72,
            heldout_points=48,
            split_recoveries=30,
            optimization_starts=5,
            optimization_maxiter=12,
            sampling_rate=0.12,
            output_dir=output_dir,
        )
    raise ValueError(f"Unknown preset: {name}")


# =============================================================================
# Pulser interacting Rydberg model
# =============================================================================

class RydbergLocalGeometry:
    """Two-segment global Rydberg pulse with four fractional controls."""

    parameter_names = ("dOmega1", "dDelta1", "dOmega2", "dDelta2")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        coords = [(i * cfg.spacing_um, 0.0) for i in range(cfg.n_atoms)]
        self.register = Register.from_coordinates(coords, prefix="q")
        self._cache: dict[tuple[float, ...], float] = {}
        self.simulator_calls = 0

        self.target_state = self._simulate_state(np.zeros(4, dtype=float))
        self.target_norm = float(self.target_state.norm())
        if not np.isfinite(self.target_norm) or self.target_norm <= 0:
            raise RuntimeError("Invalid reference state returned by Pulser simulation.")

    def build_sequence(self, z: np.ndarray) -> Sequence:
        z = np.asarray(z, dtype=float)
        if z.shape != (4,):
            raise ValueError("Expected four local pulse parameters.")

        omega1 = self.cfg.omega_1 * (1.0 + z[0])
        delta1 = self.cfg.delta_1 * (1.0 + z[1])
        omega2 = self.cfg.omega_2 * (1.0 + z[2])
        delta2 = self.cfg.delta_2 * (1.0 + z[3])

        if omega1 <= 0 or omega2 <= 0:
            raise ValueError("Perturbation produced a non-positive Rabi amplitude.")

        seq = Sequence(self.register, MockDevice)
        seq.declare_channel("rydberg", "rydberg_global")
        seq.add(
            Pulse.ConstantPulse(
                self.cfg.duration_1_ns,
                omega1,
                delta1,
                0.0,
            ),
            "rydberg",
        )
        seq.add(
            Pulse.ConstantPulse(
                self.cfg.duration_2_ns,
                omega2,
                delta2,
                0.0,
            ),
            "rydberg",
        )
        return seq

    def _simulate_state(self, z: np.ndarray):
        seq = self.build_sequence(z)
        emulator = QutipEmulator.from_sequence(
            seq,
            sampling_rate=self.cfg.sampling_rate,
            evaluation_times="Minimal",
        )
        result = emulator.run()
        self.simulator_calls += 1
        return result.get_final_state()

    def loss(self, z: np.ndarray) -> float:
        """
        Infidelity to the unperturbed many-body final state.

        This gives a device/process-local robustness landscape:
            J(z) = 1 - |<psi(0)|psi(z)>|^2.
        """
        z = np.asarray(z, dtype=float)
        key = tuple(np.round(z, 12))
        if key in self._cache:
            return self._cache[key]

        state = self._simulate_state(z)
        overlap = self.target_state.overlap(state)
        fidelity = float(np.clip(abs(overlap) ** 2, 0.0, 1.0))
        value = 1.0 - fidelity
        self._cache[key] = value
        return value

    def gradient(self, z: np.ndarray, h: float) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        grad = np.zeros(4, dtype=float)
        for j in range(4):
            e = np.zeros(4, dtype=float)
            e[j] = h
            grad[j] = (self.loss(z + e) - self.loss(z - e)) / (2.0 * h)
        return grad


# =============================================================================
# Local quadratic identification
# =============================================================================

def unit_directions(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    x = rng.normal(size=(n, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def symmetric_local_design(
    rng: np.random.Generator,
    n_points: int,
    radius: float,
    dim: int = 4,
) -> np.ndarray:
    """Antithetic random design, excluding the origin."""
    half = math.ceil(n_points / 2)
    directions = unit_directions(rng, half, dim)
    radii = radius * rng.uniform(0.30, 1.0, size=(half, 1))
    positive = directions * radii
    points = np.vstack([positive, -positive])
    return points[:n_points]


def heldout_design(
    rng: np.random.Generator,
    n_points: int,
    r_min: float,
    r_max: float,
    dim: int = 4,
) -> np.ndarray:
    directions = unit_directions(rng, n_points, dim)
    radii = rng.uniform(r_min, r_max, size=(n_points, 1))
    return directions * radii


def quadratic_features(z: np.ndarray) -> np.ndarray:
    """
    Design row for:
        J = c + g^T z + 1/2 z^T A z

    Coefficients are:
        c,
        g_i,
        0.5*z_i^2 for A_ii,
        z_i*z_j for A_ij, i<j.
    """
    z = np.asarray(z, dtype=float)
    values = [1.0]
    values.extend(z.tolist())
    values.extend((0.5 * z * z).tolist())
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            values.append(float(z[i] * z[j]))
    return np.asarray(values, dtype=float)


def unpack_quadratic(beta: np.ndarray, dim: int = 4):
    c = float(beta[0])
    g = np.asarray(beta[1 : 1 + dim], dtype=float)
    diag_start = 1 + dim
    diag_end = diag_start + dim
    a = np.zeros((dim, dim), dtype=float)
    a[np.diag_indices(dim)] = beta[diag_start:diag_end]

    k = diag_end
    for i in range(dim):
        for j in range(i + 1, dim):
            a[i, j] = beta[k]
            a[j, i] = beta[k]
            k += 1
    return c, g, 0.5 * (a + a.T)


def fit_quadratic(z: np.ndarray, y: np.ndarray, ridge: float):
    x = np.vstack([quadratic_features(row) for row in z])
    penalty = np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge * penalty, x.T @ y)
    return unpack_quadratic(beta), beta


def predict_quadratic(
    z: np.ndarray,
    c: float,
    g: np.ndarray,
    a: np.ndarray,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return c + z @ g + 0.5 * np.einsum("ni,ij,nj->n", z, a, z)


def scalar_operator(a: np.ndarray) -> np.ndarray:
    return np.eye(a.shape[0]) * np.trace(a) / a.shape[0]


def rotated_operator(a: np.ndarray, angle: float) -> np.ndarray:
    """Rotate only the first two eigendirections while preserving eigenvalues."""
    values, vectors = np.linalg.eigh(a)
    r = np.eye(a.shape[0])
    c, s = np.cos(angle), np.sin(angle)
    r[:2, :2] = np.array([[c, -s], [s, c]])
    return vectors @ r @ np.diag(values) @ r.T @ vectors.T


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = y - pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    rho = float(spearmanr(y, pred).statistic)
    return {"r2": r2, "rmse": rmse, "mae": mae, "spearman": rho}


def operator_diagnostics(a: np.ndarray) -> dict[str, float]:
    eig = np.linalg.eigvalsh(a)
    off = a - np.diag(np.diag(a))
    return {
        "min_eigenvalue": float(np.min(eig)),
        "max_eigenvalue": float(np.max(eig)),
        "condition_number_abs": float(
            np.max(np.abs(eig)) / max(np.min(np.abs(eig)), 1e-14)
        ),
        "offdiagonal_frobenius_fraction": float(
            np.linalg.norm(off, "fro") / max(np.linalg.norm(a, "fro"), 1e-14)
        ),
    }


# =============================================================================
# Recovery stability
# =============================================================================

def split_recovery_audit(
    rng: np.random.Generator,
    z: np.ndarray,
    y: np.ndarray,
    cfg: Config,
    reference_a: np.ndarray,
) -> pd.DataFrame:
    rows = []
    subset_size = max(16, int(round(cfg.split_fraction * len(z))))

    for split_id in range(cfg.split_recoveries):
        idx = rng.choice(len(z), size=subset_size, replace=False)
        (_, _, a_split), _ = fit_quadratic(z[idx], y[idx], cfg.ridge)

        rel = np.linalg.norm(a_split - reference_a, "fro") / max(
            np.linalg.norm(reference_a, "fro"), 1e-14
        )

        _, v_ref = np.linalg.eigh(reference_a)
        _, v_split = np.linalg.eigh(a_split)
        principal_alignment = float(
            abs(np.dot(v_ref[:, -1], v_split[:, -1]))
        )

        rows.append(
            {
                "split_id": split_id,
                "subset_size": subset_size,
                "relative_frobenius_error": rel,
                "principal_eigenvector_alignment": principal_alignment,
                **operator_diagnostics(a_split),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Optimization test
# =============================================================================

def regularized_inverse(a: np.ndarray, floor_fraction: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (a + a.T))
    scale = max(float(np.max(np.abs(values))), 1e-12)
    floor = floor_fraction * scale
    safe = np.maximum(np.abs(values), floor)
    return vectors @ np.diag(1.0 / safe) @ vectors.T


def capped_step(step: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(step))
    if norm <= max_norm:
        return step
    return step * (max_norm / norm)


def optimize_gradient_method(
    model: RydbergLocalGeometry,
    z0: np.ndarray,
    cfg: Config,
    method: str,
    a_inv: np.ndarray | None,
) -> list[dict[str, float | int | str]]:
    z = np.asarray(z0, dtype=float).copy()
    trace = []
    start_calls = model.simulator_calls

    # Common conservative initial learning rate.
    eta = 0.55 if method == "gradient" else 0.18

    for iteration in range(cfg.optimization_maxiter + 1):
        value = model.loss(z)
        trace.append(
            {
                "method": method,
                "iteration": iteration,
                "loss": value,
                "z_norm": float(np.linalg.norm(z)),
                "simulator_calls_since_start": model.simulator_calls - start_calls,
                **{f"z{i+1}": float(z[i]) for i in range(4)},
            }
        )
        if iteration == cfg.optimization_maxiter:
            break

        grad = model.gradient(z, cfg.finite_difference_h)
        direction = grad if method == "gradient" else a_inv @ grad
        raw_step = -eta * direction
        step = capped_step(raw_step, cfg.max_step_norm)

        # Tiny deterministic backtracking, shared by both methods.
        accepted = False
        for _ in range(5):
            candidate = z + step
            if model.loss(candidate) <= value:
                z = candidate
                accepted = True
                break
            step *= 0.5

        if not accepted:
            break

    return trace


# =============================================================================
# Plotting
# =============================================================================

def plot_operator(a: np.ndarray, names: tuple[str, ...], path: Path) -> None:
    plt.figure(figsize=(6.2, 5.3))
    im = plt.imshow(a)
    plt.colorbar(im, label="Recovered response coefficient")
    plt.xticks(range(len(names)), names, rotation=35, ha="right")
    plt.yticks(range(len(names)), names)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            plt.text(j, i, f"{a[i, j]:.3g}", ha="center", va="center")
    plt.title("Full coupled local response operator")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_heldout(y: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> None:
    plt.figure(figsize=(6.3, 5.4))
    for name, pred in predictions.items():
        plt.scatter(y, pred, s=28, alpha=0.78, label=name)
    lo = min(float(np.min(y)), *(float(np.min(v)) for v in predictions.values()))
    hi = max(float(np.max(y)), *(float(np.max(v)) for v in predictions.values()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Observed held-out infidelity")
    plt.ylabel("Predicted held-out infidelity")
    plt.title("Rydberg pulse held-out prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_optimization(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(6.5, 5.2))
    grouped = (
        df.groupby(["method", "iteration"], as_index=False)["loss"]
        .median()
        .sort_values(["method", "iteration"])
    )
    for method, sub in grouped.groupby("method"):
        plt.plot(sub["iteration"], sub["loss"], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Median infidelity")
    plt.title("Local optimization: raw gradient vs geometric preconditioning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# =============================================================================
# Experiment
# =============================================================================

def run(cfg: Config) -> dict[str, object]:
    start_time = time.time()
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] Building interacting Pulser model and reference state...")
    model = RydbergLocalGeometry(cfg)
    baseline_loss = model.loss(np.zeros(4))
    print(f"      baseline infidelity = {baseline_loss:.3e}")

    print("[2/5] Simulating local discovery perturbations...")
    z_train = symmetric_local_design(
        rng, cfg.train_points, cfg.perturbation_radius, dim=4
    )
    y_train = np.asarray([model.loss(z) for z in z_train])
    train_df = pd.DataFrame(z_train, columns=model.parameter_names)
    train_df["loss"] = y_train
    train_df.to_csv(out / "train_data.csv", index=False)

    (c, g, a_full), _ = fit_quadratic(z_train, y_train, cfg.ridge)
    a_diag = np.diag(np.diag(a_full))
    a_scalar = scalar_operator(a_full)
    a_rotated = rotated_operator(a_full, cfg.rotated_control_angle)

    print("[3/5] Running recovery stability and independent held-out prediction...")
    split_df = split_recovery_audit(
        rng, z_train, y_train, cfg, reference_a=a_full
    )
    split_df.to_csv(out / "recovery_splits.csv", index=False)

    z_test = heldout_design(
        rng,
        cfg.heldout_points,
        cfg.heldout_radius_min,
        cfg.heldout_radius_max,
        dim=4,
    )
    y_test = np.asarray([model.loss(z) for z in z_test])

    predictions = {
        "full": predict_quadratic(z_test, c, g, a_full),
        "diagonal": predict_quadratic(z_test, c, g, a_diag),
        "scalar": predict_quadratic(z_test, c, g, a_scalar),
        "rotated": predict_quadratic(z_test, c, g, a_rotated),
    }
    metrics = {
        name: regression_metrics(y_test, pred)
        for name, pred in predictions.items()
    }

    heldout_df = pd.DataFrame(z_test, columns=model.parameter_names)
    heldout_df["observed_loss"] = y_test
    for name, pred in predictions.items():
        heldout_df[f"pred_{name}"] = pred
        heldout_df[f"abs_error_{name}"] = np.abs(y_test - pred)
    heldout_df.to_csv(out / "heldout_data.csv", index=False)

    print("[4/5] Comparing local optimization methods...")
    a_inv = regularized_inverse(
        a_full, cfg.preconditioner_floor_fraction
    )
    opt_rows: list[dict[str, float | int | str]] = []
    for start_id in range(cfg.optimization_starts):
        direction = rng.normal(size=4)
        direction /= np.linalg.norm(direction)
        z0 = cfg.optimization_start_radius * direction

        for method in ("gradient", "geometric"):
            rows = optimize_gradient_method(
                model,
                z0,
                cfg,
                method,
                a_inv if method == "geometric" else None,
            )
            for row in rows:
                row["start_id"] = start_id
            opt_rows.extend(rows)

    opt_df = pd.DataFrame(opt_rows)
    opt_df.to_csv(out / "optimization_trace.csv", index=False)

    opt_summary: dict[str, dict[str, float]] = {}
    if not opt_df.empty:
        for method, sub in opt_df.groupby("method"):
            final_rows = sub.sort_values("iteration").groupby("start_id").tail(1)
            opt_summary[method] = {
                "median_final_loss": float(final_rows["loss"].median()),
                "mean_final_loss": float(final_rows["loss"].mean()),
                "median_calls": float(
                    final_rows["simulator_calls_since_start"].median()
                ),
                "median_iterations": float(final_rows["iteration"].median()),
            }

    print("[5/5] Computing gates and saving plots...")
    diag = operator_diagnostics(a_full)
    median_split_error = float(split_df["relative_frobenius_error"].median())
    median_alignment = float(
        split_df["principal_eigenvector_alignment"].median()
    )

    full_mae = metrics["full"]["mae"]
    mae_ratios = {
        name: float(metrics[name]["mae"] / max(full_mae, 1e-30))
        for name in ("diagonal", "scalar", "rotated")
    }

    geometric_better = False
    if "gradient" in opt_summary and "geometric" in opt_summary:
        geometric_better = (
            opt_summary["geometric"]["median_final_loss"]
            < opt_summary["gradient"]["median_final_loss"]
        )

    gates = {
        "G1_full_operator_has_nontrivial_offdiagonal_fraction": bool(
            diag["offdiagonal_frobenius_fraction"] > 0.10
        ),
        "G2_split_recovery_median_frobenius_error_lt_0_20": bool(
            median_split_error < 0.20
        ),
        "G3_split_principal_alignment_gt_0_90": bool(
            median_alignment > 0.90
        ),
        "G4_full_heldout_R2_positive": bool(metrics["full"]["r2"] > 0.0),
        "G5_full_heldout_spearman_gt_0_80": bool(
            metrics["full"]["spearman"] > 0.80
        ),
        "G6_full_beats_diagonal_MAE": bool(
            metrics["full"]["mae"] < metrics["diagonal"]["mae"]
        ),
        "G7_full_beats_scalar_MAE": bool(
            metrics["full"]["mae"] < metrics["scalar"]["mae"]
        ),
        "G8_full_beats_rotated_MAE": bool(
            metrics["full"]["mae"] < metrics["rotated"]["mae"]
        ),
        "G9_geometric_optimization_beats_raw_gradient": bool(
            geometric_better
        ),
    }
    gates["ALL_LOCAL_GEOMETRY_GATES_PASS"] = bool(all(gates.values()))

    summary = {
        "scope": (
            "Local interacting Rydberg emulator validation. "
            "No physical-QPU, universal-compiler, or quantum-advantage claim."
        ),
        "config": asdict(cfg),
        "pulser_version": getattr(pulser, "__version__", "unknown"),
        "baseline_loss": baseline_loss,
        "quadratic_intercept": c,
        "quadratic_gradient": g.tolist(),
        "A_full": a_full.tolist(),
        "operator_diagnostics": diag,
        "split_recovery": {
            "median_relative_frobenius_error": median_split_error,
            "median_principal_alignment": median_alignment,
        },
        "heldout_metrics": metrics,
        "heldout_mae_ratios_vs_full": mae_ratios,
        "optimization_summary": opt_summary,
        "simulator_calls": model.simulator_calls,
        "gates": gates,
        "elapsed_seconds": time.time() - start_time,
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    plot_operator(
        a_full,
        model.parameter_names,
        out / "operator_heatmap.png",
    )
    plot_heldout(
        y_test,
        predictions,
        out / "heldout_prediction.png",
    )
    plot_optimization(
        opt_df,
        out / "optimization_calls.png",
    )

    return summary


# =============================================================================


# =============================================================================
# SOURCE-ALIGNED MINIMAL CLOSURE EXTENSION
# =============================================================================
#
# This section leaves the original model, pulse construction, loss, sampling
# functions, quadratic fit, and Pulser/QutipEmulator backend unchanged.
#
# Added only:
#   1. multiple seeds,
#   2. perturbation-radius scan,
#   3. independent full-emulator optimizer comparison,
#   4. coordinate-transformation covariance audit.
#
# Primary held-out target remains the original unseen FULL-emulator infidelity.
# No surrogate is relabelled as a re-optimized simulator target.
# =============================================================================

def _csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=("debug", "screen", "formal"), default="screen")
    p.add_argument("--atoms", default="5,7,9")
    p.add_argument("--seeds", default="20260724,20260731,20260807")
    p.add_argument("--radii", default="0.025,0.040,0.055,0.070,0.085")
    p.add_argument(
        "--output-dir",
        default="pasqal_full_validation_results",
    )
    p.add_argument("--coordinate-trials", type=int, default=6)
    p.add_argument("--optimizer-starts", type=int, default=5)
    p.add_argument("--optimizer-maxiter", type=int, default=10)
    p.add_argument("--skip-radius-scan", action="store_true")
    p.add_argument("--skip-optimizer", action="store_true")
    p.add_argument("--skip-coordinate-audit", action="store_true")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[notice] Ignored notebook/kernel arguments: {unknown}")
    return args


def config_for_atoms(preset: str, seed: int, outdir: str, n_atoms: int) -> Config:
    """Original v1 settings, changing only the requested atom count."""
    cfg = preset_config(preset, seed, outdir)
    if preset == "debug":
        return replace(
            cfg,
            n_atoms=n_atoms,
            train_points=22,
            heldout_points=12,
            split_recoveries=5,
            sampling_rate=0.055,
        )
    if preset == "screen":
        return replace(
            cfg,
            n_atoms=n_atoms,
            train_points=40,
            heldout_points=24,
            split_recoveries=12,
            sampling_rate=0.08,
        )
    return replace(
        cfg,
        n_atoms=n_atoms,
        train_points=72,
        heldout_points=48,
        split_recoveries=30,
        sampling_rate=0.12,
    )


def _relative_frobenius(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.linalg.norm(a - b, "fro") / max(np.linalg.norm(a, "fro"), 1e-14)
    )


def _matched_eigenvector_alignment(a: np.ndarray, b: np.ndarray) -> float:
    """
    Minimum eigenvector overlap under the best permutation.

    This is stricter and more stable than comparing only the largest-eigenvalue
    vector when eigenvalues reorder under a coordinate transformation.
    """
    _, va = np.linalg.eigh(0.5 * (a + a.T))
    _, vb = np.linalg.eigh(0.5 * (b + b.T))
    overlap = np.abs(va.T @ vb)

    import itertools
    best = 0.0
    for perm in itertools.permutations(range(a.shape[0])):
        best = max(best, min(float(overlap[i, perm[i]]) for i in range(a.shape[0])))
    return best


def coordinate_covariance_audit(
    rng: np.random.Generator,
    z: np.ndarray,
    y: np.ndarray,
    c: float,
    g: np.ndarray,
    a: np.ndarray,
    ridge: float,
    n_trials: int,
) -> pd.DataFrame:
    """
    Audit the quadratic-coordinate transformation law without changing the model.

    Define q = T z, hence z = T^{-1}q. Then

        g_q = T^{-T} g_z,
        A_q = T^{-T} A_z T^{-1}.

    The same observed FULL-emulator losses y are refit in q coordinates and
    compared with the analytically transformed coefficients.
    """
    rows = []
    for trial in range(n_trials):
        qmat, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        scales = np.exp(rng.uniform(np.log(0.75), np.log(1.35), size=4))
        tmat = qmat @ np.diag(scales)
        tinv = np.linalg.inv(tmat)

        q_points = z @ tmat.T
        (c_fit, g_fit, a_fit), _ = fit_quadratic(q_points, y, ridge)

        g_expected = tinv.T @ g
        a_expected = tinv.T @ a @ tinv

        rows.append(
            {
                "trial": trial,
                "condition_T": float(np.linalg.cond(tmat)),
                "intercept_abs_error": float(abs(c_fit - c)),
                "gradient_relative_error": float(
                    np.linalg.norm(g_fit - g_expected)
                    / max(np.linalg.norm(g_expected), 1e-14)
                ),
                "operator_relative_fro_error": _relative_frobenius(
                    a_expected, a_fit
                ),
                "operator_matched_alignment": _matched_eigenvector_alignment(
                    a_expected, a_fit
                ),
            }
        )
    return pd.DataFrame(rows)


def run_identification_case(
    cfg: Config,
    rng: np.random.Generator,
    case_out: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Original v1 identification/prediction protocol, unchanged in substance.
    """
    model = RydbergLocalGeometry(cfg)

    z_train = symmetric_local_design(
        rng, cfg.train_points, cfg.perturbation_radius, dim=4
    )
    y_train = np.asarray([model.loss(z) for z in z_train])
    (c, g, a), _ = fit_quadratic(z_train, y_train, cfg.ridge)

    split_df = split_recovery_audit(rng, z_train, y_train, cfg, a)
    split_df["n_atoms"] = cfg.n_atoms
    split_df["seed"] = cfg.seed
    split_df["radius"] = cfg.perturbation_radius

    z_test = heldout_design(
        rng,
        cfg.heldout_points,
        cfg.heldout_radius_min,
        cfg.heldout_radius_max,
        dim=4,
    )
    y_test = np.asarray([model.loss(z) for z in z_test])

    operators = {
        "full": a,
        "diagonal": np.diag(np.diag(a)),
        "scalar": scalar_operator(a),
        "rotated": rotated_operator(a, cfg.rotated_control_angle),
    }

    metrics = {}
    heldout = pd.DataFrame(z_test, columns=model.parameter_names)
    heldout["observed_loss"] = y_test
    heldout["n_atoms"] = cfg.n_atoms
    heldout["seed"] = cfg.seed
    heldout["radius"] = cfg.perturbation_radius

    for name, operator in operators.items():
        pred = predict_quadratic(z_test, c, g, operator)
        metrics[name] = regression_metrics(y_test, pred)
        heldout[f"pred_{name}"] = pred
        heldout[f"abs_error_{name}"] = np.abs(y_test - pred)

    diag = operator_diagnostics(a)
    full_mae = metrics["full"]["mae"]

    row = {
        "n_atoms": cfg.n_atoms,
        "seed": cfg.seed,
        "radius": cfg.perturbation_radius,
        "simulator_calls_identification": model.simulator_calls,
        "baseline_loss": model.loss(np.zeros(4)),
        "recovery_median_fro_error": float(
            split_df["relative_frobenius_error"].median()
        ),
        "recovery_median_alignment": float(
            split_df["principal_eigenvector_alignment"].median()
        ),
        **diag,
        "full_r2": metrics["full"]["r2"],
        "full_spearman": metrics["full"]["spearman"],
        "full_mae": full_mae,
        "diag_mae_ratio": metrics["diagonal"]["mae"] / max(full_mae, 1e-30),
        "scalar_mae_ratio": metrics["scalar"]["mae"] / max(full_mae, 1e-30),
        "rotated_mae_ratio": metrics["rotated"]["mae"] / max(full_mae, 1e-30),
    }

    case_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(z_train, columns=model.parameter_names).assign(
        loss=y_train
    ).to_csv(case_out / "train_data.csv", index=False)
    heldout.to_csv(case_out / "heldout_data.csv", index=False)
    split_df.to_csv(case_out / "recovery_splits.csv", index=False)
    (case_out / "A_full.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "c": c,
                "g": g.tolist(),
                "A_full": a.tolist(),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return row, split_df, heldout, z_train, y_train, c, g, a


def radius_scan(
    base_cfg: Config,
    seed: int,
    radii: list[float],
    out: Path,
) -> pd.DataFrame:
    """
    Repeat the original identification protocol at each declared radius.

    Only perturbation/held-out radii are changed; pulse construction, backend,
    loss, sampling design, fit, and diagnostics are unchanged.
    """
    rows = []
    scan_out = out / "radius_scan"
    scan_out.mkdir(parents=True, exist_ok=True)

    for j, radius in enumerate(radii):
        cfg = replace(
            base_cfg,
            seed=seed + 100_000 + j,
            perturbation_radius=float(radius),
            heldout_radius_min=max(0.30 * float(radius), 1e-4),
            heldout_radius_max=1.05 * float(radius),
        )
        rng = np.random.default_rng(cfg.seed)
        case_out = scan_out / f"radius_{radius:.5f}"
        row, *_ = run_identification_case(cfg, rng, case_out)
        rows.append(row)
        print(
            f"      radius={radius:.5f} "
            f"rec={row['recovery_median_fro_error']:.4f} "
            f"align={row['recovery_median_alignment']:.4f} "
            f"R2={row['full_r2']:.6f} "
            f"off={row['offdiagonal_frobenius_fraction']:.4f}"
        )

    df = pd.DataFrame(rows).sort_values("radius")
    df.to_csv(scan_out / "radius_scan_summary.csv", index=False)
    return df


def _bounded_point(z: np.ndarray, bound: float) -> np.ndarray:
    return np.clip(np.asarray(z, dtype=float), -bound, bound)


def gf_lm_optimize(
    model: RydbergLocalGeometry,
    z0: np.ndarray,
    a: np.ndarray,
    cfg: Config,
    maxiter: int,
    coordinate_bound: float,
) -> list[dict]:
    """
    Frozen-A Levenberg-Marquardt/trust implementation on the FULL objective.
    """
    z = _bounded_point(z0, coordinate_bound)
    a_sym = 0.5 * (a + a.T)
    eig_scale = max(float(np.max(np.abs(np.linalg.eigvalsh(a_sym)))), 1e-12)
    damping = 0.10 * eig_scale
    trust = cfg.max_step_norm
    eye = np.eye(4)
    rows = []
    start_calls = model.simulator_calls

    for iteration in range(maxiter + 1):
        value = model.loss(z)
        rows.append(
            {
                "method": "GF-LM",
                "iteration": iteration,
                "loss": value,
                "simulator_calls_since_start": model.simulator_calls - start_calls,
                "z_norm": float(np.linalg.norm(z)),
                **{f"z{i+1}": float(z[i]) for i in range(4)},
            }
        )
        if iteration == maxiter:
            break

        grad = model.gradient(z, cfg.finite_difference_h)
        try:
            step = np.linalg.solve(a_sym + damping * eye, -grad)
        except np.linalg.LinAlgError:
            step = -grad

        step = capped_step(step, trust)
        predicted = -(float(grad @ step) + 0.5 * float(step @ a_sym @ step))
        if predicted <= 1e-16:
            damping *= 5.0
            trust *= 0.5
            continue

        candidate = _bounded_point(z + step, coordinate_bound)
        candidate_value = model.loss(candidate)
        actual = value - candidate_value
        rho = actual / predicted

        if actual > 0.0 and rho > 0.10:
            z = candidate
            if rho > 0.75:
                damping = max(damping / 3.0, 1e-12)
                trust = min(1.5 * trust, 2.0 * cfg.max_step_norm)
            elif rho < 0.25:
                damping *= 2.0
                trust *= 0.7
        else:
            damping *= 5.0
            trust *= 0.5

        trust = max(trust, 1e-4)

    return rows


def independent_powell_optimize(
    model: RydbergLocalGeometry,
    z0: np.ndarray,
    maxiter: int,
    coordinate_bound: float,
) -> list[dict]:
    """
    Independent derivative-free SciPy Powell optimization on the same FULL
    Pulser/QutipEmulator objective and the same local coordinate box.
    """
    start_calls = model.simulator_calls
    history: list[np.ndarray] = []

    def objective(z):
        return model.loss(_bounded_point(z, coordinate_bound))

    def callback(xk):
        history.append(_bounded_point(np.asarray(xk, dtype=float), coordinate_bound))

    bounds = [(-coordinate_bound, coordinate_bound)] * 4
    z0 = _bounded_point(z0, coordinate_bound)
    initial_value = objective(z0)

    result = minimize(
        objective,
        z0,
        method="Powell",
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": maxiter,
            "xtol": 2e-4,
            "ftol": 1e-10,
            "disp": False,
        },
    )

    rows = [
        {
            "method": "Powell-independent",
            "iteration": 0,
            "loss": initial_value,
            "simulator_calls_since_start": 0,
            "z_norm": float(np.linalg.norm(z0)),
            **{f"z{i+1}": float(z0[i]) for i in range(4)},
        }
    ]

    for iteration, z in enumerate(history, start=1):
        value = model.loss(z)
        rows.append(
            {
                "method": "Powell-independent",
                "iteration": iteration,
                "loss": value,
                "simulator_calls_since_start": model.simulator_calls - start_calls,
                "z_norm": float(np.linalg.norm(z)),
                **{f"z{i+1}": float(z[i]) for i in range(4)},
            }
        )

    z_final = _bounded_point(np.asarray(result.x, dtype=float), coordinate_bound)
    if not rows or np.linalg.norm(
        z_final - np.array([rows[-1][f"z{i+1}"] for i in range(4)])
    ) > 1e-12:
        rows.append(
            {
                "method": "Powell-independent",
                "iteration": len(rows),
                "loss": model.loss(z_final),
                "simulator_calls_since_start": model.simulator_calls - start_calls,
                "z_norm": float(np.linalg.norm(z_final)),
                **{f"z{i+1}": float(z_final[i]) for i in range(4)},
            }
        )

    return rows


def optimizer_audit(
    cfg: Config,
    seed: int,
    a: np.ndarray,
    n_starts: int,
    maxiter: int,
    out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    GF-LM versus an independent derivative-free optimizer.

    Each method receives:
      * the same initial point,
      * the same FULL Pulser/QutipEmulator loss,
      * the same coordinate box,
      * the same declared outer iteration limit.

    Separate model instances prevent cache sharing between methods.
    """
    rng = np.random.default_rng(seed + 700_000)
    coordinate_bound = max(0.12, 1.5 * cfg.optimization_start_radius)
    rows = []

    for start_id in range(n_starts):
        direction = rng.normal(size=4)
        direction /= np.linalg.norm(direction)
        z0 = cfg.optimization_start_radius * direction

        gf_model = RydbergLocalGeometry(cfg)
        gf_rows = gf_lm_optimize(
            gf_model, z0, a, cfg, maxiter, coordinate_bound
        )

        independent_model = RydbergLocalGeometry(cfg)
        independent_rows = independent_powell_optimize(
            independent_model, z0, maxiter, coordinate_bound
        )

        for row in gf_rows + independent_rows:
            row["start_id"] = start_id
            row["n_atoms"] = cfg.n_atoms
            row["seed"] = cfg.seed
            rows.append(row)

    trace = pd.DataFrame(rows)
    final = (
        trace.sort_values(["method", "start_id", "iteration"])
        .groupby(["method", "start_id"], as_index=False)
        .tail(1)
    )
    summary = (
        final.groupby("method")
        .agg(
            median_final_loss=("loss", "median"),
            mean_final_loss=("loss", "mean"),
            median_calls=("simulator_calls_since_start", "median"),
            mean_calls=("simulator_calls_since_start", "mean"),
            median_iterations=("iteration", "median"),
        )
        .reset_index()
    )

    trace.to_csv(out / "optimizer_trace.csv", index=False)
    summary.to_csv(out / "optimizer_summary.csv", index=False)
    return trace, summary


def _plot_multiseed(summary: pd.DataFrame, out: Path) -> None:
    plot_specs = [
        (
            "recovery_median_fro_error",
            "Median split recovery error",
            "multiseed_recovery.png",
        ),
        (
            "full_r2",
            "Held-out full-model R²",
            "multiseed_heldout_r2.png",
        ),
        (
            "offdiagonal_frobenius_fraction",
            "Off-diagonal Frobenius fraction",
            "multiseed_offdiagonal.png",
        ),
        (
            "diag_mae_ratio",
            "Diagonal/full held-out MAE ratio",
            "multiseed_diagonal_ablation.png",
        ),
    ]

    for metric, ylabel, filename in plot_specs:
        plt.figure(figsize=(6.7, 4.7))
        for seed, sub in summary.groupby("seed"):
            sub = sub.sort_values("n_atoms")
            plt.plot(
                sub["n_atoms"],
                sub[metric],
                marker="o",
                label=str(seed),
            )
        plt.xlabel("Number of atoms")
        plt.ylabel(ylabel)
        plt.xticks(sorted(summary["n_atoms"].unique()))
        plt.legend(title="Seed")
        plt.tight_layout()
        plt.savefig(out / filename, dpi=180)
        plt.close()


def main():
    args = parse_args()
    atoms = _csv_ints(args.atoms)
    seeds = _csv_ints(args.seeds)
    radii = _csv_floats(args.radii)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("PASQAL / PULSER — SOURCE-ALIGNED COMPLETE LOCAL-GEOMETRY VALIDATION")
    print("=" * 118)
    print(f"atoms={atoms}")
    print(f"seeds={seeds}")
    print(f"preset={args.preset}")
    print(f"radii={radii}")
    try:
        import scipy
        import qutip
        import pulser_simulation
        print(
            "versions="
            f"numpy={np.__version__}, "
            f"scipy={scipy.__version__}, "
            f"qutip={qutip.__version__}, "
            f"pulser={getattr(pulser, '__version__', 'unknown')}, "
            f"pulser_simulation={getattr(pulser_simulation, '__version__', 'unknown')}"
        )
    except Exception as version_exc:
        print(f"[warning] version report failed: {version_exc}")
    print("backend=original Pulser/QutipEmulator")
    print("pulse model=original two-segment global Rydberg pulse")
    print("loss=original infidelity to unperturbed many-body final state")
    print("heldout target=original unseen FULL-emulator infidelity")
    print()

    t0 = time.time()
    primary_rows = []
    radius_frames = []
    covariance_frames = []
    optimizer_frames = []

    for n_atoms in atoms:
        for seed in seeds:
            case_seed = seed + 1000 * n_atoms
            case_out = out / f"N{n_atoms}_seed{seed}"
            cfg = config_for_atoms(
                args.preset,
                case_seed,
                str(case_out),
                n_atoms,
            )
            rng = np.random.default_rng(case_seed)

            print(f"[N={n_atoms} seed={seed}] original identification run...")
            (
                row,
                split_df,
                heldout_df,
                z_train,
                y_train,
                c,
                g,
                a,
            ) = run_identification_case(cfg, rng, case_out)

            if not args.skip_coordinate_audit:
                covariance = coordinate_covariance_audit(
                    np.random.default_rng(case_seed + 500_000),
                    z_train,
                    y_train,
                    c,
                    g,
                    a,
                    cfg.ridge,
                    args.coordinate_trials,
                )
                covariance["n_atoms"] = n_atoms
                covariance["seed"] = seed
                covariance.to_csv(
                    case_out / "coordinate_covariance.csv",
                    index=False,
                )
                covariance_frames.append(covariance)
                row["covariance_median_operator_error"] = float(
                    covariance["operator_relative_fro_error"].median()
                )
                row["covariance_median_alignment"] = float(
                    covariance["operator_matched_alignment"].median()
                )

            if not args.skip_radius_scan:
                print("    radius scan...")
                radius_df = radius_scan(
                    cfg,
                    case_seed,
                    radii,
                    case_out,
                )
                radius_df["base_seed"] = seed
                radius_frames.append(radius_df)
                row["radius_scan_stable_points"] = int(
                    np.sum(
                        (radius_df["recovery_median_fro_error"] < 0.10)
                        & (radius_df["recovery_median_alignment"] > 0.95)
                        & (radius_df["full_r2"] > 0.95)
                    )
                )

            if not args.skip_optimizer:
                print("    independent optimizer audit...")
                _, opt_summary = optimizer_audit(
                    cfg,
                    case_seed,
                    a,
                    args.optimizer_starts,
                    args.optimizer_maxiter,
                    case_out,
                )
                opt_summary["n_atoms"] = n_atoms
                opt_summary["seed"] = seed
                optimizer_frames.append(opt_summary)
                for _, opt_row in opt_summary.iterrows():
                    method_key = (
                        str(opt_row["method"])
                        .lower()
                        .replace("-", "_")
                        .replace(" ", "_")
                    )
                    row[f"{method_key}_median_final_loss"] = float(
                        opt_row["median_final_loss"]
                    )
                    row[f"{method_key}_median_calls"] = float(
                        opt_row["median_calls"]
                    )

            primary_rows.append(row)

            print(
                f"  recovery={row['recovery_median_fro_error']:.4f} "
                f"alignment={row['recovery_median_alignment']:.4f} "
                f"R2={row['full_r2']:.6f} "
                f"rho={row['full_spearman']:.6f} "
                f"off={row['offdiagonal_frobenius_fraction']:.4f} "
                f"diag/full={row['diag_mae_ratio']:.2f}x "
                f"calls={row['simulator_calls_identification']}"
            )

    primary = pd.DataFrame(primary_rows).sort_values(["n_atoms", "seed"])
    primary.to_csv(out / "multiseed_scaling_summary.csv", index=False)

    if radius_frames:
        pd.concat(radius_frames, ignore_index=True).to_csv(
            out / "all_radius_scans.csv",
            index=False,
        )
    if covariance_frames:
        pd.concat(covariance_frames, ignore_index=True).to_csv(
            out / "all_coordinate_covariance.csv",
            index=False,
        )
    if optimizer_frames:
        pd.concat(optimizer_frames, ignore_index=True).to_csv(
            out / "all_optimizer_summaries.csv",
            index=False,
        )

    gates = {
        "M1_all_recovery_errors_lt_0_10": bool(
            np.all(primary["recovery_median_fro_error"] < 0.10)
        ),
        "M2_all_alignments_gt_0_95": bool(
            np.all(primary["recovery_median_alignment"] > 0.95)
        ),
        "M3_all_full_R2_gt_0_95": bool(
            np.all(primary["full_r2"] > 0.95)
        ),
        "M4_all_full_Spearman_gt_0_95": bool(
            np.all(primary["full_spearman"] > 0.95)
        ),
        "M5_full_beats_diagonal_everywhere": bool(
            np.all(primary["diag_mae_ratio"] > 1.0)
        ),
        "M6_full_beats_scalar_everywhere": bool(
            np.all(primary["scalar_mae_ratio"] > 1.0)
        ),
        "M7_nontrivial_offdiagonal_everywhere": bool(
            np.all(primary["offdiagonal_frobenius_fraction"] > 0.10)
        ),
    }

    if "covariance_median_operator_error" in primary.columns:
        gates["M8_coordinate_covariance_median_error_lt_0_08"] = bool(
            np.all(primary["covariance_median_operator_error"] < 0.08)
        )
    if "radius_scan_stable_points" in primary.columns:
        gates["M9_at_least_two_stable_radii_everywhere"] = bool(
            np.all(primary["radius_scan_stable_points"] >= 2)
        )

    gates["ALL_MINIMAL_CLOSURE_GATES_PASS"] = bool(all(gates.values()))

    payload = {
        "scope": (
            "Original ideal interacting Pulser/QutipEmulator 5→7→9 model, "
            "extended only by multi-seed, radius, independent-optimizer, "
            "and coordinate-covariance audits."
        ),
        "atoms": atoms,
        "seeds": seeds,
        "radii": radii,
        "preset": args.preset,
        "primary_summary": primary.to_dict(orient="records"),
        "gates": gates,
        "elapsed_seconds": time.time() - t0,
    }
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    _plot_multiseed(primary, out)

    print("\nMULTI-SEED SCALING SUMMARY")
    columns = [
        "n_atoms",
        "seed",
        "simulator_calls_identification",
        "recovery_median_fro_error",
        "recovery_median_alignment",
        "offdiagonal_frobenius_fraction",
        "full_r2",
        "full_spearman",
        "full_mae",
        "diag_mae_ratio",
        "scalar_mae_ratio",
        "rotated_mae_ratio",
    ]
    optional = [
        "covariance_median_operator_error",
        "covariance_median_alignment",
        "radius_scan_stable_points",
        "gf_lm_median_final_loss",
        "gf_lm_median_calls",
        "powell_independent_median_final_loss",
        "powell_independent_median_calls",
    ]
    columns.extend([name for name in optional if name in primary.columns])

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        300,
    ):
        print(primary[columns].to_string(index=False))

    print("\nPREDECLARED MINIMAL-CLOSURE GATES")
    print(json.dumps(gates, indent=2))
    print(f"\nElapsed: {payload['elapsed_seconds']:.2f} s")
    print(f"Outputs: {out}")
    print(
        "\nInterpretation boundary: these extensions test robustness of the "
        "original local Pulser/QutipEmulator result. They do not establish "
        "a global geodesic law or a universal computation-as-flow theorem."
    )


if __name__ == "__main__":
    main()
