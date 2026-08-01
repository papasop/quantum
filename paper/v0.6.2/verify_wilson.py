#!/usr/bin/env python3
"""Independent verification of Figure 1 Wilson upper bounds.

Recomputes the one-sided 95% Wilson upper bound from the decoded failure
counts (percent x N, N=10^6) and checks both the figure_data CSV values
and the frozen crossover claim  d*_ref=11 -> d*_flow=9.
"""
import csv
import math
import sys
from pathlib import Path

N = 1_000_000
Z = 1.6448536
TARGET = 0.025
HERE = Path(__file__).resolve().parent

def wilson_upper(k, n, z):
    ph = k / n
    denom = 1 + z * z / n
    centre = ph + z * z / (2 * n)
    margin = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return (centre + margin) / denom

ok = True
for seed in (20290107, 20290119, 20290203):
    path = HERE / f"figure_data_seed_{seed}.csv"
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    ref_pass, flow_pass = [], []
    for r in rows:
        d = int(r["d"])
        for arm in ("ref", "flow"):
            k = round(float(r[arm]) / 100 * N)
            recomputed = wilson_upper(k, N, Z)
            stored = float(r[f"wilson_{arm}"])
            err = abs(recomputed - stored)
            status = "OK" if err < 1e-7 else "MISMATCH"
            if err >= 1e-7:
                ok = False
            print(f"{seed} d={d:2d} {arm:4s} k={k:6d}  stored={stored:.8f}  recomputed={recomputed:.8f}  {status}")
            (ref_pass if arm == "ref" else flow_pass).append(d if recomputed <= TARGET else None)
    d_ref = min((d for d in ref_pass if d), default=None)
    d_flow = min((d for d in flow_pass if d), default=None)
    print(f"  -> seed {seed}: d*_ref={d_ref}, d*_flow={d_flow}  (claim: 11 -> 9)")
    if d_ref != 11 or d_flow != 9:
        ok = False

print("\nALL CHECKS PASSED" if ok else "\nCHECKS FAILED")
sys.exit(0 if ok else 1)
