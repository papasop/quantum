#!/usr/bin/env python3
"""Verify cross-file scientific consistency for the v0.6.2 release."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v0.6.0"
PAPER = ROOT / "paper" / "v0.6.2"
PDF = PAPER / "response_fibre_fault_tolerance_v0_6_2.pdf"
TEX = PAPER / "response_fibre_fault_tolerance_v0_6_2.tex"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def pdf_text(path: Path) -> tuple[int, str, str]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise SystemExit("pypdf is required for PDF consistency checks") from exc
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    creator = str((reader.metadata or {}).get("/Creator", ""))
    return len(reader.pages), creator, text


def verify_pdf_artifact(failures: list[str]) -> None:
    pages, creator, text = pdf_text(PDF)
    require(pages == 8, "v0.6.2 PDF must have 8 pages", failures)
    require("LaTeX" in creator, "v0.6.2 PDF is not LaTeX-generated", failures)
    for needle in (
        "Response-Fibre Schedule Optimization",
        "Minimum Tested Surface-Code Distance",
        "17.86",
        "11",
        "9",
        "Wilson",
        "Synthetic Fault Model",
    ):
        require(needle in text, f"v0.6.2 PDF text missing: {needle}", failures)


def maybe_compile_tex(failures: list[str]) -> None:
    tectonic = os.environ.get("TECTONIC_BIN")
    if not tectonic:
        return
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        result = subprocess.run(
            [tectonic, "--outdir", str(outdir), str(TEX)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        require(
            result.returncode == 0,
            "LaTeX compile failed:\n" + result.stdout + result.stderr,
            failures,
        )
        compiled = outdir / "response_fibre_fault_tolerance_v0_6_2.pdf"
        require(compiled.is_file(), "compiled v0.6.2 PDF missing", failures)
        if compiled.is_file():
            pages, creator, text = pdf_text(compiled)
            require(pages == 8, "compiled v0.6.2 PDF must have 8 pages", failures)
            require("LaTeX" in creator, "compiled v0.6.2 PDF is not LaTeX-generated", failures)
            require("17.86" in text and "Wilson" in text, "compiled PDF text is missing result markers", failures)


def main() -> None:
    failures: list[str] = []

    protocol = json.loads((RESULTS / "protocol.json").read_text(encoding="utf-8"))
    claim = json.loads((RESULTS / "claim_certificate.json").read_text(encoding="utf-8"))
    report_gz = (RESULTS / "report.json.gz").read_bytes()
    with gzip.open(RESULTS / "report.json.gz", "rt", encoding="utf-8") as stream:
        report = json.load(stream)

    protocol_hash = sha256_bytes(canonical_bytes(protocol))
    require(
        protocol_hash == claim["protocol_sha256"] == report["protocol_sha256"],
        "protocol canonical hash mismatch",
        failures,
    )

    report_without_self = dict(report)
    report_self = report_without_self.pop("certificate_sha256_before_self_field")
    report_hash = sha256_bytes(canonical_bytes(report_without_self))
    require(report_hash == report_self, "report self hash mismatch", failures)
    require(
        report_hash == claim["reference_report_certificate_sha256_before_self_field"],
        "claim certificate references a different report",
        failures,
    )
    require(
        sha256_bytes(report_gz) == claim["report_json_gz_sha256"],
        "compressed report byte hash mismatch",
        failures,
    )

    claim_without_self = dict(claim)
    claim_self = claim_without_self.pop("claim_certificate_sha256_before_self_field")
    require(
        sha256_bytes(canonical_bytes(claim_without_self)) == claim_self,
        "claim certificate self hash mismatch",
        failures,
    )

    outcome = claim["reported_outcome"]
    exact_pairs = {
        "fixed_distance_cases_declared": report["cases_declared"],
        "fixed_distance_cases_passing": report["secondary_fixed_distance_cases_passing"],
        "crossover_seeds_declared": report["seeds_declared"],
        "crossover_seeds_passing": report["crossover_seeds_passing"],
        "minimum_relative_decoded_failure_reduction": report["minimum_relative_decoded_failure_reduction"],
        "minimum_z_score": report["minimum_z_score"],
    }
    for key, expected in exact_pairs.items():
        require(outcome[key] == expected, f"reported outcome mismatch: {key}", failures)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    paper_tex = TEX.read_text(encoding="utf-8")
    required_hashes = (
        protocol_hash,
        report_hash,
        claim["report_json_gz_sha256"],
        claim_self,
    )
    for value in required_hashes:
        require(value in readme, f"README missing evidence hash: {value}", failures)
        require(value in paper_tex, f"paper missing evidence hash: {value}", failures)

    require("17.86%" in readme, "README minimum reduction is stale", failures)
    require("24.86" in readme, "README minimum z-score is stale", failures)

    row_pattern = re.compile(
        r"^\s*(\d{8}|)\s*&\s*(3|5|7|9|11)\s*&\s*"
        r"([0-9.]+)\s*&\s*([0-9.]+)\s*&\s*([0-9.]+)\s*&\s*"
        r"([0-9.]+)\s*\\\\$",
        re.MULTILINE,
    )
    rows: list[tuple[int, int, float, float, float, float]] = []
    current_seed: int | None = None
    for match in row_pattern.finditer(paper_tex):
        if match.group(1):
            current_seed = int(match.group(1))
        if current_seed is None:
            failures.append("paper result table begins without a seed")
            break
        rows.append((current_seed, int(match.group(2)), *map(float, match.groups()[2:])))
    require(len(rows) == 15, "paper result table does not contain 15 rows", failures)

    cases = {(case["seed"], case["distance"]): case for case in report["cases"]}
    for seed, distance, ref, flow, reduction, z_score in rows:
        case = cases.get((seed, distance))
        if case is None:
            failures.append(f"paper row absent from report: seed={seed}, d={distance}")
            continue
        expected = (
            100 * case["initial"]["decoded_logical_failure_probability"],
            100 * case["optimised"]["decoded_logical_failure_probability"],
            100 * case["difference_statistics"]["relative_failure_reduction"],
            case["difference_statistics"]["z_score"],
        )
        tolerances = (5e-5, 5e-5, 5e-3, 5e-4)
        for label, actual, target, tolerance in zip(
            ("reference", "flow", "reduction", "z-score"),
            (ref, flow, reduction, z_score),
            expected,
            tolerances,
        ):
            require(
                math.isclose(actual, target, rel_tol=0.0, abs_tol=tolerance),
                f"paper {label} mismatch: seed={seed}, d={distance}",
                failures,
            )

    checker = subprocess.run(
        [sys.executable, "verify_wilson.py"],
        cwd=PAPER,
        text=True,
        capture_output=True,
        check=False,
    )
    require(checker.returncode == 0, "paper Wilson checker failed", failures)
    verify_pdf_artifact(failures)
    maybe_compile_tex(failures)

    if failures:
        raise SystemExit("SCIENTIFIC CONSISTENCY FAILED\n" + "\n".join(failures))
    print(
        "SCIENTIFIC CONSISTENCY PASS: protocol, report, certificate, README, "
        "paper table, PDF, figure data, and Wilson crossover agree."
    )


if __name__ == "__main__":
    main()
