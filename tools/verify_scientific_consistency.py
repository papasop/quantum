#!/usr/bin/env python3
"""Verify report-derived scientific summary surfaces stay synchronized."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "v0.6.0"
README = ROOT / "README.md"
TEX = ROOT / "paper" / "response_fibre_fault_tolerance_v0_6_2.tex"
PDF = ROOT / "paper" / "response_fibre_fault_tolerance_v0_6_2.pdf"
CLAIM = RESULTS / "claim_certificate.json"
PROTOCOL = RESULTS / "protocol.json"
REPORT_GZ = RESULTS / "report.json.gz"
WILSON_CSV = RESULTS / "wilson_distance_table.csv"

EXPECTED_PROTOCOL_SHA256 = "fec91e30001712f3d9ac84c0e45a6b70f2d5ae7189d3c9ac6d1096d47505cbf6"
EXPECTED_REPORT_CERTIFICATE = "2db9620419ac5a7ff64510c65e0d391c4603b6c361fdd8aadd2d9f96165cbc79"
EXPECTED_REPORT_GZ_SHA256 = "f9edf8692aaa0f116cc6584507e7f326d184831f251669aa6e2dd2dd143bb95a"
EXPECTED_CLAIM_CERTIFICATE = "e01064c5772eff865653ad66f9a9cf466c4222cb08df95ee30f2926c3cf42ae9"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_report() -> dict[str, Any]:
    with gzip.open(REPORT_GZ, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: float, expected: float, tol: float = 1e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=tol, abs_tol=tol)


def verify_hashes(report: dict[str, Any]) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    require(sha256_bytes(canonical_json(protocol)) == EXPECTED_PROTOCOL_SHA256, "protocol hash mismatch")
    require(report["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256, "report protocol hash mismatch")

    report_without_self = dict(report)
    report_certificate = report_without_self.pop("certificate_sha256_before_self_field")
    require(sha256_bytes(canonical_json(report_without_self)) == report_certificate, "report certificate recompute failed")
    require(report_certificate == EXPECTED_REPORT_CERTIFICATE, "unexpected report certificate")
    require(sha256_bytes(REPORT_GZ.read_bytes()) == EXPECTED_REPORT_GZ_SHA256, "report.json.gz hash mismatch")

    claim = json.loads(CLAIM.read_text(encoding="utf-8"))
    claim_without_self = dict(claim)
    claim_certificate = claim_without_self.pop("claim_certificate_sha256_before_self_field")
    require(sha256_bytes(canonical_json(claim_without_self)) == claim_certificate, "claim certificate recompute failed")
    require(claim_certificate == EXPECTED_CLAIM_CERTIFICATE, "unexpected claim certificate")


def verify_report_values(report: dict[str, Any]) -> None:
    require(len(report["cases"]) == 15, "expected 15 cases")
    require(len(report["seed_crossovers"]) == 3, "expected 3 seed crossovers")
    require(report["secondary_fixed_distance_cases_passing"] == 15, "expected 15 fixed-distance passes")
    require(report["crossover_seeds_passing"] == 3, "expected 3 crossover passes")
    require(close(report["minimum_relative_decoded_failure_reduction"], 0.1786430350741669), "minimum reduction mismatch")
    require(close(report["minimum_z_score"], 24.858621249936743), "minimum z mismatch")
    require(close(report["minimum_resolved_physical_qubit_round_saving"], 0.45341380611090154), "proxy saving mismatch")
    for crossover in report["seed_crossovers"]:
        require(crossover["minimum_reference_distance"] == 11, "reference crossover distance mismatch")
        require(crossover["minimum_optimised_distance"] == 9, "optimised crossover distance mismatch")


def verify_derived_text(report: dict[str, Any]) -> None:
    readme = README.read_text(encoding="utf-8")
    tex = TEX.read_text(encoding="utf-8")
    require("17.86%" in readme, "README missing current minimum reduction")
    require("17.86\\%" in tex, "LaTeX missing current minimum reduction")
    for text, name in ((readme, "README"), (tex, "LaTeX")):
        require("24.86" in text, f"{name} missing current minimum z")
        require("11" in text and "9" in text, f"{name} missing crossover distances")
        require(EXPECTED_PROTOCOL_SHA256 in text, f"{name} missing protocol hash")
        require(EXPECTED_REPORT_CERTIFICATE in text, f"{name} missing report certificate")
        require(EXPECTED_REPORT_GZ_SHA256 in text, f"{name} missing report gzip hash")
    require("17.95%" not in readme, "README contains stale reduction")
    require("23.65" not in readme, "README contains stale z score")
    require("response_fibre_fault_tolerance_v0_6_1.pdf" not in readme, "README references stale PDF")


def verify_wilson_csv(report: dict[str, Any]) -> None:
    rows = list(csv.DictReader(WILSON_CSV.open(newline="", encoding="utf-8")))
    require(len(rows) == 30, "Wilson CSV should have 30 rows")
    keyed = {(int(row["seed"]), int(row["distance"]), row["arm"]): row for row in rows}
    cases = {(case["seed"], case["distance"]): case for case in report["cases"]}
    for crossover in report["seed_crossovers"]:
        for item in crossover["distance_table"]:
            case = cases[(crossover["seed"], item["distance"])]
            for arm, prefix in (("reference", "initial"), ("optimised", "optimised")):
                row = keyed[(crossover["seed"], item["distance"], arm)]
                require(int(row["failure_count"]) == case[prefix]["failure_count"], "CSV failure count mismatch")
                require(int(row["shots"]) == case[prefix]["shots"], "CSV shot count mismatch")


def find_latex_engine() -> list[str]:
    override = os.environ.get("TECTONIC_BIN")
    if override:
        return [override, "--outdir"]
    tectonic = shutil.which("tectonic")
    if tectonic:
        return [tectonic, "--outdir"]
    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        return [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory"]
    raise AssertionError("no LaTeX engine found; install tectonic or pdflatex")


def extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise AssertionError("pypdf is required for PDF text checks") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def verify_pdf() -> None:
    engine = find_latex_engine()
    with tempfile.TemporaryDirectory() as tmp:
        outdir = Path(tmp)
        command = engine + [str(outdir), str(TEX)]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        require(result.returncode == 0, "LaTeX compile failed:\n" + result.stdout + result.stderr)
        compiled = outdir / "response_fibre_fault_tolerance_v0_6_2.pdf"
        require(compiled.is_file(), "compiled PDF missing")
        committed_text = extract_pdf_text(PDF)
        compiled_text = extract_pdf_text(compiled)
        for needle in (
            "Schedule Response",
            "Synthetic Rotated-Surface-Code",
            "17.86%",
            "24.86",
            EXPECTED_REPORT_CERTIFICATE,
            "out-of-model validation",
        ):
            require(needle in committed_text, f"committed PDF missing {needle}")
            require(needle in compiled_text, f"compiled PDF missing {needle}")


def main() -> None:
    report = load_report()
    verify_hashes(report)
    verify_report_values(report)
    verify_derived_text(report)
    verify_wilson_csv(report)
    verify_pdf()
    print("SCIENTIFIC CONSISTENCY PASS")


if __name__ == "__main__":
    main()
