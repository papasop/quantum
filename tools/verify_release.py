#!/usr/bin/env python3
"""Verify repository hashes and compile every versioned audit."""

from __future__ import annotations

import hashlib
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.txt"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    failures: list[str] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif digest(path) != expected:
            failures.append(f"hash mismatch: {relative}")

    sources = sorted((ROOT / "src").glob("*.py"))
    for source in sources:
        try:
            py_compile.compile(str(source), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"compile failure: {source.name}: {exc}")

    if failures:
        raise SystemExit("VERIFY FAILED\n" + "\n".join(failures))
    print(f"VERIFY PASS: {len(sources)} Python audits compiled and all manifest hashes match.")


if __name__ == "__main__":
    main()
