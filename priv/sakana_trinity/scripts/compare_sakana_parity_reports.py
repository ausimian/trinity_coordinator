#!/usr/bin/env python3
"""Compare Python and Elixir Sakana SVD parity reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python_report", type=Path)
    parser.add_argument("elixir_report", type=Path)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def collect_hashes(report: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key in ["variants", "native_elixir_svd_variants", "semantic_python_component_variants"]:
        value = report.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "label" in item and "observed_bf16_sha256" in item:
                    hashes[str(item["label"])] = str(item["observed_bf16_sha256"])
    return hashes


def main() -> None:
    args = parse_args()
    py = load(args.python_report)
    ex = load(args.elixir_report)
    expected = (
        py.get("reference", {}).get("expected_bf16_sha256")
        or ex.get("reference", {}).get("expected_bf16_sha256")
    )
    py_hashes = collect_hashes(py)
    ex_hashes = collect_hashes(ex)

    print(f"expected: {expected}")
    print("\nPython variants:")
    for label, digest in py_hashes.items():
        print(f"  {label}: {digest} match={digest == expected}")

    print("\nElixir variants:")
    for label, digest in ex_hashes.items():
        print(f"  {label}: {digest} match={digest == expected}")

    print("\nCross-report identical hashes:")
    any_match = False
    for py_label, py_digest in py_hashes.items():
        for ex_label, ex_digest in ex_hashes.items():
            if py_digest == ex_digest:
                any_match = True
                print(f"  {py_label} == {ex_label}: {py_digest}")
    if not any_match:
        print("  (none)")


if __name__ == "__main__":
    main()
