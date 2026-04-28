#!/usr/bin/env python3
"""Compare Python and Elixir Sakana SVD parity reports.

The comparator intentionally distinguishes the historical stored reference hash
from the current Python baseline hash.  If the current Python report cannot
reproduce the stored reference hash, exact Elixir equality against that stored
hash is not a meaningful failure signal; compare Elixir to the current Python
baseline or provide the original svd_weights.pt to the Python debug script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python_report", type=Path)
    parser.add_argument("elixir_report", type=Path)
    parser.add_argument("--strict-reference", action="store_true",
                        help="Exit non-zero unless Python and Elixir both contain a variant matching the stored reference hash.")
    parser.add_argument("--strict-current-python", action="store_true",
                        help="Exit non-zero unless some Elixir variant matches the current Python baseline hash.")
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


def collect_zero_errors(report: dict[str, Any]) -> dict[str, Any]:
    errors: dict[str, Any] = {}
    for key in ["variants", "native_elixir_svd_variants", "semantic_python_component_variants"]:
        value = report.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "label" in item and "zero_offset_max_abs_error_vs_source" in item:
                    errors[str(item["label"])] = item["zero_offset_max_abs_error_vs_source"]
    return errors


def reference_hash(report: dict[str, Any]) -> str | None:
    return (
        report.get("reference", {}).get("expected_bf16_sha256")
        or report.get("reference", {}).get("expected_bf16_sha256")
    )


def current_python_baseline(py: dict[str, Any], py_hashes: dict[str, str]) -> tuple[str | None, str | None]:
    ref = py.get("reference", {})
    label = ref.get("current_python_baseline_label")
    digest = ref.get("current_python_baseline_bf16_sha256")
    if label and digest:
        return str(label), str(digest)
    if py_hashes:
        label, digest = next(iter(py_hashes.items()))
        return label, digest
    return None, None


def main() -> None:
    args = parse_args()
    py = load(args.python_report)
    ex = load(args.elixir_report)
    expected = reference_hash(py) or reference_hash(ex)
    py_hashes = collect_hashes(py)
    ex_hashes = collect_hashes(ex)
    py_errors = collect_zero_errors(py)
    ex_errors = collect_zero_errors(ex)
    baseline_label, baseline_digest = current_python_baseline(py, py_hashes)
    reproducible = bool(py.get("reference", {}).get("expected_hash_reproducible"))

    print(f"stored reference expected: {expected}")
    print(f"python reference hash reproducible: {reproducible}")
    print(f"current Python baseline: {baseline_label} {baseline_digest}")
    if not reproducible:
        print("note: current Python SVD did not reproduce the stored manifest hash; exact comparison to the stored hash is provenance-sensitive.")
        print("      Use --svd-weights with the original svd_weights.pt if strict historical reproduction is required.")

    print("\nPython variants:")
    for label, digest in py_hashes.items():
        print(f"  {label}: {digest} match_stored={digest == expected} zero_error={py_errors.get(label)}")

    print("\nElixir variants:")
    for label, digest in ex_hashes.items():
        print(
            f"  {label}: {digest} "
            f"match_stored={digest == expected} "
            f"match_current_python={digest == baseline_digest} "
            f"zero_error={ex_errors.get(label)}"
        )

    print("\nCross-report identical hashes:")
    any_match = False
    for py_label, py_digest in py_hashes.items():
        for ex_label, ex_digest in ex_hashes.items():
            if py_digest == ex_digest:
                any_match = True
                print(f"  {py_label} == {ex_label}: {py_digest}")
    if not any_match:
        print("  (none)")

    if args.strict_reference:
        py_ok = any(digest == expected for digest in py_hashes.values())
        ex_ok = any(digest == expected for digest in ex_hashes.values())
        if not (py_ok and ex_ok):
            raise SystemExit("strict stored-reference comparison failed")

    if args.strict_current_python:
        if not baseline_digest or not any(digest == baseline_digest for digest in ex_hashes.values()):
            raise SystemExit("strict current-Python comparison failed")


if __name__ == "__main__":
    main()
