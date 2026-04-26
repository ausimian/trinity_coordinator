#!/usr/bin/env python3
"""Convert Sakana's raw Trinity router vector from .npy to safetensors.

This is a dumb format conversion only. It preserves the full vector as one
tensor and does not split it into SVD scale offsets and router head weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


DEFAULT_TENSOR_NAME = "trinity_router_es_vector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Sakana Trinity model_iter_*.npy vector to safetensors."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("priv/sakana_trinity/artifacts/sakana_model_iter_60.npy"),
        help="Input .npy router vector.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("priv/sakana_trinity/artifacts/trinity_router_es_vector.safetensors"),
        help="Output safetensors file.",
    )
    parser.add_argument(
        "--tensor-name",
        default=DEFAULT_TENSOR_NAME,
        help=f"Tensor key to write inside the safetensors file. Default: {DEFAULT_TENSOR_NAME}",
    )
    parser.add_argument(
        "--float32",
        action="store_true",
        help="Cast the vector to float32 before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vector = np.load(args.input)

    if vector.ndim != 1:
        raise ValueError(f"Expected a 1D router vector, got shape {vector.shape}")

    if args.float32:
        vector = vector.astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({args.tensor_name: np.ascontiguousarray(vector)}, str(args.output))
    print(
        f"Wrote {args.output} with tensor {args.tensor_name} "
        f"shape={vector.shape} dtype={vector.dtype}"
    )


if __name__ == "__main__":
    main()

