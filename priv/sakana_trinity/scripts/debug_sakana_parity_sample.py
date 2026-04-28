#!/usr/bin/env python3
"""Emit side-by-side Python checkpoints for the Sakana/TRINITY SVD sample hash.

This script intentionally focuses on the single sample stored in
priv/sakana_trinity/reference/sakana_python_reference_manifest.json.  It emits
JSON with intermediate hashes/stats for:

* router-vector split and sample scale offsets,
* source tensor dtype/shape/hash,
* torch.svd and torch.linalg.svd variants,
* zero-offset reconstruction error,
* scaled singular-value normalization,
* final target-layout bf16 bytes.

It can also write a minimal semantic-component directory that the Elixir parity
Mix task can read with --components-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_REFERENCE = Path("priv/sakana_trinity/reference/sakana_python_reference_manifest.json")
DEFAULT_ROUTER_VECTOR_NPY = Path("priv/sakana_trinity/artifacts/sakana_model_iter_60.npy")
DEFAULT_OUT = Path("tmp/sakana_parity/python_sample_trace.json")
DEFAULT_COMPONENT_DIR = Path("tmp/sakana_parity/python_components")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--router-vector", type=Path, default=DEFAULT_ROUTER_VECTOR_NPY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write-components-dir", type=Path, default=DEFAULT_COMPONENT_DIR)
    parser.add_argument("--no-write-components", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--vector-dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--model-torch-dtype", default="auto", choices=["auto", "float32", "bfloat16"])
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    t = tensor.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()
    if t.dtype == torch.float16:
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor_bytes(tensor))


def tensor_prefix(tensor: torch.Tensor, count: int = 8) -> list[float]:
    flat = tensor.detach().to(torch.float32).cpu().reshape(-1)
    n = min(count, flat.numel())
    return [float(x) for x in flat[:n].tolist()]


def finite(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return float(value)


def tensor_summary(tensor: torch.Tensor, prefix_count: int = 8, alt_hashes: bool = True) -> dict[str, Any]:
    t32 = tensor.detach().to(torch.float32)
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "size": int(tensor.numel()),
        "sha256": tensor_sha256(tensor),
        "min": finite(float(t32.min().item())),
        "max": finite(float(t32.max().item())),
        "sum": finite(float(t32.sum().item())),
        "prefix_f32": tensor_prefix(tensor, prefix_count),
    }
    if alt_hashes:
        result["sha256_as_f32"] = tensor_sha256(tensor.to(torch.float32))
        result["sha256_as_bf16"] = tensor_sha256(tensor.to(torch.bfloat16))
    return result


def orient_to_shape(tensor: torch.Tensor, shape: list[int], label: str) -> torch.Tensor:
    target = tuple(int(x) for x in shape)
    if tuple(tensor.shape) == target:
        return tensor
    if tensor.ndim == 2 and tuple(tensor.T.shape) == target:
        return tensor.T.contiguous()
    raise ValueError(f"cannot orient {label} from {tuple(tensor.shape)} to {target}")


def sanitize_key(key: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "_.-") else "__" for ch in key.replace("/", "__"))


def load_router_vector(path: Path, dtype: str) -> np.ndarray:
    if path.suffix == ".npy":
        vector = np.load(path)
    else:
        from safetensors.numpy import load_file

        loaded = load_file(str(path))
        if "trinity_router_es_vector" not in loaded:
            raise KeyError(f"trinity_router_es_vector not found in {path}; keys={list(loaded)}")
        vector = loaded["trinity_router_es_vector"]

    return vector.astype(np.float32 if dtype == "float32" else np.float64, copy=False)


def reconstruct_from_torch_svd(u: torch.Tensor, s: torch.Tensor, v: torch.Tensor, offsets: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    offsets = offsets.to(dtype=s.dtype, device=s.device)
    scaled_s = s * (1.0 + offsets)
    normalization = s.sum() / scaled_s.sum()
    reconstructed = (u * scaled_s.reshape(1, -1)) @ v.T
    reconstructed = reconstructed * normalization
    return reconstructed, singular_summary(s, offsets, scaled_s, normalization)


def reconstruct_from_linalg_svd(u: torch.Tensor, s: torch.Tensor, vh: torch.Tensor, offsets: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    offsets = offsets.to(dtype=s.dtype, device=s.device)
    scaled_s = s * (1.0 + offsets)
    normalization = s.sum() / scaled_s.sum()
    reconstructed = (u * scaled_s.reshape(1, -1)) @ vh
    reconstructed = reconstructed * normalization
    return reconstructed, singular_summary(s, offsets, scaled_s, normalization)


def singular_summary(s: torch.Tensor, offsets: torch.Tensor, scaled_s: torch.Tensor, normalization: torch.Tensor) -> dict[str, Any]:
    return {
        "singular_values": tensor_summary(s, 16),
        "typed_offsets": tensor_summary(offsets, 16),
        "scaled_s": tensor_summary(scaled_s, 16),
        "sum_s": finite(float(s.detach().to(torch.float32).sum().item())),
        "sum_scaled_s": finite(float(scaled_s.detach().to(torch.float32).sum().item())),
        "normalization": finite(float(normalization.detach().to(torch.float32).item())),
    }


def max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().to(torch.float32) - right.detach().to(torch.float32)).abs().max().item())


def variant_report(label: str, reconstructed: torch.Tensor, singular: dict[str, Any], source_f32: torch.Tensor, sample: dict[str, Any]) -> dict[str, Any]:
    zero_reconstructed = reconstructed["zero"]
    adapted_reconstructed = reconstructed["adapted"]
    final = orient_to_shape(adapted_reconstructed.to(torch.bfloat16), sample["sample_reconstructed_shape"], label)
    observed = tensor_sha256(final)
    expected = sample["sample_reconstructed_bf16_sha256"]
    return {
        "label": label,
        "zero_offset_max_abs_error_vs_source": max_abs_error(zero_reconstructed, source_f32),
        "s": singular,
        "final": tensor_summary(final, 16),
        "observed_bf16_sha256" : observed,
        "expected_bf16_sha256": expected,
        "matches_expected": observed == expected,
    }


def main() -> None:
    args = parse_args()
    reference = json.loads(args.reference.read_text())
    sample = reference["sample_adapted_tensor"]
    vector = load_router_vector(args.router_vector, args.vector_dtype)
    offset_start = int(sample["offset_start"])
    offset_end = int(sample["offset_end"])
    offset_np = vector[offset_start:offset_end].copy()
    offsets = torch.from_numpy(offset_np).to(args.device)

    dtype_arg: Any
    if args.model_torch_dtype == "auto":
        dtype_arg = "auto"
    elif args.model_torch_dtype == "float32":
        dtype_arg = torch.float32
    else:
        dtype_arg = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype_arg)
    state_dict = model.state_dict()
    source_name = sample["source_name"]
    if source_name not in state_dict:
        raise KeyError(f"{source_name!r} missing; available sample keys={list(state_dict)[:20]}")

    source = state_dict[source_name].detach().to(args.device)
    source_f32 = source.to(torch.float32)

    # torch.svd matches the legacy Sakana decompose_model.py convention most closely.
    u_svd, s_svd, v_svd = torch.svd(source_f32)
    zeros = torch.zeros_like(s_svd)
    zero_svd, singular_zero_svd = reconstruct_from_torch_svd(u_svd, s_svd, v_svd, zeros)
    adapted_svd, singular_svd = reconstruct_from_torch_svd(u_svd, s_svd, v_svd, offsets.to(torch.float32))

    # torch.linalg.svd emits Vh directly.  This exposes whether the reference was produced
    # from modern linalg output or legacy torch.svd output.
    u_linalg, s_linalg, vh_linalg = torch.linalg.svd(source_f32, full_matrices=False)
    zero_linalg, _singular_zero_linalg = reconstruct_from_linalg_svd(u_linalg, s_linalg, vh_linalg, torch.zeros_like(s_linalg))
    adapted_linalg, singular_linalg = reconstruct_from_linalg_svd(u_linalg, s_linalg, vh_linalg, offsets.to(torch.float32))

    variants = [
        variant_report(
            "python_torch_svd_v_transposed_final_bf16",
            {"zero": zero_svd, "adapted": adapted_svd},
            singular_svd,
            source_f32,
            sample,
        ),
        variant_report(
            "python_linalg_svd_vh_final_bf16",
            {"zero": zero_linalg, "adapted": adapted_linalg},
            singular_linalg,
            source_f32,
            sample,
        ),
    ]

    if not args.no_write_components:
        args.write_components_dir.mkdir(parents=True, exist_ok=True)
        safe = sanitize_key(source_name)
        save_file(
            {
                f"svd.U.{safe}": u_svd.detach().cpu().contiguous(),
                f"svd.S.{safe}": s_svd.detach().cpu().contiguous(),
                f"svd.V.{safe}": v_svd.detach().cpu().contiguous(),
            },
            str(args.write_components_dir / "trinity_svf_components.safetensors"),
        )
        save_file(
            {f"svf.scale_offsets.{safe}": offsets.detach().cpu().contiguous()},
            str(args.write_components_dir / "trinity_svf_scale_offsets.safetensors"),
        )

    report = {
        "schema": "trinity_sakana_python_svd_parity_trace.v1",
        "reference": {
            "path": str(args.reference),
            "source_name": sample["source_name"],
            "elixir_name": sample["elixir_name"],
            "source_shape": sample["source_shape"],
            "sample_reconstructed_shape": sample["sample_reconstructed_shape"],
            "expected_bf16_sha256": sample["sample_reconstructed_bf16_sha256"],
        },
        "inputs": {
            "model_name": args.model_name,
            "model_torch_dtype_arg": args.model_torch_dtype,
            "source_tensor_dtype": str(source.dtype).replace("torch.", ""),
            "router_vector": str(args.router_vector),
            "router_vector_sha256": sha256_file(args.router_vector),
            "router_vector_dtype_after_load": str(vector.dtype),
            "write_components_dir": None if args.no_write_components else str(args.write_components_dir),
        },
        "source_tensor": tensor_summary(source, 16),
        "source_tensor_f32_svd_input": tensor_summary(source_f32, 16),
        "scale_offsets": tensor_summary(offsets, 16),
        "variants": variants,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote Python parity report: {args.out}")
    if not args.no_write_components:
        print(f"wrote sample Python components: {args.write_components_dir}")
    for variant in variants:
        print(
            f"{variant['label']}: {variant['observed_bf16_sha256']} "
            f"match={variant['matches_expected']} zero_error={variant['zero_offset_max_abs_error_vs_source']}"
        )


if __name__ == "__main__":
    main()
