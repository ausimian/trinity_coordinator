#!/usr/bin/env python3
"""Export Sakana Trinity coordinator artifacts to Elixir-friendly safetensors.

This script is the semantic export path. It uses the same core convention as
Sakana's supplementary Python runtime:

1. SVD components come from Qwen model weights decomposed into U/S/V tensors.
2. The trained ES vector is consumed in model state_dict order:
   selected SVD scale offsets first, router head weights second.
3. The output files are safetensors plus a JSON manifest for Elixir/Nx import.

It does not evaluate TRINITY or call providers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers import AutoModelForCausalLM


DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_VECTOR_PATH = Path("priv/sakana_trinity/artifacts/sakana_model_iter_60.npy")
DEFAULT_ES_LOG_PATH = Path("priv/sakana_trinity/reference/sakana_es_log.json")
DEFAULT_OUTPUT_DIR = Path("priv/sakana_trinity/artifacts/exported")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Sakana Trinity SVD/head artifacts to safetensors."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--router-vector", type=Path, default=DEFAULT_VECTOR_PATH)
    parser.add_argument("--es-log", type=Path, default=DEFAULT_ES_LOG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--svd-weights",
        type=Path,
        default=None,
        help=(
            "Existing svd_weights.pt. Defaults to "
            "<output-dir>/decomposed_models/<model>/svd_weights.pt"
        ),
    )
    parser.add_argument(
        "--decompose-if-missing",
        action="store_true",
        help="Generate svd_weights.pt from the base model if the file is missing.",
    )
    parser.add_argument(
        "--role-count",
        type=int,
        default=3,
        help="Number of Trinity role logits appended after agent logits.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Output dtype for scale/head tensors.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_es_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list) and data and "configs" in data[0]:
        return data[0]["configs"]

    if isinstance(data, dict) and "configs" in data:
        return data["configs"]

    if isinstance(data, dict):
        return data

    raise ValueError(f"Unsupported es_log structure in {path}")


def default_svd_path(output_dir: Path, model_name: str) -> Path:
    safe_model_name = model_name.replace("/", "_")
    return output_dir / "decomposed_models" / safe_model_name / "svd_weights.pt"


def decompose_model_to_svd(model_name: str, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    decomposed: dict[str, torch.Tensor] = {}

    for key, tensor in model.state_dict().items():
        if tensor.ndim > 1 and all(dim > 1 for dim in tensor.shape):
            u, s, v = torch.svd(tensor)
            decomposed[f"{key}.U"] = u.cpu()
            decomposed[f"{key}.S"] = s.cpu()
            decomposed[f"{key}.V"] = v.cpu()

    torch.save(decomposed, output_file)


def should_keep_svd_key(base_key: str, opt_layer_indices: list[int] | None) -> bool:
    if opt_layer_indices is None:
        return True

    if 999 in opt_layer_indices:
        return False

    if "model.layers." not in base_key:
        return True

    return any(f"model.layers.{idx}." in base_key for idx in opt_layer_indices)


def sanitize_key(key: str) -> str:
    return key.replace("/", "__")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    es_config = load_es_config(args.es_log)
    model_name = es_config.get("model_name", args.model_name) or args.model_name
    llm_names = es_config.get("llm_names", [])
    opt_layer_indices = es_config.get("opt_layer_indices")
    if opt_layer_indices is not None:
        opt_layer_indices = [int(idx) for idx in opt_layer_indices]

    svd_path = args.svd_weights or default_svd_path(args.output_dir, model_name)
    if not svd_path.exists():
        if not args.decompose_if_missing:
            raise FileNotFoundError(
                f"SVD weights not found at {svd_path}. "
                "Run with --decompose-if-missing or provide --svd-weights."
            )
        decompose_model_to_svd(model_name, svd_path)

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_state_keys = list(model.state_dict().keys())
    svd_weights = torch.load(svd_path, map_location="cpu")
    vector = np.load(args.router_vector)

    if vector.ndim != 1:
        raise ValueError(f"Expected 1D router vector, got shape {vector.shape}")

    output_count = len(llm_names) + args.role_count
    hidden_size = int(config.hidden_size)
    head_param_count = hidden_size * output_count
    scale_dtype = np.float32 if args.dtype == "float32" else np.float64
    vector = vector.astype(scale_dtype, copy=False)

    components: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    scale_entries: list[dict[str, Any]] = []
    offset = 0

    for full_key in model_state_keys:
        sv_key = f"{full_key}.S"
        if sv_key not in svd_weights:
            continue
        if not should_keep_svd_key(full_key, opt_layer_indices):
            continue

        singular_values = svd_weights[sv_key]
        scale_count = singular_values.numel()
        next_offset = offset + scale_count
        if next_offset > vector.shape[0]:
            raise ValueError(
                f"Vector ended while reading SVD scales for {full_key}: "
                f"need offset {next_offset}, vector length {vector.shape[0]}"
            )

        safe_base = sanitize_key(full_key)
        scale_chunk = torch.from_numpy(vector[offset:next_offset].copy())
        scales[f"svf.scale_offsets.{safe_base}"] = scale_chunk

        components[f"svd.U.{safe_base}"] = svd_weights[f"{full_key}.U"].cpu().contiguous()
        components[f"svd.S.{safe_base}"] = svd_weights[f"{full_key}.S"].cpu().contiguous()
        components[f"svd.V.{safe_base}"] = svd_weights[f"{full_key}.V"].cpu().contiguous()

        scale_entries.append(
            {
                "source_parameter": full_key,
                "safe_parameter": safe_base,
                "scale_tensor": f"svf.scale_offsets.{safe_base}",
                "component_tensors": {
                    "u": f"svd.U.{safe_base}",
                    "s": f"svd.S.{safe_base}",
                    "v": f"svd.V.{safe_base}",
                },
                "offset_start": offset,
                "offset_end": next_offset,
                "num_singular_values": scale_count,
                "shape": list(singular_values.shape),
            }
        )
        offset = next_offset

    head_start = offset
    head_end = head_start + head_param_count
    if head_end != vector.shape[0]:
        raise ValueError(
            "Unexpected router vector length after SVD split: "
            f"svd_count={head_start}, head_count={head_param_count}, "
            f"expected_total={head_end}, actual_total={vector.shape[0]}"
        )

    head = torch.from_numpy(vector[head_start:head_end].copy()).reshape(
        output_count,
        hidden_size,
    )
    head_tensors = {"trinity.router_head.linear.weight": head}

    component_path = args.output_dir / "trinity_svf_components.safetensors"
    scales_path = args.output_dir / "trinity_svf_scale_offsets.safetensors"
    head_path = args.output_dir / "trinity_router_head.safetensors"
    manifest_path = args.output_dir / "trinity_sakana_export_manifest.json"

    save_file(components, str(component_path))
    save_file(scales, str(scales_path))
    save_file(head_tensors, str(head_path))

    manifest = {
        "format": "trinity_sakana_safetensors_export",
        "version": 1,
        "source": {
            "model_name": model_name,
            "router_vector": str(args.router_vector),
            "router_vector_sha256": sha256_file(args.router_vector),
            "es_log": str(args.es_log),
            "es_log_sha256": sha256_file(args.es_log),
            "svd_weights": str(svd_path),
            "svd_weights_sha256": sha256_file(svd_path),
        },
        "model": {
            "hidden_size": hidden_size,
            "opt_layer_indices": opt_layer_indices,
        },
        "routing": {
            "agent_labels": llm_names,
            "role_labels": ["solver", "thinker", "verifier"][: args.role_count],
            "output_count": output_count,
            "head_type": "linear",
            "head_tensor": "trinity.router_head.linear.weight",
            "head_shape": [output_count, hidden_size],
            "head_offset_start": head_start,
            "head_offset_end": head_end,
        },
        "svf": {
            "scale_parameter_count": head_start,
            "entries": scale_entries,
            "reconstruction_formula": (
                "U @ diag(S * (1 + scale_offsets)) @ V.T, then multiply by "
                "sum(S) / sum(S * (1 + scale_offsets))"
            ),
        },
        "outputs": {
            "components": str(component_path),
            "scale_offsets": str(scales_path),
            "head": str(head_path),
        },
    }

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Wrote {component_path}")
    print(f"Wrote {scales_path}")
    print(f"Wrote {head_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
