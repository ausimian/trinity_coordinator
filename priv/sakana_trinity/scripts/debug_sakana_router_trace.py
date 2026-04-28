#!/usr/bin/env python3
"""Emit a fixed-transcript Python router trace for the adapted Qwen coordinator.

This script is intentionally runtime-facing, not SVD-facing. It loads the same
canonical Elixir artifact directory produced by `mix trinity.sakana.import_python`,
patches a Hugging Face Qwen model with the adapted tensors, extracts the
second-to-last-token hidden vector, applies the imported linear router head, and
writes a JSON trace that Elixir can compare stage by stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


DEFAULT_ARTIFACT_DIR = Path("tmp/sakana_parity/adapted_artifacts_from_python")
DEFAULT_OUT = Path("tmp/sakana_parity/python_router_trace.json")
DEFAULT_MESSAGE = "Select a TRINITY role for this reasoning task."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit Python Sakana router trace JSON.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--message", default=DEFAULT_MESSAGE)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--model-torch-dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Torch dtype used when loading the Qwen model.",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256_bytes(payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256_f32(tensor: torch.Tensor) -> str:
    array = tensor.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    return sha256_bytes(array.tobytes(order="C"))


def tensor_f32_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().float().cpu().flatten().tolist()]


def load_manifest(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "manifest.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def torch_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported torch dtype {name!r}")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def patch_model_from_artifact(
    model: torch.nn.Module,
    artifact_dir: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    state = model.state_dict()
    patched: list[dict[str, Any]] = []

    for entry in manifest["selected_tensors"]:
        source_name = entry["source_name"]
        if source_name not in state:
            raise KeyError(f"{source_name!r} not found in Hugging Face model state_dict")

        checkpoint_path = artifact_dir / entry["checkpoint_path"]
        tensor = load_file(str(checkpoint_path), device="cpu")[entry["artifact_key"]]
        target = state[source_name]

        if (
            source_name.startswith("model.layers.")
            and tensor.ndim == 2
            and tuple(tensor.T.shape) == tuple(target.shape)
        ):
            source_tensor = tensor.T.contiguous()
            orientation = "transpose"
        elif tuple(tensor.shape) == tuple(target.shape):
            source_tensor = tensor
            orientation = "as_is"
        elif tuple(tensor.T.shape) == tuple(target.shape):
            source_tensor = tensor.T.contiguous()
            orientation = "transpose"
        else:
            raise ValueError(
                f"cannot align {source_name}: artifact shape {tuple(tensor.shape)} "
                f"target shape {tuple(target.shape)}"
            )

        with torch.no_grad():
            target.copy_(source_tensor.to(dtype=target.dtype))

        patched.append(
            {
                "source_name": source_name,
                "artifact_key": entry["artifact_key"],
                "checkpoint_path": entry["checkpoint_path"],
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "artifact_shape": list(tensor.shape),
                "source_shape": list(target.shape),
                "orientation": orientation,
            }
        )

    return patched


def load_router_head(artifact_dir: Path, manifest: dict[str, Any]) -> torch.Tensor:
    path = artifact_dir / manifest["router_head_artifact"]
    tensors = load_file(str(path), device="cpu")
    return tensors[manifest["router_head_tensor_key"]].contiguous()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.artifact_dir)
    model_name = manifest.get("base_model_repo", "Qwen/Qwen3-0.6B")
    device = resolve_device(args.device)
    dtype = torch_dtype(args.model_torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()

    patched_tensors = patch_model_from_artifact(model, args.artifact_dir, manifest)
    model.to(device)

    messages = [{"role": "user", "content": args.message}]
    transcript = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
    encoded = tokenizer(transcript, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_state = outputs.hidden_states[-1]
    seq_len = int(hidden_state.shape[1])
    hidden_position = -2
    hidden_index = 0 if seq_len <= 1 else seq_len + hidden_position
    hidden_vector = hidden_state[:, hidden_index, :]

    head_weights = load_router_head(args.artifact_dir, manifest).to(device=device, dtype=torch.float32)
    route_input = hidden_vector.to(dtype=torch.float32)
    logits = route_input @ head_weights.T
    logits_1d = logits.squeeze(0)

    num_roles = 3
    output_count = int(logits_1d.shape[0])
    num_agents = output_count - num_roles
    agent_logits = logits_1d[:num_agents]
    role_logits = logits_1d[num_agents:]
    agent_id = int(torch.argmax(agent_logits).item())
    role_id = int(torch.argmax(role_logits).item())

    input_ids_list = [int(value) for value in input_ids.detach().cpu().flatten().tolist()]
    attention_shape = list(attention_mask.shape) if attention_mask is not None else None

    report = {
        "schema": "trinity_sakana_router_trace.v1",
        "runtime": "python.transformers",
        "artifact_dir": str(args.artifact_dir),
        "artifact_manifest_sha256": sha256_file(args.artifact_dir / "manifest.json"),
        "model_name": model_name,
        "model_torch_dtype": args.model_torch_dtype,
        "device": str(device),
        "messages": messages,
        "transcript": transcript,
        "transcript_sha256": sha256_bytes(transcript.encode("utf-8")),
        "input_ids": input_ids_list,
        "token_ids_sha256": sha256_json(input_ids_list),
        "attention_mask_shape": attention_shape,
        "hidden_state_shape": list(hidden_state.shape),
        "hidden_position": hidden_position,
        "hidden_index": hidden_index,
        "hidden_vector_shape": list(hidden_vector.shape),
        "hidden_vector_sha256_as_f32": tensor_sha256_f32(hidden_vector),
        "hidden_vector_prefix_f32": tensor_f32_list(hidden_vector[:, :8]),
        "hidden_vector_f32": tensor_f32_list(hidden_vector),
        "head_weight_shape": list(head_weights.shape),
        "head_weight_sha256_as_f32": tensor_sha256_f32(head_weights),
        "logits_shape": list(logits.shape),
        "logits_sha256_as_f32": tensor_sha256_f32(logits),
        "logits": tensor_f32_list(logits),
        "agent_logits": tensor_f32_list(agent_logits),
        "role_logits": tensor_f32_list(role_logits),
        "argmax_agent_id": agent_id,
        "argmax_role_id": role_id,
        "patched_tensors": patched_tensors,
        "notes": [
            "Transcript formatting intentionally mirrors TrinityCoordinator.Extractor.format_messages/1.",
            "The required router trace gate is exact token/head/argmax parity plus declared hidden/logit tolerances.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"wrote Python router trace: {args.out}")
    print(f"token_ids_sha256: {report['token_ids_sha256']}")
    print(f"hidden_vector_sha256_as_f32: {report['hidden_vector_sha256_as_f32']}")
    print(f"head_weight_sha256_as_f32: {report['head_weight_sha256_as_f32']}")
    print(f"logits_sha256_as_f32: {report['logits_sha256_as_f32']}")
    print(f"argmax_agent_id: {agent_id}")
    print(f"argmax_role_id: {role_id}")


if __name__ == "__main__":
    main()
