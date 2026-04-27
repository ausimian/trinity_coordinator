# Elixir-Native SVD Decomposition For TRINITY

This document is the implementation task list for replacing the Python
`decompose_model.py` dependency with Elixir/Nx SVD decomposition.

The goal is not to build a toy probe. The goal is to make the Qwen coordinator
path able to generate and consume SVD/SVF artifacts in Elixir, on the same GPU
runtime used by the coordinator.

## Target Contract

Given a loaded `Qwen/Qwen3-0.6B` Bumblebee model:

1. Select the same class of tensors Sakana decomposes: tensors with rank greater
   than one and every dimension greater than one.
2. Run SVD in Elixir with `Nx.LinAlg.svd/2`.
3. Store `U`, `S`, and `V` components in a deterministic manifest.
4. Reconstruct selected tensors using Sakana's SVF formula:

   ```text
   scaled_s = S * (1 + scale_offsets)
   reconstructed = U @ diag(scaled_s) @ V
   reconstructed = reconstructed * (sum(S) / sum(scaled_s))
   ```

5. Split the Sakana router vector into:
   - SVF scale offsets
   - router head weights
6. Apply the selected reconstructed tensors into the Bumblebee params tree.
7. Load router head weights into the Axon/Nx head.
8. Route from a real Qwen hidden state on `EXLA.Backend<cuda:...>`.

## Track Checklist

### Math Track

#### 1. SVD Tensor Math

- [x] Red: test decomposes a small `{m, n}` tensor into `{u, s, v}`.
- [x] Green: implement `TrinityCoordinator.Sakana.SVD.decompose_tensor/2`.
- [x] Red: test reconstructs exactly enough with zero scale offsets.
- [x] Green: implement `reconstruct/2` with Sakana's normalization formula.
- [x] QC: assert max absolute reconstruction error is below tolerance.
- [x] QC: assert tensor backend remains CUDA when input is CUDA.

#### 2. Sakana Selection Rule

- [x] Red: test rank-1 tensors and singleton-dimension tensors are skipped.
- [x] Green: implement `decomposable_tensor?/1`.
- [x] Red: test nested param containers flatten to deterministic paths.
- [x] Green: implement `flatten_tensors/1`.
- [x] Green: implement `flatten_tensor_entries/1` so original path segments are
      retained for future param-tree insertion.
- [x] QC: path ordering is stable across runs.

#### 3. Router Vector Loader

### Export Track

- [x] Red: load `trinity_router_es_vector.safetensors`.
- [x] Green: implement safetensors vector load helper.
- [x] Red: split vector into `9216` SVF scale offsets and `10240` head weights
      for the inspected Sakana run.
- [x] Green: implement split helper from manifest values, not magic constants.
- [x] QC: assert vector shape is `{19456}`.
- [x] QC: assert head shape is `{10, 1024}`.

#### 4. Qwen Structural Gate

- [x] Red: load `Qwen/Qwen3-0.6B` through `SLMProfile.qwen_coordinator/0`.
- [x] Green: select decomposable Qwen tensors from the loaded params tree.
- [x] QC: assert selected tensors are on `EXLA.Backend<cuda:...>`.
- [x] QC: assert Qwen hidden-state extraction still returns `{1, 1024}` on CUDA.

#### 5. Layer-26 SVF Gate

- [x] Red: identify layer-26 Qwen tensors in the Elixir/Bumblebee params tree.
- [x] Green: add explicit layer-filter support.
- [x] QC: assert selected singular-value count matches the Sakana vector prefix
      length: `9216`.
- [x] QC: produce a manifest that maps Elixir paths to selected tensor shapes.
- [x] QC: map a representative layer-26 tensor to its exact Sakana scale-offset
      span without running full-vocabulary SVD in the default Qwen gate.

#### 6. Sakana Vector Application

- [x] Red: apply scale offsets to selected decomposed tensors.
- [x] Green: reconstruct adapted tensors in Elixir.
- [x] Green: implement `put_tensor_entries/2` so adapted tensors can be placed
      back into nested params containers using preserved path segments.
- [x] QC: zero offsets reconstruct the original tensor within tolerance.
- [x] QC: non-zero offsets preserve original tensor shape and dtype.
- [x] QC: tensor insertion handles layer names containing dots without splitting
      the layer name incorrectly.
- [x] QC: representative Qwen tensor maps to a Sakana scale-offset span and the
      source tensor remains on CUDA.
- [x] QC: full `9216`-offset Qwen reconstruction is available as an explicit
      expensive gate, not as a default fast test.
- [ ] QC: run the explicit expensive gate on a clean GPU and record the wall
      time/VRAM outcome before treating full-backbone SVF import as routine.

#### 7. Router Head Application

### Runtime Track

- [x] Red: load the final `10240` vector values as a linear head.
- [x] Green: reshape to `{10, 1024}` and load into the existing Axon head path.
- [x] QC: route logits have shape `{1, 10}`.
- [x] QC: split route logits into `7` agent logits and `3` role logits.

#### 8. End-To-End Qwen Route Gate

- [x] Red: Qwen hidden state plus Sakana head vector produces route logits.
- [x] Green: route using real `CoordinationHead.route/5` semantics.
- [x] QC: backend assertion remains `EXLA.Backend<cuda:...>`.
- [x] QC: traces include profile, vector shape, logits shape, and backend.

## Hard Verification Commands

### Math

Fast math/loader tests:

```sh
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs
```

### Export

```sh
XLA_TARGET=cuda12 mix test --only qwen --trace
```

Focused SVD/Qwen gates used by this stream:

```sh
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs --exclude qwen --trace
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs --only qwen --trace
```

### Runtime

```sh
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs --only expensive_qwen_svd --trace
```

This gate performs SVD reconstruction across the complete Sakana-selected Qwen
tensor set for the inspected layer-26 vector, consumes all `9216` scale offsets,
and reinserts the reconstructed tensors into the Bumblebee params container. It
is intentionally opt-in because it can compile and execute large GPU SVD kernels.

Full local gate before claiming this stream complete:

```sh
XLA_TARGET=cuda12 mix format --check-formatted
XLA_TARGET=cuda12 mix test test/trinity_coordinator/sakana/svd_test.exs
XLA_TARGET=cuda12 mix test --only qwen --trace
```

## Correctness Standard

The Elixir implementation is correct when:

- SVD reconstruction with zero offsets reproduces the original tensor within
  documented numerical tolerance.
- The Sakana normalization factor is implemented exactly.
- The router vector is read from safetensors, not from Python at runtime.
- The selected scale prefix length is derived from selected SVD tensors and
  equals `9216` for the inspected Sakana run.
- The head suffix length is derived from `hidden_size * (num_agents + 3)` and
  equals `10240` for `Qwen3-0.6B` with seven agents.
- Qwen extraction and selected SVD tensors run on CUDA, not CPU.

## Non-Goals

- Do not parse PyTorch `.pt` in Elixir.
- Do not call Python at runtime.
- Do not claim paper-score reproduction from these checks alone.
- Do not use CPU-only Qwen results as acceptance evidence.
