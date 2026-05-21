# Artifact Distribution

The adapted Qwen3 router bundle is **generated output**: about 624 MB of
safetensors plus a manifest. It is gitignored, so a fresh `git clone`
does not contain it. This guide covers the two distribution flows.

## Quick Map

- **Consumer** (you cloned the repo and want to run the router):
  `mix trinity.artifact.fetch`. See §1 below.
- **Publisher** (you forked the repo and want to ship your own bundle):
  upload to a HuggingFace dataset repo with `HfHub.Commit.upload_folder/3`.
  See §2.
- **Plan B** (HuggingFace is unreachable): GitHub Release fallback. See §3.

## 1. Consumer flow — `mix trinity.artifact.fetch`

The intended fresh-clone path:

```bash
git clone https://github.com/nshkrdotcom/trinity_coordinator
cd trinity_coordinator
mix deps.get
mix trinity.artifact.fetch
XLA_TARGET=cuda12 mix run examples/qwen_router_prompt_eval.exs \
  --snapshot examples/fixtures/qwen_router_prompt_eval_logits.json \
  --determinism-runs 2
```

`mix trinity.artifact.fetch` reads
`priv/sakana_trinity/artifact_pin.json` for the source repo and the
per-file SHA-256 manifest, then downloads each file from HuggingFace
with `hf_hub`, verifying the checksum after each download. Files
already present at the destination with the correct checksum are
skipped.

### Common Options

```bash
# Default flow (fetches into priv/sakana_trinity/adapted_qwen3_0_6b_layer26/)
mix trinity.artifact.fetch

# Custom destination
mix trinity.artifact.fetch --dest /opt/trinity/bundles/v1.0.0

# Use a different pin file (fork that distributes its own bundle)
mix trinity.artifact.fetch --pin priv/forks/my_pin.json

# Air-gapped / offline (consult the HuggingFace cache only)
HF_HUB_OFFLINE=1 mix trinity.artifact.fetch --offline

# Help
mix trinity.artifact.fetch --help
```

### Cache And Storage

Downloads land in the HuggingFace cache directory used by `hf_hub`, the
default of which is `~/.cache/huggingface/`. You can override it via:

```elixir
# config/config.exs or config/runtime.exs
config :hf_hub, cache_dir: "/var/cache/huggingface"
```

The materialised bundle (`priv/sakana_trinity/adapted_qwen3_0_6b_layer26/`)
is a hard-copy of the cached files. Removing the directory and
re-running the fetch is safe — the cache makes the second run fast.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `cannot load pin "priv/sakana_trinity/artifact_pin.json"` | pin file missing or corrupt | check `git status`; the pin is committed |
| `checksum_mismatch` for one file | partial download; cache corruption | clear the file from `~/.cache/huggingface/` and retry |
| `enoent` or HTTP-level error | network/DNS or HF outage | retry; consider GH-release fallback (§3) |
| `offline mode and file not cached` | running with `--offline` against a cold cache | drop `--offline` once; subsequent runs are cached |

## 2. Publisher flow — uploading a bundle to HuggingFace

This is for forks and for the maintainer cutting a new bundle revision.

### 2.1 One-time HuggingFace setup

1. Create an account at <https://huggingface.co/>.
2. Get a write-scoped access token at
   <https://huggingface.co/settings/tokens>.
3. Set the token in your environment **for the upload command only**:

```bash
HF_TOKEN=hf_xxx iex -S mix
```

The `trinity_coordinator` library does **not** read `HF_TOKEN` from the
environment in `lib/**` (per AGENTS.md). Tokens are read only during
the explicit upload session.

### 2.2 Create the dataset repository

From `iex -S mix`:

```elixir
{:ok, %{url: url}} =
  HfHub.Repo.create(
    "your-org/trinity-coordinator-adapted-qwen3-0.6b",
    repo_type: :dataset,
    private: false,
    token: System.get_env("HF_TOKEN")
  )

IO.puts("Dataset repo created: #{url}")
```

Pick a repo name that signals (a) which underlying base model, (b) that
this is the adapted variant, (c) your ownership. The maintainer's
canonical name is `nshkrdotcom/trinity-coordinator-adapted-qwen3-0.6b`.

### 2.3 Upload the bundle

```elixir
artifact_dir = "priv/sakana_trinity/adapted_qwen3_0_6b_layer26"
repo_id      = "your-org/trinity-coordinator-adapted-qwen3-0.6b"
token        = System.get_env("HF_TOKEN")

{:ok, _info} =
  HfHub.Commit.upload_folder(
    artifact_dir,
    repo_id,
    token: token,
    repo_type: :dataset,
    commit_message: "v1.0.0: initial adapted-artifact bundle",
    ignore_patterns: ["*.log.jsonl", "*.tmp", ".DS_Store"]
  )
```

LFS upload is automatic for files ≥ 10 MB — the two 297 MB tensors
(`0001_embedder…` and `0009_language_modeling_head…`) ride LFS; the
nine smaller files (4–6 MB each) ride a normal commit.

### 2.4 Tag the revision

```elixir
{:ok, _} =
  HfHub.Git.create_tag(
    repo_id,
    "v1.0.0",
    repo_type: :dataset,
    message: "Initial public release of the adapted Qwen3 router bundle",
    token: token
  )
```

### 2.5 Regenerate the pin

After publishing, regenerate `artifact_pin.json` so the fetch task
points at the new revision:

```bash
XLA_TARGET=cuda12 mix run build_support/build_artifact_pin.exs \
  --manifest priv/sakana_trinity/adapted_qwen3_0_6b_layer26/manifest.json \
  --bundle-dir priv/sakana_trinity/adapted_qwen3_0_6b_layer26 \
  --repo-id your-org/trinity-coordinator-adapted-qwen3-0.6b \
  --revision v1.0.0 \
  --out priv/sakana_trinity/artifact_pin.json
```

Commit the updated pin file to your fork. Downstream users get the new
bundle on next `mix trinity.artifact.fetch`.

### 2.6 Smoke check after publishing

From a throwaway scratch project:

```elixir
{:ok, path} =
  HfHub.Download.hf_hub_download(
    repo_id: "your-org/trinity-coordinator-adapted-qwen3-0.6b",
    filename: "manifest.json",
    repo_type: :dataset,
    revision: "v1.0.0"
  )

{:ok, json} = path |> File.read!() |> Jason.decode()
assert json["router_head_shape"] == [10, 1024]
```

If that fetches and decodes cleanly, the rest of the bundle is
addressable by the same path.

## 3. Plan B — GitHub Release fallback

When HuggingFace is unreachable (firewalled CI, HF outage,
self-hosted CI behind a proxy), publish a tarball as a GitHub Release
asset.

### 3.1 Publisher side

```bash
cd priv/sakana_trinity
tar czf adapted_qwen3_0_6b_layer26-v1.0.0.tar.gz \
  adapted_qwen3_0_6b_layer26/
sha256sum adapted_qwen3_0_6b_layer26-v1.0.0.tar.gz

gh release create v1.0.0-artifact-only \
  --title "Adapted Qwen3-0.6B bundle v1.0.0" \
  --notes "Mirror of the HuggingFace dataset bundle." \
  adapted_qwen3_0_6b_layer26-v1.0.0.tar.gz
```

GitHub Release assets cap at 2 GB; the 624 MB bundle is comfortably
under that.

### 3.2 Consumer side

```bash
TARBALL=adapted_qwen3_0_6b_layer26-v1.0.0.tar.gz
curl -L -o /tmp/$TARBALL \
  https://github.com/your-org/trinity_coordinator/releases/download/v1.0.0-artifact-only/$TARBALL

echo "<expected-sha256>  /tmp/$TARBALL" | sha256sum -c -
tar xzf /tmp/$TARBALL -C priv/sakana_trinity/
```

The `mix trinity.artifact.fetch` task does not auto-fall-back to GitHub
Releases; this path is a manual mirror for cases where the HF flow
cannot be used.

## 4. Versioning Conventions

The bundle has its own SemVer identity, independent of the
`trinity_coordinator` Hex version. The pin file tracks both:

- `repo_id` — HuggingFace dataset repo holding the bundle
- `revision` — the bundle's SemVer tag (e.g. `v1.0.0`)
- `manifest_sha256` — SHA of the bundle's own manifest, so a pin and a
  manifest cannot silently disagree
- `files[].sha256` — per-file SHA, fed straight into
  `hf_hub`'s `verify_checksum: true` + `expected_sha256:`

When you cut a new bundle revision, you bump the `revision` (and
likely the `manifest_sha256` and a subset of `files[].sha256`). The
project's Hex version does not need to move unless the change requires
a code update on the consumer side (e.g. a manifest schema bump).

## 5. References

- [`mix trinity.artifact.fetch`](`Mix.Tasks.Trinity.Artifact.Fetch`)
  — the consumer command.
- [`TrinityCoordinator.ArtifactFetch`](`TrinityCoordinator.ArtifactFetch`)
  — the library backing the task.
- [`TrinityCoordinator.ArtifactFetch.Pin`](`TrinityCoordinator.ArtifactFetch.Pin`)
  — the pin schema.
- [`HfHub.Download.hf_hub_download/1`](https://hexdocs.pm/hf_hub/HfHub.Download.html)
  — the underlying file fetch.
- [`HfHub.Commit.upload_folder/3`](https://hexdocs.pm/hf_hub/HfHub.Commit.html)
  — bundle upload.
- [Runtime Profiles](runtime_profiles.md) — when you'd want to
  regenerate the bundle yourself.
