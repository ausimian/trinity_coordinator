import Config

# Runtime configuration boundary.
#
# Per AGENTS.md and Elixir runtime-configuration best practices, this is
# the only file (alongside `config/config.exs`) where the project reads
# OS environment variables. Nothing under `lib/**` calls `System.get_env/1`
# (and similar) directly; the boundary lives here.
#
# Library knobs we forward into application config:
#
#   * `HF_TOKEN`         -> `:hf_hub, :token`
#   * `HF_HUB_CACHE`     -> `:hf_hub, :cache_dir`
#   * `HF_HOME`          -> `:hf_hub, :cache_dir` (joined with "hub")
#   * `HF_HUB_OFFLINE`   -> `:hf_hub, :offline`
#
# Hosts that consume `trinity_coordinator` as a library can replicate or
# replace any of these in their own `config/runtime.exs`.

if token = System.get_env("HF_TOKEN") do
  config :hf_hub, token: token
end

cond do
  dir = System.get_env("HF_HUB_CACHE") ->
    config :hf_hub, cache_dir: dir

  dir = System.get_env("HF_HOME") ->
    config :hf_hub, cache_dir: Path.join(dir, "hub")

  true ->
    :ok
end

if System.get_env("HF_HUB_OFFLINE") in ~w(1 true) do
  config :hf_hub, offline: true
end
