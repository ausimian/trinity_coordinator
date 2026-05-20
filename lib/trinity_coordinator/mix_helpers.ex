defmodule TrinityCoordinator.MixHelpers do
  @moduledoc """
  Shared helpers for the `trinity.*` Mix tasks.

  These helpers convert coordinator-load and artifact-load errors into readable
  `Mix.raise/1` messages so that operator-facing tasks never surface
  `MatchError` for expected setup failures (missing CUDA, missing artifact
  directory, manifest drift).

  This module lives outside the `Mix.Tasks.*` namespace by design: any module
  under `lib/mix/tasks/` is picked up by `mix help` as a task, and a
  non-`use Mix.Task` module placed there would produce confusing diagnostics.
  """

  alias TrinityCoordinator.Sakana.Coordinator

  @typedoc "Reason returned by `Coordinator.load/1` when it fails."
  @type load_reason :: term()

  @doc """
  Loads a coordinator via `TrinityCoordinator.Sakana.Coordinator.load/1`, or
  raises a `Mix.Error` with a readable, prefixed message.

  `opts` is the same keyword list accepted by `Coordinator.load/1`, including
  but not limited to `:artifact_dir`, `:num_roles`, `:backend`, and
  `:require_cuda`. Forwarded as-is.

  ## Examples

      iex> _coordinator = TrinityCoordinator.MixHelpers.load_coordinator!(
      ...>   artifact_dir: "priv/sakana_trinity/adapted_qwen3_0_6b_layer26"
      ...> )
  """
  @spec load_coordinator!(keyword()) :: map()
  def load_coordinator!(opts) when is_list(opts) do
    case Coordinator.load(opts) do
      {:ok, coordinator} ->
        coordinator

      {:error, reason} ->
        Mix.raise("coordinator load failed: " <> format_load_error(reason))
    end
  end

  @doc """
  Formats a coordinator load failure reason into a single-line operator string.

  Unwraps the `{:coordinator_load_error, message}` shape produced by
  `Coordinator.load/1`'s rescue clause; falls back to `inspect/1` for any
  other shape.
  """
  @spec format_load_error(load_reason()) :: String.t()
  def format_load_error({:coordinator_load_error, message}) when is_binary(message),
    do: message

  def format_load_error(reason), do: inspect(reason)
end
