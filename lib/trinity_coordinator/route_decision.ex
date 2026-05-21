defmodule TrinityCoordinator.RouteDecision do
  @moduledoc """
  Structured representation of one router decision.

  Wraps the informal map returned by `TrinityCoordinator.CoordinationHead.route/6`
  in an explicit struct with named fields and a stable `:transcript_hash`
  fingerprint. Downstream code that wants type-safe access to a route should
  prefer this struct; legacy callers that read `route.agent_id` from the raw
  map continue to work unchanged.

  Fields:

    * `:agent_id` — chosen agent slot (`0..6` for the canonical artifact)
    * `:role_id` — chosen role index (0=Worker, 1=Thinker, 2=Verifier)
    * `:role_name` — string name ("Worker" / "Thinker" / "Verifier")
    * `:agent_logits`, `:role_logits`, `:logits` — Nx tensors
    * `:margins` — `%{agent: float(), role: float()}` top-1 minus top-2 logit
      margins, computed by `from_route/3`
    * `:selection_modes` — `%{agent: atom(), role: atom()}` (default
      `%{agent: :argmax, role: :argmax}`)
    * `:transcript_hash` — sha256 over the transcript that produced this route
    * `:artifact_identity` — optional map; e.g. router head sha256 + artifact dir
  """

  alias TrinityCoordinator.Trace.Hash

  @enforce_keys [:agent_id, :role_id, :role_name]
  defstruct [
    :agent_id,
    :role_id,
    :role_name,
    :agent_logits,
    :role_logits,
    :logits,
    :margins,
    :selection_modes,
    :transcript_hash,
    :artifact_identity
  ]

  @type margin :: %{required(:agent) => float(), required(:role) => float()}

  @type selection_mode :: :argmax | :softmax | :sample

  @type t :: %__MODULE__{
          agent_id: non_neg_integer(),
          role_id: 0 | 1 | 2,
          role_name: String.t(),
          agent_logits: Nx.Tensor.t() | nil,
          role_logits: Nx.Tensor.t() | nil,
          logits: Nx.Tensor.t() | nil,
          margins: margin() | nil,
          selection_modes:
            %{required(:agent) => selection_mode(), required(:role) => selection_mode()}
            | nil,
          transcript_hash: String.t() | nil,
          artifact_identity: map() | nil
        }

  @doc """
  Builds a `%RouteDecision{}` from the raw map returned by
  `CoordinationHead.route/6`.

  `messages` (or a binary transcript hash) is used to compute
  `:transcript_hash`; supply `nil` if you don't have it. `opts` may include
  `:artifact_identity` (a small map describing the artifact provenance), and
  `:role_name` if you want to override the default lookup.
  """
  @spec from_route(map(), [map()] | String.t() | nil, keyword()) :: t()
  def from_route(route, messages_or_hash \\ nil, opts \\ []) when is_map(route) do
    role_id = Map.fetch!(route, :role_id)
    role_name = Keyword.get(opts, :role_name) || default_role_name(role_id)

    transcript_hash =
      case messages_or_hash do
        nil -> nil
        hash when is_binary(hash) -> hash
        msgs when is_list(msgs) -> Hash.messages(msgs)
      end

    selection_modes = %{
      agent: Map.get(route, :agent_selection_mode, :argmax),
      role: Map.get(route, :role_selection_mode, :argmax)
    }

    margins = %{
      agent: tensor_top_margin(Map.get(route, :agent_logits)),
      role: tensor_top_margin(Map.get(route, :role_logits))
    }

    %__MODULE__{
      agent_id: Map.fetch!(route, :agent_id),
      role_id: role_id,
      role_name: role_name,
      agent_logits: Map.get(route, :agent_logits),
      role_logits: Map.get(route, :role_logits),
      logits: Map.get(route, :logits),
      margins: margins,
      selection_modes: selection_modes,
      transcript_hash: transcript_hash,
      artifact_identity: Keyword.get(opts, :artifact_identity)
    }
  end

  defp default_role_name(0), do: "Worker"
  defp default_role_name(1), do: "Thinker"
  defp default_role_name(2), do: "Verifier"
  defp default_role_name(other), do: "role_#{other}"

  defp tensor_top_margin(nil), do: nil

  defp tensor_top_margin(%Nx.Tensor{} = t) do
    case t |> Nx.to_flat_list() |> Enum.sort(:desc) do
      [a, b | _] -> a - b
      [_] -> 0.0
      [] -> nil
    end
  end

  @doc """
  Returns a JSON-encodable map representation of this struct, omitting Nx
  tensors. Use this when persisting to trace logs or shipping across
  process boundaries.
  """
  @spec to_trace_map(t()) :: map()
  def to_trace_map(%__MODULE__{} = rd) do
    %{
      agent_id: rd.agent_id,
      role_id: rd.role_id,
      role_name: rd.role_name,
      margins: rd.margins,
      selection_modes: rd.selection_modes,
      transcript_hash: rd.transcript_hash,
      artifact_identity: rd.artifact_identity
    }
  end
end
