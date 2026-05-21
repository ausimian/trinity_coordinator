defmodule TrinityCoordinator.RouteDecisionTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.RouteDecision

  describe "from_route/3" do
    test "wraps an argmax route and computes margins from logits tensors" do
      route = %{
        agent_id: 4,
        role_id: 2,
        agent_logits: Nx.tensor([1.0, 0.5, -2.0, 3.5, 9.0, -1.0, 0.0], type: :f32),
        role_logits: Nx.tensor([0.2, 0.1, 5.5], type: :f32),
        logits: Nx.tensor([[1.0, 0.5, -2.0, 3.5, 9.0, -1.0, 0.0, 0.2, 0.1, 5.5]], type: :f32),
        agent_selection_mode: :argmax,
        role_selection_mode: :argmax
      }

      rd = RouteDecision.from_route(route, nil)

      assert rd.agent_id == 4
      assert rd.role_id == 2
      assert rd.role_name == "Verifier"
      assert rd.selection_modes == %{agent: :argmax, role: :argmax}

      assert_in_delta rd.margins.agent, 9.0 - 3.5, 1.0e-5
      assert_in_delta rd.margins.role, 5.5 - 0.2, 1.0e-5
    end

    test "computes transcript_hash from a messages list" do
      route = %{agent_id: 0, role_id: 0}
      msgs = [%{role: "user", content: "hello"}]
      rd = RouteDecision.from_route(route, msgs)
      assert is_binary(rd.transcript_hash)
      assert String.length(rd.transcript_hash) == 64
    end

    test "accepts a pre-computed transcript hash binary" do
      route = %{agent_id: 1, role_id: 1}
      hash = String.duplicate("a", 64)
      rd = RouteDecision.from_route(route, hash)
      assert rd.transcript_hash == hash
      assert rd.role_name == "Thinker"
    end

    test "carries an artifact_identity map when supplied" do
      route = %{agent_id: 0, role_id: 0}
      identity = %{router_head_sha256: "deadbeef", artifact_dir: "priv/x"}
      rd = RouteDecision.from_route(route, nil, artifact_identity: identity)
      assert rd.artifact_identity == identity
    end

    test "missing logit tensors give nil margins" do
      route = %{agent_id: 0, role_id: 0}
      rd = RouteDecision.from_route(route, nil)
      assert rd.margins == %{agent: nil, role: nil}
    end
  end

  describe "to_trace_map/1" do
    test "produces a JSON-safe map with no Nx tensors" do
      route = %{
        agent_id: 4,
        role_id: 2,
        agent_logits: Nx.tensor([1.0, 0.5, -2.0, 3.5, 9.0, -1.0, 0.0], type: :f32),
        role_logits: Nx.tensor([0.2, 0.1, 5.5], type: :f32),
        logits: Nx.tensor([[1.0, 0.5, -2.0, 3.5, 9.0, -1.0, 0.0, 0.2, 0.1, 5.5]], type: :f32)
      }

      rd =
        RouteDecision.from_route(route, [%{role: "user", content: "x"}],
          artifact_identity: %{a: 1}
        )

      m = RouteDecision.to_trace_map(rd)

      assert m.agent_id == 4
      assert m.role_id == 2
      assert m.role_name == "Verifier"
      assert m.margins.agent > 0.0
      assert m.margins.role > 0.0
      assert m.selection_modes == %{agent: :argmax, role: :argmax}
      assert is_binary(m.transcript_hash)
      assert m.artifact_identity == %{a: 1}

      # JSON-encodable: no Nx tensors anywhere.
      refute Enum.any?(Map.values(m), fn v -> match?(%Nx.Tensor{}, v) end)
      assert {:ok, _} = Jason.encode(m)
    end

    test "safe when artifact_identity is nil" do
      rd = RouteDecision.from_route(%{agent_id: 0, role_id: 0}, nil)
      m = RouteDecision.to_trace_map(rd)
      assert m.artifact_identity == nil
      assert {:ok, _} = Jason.encode(m)
    end
  end
end
