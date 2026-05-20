defmodule TrinityCoordinator.Sakana.TinyArtifactSmokeTest do
  @moduledoc """
  CPU-only smoke for the canonical manifest + router head + routing-state path.

  Loads the committed tiny synthetic fixture
  (`test/fixtures/sakana_tiny_artifact/`), exercises the full artifact loader
  + `Head.build_routing_state/2` + `Head.assert_shape_invariants!/2`, and
  forwards a synthetic hidden state through the built Axon model.

  No Bumblebee. No Qwen. No CUDA. The test asserts wall-clock under a generous
  CPU budget so a future regression that pulls in heavyweight init fails the
  metric instead of just slowing CI silently.
  """

  use ExUnit.Case, async: true

  alias TrinityCoordinator.Sakana.{Artifact, Head}
  alias TrinityCoordinator.Test.SakanaTinyArtifactFactory

  @cpu_wall_budget_ms 5_000

  test "loads the tiny fixture and routes a synthetic hidden state without CUDA" do
    started = System.monotonic_time(:millisecond)

    fixture_dir = SakanaTinyArtifactFactory.fixture_dir()
    {num_agents, num_roles, hidden} = SakanaTinyArtifactFactory.dimensions()

    # Canonical artifact load — same code path the CUDA Coordinator.load/1 uses.
    {:ok, manifest} = Artifact.load_manifest(fixture_dir)
    assert manifest["router_head_shape"] == [num_agents + num_roles, hidden]
    assert manifest["status"] == "complete"

    head_weights = Artifact.load_router_head!(fixture_dir, manifest: manifest)
    assert Nx.shape(head_weights) == {num_agents + num_roles, hidden}

    # Routing state build — uses CoordinationHead + Axon init, no Bumblebee.
    {:ok, head_state} =
      Head.build_routing_state(head_weights, num_roles: num_roles, backend: Nx.BinaryBackend)

    assert head_state.num_agents == num_agents
    assert head_state.num_roles == num_roles
    assert head_state.hidden_size == hidden

    # Shape-invariant gate that Phase 4 added to Coordinator.load/1.
    assert :ok = Head.assert_shape_invariants!(head_state, manifest)

    # Forward pass through the routing head with a synthetic hidden vector.
    hidden_vector = Nx.tensor(List.duplicate(0.5, hidden), type: :f32) |> Nx.new_axis(0)
    {_init_fn, predict_fn} = Axon.build(head_state.model)
    logits = predict_fn.(head_state.params, %{"hidden_state" => hidden_vector})

    assert Nx.shape(logits) == {1, num_agents + num_roles}

    # Argmax slicing into agent vs role using the same convention as CoordinationHead.
    {agent_id, role_id} = argmax_pair(logits, num_agents)
    assert agent_id in 0..(num_agents - 1)
    assert role_id in 0..(num_roles - 1)

    elapsed = System.monotonic_time(:millisecond) - started

    assert elapsed < @cpu_wall_budget_ms,
           "tiny smoke ran in #{elapsed}ms; budget is #{@cpu_wall_budget_ms}ms"
  end

  defp argmax_pair(logits, num_agents) do
    flat = logits |> Nx.squeeze(axes: [0]) |> Nx.to_flat_list()
    {agent_logits, role_logits} = Enum.split(flat, num_agents)

    {Enum.with_index(agent_logits) |> Enum.max_by(&elem(&1, 0)) |> elem(1),
     Enum.with_index(role_logits) |> Enum.max_by(&elem(&1, 0)) |> elem(1)}
  end

  test "loads when the head is rewritten in a temp directory (factory parity)" do
    dir =
      Path.join(System.tmp_dir!(), "tiny_artifact_parity_#{System.unique_integer([:positive])}")

    on_exit(fn -> File.rm_rf!(dir) end)

    head_sha = SakanaTinyArtifactFactory.write!(dir)

    {:ok, manifest} = Artifact.load_manifest(dir)
    assert manifest["router_head_sha256"] == head_sha

    head_weights = Artifact.load_router_head!(dir, manifest: manifest)

    {:ok, state} =
      Head.build_routing_state(head_weights, num_roles: 3, backend: Nx.BinaryBackend)

    assert :ok = Head.assert_shape_invariants!(state, manifest)
  end

  test "tiny fixture sha256 matches the committed manifest declaration" do
    committed_manifest =
      Path.join(SakanaTinyArtifactFactory.fixture_dir(), "manifest.json")
      |> File.read!()
      |> Jason.decode!()

    computed_sha =
      Path.join(SakanaTinyArtifactFactory.fixture_dir(), "router_head.safetensors")
      |> File.read!()
      |> then(&:crypto.hash(:sha256, &1))
      |> Base.encode16(case: :lower)

    assert committed_manifest["router_head_sha256"] == computed_sha
  end
end
