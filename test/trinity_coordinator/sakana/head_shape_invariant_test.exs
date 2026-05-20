defmodule TrinityCoordinator.Sakana.HeadShapeInvariantTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.Sakana.Head

  defp head_state(num_agents, num_roles, hidden) do
    %{num_agents: num_agents, num_roles: num_roles, hidden_size: hidden}
  end

  defp manifest(shape), do: %{"router_head_shape" => shape}

  describe "assert_shape_invariants!/2" do
    test "returns :ok when manifest router_head_shape matches built head" do
      assert :ok =
               Head.assert_shape_invariants!(
                 head_state(7, 3, 1024),
                 manifest([10, 1024])
               )
    end

    test "raises when agent+role count disagrees with output dim" do
      err =
        try do
          Head.assert_shape_invariants!(head_state(6, 3, 1024), manifest([10, 1024]))
        rescue
          e -> e
        end

      assert %ArgumentError{} = err
      assert String.contains?(Exception.message(err), "output_count=10")
      assert String.contains?(Exception.message(err), "= 9")
    end

    test "raises when hidden size disagrees" do
      err =
        try do
          Head.assert_shape_invariants!(head_state(7, 3, 512), manifest([10, 1024]))
        rescue
          e -> e
        end

      assert %ArgumentError{} = err
      assert String.contains?(Exception.message(err), "hidden-size")
      assert String.contains?(Exception.message(err), "1024")
    end

    test "falls back to python_semantic_manifest.routing.head_shape when router_head_shape missing" do
      manifest = %{
        "python_semantic_manifest" => %{"routing" => %{"head_shape" => [10, 1024]}}
      }

      assert :ok = Head.assert_shape_invariants!(head_state(7, 3, 1024), manifest)
    end

    test "raises when manifest declares no shape at all" do
      err =
        try do
          Head.assert_shape_invariants!(head_state(7, 3, 1024), %{})
        rescue
          e -> e
        end

      assert %ArgumentError{} = err
      assert String.contains?(Exception.message(err), "malformed")
    end
  end
end
