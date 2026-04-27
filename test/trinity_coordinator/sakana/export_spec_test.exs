defmodule TrinityCoordinator.Sakana.ExportSpecTest do
  use ExUnit.Case, async: true

  alias TrinityCoordinator.Sakana.ExportSpec

  test "default qwen export spec preserves inspected Sakana constants" do
    spec = ExportSpec.qwen3_0_6b_layer26()

    assert spec.hidden_size == 1_024
    assert spec.num_agents == 7
    assert spec.num_roles == 3
    assert ExportSpec.output_count(spec) == 10
    assert spec.scale_offset_count == 9_216
    assert ExportSpec.head_param_count(spec) == 10_240
    assert ExportSpec.source_vector_size(spec) == 19_456
    assert spec.selected_layer_indices == [26]
    assert spec.source_vector_tensor == "trinity_router_es_vector"
    assert spec.router_head_tensor_key == "trinity_router_head"
    assert {:ok, ^spec} = ExportSpec.validate(spec)
  end

  test "rejects invalid specs" do
    spec = %{ExportSpec.qwen3_0_6b_layer26() | hidden_size: 0}
    assert {:error, {:invalid_export_spec, :hidden_size}} = ExportSpec.validate(spec)

    assert {:error, {:unsupported_export_spec, :unknown}} = ExportSpec.resolve(:unknown)
  end
end
