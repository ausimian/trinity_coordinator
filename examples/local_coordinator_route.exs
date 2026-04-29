defmodule Examples.LocalCoordinatorRoute do
  @moduledoc false

  alias TrinityCoordinator.{HITL, RoleInjector, Runtime, Trace}
  alias TrinityCoordinator.Sakana.{Artifact, Coordinator}

  @default_artifact_dir "tmp/sakana_parity/adapted_artifacts_from_python"
  @default_prompt "Select a TRINITY role for this reasoning task."

  def main(argv) do
    Application.ensure_all_started(:trinity_coordinator)

    {opts, rest, errors} =
      argv
      |> normalize_argv()
      |> OptionParser.parse(
        strict: [
          artifact_dir: :string,
          prompt: :string
        ]
      )

    unless rest == [], do: raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: raise("Invalid options: #{inspect(errors)}")

    artifact_dir = Keyword.get(opts, :artifact_dir, @default_artifact_dir)
    prompt = Keyword.get(opts, :prompt, @default_prompt)
    manifest_path = Artifact.manifest_path(artifact_dir)

    ensure_manifest!(manifest_path)

    HITL.banner("TRINITY EXAMPLE: LOCAL COORDINATOR ROUTE")
    Runtime.put_cuda_backend!()

    {:ok, coordinator} = Coordinator.load(artifact_dir: artifact_dir)

    messages = [%{role: "user", content: prompt}]
    {:ok, routed} = Coordinator.route_messages(coordinator, messages)

    print_report(artifact_dir, manifest_path, prompt, messages, coordinator, routed)
  end

  defp normalize_argv(["--" | rest]), do: rest
  defp normalize_argv(argv), do: argv

  defp ensure_manifest!(manifest_path) do
    unless File.exists?(manifest_path) do
      raise """
      Missing adapted artifact manifest: #{manifest_path}

      Build or import the canonical artifact bundle first, then rerun:

          XLA_TARGET=cuda12 mix run examples/local_coordinator_route.exs -- \\
            --artifact-dir tmp/sakana_parity/adapted_artifacts_from_python
      """
    end
  end

  defp print_report(artifact_dir, manifest_path, prompt, messages, coordinator, routed) do
    extraction = routed.extraction
    route = routed.route
    manifest_hash = Artifact.file_sha256!(manifest_path)

    IO.puts("""

    Input
      prompt: #{prompt}
      transcript_hash: #{Trace.Hash.messages(messages)}

    Artifact
      dir: #{artifact_dir}
      manifest_path: #{manifest_path}
      manifest_sha256: #{manifest_hash}
      status: #{coordinator.manifest["status"]}
      layout: #{coordinator.manifest["artifact_layout"]}
      base_model: #{coordinator.manifest["base_model_repo"]}
      source_vector_sha256: #{coordinator.manifest["source_vector_sha256"]}
      selected_tensor_count: #{coordinator.manifest["selected_tensor_count"]}
      selected_singular_value_count: #{coordinator.manifest["selected_singular_value_count"]}
      router_head_shape: #{inspect(coordinator.manifest["router_head_shape"])}

    Tokenization And Hidden Extraction
      formatted_transcript:
    #{indent(extraction.transcript, 4)}
      input_shapes: #{inspect(extraction.input_shapes)}
      input_ids: #{inspect(tensor_list(extraction.input_ids))}
      hidden_state_shape: #{inspect(extraction.hidden_state_shape)}
      hidden_position: #{inspect(extraction.hidden_position)}
      hidden_index: #{inspect(extraction.hidden_index)}
      route_vector_shape: #{inspect(extraction.vector_shape)}
      route_vector_backend: #{Runtime.tensor_backend(extraction.vector)}
      route_vector_hash: #{Trace.Hash.tensor(extraction.vector_snapshot)}

    Router
      logits_shape: #{inspect(Nx.shape(route.logits))}
      logits: #{inspect(round_list(Nx.to_flat_list(Nx.squeeze(route.logits, axes: [0]))))}
      agent_logits: #{inspect(round_list(Nx.to_flat_list(route.agent_logits)))}
      role_logits: #{inspect(round_list(Nx.to_flat_list(route.role_logits)))}
      selected_agent_id: #{route.agent_id}
      selected_agent_name: #{Map.fetch!(agent_names(), route.agent_id)}
      selected_role_id: #{route.role_id}
      selected_role_name: #{RoleInjector.role_name(route.role_id)}

    Boundary
      provider_calls: none
      purpose: prove the adapted local Qwen coordinator loads, extracts a hidden vector, and produces a real local route without provider dispatch.
    """)
  end

  defp agent_names do
    %{
      0 => "gpt-5",
      1 => "claude-sonnet-4-20250514",
      2 => "gemini-2.5-pro",
      3 => "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
      4 => "google/gemma-3-27b-it",
      5 => "Qwen/Qwen3-32B (reasoning)",
      6 => "Qwen/Qwen3-32B (direct)"
    }
  end

  defp tensor_list(nil), do: nil
  defp tensor_list(%Nx.Tensor{} = tensor), do: Nx.to_flat_list(tensor)

  defp round_list(values) do
    Enum.map(values, fn
      value when is_float(value) -> Float.round(value, 5)
      value -> value
    end)
  end

  defp indent(text, spaces) do
    padding = String.duplicate(" ", spaces)

    text
    |> String.split("\n")
    |> Enum.map_join("\n", &(padding <> &1))
  end
end

Examples.LocalCoordinatorRoute.main(System.argv())
