defmodule Examples.QwenRouterPromptEval do
  @moduledoc false

  require Logger

  alias TrinityCoordinator.{HITL, RoleInjector, Runtime, Trace}
  alias TrinityCoordinator.Sakana.{Artifact, Coordinator}

  @default_artifact_dir "priv/sakana_trinity/adapted_qwen3_0_6b_layer26"
  @child_env "TRINITY_QWEN_ROUTER_PROMPT_EVAL_CHILD"
  @native_log_path "tmp/examples/qwen_router_prompt_eval.native.log"

  @agent_names %{
    0 => "gpt-5",
    1 => "claude-sonnet-4-20250514",
    2 => "gemini-2.5-pro",
    3 => "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    4 => "google/gemma-3-27b-it",
    5 => "Qwen/Qwen3-32B (reasoning)",
    6 => "Qwen/Qwen3-32B (direct)"
  }

  @cases [
    %{
      id: "math_direct",
      expected: %{agent_id: 4, role_id: 2},
      messages: [
        %{role: "user", content: "What is 17 + 25? Answer briefly."}
      ]
    },
    %{
      id: "math_proof",
      expected: %{agent_id: 0, role_id: 0},
      messages: [
        %{
          role: "user",
          content:
            "Prove that the sum of the first n odd positive integers is n squared. Route this to the best next role."
        }
      ]
    },
    %{
      id: "code_debug",
      expected: %{agent_id: 0, role_id: 0},
      messages: [
        %{
          role: "user",
          content:
            "A Python function mutates its default list argument across calls. Identify the bug and propose the smallest fix."
        }
      ]
    },
    %{
      id: "security_review",
      expected: %{agent_id: 4, role_id: 1},
      messages: [
        %{
          role: "user",
          content:
            "Review this login flow for security risks: passwords are hashed, sessions are cookies, and reset tokens never expire."
        }
      ]
    },
    %{
      id: "planning",
      expected: %{agent_id: 4, role_id: 0},
      messages: [
        %{
          role: "user",
          content:
            "Create a concise implementation plan for migrating a small Elixir service from in-memory state to Postgres."
        }
      ]
    },
    %{
      id: "verification_after_worker",
      expected: %{agent_id: 4, role_id: 2},
      messages: [
        %{role: "user", content: "Calculate 6 * 7 and verify the answer."},
        %{role: "assistant", content: "Worker answer: 6 * 7 = 42."}
      ]
    },
    %{
      id: "needs_revision",
      expected: %{agent_id: 4, role_id: 2},
      messages: [
        %{role: "user", content: "Check whether the answer is correct: 19 + 24 = 41."},
        %{role: "assistant", content: "Worker answer: 19 + 24 = 41."}
      ]
    },
    %{
      id: "ambiguous_decomposition",
      expected: %{agent_id: 0, role_id: 2},
      messages: [
        %{
          role: "user",
          content:
            "This problem has unclear requirements. We may need to split it into assumptions, risks, and a concrete next action."
        }
      ]
    },
    %{
      id: "creative_but_constrained",
      expected: %{agent_id: 4, role_id: 2},
      messages: [
        %{
          role: "user",
          content:
            "Draft a friendly but precise support reply explaining a billing correction. Keep it under 120 words."
        }
      ]
    },
    %{
      id: "longer_context",
      expected: %{agent_id: 4, role_id: 2},
      messages: [
        %{
          role: "system",
          content:
            "You are routing work inside a three-role TRINITY loop. Worker solves, Thinker plans or redirects, Verifier checks."
        },
        %{
          role: "user",
          content:
            "Given a release checklist, identify the next best role. The feature compiles, unit tests pass, docs changed, but no smoke test has been run yet."
        }
      ]
    },
    %{
      id: "provider_failure_triage",
      expected: %{agent_id: 4, role_id: 2},
      messages: [
        %{
          role: "user",
          content:
            "The last provider call timed out after 30 seconds. Decide whether to retry, ask a thinker for a smaller plan, or verify the partial answer."
        }
      ]
    },
    %{
      id: "final_answer_check",
      expected: %{agent_id: 4, role_id: 0},
      messages: [
        %{role: "user", content: "Solve and then verify: the capital of France is Paris."},
        %{role: "assistant", content: "Worker answer: Paris is the capital of France."},
        %{role: "assistant", content: "Thinker note: this is a factual lookup and likely ready."}
      ]
    }
  ]

  def main(argv) do
    argv = normalize_argv(argv)
    maybe_reexec_with_suppressed_stderr!(argv)

    unless "--debug-native-logs" in argv do
      Logger.configure(level: :error)
    end

    Application.ensure_all_started(:trinity_coordinator)

    {opts, rest, errors} =
      argv
      |> OptionParser.parse(
        strict: [
          artifact_dir: :string,
          case: :keep,
          debug_native_logs: :boolean,
          list_cases: :boolean,
          no_assert: :boolean,
          show_logits: :boolean,
          verbose: :boolean
        ]
      )

    unless rest == [], do: raise("Unexpected arguments: #{inspect(rest)}")
    unless errors == [], do: raise("Invalid options: #{inspect(errors)}")

    if Keyword.get(opts, :list_cases, false) do
      list_cases()
    else
      run_eval!(opts)
    end
  end

  defp normalize_argv(["--" | rest]), do: rest
  defp normalize_argv(argv), do: argv

  defp maybe_reexec_with_suppressed_stderr!(argv) do
    cond do
      System.get_env(@child_env) == "1" ->
        :ok

      "--list-cases" in argv ->
        :ok

      "--debug-native-logs" in argv ->
        :ok

      true ->
        File.mkdir_p!(Path.dirname(@native_log_path))
        File.rm(@native_log_path)

        shell = """
        stderr_path=$1
        script_path=$2
        shift 2
        export #{@child_env}=1
        mix run "$script_path" -- "$@" 2>"$stderr_path"
        """

        {_output, status} =
          System.cmd("sh", ["-c", shell, "sh", @native_log_path, __ENV__.file | argv],
            into: IO.stream(:stdio, :line)
          )

        if status != 0 do
          IO.puts(:stderr, "\nqwen_router_prompt_eval failed.")
          IO.puts(:stderr, "Native/framework logs were captured at #{@native_log_path}.")
          IO.puts(:stderr, "Re-run with --debug-native-logs to see them inline.\n")
          print_stderr_tail(@native_log_path)
        end

        System.halt(status)
    end
  end

  defp print_stderr_tail(path) do
    if File.exists?(path) do
      path
      |> File.read!()
      |> String.split("\n")
      |> Enum.reject(&(&1 == ""))
      |> Enum.take(-40)
      |> case do
        [] ->
          :ok

        lines ->
          IO.puts(:stderr, "Last stderr lines:")
          Enum.each(lines, &IO.puts(:stderr, &1))
      end
    end
  end

  defp list_cases do
    Enum.each(@cases, fn case_spec ->
      IO.puts(case_spec.id)
    end)
  end

  defp run_eval!(opts) do
    artifact_dir = Keyword.get(opts, :artifact_dir, @default_artifact_dir)
    manifest_path = Artifact.manifest_path(artifact_dir)
    ensure_manifest!(manifest_path)

    selected_cases = select_cases!(Keyword.get_values(opts, :case))
    assert? = not Keyword.get(opts, :no_assert, false)
    show_logits? = Keyword.get(opts, :show_logits, false)
    verbose? = Keyword.get(opts, :verbose, false) or show_logits?

    HITL.banner("QWEN ROUTER PROMPT EVAL")
    Runtime.put_cuda_backend!()

    {:ok, coordinator} = Coordinator.load(artifact_dir: artifact_dir)

    IO.puts("""

    What this does
      Loads the local adapted Qwen router once, sends fixed transcripts through it,
      and checks whether the selected agent slot and role match expectations.

      No external LLM/provider calls are made.

    Agent labels
      The agent names are labels from the original Sakana checkpoint.
      Example: agent 4 is labeled "google/gemma-3-27b-it".
      This eval does not call Gemma; it only reports that the router selected
      checkpoint slot 4.

    Artifact
      #{artifact_dir}

    Model
      base router model: #{coordinator.manifest["base_model_repo"]}
      router head shape: #{inspect(coordinator.manifest["router_head_shape"])}
      assertion mode: #{if(assert?, do: "strict", else: "report only")}

    Native logs
      hidden in normal mode: #{@native_log_path}
      use --debug-native-logs to print XLA/CUDA compiler logs inline
    """)

    results =
      selected_cases
      |> Enum.with_index(1)
      |> Enum.map(fn {case_spec, index} ->
        route_case!(
          coordinator,
          case_spec,
          index,
          length(selected_cases),
          assert?,
          show_logits?,
          verbose?
        )
      end)

    failures = Enum.filter(results, &(&1.status == :fail))

    if failures == [] do
      print_summary(results)
    else
      ids = failures |> Enum.map(& &1.id) |> Enum.join(", ")
      raise "qwen_router_prompt_eval failed cases=#{ids}"
    end
  end

  defp select_cases!([]), do: @cases

  defp select_cases!(ids) do
    cases_by_id = Map.new(@cases, &{&1.id, &1})

    Enum.map(ids, fn id ->
      Map.get(cases_by_id, id) || raise("Unknown case #{inspect(id)}. Run with --list-cases.")
    end)
  end

  defp ensure_manifest!(manifest_path) do
    unless File.exists?(manifest_path) do
      raise """
      Missing adapted artifact manifest: #{manifest_path}

      Install the canonical artifact bundle at #{@default_artifact_dir}, or pass
      --artifact-dir path/to/adapted_qwen3_0_6b_layer26.
      """
    end
  end

  defp route_case!(coordinator, case_spec, index, total, assert?, show_logits?, verbose?) do
    {:ok, routed} = Coordinator.route_messages(coordinator, case_spec.messages)

    route = routed.route
    expected = case_spec.expected
    actual = %{agent_id: route.agent_id, role_id: route.role_id}

    status =
      cond do
        not assert? -> :report
        expectation_matches?(expected, actual) -> :ok
        true -> :fail
      end

    print_case(case_spec, routed, index, total, status, show_logits?, verbose?)

    %{id: case_spec.id, status: status, role_id: route.role_id, agent_id: route.agent_id}
  end

  defp expectation_matches?(expected, actual) do
    expected.agent_id == actual.agent_id and expected.role_id == actual.role_id
  end

  defp print_case(case_spec, routed, index, total, status, show_logits?, verbose?) do
    extraction = routed.extraction
    route = routed.route
    expected = case_spec.expected
    token_count = extraction.input_ids |> Nx.to_flat_list() |> length()
    prompt = format_prompt(case_spec.messages)

    IO.puts("""

    [#{index}/#{total}] #{case_spec.id} - #{status_label(status)}

    Prompt sent to router:
    #{indent(prompt, 2)}

    Expected route:
      agent #{expected.agent_id}: #{Map.fetch!(@agent_names, expected.agent_id)}
      role  #{expected.role_id}: #{RoleInjector.role_name(expected.role_id)}

    Router returned:
      agent #{route.agent_id}: #{Map.fetch!(@agent_names, route.agent_id)}
      role  #{route.role_id}: #{RoleInjector.role_name(route.role_id)}

    Router input tokens: #{token_count}
    """)

    if verbose? do
      IO.puts("""
      Debug:
        hidden_index: #{extraction.hidden_index}
        transcript_hash: #{Trace.Hash.messages(case_spec.messages)}
        route_vector_hash: #{Trace.Hash.tensor(extraction.vector_snapshot)}
        agent_margin: #{format_float(top_margin(route.agent_logits))}
        role_margin: #{format_float(top_margin(route.role_logits))}
      """)
    end

    if show_logits? do
      IO.puts("      agent_logits: #{inspect(round_list(Nx.to_flat_list(route.agent_logits)))}")
      IO.puts("      role_logits: #{inspect(round_list(Nx.to_flat_list(route.role_logits)))}")
    end
  end

  defp status_label(:ok), do: "PASS"
  defp status_label(:fail), do: "FAIL"
  defp status_label(:report), do: "REPORT ONLY"

  defp format_prompt(messages) do
    messages
    |> Enum.map(fn message ->
      role = Map.get(message, :role, Map.get(message, "role"))
      content = Map.get(message, :content, Map.get(message, "content"))
      "#{role}: #{content}"
    end)
    |> Enum.join("\n")
  end

  defp indent(text, spaces) do
    padding = String.duplicate(" ", spaces)

    text
    |> String.split("\n")
    |> Enum.map_join("\n", &(padding <> &1))
  end

  defp print_summary(results) do
    passed = Enum.count(results, &(&1.status in [:ok, :report]))
    failed = Enum.count(results, &(&1.status == :fail))

    role_counts =
      results
      |> Enum.frequencies_by(& &1.role_id)
      |> Enum.sort()
      |> Enum.map_join(", ", fn {role_id, count} ->
        "#{RoleInjector.role_name(role_id)}=#{count}"
      end)

    agent_counts =
      results
      |> Enum.frequencies_by(& &1.agent_id)
      |> Enum.sort()
      |> Enum.map_join(", ", fn {agent_id, count} ->
        "#{agent_id}=#{count}"
      end)

    IO.puts("""

    Summary
      passed: #{passed}
      failed: #{failed}
      roles selected: #{role_counts}
      agent slots selected: #{agent_counts}

    PASS qwen_router_prompt_eval
    """)
  end

  defp top_margin(%Nx.Tensor{} = logits) do
    logits
    |> Nx.to_flat_list()
    |> Enum.sort(:desc)
    |> case do
      [first, second | _] -> first - second
      [_only] -> 0.0
      [] -> 0.0
    end
  end

  defp round_list(values) do
    Enum.map(values, fn
      value when is_float(value) -> Float.round(value, 5)
      value -> value
    end)
  end

  defp format_float(value) when is_float(value), do: :erlang.float_to_binary(value, decimals: 5)
  defp format_float(value), do: inspect(value)
end

Examples.QwenRouterPromptEval.main(System.argv())
