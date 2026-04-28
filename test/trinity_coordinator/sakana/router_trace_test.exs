defmodule TrinityCoordinator.Sakana.RouterTraceTest do
  use ExUnit.Case, async: true

  alias Mix.Tasks.Trinity.Sakana.RouterTrace

  test "comparison passes exact ids with hidden/logit values inside tolerances" do
    python = synthetic_report()

    elixir =
      python
      |> put_in(["hidden_vector_f32"], [1.001, 1.999, 3.001])
      |> put_in(["logits"], [0.11, 0.19, 0.31])

    comparison =
      RouterTrace.compare_reports(python, elixir, %{
        hidden_max_abs_error: 0.01,
        hidden_mean_abs_error: 0.01,
        hidden_min_cosine: 0.99,
        hidden_max_relative_l2: 0.01,
        logits_max_abs_error: 0.02,
        logits_mean_abs_error: 0.02,
        logits_min_cosine: 0.99,
        logits_max_relative_l2: 0.05
      })

    assert comparison["failed_required"] == 0
  end

  test "comparison fails required argmax mismatch" do
    python = synthetic_report()
    elixir = Map.put(python, "argmax_role_id", 1)

    comparison = RouterTrace.compare_reports(python, elixir)

    assert comparison["failed_required"] == 1
    assert Enum.any?(comparison["checks"], &(&1["name"] == "argmax_role_id" and not &1["passed"]))
  end

  test "argument parser supports declared trace tolerances" do
    opts =
      RouterTrace.parse_args!([
        "--artifact-dir",
        "tmp/artifacts",
        "--python-report",
        "tmp/python.json",
        "--out",
        "tmp/elixir.json",
        "--hidden-max-abs",
        "0.1",
        "--hidden-mean-abs",
        "0.01",
        "--hidden-min-cosine",
        "0.98",
        "--hidden-max-relative-l2",
        "0.12",
        "--logits-max-abs",
        "0.2",
        "--logits-mean-abs",
        "0.02",
        "--logits-min-cosine",
        "0.97",
        "--logits-max-relative-l2",
        "0.13"
      ])

    assert opts.artifact_dir == "tmp/artifacts"
    assert opts.python_report == "tmp/python.json"
    assert opts.out == "tmp/elixir.json"
    assert opts.tolerances.hidden_max_abs_error == 0.1
    assert opts.tolerances.hidden_mean_abs_error == 0.01
    assert opts.tolerances.hidden_min_cosine == 0.98
    assert opts.tolerances.hidden_max_relative_l2 == 0.12
    assert opts.tolerances.logits_max_abs_error == 0.2
    assert opts.tolerances.logits_mean_abs_error == 0.02
    assert opts.tolerances.logits_min_cosine == 0.97
    assert opts.tolerances.logits_max_relative_l2 == 0.13
  end

  defp synthetic_report do
    %{
      "schema" => "trinity_sakana_router_trace.v1",
      "transcript_sha256" => "transcript",
      "token_ids_sha256" => "tokens",
      "input_ids" => [1, 2, 3],
      "head_weight_sha256_as_f32" => "head",
      "hidden_vector_shape" => [1, 3],
      "logits_shape" => [1, 3],
      "hidden_vector_f32" => [1.0, 2.0, 3.0],
      "logits" => [0.1, 0.2, 0.3],
      "argmax_agent_id" => 0,
      "argmax_role_id" => 2
    }
  end
end
