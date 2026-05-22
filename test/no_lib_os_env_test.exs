defmodule TrinityCoordinator.NoLibOsEnvTest do
  @moduledoc """
  Regression armor for the AGENTS.md rule:

      Runtime application code under `lib/**` must not call direct OS env APIs
      such as `System.get_env`, `System.fetch_env`, `System.put_env`, or
      `System.delete_env`.

  Boundaries belong in `config/runtime.exs` (or `build_support/` for
  build-time validators).

  This test scans every tracked Elixir source file under `lib/` for the
  forbidden token shapes. The check is intentionally textual; it catches
  the actual call shape we want to forbid and a couple of moduledoc strings
  that *describe* the rule are explicitly allow-listed.
  """

  use ExUnit.Case, async: true

  @forbidden ~w(
    System.get_env
    System.fetch_env
    System.fetch_env!
    System.put_env
    System.delete_env
  )

  test "no lib/** Elixir source calls direct OS env APIs" do
    hits =
      tracked_lib_files()
      |> Enum.flat_map(&forbidden_hits/1)

    assert hits == [],
           "Found direct OS env API call(s) under lib/**. Move env reads to " <>
             "config/runtime.exs (or build_support/ for build-time validators). " <>
             "Hits:\n" <> Enum.map_join(hits, "\n", fn {p, t} -> "  #{p} -> #{t}" end)
  end

  defp tracked_lib_files do
    {out, 0} = System.cmd("git", ["ls-files", "lib/"])

    out
    |> String.split("\n", trim: true)
    |> Enum.filter(&(String.ends_with?(&1, ".ex") or String.ends_with?(&1, ".exs")))
  end

  defp forbidden_hits(path) do
    body = File.read!(path)

    Enum.flat_map(@forbidden, fn token ->
      body
      |> String.split("\n")
      |> Enum.with_index(1)
      |> Enum.filter(fn {line, _idx} ->
        String.contains?(line, token) and not docstring_or_comment_line?(line)
      end)
      |> Enum.map(fn {_line, idx} -> {path, "#{token} (line #{idx})"} end)
    end)
  end

  # Returns true if the line is a `#`-comment or sits inside a docstring
  # context where mentioning the API name is documentation, not a call.
  # The check is conservative — a leading `#` (with optional indent) or any
  # backtick-quoted `System.get_env/1` reference is treated as docs.
  defp docstring_or_comment_line?(line) do
    trimmed = String.trim_leading(line)
    String.starts_with?(trimmed, "#") or String.contains?(line, "`System.")
  end
end
