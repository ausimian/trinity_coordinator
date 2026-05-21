defmodule DependencySourcesTest do
  @moduledoc """
  Regression coverage for `DependencySources`. These tests focus on the
  `:path`-vs-`:github` fallback boundary, which broke once when a fresh
  `mix deps.get` materialised sibling deps under `deps/` and the helper
  mistook them for developer-workspace checkouts.
  """

  use ExUnit.Case, async: true

  setup do
    tmp = Path.join(System.tmp_dir!(), "dep_src_test_#{System.unique_integer([:positive])}")
    File.mkdir_p!(tmp)
    on_exit(fn -> File.rm_rf!(tmp) end)
    {:ok, tmp: tmp}
  end

  defp write_config!(repo_root, config_body) do
    File.mkdir_p!(Path.join(repo_root, "build_support"))

    File.write!(
      Path.join(repo_root, "build_support/dependency_sources.config.exs"),
      config_body
    )
  end

  describe "helper version" do
    test "is at least 2 (path-guard bug fix)" do
      assert DependencySources.helper_version() >= 2
    end
  end

  describe "path vs github fallback when repo_root is a real workspace" do
    test "picks :path when the sibling checkout exists", %{tmp: tmp} do
      # Real workspace layout: parent contains repo_a/ AND sibling/
      parent = Path.join(tmp, "workspace")
      repo = Path.join(parent, "repo_a")
      sibling = Path.join(parent, "sibling")
      File.mkdir_p!(repo)
      File.mkdir_p!(sibling)

      write_config!(repo, """
      %{
        deps: %{
          sibling: %{
            path: "../sibling",
            github: %{repo: "owner/sibling", branch: "main"},
            default_order: [:path, :github]
          }
        }
      }
      """)

      tuple = DependencySources.dep(:sibling, repo)
      {:sibling, opts} = tuple
      assert Keyword.has_key?(opts, :path)
      refute Keyword.has_key?(opts, :github)
    end

    test "falls through to :github when no sibling exists", %{tmp: tmp} do
      parent = Path.join(tmp, "workspace_no_sibling")
      repo = Path.join(parent, "repo_a")
      File.mkdir_p!(repo)

      write_config!(repo, """
      %{
        deps: %{
          sibling: %{
            path: "../sibling",
            github: %{repo: "owner/sibling", branch: "main"},
            default_order: [:path, :github]
          }
        }
      }
      """)

      tuple = DependencySources.dep(:sibling, repo)
      {:sibling, [github: "owner/sibling", branch: "main"]} = tuple
    end
  end

  describe "path-guard: refuses sibling deps under a Mix deps/ ancestor" do
    test "when repo_root is itself a Mix-managed dep, :path candidate that resolves into deps/ is rejected and :github is used",
         %{tmp: tmp} do
      # Simulate: a parent `mix deps.get` has fetched two siblings into
      # deps/. From repo_a's perspective, `../sibling` resolves into
      # deps/sibling, which exists. The guard must reject it.
      parent = Path.join(tmp, "fresh_clone")
      deps_dir = Path.join(parent, "deps")
      repo = Path.join(deps_dir, "repo_a")
      sibling = Path.join(deps_dir, "sibling")
      File.mkdir_p!(repo)
      File.mkdir_p!(sibling)

      write_config!(repo, """
      %{
        deps: %{
          sibling: %{
            path: "../sibling",
            github: %{repo: "owner/sibling", branch: "main"},
            default_order: [:path, :github]
          }
        }
      }
      """)

      tuple = DependencySources.dep(:sibling, repo)
      {:sibling, [github: "owner/sibling", branch: "main"]} = tuple
    end

    test "real workspace next to deps/ still picks :path (no false positive)",
         %{tmp: tmp} do
      # repo_root is NOT under deps/, but the candidate happens to live
      # next to a different repo's deps/. The guard should not trip.
      parent = Path.join(tmp, "real_workspace_2")
      repo = Path.join(parent, "repo_a")
      sibling = Path.join(parent, "sibling")
      File.mkdir_p!(repo)
      File.mkdir_p!(sibling)
      # Decoy deps/ in some unrelated subtree.
      File.mkdir_p!(Path.join(parent, "decoy/deps/something"))

      write_config!(repo, """
      %{
        deps: %{
          sibling: %{
            path: "../sibling",
            github: %{repo: "owner/sibling", branch: "main"},
            default_order: [:path, :github]
          }
        }
      }
      """)

      tuple = DependencySources.dep(:sibling, repo)
      {:sibling, opts} = tuple
      assert Keyword.has_key?(opts, :path)
    end

    test "list-of-paths form: each candidate is guarded independently",
         %{tmp: tmp} do
      parent = Path.join(tmp, "fresh_clone_list")
      deps_dir = Path.join(parent, "deps")
      repo = Path.join(deps_dir, "repo_a")
      bad_sibling = Path.join(deps_dir, "sibling")
      File.mkdir_p!(repo)
      File.mkdir_p!(bad_sibling)

      write_config!(repo, ~s"""
      %{
        deps: %{
          sibling: %{
            path: ["../sibling", "../../somewhere_else"],
            github: %{repo: "owner/sibling", branch: "main"},
            default_order: [:path, :github]
          }
        }
      }
      """)

      # Both candidates: "../sibling" resolves under deps/ (rejected),
      # "../../somewhere_else" does not exist (rejected) -> :github.
      tuple = DependencySources.dep(:sibling, repo)
      {:sibling, [github: "owner/sibling", branch: "main"]} = tuple
    end
  end
end
