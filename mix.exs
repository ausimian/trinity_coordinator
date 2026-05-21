unless Code.ensure_loaded?(DependencySources) do
  Code.require_file("build_support/dependency_sources.exs", __DIR__)
end

unless Code.ensure_loaded?(XlaTargetValidator) do
  Code.require_file("build_support/xla_target_validator.exs", __DIR__)
end

# Load the XlaEnvPreflight Mix compiler eagerly so Mix can find it before
# the project's own `lib/` tree compiles. Compilers referenced from a
# project's `:compilers` list must be loadable before `mix compile`
# starts; placing the module under `lib/` would create a chicken-and-egg
# bootstrap problem (Mix would look for the compiler module before it
# has had a chance to compile `lib/`).
unless Code.ensure_loaded?(Mix.Tasks.Compile.XlaEnvPreflight) do
  Code.require_file("build_support/mix_tasks_compile_xla_env_preflight.exs", __DIR__)
end

# Eager XLA_TARGET preflight: a project's :compilers list runs AFTER
# dependency compilation, so the in-project preflight compiler alone
# would not catch the EXLA-side failure mode. Calling the validator
# here, at mix.exs top level, catches mix test, mix compile,
# mix deps.compile, and mix deps.update -- all of which evaluate
# mix.exs before touching deps.
#
# When this mix.exs is being evaluated as a transitive dependency
# (our directory sits inside some parent project's deps/), defer
# entirely to that parent project's build configuration.
XlaTargetValidator.validate_root_project!(__DIR__)

defmodule TrinityCoordinator.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :trinity_coordinator,
      version: @version,
      elixir: "~> 1.18",
      description:
        "An Elixir implementation of the TRINITY multi-agent orchestration router for routing language-model calls through a compact hidden-state router.",
      source_url: "https://github.com/nshkrdotcom/trinity_coordinator",
      homepage_url: "https://github.com/nshkrdotcom/trinity_coordinator",
      start_permanent: Mix.env() == :prod,
      package: package(),
      docs: docs(),
      compilers: [:xla_env_preflight] ++ Mix.compilers(),
      elixirc_paths: elixirc_paths(Mix.env()),
      dialyzer: [
        plt_add_apps: [:mix],
        plt_file: {:no_warn, "priv/plts/dialyzer.plt"}
      ],
      deps: deps()
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  def cli do
    [
      preferred_envs: [
        dialyzer: :dev,
        credo: :dev
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
      {:nx, "~> 0.9"},
      {:axon, "~> 0.7"},
      # Pinned to a Qwen3-supporting commit until a Bumblebee Hex release
      # lands that includes it. To unpin, follow
      # docs/bumblebee_unpin_playbook.md.
      {:bumblebee,
       github: "elixir-nx/bumblebee",
       ref: "0fd8114cf5429af9236f100f3350986e9d823c02",
       override: true},
      {:exla, "~> 0.9"},
      DependencySources.dep(:inference, __DIR__),
      DependencySources.dep(:agent_session_manager, __DIR__),
      DependencySources.dep(:gemini_cli_sdk, __DIR__),
      {:req, "~> 0.5"},
      {:hf_hub, "~> 0.2"},
      {:credo, "~> 1.7", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: :dev, runtime: false},
      {:ex_doc, "~> 0.38", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      name: :trinity_coordinator,
      licenses: ["MIT"],
      maintainers: ["nshkrdotcom"],
      links: %{
        GitHub: "https://github.com/nshkrdotcom/trinity_coordinator"
      },
      files: [
        "lib",
        "mix.exs",
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        "AGENTS.md",
        "assets",
        "build_support",
        "examples",
        "guides",
        "docs/*.md"
      ]
    ]
  end

  defp docs do
    [
      main: "overview",
      source_ref: "v#{@version}",
      source_url: "https://github.com/nshkrdotcom/trinity_coordinator",
      extras: [
        {"README.md", [filename: "overview", title: "Overview"]},
        {"examples/README.md", [filename: "examples", title: "Examples"]},
        "guides/onboarding.md",
        "guides/current_direction.md",
        "guides/system_architecture.md",
        "guides/python_parity_reconstruction.md",
        "guides/stage_checks_and_tolerances.md",
        "guides/artifacts_and_export.md",
        "guides/svd_generation_runbook.md",
        "guides/service_buildout.md",
        "guides/provider_service_hardening.md",
        "guides/operations_qc.md",
        "guides/troubleshooting.md",
        "docs/sakana_svd_byte_match_rigor_plan.md",
        "docs/sakana_svd_parity_debug_checklist.md",
        "docs/elixir_svd_decomposition.md",
        "docs/production_qwen_slm_profile.md",
        "docs/coordination_head_variants.md",
        "docs/trace_persistence.md",
        "docs/configurable_provider_pools.md",
        "docs/agent_slot_provider_mapping.md",
        "docs/production_runbook.md",
        "docs/bumblebee_unpin_playbook.md",
        "docs/provider_smoke_tests.md",
        "CHANGELOG.md",
        "LICENSE"
      ],
      groups_for_extras: [
        Project: ["README.md", "CHANGELOG.md", "LICENSE"],
        Examples: ["examples/README.md"],
        "Start Here": [
          "guides/onboarding.md",
          "guides/current_direction.md",
          "guides/system_architecture.md"
        ],
        "Parity Guides": [
          "guides/python_parity_reconstruction.md",
          "guides/stage_checks_and_tolerances.md",
          "guides/artifacts_and_export.md",
          "guides/svd_generation_runbook.md"
        ],
        "Service Buildout": [
          "guides/service_buildout.md",
          "guides/provider_service_hardening.md",
          "guides/operations_qc.md",
          "guides/troubleshooting.md"
        ],
        "Operator Runbooks": [
          "docs/agent_slot_provider_mapping.md",
          "docs/production_runbook.md",
          "docs/bumblebee_unpin_playbook.md"
        ],
        "Reference Notes": [
          "docs/sakana_svd_byte_match_rigor_plan.md",
          "docs/sakana_svd_parity_debug_checklist.md",
          "docs/elixir_svd_decomposition.md",
          "docs/production_qwen_slm_profile.md",
          "docs/coordination_head_variants.md",
          "docs/trace_persistence.md",
          "docs/configurable_provider_pools.md",
          "docs/provider_smoke_tests.md"
        ]
      ],
      groups_for_modules: [
        Core: [
          TrinityCoordinator,
          TrinityCoordinator.Extractor,
          TrinityCoordinator.CoordinationHead,
          TrinityCoordinator.Orchestrator
        ],
        Runtime: [
          TrinityCoordinator.Runtime,
          TrinityCoordinator.StateManager,
          TrinityCoordinator.RoleInjector,
          TrinityCoordinator.Thinker,
          TrinityCoordinator.Verifier,
          TrinityCoordinator.AgentPool,
          TrinityCoordinator.AgentPool.Adapter,
          TrinityCoordinator.AgentPool.Inference,
          TrinityCoordinator.AgentPool.OpenAI
        ]
      ],
      skip_undefined_reference_warnings_on: ["CHANGELOG.md"]
    ]
  end
end
