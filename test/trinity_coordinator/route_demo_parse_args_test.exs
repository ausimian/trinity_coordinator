defmodule Mix.Tasks.Trinity.Route.Demo.ParseArgsTest do
  use ExUnit.Case, async: true

  alias Mix.Tasks.Trinity.Route.Demo

  describe "parse_args!/1 — mock alias" do
    test "--mock-provider sets mock?: true" do
      opts = Demo.parse_args!(["--mock-provider"])
      assert opts.mock? == true
    end

    test "--mock still sets mock?: true (alias)" do
      opts = Demo.parse_args!(["--mock"])
      assert opts.mock? == true
    end

    test "both flags together are accepted and equivalent" do
      opts = Demo.parse_args!(["--mock-provider", "--mock"])
      assert opts.mock? == true
    end

    test "neither flag and without --allow-live leaves mock?: false" do
      opts = Demo.parse_args!([])
      assert opts.mock? == false
      assert opts.allow_live? == false
    end

    test "--allow-live without mock leaves mock?: false" do
      opts = Demo.parse_args!(["--allow-live"])
      assert opts.mock? == false
      assert opts.allow_live? == true
    end
  end
end
