defmodule ErlCuda do
  @moduledoc """
  Public API for launching GPU kernels from Erlang/Elixir.
  """

  alias ErlCuda.Native

  @doc """
  Launches a named kernel asynchronously. Returns `{:ok, job_id}` immediately.
  The caller later receives `{:erlcuda, job_id, {:ok, result} | {:error, reason}}`.
  """
  def launch(:vector_add, [a, b]) do
    Native.launch_vector_add(a, b)
  end
end
