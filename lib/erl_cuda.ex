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

  @doc """
  Synchronous wrapper around `launch/2`. Raises if the kernel errors or the
  result does not arrive within `timeout` milliseconds.
  """
  def launch!(kernel, args, timeout \\ 5_000) do
    {:ok, job_id} = launch(kernel, args)

    receive do
      {:erlcuda, ^job_id, {:ok, result}} -> result
      {:erlcuda, ^job_id, {:error, reason}} -> raise "GPU kernel failed: #{inspect(reason)}"
    after
      timeout -> raise "GPU kernel timed out after #{timeout}ms"
    end
  end
end
