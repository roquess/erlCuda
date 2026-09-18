defmodule ErlCuda do
  @moduledoc """
  Public API for launching GPU kernels from Erlang/Elixir.
  """

  alias ErlCuda.Native

  @doc """
  Launches a named kernel asynchronously. Returns `{:ok, job_id}` immediately,
  or `{:error, :invalid_device}` if `opts[:device]` doesn't exist on this
  machine. The caller later receives
  `{:erlcuda, job_id, {:ok, result} | {:error, reason}}`.

  ## Options

    * `:device` - GPU device ordinal to run on. Defaults to `0`.
  """
  def launch(:vector_add, [a, b], opts \\ []) do
    device = Keyword.get(opts, :device, 0)
    Native.launch_vector_add(a, b, device)
  end

  @doc """
  Synchronous wrapper around `launch/3`. Raises if the kernel errors or the
  result does not arrive within `opts[:timeout]` milliseconds.

  ## Options

    * `:device` - GPU device ordinal to run on. Defaults to `0`.
    * `:timeout` - milliseconds to wait for the result. Defaults to `5_000`.
  """
  def launch!(kernel, args, opts \\ []) do
    timeout = Keyword.get(opts, :timeout, 5_000)

    case launch(kernel, args, opts) do
      {:ok, job_id} ->
        receive do
          {:erlcuda, ^job_id, {:ok, result}} -> result
          {:erlcuda, ^job_id, {:error, reason}} -> raise "GPU kernel failed: #{inspect(reason)}"
        after
          timeout -> raise "GPU kernel timed out after #{timeout}ms"
        end

      {:error, reason} ->
        raise "GPU kernel failed to launch: #{inspect(reason)}"
    end
  end
end
