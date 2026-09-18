vector_len = 1_000
repetitions = 100

a = for i <- 0..(vector_len - 1), do: i * 1.0
b = for i <- 0..(vector_len - 1), do: i * 2.0

# Warm-up: absorbs the one-time cost of spawning the device-0 worker thread
# and constructing its CudaBackend (CUDA context creation, PTX module
# JIT-loading), neither of which reflects steady-state per-job latency.
ErlCuda.launch!(:vector_add, [a, b])

total_us =
  Enum.reduce(1..repetitions, 0, fn _, acc ->
    {time_us, _result} = :timer.tc(fn -> ErlCuda.launch!(:vector_add, [a, b]) end)
    acc + time_us
  end)

mean_us = total_us / repetitions
IO.puts("erlcuda_mean_us: #{:erlang.float_to_binary(mean_us, decimals: 1)}")
