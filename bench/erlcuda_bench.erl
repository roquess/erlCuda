%% Full-stack vector_add latency benchmark: erlcuda:launch_sync/2 -> NIF ->
%% mpsc channel -> GPU worker thread -> message send -> receive. Compare
%% against native/erlcuda_nif/src/bin/bench_pure_cuda.rs, which measures the
%% same computation with no BEAM/NIF/channel involved at all. See the
%% Benchmarks section of README.md for how to run both and how to read the
%% results.
-module(erlcuda_bench).

-export([run/0]).

-define(VECTOR_LEN, 1000).
-define(REPETITIONS, 100).

run() ->
    A = [I * 1.0 || I <- lists:seq(0, ?VECTOR_LEN - 1)],
    B = [I * 2.0 || I <- lists:seq(0, ?VECTOR_LEN - 1)],

    % Warm-up: absorbs the one-time cost of spawning the device-0 worker
    % thread and constructing its CudaBackend (CUDA context creation, PTX
    % module JIT-loading), neither of which reflects steady-state per-job
    % latency.
    _ = erlcuda:launch_sync(vector_add, [A, B]),

    TotalUs = lists:foldl(
        fun(_, Acc) ->
            {TimeUs, _Result} = timer:tc(fun() -> erlcuda:launch_sync(vector_add, [A, B]) end),
            Acc + TimeUs
        end,
        0,
        lists:seq(1, ?REPETITIONS)
    ),

    MeanUs = TotalUs / ?REPETITIONS,
    io:format("erlcuda_mean_us: ~.1f~n", [MeanUs]).
