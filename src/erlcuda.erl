-module(erlcuda).

-moduledoc """
Launches GPU kernels (`vector_add`, `reduce`, `dot_product`, `matmul`) from
Erlang, either asynchronously (`launch/2,3`) or synchronously (`launch_sync/2,3`).

Each kernel runs on a dedicated GPU worker thread; jobs targeting the same
device and kernel type may be opportunistically batched together.
""".

-export([launch/2, launch/3, launch_sync/2, launch_sync/3]).

-doc """
Equivalent to `launch/3` with no options (so `device` defaults to `0`).
""".
-spec launch(atom(), list()) -> {ok, non_neg_integer()} | {error, term()}.
launch(Kernel, Args) ->
    launch(Kernel, Args, []).

-doc """
Launches a named kernel asynchronously. Returns `{ok, JobId}` immediately,
or `{error, invalid_device}` if the `device` option doesn't exist on this
machine. The caller later receives `{erlcuda, JobId, {ok, Result} |
{error, Reason}}`.

Kernels:
- `launch(vector_add, [A, B], Opts)` - elementwise addition
- `launch(reduce, [A], Opts)` - parallel sum reduction
- `launch(dot_product, [A, B], Opts)` - dot product
- `launch(matmul, [A, B, M, N, K], Opts)` - `C = A * B`, `A` is `MxK`,
  `B` is `KxN`, both flattened row-major

Options:
- `device` - GPU device ordinal to run on. Defaults to `0`.
""".
-spec launch(atom(), list(), proplists:proplist()) ->
    {ok, non_neg_integer()} | {error, term()}.
launch(vector_add, [A, B], Opts) ->
    Device = proplists:get_value(device, Opts, 0),
    erlcuda_nif:launch_vector_add(A, B, Device);
launch(reduce, [A], Opts) ->
    Device = proplists:get_value(device, Opts, 0),
    erlcuda_nif:launch_reduce(A, Device);
launch(dot_product, [A, B], Opts) ->
    Device = proplists:get_value(device, Opts, 0),
    erlcuda_nif:launch_dot_product(A, B, Device);
launch(matmul, [A, B, M, N, K], Opts) ->
    Device = proplists:get_value(device, Opts, 0),
    erlcuda_nif:launch_matmul(A, B, M, N, K, Device).

-doc """
Equivalent to `launch_sync/3` with no options (so `device` defaults to `0`
and `timeout` defaults to `5000` milliseconds).
""".
-spec launch_sync(atom(), list()) -> list().
launch_sync(Kernel, Args) ->
    launch_sync(Kernel, Args, []).

-doc """
Synchronous wrapper around `launch/3`. Raises (via `erlang:error/1`) if the
kernel errors or the result does not arrive within the `timeout` option
(milliseconds, default `5000`). Also raises immediately, without ever
entering `receive`, if the launch itself is rejected (e.g. an invalid
`device`). Named `launch_sync` rather than `launch!` since `!` is Erlang's
send operator, not an idiomatic function-name character.
""".
-spec launch_sync(atom(), list(), proplists:proplist()) -> list().
launch_sync(Kernel, Args, Opts) ->
    Timeout = proplists:get_value(timeout, Opts, 5000),
    case launch(Kernel, Args, Opts) of
        {ok, JobId} ->
            receive
                {erlcuda, JobId, {ok, Result}} -> Result;
                {erlcuda, JobId, {error, Reason}} ->
                    erlang:error({gpu_kernel_failed, Reason})
            after Timeout ->
                erlang:error({gpu_kernel_timeout, Timeout})
            end;
        {error, Reason} ->
            erlang:error({gpu_kernel_launch_failed, Reason})
    end.
