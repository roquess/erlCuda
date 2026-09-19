-module(erlcuda_tests).

-include_lib("eunit/include/eunit.hrl").

launch_computes_vector_add_asynchronously_test() ->
    {ok, JobId} = erlcuda:launch(vector_add, [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
    receive
        {erlcuda, JobId, {ok, Result}} ->
            ?assertEqual([11.0, 22.0, 33.0], Result)
    after 1000 ->
        ?assert(false)
    end.

launch_sync_returns_the_result_synchronously_test() ->
    ?assertEqual([4.0, 6.0], erlcuda:launch_sync(vector_add, [[1.0, 2.0], [3.0, 4.0]])).

launch_sync_raises_on_mismatched_lengths_test() ->
    ?assertError(
        {gpu_kernel_failed, Reason},
        erlcuda:launch_sync(vector_add, [[1.0], [1.0, 2.0]])
    ),
    % Re-run just to inspect the reason via a catch, proving it actually
    % mentions the length mismatch rather than some unrelated failure.
    try
        erlcuda:launch_sync(vector_add, [[1.0], [1.0, 2.0]])
    catch
        error:{gpu_kernel_failed, Msg} ->
            ?assert(is_binary(Msg) orelse is_list(Msg)),
            MsgStr = case is_binary(Msg) of
                true -> binary_to_list(Msg);
                false -> Msg
            end,
            ?assert(string:str(MsgStr, "length mismatch") > 0)
    end.

launch_correlates_results_for_many_concurrent_jobs_test() ->
    Jobs = [
        begin
            % Each job's inputs are chosen so its expected sum is unique
            % across the whole batch, so a cross-talk bug (job N receiving
            % job M's result) would produce a mismatched sum instead of
            % silently passing.
            A = [I * 1.0],
            B = [I * 10.0],
            {ok, JobId} = erlcuda:launch(vector_add, [A, B]),
            {JobId, I * 11.0}
        end
        || I <- lists:seq(1, 10)
    ],
    JobIds = [JobId || {JobId, _ExpectedSum} <- Jobs],
    ?assertEqual(length(lists:usort(JobIds)), length(JobIds)),
    lists:foreach(
        fun({JobId, ExpectedSum}) ->
            receive
                {erlcuda, JobId, {ok, [ExpectedSum]}} -> ok
            after 1000 ->
                ?assert(false)
            end
        end,
        Jobs
    ).

launch_runs_on_device_0_explicitly_test() ->
    ?assertEqual(
        [4.0, 6.0],
        erlcuda:launch_sync(vector_add, [[1.0, 2.0], [3.0, 4.0]], [{device, 0}])
    ).

launch_returns_invalid_device_for_out_of_range_device_synchronously_test() ->
    ?assertEqual(
        {error, invalid_device},
        erlcuda:launch(vector_add, [[1.0], [1.0]], [{device, 99}])
    ).

launch_sync_raises_for_out_of_range_device_without_ever_receiving_test() ->
    % No job is ever enqueued for an invalid device, so launch_sync/3 must
    % raise from the {error, Reason} branch synchronously, before it would
    % block on `receive`. If it incorrectly fell through to `receive`, this
    % test would hang until the default 5000ms timeout instead of failing
    % fast.
    ?assertError(
        {gpu_kernel_launch_failed, invalid_device},
        erlcuda:launch_sync(vector_add, [[1.0], [1.0]], [{device, 99}])
    ).

launch_sync_reduces_a_vector_test() ->
    ?assertEqual([10.0], erlcuda:launch_sync(reduce, [[1.0, 2.0, 3.0, 4.0]])).

launch_sync_computes_a_dot_product_test() ->
    ?assertEqual(
        [32.0],
        erlcuda:launch_sync(dot_product, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ).

launch_sync_multiplies_matrices_test() ->
    % [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]], flattened row-major
    A = [1.0, 2.0, 3.0, 4.0],
    B = [5.0, 6.0, 7.0, 8.0],
    ?assertEqual([19.0, 22.0, 43.0, 50.0], erlcuda:launch_sync(matmul, [A, B, 2, 2, 2])).
