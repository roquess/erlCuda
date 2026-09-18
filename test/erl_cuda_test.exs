defmodule ErlCudaTest do
  use ExUnit.Case

  test "launch/2 computes vector_add asynchronously" do
    {:ok, job_id} = ErlCuda.launch(:vector_add, [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    assert_receive {:erlcuda, ^job_id, {:ok, [11.0, 22.0, 33.0]}}, 1_000
  end

  test "launch!/2 returns the result synchronously" do
    assert ErlCuda.launch!(:vector_add, [[1.0, 2.0], [3.0, 4.0]]) == [4.0, 6.0]
  end

  test "launch!/2 raises on mismatched lengths" do
    assert_raise RuntimeError, ~r/length mismatch/, fn ->
      ErlCuda.launch!(:vector_add, [[1.0], [1.0, 2.0]])
    end
  end

  test "launch/2 correctly correlates results for many concurrent jobs" do
    jobs =
      for i <- 1..10 do
        # Each job's inputs are chosen so its expected sum is unique across
        # the whole batch, so a cross-talk bug (job N receiving job M's
        # result) would produce a mismatched sum instead of silently passing.
        a = [i * 1.0]
        b = [i * 10.0]
        {:ok, job_id} = ErlCuda.launch(:vector_add, [a, b])
        {job_id, i * 11.0}
      end

    # All job ids must be distinct, otherwise the assertions below couldn't
    # tell correlated results apart from coincidentally-matching ones.
    job_ids = Enum.map(jobs, fn {job_id, _expected_sum} -> job_id end)
    assert length(Enum.uniq(job_ids)) == length(job_ids)

    for {job_id, expected_sum} <- jobs do
      assert_receive {:erlcuda, ^job_id, {:ok, [^expected_sum]}}, 1_000
    end
  end

  test "launch/3 runs on device 0 explicitly" do
    assert ErlCuda.launch!(:vector_add, [[1.0, 2.0], [3.0, 4.0]], device: 0) == [4.0, 6.0]
  end

  test "launch/3 returns {:error, :invalid_device} for an out-of-range device, synchronously" do
    assert {:error, :invalid_device} = ErlCuda.launch(:vector_add, [[1.0], [1.0]], device: 99)
  end

  test "launch!/3 raises for an out-of-range device without ever entering receive" do
    # No job is ever enqueued for an invalid device, so launch!/3 must raise
    # from the {:error, reason} branch synchronously, before it would block
    # on `receive`. If it incorrectly fell through to `receive`, this test
    # would hang until the default 5_000ms timeout instead of failing fast.
    assert_raise RuntimeError, ~r/GPU kernel failed to launch/, fn ->
      ErlCuda.launch!(:vector_add, [[1.0], [1.0]], device: 99)
    end
  end

  test "the Erlang wrapper's launch/3 computes vector_add asynchronously" do
    {:ok, job_id} = :erlcuda.launch(:vector_add, [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], device: 0)

    assert_receive {:erlcuda, ^job_id, {:ok, [11.0, 22.0, 33.0]}}, 1_000
  end

  test "the Erlang wrapper's launch_sync/2 returns the result synchronously" do
    assert :erlcuda.launch_sync(:vector_add, [[1.0, 2.0], [3.0, 4.0]]) == [4.0, 6.0]
  end
end
