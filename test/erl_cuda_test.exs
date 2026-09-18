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
end
