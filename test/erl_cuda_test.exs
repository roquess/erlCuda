defmodule ErlCudaTest do
  use ExUnit.Case

  test "native module loads and returns the not_implemented stub" do
    assert {:error, :not_implemented} = ErlCuda.Native.launch_vector_add([1.0], [2.0])
  end
end
