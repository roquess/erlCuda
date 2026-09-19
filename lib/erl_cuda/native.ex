defmodule ErlCuda.Native do
  use Rustler, otp_app: :erl_cuda, crate: "erlcuda_nif"

  def launch_vector_add(_a, _b, _device), do: :erlang.nif_error(:nif_not_loaded)
  def launch_reduce(_a, _device), do: :erlang.nif_error(:nif_not_loaded)
  def launch_dot_product(_a, _b, _device), do: :erlang.nif_error(:nif_not_loaded)
  def launch_matmul(_a, _b, _m, _n, _k, _device), do: :erlang.nif_error(:nif_not_loaded)
end
