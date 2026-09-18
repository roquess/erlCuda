defmodule ErlCuda.Native do
  use Rustler, otp_app: :erl_cuda, crate: "erlcuda_nif"

  def launch_vector_add(_a, _b, _device), do: :erlang.nif_error(:nif_not_loaded)
end
