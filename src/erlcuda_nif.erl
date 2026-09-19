-module(erlcuda_nif).

-moduledoc """
Internal NIF-loading layer for `erlcuda`. Hand-rolled, replacing what
Rustler's Mix-side compiler used to do automatically. The Rust crate itself
(`native/erlcuda_nif/`) is a completely generic Rustler/NIF-ABI shared
library — nothing about it is Elixir-specific; only the BEAM-side loading
glue differs between Mix and plain Erlang, which is what this module
provides.

End users should call `erlcuda` instead of this module directly — the stub
functions below only exist to be replaced by native code via `-on_load/1`.
""".

-on_load(init/0).

-export([launch_vector_add/3, launch_reduce/2, launch_dot_product/3, launch_matmul/6]).

-doc false.
init() ->
    SoName = filename:join(code:priv_dir(erlcuda), "erlcuda_nif"),
    erlang:load_nif(SoName, 0).

-doc false.
launch_vector_add(_A, _B, _Device) -> erlang:nif_error(nif_not_loaded).
-doc false.
launch_reduce(_A, _Device) -> erlang:nif_error(nif_not_loaded).
-doc false.
launch_dot_product(_A, _B, _Device) -> erlang:nif_error(nif_not_loaded).
-doc false.
launch_matmul(_A, _B, _M, _N, _K, _Device) -> erlang:nif_error(nif_not_loaded).
