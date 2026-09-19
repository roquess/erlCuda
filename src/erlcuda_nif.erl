%% Hand-rolled NIF loader, replacing what Rustler's Mix-side compiler used
%% to do automatically. The Rust crate itself (`native/erlcuda_nif/`) is a
%% completely generic Rustler/NIF-ABI shared library — nothing about it is
%% Elixir-specific; only the BEAM-side loading glue differs between Mix and
%% plain Erlang, which is what this module provides.
-module(erlcuda_nif).

-on_load(init/0).

-export([launch_vector_add/3, launch_reduce/2, launch_dot_product/3, launch_matmul/6]).

init() ->
    SoName = filename:join(code:priv_dir(erlcuda), "erlcuda_nif"),
    erlang:load_nif(SoName, 0).

launch_vector_add(_A, _B, _Device) -> erlang:nif_error(nif_not_loaded).
launch_reduce(_A, _Device) -> erlang:nif_error(nif_not_loaded).
launch_dot_product(_A, _B, _Device) -> erlang:nif_error(nif_not_loaded).
launch_matmul(_A, _B, _M, _N, _K, _Device) -> erlang:nif_error(nif_not_loaded).
