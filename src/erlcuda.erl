%% Pure-delegation Erlang entry point for erlCuda, so Erlang callers don't
%% need to write 'Elixir.ErlCuda':launch(...) directly. No logic lives here:
%% every function forwards straight to the ErlCuda Elixir module. Elixir
%% keyword lists and Erlang proplists share the same representation
%% ([{atom(), term()}]), so options like [{device, 1}] pass through
%% unchanged in either direction.
-module(erlcuda).

-export([launch/2, launch/3, launch_sync/2, launch_sync/3]).

launch(Kernel, Args) ->
    launch(Kernel, Args, []).

launch(Kernel, Args, Opts) ->
    'Elixir.ErlCuda':launch(Kernel, Args, Opts).

%% Delegates to Elixir's launch!/2, named without the trailing "!" since
%% that's the Erlang send operator, not an idiomatic function-name
%% character.
launch_sync(Kernel, Args) ->
    launch_sync(Kernel, Args, []).

launch_sync(Kernel, Args, Opts) ->
    'Elixir.ErlCuda':'launch!'(Kernel, Args, Opts).
