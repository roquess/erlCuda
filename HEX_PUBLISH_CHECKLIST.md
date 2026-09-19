# hex.pm Publish Checklist (erlcuda 0.1.0)

One-time runbook for publishing `erlcuda` to hex.pm. Delete this file after
publishing if you don't want it lingering in the repo (or keep it as a
template for future 0.2.0+ releases).

## Already verified (2026-09-19, clean-state run)

- `rebar3 compile`: clean, no warnings beyond the known-benign
  `rustc_codegen_nvvm` PATH message.
- `rebar3 eunit`: **10/10** tests passed.
- `cargo test --release` (in `native/erlcuda_nif`): **29/29** tests passed
  (2 suites).
- `rebar3 hex build -u -o <scratch dir>` tarball contents inspected: clean.
  No `native/erlcuda_nif/target/`, no `kernels/target/`, no `.wolf/`, no
  stale `priv/erlcuda_nif.dll`. All real source (Rust sources, `Cargo.toml`/
  `Cargo.lock`, `build.rs`, Erlang sources, `rebar.config`, `rebar.lock`,
  `README.md`, `CHANGELOG.md`, `LICENSE`) present.
- Docs build (`ex_doc`) generated successfully alongside the package tarball.

You are not starting from scratch — the package is publish-ready as far as
automated checks can confirm.

## One-time setup

- [ ] `rebar3 hex user auth` (interactive; needs your hex.pm account
      credentials and a local encryption passphrase for the stored API key).
- [ ] Confirm this is the **intended hex.pm account** (personal vs. an
      org/team account) before proceeding — not something this checklist
      can determine for you.

## Before publishing

- [ ] Final skim of `CHANGELOG.md` for accuracy (version, date, entries).
- [ ] Final skim of `README.md` for accuracy (install instructions, badges,
      examples still match the code).
- [ ] Remember: **hex.pm publishes are effectively permanent** for a given
      version number. A version can be "retired" (flagged/deprecated) but
      not deleted or reused. If in doubt, double-check before confirming.

## Publish

```bash
rebar3 hex publish
```

- Review the printed file list carefully when prompted — this is your last
  chance to catch anything unexpected in the package contents.
- Confirm when prompted to actually push the release to hex.pm.

## After publishing

- [ ] Verify the package page on `https://hex.pm/packages/erlcuda`.
- [ ] Verify docs rendered correctly on `https://hexdocs.pm/erlcuda`.
- [ ] Optionally delete this file (`HEX_PUBLISH_CHECKLIST.md`) — its job is
      done once 0.1.0 is live. Keep it around if you'd rather reuse/update
      it for the next release.
