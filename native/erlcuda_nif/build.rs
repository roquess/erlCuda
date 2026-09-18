use std::env;
use std::path::PathBuf;

use cuda_builder::CudaBuilder;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=../../kernels");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernels_dir = manifest_dir.join("../../kernels");

    // `cuda_builder` shells out to a nested `cargo build` in `kernels_dir` to
    // compile the kernel crate with the nightly toolchain pinned by
    // `kernels/rust-toolchain.toml`. Cargo/rustup set `RUSTUP_TOOLCHAIN` in the
    // environment of this build script (since *this* crate is built with
    // stable), and that env var propagates to the nested `cargo` invocation,
    // overriding the directory-based nightly override and making the nested
    // build fail with "the `-Z` flag is only accepted on the nightly channel".
    // Removing it here lets rustup's normal directory-based toolchain
    // resolution (via `kernels/rust-toolchain.toml`) take effect for the
    // nested build, while this crate itself keeps building on stable.
    unsafe {
        env::remove_var("RUSTUP_TOOLCHAIN");
        env::remove_var("RUSTC");
        env::remove_var("RUSTC_WRAPPER");
        env::remove_var("RUSTC_WORKSPACE_WRAPPER");
        env::remove_var("RUSTDOC");
    }

    CudaBuilder::new(kernels_dir)
        .copy_to(out_path.join("kernels.ptx"))
        .build()
        .unwrap();
}
