use std::env;
use std::path::PathBuf;

use cuda_builder::CudaBuilder;

const CODEGEN_NVVM_DYLIB_NAMES: [&str; 3] = [
    "rustc_codegen_nvvm.dll",
    "librustc_codegen_nvvm.so",
    "librustc_codegen_nvvm.dylib",
];

/// The Rust-CUDA commit this crate's `cust`/`cuda_builder` git dependencies
/// (and `cust`'s own transitive `cust_raw`) are pinned to (see `Cargo.toml`'s
/// `rev = "..."` fields).
/// Cargo's git-checkout cache under `.cargo/git/checkouts/` is keyed by
/// repository URL, not by revision, so a machine that has ever built
/// `rustc_codegen_nvvm` for a *different* rev of `Rust-GPU/rust-cuda` (e.g.
/// for an unrelated project) could otherwise have its checkout picked up
/// here too. Each checkout's revision subdirectory is named after Cargo's
/// abbreviated (short) commit id, so this is checked as a prefix match.
const PINNED_RUST_CUDA_REV: &str = "6a836d9236fc38e0fa7a71f7bdeda7a8f82bc8d5";

fn is_codegen_nvvm_already_on_path() -> bool {
    let Some(path_var) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&path_var)
        .any(|dir| CODEGEN_NVVM_DYLIB_NAMES.iter().any(|name| dir.join(name).is_file()))
}

fn cargo_home() -> Option<PathBuf> {
    if let Some(dir) = env::var_os("CARGO_HOME") {
        return Some(PathBuf::from(dir));
    }
    let home = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".cargo"))
}

/// Searches `<cargo home>/git/checkouts/rust-cuda-*/*/target/release/` for an
/// already-built `rustc_codegen_nvvm` dynamic library, from any prior manual
/// build following this project's README instructions. Returns the
/// directory containing it, if found. Only a checkout of `PINNED_RUST_CUDA_REV`
/// itself is accepted (see that constant's doc comment) — this only ever
/// finds a build made from this project's own pinned commit, via Cargo's own
/// git-dependency checkout mechanism (not, for example, an independently
/// `git clone`d copy built elsewhere).
fn find_prebuilt_codegen_nvvm_dir() -> Option<PathBuf> {
    let checkouts_dir = cargo_home()?.join("git").join("checkouts");
    for repo_entry in std::fs::read_dir(&checkouts_dir).ok()?.flatten() {
        if !repo_entry.file_name().to_string_lossy().starts_with("rust-cuda-") {
            continue;
        }
        let Ok(rev_entries) = std::fs::read_dir(repo_entry.path()) else {
            continue;
        };
        for rev_entry in rev_entries.flatten() {
            let rev_dir_name = rev_entry.file_name();
            if !PINNED_RUST_CUDA_REV.starts_with(rev_dir_name.to_string_lossy().as_ref()) {
                continue;
            }
            let release_dir = rev_entry.path().join("target").join("release");
            if CODEGEN_NVVM_DYLIB_NAMES.iter().any(|name| release_dir.join(name).is_file()) {
                return Some(release_dir);
            }
        }
    }
    None
}

/// Auto-extends this build script's own `PATH` with an already-built
/// `rustc_codegen_nvvm` (if found) and CUDA's `nvvm/bin` (from `CUDA_PATH`),
/// so `CudaBuilder`'s nested `cargo` invocation (which inherits this
/// process's environment) can find the codegen backend without requiring
/// the developer to manually export `PATH` every shell session. If nothing
/// is found, this is a no-op and the existing manual-setup error path
/// (documented in README.md) still applies unchanged.
fn extend_path_for_cuda_codegen_backend() {
    if is_codegen_nvvm_already_on_path() {
        return;
    }
    let Some(codegen_dir) = find_prebuilt_codegen_nvvm_dir() else {
        return;
    };

    let mut dirs = vec![codegen_dir];
    if let Some(cuda_path) = env::var_os("CUDA_PATH") {
        dirs.push(PathBuf::from(cuda_path).join("nvvm").join("bin"));
    }
    if let Some(existing) = env::var_os("PATH") {
        dirs.extend(env::split_paths(&existing));
    }

    if let Ok(new_path) = env::join_paths(dirs) {
        println!(
            "cargo::warning=erlCuda: auto-added a previously-built rustc_codegen_nvvm to PATH for this build"
        );
        unsafe {
            env::set_var("PATH", new_path);
        }
    }
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=../../kernels");

    extend_path_for_cuda_codegen_backend();

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
