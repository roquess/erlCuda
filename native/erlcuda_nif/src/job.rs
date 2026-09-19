use rustler::types::LocalPid;

/// One request to run a GPU kernel. New kernel types add a variant here —
/// see `docs/superpowers/specs/2026-09-19-additional-kernels-design.md` for
/// the full set planned beyond `VectorAdd`.
pub enum Command {
    VectorAdd { a: Vec<f32>, b: Vec<f32> },
    Reduce { a: Vec<f32> },
    DotProduct { a: Vec<f32>, b: Vec<f32> },
}

pub struct Job {
    pub id: u64,
    pub pid: LocalPid,
    pub command: Command,
}
