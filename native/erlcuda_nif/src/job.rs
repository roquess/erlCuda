use rustler::types::LocalPid;

pub struct Job {
    pub id: u64,
    pub pid: LocalPid,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}
