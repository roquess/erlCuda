pub trait Backend {
    fn vector_add(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String>;
}

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn vector_add(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        if a.len() != b.len() {
            return Err(format!("length mismatch: {} vs {}", a.len(), b.len()));
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_equal_length_vectors() {
        let mut backend = CpuBackend;
        let result = backend
            .vector_add(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0])
            .unwrap();
        assert_eq!(result, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let mut backend = CpuBackend;
        let err = backend.vector_add(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(err.contains("length mismatch"));
    }
}
