mod atoms {
    rustler::atoms! {
        error,
        not_implemented,
    }
}

#[rustler::nif]
fn launch_vector_add(_a: Vec<f32>, _b: Vec<f32>) -> (rustler::Atom, rustler::Atom) {
    (atoms::error(), atoms::not_implemented())
}

rustler::init!("Elixir.ErlCuda.Native");
