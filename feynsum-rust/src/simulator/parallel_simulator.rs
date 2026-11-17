mod state;
mod state_expander;

pub use state::State;

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::circuit::Circuit;
use crate::config::Config;
use crate::gate_scheduler;
use crate::kernel_stats::{json_usizes, secs};
use crate::profile;
use crate::types::{AtomicBasisIdx, BasisIdx, Complex, Real};

pub use state::SparseStateTable;
pub use state_expander::expand_sparse;
pub use state_expander::{ExpandMethod, ExpandResult};

pub fn run<B: BasisIdx, AB: AtomicBasisIdx<B>>(
    config: &Config,
    circuit: Circuit<B>,
) -> State<B, AB> {
    let num_gates = circuit.num_gates();
    let num_qubits = circuit.num_qubits;

    let mut num_gates_visited = 0;
    let mut state = initial_state::<B, AB>(config, num_qubits);
    let mut num_nonzeros = state.num_nonzeros();
    let mut num_gate_apps = 0;
    let mut prev_num_nonzeros = num_nonzeros;
    let mut expand_time = 0.0;
    let mut kernel_idx = 0;

    let mut stats_out = config.kernel_stats.as_ref().map(|path| {
        BufWriter::new(File::create(path).expect("cannot create --kernel-stats file"))
    });

    let mut gate_scheduler = gate_scheduler::create_gate_scheduler(config, &circuit);

    log::info!("starting gate application loop.");

    let (duration, _) = profile!(loop {
        let these_gate_indices = gate_scheduler.pick_next_gates();
        let these_gates = these_gate_indices
            .iter()
            .map(|&idx| &circuit.gates[idx])
            .collect::<Vec<_>>();

        log::debug!("applying gates: {:?}", these_gates);

        if these_gates.is_empty() {
            break;
        }

        let num_gates_visited_here = these_gates.len();

        let ExpandResult {
            state: new_state,
            num_nonzeros: new_num_nonzeros,
            num_gate_apps: num_gate_apps_here,
            method,
            duration,
            phases,
            walk,
        } = state_expander::expand(these_gates, config, num_qubits, prev_num_nonzeros, state);
        expand_time += secs(duration);

        if let Some(out) = stats_out.as_mut() {
            let method_name = match method {
                ExpandMethod::Sparse => "sparse",
                ExpandMethod::PushDense => "push_dense",
                ExpandMethod::PullDense => "pull_dense",
            };
            let walk_fields = match walk {
                Some(w) => format!(
                    "\"t_walk\":{:e},\"walk_apps\":{},\"leaves\":{}",
                    secs(w.duration),
                    w.gate_apps,
                    w.leaves
                ),
                None => "\"t_walk\":null,\"walk_apps\":null,\"leaves\":null".to_string(),
            };
            writeln!(
                out,
                "{{\"kernel\":{},\"gates\":{},\"len\":{},\"method\":\"{}\",\"nnz_in\":{},\"nnz_out\":{},\"gate_apps\":{},\"t_total\":{:e},\"t_expected_scan\":{:e},\"t_alloc\":{:e},\"t_gather\":{:e},\"t_walk_and_insert\":{:e},\"t_resize\":{:e},\"num_resizes\":{},\"t_count\":{:e},\"output_capacity\":{},{}}}",
                kernel_idx,
                json_usizes(&these_gate_indices),
                num_gates_visited_here,
                method_name,
                num_nonzeros,
                new_num_nonzeros,
                num_gate_apps_here,
                secs(duration),
                phases.expected_scan,
                phases.alloc,
                phases.gather,
                phases.walk_and_insert,
                phases.resize,
                phases.num_resizes,
                phases.count,
                phases.output_capacity,
                walk_fields
            )
            .expect("cannot write --kernel-stats file");
        }
        kernel_idx += 1;

        let density = {
            let max_num_states: u64 = 1 << num_qubits;
            num_nonzeros as Real / max_num_states as Real
        };

        let throughput = (num_gate_apps_here as Real / 1e6) / duration.as_secs_f32();

        println!(
            "gate: {:<3} density: {:.8} nonzero: {:>10} hop: {:<2} {} time: {:.4}s throughput: {:.2}M gates/s",
            num_gates_visited,
            density,
            num_nonzeros,
            num_gates_visited_here,
            method,
            duration.as_secs_f32(),
            throughput
        );

        num_gates_visited += num_gates_visited_here;
        num_gate_apps += num_gate_apps_here;
        prev_num_nonzeros = num_nonzeros;
        num_nonzeros = new_num_nonzeros;
        state = new_state;
    });

    let final_density = {
        let max_num_states: u64 = 1 << num_qubits;
        num_nonzeros as f64 / max_num_states as f64
    };

    println!(
        "gate: {:<2} density: {:.8} nonzero: {:>10}\ngate app count: {}, time: {}s",
        num_gates_visited,
        final_density,
        num_nonzeros,
        num_gate_apps,
        duration.as_secs_f32()
    );
    println!(
        "kernels: {}, expand time (excluding --kernel-stats instrumentation): {}s",
        kernel_idx, expand_time
    );
    if let Some(mut out) = stats_out {
        out.flush().expect("cannot write --kernel-stats file");
    }

    assert!(num_gates_visited >= num_gates);
    state
}

fn initial_state<B: BasisIdx, AB: AtomicBasisIdx<B>>(
    config: &Config,
    num_qubits: usize,
) -> State<B, AB> {
    let n = config.init_random_nonzeros;
    if n == 0 {
        return State::Sparse(SparseStateTable::singleton(
            num_qubits,
            B::zeros(),
            Complex::new(1.0, 0.0),
            config.maxload,
            1,
        ));
    }
    assert!(
        num_qubits >= 64 || (n as u128) * 2 <= (1u128 << num_qubits),
        "--init-random-nonzeros must be at most half of the 2^{} basis states",
        num_qubits
    );
    // splitmix64
    let mut seed = config.seed;
    let mut next = move || {
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let weight = Complex::new(1.0 / (n as Real).sqrt(), 0.0);
    let mut seen = HashSet::with_capacity(n);
    let mut entries = Vec::with_capacity(n);
    while entries.len() < n {
        let mut bidx = B::zeros();
        let mut bits = 0;
        for qi in 0..num_qubits {
            if qi % 64 == 0 {
                bits = next();
            }
            if (bits >> (qi % 64)) & 1 == 1 {
                bidx = bidx.set(qi);
            }
        }
        if seen.insert(bidx.clone()) {
            entries.push((bidx, weight));
        }
    }
    log::info!("initial state: {} random basis states", n);
    State::Sparse(SparseStateTable::from_unique_entries(
        num_qubits,
        config.maxload,
        entries,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;
    use crate::types::constants;
    use crate::types::BasisIdx64;
    use approx::abs_diff_eq;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn test_run() {
        let config = Config::default();
        // bv_n30.qasm
        let circuit = Circuit::<BasisIdx64>::new(
            parser::parse_program(
                r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q0[30];
creg c0[30];
h q0[0];
h q0[1];
h q0[2];
h q0[3];
h q0[4];
h q0[5];
h q0[6];
h q0[7];
h q0[8];
h q0[9];
h q0[10];
h q0[11];
h q0[12];
h q0[13];
h q0[14];
h q0[15];
h q0[16];
h q0[17];
h q0[18];
h q0[19];
h q0[20];
h q0[21];
h q0[22];
h q0[23];
h q0[24];
h q0[25];
h q0[26];
h q0[27];
h q0[28];
x q0[29];
h q0[29];
//barrier q0[0],q0[1],q0[2],q0[3],q0[4],q0[5],q0[6],q0[7],q0[8],q0[9],q0[10],q0[11],q0[12],q0[13],q0[14],q0[15],q0[16],q0[17],q0[18],q0[19],q0[20],q0[21],q0[22],q0[23],q0[24],q0[25],q0[26],q0[27],q0[28],q0[29];
cx q0[0],q0[29];
cx q0[4],q0[29];
cx q0[5],q0[29];
cx q0[7],q0[29];
cx q0[8],q0[29];
cx q0[10],q0[29];
cx q0[11],q0[29];
cx q0[13],q0[29];
cx q0[15],q0[29];
cx q0[17],q0[29];
cx q0[21],q0[29];
cx q0[22],q0[29];
cx q0[23],q0[29];
cx q0[24],q0[29];
cx q0[25],q0[29];
cx q0[26],q0[29];
cx q0[27],q0[29];
cx q0[28],q0[29];
//barrier q0[0],q0[1],q0[2],q0[3],q0[4],q0[5],q0[6],q0[7],q0[8],q0[9],q0[10],q0[11],q0[12],q0[13],q0[14],q0[15],q0[16],q0[17],q0[18],q0[19],q0[20],q0[21],q0[22],q0[23],q0[24],q0[25],q0[26],q0[27],q0[28],q0[29];
h q0[0];
h q0[1];
h q0[2];
h q0[3];
h q0[4];
h q0[5];
h q0[6];
h q0[7];
h q0[8];
h q0[9];
h q0[10];
h q0[11];
h q0[12];
h q0[13];
h q0[14];
h q0[15];
h q0[16];
h q0[17];
h q0[18];
h q0[19];
h q0[20];
h q0[21];
h q0[22];
h q0[23];
h q0[24];
h q0[25];
h q0[26];
h q0[27];
h q0[28];
            "#,
            )
            .unwrap(),
        )
        .unwrap();

        let state = run::<BasisIdx64, AtomicU64>(&config, circuit);

        //        println!("{:?}", state);

        //assert!(matches!(state, State::Sparse(SparseStateTable { .. })));

        let table = match state {
            //State::Sparse(table) => table,
            State::Sparse(table) => table,
            _ => panic!(),
        };

        assert_eq!(table.num_nonzeros(), 2);
        let nonzero_field_1 = table
            .get(&BasisIdx64::new("11111111000101010110110110001"))
            .unwrap();
        assert!(abs_diff_eq!(
            nonzero_field_1.re,
            constants::RECP_SQRT_2,
            epsilon = 0.0001
        ));

        let nonzero_field_2 = table
            .get(&BasisIdx64::new("111111111000101010110110110001"))
            .unwrap();
        assert!(abs_diff_eq!(
            nonzero_field_2.re,
            -constants::RECP_SQRT_2,
            epsilon = 0.0001
        ));
    }
}
