use std::fmt::Display;
use std::str::FromStr;

use crate::circuit::Circuit;
use crate::config::Config;
use crate::types::BasisIdx;

mod greedy_finish_qubit_gate_scheduler;
mod greedy_nonbranching_gate_scheduler;
mod naive_gate_scheduler;
mod utility;

pub use greedy_finish_qubit_gate_scheduler::{
    GreedyFinishQubitFusedGateScheduler, GreedyFinishQubitGateScheduler,
};
pub use greedy_nonbranching_gate_scheduler::GreedyNonbranchingGateScheduler;
pub use naive_gate_scheduler::NaiveGateScheduler;

#[derive(Debug, Copy, Clone)]
pub enum GateSchedulingPolicy {
    Naive,
    GreedyNonbranching,
    GreedyFinishQubit,
    /// greedy-finish-qubit with kernel fusion, ported from feynsum-sml's
    /// `GateSchedulerGreedyFinishQubit.pickNext` (the scheduler used in the paper)
    GreedyFinishQubitFused,
}

impl FromStr for GateSchedulingPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "naive" => Ok(GateSchedulingPolicy::Naive),
            "greedy-nonbranching" | "gnb" => Ok(GateSchedulingPolicy::GreedyNonbranching),
            "greedy-finish-qubit" | "gfq" => Ok(GateSchedulingPolicy::GreedyFinishQubit),
            "greedy-finish-qubit-fused" | "gfq-fused" => {
                Ok(GateSchedulingPolicy::GreedyFinishQubitFused)
            }
            _ => Err(format!(
                "unknown gate scheduling policy: {}; valid values are: naive, gnb, gfq, gfq-fused",
                s
            )),
        }
    }
}

impl Display for GateSchedulingPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GateSchedulingPolicy::Naive => write!(f, "naive"),
            GateSchedulingPolicy::GreedyNonbranching => write!(f, "greedy-nonbranching"),
            GateSchedulingPolicy::GreedyFinishQubit => write!(f, "greedy-finish-qubit"),
            GateSchedulingPolicy::GreedyFinishQubitFused => {
                write!(f, "greedy-finish-qubit-fused")
            }
        }
    }
}

pub trait GateScheduler {
    fn pick_next_gates(&mut self) -> Vec<usize>;
}

pub fn create_gate_scheduler<'a, B: BasisIdx>(
    config: &Config,
    circuit: &'a Circuit<B>,
) -> Box<dyn GateScheduler + 'a> {
    let gate_scheduling_policy = config.gate_scheduling_policy;
    let disable_gate_fusion = config.disable_gate_fusion;

    let num_gates = circuit.num_gates();
    let num_qubits = circuit.num_qubits;
    let gate_touches = circuit
        .gates
        .iter()
        .map(|gate| gate.touches.as_slice())
        .collect();
    let gate_is_branching = circuit
        .gates
        .iter()
        .map(|gate| gate.is_branching())
        .collect();

    match gate_scheduling_policy {
        GateSchedulingPolicy::Naive => {
            log::info!("using naive gate scheduler");
            Box::new(NaiveGateScheduler::new(num_gates))
        }
        GateSchedulingPolicy::GreedyNonbranching => {
            log::info!("using greedy nonbranching gate scheduler");
            Box::new(GreedyNonbranchingGateScheduler::new(
                num_gates,
                num_qubits,
                gate_touches,
                gate_is_branching,
                disable_gate_fusion,
            ))
        }
        GateSchedulingPolicy::GreedyFinishQubit => {
            log::info!("using greedy nonbranching gate scheduler");
            Box::new(GreedyFinishQubitGateScheduler::new(
                num_gates,
                num_qubits,
                gate_touches,
            ))
        }
        GateSchedulingPolicy::GreedyFinishQubitFused => {
            log::info!(
                "using greedy finish qubit gate scheduler with fusion (max branching stride {})",
                config.max_branching_stride
            );
            // feynsum-sml decides single-qubit unitaries (u, u1, u2, u3, rx) by
            // their matrix zero pattern; mirror that so kernels match the SML ones.
            let gate_is_branching_sml = circuit
                .gates
                .iter()
                .map(|gate| gate.is_branching_sml())
                .collect();
            Box::new(GreedyFinishQubitFusedGateScheduler::new(
                num_gates,
                num_qubits,
                gate_touches,
                gate_is_branching_sml,
                config.max_branching_stride,
                disable_gate_fusion,
            ))
        }
    }
}
