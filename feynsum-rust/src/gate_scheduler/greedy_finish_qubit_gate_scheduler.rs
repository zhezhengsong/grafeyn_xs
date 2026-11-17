use super::{utility, GateScheduler};
use crate::types::{GateIndex, QubitIndex};

pub struct GreedyFinishQubitGateScheduler<'a> {
    frontier: Vec<GateIndex>,
    num_gates: usize,
    num_qubits: usize,
    gate_touches: Vec<&'a [QubitIndex]>,
}

impl<'a> GateScheduler for GreedyFinishQubitGateScheduler<'a> {
    fn pick_next_gates(&mut self) -> Vec<GateIndex> {
        let unfinished_qubit = (0..self.num_qubits).find(|qi| self.frontier[*qi] < self.num_gates);

        match unfinished_qubit {
            Some(qi) => self.make_progress_on_qubit(qi),
            None => Vec::new(),
        }
    }
}

impl<'a> GreedyFinishQubitGateScheduler<'a> {
    pub fn new(num_gates: usize, num_qubits: usize, gate_touches: Vec<&'a [QubitIndex]>) -> Self {
        log::debug!(
            "initializing greedy finish qubit gate scheduler with {} gates and {} qubits",
            num_gates,
            num_qubits
        );

        GreedyFinishQubitGateScheduler {
            frontier: (0..num_qubits)
                .map(|qi| utility::next_touch(num_gates, &gate_touches, qi, 0))
                .collect(),
            num_gates,
            num_qubits,
            gate_touches,
        }
    }

    fn make_progress_on_qubit(&mut self, qi: QubitIndex) -> Vec<GateIndex> {
        let desired_gate = self.frontier[qi];

        if utility::okay_to_visit(
            self.num_gates,
            &self.gate_touches,
            &self.frontier,
            desired_gate,
        ) {
            utility::mark_as_visit(
                self.num_gates,
                &self.gate_touches,
                &mut self.frontier,
                desired_gate,
            );
            vec![desired_gate]
        } else {
            let dependency = self.gate_touches[desired_gate]
                .iter()
                .find(|qj| self.frontier[**qj] < desired_gate)
                .expect("since desired_gate is not okay to visit, there must be a dependency");

            self.make_progress_on_qubit(*dependency)
        }
    }
}

/// Port of `pickNext` in feynsum-sml's `GateSchedulerGreedyFinishQubit.sml`:
/// keep taking the next greedy-finish-qubit gate until the kernel contains
/// `max_branching_stride` branching gates (or the circuit is exhausted).
pub struct GreedyFinishQubitFusedGateScheduler<'a> {
    inner: GreedyFinishQubitGateScheduler<'a>,
    gate_is_branching: Vec<bool>,
    max_branching_stride: usize,
    disable_gate_fusion: bool,
}

impl<'a> GateScheduler for GreedyFinishQubitFusedGateScheduler<'a> {
    fn pick_next_gates(&mut self) -> Vec<GateIndex> {
        if self.disable_gate_fusion {
            return self.inner.pick_next_gates();
        }
        let mut next_gates = Vec::new();
        let mut num_branching_so_far = 0;
        while num_branching_so_far < self.max_branching_stride {
            // the unfused scheduler returns at most one (already visited) gate
            match self.inner.pick_next_gates().pop() {
                None => break,
                Some(gi) => {
                    if self.gate_is_branching[gi] {
                        num_branching_so_far += 1;
                    }
                    next_gates.push(gi);
                }
            }
        }
        next_gates
    }
}

impl<'a> GreedyFinishQubitFusedGateScheduler<'a> {
    pub fn new(
        num_gates: usize,
        num_qubits: usize,
        gate_touches: Vec<&'a [QubitIndex]>,
        gate_is_branching: Vec<bool>,
        max_branching_stride: usize,
        disable_gate_fusion: bool,
    ) -> Self {
        assert_eq!(gate_is_branching.len(), num_gates);
        Self {
            inner: GreedyFinishQubitGateScheduler::new(num_gates, num_qubits, gate_touches),
            gate_is_branching,
            max_branching_stride,
            disable_gate_fusion,
        }
    }
}
