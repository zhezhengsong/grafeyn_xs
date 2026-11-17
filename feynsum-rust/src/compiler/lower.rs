//! Lowering: circuit gates -> IR ops (see `ir.rs` for the op set).
//!
//! Each case mirrors `GateDefn::push_apply` in `circuit/gate.rs`; the
//! per-kernel checker (`check.rs`) compares the two on every input.

use std::f64::consts::FRAC_1_SQRT_2;
use std::fmt;

use super::ir::{Kernel, Op, C64};
use crate::circuit::{Circuit, Gate, GateDefn};
use crate::types::{BasisIdx, QubitIndex};
use crate::utility;

#[derive(Debug, Clone, PartialEq)]
pub enum LowerError {
    /// The gate has no lowering yet (fsim, unknown gates); the kernel has to
    /// stay on the interpreter.
    Unsupported { gate_index: usize, name: String },
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LowerError::Unsupported { gate_index, name } => {
                write!(f, "gate {} ({}) has no IR lowering", gate_index, name)
            }
        }
    }
}

pub fn lower_kernel<B: BasisIdx>(
    circuit: &Circuit<B>,
    gate_indices: &[usize],
) -> Result<Kernel, LowerError> {
    let mut ops = Vec::new();
    for &gi in gate_indices {
        let gate = &circuit.gates[gi];
        match lower_gate(gate) {
            Some(gate_ops) => ops.extend(gate_ops),
            None => {
                return Err(LowerError::Unsupported {
                    gate_index: gi,
                    name: gate.name().to_string(),
                })
            }
        }
    }
    Ok(Kernel {
        gates: gate_indices.to_vec(),
        ops,
    })
}

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// e^{i theta}
fn cis(theta: f64) -> C64 {
    C64::new(theta.cos(), theta.sin())
}

fn phase(qubit: QubitIndex, c0: C64, c1: C64) -> Op {
    Op::Phase {
        controls: vec![],
        qubit,
        c0,
        c1,
    }
}

fn flip(controls: Vec<QubitIndex>, target: QubitIndex) -> Op {
    Op::Flip {
        controls,
        targets: vec![target],
    }
}

fn branch(qubit: QubitIndex, m: [[C64; 2]; 2]) -> Vec<Op> {
    vec![Op::Branch { qubit, m }, Op::Prune]
}

/// Lower a general single-qubit unitary [[a, b], [c, d]] the way
/// `single_qubit_unitary_push` applies it: anti-diagonal -> phase then flip,
/// diagonal -> phase, otherwise branch. The zero pattern is read from the
/// interpreter's own f32 matrix so the decision matches it (and gfq-fused).
fn single_qubit<B: BasisIdx>(gate: &Gate<B>, target: QubitIndex, m: [[C64; 2]; 2]) -> Vec<Op> {
    let (a32, b32, c32, d32) = gate
        .defn
        .single_qubit_matrix()
        .expect("single-qubit unitary");
    let z = utility::is_zero;
    if z(a32) && z(d32) {
        // |0> -> c|1>, |1> -> b|0>: weight factor depends on the input bit, then flip
        vec![phase(target, m[1][0], m[0][1]), flip(vec![], target)]
    } else if z(b32) && z(c32) {
        vec![phase(target, m[0][0], m[1][1])]
    } else {
        branch(target, m)
    }
}

fn u_matrix(theta: f64, phi: f64, lambda: f64) -> [[C64; 2]; 2] {
    let (cos, sin) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    [
        [c(cos, 0.0), -cis(lambda) * sin],
        [cis(phi) * sin, cis(phi + lambda) * cos],
    ]
}

/// Ops for one gate, or `None` when the gate is not supported.
pub fn lower_gate<B: BasisIdx>(gate: &Gate<B>) -> Option<Vec<Op>> {
    let one = c(1.0, 0.0);
    let r = FRAC_1_SQRT_2;
    let ops = match gate.defn {
        GateDefn::X(q) => vec![flip(vec![], q)],
        GateDefn::CX { control, target } => vec![flip(vec![control], target)],
        GateDefn::CCX {
            control1,
            control2,
            target,
        } => vec![flip(vec![control1, control2], target)],
        GateDefn::Swap { target1, target2 } => vec![Op::Swap {
            controls: vec![],
            a: target1,
            b: target2,
        }],
        GateDefn::CSwap {
            control,
            target1,
            target2,
        } => vec![Op::Swap {
            controls: vec![control],
            a: target1,
            b: target2,
        }],
        // Y|0> = i|1>, Y|1> = -i|0>: the factor depends on the bit before the flip
        GateDefn::PauliY(q) => vec![phase(q, c(0.0, 1.0), c(0.0, -1.0)), flip(vec![], q)],
        GateDefn::PauliZ(q) => vec![phase(q, one, c(-1.0, 0.0))],
        GateDefn::S(q) => vec![phase(q, one, c(0.0, 1.0))],
        GateDefn::Sdg(q) => vec![phase(q, one, c(0.0, -1.0))],
        GateDefn::T(q) => vec![phase(q, one, c(r, r))],
        GateDefn::Tdg(q) => vec![phase(q, one, c(r, -r))],
        GateDefn::Phase { rot, target } => vec![phase(target, one, cis(rot as f64))],
        GateDefn::RZ { rot, target } => {
            let half = rot as f64 / 2.0;
            vec![phase(target, cis(-half), cis(half))]
        }
        GateDefn::CZ { control, target } => vec![Op::Phase {
            controls: vec![control],
            qubit: target,
            c0: one,
            c1: c(-1.0, 0.0),
        }],
        GateDefn::CPhase {
            control,
            target,
            rot,
        } => vec![Op::Phase {
            controls: vec![control],
            qubit: target,
            c0: one,
            c1: cis(rot as f64),
        }],
        GateDefn::Hadamard(q) => branch(q, [[c(r, 0.0), c(r, 0.0)], [c(r, 0.0), c(-r, 0.0)]]),
        GateDefn::RY { rot, target } => {
            let (cos, sin) = ((rot as f64 / 2.0).cos(), (rot as f64 / 2.0).sin());
            branch(target, [[c(cos, 0.0), c(-sin, 0.0)], [c(sin, 0.0), c(cos, 0.0)]])
        }
        GateDefn::SqrtX(q) => {
            let (p, m) = (c(0.5, 0.5), c(0.5, -0.5));
            branch(q, [[p, m], [m, p]])
        }
        GateDefn::SqrtXdg(q) => {
            let (p, m) = (c(0.5, 0.5), c(0.5, -0.5));
            branch(q, [[m, p], [p, m]])
        }
        GateDefn::RX { rot, target } => {
            let (cos, sin) = ((rot as f64 / 2.0).cos(), (rot as f64 / 2.0).sin());
            let m = [[c(cos, 0.0), c(0.0, -sin)], [c(0.0, -sin), c(cos, 0.0)]];
            single_qubit(gate, target, m)
        }
        GateDefn::U {
            target,
            theta,
            phi,
            lambda,
        } => single_qubit(
            gate,
            target,
            u_matrix(theta as f64, phi as f64, lambda as f64),
        ),
        GateDefn::FSim { .. } | GateDefn::Other { .. } => return None,
    };
    Some(ops)
}
