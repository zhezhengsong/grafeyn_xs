//! Kernel IR.
//!
//! A kernel is a straight-line list of gates that every Feynman path walks
//! before the paths are summed into the next state. The interpreter
//! (`parallel_simulator::state_expander::apply_gates1`) dispatches on ~20 gate
//! kinds, one closure call per gate per path. Here every supported gate is
//! lowered to a handful of primitive operations on one path, i.e. a basis
//! state `b` (one bit per qubit) and a complex weight `w`:
//!
//! | op        | effect on one path                                             | gates            |
//! |-----------|----------------------------------------------------------------|------------------|
//! | `Flip`    | if all `controls` are 1: flip every bit in `targets`           | x, cx, ccx (+ y, anti-diagonal u) |
//! | `Swap`    | if all `controls` are 1: exchange bits `a` and `b`             | swap, cswap      |
//! | `Phase`   | if all `controls` are 1: `w *= if b[qubit] { c1 } else { c0 }` | z, s, t, rz, phase, u1, cz, cphase (+ y) |
//! | `Branch`  | one path becomes two: bit `j = b[qubit]` continues as 0 with `w*m[0][j]` and as 1 with `w*m[1][j]` | h, rx, ry, sx, sxdg, u |
//! | `Prune`   | drop paths whose weight is zero                                 | after each branch |
//!
//! Semantics are defined by [`eval`]: one input path yields at most
//! `2^(number of Branch ops)` output paths ("leaves"); the simulator adds the
//! leaves into the next state. Two kernels are equivalent when, for every input
//! basis state, the sums of their leaves agree (see `check.rs`).
//!
//! Design choices:
//! - Ops name qubits by index, so the IR does not depend on the basis-index
//!   width (`BasisIdx64` or `BasisIdxUnlimited`); backends turn index lists
//!   into machine-word masks.
//! - Constants are `f64`. Gate parameters come in as `f32` (the host precision);
//!   folding phases at compile time in `f64` and rounding once at code
//!   generation loses less than the interpreter does.
//! - Whether a single-qubit unitary branches is decided exactly as the
//!   interpreter and `gfq-fused` decide it (f32 matrix, zero threshold 1e-8),
//!   so a lowered kernel has the same number of leaves the scheduler assumed.

use std::fmt;

use num::complex::Complex;

use crate::types::{constants, BasisIdx, QubitIndex};

pub type C64 = Complex<f64>;

/// Same threshold as the interpreter (`utility::is_zero`).
pub const ZERO_THRESHOLD: f64 = constants::ZERO_THRESHOLD as f64;

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// If every control bit is 1, flip every target bit. `controls` may be empty.
    Flip {
        controls: Vec<QubitIndex>,
        targets: Vec<QubitIndex>,
    },
    /// If every control bit is 1, exchange bits `a` and `b`.
    Swap {
        controls: Vec<QubitIndex>,
        a: QubitIndex,
        b: QubitIndex,
    },
    /// If every control bit is 1, multiply the weight by `c1` when bit `qubit`
    /// is 1 and by `c0` when it is 0; otherwise leave the weight unchanged.
    Phase {
        controls: Vec<QubitIndex>,
        qubit: QubitIndex,
        c0: C64,
        c1: C64,
    },
    /// Split the path on `qubit`. With `j = b[qubit]`, continue as
    /// `(b[qubit := 0], w * m[0][j])` and `(b[qubit := 1], w * m[1][j])`,
    /// i.e. `m` is the gate's 2x2 matrix with the input bit selecting a column.
    Branch { qubit: QubitIndex, m: [[C64; 2]; 2] },
    /// Drop paths whose weight is zero (both parts below `ZERO_THRESHOLD`).
    Prune,
}

impl Op {
    /// Every qubit the op reads or writes.
    pub fn qubits(&self) -> Vec<QubitIndex> {
        match self {
            Op::Flip { controls, targets } => controls.iter().chain(targets).copied().collect(),
            Op::Swap { controls, a, b } => controls.iter().copied().chain([*a, *b]).collect(),
            Op::Phase {
                controls, qubit, ..
            } => controls.iter().copied().chain([*qubit]).collect(),
            Op::Branch { qubit, .. } => vec![*qubit],
            Op::Prune => vec![],
        }
    }
}

/// A lowered kernel.
#[derive(Debug, Clone)]
pub struct Kernel {
    /// Indices of the source gates in the circuit, in kernel order.
    pub gates: Vec<usize>,
    pub ops: Vec<Op>,
}

impl Kernel {
    /// Sorted distinct qubits touched by the kernel (its `k` is the length).
    pub fn touched(&self) -> Vec<QubitIndex> {
        let mut qs: Vec<QubitIndex> = self.ops.iter().flat_map(|op| op.qubits()).collect();
        qs.sort_unstable();
        qs.dedup();
        qs
    }

    pub fn num_branches(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, Op::Branch { .. }))
            .count()
    }

    /// Upper bound on the number of leaves per input path.
    pub fn max_leaves(&self) -> usize {
        1 << self.num_branches()
    }
}

pub fn is_zero(c: C64) -> bool {
    c.re.abs() < ZERO_THRESHOLD && c.im.abs() < ZERO_THRESHOLD
}

fn controls_hold<B: BasisIdx>(b: &B, controls: &[QubitIndex]) -> bool {
    controls.iter().all(|&q| b.get(q))
}

/// Reference semantics: run one input path through `ops` and return the
/// leaves (zero-weight leaves are dropped, as the interpreter does before
/// inserting into the table).
pub fn eval<B: BasisIdx>(ops: &[Op], bidx: B, weight: C64) -> Vec<(B, C64)> {
    let mut paths = if is_zero(weight) {
        vec![]
    } else {
        vec![(bidx, weight)]
    };
    for op in ops {
        match op {
            Op::Flip { controls, targets } => {
                for (b, _) in paths.iter_mut() {
                    if controls_hold(b, controls) {
                        for &t in targets {
                            *b = b.flip(t);
                        }
                    }
                }
            }
            Op::Swap { controls, a, b: c } => {
                for (b, _) in paths.iter_mut() {
                    if controls_hold(b, controls) {
                        *b = b.swap(*a, *c);
                    }
                }
            }
            Op::Phase {
                controls,
                qubit,
                c0,
                c1,
            } => {
                for (b, w) in paths.iter_mut() {
                    if controls_hold(b, controls) {
                        *w *= if b.get(*qubit) { *c1 } else { *c0 };
                    }
                }
            }
            Op::Branch { qubit, m } => {
                let mut next = Vec::with_capacity(2 * paths.len());
                for (b, w) in paths {
                    let j = b.get(*qubit) as usize;
                    next.push((b.unset(*qubit), w * m[0][j]));
                    next.push((b.set(*qubit), w * m[1][j]));
                }
                paths = next;
            }
            Op::Prune => paths.retain(|(_, w)| !is_zero(*w)),
        }
    }
    paths.retain(|(_, w)| !is_zero(*w));
    paths
}

// ---------------------------------------------------------------------------
// printing

struct C(C64);

impl fmt::Display for C {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (re, im) = (self.0.re, self.0.im);
        let tiny = |x: f64| x.abs() < 5e-5;
        match (tiny(re), tiny(im)) {
            (_, true) => write!(f, "{:.4}", re),
            (true, false) => write!(f, "{:.4}i", im),
            (false, false) => write!(f, "{:.4}{}{:.4}i", re, if im < 0.0 { "-" } else { "+" }, im.abs()),
        }
    }
}

fn qs(qubits: &[QubitIndex]) -> String {
    qubits.iter().map(|q| format!("q{}", q)).collect::<Vec<_>>().join(",")
}

fn cond(controls: &[QubitIndex]) -> String {
    if controls.is_empty() {
        String::new()
    } else {
        format!(
            " if {}",
            controls.iter().map(|q| format!("q{}", q)).collect::<Vec<_>>().join("&")
        )
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Flip { controls, targets } => write!(f, "flip {}{}", qs(targets), cond(controls)),
            Op::Swap { controls, a, b } => write!(f, "swap q{},q{}{}", a, b, cond(controls)),
            Op::Phase {
                controls,
                qubit,
                c0,
                c1,
            } => write!(
                f,
                "phase q{} ? {} : {}{}",
                qubit,
                C(*c1),
                C(*c0),
                cond(controls)
            ),
            Op::Branch { qubit, m } => write!(
                f,
                "branch q{} [[{}, {}], [{}, {}]]",
                qubit,
                C(m[0][0]),
                C(m[0][1]),
                C(m[1][0]),
                C(m[1][1])
            ),
            Op::Prune => write!(f, "prune"),
        }
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "kernel gates={:?} k={} max_leaves={}",
            self.gates,
            self.touched().len(),
            self.max_leaves()
        )?;
        for op in &self.ops {
            writeln!(f, "  {}", op)?;
        }
        Ok(())
    }
}
