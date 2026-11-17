//! Per-kernel differential check: lowered IR vs. the interpreter's
//! `push_apply`, on the bits the kernel touches.
//!
//! For a kernel touching `k` qubits, every one of the `2^k` assignments of
//! those bits is tried when `k <= EXHAUSTIVE_MAX_K`, otherwise `SAMPLES`
//! random ones. Each assignment is embedded in two backgrounds for the other
//! qubits (all zeros, and a random pattern), which catches lowerings that
//! touch the wrong qubit. Both sides' leaves are summed per basis state and
//! compared; basis states are compared by their bits, not by the `BasisIdx`
//! value, because `BasisIdxUnlimited` equality also compares vector lengths.

use std::collections::HashMap;
use std::fmt;

use super::ir::{self, C64};
use super::lower::{lower_kernel, LowerError};
use crate::circuit::{Circuit, Gate, PushApplicable, PushApplyOutput};
use crate::config::Config;
use crate::gate_scheduler;
use crate::types::{BasisIdx, Complex, QubitIndex};
use crate::utility;

pub const EXHAUSTIVE_MAX_K: usize = 12;
pub const SAMPLES: usize = 512;
/// samples for very long kernels (each sample walks every gate twice)
pub const SAMPLES_LONG: usize = 64;
pub const LONG_KERNEL: usize = 10_000;
/// absolute tolerance on the summed output weights (input weight has norm 1)
pub const TOLERANCE: f64 = 1e-5;

#[derive(Debug, Default)]
pub struct CheckReport {
    pub kernels: usize,
    pub unsupported: Vec<LowerError>,
    pub exhaustive: usize,
    pub sampled: usize,
    pub inputs: usize,
    pub max_err: f64,
    pub max_leaves_seen: usize,
    pub leaf_bound_violations: usize,
    pub failures: Vec<String>,
}

impl CheckReport {
    pub fn ok(&self) -> bool {
        self.failures.is_empty() && self.leaf_bound_violations == 0
    }
}

impl fmt::Display for CheckReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "kernels {} | lowered {} | unsupported {} | exhaustive {} | sampled {} | inputs {}",
            self.kernels,
            self.kernels - self.unsupported.len(),
            self.unsupported.len(),
            self.exhaustive,
            self.sampled,
            self.inputs
        )?;
        writeln!(
            f,
            "max |ir - interpreter| = {:.3e} (tolerance {:.0e}) | max leaves per input {} | leaf-bound violations {}",
            self.max_err, TOLERANCE, self.max_leaves_seen, self.leaf_bound_violations
        )?;
        for e in self.unsupported.iter().take(5) {
            writeln!(f, "unsupported: {}", e)?;
        }
        for msg in self.failures.iter().take(10) {
            writeln!(f, "FAIL {}", msg)?;
        }
        write!(f, "{}", if self.ok() { "OK" } else { "FAILED" })
    }
}

pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The interpreter's semantics for one input path, as a loop over gates
/// (same push_apply calls and zero pruning as `apply_gates1`, no recursion).
pub fn interpret<B: BasisIdx>(gates: &[&Gate<B>], bidx: B, weight: Complex) -> Vec<(B, Complex)> {
    let mut paths = vec![(bidx, weight)];
    for gate in gates {
        let mut next = Vec::with_capacity(paths.len() * 2);
        for (b, w) in paths {
            if utility::is_zero(w) {
                continue;
            }
            match gate.push_apply(b, w) {
                PushApplyOutput::Nonbranching(b1, w1) => next.push((b1, w1)),
                PushApplyOutput::Branching((b1, w1), (b2, w2)) => {
                    next.push((b1, w1));
                    next.push((b2, w2));
                }
            }
        }
        paths = next;
    }
    paths.retain(|(_, w)| utility::is_nonzero(*w));
    paths
}

fn bits_key<B: BasisIdx>(b: &B, num_qubits: usize) -> Vec<u64> {
    let mut words = vec![0u64; (num_qubits + 63) / 64];
    for q in 0..num_qubits {
        if b.get(q) {
            words[q / 64] |= 1 << (q % 64);
        }
    }
    words
}

fn with_bits<B: BasisIdx>(mut b: B, qubits: &[QubitIndex], assignment: u64) -> B {
    for (i, &q) in qubits.iter().enumerate() {
        b = if (assignment >> i) & 1 == 1 { b.set(q) } else { b.unset(q) };
    }
    b
}

fn random_background<B: BasisIdx>(num_qubits: usize, rng: &mut SplitMix64) -> B {
    let mut b = B::zeros();
    let mut bits = 0;
    for q in 0..num_qubits {
        if q % 64 == 0 {
            bits = rng.next();
        }
        if (bits >> (q % 64)) & 1 == 1 {
            b = b.set(q);
        }
    }
    b
}

/// Check one lowered kernel against the interpreter, accumulating into `report`.
fn check_kernel<B: BasisIdx>(
    circuit: &Circuit<B>,
    kernel_idx: usize,
    kernel: &ir::Kernel,
    rng: &mut SplitMix64,
    report: &mut CheckReport,
) {
    let gates: Vec<&Gate<B>> = kernel.gates.iter().map(|&gi| &circuit.gates[gi]).collect();
    let touched = kernel.touched();
    let k = touched.len();
    let assignments: Vec<u64> = if k <= EXHAUSTIVE_MAX_K {
        report.exhaustive += 1;
        (0..(1u64 << k)).collect()
    } else {
        report.sampled += 1;
        let n = if gates.len() > LONG_KERNEL { SAMPLES_LONG } else { SAMPLES };
        (0..n)
            .map(|_| {
                // k may exceed 64: only the low 64 touched qubits get random bits from
                // `assignment`; the background below randomizes the rest
                rng.next()
            })
            .collect()
    };
    let backgrounds = [B::zeros(), random_background::<B>(circuit.num_qubits, rng)];
    let w_in = Complex::new(0.6, 0.8);
    let mut failures = 0;

    for background in backgrounds.iter() {
        for &a in &assignments {
            let mut input = with_bits(background.clone(), &touched[..k.min(64)], a);
            if k > 64 {
                // randomize touched qubits beyond the first 64 as well
                let extra = random_background::<B>(circuit.num_qubits, rng);
                for &q in &touched[64..] {
                    input = if extra.get(q) { input.set(q) } else { input.unset(q) };
                }
            }
            report.inputs += 1;

            let mut expected: HashMap<Vec<u64>, C64> = HashMap::new();
            for (b, w) in interpret(&gates, input.clone(), w_in) {
                *expected.entry(bits_key(&b, circuit.num_qubits)).or_default() +=
                    C64::new(w.re as f64, w.im as f64);
            }
            let leaves = ir::eval(&kernel.ops, input.clone(), C64::new(0.6, 0.8));
            report.max_leaves_seen = report.max_leaves_seen.max(leaves.len());
            if leaves.len() > kernel.max_leaves() {
                report.leaf_bound_violations += 1;
            }
            let mut actual: HashMap<Vec<u64>, C64> = HashMap::new();
            for (b, w) in leaves {
                *actual.entry(bits_key(&b, circuit.num_qubits)).or_default() += w;
            }

            let mut err: f64 = 0.0;
            for (key, w) in &expected {
                err = err.max((w - actual.get(key).copied().unwrap_or_default()).norm());
            }
            for (key, w) in &actual {
                if !expected.contains_key(key) {
                    err = err.max(w.norm());
                }
            }
            report.max_err = report.max_err.max(err);
            if err > TOLERANCE {
                failures += 1;
                if failures <= 2 {
                    let names: Vec<&str> = gates.iter().map(|g| g.name()).collect();
                    report.failures.push(format!(
                        "kernel {} gates {:?} {:?}: input bits {:?} -> error {:.3e}",
                        kernel_idx,
                        kernel.gates,
                        names,
                        touched.iter().filter(|&&q| input.get(q)).collect::<Vec<_>>(),
                        err
                    ));
                }
            }
        }
    }
}

/// Lower and check every kernel the configured scheduler produces.
pub fn check_circuit<B: BasisIdx>(config: &Config, circuit: &Circuit<B>, seed: u64) -> CheckReport {
    let mut report = CheckReport::default();
    let mut rng = SplitMix64(seed);
    let mut scheduler = gate_scheduler::create_gate_scheduler(config, circuit);
    loop {
        let gate_indices = scheduler.pick_next_gates();
        if gate_indices.is_empty() {
            break;
        }
        let kernel_idx = report.kernels;
        report.kernels += 1;
        match lower_kernel(circuit, &gate_indices) {
            Err(e) => report.unsupported.push(e),
            Ok(kernel) => check_kernel(circuit, kernel_idx, &kernel, &mut rng, &mut report),
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::GateDefn;
    use crate::gate_scheduler::GateSchedulingPolicy;
    use crate::types::{BasisIdx64, BasisIdxUnlimited, Real};
    use std::f32::consts::PI;

    /// A random circuit over every gate kind that has a lowering, with angles
    /// drawn from {0, pi/2, pi, random} so diagonal / anti-diagonal / degenerate
    /// matrices (u1, u3(pi,..), rx(0), ry(0)) all show up.
    fn random_circuit<B: BasisIdx>(num_qubits: usize, num_gates: usize, seed: u64) -> Circuit<B> {
        let mut rng = SplitMix64(seed);
        let angle = |rng: &mut SplitMix64| -> Real {
            match rng.below(4) {
                0 => 0.0,
                1 => PI / 2.0,
                2 => PI,
                _ => (rng.next() % 10_000) as Real / 10_000.0 * 2.0 * PI - PI,
            }
        };
        let distinct = |rng: &mut SplitMix64, n: usize| -> Vec<usize> {
            let mut qs = Vec::new();
            while qs.len() < n {
                let q = rng.below(num_qubits);
                if !qs.contains(&q) {
                    qs.push(q);
                }
            }
            qs
        };
        let mut gates = Vec::new();
        for _ in 0..num_gates {
            let kind = rng.below(23);
            let q = distinct(&mut rng, 3);
            let defn = match kind {
                0 => GateDefn::X(q[0]),
                1 => GateDefn::PauliY(q[0]),
                2 => GateDefn::PauliZ(q[0]),
                3 => GateDefn::Hadamard(q[0]),
                4 => GateDefn::S(q[0]),
                5 => GateDefn::Sdg(q[0]),
                6 => GateDefn::T(q[0]),
                7 => GateDefn::Tdg(q[0]),
                8 => GateDefn::SqrtX(q[0]),
                9 => GateDefn::SqrtXdg(q[0]),
                10 => GateDefn::RX { rot: angle(&mut rng), target: q[0] },
                11 => GateDefn::RY { rot: angle(&mut rng), target: q[0] },
                12 => GateDefn::RZ { rot: angle(&mut rng), target: q[0] },
                13 => GateDefn::Phase { rot: angle(&mut rng), target: q[0] },
                14 | 15 => GateDefn::U {
                    target: q[0],
                    theta: angle(&mut rng),
                    phi: angle(&mut rng),
                    lambda: angle(&mut rng),
                },
                16 => GateDefn::CX { control: q[0], target: q[1] },
                17 => GateDefn::CZ { control: q[0], target: q[1] },
                18 => GateDefn::CPhase { control: q[0], target: q[1], rot: angle(&mut rng) },
                19 => GateDefn::Swap { target1: q[0], target2: q[1] },
                20 => GateDefn::CCX { control1: q[0], control2: q[1], target: q[2] },
                21 => GateDefn::CSwap { control: q[0], target1: q[1], target2: q[2] },
                _ => GateDefn::U { target: q[0], theta: 0.0, phi: 0.0, lambda: angle(&mut rng) }, // u1
            };
            gates.push(Gate::new(defn));
        }
        Circuit { num_qubits, gates }
    }

    fn config(policy: GateSchedulingPolicy, stride: usize) -> Config {
        Config {
            gate_scheduling_policy: policy,
            max_branching_stride: stride,
            ..Config::default()
        }
    }

    #[test]
    fn eval_hadamard_then_cx() {
        let circuit = Circuit::<BasisIdx64> {
            num_qubits: 2,
            gates: vec![
                Gate::new(GateDefn::Hadamard(0)),
                Gate::new(GateDefn::CX { control: 0, target: 1 }),
            ],
        };
        let kernel = lower_kernel(&circuit, &[0, 1]).unwrap();
        let mut leaves = ir::eval(&kernel.ops, BasisIdx64::zeros(), C64::new(1.0, 0.0));
        leaves.sort_by_key(|(b, _)| b.as_idx());
        assert_eq!(leaves.len(), 2);
        assert_eq!(leaves[0].0.as_idx(), 0b00);
        assert_eq!(leaves[1].0.as_idx(), 0b11);
        for (_, w) in leaves {
            assert!((w - C64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0)).norm() < 1e-12);
        }
    }

    #[test]
    fn random_circuits_match_interpreter() {
        let policies = [
            (GateSchedulingPolicy::GreedyFinishQubitFused, 2),
            (GateSchedulingPolicy::GreedyFinishQubitFused, 4),
            (GateSchedulingPolicy::GreedyNonbranching, 2),
            (GateSchedulingPolicy::Naive, 2),
        ];
        for seed in 0..40 {
            let circuit = random_circuit::<BasisIdx64>(6, 80, seed);
            for (policy, stride) in policies {
                let report = check_circuit(&config(policy, stride), &circuit, seed);
                assert!(report.unsupported.is_empty());
                assert!(report.ok(), "seed {} {:?}/{}:\n{}", seed, policy, stride, report);
                assert_eq!(report.sampled, 0, "6 qubits must be checked exhaustively");
            }
        }
    }

    #[test]
    fn unlimited_basis_idx_matches_interpreter() {
        for seed in 0..5 {
            let circuit = random_circuit::<BasisIdxUnlimited>(70, 120, 100 + seed);
            let report = check_circuit(
                &config(GateSchedulingPolicy::GreedyFinishQubitFused, 2),
                &circuit,
                seed,
            );
            assert!(report.ok(), "seed {}:\n{}", seed, report);
        }
    }

    #[test]
    fn wrong_lowering_is_caught() {
        // swap the two columns of H: a deliberately wrong kernel must fail
        let circuit = Circuit::<BasisIdx64> {
            num_qubits: 1,
            gates: vec![Gate::new(GateDefn::Hadamard(0))],
        };
        let mut kernel = lower_kernel(&circuit, &[0]).unwrap();
        if let ir::Op::Branch { m, .. } = &mut kernel.ops[0] {
            m[1].swap(0, 1);
        }
        let mut report = CheckReport::default();
        check_kernel(&circuit, 0, &kernel, &mut SplitMix64(1), &mut report);
        assert!(!report.failures.is_empty());
    }
}
