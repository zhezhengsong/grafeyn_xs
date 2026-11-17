//! Instrumentation for the kernel-compilation study:
//! - `dump_static_kernels`: run the gate scheduler to exhaustion (it never
//!   looks at the state) and write the kernel sequence as JSON lines.
//! - `walk_only`: re-walk a kernel's Feynman paths without touching any table,
//!   to separate path-walking time from table-insertion time.

use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::circuit::{Circuit, Gate, PushApplicable, PushApplyOutput};
use crate::config::Config;
use crate::gate_scheduler;
use crate::types::{BasisIdx, Complex};
use crate::utility;

pub fn dump_static_kernels<B: BasisIdx>(
    config: &Config,
    circuit: &Circuit<B>,
    path: &Path,
) -> io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    let mut scheduler = gate_scheduler::create_gate_scheduler(config, circuit);
    let mut num_gates_seen = 0;
    let mut kernel_idx = 0;

    loop {
        let gate_indices = scheduler.pick_next_gates();
        if gate_indices.is_empty() {
            break;
        }
        num_gates_seen += gate_indices.len();
        let gates: Vec<&Gate<B>> = gate_indices.iter().map(|&gi| &circuit.gates[gi]).collect();

        let mut touched: Vec<usize> = gates.iter().flat_map(|g| g.touches.iter().copied()).collect();
        touched.sort_unstable();
        touched.dedup();

        let mut line = String::new();
        write!(line, "{{\"kernel\":{},\"gates\":{}", kernel_idx, json_usizes(&gate_indices)).unwrap();
        write!(line, ",\"names\":[").unwrap();
        for (i, g) in gates.iter().enumerate() {
            write!(line, "{}\"{}\"", if i > 0 { "," } else { "" }, g.name()).unwrap();
        }
        write!(line, "],\"qubits\":[").unwrap();
        for (i, g) in gates.iter().enumerate() {
            write!(line, "{}{}", if i > 0 { "," } else { "" }, json_usizes(&g.touches)).unwrap();
        }
        write!(line, "],\"params\":[").unwrap();
        for (i, g) in gates.iter().enumerate() {
            let ps: Vec<String> = g.params().iter().map(|p| format!("{}", p)).collect();
            write!(line, "{}[{}]", if i > 0 { "," } else { "" }, ps.join(",")).unwrap();
        }
        write!(line, "],\"branching_rust\":[").unwrap();
        for (i, g) in gates.iter().enumerate() {
            write!(line, "{}{}", if i > 0 { "," } else { "" }, g.is_branching() as u8).unwrap();
        }
        write!(line, "],\"branching_sml\":[").unwrap();
        for (i, g) in gates.iter().enumerate() {
            write!(line, "{}{}", if i > 0 { "," } else { "" }, g.is_branching_sml() as u8).unwrap();
        }
        writeln!(line, "],\"len\":{},\"k\":{}}}", gates.len(), touched.len()).unwrap();
        out.write_all(line.as_bytes())?;
        kernel_idx += 1;
    }

    assert_eq!(
        num_gates_seen,
        circuit.num_gates(),
        "scheduler must visit every gate exactly once"
    );
    out.flush()
}

pub fn json_usizes(xs: &[usize]) -> String {
    let parts: Vec<String> = xs.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(","))
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WalkProfile {
    pub duration: Duration,
    /// gate applications performed while walking
    pub gate_apps: usize,
    /// paths that reached the end of the kernel with a nonzero weight
    /// (= insertions the real expansion attempts)
    pub leaves: usize,
}

/// Walk every Feynman path of `gates` from each input element exactly like the
/// sparse expansion does (same push_apply, same zero pruning), but drop the
/// leaves instead of inserting them.
pub fn walk_only<B: BasisIdx>(gates: &[&Gate<B>], input: &[(B, Complex)]) -> WalkProfile {
    let start = Instant::now();
    let (gate_apps, leaves, checksum) = input
        .par_iter()
        .fold(
            || (0usize, 0usize, 0.0f32),
            |mut acc, (bidx, weight)| {
                walk(gates, 0, bidx.clone(), *weight, &mut acc);
                acc
            },
        )
        .reduce(|| (0, 0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    let duration = start.elapsed();
    std::hint::black_box(checksum);
    WalkProfile {
        duration,
        gate_apps,
        leaves,
    }
}

fn walk<B: BasisIdx>(
    gates: &[&Gate<B>],
    gatenum: usize,
    bidx: B,
    weight: Complex,
    acc: &mut (usize, usize, f32),
) {
    if utility::is_zero(weight) {
        return;
    }
    if gatenum >= gates.len() {
        acc.1 += 1;
        acc.2 += weight.re;
        std::hint::black_box(bidx);
        return;
    }
    acc.0 += 1;
    match gates[gatenum].push_apply(bidx, weight) {
        PushApplyOutput::Nonbranching(b, w) => walk(gates, gatenum + 1, b, w, acc),
        PushApplyOutput::Branching((b1, w1), (b2, w2)) => {
            walk(gates, gatenum + 1, b1, w1, acc);
            walk(gates, gatenum + 1, b2, w2, acc);
        }
    }
}

pub fn secs(d: Duration) -> f64 {
    d.as_secs_f64()
}
