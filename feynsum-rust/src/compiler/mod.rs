//! Ahead-of-time compilation of GraFeyn kernels (proj.pdf, project 3).
//!
//! Step 1 (this module so far): the kernel IR (`ir`), lowering from circuit
//! gates (`lower`), and a per-kernel differential checker against the
//! interpreter (`check`). Passes and backends come next.

pub mod check;
pub mod ir;
pub mod lower;

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::circuit::Circuit;
use crate::config::Config;
use crate::gate_scheduler;
use crate::types::BasisIdx;

/// Write the lowered IR of every kernel (text, one block per kernel).
pub fn dump_ir<B: BasisIdx>(config: &Config, circuit: &Circuit<B>, path: &Path) -> io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    let mut scheduler = gate_scheduler::create_gate_scheduler(config, circuit);
    let mut idx = 0;
    loop {
        let gate_indices = scheduler.pick_next_gates();
        if gate_indices.is_empty() {
            break;
        }
        write!(out, "#{} ", idx)?;
        match lower::lower_kernel(circuit, &gate_indices) {
            Ok(kernel) => write!(out, "{}", kernel)?,
            Err(e) => writeln!(out, "kernel gates={:?} NOT LOWERED: {}", gate_indices, e)?,
        }
        idx += 1;
    }
    out.flush()
}
