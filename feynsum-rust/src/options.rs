use std::path::PathBuf;
use structopt::StructOpt;

use crate::gate_scheduler::GateSchedulingPolicy;
use crate::simulator::Simulator;
use crate::types::Real;

#[derive(Debug, StructOpt)]
#[structopt(name = "feynsum", about = "Feynsum quantum simulator")]
pub struct Options {
    #[structopt(
        parse(from_os_str),
        name = "input",
        short = "i",
        long = "input",
        help = "path to the input qasm file"
    )]
    pub input: PathBuf,

    #[structopt(
        parse(from_os_str),
        name = "output",
        short = "o",
        long = "output",
        help = "output file path to dump densities to. if not specified, densities are not printed"
    )]
    pub output: Option<PathBuf>,

    #[structopt(
        name = "gate scheduling policy",
        long = "scheduler",
        short = "s",
        default_value = "naive",
        help = "gate scheduling policy to use"
    )]
    pub gate_schduling_policy: GateSchedulingPolicy,

    #[structopt(long = "dense-threshold", default_value = "0.25")]
    pub dense_threshold: Real,

    #[structopt(long = "pull-threshold", default_value = "0.8")]
    pub pull_threshold: Real,

    #[structopt(long = "disable-gate-fusion")]
    pub disable_gate_fusion: bool,

    #[structopt(
        name = "simulator",
        long = "simulator",
        default_value = "parallel",
        help = "simulator to use"
    )]
    pub simulator: Simulator,

    #[structopt(
        name = "parallelism",
        long = "parallelism",
        default_value = "1",
        help = "number of threads to use",
        required_if("simulator", "parallel")
    )]
    pub parallelism: usize,

    #[structopt(
        name = "block size",
        long = "block-size",
        default_value = "10000",
        help = "block size for parallelism"
    )]
    pub block_size: usize,

    #[structopt(
        long = "max-branching-stride",
        default_value = "2",
        help = "gfq-fused: close a kernel after this many branching gates (feynsum-sml's scheduler-max-branching-stride)"
    )]
    pub max_branching_stride: usize,

    #[structopt(
        parse(from_os_str),
        long = "dump-kernels",
        help = "write the static kernel sequence (JSON lines) to this file and exit without simulating"
    )]
    pub dump_kernels: Option<PathBuf>,

    #[structopt(
        parse(from_os_str),
        long = "kernel-stats",
        help = "parallel simulator: write per-kernel runtime statistics (JSON lines) to this file; adds a separate walk-only timing pass per kernel"
    )]
    pub kernel_stats: Option<PathBuf>,

    #[structopt(
        long = "init-random-nonzeros",
        default_value = "0",
        help = "parallel simulator: start from N distinct random basis states with equal weights instead of |0...0>"
    )]
    pub init_random_nonzeros: usize,

    #[structopt(
        parse(from_os_str),
        long = "dump-ir",
        help = "write the lowered IR of every kernel (text) to this file and exit without simulating"
    )]
    pub dump_ir: Option<PathBuf>,

    #[structopt(
        long = "check-ir",
        help = "lower every kernel to IR and check it against the interpreter on the bits it touches; exit without simulating"
    )]
    pub check_ir: bool,

    #[structopt(long = "seed", default_value = "1", help = "seed for --init-random-nonzeros and --check-ir")]
    pub seed: u64,
}
