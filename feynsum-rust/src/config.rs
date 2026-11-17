use std::path::PathBuf;

use crate::gate_scheduler::GateSchedulingPolicy;
use crate::options::Options;
use crate::types::Real;

pub struct Config {
    pub block_size: usize,
    #[allow(dead_code)]
    pub maxload: Real,
    pub gate_scheduling_policy: GateSchedulingPolicy, // TODO: Add pullThreshold
    pub disable_gate_fusion: bool,
    pub dense_threshold: Real,
    pub pull_threshold: Real,
    pub max_branching_stride: usize,
    pub kernel_stats: Option<PathBuf>,
    pub init_random_nonzeros: usize,
    pub seed: u64,
}

impl Config {
    pub fn new(options: &Options) -> Self {
        Self {
            block_size: options.block_size,
            maxload: 0.75, // FIXME
            gate_scheduling_policy: options.gate_schduling_policy,
            disable_gate_fusion: options.disable_gate_fusion,
            dense_threshold: options.dense_threshold,
            pull_threshold: options.pull_threshold,
            max_branching_stride: options.max_branching_stride,
            kernel_stats: options.kernel_stats.clone(),
            init_random_nonzeros: options.init_random_nonzeros,
            seed: options.seed,
        }
    }
}

// used for testing
impl Default for Config {
    fn default() -> Self {
        Self {
            block_size: 10_000,
            maxload: 0.75, // FIXME
            gate_scheduling_policy: GateSchedulingPolicy::GreedyNonbranching,
            disable_gate_fusion: false,
            dense_threshold: 0.25,
            pull_threshold: 0.8,
            max_branching_stride: 2,
            kernel_stats: None,
            init_random_nonzeros: 0,
            seed: 1,
        }
    }
}
