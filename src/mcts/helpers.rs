use std::time::Instant;

use crate::{mcts::MctsParams, tree::Node};

pub struct SearchHelpers;

impl SearchHelpers {
    /// CPUCT
    ///
    /// Larger value implies more exploration.
    pub fn get_cpuct(params: &MctsParams, node: &Node, is_root: bool) -> f32 {
        // baseline CPUCT value
        let mut cpuct = if is_root {
            params.root_cpuct()
        } else {
            params.cpuct()
        };

        // scale CPUCT as visits increase
        let scale = params.cpuct_visits_scale() * 128.0;
        cpuct *= 1.0 + ((node.visits() as f32 + scale) / scale).ln();

        // scale CPUCT with variance of Q
        if node.visits() > 1 {
            let mut frac = node.var().sqrt() / params.cpuct_var_scale();
            frac += (1.0 - frac) / (1.0 + params.cpuct_var_warmup() * node.visits() as f32);
            cpuct *= 1.0 + params.cpuct_var_weight() * (frac - 1.0);
        }

        cpuct
    }

    /// Base Exploration Scaling
    ///
    /// Larger value implies more exploration.
    fn base_explore_scaling(params: &MctsParams, node: &Node) -> f32 {
        (params.expl_tau() * (node.visits().max(1) as f32).ln()).exp()
    }

    /// Exploration Scaling
    ///
    /// Larger value implies more exploration.
    pub fn get_explore_scaling(params: &MctsParams, node: &Node) -> f32 {
        let mut scale = Self::base_explore_scaling(params, node);
        let gini = node.gini_impurity();

        let factor = if cfg!(feature = "datagen") {
            // inverse-gini formula for datagen
            (0.679 - 1.634 * (1.635 - gini).ln()).max(0.3581)
        } else {
            // normal formula
            (params.gini_base() - params.gini_ln_multiplier() * (gini + 0.001).ln())
                .min(params.gini_min())
        };

        scale *= factor;
        scale
    }

    /// Common depth PST
    pub fn get_pst(depth: usize, q: f32, params: &MctsParams) -> f32 {
        let scalar = q - q.min(params.winning_pst_threshold());
        let t = scalar / (1.0 - params.winning_pst_threshold());
        let base_pst = 1.0 - params.base_pst_adjustment()
            + ((depth as f32) - params.root_pst_adjustment()).powf(-params.depth_pst_adjustment());
        base_pst + (params.winning_pst_max() - base_pst) * t
    }

    /// First Play Urgency
    ///
    /// #### Note
    /// Must return a value in [0, 1].
    pub fn get_fpu(node: &Node) -> f32 {
        1.0 - node.q()
    }

    /// Get a predicted win probability for an action
    ///
    /// #### Note
    /// Must return a value in [0, 1].
    pub fn get_action_value(node: &Node, fpu: f32) -> f32 {
        if node.visits() == 0 {
            fpu
        } else {
            node.q()
        }
    }

    /// Calculates the maximum allowed time usage for a search
    ///
    /// #### Note
    /// This will be overriden by a `go movetime` command,
    /// and a move overhead will be applied to this, so no
    /// need for it here.
    pub fn get_time(time: u64, increment: Option<u64>) -> (u128, u128) {
        let inc = increment.unwrap_or(0) as u128;
        let allocation = ((time as u128 + inc) * 16 / 100).max(1);

        (allocation, allocation)
    }

    pub fn soft_time_cutoff(timer: &Instant, time: u128) -> bool {
        timer.elapsed().as_millis() >= time
    }
}
