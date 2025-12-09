use std::time::Instant;

use crate::{
    mcts::{MctsParams, Searcher},
    tree::Node,
};

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
    pub fn get_time(
        time: u64,
        increment: Option<u64>,
        _ply: u32,
        movestogo: Option<u64>,
        params: &MctsParams,
    ) -> (u128, u128) {
        if let Some(mtg) = movestogo {
            // Cyclic time control (x moves in y seconds)
            let max_time = (time as f64 / (mtg as f64).clamp(1.0, 30.0)) as u128;
            (max_time, max_time)
        } else {
            let mtg = params.tm_mtg() as f64;
            let inc = increment.unwrap_or(0) as f64;
            let time_val = time as f64;

            let effective_time = (time_val + inc * (mtg - 1.0)).max(1.0);

            let opt_time = (effective_time * params.tm_opt_base()) as u128;
            let max_time = ((time_val * params.tm_hard_limit()).min(time_val)) as u128;

            (opt_time, max_time)
        }
    }

    pub fn soft_time_cutoff(
        searcher: &Searcher,
        timer: &Instant,
        previous_score: f32,
        best_move_changes: i32,
        nodes: usize,
        time: u128,
    ) -> (bool, f32) {
        let elapsed = timer.elapsed().as_millis();

        // Use more time if our eval is falling, and vice versa
        let (_, mut score) = searcher.get_pv(0);
        score = Searcher::get_cp(score);
        let eval_diff = if previous_score == f32::NEG_INFINITY {
            0.0
        } else {
            previous_score - score
        };

        let feval_scale = (1.0 + eval_diff.max(0.0) * searcher.params.tm_feval_scale())
            .min(searcher.params.tm_feval_max());

        // Use more time if our best move is changing frequently
        let bmi_scale = (1.0 + best_move_changes as f32 * searcher.params.tm_bmi_scale())
            .min(searcher.params.tm_bmi_max());

        // Use less time if our best move has a large percentage of visits, and vice versa
        let (best_child_ptr, _, _) = searcher.get_best_action(searcher.tree.root_node());
        let ratio = if nodes > 0 {
            searcher.tree[best_child_ptr].visits() as f32 / nodes as f32
        } else {
            0.0
        };

        let bmv_scale =
            (1.0 + (1.0 - ratio) * searcher.params.tm_bmv_scale()).min(searcher.params.tm_bmv_max());

        let total_time = (time as f32 * feval_scale * bmi_scale * bmv_scale) as u128;

        (elapsed >= total_time, score)
    }
}
