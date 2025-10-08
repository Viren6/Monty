use crate::{
    chess::{ChessState, GameState},
    tree::{Node, NodePtr},
};

use super::{SearchHelpers, Searcher};

#[derive(Clone, Copy, Debug, Default)]
pub struct NodeEvaluation {
    pub value: f32,
    pub draw: f32,
}

impl NodeEvaluation {
    fn new(value: f32, draw: f32) -> Self {
        Self { value, draw }
    }

    fn flipped(self) -> Self {
        Self {
            value: 1.0 - self.value,
            draw: self.draw,
        }
    }
}

pub fn perform_one(
    searcher: &Searcher,
    pos: &mut ChessState,
    ptr: NodePtr,
    depth: &mut usize,
    thread_id: usize,
) -> Option<NodeEvaluation> {
    *depth += 1;

    let cur_hash = pos.hash();
    let mut child_hash: Option<u64> = None;
    let tree = searcher.tree;
    let node = &tree[ptr];

    let eval = if node.is_terminal() || node.visits() == 0 {
        if node.visits() == 0 {
            node.set_state(pos.game_state());
        }

        // probe hash table to use in place of network
        if node.state() == GameState::Ongoing {
            if let Some(entry) = tree.probe_hash(cur_hash) {
                NodeEvaluation::new(entry.q(), entry.d())
            } else {
                get_utility(searcher, ptr, pos)
            }
        } else {
            get_utility(searcher, ptr, pos)
        }
    } else {
        // expand node on the second visit
        if node.is_not_expanded() {
            tree.expand_node(
                ptr,
                pos,
                searcher.params,
                searcher.policy,
                *depth,
                thread_id,
            )?;
        }

        // this node has now been accessed so we need to move its
        // children across if they are in the other tree half
        tree.fetch_children(ptr, thread_id)?;

        // select action to take via PUCT
        let stm = pos.stm();
        let action = pick_action(searcher, ptr, node);

        let child_ptr = node.actions() + action;

        let mov = tree[child_ptr].parent_move();

        pos.make_move(mov);

        // capture child hash (value is stored from the side to move at this child)
        child_hash = Some(pos.hash());

        tree[child_ptr].inc_threads();

        // acquire lock to avoid issues with desynced setting of
        // game state between threads when threads > 1
        let lock = if tree[child_ptr].visits() == 0 {
            Some(node.actions_mut())
        } else {
            None
        };

        // descend further
        let maybe_eval = perform_one(searcher, pos, child_ptr, depth, thread_id);

        drop(lock);

        tree[child_ptr].dec_threads();

        let eval_from_child = maybe_eval?;

        if tree[child_ptr].state() == GameState::Ongoing {
            tree.update_butterfly(stm, mov, eval_from_child.value, searcher.params);
        }

        tree.propogate_proven_mates(ptr, tree[child_ptr].state());

        eval_from_child
    };

    // store value for the side to move at the visited node in TT
    if let Some(h) = child_hash {
        let child_eval = eval.flipped();
        tree.push_hash(h, child_eval.value, child_eval.draw);
    } else {
        tree.push_hash(cur_hash, eval.value, eval.draw);
    }

    // flip perspective and backpropagate
    let flipped = eval.flipped();
    tree.update_node_stats(ptr, flipped.value, flipped.draw, thread_id);
    Some(flipped)
}

fn get_utility(searcher: &Searcher, ptr: NodePtr, pos: &ChessState) -> NodeEvaluation {
    match searcher.tree[ptr].state() {
        GameState::Ongoing => {
            let evaluation = pos.evaluate_wdl(searcher.value, searcher.params);
            NodeEvaluation::new(evaluation.adjusted.score(), evaluation.adjusted.draw)
        }
        GameState::Draw => NodeEvaluation::new(0.5, 1.0),
        GameState::Lost(_) => NodeEvaluation::new(0.0, 0.0),
        GameState::Won(_) => NodeEvaluation::new(1.0, 0.0),
    }
}

fn pick_action(searcher: &Searcher, ptr: NodePtr, node: &Node) -> usize {
    let is_root = ptr == searcher.tree.root_node();

    let cpuct = SearchHelpers::get_cpuct(searcher.params, node, is_root);
    let fpu = SearchHelpers::get_fpu(node);
    let expl_scale = SearchHelpers::get_explore_scaling(searcher.params, node);

    let expl = cpuct * expl_scale;

    let actions_ptr = node.actions();
    let mut acc = 0.0;
    let mut k = 0;
    while k < node.num_actions() && acc < searcher.params.policy_top_p() {
        acc += searcher.tree[actions_ptr + k].policy();
        k += 1;
    }
    let mut limit = k.max(searcher.params.min_policy_actions() as usize);
    let mut thresh = 1u64 << (searcher.params.visit_threshold_power() as u32);
    while node.visits() >= thresh && limit < node.num_actions() {
        limit += 2;
        thresh = thresh.checked_shl(1).unwrap_or(u64::MAX);
    }
    limit = limit.min(node.num_actions());

    searcher
        .tree
        .get_best_child_by_key_lim(ptr, limit, |child| {
            let mut q = SearchHelpers::get_action_value(child, fpu);

            // virtual loss
            let threads = f64::from(child.threads());
            if threads > 0.0 {
                let visits = child.visits() as f64;
                let q2 = f64::from(q) * visits
                    / (visits + 1.0 + searcher.params.virtual_loss_weight() * (threads - 1.0));
                q = q2 as f32;
            }

            let u = expl * child.policy() / (1 + child.visits()) as f32;

            q + u
        })
}
