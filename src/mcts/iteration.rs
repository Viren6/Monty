use crate::{
    chess::{ChessState, GameState, Move},
    tree::{Node, NodePtr},
};

use super::{SearchHelpers, Searcher};

fn get_utility(searcher: &Searcher, ptr: NodePtr, pos: &ChessState) -> f32 {
    match searcher.tree[ptr].state() {
        GameState::Ongoing => pos.get_value_wdl(searcher.value, searcher.params),
        GameState::Draw => 0.5,
        GameState::Lost(_) => 0.0,
        GameState::Won(_) => 1.0,
    }
}

fn pick_action(searcher: &Searcher, ptr: NodePtr, node: &Node) -> usize {
    let is_root = ptr == searcher.tree.root_node();

    let cpuct = SearchHelpers::get_cpuct(searcher.params, node, is_root);
    let fpu = SearchHelpers::get_fpu(node);
    let expl_scale = SearchHelpers::get_explore_scaling(searcher.params, node);

    let expl = cpuct * expl_scale;

    searcher.tree.get_best_child_by_key(ptr, |child| {
        let mut q = SearchHelpers::get_action_value(child, fpu);

        // virtual loss
        let threads = f64::from(child.threads());
        if threads > 0.0 {
            let visits = f64::from(child.visits());
            let q2 = f64::from(q) * visits
                / (visits + 1.0 + searcher.params.virtual_loss_weight() * (threads - 1.0));
            q = q2 as f32;
        }

        let u = expl * child.policy() / (1 + child.visits()) as f32;

        q + u
    })
}

pub struct PpbResult {
    pub value: f32,
    pub new_path: Option<Vec<Move>>,
}

pub fn perform_one_ppb(
    searcher: &Searcher,
    pos: &mut ChessState,
    ptr: NodePtr,
    depth: &mut usize,
    thread_id: usize,
    path: &mut Vec<Move>,
) -> Option<PpbResult> {
    *depth += 1;

    let hash = pos.hash();
    let tree = searcher.tree;
    let node = &tree[ptr];

    if node.is_terminal() || node.visits() == 0 {
        if node.visits() == 0 {
            node.set_state(pos.game_state());
        }

        let mut u = if node.state() == GameState::Ongoing {
            if let Some(entry) = tree.probe_hash(hash) {
                entry.q()
            } else {
                get_utility(searcher, ptr, pos)
            }
        } else {
            get_utility(searcher, ptr, pos)
        };

        u = 1.0 - u;
        let new_q = node.update(u);
        tree.push_hash(hash, 1.0 - new_q);

        return Some(PpbResult { value: u, new_path: None });
    }

    if node.is_not_expanded() {
        tree.expand_node(ptr, pos, searcher.params, searcher.policy, *depth, thread_id)?;
    }

    tree.fetch_children(ptr, thread_id)?;

    let action = pick_action(searcher, ptr, node);
    let child_ptr = node.actions() + action;
    let mov = tree[child_ptr].parent_move();

    pos.make_move(mov);
    path.push(mov);

    tree[child_ptr].inc_threads();
    let lock = if tree[child_ptr].visits() == 0 {
        Some(node.actions_mut())
    } else {
        None
    };

    let child_res = perform_one_ppb(searcher, pos, child_ptr, depth, thread_id, path);

    drop(lock);

    tree[child_ptr].dec_threads();

    let child_res = child_res?;

    tree.propogate_proven_mates(ptr, tree[child_ptr].state());

    path.pop();

    let u = 1.0 - child_res.value;
    let new_q = node.update(u);
    tree.push_hash(hash, 1.0 - new_q);

    if let Some(new_path) = child_res.new_path {
        return Some(PpbResult { value: u, new_path: Some(new_path) });
    }

    if !node.is_not_expanded() {
        let best = pick_action(searcher, ptr, node);
        if best == action {
            let mut new_path = path.clone();
            new_path.push(mov);
            return Some(PpbResult { value: u, new_path: Some(new_path) });
        }
    }

    Some(PpbResult { value: u, new_path: None })
}
