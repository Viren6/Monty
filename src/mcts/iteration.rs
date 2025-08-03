use std::sync::atomic::Ordering;

use crate::{
    chess::{ChessState, GameState}, mcts::DEBUG, tree::{Node, NodePtr}
};

use super::{SearchHelpers, Searcher};

pub fn perform_one(
    searcher: &Searcher,
    pos: &mut ChessState,
    ptr: NodePtr,
    depth: &mut usize,
    thread_id: usize,
) -> Option<f32> {
    *depth += 1;

    let hash = pos.hash();
    let tree = searcher.tree;
    let node = &tree[ptr];

    if DEBUG.load(Ordering::Relaxed) {
        println!("New perform one with node: ({}, {}), visits: {}", ptr.half(), ptr.idx(), node.visits());
        print_board(pos);
    }

    let mut u = if node.is_terminal() || node.visits() == 0 {
        if DEBUG.load(Ordering::Relaxed) {
            println!("score 👀 -> {} visits, terminal {}", node.visits(), node.is_terminal());
        }

        if node.visits() == 0 {
            node.set_state(pos.game_state());
        }

        // probe hash table to use in place of network
        if node.state() == GameState::Ongoing {
            if let Some(entry) = tree.probe_hash(hash) {
                entry.q()
            } else {
                get_utility(searcher, ptr, pos)
            }
        } else {
            get_utility(searcher, ptr, pos)
        }
    } else {
        // expand node on the second visit
        if node.is_not_expanded() {
            if DEBUG.load(Ordering::Relaxed) {
                println!("expand!");
            }

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

        if DEBUG.load(Ordering::Relaxed) { 
            println!("Selecting new action for node ({}, {})", ptr.half(), ptr.idx());
        }
        
        // select action to take via PUCT
        let action = pick_action(searcher, ptr, node);

        let child_ptr = node.actions() + action;

        let mov = tree[child_ptr].parent_move();

        pos.make_move(mov);

        if DEBUG.load(Ordering::Relaxed) { 
            println!("New action ({}, {}) -> {}", child_ptr.half(), child_ptr.idx(), mov);
        }

        tree[child_ptr].inc_threads();

        // acquire lock to avoid issues with desynced setting of
        // game state between threads when threads > 1
        let lock = if tree[child_ptr].visits() == 0 {
            Some(node.actions_mut())
        } else {
            None
        };

        // descend further
        let maybe_u = perform_one(searcher, pos, child_ptr, depth, thread_id);

        drop(lock);

        tree[child_ptr].dec_threads();

        let u = maybe_u?;

        tree.propogate_proven_mates(ptr, tree[child_ptr].state());

        u
    };

    // node scores are stored from the perspective
    // **of the parent**, as they are usually only
    // accessed from the parent's POV
    u = 1.0 - u;

    let new_q = node.update(u);
    tree.push_hash(hash, 1.0 - new_q);

    Some(u)
}

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

pub fn print_board(pos: &ChessState) {
    let piece_icons: [[&str; 6]; 2] = [
        [" P", " N", " B", " R", " Q", " K"],
        [" p", " n", " b", " r", " q", " k"],
    ];

    let mut info = Vec::new();
    let fen = format!("FEN: Nope");
    info.push(fen.as_str());
    let zobrist = format!(
        "Zobrist Key: {}",
        pos.hash()
    );
    info.push(zobrist.as_str());
    let castle_rights = format!(
        "Castle Rights: {}",
        pos.board().rights()
    );
    info.push(castle_rights.as_str());
    let side_sign = format!(
        "Side To Move: {}",
        pos.stm()
    );
    info.push(side_sign.as_str());
    let en_passant = format!(
        "En Passant: {}",
        pos.board().enp_sq()
    );
    info.push(en_passant.as_str());
    let half_moves = format!(
        "Half Moves: {}",
        pos.board().halfm()
    );
    info.push(half_moves.as_str());
    let in_check = format!(
        "In Check: {}",
        String::new()
    );
    info.push(in_check.as_str());
    let phase = format!(
        "Phase: {}",
        String::new()
    );
    info.push(phase.as_str());

    let mut result = " -----------------\n".to_string();
    for rank in 0..8 {
        result += "|";
        for file in 0..8 {
            let square = if pos.stm() == 0 { 7 - rank } else { rank } * 8 + if pos.stm() == 0 { file } else { 7 - file };
            
            if square == pos.board().enp_sq() && pos.board().enp_sq() != 0 {
                result += " x";
                continue;
            }

            let piece_type = pos.board().get_pc(1 << square);
            let piece_side = usize::from(pos.board().bbs()[1] & (1 << square) > 0);
            if piece_type == 0 {
                result += " .";
            } else {
                result += piece_icons[piece_side][piece_type - 2];
            }
        }
        result += format!(" | {}", info[rank as usize])
            .as_str();
        result += "\n".to_string().as_str();
    }
    result += " -----------------\n";
    println!("{}", result);
}