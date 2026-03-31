use std::{
    collections::VecDeque,
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::Duration,
};

use crate::{
    chess::ChessState,
    lc0::{
        mapping::monty_move_to_lc0_index,
        worker::{Lc0Result, Lc0Worker},
    },
    tree::{Node, NodePtr, Tree},
};

pub struct Lc0Config {
    pub num_workers: usize,
    pub network_path: String,
    pub backend: String,
    pub batch_size: usize,
    pub value_weight: u64,
    pub chess960: bool,
}

impl Default for Lc0Config {
    fn default() -> Self {
        Self {
            num_workers: 0,
            network_path: String::new(),
            backend: "onnx-trt".to_string(),
            batch_size: 32,
            value_weight: 128,
            chess960: false,
        }
    }
}

struct PendingBatch {
    node_ptrs: Vec<NodePtr>,
    positions: Vec<ChessState>,
    tree_half: usize,
    actual_count: usize,
}

pub struct Lc0Coordinator {
    workers: Vec<Lc0Worker>,
    pub config: Lc0Config,
}

impl Lc0Coordinator {
    pub fn new(config: Lc0Config) -> Self {
        Self {
            workers: Vec::new(),
            config,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.num_workers > 0 && !self.config.network_path.is_empty()
    }

    /// Spawn worker processes. Called when config changes.
    /// Workers start loading the network immediately in the background.
    pub fn prepare(&mut self) {
        // Kill existing workers
        self.workers.clear();

        if !self.is_enabled() {
            return;
        }

        eprintln!(
            "info string spawning {} lc0 workers (batch_size={})",
            self.config.num_workers, self.config.batch_size
        );

        for _ in 0..self.config.num_workers {
            self.workers.push(Lc0Worker::spawn(
                &self.config.network_path,
                &self.config.backend,
                self.config.batch_size,
                self.config.chess960,
            ));
        }
    }

    /// Spawn workers if needed and block until all are ready.
    /// Called from `isready` handler.
    pub fn ensure_ready(&mut self) {
        if self.is_enabled() && self.workers.is_empty() {
            self.prepare();
        }
        for worker in &self.workers {
            worker.wait_ready();
        }
    }

    pub fn run_loop(&self, tree: &Tree, abort: &AtomicBool) {
        if self.workers.is_empty() {
            return;
        }

        let mut pending_batches: Vec<Option<PendingBatch>> =
            (0..self.workers.len()).map(|_| None).collect();

        let mut total_applied = 0usize;
        let value_weight = self.config.value_weight;
        let batch_size = self.config.batch_size;

        while !abort.load(Ordering::Relaxed) {
            let current_half = tree.half();

            // 1. Check workers for completed results
            for (i, worker) in self.workers.iter().enumerate() {
                if !worker.is_busy() {
                    continue;
                }

                if let Some(results) = worker.try_recv() {
                    worker.set_busy(false);

                    if let Some(pending) = pending_batches[i].take() {
                        // Discard if tree half changed
                        if pending.tree_half != current_half {
                            continue;
                        }

                        let applied = apply_results(
                            tree,
                            &results,
                            &pending.node_ptrs,
                            &pending.positions,
                            pending.actual_count,
                            value_weight,
                        );
                        total_applied += applied;

                        if applied > 0 && total_applied % 50 == 0 {
                            eprintln!(
                                "info string lc0 refined {} nodes total",
                                total_applied
                            );
                        }
                    }
                }
            }

            // 2. If a worker is idle, find candidates and dispatch
            for (i, worker) in self.workers.iter().enumerate() {
                if worker.is_busy() {
                    continue;
                }

                let candidates = find_candidates(
                    tree,
                    tree.root_position(),
                    batch_size,
                );

                if candidates.is_empty() {
                    break;
                }

                let actual_count = candidates.len();
                let mut fens: Vec<String> = Vec::with_capacity(actual_count);
                let mut node_ptrs: Vec<NodePtr> = Vec::with_capacity(actual_count);
                let mut positions: Vec<ChessState> = Vec::with_capacity(actual_count);

                for (ptr, pos) in &candidates {
                    fens.push(pos.board().as_fen());
                    node_ptrs.push(*ptr);
                    positions.push(pos.clone());
                }

                worker.send_batch(&fens);

                pending_batches[i] = Some(PendingBatch {
                    node_ptrs,
                    positions,
                    tree_half: current_half,
                    actual_count,
                });
            }

            thread::sleep(Duration::from_millis(1));
        }
    }
}

/// BFS from root to find highest-visit expanded nodes with lc0_status == 0.
/// Returns up to `batch_size` candidates sorted by visits descending.
fn find_candidates(
    tree: &Tree,
    root_pos: &ChessState,
    batch_size: usize,
) -> Vec<(NodePtr, ChessState)> {
    struct Candidate {
        ptr: NodePtr,
        pos: ChessState,
        visits: u64,
    }

    let root_ptr = tree.root_node();
    let root_node = &tree[root_ptr];

    if !root_node.has_children() {
        return Vec::new();
    }

    // BFS with depth limit
    let mut queue: VecDeque<(NodePtr, ChessState)> = VecDeque::new();
    queue.push_back((root_ptr, root_pos.clone()));

    let mut candidates: Vec<Candidate> = Vec::new();
    let max_depth = 6;
    let mut depth_markers: VecDeque<usize> = VecDeque::new();
    depth_markers.push_back(1);
    let mut current_depth = 0;

    while let Some((ptr, pos)) = queue.pop_front() {
        // Track depth
        if let Some(front) = depth_markers.front_mut() {
            *front -= 1;
            if *front == 0 {
                depth_markers.pop_front();
                current_depth += 1;
            }
        }

        let node = &tree[ptr];

        if !node.has_children() {
            continue;
        }

        // This node is expanded - check if it's a candidate
        if node.lc0_status() == Node::LC0_UNPROCESSED && node.try_mark_lc0_pending() {
            candidates.push(Candidate {
                ptr,
                pos: pos.clone(),
                visits: node.visits(),
            });
        }

        // Expand children into BFS if within depth limit
        if current_depth < max_depth {
            let first_child = node.actions();
            let num_actions = node.num_actions();
            let mut children_count = 0;

            for action in 0..num_actions {
                let child_ptr = first_child + action;
                let child = &tree[child_ptr];

                if child.has_children() {
                    let mut child_pos = pos.clone();
                    child_pos.make_move(child.parent_move());
                    queue.push_back((child_ptr, child_pos));
                    children_count += 1;
                }
            }

            if children_count > 0 {
                depth_markers.push_back(children_count);
            }
        }
    }

    // Sort by visits descending, take top batch_size
    candidates.sort_unstable_by(|a, b| b.visits.cmp(&a.visits));
    candidates.truncate(batch_size);

    candidates
        .into_iter()
        .map(|c| (c.ptr, c.pos))
        .collect()
}

/// Apply LC0 results to tree nodes: replace policy and inject value.
fn apply_results(
    tree: &Tree,
    results: &[Lc0Result],
    node_ptrs: &[NodePtr],
    positions: &[ChessState],
    actual_count: usize,
    value_weight: u64,
) -> usize {
    let mut applied = 0;

    for i in 0..actual_count.min(results.len()).min(node_ptrs.len()) {
        let ptr = node_ptrs[i];
        let pos = &positions[i];
        let result = &results[i];
        let node = &tree[ptr];

        // Validate result
        if result.value.is_nan() || result.value.is_infinite() {
            node.mark_lc0_done();
            continue;
        }

        if result.policy_logits.is_empty() {
            node.mark_lc0_done();
            continue;
        }

        // Apply policy: map lc0 logits to children
        let first_child = node.actions();
        let num_actions = node.num_actions();

        // Build lc0 index -> logit lookup
        let mut lc0_logits = [f32::NEG_INFINITY; 1858];
        for &(idx, logit) in &result.policy_logits {
            if idx < 1858 && logit.is_finite() {
                lc0_logits[idx] = logit;
            }
        }

        // Map children to lc0 logits and compute softmax
        let mut child_logits = vec![f32::NEG_INFINITY; num_actions];
        let mut max_logit = f32::NEG_INFINITY;

        for action in 0..num_actions {
            let child = &tree[first_child + action];
            let mov = child.parent_move();

            if let Some(lc0_idx) = monty_move_to_lc0_index(mov, pos) {
                let logit = lc0_logits[lc0_idx];
                child_logits[action] = logit;
                if logit > max_logit {
                    max_logit = logit;
                }
            }
        }

        if max_logit == f32::NEG_INFINITY {
            node.mark_lc0_done();
            continue;
        }

        // Softmax
        let mut sum_exp = 0.0f32;
        let mut probs = vec![0.0f32; num_actions];
        for (action, logit) in child_logits.iter().enumerate() {
            if *logit > f32::NEG_INFINITY {
                let p = (*logit - max_logit).exp();
                probs[action] = p;
                sum_exp += p;
            }
        }

        if sum_exp <= 0.0 {
            node.mark_lc0_done();
            continue;
        }

        // Set policy on children
        let scale = 1.0 / sum_exp;
        for action in 0..num_actions {
            let policy = probs[action] * scale;
            tree[first_child + action].set_policy(policy);
        }

        // Inject value as value_weight virtual visits
        // LC0 value is -1..1 from STM perspective, convert to 0..1
        let score = ((result.value + 1.0) / 2.0).clamp(0.0, 1.0);
        let draw = result.draw.clamp(0.0, 1.0);

        const QUANT: i32 = 16384 * 4;
        let q = f64::from(score) * f64::from(QUANT);
        let d = f64::from(draw) * f64::from(QUANT);
        let w = value_weight;

        use crate::tree::node::NodeStatsDelta;
        let delta = NodeStatsDelta {
            visits: w,
            sum_q: (q * w as f64) as u64,
            sum_sq_q: (q * q * w as f64) as u64,
            draws: (d * w as f64) as u64,
        };
        node.apply_delta(delta);

        node.mark_lc0_done();
        applied += 1;
    }

    applied
}
