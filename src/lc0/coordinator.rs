use std::{
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    thread,
    time::{Duration, Instant},
};

use crate::{
    chess::{Castling, Position},
    lc0::{
        ffi::{Lc0FfiWorker, lc0_init_shared},
        mapping::monty_move_to_lc0_index,
        worker::Lc0Result,
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
    positions: Vec<(Position, Castling)>,
    tree_half: usize,
    actual_count: usize,
}

pub struct Lc0Coordinator {
    workers: Vec<Lc0FfiWorker>,
    pub config: Lc0Config,
    total_refined: AtomicUsize,
}

impl Lc0Coordinator {
    pub fn new(config: Lc0Config) -> Self {
        Self {
            workers: Vec::new(),
            config,
            total_refined: AtomicUsize::new(0),
        }
    }

    pub fn reset_refined(&self) {
        self.total_refined.store(0, Ordering::Relaxed);
    }

    pub fn total_refined(&self) -> usize {
        self.total_refined.load(Ordering::Relaxed)
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
            "info string loading lc0 network, spawning {} workers (batch_size={})",
            self.config.num_workers, self.config.batch_size
        );

        // Single network init - one copy on GPU shared by all workers
        let shared_handle = lc0_init_shared(
            &self.config.network_path,
            &self.config.backend,
            self.config.chess960,
            self.config.batch_size,
        );

        for _ in 0..self.config.num_workers {
            self.workers.push(Lc0FfiWorker::new(shared_handle));
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

        let value_weight = self.config.value_weight;
        let batch_size = self.config.batch_size;
        let mut last_report = Instant::now();

        // Local buffer of candidates, fed by the expansion queue
        let mut candidate_pool: Vec<(NodePtr, Position, Castling)> = Vec::new();

        while !abort.load(Ordering::Relaxed) && !tree.is_full() {
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

                        if applied > 0 {
                            self.total_refined.fetch_add(applied, Ordering::Relaxed);
                            if last_report.elapsed() >= Duration::from_secs(1) {
                                eprintln!(
                                    "info string lc0 refined {} nodes total",
                                    self.total_refined.load(Ordering::Relaxed)
                                );
                                last_report = Instant::now();
                            }
                        }
                    }
                }
            }

            // 2. Drain newly expanded nodes from the tree's queue
            let new_nodes = tree.drain_lc0_queue();
            candidate_pool.extend(new_nodes);

            // Cap pool size to avoid unbounded growth
            const MAX_POOL: usize = 1024;
            if candidate_pool.len() > MAX_POOL * 2 {
                // O(n) partial sort to keep the top MAX_POOL by visits
                candidate_pool.select_nth_unstable_by(MAX_POOL, |a, b| {
                    tree[b.0].visits().cmp(&tree[a.0].visits())
                });
                candidate_pool.truncate(MAX_POOL);
            }

            // 3. If a worker is idle, pick best candidates and dispatch
            let idle_count = self.workers.iter().filter(|w| !w.is_busy()).count();
            if idle_count > 0 && !candidate_pool.is_empty() {
                let needed = batch_size * idle_count;
                let selected = select_top_candidates(tree, &mut candidate_pool, needed);

                if !selected.is_empty() {
                    let mut resolved_iter = selected.into_iter();

                    for (i, worker) in self.workers.iter().enumerate() {
                        if worker.is_busy() {
                            continue;
                        }

                        let batch: Vec<(NodePtr, Position, Castling)> =
                            resolved_iter.by_ref().take(batch_size).collect();

                        if batch.is_empty() {
                            break;
                        }

                        let actual_count = batch.len();
                        let mut boards: Vec<Position> = Vec::with_capacity(actual_count);
                        let mut node_ptrs: Vec<NodePtr> = Vec::with_capacity(actual_count);
                        let mut positions: Vec<(Position, Castling)> = Vec::with_capacity(actual_count);

                        for &(ptr, board, castling) in &batch {
                            boards.push(board);
                            node_ptrs.push(ptr);
                            positions.push((board, castling));
                        }

                        worker.send_batch(&boards);

                        pending_batches[i] = Some(PendingBatch {
                            node_ptrs,
                            positions,
                            tree_half: current_half,
                            actual_count,
                        });
                    }
                }
            }

            thread::sleep(Duration::from_millis(1));
        }
    }
}

/// Select top candidates by visit count from the pool.
/// Removes selected ones from the pool. Pool is bounded to ~1024 entries.
fn select_top_candidates(
    tree: &Tree,
    pool: &mut Vec<(NodePtr, Position, Castling)>,
    limit: usize,
) -> Vec<(NodePtr, Position, Castling)> {
    if pool.is_empty() {
        return Vec::new();
    }

    if pool.len() <= limit {
        return pool
            .drain(..)
            .filter(|(ptr, _, _)| {
                tree[*ptr].lc0_status() == Node::LC0_UNPROCESSED
                    && tree[*ptr].try_mark_lc0_pending()
            })
            .collect();
    }

    // O(n) partial sort: partition so top `limit` by visits are at the front
    let pivot = limit.min(pool.len() - 1);
    pool.select_nth_unstable_by(pivot, |a, b| tree[b.0].visits().cmp(&tree[a.0].visits()));

    pool.drain(..=pivot)
        .filter(|(ptr, _, _)| {
            tree[*ptr].lc0_status() == Node::LC0_UNPROCESSED
                && tree[*ptr].try_mark_lc0_pending()
        })
        .collect()
}

/// Apply LC0 results to tree nodes: replace policy and inject value.
fn apply_results(
    tree: &Tree,
    results: &[Lc0Result],
    node_ptrs: &[NodePtr],
    positions: &[(Position, Castling)],
    actual_count: usize,
    value_weight: u64,
) -> usize {
    let mut applied = 0;

    for i in 0..actual_count.min(results.len()).min(node_ptrs.len()) {
        let ptr = node_ptrs[i];
        let (board, castling) = &positions[i];
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

            if let Some(lc0_idx) = monty_move_to_lc0_index(mov, board, castling) {
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
        // Nodes store Q from the parent's perspective, so flip: 1.0 - stm_score
        let lc0_stm_score = ((result.value + 1.0) / 2.0).clamp(0.0, 1.0);
        let score = 1.0 - lc0_stm_score;
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
