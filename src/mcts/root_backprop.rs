use std::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, Ordering},
};

use crate::tree::{Node, Tree};

#[derive(Default)]
struct ThreadAccum {
    visits: u32,
    sum_q: u64,
    sum_sq_q: u64,
}

impl ThreadAccum {
    #[inline]
    fn reset(&mut self) {
        self.visits = 0;
        self.sum_q = 0;
        self.sum_sq_q = 0;
    }
}

#[repr(align(64))]
struct ThreadAccumCell {
    inner: UnsafeCell<ThreadAccum>,
}

impl Default for ThreadAccumCell {
    fn default() -> Self {
        Self {
            inner: UnsafeCell::new(ThreadAccum::default()),
        }
    }
}

unsafe impl Sync for ThreadAccumCell {}

fn compute_flush_threshold(threads: usize) -> u32 {
	threads.try_into().unwrap()
}

pub(crate) struct RootBackprop<'a> {
    tree: &'a Tree,
    per_thread: Vec<ThreadAccumCell>,
    flush_threshold: u32,
    activation_visits: u32,
    activated: AtomicBool,
}

unsafe impl Sync for RootBackprop<'_> {}

impl<'a> RootBackprop<'a> {
    pub(crate) fn new(tree: &'a Tree, threads: usize) -> Self {
        let flush_threshold = compute_flush_threshold(threads).max(1);
        let activation_visits = flush_threshold
            .saturating_mul(threads as u32)
            .max(flush_threshold);

        Self {
            tree,
            per_thread: (0..threads).map(|_| ThreadAccumCell::default()).collect(),
            flush_threshold,
            activation_visits,
            activated: AtomicBool::new(false),
        }
    }

    #[inline]
    fn thread_accum(&self, thread_id: usize) -> &ThreadAccumCell {
        &self.per_thread[thread_id]
    }

    #[inline]
    fn root_node(&self) -> &Node {
        let ptr = self.tree.root_node();
        &self.tree[ptr]
    }

    pub(crate) fn record(&self, thread_id: usize, value: f32) {
        let node = self.root_node();

        if !self.activated.load(Ordering::Relaxed) {
            if node.visits() < self.activation_visits {
                node.update(value);
                return;
            }

            self.activated.store(true, Ordering::Relaxed);
        }

        let cell = self.thread_accum(thread_id);
        let accum = unsafe { &mut *cell.inner.get() };

        let q = Node::quantize_value(value);
        accum.visits += 1;
        accum.sum_q += q;
        accum.sum_sq_q += q * q;

        if accum.visits >= self.flush_threshold {
            node.apply_batch_stats(accum.visits, accum.sum_q, accum.sum_sq_q);
            accum.reset();
        }
    }

    pub(crate) fn flush(&self, thread_id: usize) {
        let accum = unsafe { &mut *self.thread_accum(thread_id).inner.get() };

        if accum.visits == 0 {
            return;
        }

        self.root_node()
            .apply_batch_stats(accum.visits, accum.sum_q, accum.sum_sq_q);
        accum.reset();
    }

    pub(crate) fn flush_all(&self) {
        for thread_id in 0..self.per_thread.len() {
            self.flush(thread_id);
        }
    }
}