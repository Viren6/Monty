use std::sync::atomic::{AtomicUsize, Ordering};

use crate::numa::NumaBuffer;

#[repr(align(64))]
#[derive(Default)]
pub struct ThreadStats {
    total_nodes: AtomicUsize,
    total_iters: AtomicUsize,
    main_iters: AtomicUsize,
    seldepth: AtomicUsize,
}

pub struct SearchStats {
    per_thread: NumaBuffer<ThreadStats>, // accessed only by corresponding thread
    pub avg_depth: AtomicUsize,
}

impl SearchStats {
    pub fn new(threads: usize) -> Self {
        Self {
            per_thread: NumaBuffer::new(threads),
            avg_depth: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub fn add_iter(&self, tid: usize, depth: usize, main: bool) {
        let stats = self.per_thread.init_with(tid, ThreadStats::default);
        stats.total_iters.fetch_add(1, Ordering::Relaxed);
        stats.total_nodes.fetch_add(depth, Ordering::Relaxed);
        stats
            .seldepth
            .fetch_max(depth.saturating_sub(1), Ordering::Relaxed);
        if main {
            stats.main_iters.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn total_iters(&self) -> usize {
        (0..self.per_thread.len())
            .filter_map(|idx| self.per_thread.try_get(idx))
            .map(|c| c.total_iters.load(Ordering::Relaxed))
            .sum()
    }

    pub fn total_nodes(&self) -> usize {
        (0..self.per_thread.len())
            .filter_map(|idx| self.per_thread.try_get(idx))
            .map(|c| c.total_nodes.load(Ordering::Relaxed))
            .sum()
    }

    pub fn main_iters(&self) -> usize {
        (0..self.per_thread.len())
            .filter_map(|idx| self.per_thread.try_get(idx))
            .map(|c| c.main_iters.load(Ordering::Relaxed))
            .sum()
    }

    pub fn seldepth(&self) -> usize {
        (0..self.per_thread.len())
            .filter_map(|idx| self.per_thread.try_get(idx))
            .map(|c| c.seldepth.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0)
    }
}
