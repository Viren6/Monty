use std::sync::atomic::{AtomicUsize, Ordering};

#[repr(align(64))]
#[derive(Default)]
pub struct ThreadStats {
    total_nodes: AtomicUsize,
    total_iters: AtomicUsize,
    main_iters: AtomicUsize,
    seldepth: AtomicUsize,
}

pub struct SearchStats {
    per_thread: Vec<ThreadStats>, // accessed only by corresponding thread
    pub avg_depth: AtomicUsize,
    next_main_time_check: AtomicUsize,
    next_opt_time_check: AtomicUsize,
    next_best_move_reset: AtomicUsize,
    #[cfg(not(feature = "uci-minimal"))]
    next_uci_report: AtomicUsize,
}

impl SearchStats {
    pub fn new(threads: usize) -> Self {
        Self {
            per_thread: (0..threads).map(|_| ThreadStats::default()).collect(),
            avg_depth: AtomicUsize::new(0),
            next_main_time_check: AtomicUsize::new(128),
            next_opt_time_check: AtomicUsize::new(4096),
            next_best_move_reset: AtomicUsize::new(16384),
            #[cfg(not(feature = "uci-minimal"))]
            next_uci_report: AtomicUsize::new(8192),
        }
    }

    fn advance_threshold(threshold: &AtomicUsize, step: usize, iters: usize) -> bool {
        let mut current = threshold.load(Ordering::Relaxed);
        loop {
            if iters < current {
                return false;
            }

            match threshold.compare_exchange(
                current,
                current + step,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    pub fn should_check_main_time(&self, iters: usize) -> bool {
        Self::advance_threshold(&self.next_main_time_check, 128, iters)
    }

    pub fn should_check_opt_time(&self, iters: usize) -> bool {
        Self::advance_threshold(&self.next_opt_time_check, 4096, iters)
    }

    pub fn should_reset_best_move(&self, iters: usize) -> bool {
        Self::advance_threshold(&self.next_best_move_reset, 16384, iters)
    }

    #[cfg(not(feature = "uci-minimal"))]
    pub fn should_emit_uci_report(&self, iters: usize) -> bool {
        Self::advance_threshold(&self.next_uci_report, 8192, iters)
    }

    #[inline]
    pub fn add_iter(&self, tid: usize, depth: usize, main: bool) {
        let stats = &self.per_thread[tid];
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
        self.per_thread
            .iter()
            .map(|c| c.total_iters.load(Ordering::Relaxed))
            .sum()
    }

    pub fn total_nodes(&self) -> usize {
        self.per_thread
            .iter()
            .map(|c| c.total_nodes.load(Ordering::Relaxed))
            .sum()
    }

    pub fn main_iters(&self) -> usize {
        self.per_thread
            .iter()
            .map(|c| c.main_iters.load(Ordering::Relaxed))
            .sum()
    }

    pub fn seldepth(&self) -> usize {
        self.per_thread
            .iter()
            .map(|c| c.seldepth.load(Ordering::Relaxed))
            .max()
            .unwrap_or(0)
    }
}
