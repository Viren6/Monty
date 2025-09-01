use std::sync::atomic::{AtomicU64, Ordering};

pub struct InflightTable {
    table: Vec<AtomicU64>,
}

impl InflightTable {
    pub fn new(size: usize, _threads: usize) -> Self {
        let mut table = Vec::with_capacity(size);
        table.resize_with(size, || AtomicU64::new(0));
        Self { table }
    }

    fn idx(&self, hash: u64) -> usize {
        (hash as usize) % self.table.len()
    }

    pub fn try_insert(&self, hash: u64) -> bool {
        let idx = self.idx(hash);
        self.table[idx]
            .compare_exchange(0, hash, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    pub fn contains(&self, hash: u64) -> bool {
        let idx = self.idx(hash);
        self.table[idx].load(Ordering::Relaxed) == hash
    }

    pub fn remove(&self, hash: u64) {
        let idx = self.idx(hash);
        let _ = self.table[idx].compare_exchange(hash, 0, Ordering::Release, Ordering::Relaxed);
    }

    pub fn clear(&self, threads: usize) {
        let chunk_size = self.table.len().div_ceil(threads);
        std::thread::scope(|s| {
            for chunk in self.table.chunks(chunk_size) {
                s.spawn(move || {
                    for entry in chunk {
                        entry.store(0, Ordering::Relaxed);
                    }
                });
            }
        });
    }
}