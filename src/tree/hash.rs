use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Default)]
pub struct HashEntry {
    pub q: f32,
    pub visits: u32,
}

impl HashEntry {
    pub fn q(&self) -> f32 {
        self.q
    }
}

#[derive(Default)]
#[repr(align(16))]
struct HashEntryInternal {
    key: AtomicU64,
    data: AtomicU64, // Packed: high 32 bits = visits, low 32 bits = fixed-point Q
}

const Q_SCALE: f64 = (1u64 << 32) as f64;

impl HashEntryInternal {
    #[inline(always)]
    fn unpack_data(data: u64) -> (u32, f64) {
        let visits = (data >> 32) as u32;
        let q_fixed = (data & 0xFFFFFFFF) as u32;
        let q = q_fixed as f64 / Q_SCALE;
        (visits, q)
    }

    #[inline(always)]
    fn pack_data(visits: u32, q: f64) -> u64 {
        let q_fixed = (q * Q_SCALE).min((u32::MAX as f64) - 1.0) as u32;
        ((visits as u64) << 32) | (q_fixed as u64)
    }
}

pub struct HashTable {
    table: Vec<HashEntryInternal>,
    mask: u64,
}

impl HashTable {
    pub fn new(size: usize, threads: usize) -> Self {
        // Ensure size is power of 2 for fast indexing
        let size = size.next_power_of_two();
        let mask = (size as u64) - 1;
        let chunk_size = size.div_ceil(threads.max(1));

        let mut table = HashTable { 
            table: Vec::new(),
            mask 
        };
        table.table.reserve_exact(size);

        unsafe {
            use std::mem::{size_of, MaybeUninit};
            let ptr = table.table.as_mut_ptr().cast();
            let uninit: &mut [MaybeUninit<u8>] =
                std::slice::from_raw_parts_mut(ptr, size * size_of::<HashEntryInternal>());

            std::thread::scope(|s| {
                for chunk in uninit.chunks_mut(chunk_size) {
                    s.spawn(|| {
                        chunk.as_mut_ptr().write_bytes(0, chunk.len());
                    });
                }
            });

            table.table.set_len(size);
        }

        table
    }

    pub fn clear(&mut self, threads: usize) {
        let chunk_size = self.table.len().div_ceil(threads.max(1));

        std::thread::scope(|s| {
            for chunk in self.table.chunks_mut(chunk_size) {
                s.spawn(|| {
                    unsafe {
                         let ptr = chunk.as_mut_ptr() as *mut u8;
                         ptr.write_bytes(0, chunk.len() * std::mem::size_of::<HashEntryInternal>());
                    }
                });
            }
        });
    }

    #[inline(always)]
    pub fn get(&self, hash: u64) -> Option<HashEntry> {
        // Fast index using AND mask (since size is power of 2)
        let idx = (hash & self.mask) as usize;
        // Unsafe get check removal could be faster but Vec bounds check is cheap enough usually.
        // To be safe and fast, use get_unchecked if we trust logic, but standard index is fine.
        let entry = unsafe { self.table.get_unchecked(idx) };

        let key = entry.key.load(Ordering::Relaxed);
        if key == hash {
            let data = entry.data.load(Ordering::Relaxed);
            // Verify key again to reduce tearing probability
            if entry.key.load(Ordering::Relaxed) == hash {
                let (visits, q) = HashEntryInternal::unpack_data(data);
                return Some(HashEntry { q: q as f32, visits });
            }
        }
        None
    }

    #[inline(always)]
    pub fn push(&self, hash: u64, val: f32) {
        let idx = (hash & self.mask) as usize;
        let entry = unsafe { self.table.get_unchecked(idx) };
        let val_f64 = val as f64;

        let key = entry.key.load(Ordering::Relaxed);

        if key == hash {
            // Update existing entry
            // We use a "lazy" update without CAS loop to avoid contention.
            // This is lossy but much faster.
            let current_data = entry.data.load(Ordering::Relaxed);
            let (visits, q) = HashEntryInternal::unpack_data(current_data);
            
            let new_visits = visits.saturating_add(1);
            // Incremental average: Q_new = Q_old + (Val - Q_old) / N
            let new_q = q + (val_f64 - q) / (new_visits as f64);
            
            let new_data = HashEntryInternal::pack_data(new_visits, new_q);
            entry.data.store(new_data, Ordering::Relaxed);
        } else {
            // Collision or Empty
            // Replacement strategy:
            // If empty (key == 0), always take.
            // If collision, replace if existing entry has low visits (low confidence).
            // This protects valuable high-visit nodes from being overwritten by new low-visit nodes.
            
            let replace = if key == 0 {
                true
            } else {
                let current_data = entry.data.load(Ordering::Relaxed);
                let (visits, _) = HashEntryInternal::unpack_data(current_data);
                // Threshold: if visits are low, we can replace.
                // Heuristic: 1 or 2 visits is "noise" or "new".
                visits <= 2
            };

            if replace {
                // We replace. 
                // Store Data first, then Key? Or Key then Data?
                // If Key first: Readers see new Key, old Data (garbage).
                // If Data first: Readers see old Key, new Data (mismatch, ignored).
                // So Data first is safer.
                let new_data = HashEntryInternal::pack_data(1, val_f64);
                entry.data.store(new_data, Ordering::Relaxed);
                entry.key.store(hash, Ordering::Relaxed);
            }
        }
    }
}
