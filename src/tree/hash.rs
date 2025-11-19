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
    fn unpack_data(data: u64) -> (u32, f64) {
        let visits = (data >> 32) as u32;
        let q_fixed = (data & 0xFFFFFFFF) as u32;
        let q = q_fixed as f64 / Q_SCALE; // Q is avg_q
        (visits, q)
    }

    fn pack_data(visits: u32, q: f64) -> u64 {
        let q_fixed = (q * Q_SCALE).min((u32::MAX as f64) - 1.0) as u32; // Clamp
        ((visits as u64) << 32) | (q_fixed as u64)
    }
}

pub struct HashTable {
    table: Vec<HashEntryInternal>,
}

impl HashTable {
    pub fn new(size: usize, threads: usize) -> Self {
        let chunk_size = size.div_ceil(threads.max(1));

        let mut table = HashTable { table: Vec::new() };
        table.table.reserve_exact(size);

        // Initialize with zeroed memory (safe for Atomics)
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
                    // Resetting to default (0) is sufficient
                    // Using ptr write_bytes is faster than iterating
                    unsafe {
                         let ptr = chunk.as_mut_ptr() as *mut u8;
                         ptr.write_bytes(0, chunk.len() * std::mem::size_of::<HashEntryInternal>());
                    }
                });
            }
        });
    }

    pub fn get(&self, hash: u64) -> Option<HashEntry> {
        let idx = hash % (self.table.len() as u64);
        let entry = &self.table[idx as usize];

        let key = entry.key.load(Ordering::Relaxed);
        if key == hash {
            let data = entry.data.load(Ordering::Relaxed);
            // Verify key again to avoid tearing/race where key changed but data is old or vice versa
            // A simple double-check isn't perfect but good enough for lock-free TT
            if entry.key.load(Ordering::Relaxed) == hash {
                let (visits, q) = HashEntryInternal::unpack_data(data);
                return Some(HashEntry { q: q as f32, visits });
            }
        }
        None
    }

    pub fn push(&self, hash: u64, val: f32) {
        let idx = hash % (self.table.len() as u64);
        let entry = &self.table[idx as usize];
        let val_f64 = val as f64;

        // Load key
        let key = entry.key.load(Ordering::Relaxed);

        if key == hash {
            // Update existing
            let mut current_data = entry.data.load(Ordering::Relaxed);
            loop {
                let (visits, q) = HashEntryInternal::unpack_data(current_data);
                
                // Incremental average update
                // NewAvg = (OldAvg * OldVisits + Val) / (OldVisits + 1)
                let total_q = q * (visits as f64) + val_f64;
                let new_visits = visits + 1;
                let new_q = total_q / (new_visits as f64);
                
                let new_data = HashEntryInternal::pack_data(new_visits, new_q);

                match entry.data.compare_exchange_weak(
                    current_data,
                    new_data,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(x) => current_data = x,
                }
            }
        } else if key == 0 {
             // Try to claim empty slot
             // First CAS key
             if entry.key.compare_exchange(0, hash, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                 // Successfully claimed, now set data
                 // We can just overwrite data because we own the key (conceptually)
                 // But another thread might have seen key=hash and tried to update data.
                 // So we should use store or CAS. 
                 // Since we just set key from 0, data should be 0.
                 // But better be safe:
                 let new_data = HashEntryInternal::pack_data(1, val_f64);
                 entry.data.store(new_data, Ordering::Relaxed);
             } else {
                 // Failed to claim, someone else wrote key. Recurse/Retry?
                 // Just drop this update or retry once.
                 // Simpler to just drop.
             }
        } else {
            // Collision (key != hash and key != 0)
            // Heuristic: Replace if stored visits is small (e.g. <= 1)
            // This allows new positions to eventually take over stale ones, 
            // but protects valuable high-visit nodes.
            let current_data = entry.data.load(Ordering::Relaxed);
            let (visits, _) = HashEntryInternal::unpack_data(current_data);
            
            if visits <= 1 {
                 // Replace
                 // We need to update key AND data.
                 // This is racy. We might update key, then another thread updates data for NEW key using OLD data...
                 // "Hyena" hashing or similar helps.
                 // For simplicity, we just overwrite.
                 // Data first or Key first?
                 // If we set Key first, other threads might read garbage Data.
                 // If we set Data first, other threads might read new Data with old Key (mismatch, ignored).
                 // So Data first seems safer for readers.
                 let new_data = HashEntryInternal::pack_data(1, val_f64);
                 entry.data.store(new_data, Ordering::Relaxed);
                 entry.key.store(hash, Ordering::Relaxed);
            }
        }
    }
}
