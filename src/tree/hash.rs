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
        let idx = (hash & self.mask) as usize;
        // Check Primary Slot
        let entry0 = unsafe { self.table.get_unchecked(idx) };
        if let Some(e) = self.read_entry(entry0, hash) {
            return Some(e);
        }

        // Check Secondary Slot (XOR pair)
        let entry1 = unsafe { self.table.get_unchecked(idx ^ 1) };
        if let Some(e) = self.read_entry(entry1, hash) {
            return Some(e);
        }

        None
    }

    #[inline(always)]
    fn read_entry(&self, entry: &HashEntryInternal, hash: u64) -> Option<HashEntry> {
        let key = entry.key.load(Ordering::Relaxed);
        if key == hash {
            let data = entry.data.load(Ordering::Relaxed);
            if entry.key.load(Ordering::Relaxed) == hash {
                let (visits, q) = HashEntryInternal::unpack_data(data);
                return Some(HashEntry { q: q as f32, visits });
            }
        }
        None
    }

    #[inline(always)]
    fn update_entry(&self, entry: &HashEntryInternal, val: f64) {
        let current_data = entry.data.load(Ordering::Relaxed);
        let (visits, q) = HashEntryInternal::unpack_data(current_data);
        
        let new_visits = visits.saturating_add(1);
        // Moving average
        let new_q = q + (val - q) / (new_visits as f64);
        
        let new_data = HashEntryInternal::pack_data(new_visits, new_q);
        entry.data.store(new_data, Ordering::Relaxed);
    }

    #[inline(always)]
    fn overwrite_entry(&self, entry: &HashEntryInternal, hash: u64, val: f64) {
        // Determine order: Data then Key is safer for readers (prevents reading garbage data with new key)
        // But readers check Key twice, so we want to ensure Key matches Data.
        // 1. Write Data (visits=1, q=val)
        let new_data = HashEntryInternal::pack_data(1, val);
        entry.data.store(new_data, Ordering::Relaxed);
        // 2. Write Key
        entry.key.store(hash, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn push(&self, hash: u64, val: f32) {
        let idx = (hash & self.mask) as usize;
        // Define slots as `idx` and `idx ^ 1`
        let idx0 = idx;
        let idx1 = idx ^ 1;
        
        let entry0 = unsafe { self.table.get_unchecked(idx0) };
        let entry1 = unsafe { self.table.get_unchecked(idx1) };
        
        let val_f64 = val as f64;

        // 1. Try to Update Existing
        if entry0.key.load(Ordering::Relaxed) == hash {
            self.update_entry(entry0, val_f64);
            return;
        }
        if entry1.key.load(Ordering::Relaxed) == hash {
            self.update_entry(entry1, val_f64);
            return;
        }

        // 2. Insert New
        // Load metadata to decide where to put it
        let d0 = entry0.data.load(Ordering::Relaxed);
        let (v0, _) = HashEntryInternal::unpack_data(d0);
        let k0 = entry0.key.load(Ordering::Relaxed);

        if k0 == 0 { // Empty slot 0
            self.overwrite_entry(entry0, hash, val_f64);
            return;
        }

        let d1 = entry1.data.load(Ordering::Relaxed);
        let (v1, _) = HashEntryInternal::unpack_data(d1);
        let k1 = entry1.key.load(Ordering::Relaxed);

        if k1 == 0 { // Empty slot 1
            self.overwrite_entry(entry1, hash, val_f64);
            return;
        }

        // Both occupied. Replacement strategy.
        // We want to preserve the entry with HIGHER visits in one of the slots (usually slot 0).
        // And put the new entry in the other.
        
        // If Slot 1 is better than Slot 0, promote Slot 1 to Slot 0.
        if v1 > v0 {
             // Move 1 -> 0
             // We just overwrite 0 with 1's content
             // We need to read 1's Q again or use packed data
             // Ideally copy atomic to atomic? No.
             // Just overwrite 0 with what we read from 1.
             // Since we are lock-free, data might be slightly stale, but that's fine.
             entry0.data.store(d1, Ordering::Relaxed);
             entry0.key.store(k1, Ordering::Relaxed);
             
             // Now overwrite Slot 1 with NEW
             self.overwrite_entry(entry1, hash, val_f64);
        } else {
            // Slot 0 is better (or equal).
            // Overwrite Slot 1 with NEW (Victimizing Slot 1).
            self.overwrite_entry(entry1, hash, val_f64);
        }
    }
}
