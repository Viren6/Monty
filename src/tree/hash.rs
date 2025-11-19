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
    data: AtomicU64, // Packed: 32 visits | 16 checksum | 16 q
}

// Q mapped to 0..u16::MAX
const Q_SCALE: f64 = 65535.0;

impl HashEntryInternal {
    #[inline(always)]
    fn checksum(hash: u64) -> u16 {
        (hash >> 48) as u16
    }

    #[inline(always)]
    fn unpack_data(data: u64) -> (u32, u16, f64) {
        let visits = (data >> 32) as u32;
        let checksum = ((data >> 16) & 0xFFFF) as u16;
        let q_int = (data & 0xFFFF) as u16;
        let q = q_int as f64 / Q_SCALE;
        (visits, checksum, q)
    }

    #[inline(always)]
    fn pack_data(visits: u32, checksum: u16, q: f64) -> u64 {
        let q_int = (q * Q_SCALE).clamp(0.0, 65535.0) as u64;
        let checksum = checksum as u64;
        let visits = visits as u64;
        (visits << 32) | (checksum << 16) | q_int
    }
}

pub struct HashTable {
    table: Vec<HashEntryInternal>,
    mask: u64,
}

impl HashTable {
    pub fn new(size: usize, threads: usize) -> Self {
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
        
        // Probe Slot 0
        let entry0 = unsafe { self.table.get_unchecked(idx) };
        if let Some(e) = self.read_entry(entry0, hash) {
            return Some(e);
        }

        // Probe Slot 1 (XOR pair)
        let entry1 = unsafe { self.table.get_unchecked(idx ^ 1) };
        if let Some(e) = self.read_entry(entry1, hash) {
            return Some(e);
        }

        None
    }

    #[inline(always)]
    fn read_entry(&self, entry: &HashEntryInternal, hash: u64) -> Option<HashEntry> {
        // Double-check locking pattern + Checksum
        let k1 = entry.key.load(Ordering::Relaxed);
        if k1 == hash {
            let data = entry.data.load(Ordering::Relaxed);
            let k2 = entry.key.load(Ordering::Relaxed);
            
            if k2 == hash {
                let (visits, checksum, q) = HashEntryInternal::unpack_data(data);
                // Verify checksum matches the hash's upper bits
                if checksum == HashEntryInternal::checksum(hash) {
                    return Some(HashEntry { q: q as f32, visits });
                }
            }
        }
        None
    }

    #[inline(always)]
    pub fn push(&self, hash: u64, val: f32) {
        let idx = (hash & self.mask) as usize;
        let idx0 = idx;
        let idx1 = idx ^ 1;
        
        let entry0 = unsafe { self.table.get_unchecked(idx0) };
        let entry1 = unsafe { self.table.get_unchecked(idx1) };
        
        let val_f64 = val as f64;
        let cs = HashEntryInternal::checksum(hash);

        // 1. Try Update
        if entry0.key.load(Ordering::Relaxed) == hash {
            self.update_entry(entry0, val_f64, cs);
            return;
        }
        if entry1.key.load(Ordering::Relaxed) == hash {
            self.update_entry(entry1, val_f64, cs);
            return;
        }

        // 2. Insert / Replace
        let d0 = entry0.data.load(Ordering::Relaxed);
        let (v0, _, _) = HashEntryInternal::unpack_data(d0);
        let k0 = entry0.key.load(Ordering::Relaxed);

        if k0 == 0 {
            self.overwrite_entry(entry0, hash, val_f64, cs);
            return;
        }

        let d1 = entry1.data.load(Ordering::Relaxed);
        let (v1, _, _) = HashEntryInternal::unpack_data(d1);
        let k1 = entry1.key.load(Ordering::Relaxed);

        if k1 == 0 {
            self.overwrite_entry(entry1, hash, val_f64, cs);
            return;
        }

        // Replacement Strategy: Depth-Preferred / Visit-Preferred
        // Promote better entry to Slot 0, victimize Slot 1.
        if v1 > v0 {
            // Promote Slot 1 -> Slot 0
            entry0.data.store(d1, Ordering::Relaxed);
            entry0.key.store(k1, Ordering::Relaxed);
            
            // New -> Slot 1
            self.overwrite_entry(entry1, hash, val_f64, cs);
        } else {
            // Slot 0 is good. Overwrite Slot 1.
            self.overwrite_entry(entry1, hash, val_f64, cs);
        }
    }

    #[inline(always)]
    fn update_entry(&self, entry: &HashEntryInternal, val: f64, checksum: u16) {
        let current_data = entry.data.load(Ordering::Relaxed);
        let (visits, _, q) = HashEntryInternal::unpack_data(current_data);
        
        let new_visits = visits.saturating_add(1);
        let new_q = q + (val - q) / (new_visits as f64);
        
        let new_data = HashEntryInternal::pack_data(new_visits, checksum, new_q);
        entry.data.store(new_data, Ordering::Relaxed);
    }

    #[inline(always)]
    fn overwrite_entry(&self, entry: &HashEntryInternal, hash: u64, val: f64, checksum: u16) {
        // Write Data first, then Key.
        // If Reader reads new Data with old Key, Checksum check (based on Key) will fail 99.998%.
        // If Reader reads old Data with new Key, Checksum check (based on Key) will fail 99.998%.
        let new_data = HashEntryInternal::pack_data(1, checksum, val);
        entry.data.store(new_data, Ordering::Relaxed);
        entry.key.store(hash, Ordering::Relaxed);
    }
}
