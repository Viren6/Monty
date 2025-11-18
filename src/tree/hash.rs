use crate::tree::NodePtr;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Default)]
pub struct HashEntry {
    pub hash: u16,
    pub q: u16,
    pub node: NodePtr,
}

impl HashEntry {
    pub fn q(&self) -> f32 {
        f32::from(self.q) / f32::from(u16::MAX)
    }
}

#[derive(Default)]
struct HashEntryInternal {
    // 3 AtomicU64s
    // word0: node (64)
    // word1: hash (64)
    // word2: version(16) | q(48)
    node: AtomicU64,
    hash: AtomicU64,
    data: AtomicU64,
}

impl HashEntryInternal {
    fn pack_data(q: u64, version: u16) -> u64 {
        (u64::from(version) << 48) | (q & 0xFFFF_FFFF_FFFF)
    }

    fn unpack_data(data: u64) -> (u16, u64) {
        let version = (data >> 48) as u16;
        let q = data & 0xFFFF_FFFF_FFFF;
        (version, q)
    }
}

pub struct HashTable {
    table: Vec<HashEntryInternal>,
}

impl HashTable {
    pub fn new(size: usize, threads: usize) -> Self {
        let chunk_size = size.div_ceil(threads);

        let mut table = HashTable { table: Vec::new() };
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
        let chunk_size = self.table.len().div_ceil(threads);

        std::thread::scope(|s| {
            for chunk in self.table.chunks_mut(chunk_size) {
                s.spawn(|| {
                    for entry in chunk.iter_mut() {
                        *entry = HashEntryInternal::default();
                    }
                });
            }
        });
    }

    fn key(hash: u64) -> u16 {
        (hash >> 48) as u16
    }

    pub fn get(&self, hash: u64) -> Option<HashEntry> {
        let idx = hash % (self.table.len() as u64);
        let entry = &self.table[idx as usize];

        // Optimistic read
        let d1 = entry.data.load(Ordering::Acquire);
        let (v1, _) = HashEntryInternal::unpack_data(d1);

        // If version is odd, it's being written
        if v1 & 1 != 0 {
            return None;
        }

        let stored_hash = entry.hash.load(Ordering::Relaxed);
        if stored_hash != hash {
            return None;
        }

        let node_val = entry.node.load(Ordering::Relaxed);
        let d2 = entry.data.load(Ordering::Acquire);
        let (v2, q) = HashEntryInternal::unpack_data(d2);

        if v1 == v2 {
            // Downcast 48-bit Q to 16-bit Q for HashEntry
            // q is q_val * 2^48
            // we want q_val * 2^16 = q / 2^32
            let q16 = (q >> 32) as u16;
            
            Some(HashEntry {
                hash: Self::key(hash),
                q: q16,
                node: NodePtr::from_raw(node_val),
            })
        } else {
            None
        }
    }

    pub fn push(&self, hash: u64, q_val: f32, node: NodePtr) {
        let idx = hash % (self.table.len() as u64);
        let entry = &self.table[idx as usize];

        // Fixed point conversion: q * 2^48
        let q_fp = (f64::from(q_val) * ((1u64 << 48) as f64)) as u64;
        let q_fp = q_fp.min((1u64 << 48) - 1); // clamp to 48 bits

        // Optimistic write
        let d_old = entry.data.load(Ordering::Relaxed);
        let (v_old, q_old) = HashEntryInternal::unpack_data(d_old);

        if v_old & 1 != 0 {
            return;
        }

        let v_new_locked = v_old.wrapping_add(1);
        let d_locked = HashEntryInternal::pack_data(q_old, v_new_locked); // preserve old q temporarily

        if entry
            .data
            .compare_exchange(d_old, d_locked, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return;
        }

        // Write payload
        entry.node.store(node.inner(), Ordering::Relaxed);
        entry.hash.store(hash, Ordering::Relaxed);

        // Unlock
        let v_new = v_new_locked.wrapping_add(1);
        let d_new = HashEntryInternal::pack_data(q_fp, v_new);
        entry.data.store(d_new, Ordering::Release);
    }
}
