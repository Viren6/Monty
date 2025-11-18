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
    // data: version(16) | q(16) | hash_high(32)
    data: AtomicU64,
    node: AtomicU64,
}

impl HashEntryInternal {
    // Bit layout for data:
    // 0-15: Version (16 bits)
    // 16-31: Q (16 bits)
    // 32-63: Hash High (32 bits)

    fn pack_data(hash_high: u32, q: u16, version: u16) -> u64 {
        u64::from(version) | (u64::from(q) << 16) | (u64::from(hash_high) << 32)
    }

    fn unpack_data(data: u64) -> (u32, u16, u16) {
        let version = (data & 0xFFFF) as u16;
        let q = ((data >> 16) & 0xFFFF) as u16;
        let hash_high = ((data >> 32) & 0xFFFF_FFFF) as u32;
        (hash_high, q, version)
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

    fn key_high(hash: u64) -> u32 {
        (hash >> 32) as u32
    }

    fn key_u16(hash: u64) -> u16 {
        (hash >> 48) as u16
    }

    pub fn get(&self, hash: u64) -> Option<HashEntry> {
        let idx = hash % (self.table.len() as u64);
        let entry = &self.table[idx as usize];
        let key = Self::key_high(hash);

        // Optimistic read (Seqlock-like)
        let d1 = entry.data.load(Ordering::Acquire);
        let (h1, _, v1) = HashEntryInternal::unpack_data(d1);

        // If hash doesn't match, no need to read further
        if h1 != key {
            return None;
        }

        // If version is odd, it's being written to.
        if v1 & 1 != 0 {
            return None;
        }

        let node_val = entry.node.load(Ordering::Relaxed);
        let d2 = entry.data.load(Ordering::Acquire);
        let (_, _, v2) = HashEntryInternal::unpack_data(d2);

        if v1 == v2 {
            let (_, q, _) = HashEntryInternal::unpack_data(d1);
            Some(HashEntry {
                hash: Self::key_u16(hash),
                q,
                node: NodePtr::from_raw(node_val),
            })
        } else {
            None
        }
    }

    pub fn push(&self, hash: u64, q_val: f32, node: NodePtr) {
        let idx = hash % (self.table.len() as u64);
        let entry = &self.table[idx as usize];

        let key = Self::key_high(hash);
        let q = (q_val * f32::from(u16::MAX)) as u16;

        // Optimistic write (Try-lock)
        let d_old = entry.data.load(Ordering::Relaxed);
        let (h_old, _, v_old) = HashEntryInternal::unpack_data(d_old);

        // If locked (odd), abort
        if v_old & 1 != 0 {
            return;
        }

        // Try to lock: increment version to odd
        let v_new_locked = v_old.wrapping_add(1);
        let d_locked = HashEntryInternal::pack_data(h_old, 0, v_new_locked);

        // Compare exchange to claim
        if entry
            .data
            .compare_exchange(d_old, d_locked, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return; // Failed to lock
        }

        // We are now the writer.
        entry.node.store(node.inner(), Ordering::Relaxed);

        // Unlock: increment version to even, set new data
        let v_new = v_new_locked.wrapping_add(1);
        let d_new = HashEntryInternal::pack_data(key, q, v_new);
        entry.data.store(d_new, Ordering::Release);
    }
}
