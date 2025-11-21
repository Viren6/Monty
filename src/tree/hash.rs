use std::sync::atomic::{AtomicU64, Ordering};

use super::NodePtr;

#[derive(Clone, Copy, Debug, Default)]
pub struct HashEntry {
    hash: u16,
    q: u16,
    gen: u8,
    half: u8,
    node_idx: u32,
}

impl HashEntry {
    pub fn q(&self) -> f32 {
        f32::from(self.q) / f32::from(u16::MAX)
    }

    pub fn node_idx(&self) -> usize {
        self.node_idx as usize
    }

    pub fn gen(&self) -> u8 {
        self.gen
    }

    pub fn half(&self) -> u8 {
        self.half
    }
}

#[derive(Default)]
struct HashEntryInternal(AtomicU64);

impl Clone for HashEntryInternal {
    fn clone(&self) -> Self {
        Self(AtomicU64::new(self.0.load(Ordering::Relaxed)))
    }
}

impl From<&HashEntryInternal> for HashEntry {
    fn from(value: &HashEntryInternal) -> Self {
        let val = value.0.load(Ordering::Relaxed);
        let hash = val as u16;
        let q = (val >> 16) as u16;
        let gen = ((val >> 32) & 0xF) as u8;
        let half = ((val >> 36) & 1) as u8;
        let node_idx = (val >> 37) as u32;
        Self {
            hash,
            q,
            gen,
            half,
            node_idx,
        }
    }
}

impl From<HashEntry> for u64 {
    fn from(value: HashEntry) -> Self {
        let hash = u64::from(value.hash);
        let q = u64::from(value.q) << 16;
        let gen = u64::from(value.gen & 0xF) << 32;
        let half = u64::from(value.half & 1) << 36;
        let node_idx = u64::from(value.node_idx) << 37;
        hash | q | gen | half | node_idx
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

    pub fn fetch(&self, hash: u64) -> HashEntry {
        let idx = hash % (self.table.len() as u64);
        HashEntry::from(&self.table[idx as usize])
    }

    fn key(hash: u64) -> u16 {
        (hash >> 48) as u16
    }

    pub fn get(&self, hash: u64) -> Option<HashEntry> {
        let entry = self.fetch(hash);

        if entry.hash == Self::key(hash) {
            Some(entry)
        } else {
            None
        }
    }

    pub fn push(&self, hash: u64, q: f32, node: NodePtr, gen: u8) {
        let idx = hash % (self.table.len() as u64);
        let node_idx = node.idx();
        let half = if node.half() { 1 } else { 0 };

        // 27 bits = 134,217,727
        let (stored_idx, stored_gen, stored_half) = if node_idx < (1 << 27) {
            (node_idx as u32, gen, half)
        } else {
            (0, 0, 0) // Invalid/Root
        };

        let entry = HashEntry {
            hash: Self::key(hash),
            q: (q * f32::from(u16::MAX)) as u16,
            gen: stored_gen,
            half: stored_half,
            node_idx: stored_idx,
        };

        self.table[idx as usize]
            .0
            .store(u64::from(entry), Ordering::Relaxed)
    }
}
