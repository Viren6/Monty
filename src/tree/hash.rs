use std::sync::atomic::{AtomicU64, Ordering};

use super::NodePtr;

#[derive(Clone, Copy, Debug, Default)]
#[repr(align(16))]
pub struct HashEntry {
    pub hash: u16,
    pub q: u64,
    pub gen: u8,
    pub half: u8,
    pub node_idx: u32,
}

impl HashEntry {
    pub fn q(&self) -> f32 {
        // Use f64 for intermediate calculation to maintain precision before casting back to f32
        (self.q as f64 / u64::MAX as f64) as f32
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
#[repr(align(16))]
struct HashEntryInternal {
    key: AtomicU64,
    data: AtomicU64,
}

impl HashEntryInternal {
    fn read(&self) -> (u64, u64) {
        let k1 = self.key.load(Ordering::Relaxed);
        let d = self.data.load(Ordering::Relaxed);
        let k2 = self.key.load(Ordering::Relaxed);
        if k1 == k2 {
            (k1, d)
        } else {
            (0, 0) // Torn read detected, return invalid
        }
    }

    fn write(&self, key: u64, data: u64) {
        // Invalidate key first to prevent torn reads during update
        self.key.store(0, Ordering::Relaxed);
        self.data.store(data, Ordering::Relaxed);
        self.key.store(key, Ordering::Relaxed);
    }
}

impl From<&HashEntryInternal> for HashEntry {
    fn from(value: &HashEntryInternal) -> Self {
        let (key, data) = value.read();
        let q = data;
        let hash = key as u16;
        let gen = (key >> 16) as u8;
        let half = (key >> 24) as u8;
        let node_idx = (key >> 32) as u32;
        Self {
            hash,
            q,
            gen,
            half,
            node_idx,
        }
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
                        entry.key.store(0, Ordering::Relaxed);
                        entry.data.store(0, Ordering::Relaxed);
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

        // u32::MAX
        let (stored_idx, stored_gen, stored_half) = if node_idx <= u32::MAX as usize {
            (node_idx as u32, gen, half)
        } else {
            (0, 0, 0) // Invalid/Root
        };

        let key = (stored_idx as u64) << 32
            | (stored_half as u64) << 24
            | (stored_gen as u64) << 16
            | Self::key(hash) as u64;
        
        let data = (q as f64 * u64::MAX as f64) as u64;

        self.table[idx as usize].write(key, data);
    }
}
