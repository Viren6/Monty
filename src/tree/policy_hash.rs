use std::sync::atomic::{AtomicU64, Ordering};

use super::NodePtr;

#[derive(Clone, Copy, Debug, Default)]
pub struct PolicyEntry {
    ptr: u32,
    temp: u16,
    hash: u32,
}

impl PolicyEntry {
    pub fn ptr(&self) -> NodePtr {
        NodePtr::from_raw(self.ptr)
    }

    pub fn temp(&self) -> u16 {
        self.temp
    }
}

#[derive(Default)]
struct PolicyEntryInternal(AtomicU64);

impl Clone for PolicyEntryInternal {
    fn clone(&self) -> Self {
        Self(AtomicU64::new(self.0.load(Ordering::Relaxed)))
    }
}

impl From<&PolicyEntryInternal> for PolicyEntry {
    fn from(value: &PolicyEntryInternal) -> Self {
        let v = value.0.load(Ordering::Relaxed);
        PolicyEntry {
            ptr: v as u32,
            temp: ((v >> 32) & 0xFFFF) as u16,
            hash: (v >> 32) as u32,
        }
    }
}

impl From<PolicyEntry> for u64 {
    fn from(value: PolicyEntry) -> Self {
        u64::from(value.ptr) | (u64::from(value.temp) << 32) | (u64::from(value.hash) << 48)
    }
}

pub struct PolicyTable {
    table: Vec<PolicyEntryInternal>,
}

impl PolicyTable {
    pub fn new(size: usize, threads: usize) -> Self {
        let chunk_size = size.div_ceil(threads);
        let mut table = PolicyTable { table: Vec::new() };
        table.table.reserve_exact(size);
        unsafe {
            use std::mem::{size_of, MaybeUninit};
            let ptr = table.table.as_mut_ptr().cast();
            let uninit: &mut [MaybeUninit<u8>] =
                std::slice::from_raw_parts_mut(ptr, size * size_of::<PolicyEntryInternal>());
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
                        *entry = PolicyEntryInternal::default();
                    }
                });
            }
        });
    }

    pub fn fetch(&self, hash: u64) -> PolicyEntry {
        let idx = hash % (self.table.len() as u64);
        PolicyEntry::from(&self.table[idx as usize])
    }

    fn key(hash: u64) -> u32 {
        (hash >> 32) as u32
    }

    pub fn get(&self, hash: u64) -> Option<PolicyEntry> {
        let entry = self.fetch(hash);
        if entry.hash == Self::key(hash) {
            Some(entry)
        } else {
            None
        }
    }

    pub fn push(&self, hash: u64, ptr: NodePtr, temp: u16) {
        let idx = hash % (self.table.len() as u64);
        let entry = PolicyEntry {
            ptr: ptr.inner(),
            temp,
            hash: Self::key(hash),
        };
        self.table[idx as usize]
            .0
            .store(u64::from(entry), Ordering::Relaxed);
    }
}

pub fn temp_to_u16(temp: f32) -> u16 {
    (temp.clamp(0.0, 2.0) / 2.0 * f32::from(u16::MAX)) as u16
}

pub fn u16_to_temp(t: u16) -> f32 {
    f32::from(t) / f32::from(u16::MAX) * 2.0
}