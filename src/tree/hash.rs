use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Default)]
pub struct HashEntry {
    hash: u16,
    ptr: u32,
}

impl HashEntry {
    pub fn ptr(&self) -> super::NodePtr {
        super::NodePtr::from_raw(self.ptr)
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
        unsafe { std::mem::transmute(value.0.load(Ordering::Relaxed)) }
    }
}

impl From<HashEntry> for u64 {
    fn from(value: HashEntry) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

pub struct HashTable {
    table: Vec<HashEntryInternal>,
}

impl HashTable {
    pub fn new(size: usize, threads: usize) -> Self {
        let size = size.max(1);
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

    pub fn push(&self, hash: u64, ptr: super::NodePtr) {
        if self.table.is_empty() {
            return;
        }
        let idx = hash % (self.table.len() as u64);

        let entry = HashEntry {
            hash: Self::key(hash),
            ptr: ptr.inner(),
        };

        self.table[idx as usize]
            .0
            .store(u64::from(entry), Ordering::Relaxed)
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }
}
