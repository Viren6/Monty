use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Default)]
pub struct HashEntry {
    hash: u16,
    q: u16,
    visits: u32,
}

impl HashEntry {
    pub fn q(&self) -> f32 {
        f32::from(self.q) / f32::from(u16::MAX)
    }

    pub fn visits(&self) -> u32 {
        self.visits
    }

    fn from_raw(raw: u64) -> Self {
        HashEntry {
            hash: (raw >> 48) as u16,
            q: ((raw >> 32) & 0xFFFF) as u16,
            visits: raw as u32,
        }
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
        let raw = value.0.load(Ordering::Relaxed);
        HashEntry::from_raw(raw)
    }
}

impl From<HashEntry> for u64 {
    fn from(value: HashEntry) -> Self {
        (u64::from(value.hash) << 48) | (u64::from(value.q) << 32) | u64::from(value.visits)
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

        if entry.visits > 0 && entry.hash == Self::key(hash) {
            Some(entry)
        } else {
            None
        }
    }

    pub fn push(&self, hash: u64, q: f32) {
        let idx = hash % (self.table.len() as u64);

        let key = Self::key(hash);
        let q = (q.clamp(0.0, 1.0) * f32::from(u16::MAX)) as u16;
        let cell = &self.table[idx as usize].0;

        let mut current = cell.load(Ordering::Relaxed);

        loop {
            let mut entry = HashEntry::from_raw(current);

            if entry.visits == 0 || entry.hash != key {
                entry.hash = key;
                entry.q = q;
                entry.visits = 1;
            } else {
                let visits = entry.visits.saturating_add(1);
                let total = f64::from(entry.q) * f64::from(entry.visits) + f64::from(q);
                entry.q = (total / f64::from(visits))
                    .round()
                    .clamp(0.0, f64::from(u16::MAX)) as u16;
                entry.visits = visits;
            }

            match cell.compare_exchange(
                current,
                u64::from(entry),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}
