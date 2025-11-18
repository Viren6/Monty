use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct HashEntry {
    hash: u16,
    visits: u16,
    q_bits: u32,
}

impl HashEntry {
    pub fn q(&self) -> f32 {
        f32::from_bits(self.q_bits)
    }

    pub fn visits(&self) -> u16 {
        self.visits
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

        if entry.hash == Self::key(hash) && entry.visits() > 0 {
            Some(entry)
        } else {
            None
        }
    }

    pub fn push(&self, hash: u64, q: f32) {
        let idx = hash % (self.table.len() as u64);
        let key = Self::key(hash);
        let cell = &self.table[idx as usize].0;

        let mut current = cell.load(Ordering::Relaxed);
        loop {
            let mut entry: HashEntry = unsafe { std::mem::transmute(current) };

            if entry.hash != key {
                entry = HashEntry {
                    hash: key,
                    visits: 1,
                    q_bits: q.clamp(0.0, 1.0).to_bits(),
                };
            } else {
                let visits = entry.visits.saturating_add(1).min(u16::MAX);
                let blended = {
                    let stored = f32::from_bits(entry.q_bits);
                    let total = f32::from(entry.visits);
                    let mut value = q;
                    if !value.is_finite() {
                        value = 0.5;
                    }
                    let numerator = stored * total + value;
                    (numerator / (total + 1.0)).clamp(0.0, 1.0)
                };

                entry = HashEntry {
                    hash: key,
                    visits,
                    q_bits: blended.to_bits(),
                };
            }

            let new_value: u64 = entry.into();
            match cell.compare_exchange(current, new_value, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}
