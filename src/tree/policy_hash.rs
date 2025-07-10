use std::sync::Mutex;
use crate::chess::Move;

#[derive(Clone)]
pub struct PolicyEntry {
    hash: u64,
    data: Vec<(u16, f32)>,
}

pub struct PolicyHashTable {
    table: Vec<Mutex<Option<PolicyEntry>>>,
}

impl PolicyHashTable {
    pub fn new(size: usize, _threads: usize) -> Self {
        let mut table = Vec::with_capacity(size);
        for _ in 0..size {
            table.push(Mutex::new(None));
        }
        Self { table }
    }

    pub fn clear(&mut self, _threads: usize) {
        for entry in &mut self.table {
            *entry.lock().unwrap() = None;
        }
    }

    pub fn get(&self, hash: u64) -> Option<Vec<(Move, f32)>> {
        let idx = hash % self.table.len() as u64;
        let entry = self.table[idx as usize].lock().unwrap();
        entry.as_ref().and_then(|e| {
            if e.hash == hash {
                Some(e.data.iter().map(|&(m, v)| (Move::from(m), v)).collect())
            } else {
                None
            }
        })
    }

    pub fn push(&self, hash: u64, data: Vec<(Move, f32)>) {
        let idx = hash % self.table.len() as u64;
        let mut entry = self.table[idx as usize].lock().unwrap();
        *entry = Some(PolicyEntry {
            hash,
            data: data.into_iter().map(|(m, v)| (u16::from(m), v)).collect(),
        });
    }
}