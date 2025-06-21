use std::collections::HashMap;
use std::sync::RwLock;

use crate::chess::Move;

#[derive(Clone)]
pub struct DagEntry {
    pub moves: Vec<(Move, f32)>,
}

pub struct Dag {
    map: RwLock<HashMap<u64, DagEntry>>,
}

impl Dag {
    pub fn new() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, hash: u64) -> Option<DagEntry> {
        self.map.read().unwrap().get(&hash).cloned()
    }

    pub fn insert(&self, hash: u64, entry: DagEntry) {
        self.map.write().unwrap().insert(hash, entry);
    }

    pub fn clear(&self) {
        self.map.write().unwrap().clear();
    }
}