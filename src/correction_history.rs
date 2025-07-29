use std::sync::atomic::{AtomicI32, Ordering};

use crate::chess::Board;

/// Parameters for correction history.
const CORRHIST_SIZE: usize = 1 << 16; // 65,536 entries
const CORRHIST_WEIGHT_SCALE: i32 = 256;
const CORRHIST_Q_SCALE: i32 = 1 << 30; // quantisation for q values

pub struct CorrectionHistory {
    table: Vec<AtomicI32>,
}

impl Default for CorrectionHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl CorrectionHistory {
    /// Create a new correction history table
    pub fn new() -> Self {
        Self {
            table: (0..CORRHIST_SIZE).map(|_| AtomicI32::new(0)).collect(),
        }
    }

    #[inline]
    fn index(&self, board: &Board) -> usize {
        board.hash() as usize % CORRHIST_SIZE
    }

    /// Return the current q correction for the board
    pub fn get(&self, board: &Board) -> f32 {
        self.table[self.index(board)].load(Ordering::Relaxed) as f32 / CORRHIST_Q_SCALE as f32
    }

    /// Adjust a q evaluation with correction history
    pub fn apply(&self, board: &Board, q: f32) -> f32 {
        q + self.get(board)
    }

    /// Update correction history using visits and evaluation difference
    pub fn update(&self, board: &Board, diff: f32, diff_visits: i32) {
        let idx = self.index(board);
        let entry = self.table[idx].load(Ordering::Relaxed);
        let scaled_diff = (diff * CORRHIST_Q_SCALE as f32) as i32;
        let new_weight = diff_visits.min(CORRHIST_WEIGHT_SCALE);
        let value = (entry * (CORRHIST_WEIGHT_SCALE - new_weight) + scaled_diff * new_weight)
            / CORRHIST_WEIGHT_SCALE;
        self.table[idx].store(value, Ordering::Relaxed);
    }
}
