use std::sync::atomic::{AtomicI16, AtomicU64, Ordering};

use crate::chess::{cp_from_score, ChessState};

const BUCKETS: usize = 16384;

pub struct ValueHistory {
    table: Vec<AtomicI16>,
    corrections: AtomicU64,
    updates: AtomicU64,
}

impl ValueHistory {
    pub fn new() -> Self {
        let mut table = Vec::with_capacity(BUCKETS * 2);
        table.extend((0..BUCKETS * 2).map(|_| AtomicI16::new(0)));

        Self {
            table,
            corrections: AtomicU64::new(0),
            updates: AtomicU64::new(0),
        }
    }

    pub fn clear(&self) {
        for entry in &self.table {
            entry.store(0, Ordering::Relaxed);
        }
        self.corrections.store(0, Ordering::Relaxed);
        self.updates.store(0, Ordering::Relaxed);
    }

    pub fn correct_cp(&self, pos: &ChessState, cp: i32) -> i32 {
        let idx = Self::index(pos);
        let adj = self.table[idx].load(Ordering::Relaxed) as i32;
        let corrected = cp + adj / 16;
        corrected
    }

    pub fn update(&self, pos: &ChessState, predicted: f32, actual: f32, visits: u16) {
        let visits = visits.max(1);
        let predicted = predicted.clamp(1e-6, 1.0 - 1e-6);
        let actual = actual.clamp(1e-6, 1.0 - 1e-6);

        let predicted_cp = cp_from_score(predicted);
        let actual_cp = cp_from_score(actual);

        const MIN_DIV: i32 = 1;
        const MAX_DIV: i32 = 16;
        const MAX_VISITS: u16 = 8192;

        let divisor = MIN_DIV
            + (MAX_DIV - MIN_DIV) * i32::from(visits.min(MAX_VISITS)) / i32::from(MAX_VISITS);

        let mut bonus = (predicted_cp - actual_cp) / divisor;
        bonus = bonus.clamp(-256, 256);

        let idx = Self::index(pos);
        let cell = &self.table[idx];
        let mut current = cell.load(Ordering::Relaxed);

        while bonus != 0 {
            let delta = scale_bonus(current, bonus);
            let new = current.saturating_add(delta);

            match cell.compare_exchange(current, new, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }

        if bonus != 0 {
            self.updates.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn index(pos: &ChessState) -> usize {
        let bucket = (pos.pawn_key() as usize) & (BUCKETS - 1);
        bucket * 2 + pos.stm()
    }
}

fn scale_bonus(score: i16, bonus: i32) -> i16 {
    let bonus = bonus.clamp(i16::MIN as i32, i16::MAX as i32);
    let reduction = i32::from(score) * bonus.abs() / 1024;
    let adjusted = bonus - reduction;
    adjusted.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}
