use std::sync::atomic::{AtomicI16, Ordering};

const TABLE_SIZE: usize = 16384;
const EVAL_SCALE: f32 = 400.0;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

fn score_to_cp(score: f32) -> i32 {
    (EVAL_SCALE * inverse_sigmoid(score)).round() as i32
}

fn cp_to_score(cp: i32) -> f32 {
    sigmoid(cp as f32 / EVAL_SCALE)
}

fn scale_bonus(score: i16, bonus: i32) -> i16 {
    let bonus = bonus.clamp(i16::MIN as i32, i16::MAX as i32);
    let adjusted = bonus - i32::from(score) * bonus.abs() / 1024;
    adjusted.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

pub struct ValueHistory {
    table: Vec<AtomicI16>,
}

impl ValueHistory {
    pub fn new() -> Self {
        let mut table = Vec::with_capacity(TABLE_SIZE * 2);
        table.extend((0..TABLE_SIZE * 2).map(|_| AtomicI16::new(0)));
        Self { table }
    }

    pub fn clear(&self) {
        for entry in &self.table {
            entry.store(0, Ordering::Relaxed);
        }
    }

    fn idx(&self, pawn_hash: u64, stm: usize) -> usize {
        ((pawn_hash as usize) & (TABLE_SIZE - 1)) * 2 + stm
    }

    pub fn correct_score(&self, pawn_hash: u64, stm: usize, score: f32) -> f32 {
        let score = score.clamp(0.001, 0.999);
        let cp = score_to_cp(score);
        let hist = self.table[self.idx(pawn_hash, stm)].load(Ordering::Relaxed) as i32;
        let corrected_cp = cp + hist / 16;
        cp_to_score(corrected_cp)
    }

    pub fn update(&self, pawn_hash: u64, stm: usize, q: f32, score: f32, num_visits: u64) {
        let score = score.clamp(0.001, 0.999);
        let q = q.clamp(0.001, 0.999);

        if !score.is_finite() || !q.is_finite() {
            return;
        }

        let q_cp = score_to_cp(q);
        let score_cp = score_to_cp(score);

        const MIN_DIV: i32 = 1;
        const MAX_DIV: i32 = 16;
        const MAX_VISITS: u64 = 8192;

        let divisor =
            MIN_DIV + (MAX_DIV - MIN_DIV) * num_visits.min(MAX_VISITS) as i32 / MAX_VISITS as i32;
        let bonus = ((q_cp - score_cp) / divisor).clamp(-256, 256);

        let cell = &self.table[self.idx(pawn_hash, stm)];
        let mut current = cell.load(Ordering::Relaxed);

        loop {
            let delta = scale_bonus(current, bonus);
            let new = current.saturating_add(delta);

            match cell.compare_exchange(current, new, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}
