use std::sync::atomic::{AtomicI16, Ordering};

pub const VALUE_EVAL_SCALE: f32 = 400.0;

fn prob_to_cp(score: f32) -> i32 {
    (-VALUE_EVAL_SCALE * ((1.0 / score) - 1.0).ln()).round() as i32
}

fn cp_to_prob(cp: i32) -> f32 {
    1.0 / (1.0 + (-(cp as f32) / VALUE_EVAL_SCALE).exp())
}

fn scale_bonus(score: i16, bonus: i32) -> i16 {
    let bonus = bonus.clamp(i16::MIN as i32, i16::MAX as i32);
    let reduction = i32::from(score) * bonus.abs() / 1024;
    let adjusted = bonus - reduction;
    adjusted.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

pub struct ValueHistory {
    table: Vec<[AtomicI16; 2]>,
}

impl ValueHistory {
    pub fn new() -> Self {
        let mut table = Vec::with_capacity(16384);
        table.extend((0..16384).map(|_| [AtomicI16::new(0), AtomicI16::new(0)]));

        Self { table }
    }

    fn entry(&self, pawn_hash: u64, side: usize) -> &AtomicI16 {
        &self.table[pawn_hash as usize & 16383][side]
    }

    pub fn clear(&self) {
        for bucket in &self.table {
            for entry in bucket {
                entry.store(0, Ordering::Relaxed);
            }
        }
    }

    pub fn correct_cp(&self, pawn_hash: u64, side: usize, score_cp: i32) -> i32 {
        let entry = self.entry(pawn_hash, side).load(Ordering::Relaxed);
        score_cp + i32::from(entry) / 16
    }

    pub fn update(&self, pawn_hash: u64, side: usize, q: f32, score: f32, num_visits: u16) {
        let q = q.clamp(0.001, 0.999);
        let score = score.clamp(0.001, 0.999);

        let q_cp = prob_to_cp(q);
        let score_cp = prob_to_cp(score);

        const MIN_DIV: i32 = 1;
        const MAX_DIV: i32 = 8;
        const MAX_VISITS: u16 = 1000;

        let divisor = MIN_DIV
            + (MAX_DIV - MIN_DIV) * i32::from(num_visits.min(MAX_VISITS)) / i32::from(MAX_VISITS);

        let bonus = ((q_cp - score_cp) / divisor).clamp(-256, 256);
        let cell = self.entry(pawn_hash, side);

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

    pub fn score_to_cp(score: f32) -> i32 {
        prob_to_cp(score)
    }

    pub fn cp_to_score(cp: i32) -> f32 {
        cp_to_prob(cp)
    }
}
