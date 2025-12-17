use std::sync::atomic::{AtomicI16, Ordering};

const TABLE_SIZE: usize = 1 << 14;

#[derive(Default)]
pub struct ValueHistory {
    table: Vec<[AtomicI16; 2]>,
}

impl ValueHistory {
    pub fn new() -> Self {
        let mut table = Vec::with_capacity(TABLE_SIZE);
        table.extend((0..TABLE_SIZE).map(|_| [AtomicI16::new(0), AtomicI16::new(0)]));
        Self { table }
    }

    pub fn clear(&self) {
        for entry in &self.table {
            entry[0].store(0, Ordering::Relaxed);
            entry[1].store(0, Ordering::Relaxed);
        }
    }

    #[must_use]
    pub fn correct_cp(&self, pawn_hash: u64, stm: usize, score_cp: i32) -> i32 {
        let entry = self.entry(pawn_hash, stm);
        score_cp + i32::from(entry.load(Ordering::Relaxed)) / 16
    }

    pub fn update(&self, pawn_hash: u64, stm: usize, q: f32, score: f32, visits: u16) {
        let q = q.clamp(0.001, 0.999);
        let score = score.clamp(0.001, 0.999);

        let q_cp = score_to_cp(q);
        let score_cp = score_to_cp(score);

        const MIN_DIV: i32 = 1;
        const MAX_DIV: i32 = 16;
        const MAX_VISITS: u16 = 8192;

        let divisor = MIN_DIV
            + (MAX_DIV - MIN_DIV) * i32::from(visits.min(MAX_VISITS)) / i32::from(MAX_VISITS);

        let bonus = ((q_cp - score_cp) / divisor).clamp(-256, 256);

        if bonus == 0 {
            return;
        }

        let cell = self.entry(pawn_hash, stm);

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

    fn entry(&self, pawn_hash: u64, stm: usize) -> &AtomicI16 {
        &self.table[index(pawn_hash)][stm]
    }
}

fn score_to_cp(score: f32) -> i32 {
    const K: f32 = 400.0;
    let score = score.clamp(0.001, 0.999);
    (-K * ((1.0 / score) - 1.0).ln()).round() as i32
}

fn scale_bonus(score: i16, bonus: i32) -> i16 {
    let adjusted = bonus - i32::from(score) * bonus.abs() / 1024;
    adjusted.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

fn index(pawn_hash: u64) -> usize {
    (pawn_hash as usize) & (TABLE_SIZE - 1)
}
