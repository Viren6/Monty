use std::sync::atomic::{AtomicI16, Ordering};

use crate::chess::Board;

use super::threats;

const MAX_CORRECTION: i16 = 400;
const DECAY: i16 = 32; // smoothing factor

static TABLE: [AtomicI16; threats::TOTAL] = [const { AtomicI16::new(0) }; threats::TOTAL];

/// Return the total correction for a board by summing the corrections for all
/// threat features present in the position.
pub fn correction(board: &Board) -> i32 {
    let mut total = 0i32;
    threats::map_features(board, |feat| {
        total += i32::from(TABLE[feat].load(Ordering::Relaxed));
    });
    total
}

/// Update the correction history for all features in the given board with the
/// provided difference between searched and evaluated score (in centipawns).
pub fn update(board: &Board, diff: i32) {
    threats::map_features(board, |feat| {
        let entry = &TABLE[feat];
        let cur = entry.load(Ordering::Relaxed) as i32;
        // Exponential moving average towards `diff`.
        let mut new = cur + ((diff - cur) / i32::from(DECAY));
        new = new.clamp(-(MAX_CORRECTION as i32), MAX_CORRECTION as i32);
        entry.store(new as i16, Ordering::Relaxed);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::ChessState;

    // Basic smoke test to ensure correction functions work and use threat mapping.
    #[test]
    fn test_correction_update() {
        let pos = ChessState::default();
        // ensure zero correction initially
        assert_eq!(correction(&pos.board()), 0);
        update(&pos.board(), 100);
        let c = correction(&pos.board());
        assert!(c != 0);
    }
}