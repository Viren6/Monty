use bullet_lib::game::inputs::SparseInputType;
use crate::structs::WdlPosition;

use monty::networks::value::threats::{map_features, TOTAL};

#[derive(Clone, Copy, Default)]
pub struct ThreatInputs;
impl SparseInputType for ThreatInputs {
    type RequiredDataType = WdlPosition;

    fn num_inputs(&self) -> usize {
        TOTAL
    }

    fn max_active(&self) -> usize {
        128
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &WdlPosition, mut f: F) {
        let mut bbs = [0; 8];
        for (pc, sq) in pos.board.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        map_features(bbs, 0, |stm| f(stm, stm));
    }

    fn shorthand(&self) -> String {
        format!("{TOTAL}")
    }

    fn description(&self) -> String {
        "Threat inputs".to_string()
    }
}
