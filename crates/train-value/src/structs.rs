use bullet_lib::game::formats::bulletformat::{BulletFormat, ChessBoard};
use std::str::FromStr;

#[derive(Clone, Copy, Default)]
pub struct WdlPosition {
    pub board: ChessBoard,
    pub wdl: [f32; 3],
}

impl WdlPosition {
    pub fn new(board: ChessBoard, wdl: [f32; 3]) -> Self {
        Self { board, wdl }
    }
}

pub struct WdlIterator {
    board_iter: <ChessBoard as IntoIterator>::IntoIter,
}

impl Iterator for WdlIterator {
    type Item = (u8, usize);

    fn next(&mut self) -> Option<Self::Item> {
        self.board_iter.next().map(|(pc, sq)| (pc, usize::from(sq)))
    }
}

impl IntoIterator for WdlPosition {
    type Item = (u8, usize);
    type IntoIter = WdlIterator;

    fn into_iter(self) -> Self::IntoIter {
        WdlIterator {
            board_iter: self.board.into_iter(),
        }
    }
}

impl BulletFormat for WdlPosition {
    type FeatureType = (u8, usize);
    const HEADER_SIZE: usize = ChessBoard::HEADER_SIZE;

    fn score(&self) -> i16 {
        self.board.score()
    }

    fn result(&self) -> f32 {
        self.wdl[0] + 0.5 * self.wdl[1]
    }

    fn result_idx(&self) -> usize {
        0
    }

    fn set_result(&mut self, _result: f32) {
        panic!("Cannot set result on WdlPosition");
    }
}

impl FromStr for WdlPosition {
    type Err = <ChessBoard as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let board = ChessBoard::from_str(s)?;
        Ok(Self {
            board,
            wdl: [0.33, 0.33, 0.33], // Default placeholder, will be overridden by loader or eval
        })
    }
}
