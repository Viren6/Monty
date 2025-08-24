use crate::{Destination, Rand};

use monty::{
    chess::{ChessState, GameState, Move},
    networks::{PolicyNetwork, ValueNetwork},
};
use montyformat::{MontyFormat, MontyValueFormat, SearchData};

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

pub struct DatagenThread<'a> {
    rng: Rand,
    dest: Arc<Mutex<Destination>>,
    stop: &'a AtomicBool,
    book: Option<Vec<&'a str>>,
}

impl<'a> DatagenThread<'a> {
    pub fn new(
        stop: &'a AtomicBool,
        book: Option<Vec<&'a str>>,
        dest: Arc<Mutex<Destination>>,
    ) -> Self {
        Self {
            rng: Rand::with_seed(),
            dest,
            stop,
            book,
        }
    }

    pub fn run(&mut self, output_policy: bool, policy: &PolicyNetwork, value: &ValueNetwork) {
        loop {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }

            self.run_game(policy, value, output_policy);
        }
    }

    fn run_game(&mut self, policy: &PolicyNetwork, value: &ValueNetwork, output_policy: bool) {
        let mut position = if let Some(book) = &self.book {
            let idx = self.rng.rand_int() as usize % book.len();
            ChessState::from_fen(book[idx])
        } else {
            ChessState::from_fen(ChessState::STARTPOS)
        };

        let mut result = 0.5;

        let pos = position.board();

        let montyformat_position = montyformat::chess::Position::from_raw(
            pos.bbs(),
            pos.stm() > 0,
            pos.enp_sq(),
            pos.rights(),
            pos.halfm(),
            pos.fullm(),
        );

        let montyformat_castling = montyformat::chess::Castling::from_raw(
            &montyformat_position,
            position.castling().rook_files(),
        );

        let mut value_game = MontyValueFormat {
            startpos: montyformat_position,
            castling: montyformat_castling,
            result: 0.5,
            moves: Vec::new(),
        };

        let mut policy_game = MontyFormat::new(montyformat_position, montyformat_castling);

        let mut total_iters = 0usize;
        let mut searches = 0usize;

        // play out game using raw network outputs
        loop {
            if self.stop.load(Ordering::Relaxed) {
                return;
            }

            let mut moves = Vec::new();
            position.map_legal_moves(|mov| moves.push(mov));
            if moves.is_empty() {
                break;
            }

            let hl = policy.hl(&position.board());

            let mut dist = Vec::with_capacity(moves.len());
            let mut best_move: Move = moves[0];
            let mut best_score = f32::NEG_INFINITY;

            for mov in moves {
                let p = policy.get(&position.board(), &mov, &hl);
                let mf_move = montyformat::chess::Move::from(u16::from(mov));
                let visits = (p * 65535.0) as u32;
                dist.push((mf_move, visits));
                if p > best_score {
                    best_score = p;
                    best_move = mov;
                }
            }

            let (win, draw, _) = value.eval(&position.board());
            let score = win + draw / 2.0;
            let mf_best_move = montyformat::chess::Move::from(u16::from(best_move));

            if output_policy {
                let search_data = SearchData::new(mf_best_move, score, Some(dist));
                policy_game.push(search_data);
            } else {
                value_game.push(position.stm(), mf_best_move, score);
            }

            searches += 1;
            total_iters += 1;

            position.make_move(best_move);

            let game_state = position.game_state();
            match game_state {
                GameState::Ongoing => {}
                GameState::Draw => break,
                GameState::Lost(_) => {
                    if position.stm() == 1 {
                        result = 1.0;
                    } else {
                        result = 0.0;
                    }
                    break;
                }
                GameState::Won(_) => {
                    if position.stm() == 1 {
                        result = 0.0;
                    } else {
                        result = 1.0;
                    }
                    break;
                }
            }
        }

        if output_policy {
            policy_game.result = result;
        } else {
            value_game.result = result;
        }

        if self.stop.load(Ordering::Relaxed) {
            return;
        }

        let mut dest = self.dest.lock().unwrap();

        if output_policy {
            dest.push_policy(&policy_game, self.stop, searches, total_iters);
        } else {
            dest.push(&value_game, self.stop, searches, total_iters);
        }
    }
}
