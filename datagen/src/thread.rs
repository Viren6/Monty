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
        let mut temp = 0.8f32;

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

            let mut probs = Vec::with_capacity(moves.len());
            let mut max = f32::NEG_INFINITY;

            for mov in moves {
                let p = policy.get(&position.board(), &mov, &hl);
                max = max.max(p);
                probs.push((mov, p));
            }

            let pst = 1.0f32;
            let mut total = 0.0f32;
            for (_, p) in probs.iter_mut() {
                *p = ((*p - max) / pst).exp();
                total += *p;
            }

            if total <= 0.0 {
                total = 1.0;
            }

            let mut dist = Vec::with_capacity(probs.len());
            for (mov, p) in probs.iter_mut() {
                *p /= total;
                let mf_move = montyformat::chess::Move::from(u16::from(*mov));
                let visits = (*p * 65535.0) as u32;
                dist.push((mf_move, visits));
            }

            let best_move: Move = if temp == 0.0 {
                probs
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(m, _)| *m)
                    .unwrap()
            } else {
                let t = 1.0 / f64::from(temp);
                let mut weights = Vec::with_capacity(probs.len());
                let mut total = 0.0f64;
                for (_, p) in &probs {
                    let w = f64::from(*p).powf(t);
                    weights.push(w);
                    total += w;
                }
                let rand = f64::from(self.rng.rand_int()) / f64::from(u32::MAX);
                let mut cumulative = 0.0f64;
                let mut chosen = probs[probs.len() - 1].0;
                for (i, weight) in weights.iter().enumerate() {
                    cumulative += *weight;
                    if cumulative / total > rand {
                        chosen = probs[i].0;
                        break;
                    }
                }
                chosen
            };

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

            temp *= 0.9;
            if temp <= 0.2 {
                temp = 0.0;
            }

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
