use crate::{
    mcts::MctsParams,
    networks::{Accumulator, PolicyNetwork, ValueNetwork, POLICY_L1},
};

pub use montyformat::chess::{Attacks, Castling, GameState, Move, Position};

#[derive(Clone, Copy, Debug, Default)]
pub struct EvalWdl {
    pub win: f32,
    pub draw: f32,
    pub loss: f32,
}

impl EvalWdl {
    pub fn new(win: f32, draw: f32, loss: f32) -> Self {
        let mut res = Self { win, draw, loss };
        res.normalize();
        res
    }

    fn normalize(&mut self) {
        let sum = self.win + self.draw + self.loss;
        if sum > 0.0 {
            let inv = 1.0 / sum;
            self.win *= inv;
            self.draw *= inv;
            self.loss *= inv;
        }
    }

    pub fn score(&self) -> f32 {
        self.win + 0.5 * self.draw
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Evaluation {
    pub raw: EvalWdl,
    pub adjusted: EvalWdl,
}

#[derive(Clone)]
pub struct ChessState {
    board: Position,
    castling: Castling,
    stack: Vec<u64>,
}

impl Default for ChessState {
    fn default() -> Self {
        Self::from_fen(Self::STARTPOS)
    }
}

impl ChessState {
    pub const STARTPOS: &'static str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    #[cfg(feature = "datagen")]
    pub const BENCH_DEPTH: usize = 4;

    #[cfg(not(feature = "datagen"))]
    pub const BENCH_DEPTH: usize = 6;

    pub fn board(&self) -> Position {
        self.board
    }

    pub fn castling(&self) -> Castling {
        self.castling
    }

    pub fn conv_mov_to_str(&self, mov: Move) -> String {
        mov.to_uci(&self.castling)
    }

    pub fn from_fen(fen: &str) -> Self {
        let mut castling = Castling::default();
        let board = Position::parse_fen(fen, &mut castling);

        Self {
            board,
            castling,
            stack: Vec::new(),
        }
    }

    pub fn map_legal_moves<F: FnMut(Move)>(&self, f: F) {
        self.board.map_legal_moves(&self.castling, f);
    }

    pub fn game_state(&self) -> GameState {
        self.board.game_state(&self.castling, &self.stack)
    }

    pub fn hash(&self) -> u64 {
        self.board.hash()
    }

    pub fn make_move(&mut self, mov: Move) {
        self.stack.push(self.board.hash());
        self.board.make(mov, &self.castling);

        if self.board.halfm() == 0 {
            self.stack.clear();
        }
    }

    pub fn stm(&self) -> usize {
        self.board.stm()
    }

    pub fn map_moves_with_policies<F: FnMut(Move, f32)>(&self, policy: &PolicyNetwork, mut f: F) {
        let hl = policy.hl(&self.board);

        self.map_legal_moves(|mov| {
            let policy = policy.get(&self.board, &mov, &hl);
            f(mov, policy);
        });
    }

    pub fn get_policy_hl(&self, policy: &PolicyNetwork) -> Accumulator<i16, { POLICY_L1 / 2 }> {
        policy.hl(&self.board)
    }

    pub fn get_policy(
        &self,
        mov: Move,
        hl: &Accumulator<i16, { POLICY_L1 / 2 }>,
        policy: &PolicyNetwork,
    ) -> f32 {
        policy.get(&self.board, &mov, hl)
    }

    #[cfg(not(feature = "datagen"))]
    fn piece_count(&self, piece: usize) -> i32 {
        self.board.piece(piece).count_ones() as i32
    }

    pub fn evaluate_wdl(&self, value: &ValueNetwork, params: &MctsParams) -> Evaluation {
        let (win, draw, loss) = value.eval(&self.board);
        let raw = EvalWdl::new(win, draw, loss);
        let adjusted = apply_contempt(raw, params.contempt() as f32);
        Evaluation { raw, adjusted }
    }

    pub fn get_value(&self, value: &ValueNetwork, params: &MctsParams) -> i32 {
        const K: f32 = 400.0;
        let evaluation = self.evaluate_wdl(value, params);
        let score = evaluation.adjusted.score();
        let cp = (-K * (1.0 / score.clamp(0.0, 1.0) - 1.0).ln()) as i32;

        #[cfg(not(feature = "datagen"))]
        {
            use montyformat::chess::consts::Piece;

            let mut mat = self.piece_count(Piece::KNIGHT) * params.knight_value()
                + self.piece_count(Piece::BISHOP) * params.bishop_value()
                + self.piece_count(Piece::ROOK) * params.rook_value()
                + self.piece_count(Piece::QUEEN) * params.queen_value();

            mat = params.material_offset() + mat / params.material_div1();

            cp * mat / params.material_div2()
        }

        #[cfg(feature = "datagen")]
        cp
    }

    pub fn get_value_wdl(&self, value: &ValueNetwork, params: &MctsParams) -> f32 {
        self.evaluate_wdl(value, params).adjusted.score()
    }

    pub fn perft(&self, depth: usize) -> u64 {
        perft::<true, true>(&self.board, depth as u8, &self.castling)
    }

    pub fn display(&self, policy: &PolicyNetwork) {
        let mut moves = Vec::new();
        let mut max = f32::NEG_INFINITY;
        self.map_moves_with_policies(policy, |mov, policy| {
            moves.push((mov, policy));

            if policy > max {
                max = policy;
            }
        });

        let mut total = 0.0;

        for (_, policy) in moves.iter_mut() {
            *policy = (*policy - max).exp();
            total += *policy;
        }

        for (_, policy) in moves.iter_mut() {
            *policy /= total;
        }

        let mut w = [0f32; 64];
        let mut count = [0; 64];

        for &(mov, policy) in moves.iter() {
            let fr = usize::from(mov.src());
            let to = usize::from(mov.to());

            w[fr] = w[fr].max(policy);
            w[to] = w[to].max(policy);

            count[fr] += 1;
            count[to] += 1;
        }

        let pcs = [
            ['p', 'n', 'b', 'r', 'q', 'k'],
            ['P', 'N', 'B', 'R', 'Q', 'K'],
        ];

        println!("+-----------------+");

        for i in (0..8).rev() {
            print!("|");

            for j in 0..8 {
                let sq = 8 * i + j;
                let pc = self.board.get_pc(1 << sq);
                let ch = if pc != 0 {
                    let is_white = self.board.piece(0) & (1 << sq) > 0;
                    pcs[usize::from(is_white)][pc - 2]
                } else {
                    '.'
                };

                if count[sq] > 0 {
                    let g = (255.0 * (2.0 * w[sq]).min(1.0)) as u8;
                    let r = 255 - g;
                    print!(" \x1b[38;2;{r};{g};0m{ch}\x1b[0m");
                } else {
                    print!(" \x1b[34m{ch}\x1b[0m");
                }
            }

            println!(" |");
        }

        println!("+-----------------+");
    }
}

fn apply_contempt(raw: EvalWdl, contempt: f32) -> EvalWdl {
    if contempt == 0.0 {
        return raw;
    }

    let v = raw.win - raw.loss;
    let d = raw.draw;
    let w = (1.0 + v - d) * 0.5;
    let l = (1.0 - v - d) * 0.5;
    const EPS: f32 = 1e-4;

    if w <= EPS || l <= EPS || w >= 1.0 - EPS || l >= 1.0 - EPS {
        return raw;
    }

    let a = (1.0 / l - 1.0).ln();
    let b = (1.0 / w - 1.0).ln();
    let denom = a + b;

    if !denom.is_finite() || denom.abs() < 1e-6 {
        return raw;
    }

    let s = 2.0 / denom;
    let mu = (a - b) / denom;

    let delta_mu = contempt * std::f32::consts::LN_10 / 400.0;
    let mu_new = (mu + delta_mu).clamp(-8.0, 8.0);
    let s_new = s;

    let logistic = |x: f32| 1.0 / (1.0 + (-x).exp());
    let w_new = logistic((-1.0 + mu_new) / s_new);
    let l_new = logistic((-1.0 - mu_new) / s_new);
    let mut d_new = (1.0 - w_new - l_new).max(0.0);

    if d_new > 1.0 {
        d_new = 1.0;
    }

    EvalWdl::new(w_new, d_new, l_new)
}

fn perft<const ROOT: bool, const BULK: bool>(
    pos: &Position,
    depth: u8,
    castling: &Castling,
) -> u64 {
    let mut count = 0;

    if BULK && !ROOT && depth == 1 {
        pos.map_legal_moves(castling, |_| count += 1);
    } else {
        let leaf = depth == 1;

        pos.map_legal_moves(castling, |mov| {
            let mut tmp = *pos;
            tmp.make(mov, castling);

            let num = if !BULK && leaf {
                1
            } else {
                perft::<false, BULK>(&tmp, depth - 1, castling)
            };

            count += num;

            if ROOT {
                println!("{}: {num}", mov.to_uci(castling));
            }
        });
    }

    count
}
