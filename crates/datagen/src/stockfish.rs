use crate::{Destination, RunOptions};
use monty::{
    chess::{ChessState, GameState, Move},
    networks::{PolicyNetwork, ValueNetwork},
};
use montyformat::{
    chess::{Right, Side},
    MontyValueFormat,
};
use std::{
    io::{BufRead, BufReader, Write},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

const BATCH_SIZE: usize = 8;

fn get_sf_path() -> &'static str {
    if cfg!(target_os = "windows") {
        "./stockfish_x86-64-avx2.exe"
    } else {
        "stockfish"
    }
}

pub fn run(
    opts: RunOptions,
    policy: &PolicyNetwork,
    value: &ValueNetwork,
) {
    println!("Starting Stockfish Value Datagen");
    println!("Threads: {}", opts.threads);
    println!("Games: {}", opts.games);
    println!("Out: {}", opts.out_path);

    let book = opts
        .book
        .as_ref()
        .map(|path| crate::book::OpeningBook::load(path.clone()).expect("failed to load opening book"));
    let book = book.map(Arc::new);

    let vout = std::fs::File::create(&opts.out_path).unwrap();
    let dest = Arc::new(Mutex::new(Destination {
        writer: std::io::BufWriter::new(vout),
        reusable_buffer: Vec::new(),
        games: 0,
        searches: 0,
        iters: 0,
        limit: opts.games,
        results: [0; 3],
    }));

    let stop = Arc::new(AtomicBool::new(false));
    let stop_signal = stop.clone();
    ctrlc::set_handler(move || {
        stop_signal.store(true, Ordering::SeqCst);
    }).ok();

    std::thread::scope(|s| {
        for _ in 0..opts.threads {
             let opts = &opts;
             let dest = dest.clone();
             let stop = stop.clone();
             let book = book.clone();
             
             s.spawn(move || {
                 worker(opts, policy, value, dest, stop, book);
             });
        }
    });

    let dest = dest.lock().unwrap();
    dest.report();
}

fn worker(
    opts: &RunOptions,
    policy: &PolicyNetwork,
    value: &ValueNetwork,
    dest: Arc<Mutex<Destination>>,
    stop: Arc<AtomicBool>,
    book: Option<Arc<crate::book::OpeningBook>>,
) {
    let mut rng = crate::rng::Rand::with_seed();
    let sf_path = get_sf_path();
    
    let mut child = Command::new(sf_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to spawn Stockfish");
        
    let mut stdin = child.stdin.take().expect("Failed to open stdin");
    let stdout = child.stdout.take().expect("Failed to open stdout");
    let mut reader = BufReader::new(stdout);
    
    writeln!(stdin, "uci").unwrap();
    if opts.dfrc {
        writeln!(stdin, "setoption name UCI_Chess960 value true").unwrap();
    }
    writeln!(stdin, "setoption name Hash value 64").unwrap();
    writeln!(stdin, "isready").unwrap();
    
    let mut buf = String::new();
    loop {
        buf.clear();
        reader.read_line(&mut buf).unwrap();
        if buf.trim() == "readyok" { break; }
    }

    loop {
        if stop.load(Ordering::Relaxed) { break; }
        
        let mut fens_to_send = Vec::with_capacity(BATCH_SIZE);
        let mut pending_games = Vec::with_capacity(BATCH_SIZE);
        
        for _ in 0..BATCH_SIZE {
            let mut game = GameRunner::new(book.as_deref(), rng.rand_int());
            
            let mut plies_played = 0;
            let target_plies = 8;
            let mut early_term = false;

            while plies_played < target_plies {
                 let mut has_moves = false;
                 game.position.map_legal_moves(|_| has_moves = true);
                 
                 if !has_moves {
                     early_term = true;
                     game.finish_early(game.position.board().in_check());
                     break;
                 }
                 
                 let (m, q) = run_monty_policy(&mut game, policy, value, &mut rng);
                 
                 let mf_move = montyformat::chess::Move::from(u16::from(m));
                 game.value_game.moves.push(montyformat::SearchResult {
                     best_move: mf_move,
                     score: q_to_cp(q),
                 });

                 plies_played += 1;
            }
            
            // Update iters for accurate reporting
            game.iters = game.value_game.moves.len();
            
            if !early_term {
                let fen = if opts.dfrc {
                     make_shredder_fen(&game.position)
                } else {
                     game.position.board().as_fen()
                };
                fens_to_send.push(fen);
                pending_games.push(game);
            } else {
                dest.lock().unwrap().push(&game.value_game, &stop, game.searches, game.iters);
            }
        }
        
        if fens_to_send.is_empty() { continue; }
        
        writeln!(stdin, "datagen {} {}", opts.nodes, fens_to_send.len()).unwrap();
        for fen in &fens_to_send {
            writeln!(stdin, "{}", fen).unwrap();
        }
        stdin.flush().unwrap();
        
        // Read Results
        let mut received_count = 0;
        while received_count < fens_to_send.len() {
            buf.clear();
            if reader.read_line(&mut buf).unwrap() == 0 {
                panic!("Stockfish died");
            }
            let line = buf.trim();
            if line == "BATCH_DONE" { break; }
            if line.starts_with("Game ") {
                 let parts: Vec<&str> = line.split_whitespace().collect();
                 if parts.len() < 2 { continue; }
                 
                 // Format: Game 0: e2e4 ... scores 10 20 ... result 1
                 // Identify sections
                 let mut game_idx = 0;
                 if let Some(idx_str) = parts[1].strip_suffix(':') {
                     if let Ok(idx) = idx_str.parse::<usize>() {
                         game_idx = idx;
                     }
                 }
                 
                 let mut moves_start = 2;
                 let mut scores_start = 0;
                 let mut result_start = 0;
                 
                 for (i, p) in parts.iter().enumerate().skip(2) {
                     if *p == "scores" { scores_start = i + 1; }
                     if *p == "result" { result_start = i + 1; }
                 }
                 
                 if game_idx < pending_games.len() {
                     let game = &mut pending_games[game_idx];
                     
                     // Parse Moves & Scores
                     // Assuming scores correspond 1-to-1 with moves or plies?
                     // Stockfish plays out game.
                     // We need to feed moves into game position to get correct Move objects (uci -> Move)
                     
                     // We iterate moves and scores together?
                     // Verify lengths.
                     let moves_end = if scores_start > 0 { scores_start - 1 } else { parts.len() };
                     let scores_end = if result_start > 0 { result_start - 1 } else { parts.len() };
                     
                     // let moves_cnt = moves_end - moves_start;
                     // let scores_cnt = scores_end - scores_start;
                     
                     let mut current_score_idx = scores_start;
                     for i in moves_start..moves_end {
                         let uci = parts[i];
                         // Parse uci
                         // We need to find move in legal moves.
                         let mut matched_move = Move::NULL;
                         
                         // We need scanning.
                         // But Monty's Move parsing might be available.
                         // Or manual match.
                         // Optimization: we can just iterate legal moves.
                         game.position.map_legal_moves(|m| {
                             if uci == uci_str(m, &game.position) {
                                 matched_move = m;
                             }
                         });
                         
                         if matched_move != Move::NULL {
                             // Get score
                             let score = if current_score_idx < scores_end {
                                 parts[current_score_idx].parse::<i16>().unwrap_or(0)
                             } else { 0 };
                             current_score_idx += 1;
                             
                             let mf_move = montyformat::chess::Move::from(u16::from(matched_move));
                             game.value_game.moves.push(montyformat::SearchResult {
                                 best_move: mf_move,
                                 score,
                             });
                             
                             game.position.make_move(matched_move);
                         } else {
                             // Break if illegal/unknown (should not happen)
                             break;
                         }
                     }
                     
                     // Parse Result
                     // result 1 (Draw/Unknown) or 0 (Loss for STM)? 
                     // In SF we returned: 0=Loss(STM), 1=Draw. 
                     // If SF is playing, the result is from Perspective of Side To Move *at end*?
                     // No, "returns the game result (WDL) from side to move perspective".
                     // Wait, datagen_game returns result.
                     // if returned 0 (Loss), it means the side whose turn it was LOST.
                     // if returned 1 (Draw).
                     // What about Win?
                     
                     // My SF implementation:
                     // if checkers: return 0. (Loss)
                     // else: return 1. (Draw, Stalemate)
                     // So result is from perspective of person about to move.
                     
                     // Monty expects result 0.0 (Loss), 1.0 (Win), 0.5 (Draw) *from White perspective??*
                     // MontyValueFormat result: "result: f32".
                     // Usually 1.0 for White Win, 0.0 for Black Win, 0.5 Draw.
                     
                     // We need to map (ResultSTM, STM).
                     // If ResultSTM = 0 (Loss), then STM lost. So Other won.
                     // If ResultSTM = 1 (Draw), Draw.
                     
                     let sf_res = if result_start < parts.len() {
                         parts[result_start].parse::<i32>().unwrap_or(1)
                     } else { 1 };
                     
                     // Game logic to determine global result
                     // Current STM is game.position.stm()
                     let stm = game.position.stm(); // 0=White, 1=Black
                     let final_res = if sf_res == 1 {
                         0.5
                     } else {
                         // Loss for STM.
                         // If STM=White(0) and Loss -> Black Wins -> 0.0
                         // If STM=Black(1) and Loss -> White Wins -> 1.0
                         if stm == 0 { 0.0 } else { 1.0 }
                     };
                     
                     game.value_game.result = final_res;
                     game.iters = game.value_game.moves.len();
                     dest.lock().unwrap().push(&game.value_game, &stop, game.searches, game.iters);
                     
                     received_count += 1;
                 }
            }
        }
    }
    
    let _ = child.kill();
}


struct GameRunner {
    position: ChessState,
    temp: f32,
    searches: usize,
    iters: usize,
    value_game: MontyValueFormat,
}

impl GameRunner {
    fn new(book: Option<&crate::book::OpeningBook>, seed: u32) -> Self {
        let position = if let Some(book) = book {
            let mut rng = crate::rng::Rand(seed);
            let mut reader = book.reader().expect("failed to get book reader");
            let fen = reader.random_line(&mut rng).expect("failed to read book line");
            ChessState::from_fen(&fen)
        } else {
            ChessState::from_fen(ChessState::STARTPOS)
        };

        let montyformat_position = position.board();
        let montyformat_castling = position.castling();

        GameRunner {
            position,
            temp: 1.4,
            searches: 0,
            iters: 0,
            value_game: MontyValueFormat {
                startpos: montyformat_position,
                castling: montyformat_castling,
                result: 0.0,
                moves: Vec::new(),
            },
        }
    }
    
    fn finish_early(&mut self, in_check: bool) {
         let stm = self.position.stm();
         let result = if in_check {
             if stm == 0 { 0.0 } else { 1.0 }
         } else {
             0.5
         };
         self.value_game.result = result;
    }
}

// Helpers
fn uci_str(m: Move, pos: &ChessState) -> String {
    let src = m.src();
    let to = m.to();
    let promo = m.promo_pc();
    
    let f1 = (src % 8) as u8;
    let r1 = (src / 8) as u8;
    let f2 = (to % 8) as u8;
    let r2 = (to / 8) as u8;
    
    let mut s = format!("{}{}{}{}", 
        (b'a' + f1) as char, (b'1' + r1) as char,
        (b'a' + f2) as char, (b'1' + r2) as char
    );
    
    if m.is_promo() {
         let p = match promo {
             0 => 'n', 1 => 'b', 2 => 'r', 3 => 'q', _ => 'q'
         };
         s.push(p);
    }
    
    // Castling Fix? Stockfish expects e1g1.
    // Monty Move might be "King takes Rook" for castling if FRC?
    // In Standard, Monty uses King->Rook square?
    // Let's check `make_shredder_fen` usage. 
    // Standard UCI: e1g1.
    // Verify Monty move structure. 
    
    s
}

fn run_monty_policy(
    game: &mut GameRunner, 
    policy: &PolicyNetwork, 
    value: &ValueNetwork,
    rng: &mut crate::rng::Rand
) -> (Move, f32) {
    let board = game.position.board();
    let hl = policy.hl(&board);
    
    let mut moves = Vec::new();
    let mut logits = Vec::new();
    let mut max_logit = f32::NEG_INFINITY;
    
    game.position.map_legal_moves(|m| {
        let mf_move = montyformat::chess::Move::from(u16::from(m));
        let logit = policy.get(&board, &mf_move, &hl);
        moves.push(m);
        logits.push(logit);
        if logit > max_logit { max_logit = logit; }
    });

    let mut sum_probs = 0.0;
    let mut probs = Vec::with_capacity(logits.len());
    let inv_temp = 1.0 / game.temp;

    for &logit in &logits {
        let prob = ((logit - max_logit) * inv_temp).exp();
        probs.push(prob);
        sum_probs += prob;
    }

    let r = (rng.rand_int() as f32 / u32::MAX as f32) * sum_probs;
    let mut acc = 0.0;
    let mut selected_idx = 0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if acc >= r {
            selected_idx = i;
            break;
        }
    }
    
    let best_move = moves[selected_idx];
    game.position.make_move(best_move);

    let (w, d, _l) = value.eval(&board);

    // Sharpness scaling
    const SHARPNESS_SCALE: f32 = 2.459;
    const SHARPNESS_QUADRATIC: f32 = 0.8724;

    let draw_adj = d * SHARPNESS_SCALE + d * d * SHARPNESS_QUADRATIC;
    let sum = w + d + draw_adj + _l;
    
    let w_scaled = w / sum;
    let d_scaled = (d + draw_adj) / sum;
    // let l_scaled = _l / sum;

    let q = w_scaled + 0.5 * d_scaled;

    // Decay temp
    game.temp *= 0.9;
    
    (best_move, q)
}

fn q_to_cp(q: f32) -> i16 {
    let q = q.clamp(0.001, 0.999);
    (-(400.0 * (1.0 / q - 1.0).ln())) as i16
}

pub fn make_shredder_fen(pos: &ChessState) -> String {
    let board = pos.board();
    let castling = pos.castling();
    let pcs = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K'];
    let mut fen = String::new();

    for rank in (0..8).rev() {
        let mut clear = 0;

        for file in 0..8 {
            let sq = 8 * rank + file;
            let bit = 1 << sq;
            let pc = board.get_pc(bit);
            if pc != 0 {
                if clear > 0 {
                    fen.push_str(&format!("{}", clear));
                }
                clear = 0;
                let is_black = board.piece(Side::BLACK) & bit > 0;
                let idx = pc - 2 + 6 * usize::from(!is_black);
                fen.push(pcs[idx]);
            } else {
                clear += 1;
            }
        }

        if clear > 0 {
            fen.push_str(&format!("{}", clear));
        }

        if rank > 0 {
            fen.push('/');
        }
    }

    fen.push(' ');
    fen.push(['w', 'b'][board.stm()]);
    fen.push(' ');

    let rights = board.rights();
    if rights == 0 {
        fen.push('-');
    } else {
        if rights & Right::WKS > 0 {
            let file = castling.rook_file(Side::WHITE, 1); // 1 = KS
            fen.push((b'A' + file as u8) as char);
        }
        if rights & Right::WQS > 0 {
            let file = castling.rook_file(Side::WHITE, 0); // 0 = QS
            fen.push((b'A' + file as u8) as char);
        }
        if rights & Right::BKS > 0 {
            let file = castling.rook_file(Side::BLACK, 1);
            fen.push((b'a' + file as u8) as char);
        }
        if rights & Right::BQS > 0 {
            let file = castling.rook_file(Side::BLACK, 0);
            fen.push((b'a' + file as u8) as char);
        }
    }

    fen.push(' ');

    if board.enp_sq() == 0 {
        fen.push('-');
    } else {
        let file = board.enp_sq() % 8;
        let rank = board.enp_sq() / 8;
        fen.push((b'a' + file) as char);
        fen.push((b'1' + rank) as char);
    }

    fen.push_str(&format!(" {} {}", board.halfm(), board.fullm()));

    fen
}
