use crate::{Destination, RunOptions};
use monty::{
    chess::{ChessState, Move},
    networks::PolicyNetwork,
};
use montyformat::{
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

const BATCH_SIZE: usize = 64;

const SF_BIN: &[u8] = include_bytes!("../stockfish_bin");

fn prepare_sf_path() -> String {
    if cfg!(target_os = "windows") {
        "./stockfish_x86-64-avx2.exe".to_string()
    } else {
        let path = std::path::Path::new("./stockfish_embedded");
        if path.exists() {
            return "./stockfish_embedded".to_string();
        }

        match std::fs::write("stockfish_embedded", SF_BIN) {
            Ok(_) => {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mut perms = std::fs::metadata("stockfish_embedded").unwrap().permissions();
                    perms.set_mode(0o755);
                    std::fs::set_permissions("stockfish_embedded", perms).unwrap();
                }
                "./stockfish_embedded".to_string()
            }
            Err(_) => "stockfish".to_string(),
        }
    }
}

pub fn run(
    opts: RunOptions,
    policy: &PolicyNetwork,
) {
    println!("Starting Stockfish Value Datagen");
    println!("Threads: {}", opts.threads);
    println!("Games: {}", opts.games);
    println!("Out: {}", opts.out_path);

    // Prepare Stockfish binary (extract only once if needed)
    let sf_path = prepare_sf_path();

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
             let sf_path = sf_path.clone();
             
             s.spawn(move || {
                 worker(opts, policy, dest, stop, book, sf_path);
             });
        }
    });

    let dest = dest.lock().unwrap();
    dest.report();
}

fn worker(
    opts: &RunOptions,
    policy: &PolicyNetwork,
    dest: Arc<Mutex<Destination>>,
    stop: Arc<AtomicBool>,
    book: Option<Arc<crate::book::OpeningBook>>,
    sf_path: String,
) {
    let mut rng = crate::rng::Rand::with_seed();
    
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
            let mut game = GameRunner::new(book.as_deref(), rng.rand_int(), opts.dfrc);
            
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
                 
                 let m = run_monty_policy(&mut game, policy, &mut rng);
                 // We do NOT record these moves
                 game.position.make_move(m);

                 plies_played += 1;
            }
            
            if !early_term {
                // Update startpos to current position (after 8 plies)
                game.value_game.startpos = game.position.board();
                game.value_game.castling = game.position.castling();
                pending_games.push(Some(game));
            } else {
                game.iters = game.value_game.moves.len();
                dest.lock().unwrap().push(&game.value_game, &stop, game.searches, game.iters);
                pending_games.push(None);
            }
        }
            
        let mut batch_mapping = Vec::new();
        for (i, slot) in pending_games.iter().enumerate() {
            if let Some(game) = slot {
                fens_to_send.push(game.position.board().as_fen());
                batch_mapping.push(i);
            }
        }
        
        if fens_to_send.is_empty() { continue; }
        
        writeln!(stdin, "datagen {} {}", opts.nodes, fens_to_send.len()).unwrap();
        stdin.flush().unwrap();

        for fen in &fens_to_send {
            writeln!(stdin, "{}", fen).unwrap();
        }
        stdin.flush().unwrap();
        
        loop {
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
                 let mut sf_idx = 0;
                 if let Some(idx_str) = parts[1].strip_suffix(':') {
                     if let Ok(idx) = idx_str.parse::<usize>() {
                         sf_idx = idx;
                     }
                 }
                 
                 if sf_idx >= batch_mapping.len() {
                     eprintln!("Stockfish returned invalid game index: {}", sf_idx);
                     continue;
                 }
                 let game_idx = batch_mapping[sf_idx];
                 
                 let moves_start = 2;
                 let mut scores_start = 0;
                 let mut result_start = 0;
                 
                 for (i, p) in parts.iter().enumerate().skip(2) {
                     if *p == "scores" { scores_start = i + 1; }
                     if *p == "result" { result_start = i + 1; }
                 }
                 
                 if game_idx < pending_games.len() {
                     if let Some(game) = &mut pending_games[game_idx] {
                     
                     // Parse Moves & Scores
                     let moves_end = if scores_start > 0 { scores_start - 1 } else { parts.len() };
                     let scores_end = if result_start > 0 { result_start - 1 } else { parts.len() };
                     
                     let mut current_score_idx = scores_start;
                     for i in moves_start..moves_end {
                         let uci = parts[i];
                         // Parse uci
                         // We need to find move in legal moves.
                         let mut matched_move = Move::NULL;
                         
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
                            panic!("Mismatch! UCI: '{}' FEN: '{}' (sf_idx: {}, game_idx: {})", uci, fens_to_send[sf_idx], sf_idx, game_idx);
                        }
                     }
                     
                     // Parse Result
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
                         if stm == 0 { 0.0 } else { 1.0 }
                     };
                     
                     game.value_game.result = final_res;
                     game.searches = game.value_game.moves.len();
                     /*println!("Game Inserted");
                     let mut temp_pos = game.value_game.startpos;
                     println!("Initial FEN: {}", temp_pos.as_fen());
                     for res in &game.value_game.moves {
                         println!("Move: {}, Score: {}, FEN after:", res.best_move.to_uci(&game.value_game.castling), res.score);
                         let m = Move::from(u16::from(res.best_move));
                         temp_pos.make(res.best_move, &game.value_game.castling);
                         println!("{}", temp_pos.as_fen());
                     }
                     println!("Result: {}", game.value_game.result);*/
                     dest.lock().unwrap().push(&game.value_game, &stop, game.searches, game.iters);
                     
                     }
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
    fn new(book: Option<&crate::book::OpeningBook>, seed: u32, dfrc: bool) -> Self {
        let mut position = if let Some(book) = book {
            let mut rng = crate::rng::Rand(seed);
            let mut reader = book.reader().expect("failed to get book reader");
            let fen = reader.random_line(&mut rng).expect("failed to read book line");
            ChessState::from_fen(&fen)
        } else {
            ChessState::from_fen(ChessState::STARTPOS)
        };

        if dfrc {
            position.set_chess960(true);
        }

        let montyformat_position = position.board();
        let montyformat_castling = position.castling();

        GameRunner {
            position,
            temp: 1.0,
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
    m.to_uci(&pos.castling())
}

fn run_monty_policy(
    game: &mut GameRunner, 
    policy: &PolicyNetwork, 
    rng: &mut crate::rng::Rand
) -> Move {
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
    
    best_move
}


