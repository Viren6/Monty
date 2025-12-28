use crate::{Destination, RunOptions};
use monty::{
    chess::{ChessState, GameState, Move},
};
use montyformat::{MontyFormat, MontyValueFormat, SearchData};
use std::{
    io::{BufRead, BufReader, Write},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

const BATCH_SIZE: usize = 1024;
// In a real scenario, this path might be dynamic or configured via env var, 
// but user requested variable to be in script.
const LC0_NETWORK_PATH: &str = r"C:\Users\viren\Documents\GitHub\Monty0\bt4-1024x15x32h-swa-6147500.pb.gz";

struct GameRunner {
    position: ChessState,
    temp: f32,
    searches: usize,
    iters: usize,
    policy_game: MontyFormat,
    #[allow(dead_code)]
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
            policy_game: MontyFormat::new(montyformat_position, montyformat_castling),
            value_game: MontyValueFormat {
                startpos: montyformat_position,
                castling: montyformat_castling,
                result: 0.0,
                moves: Vec::new(),
            },
        }
    }

    fn reset(&mut self, book: Option<&crate::book::OpeningBook>, seed: u32) {
        *self = Self::new(book, seed);
    }
}

// ... (skipping run_policy_datagen usually, but I need to target process_game which is far below)
// I will split this into two chunks if needed, but context helps.
// Actually, StartLine is 44. I'll just do the Constructor chunk first.

#[allow(unused)]
fn dummy() {}

static TOTAL_FAILURES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static UNRESOLVED_FAILURES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn get_exe_path() -> &'static str {
    if cfg!(target_os = "windows") {
        "./lc0_inference_standalone/lc0_inference.exe"
    } else {
        "./lc0_inference_standalone/lc0_inference"
    }
}

pub fn run_policy_datagen(
    opts: RunOptions,
) {
    println!("Starting LC0 Datagen with BATCH_SIZE={}", BATCH_SIZE);
    println!("Using Network: {}", LC0_NETWORK_PATH);

    let exe_path = get_exe_path();

    let mut child = Command::new(exe_path)
        .arg(LC0_NETWORK_PATH)
        .arg(BATCH_SIZE.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to spawn lc0_inference. Make sure it is compiled and in lc0_inference directory.");

    let mut stdin = child.stdin.take().expect("Failed to open stdin");
    let stdout = child.stdout.take().expect("Failed to open stdout");
    let mut reader = BufReader::new(stdout);

    // Opening book
    let book = opts
        .book
        .map(|path| crate::book::OpeningBook::load(path).expect("failed to load opening book"));
    let book_ref = book.as_ref();

    // Destination
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
    
    // Graceful Shutdown
    let stop_signal = stop.clone();
    ctrlc::set_handler(move || {
        stop_signal.store(true, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let mut rng = crate::rng::Rand::with_seed();
    let mut games: Vec<GameRunner> = (0..BATCH_SIZE)
        .map(|_| GameRunner::new(book_ref, rng.rand_int()))
        .collect();

    let mut buffer = String::new();

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }

        // 1. Prepare Games (Handle Terminal States)
        for game in &mut games {
            let mut moves = 0;
            game.position.map_legal_moves(|_| moves += 1);
            
            if moves == 0 {
                let in_check = game.position.board().in_check();
                
                let result = if in_check {
                    if game.position.stm() == 0 { 0.0 } else { 1.0 }
                } else {
                    0.5
                };

                if opts.policy_data {
                    game.policy_game.result = result;
                    dest.lock().unwrap().push_policy(&game.policy_game, &stop, game.searches, game.iters);
                } else {
                    game.value_game.result = result;
                    dest.lock().unwrap().push(&game.value_game, &stop, game.searches, game.iters);
                }
                
                game.reset(book_ref, rng.rand_int());
            }
        }

        // 2. Send FENs
        for game in &games {
            let fen = game.position.board().as_fen();
            writeln!(stdin, "{}", fen).unwrap();
        }
        stdin.flush().unwrap();

        // 2. Read Results
        let mut game_idx = 0;
        let mut current_policy = [f32::NEG_INFINITY; 1858]; 
        let mut current_value = 0.0f32;
        let mut reading_fen = String::new();
        
        loop {
            buffer.clear();
            if reader.read_line(&mut buffer).unwrap() == 0 {
                // EOF
                panic!("LC0 process died unexpectedly");
            }
            let line = buffer.trim();
            if line == "BATCH_DONE" {
                break;
            }
            
            if line.starts_with("FEN:") {
                 // New game starting in stream
                 current_policy = [f32::NEG_INFINITY; 1858];
                 current_value = 0.0;
                 if let Some(f) = line.strip_prefix("FEN: ") {
                     reading_fen = f.trim().to_string();
                 }
            } else if line.starts_with("Value:") {
                if let Some(val_str) = line.split_whitespace().nth(1) {
                     current_value = val_str.parse().unwrap_or(0.0);
                }
            } else if line.starts_with("Policy (Logits):") {
                // Parse "idx:logit"
                let content = line.trim_start_matches("Policy (Logits):").trim();
                for token in content.split_whitespace() {
                    if let Some((idx_str, val_str)) = token.split_once(':') {
                        if let (Ok(idx), Ok(val)) = (idx_str.parse::<usize>(), val_str.parse::<f32>()) {
                           if idx < 1858 {
                               current_policy[idx] = val;
                           }
                        }
                    }
                }
                
                // Trigger processing after Policy line
                if game_idx < BATCH_SIZE {
                    let game = &mut games[game_idx];

                    // VALIDATION
                    let mut valid = true;
                    if current_value.is_nan() || current_value.is_infinite() {
                        valid = false;
                        println!("ERROR: Na/Inf Value for FEN: {}", reading_fen);
                    }
                    
                    let mut has_finite = false;
                    if valid {
                        for &p in &current_policy {
                            if p.is_nan() || p.is_infinite() {
                                // If we see NaN or POS_INFINITY: BAD.
                                // If we see NEG_INFINITY, that's fine (unplayed move).
                                if p == f32::NEG_INFINITY { continue; }
                                valid = false;
                                break;
                            }
                            has_finite = true;
                        }
                    }

                    // Strict "No Finite Logits" handling (Terminal State Disagreement)
                    // If Monty thought there were moves, but LC0 returns NO finite logits, 
                    // it means the position was actually terminal (Checkmate or Stalemate).
                    if valid && !has_finite {
                        // Handle as Terminal State
                        let in_check = game.position.board().in_check();
                        
                        let result = if in_check {
                             // Checkmate: Loss for STM
                             if game.position.stm() == 0 { 0.0 } else { 1.0 }
                        } else {
                             // Stalemate: Draw
                             0.5
                        };
                        
                        // Log event but don't panic
                        println!("Info: LC0 detected terminal state for FEN: {}. Result: {}", reading_fen, result);

                        if opts.policy_data {
                            game.policy_game.result = result;
                            dest.lock().unwrap().push_policy(&game.policy_game, &stop, game.searches, game.iters);
                        } else {
                            game.value_game.result = result;
                            dest.lock().unwrap().push(&game.value_game, &stop, game.searches, game.iters);
                        }

                        game.reset(book_ref, rng.rand_int());
                        game_idx += 1;
                        continue; 
                    }

                    if !valid {
                        TOTAL_FAILURES.fetch_add(1, Ordering::Relaxed);
                        println!("Validation FAILED for FEN: {}", reading_fen);
                        
                        // RETRY MECHANISM
                        let mut resolved = false;
                        for attempt in 1..=3 {
                            println!("Attempting Retry {}/3...", attempt);
                            if let Some((retry_pol, retry_val)) = run_single_inference_retry(&reading_fen) {
                                current_policy = retry_pol;
                                current_value = retry_val;
                                resolved = true;
                                println!("Retry SUCCESS.");
                                break;
                            }
                        }

                        if !resolved {
                            UNRESOLVED_FAILURES.fetch_add(1, Ordering::Relaxed);
                            println!("All retries FAILED. Using UNIFORM FALLBACK.");
                            
                            // FALLBACK to Uniform
                            current_value = 0.0;
                            current_policy = [0.0; 1858];
                        }
                    }

                    process_game(game, &current_policy, current_value, &dest, &stop, &mut rng, opts.policy_data, book_ref);
                    game_idx += 1;
                }
            } else if line.starts_with("Policy (Top > 1%):") {
                 // Legacy ignore
            }
        }
    }
    
    let _ = child.kill();
    
    println!("Datagen Finished.");
    println!("Total Failures: {}", TOTAL_FAILURES.load(Ordering::Relaxed));
    println!("Unresolved Failures (Fallbacks): {}", UNRESOLVED_FAILURES.load(Ordering::Relaxed));
}

fn run_single_inference_retry(fen: &str) -> Option<([f32; 1858], f32)> {
    let exe_path = get_exe_path();
    
    // Spawn fresh process with batch_size=1
    let mut child = Command::new(exe_path)
        .arg(LC0_NETWORK_PATH)
        .arg("1") 
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null()) // validation is noisy enough
        .spawn()
        .ok()?;

    {
        let stdin = child.stdin.as_mut()?;
        writeln!(stdin, "{}", fen).ok()?;
    } // close stdin to signal we are done sending? actually tool waits for newlines.
    
    let stdout = child.stdout.take()?;
    let mut reader = BufReader::new(stdout);
    let mut buffer = String::new();
    
    let mut policy = [f32::NEG_INFINITY; 1858];
    let mut value = 0.0f32;
    let mut found_policy = false;
    let mut found_value = false;

    loop {
        buffer.clear();
        if reader.read_line(&mut buffer).unwrap_or(0) == 0 { break; }
        let line = buffer.trim();
        if line == "BATCH_DONE" { break; }
        
        if line.starts_with("Value:") {
            if let Some(val_str) = line.split_whitespace().nth(1) {
                if let Ok(v) = val_str.parse::<f32>() {
                    value = v;
                    found_value = true;
                }
            }
        } else if line.starts_with("Policy (Logits):") {
             let content = line.trim_start_matches("Policy (Logits):").trim();
             for token in content.split_whitespace() {
                if let Some((idx_str, val_str)) = token.split_once(':') {
                    if let (Ok(idx), Ok(val)) = (idx_str.parse::<usize>(), val_str.parse::<f32>()) {
                       if idx < 1858 {
                           policy[idx] = val;
                       }
                    }
                }
             }
             found_policy = true;
        }
    }
    
    let _ = child.kill();

    if found_policy && found_value {
        // Validate again!
        if value.is_nan() || value.is_infinite() { return None; }
        let mut has_finite = false;
        for &p in &policy {
             if p.is_nan() || (p.is_infinite() && p == f32::INFINITY) { return None; }
             if p.is_finite() { has_finite = true; }
        }
        if !has_finite { return None; }
        
        return Some((policy, value));
    }
    
    None
}

fn process_game(
    game: &mut GameRunner,
    policy_probs: &[f32; 1858],
    lc0_value: f32,
    dest: &Arc<Mutex<Destination>>,
    stop: &AtomicBool,
    rng: &mut crate::rng::Rand,
    output_policy: bool,
    book: Option<&crate::book::OpeningBook>,
) {
    let mut moves = Vec::new();
    game.position.map_legal_moves(|mov| moves.push(mov));

    if moves.is_empty() {
        game.reset(book, rng.rand_int());
        return;
    }

    // 1. Compute Legal Move Probabilities from Logits (Softmax over Legal Moves)
    let mut dist = Vec::with_capacity(moves.len());
    let mut legal_logits = Vec::with_capacity(moves.len());
    let mut max_legal_logit = f32::NEG_INFINITY;
    
    let stm = game.position.stm(); // 0=White, 1=Black

    // Collect logits for legal moves
    for mov in &moves {
         // LC0 output is always relative (White perspective).
         // If we are Black, we must mirror the move to find the corresponding LC0 index.
         let lookup_move = if stm == 1 {
             let src = u16::from(mov.src());
             let dst = u16::from(mov.to());
             
             // Vertical flip: square ^ 56
             let src_mir = src ^ 56;
             let dst_mir = dst ^ 56;
             
             monty::chess::Move::new(src_mir, dst_mir, mov.flag())
         } else {
             *mov
         };

         // FIX: Map Castling to King-takes-Rook (FRC style) for LC0
         let flag = lookup_move.flag();
         let lookup_move = if flag == 2 || flag == 3 {
             let k_to = u16::from(lookup_move.to());
             let r_to = if k_to == 6 { // g1
                  7 // h1
             } else if k_to == 2 { // c1
                  0 // a1
             } else {
                  k_to 
             };
             monty::chess::Move::new(u16::from(lookup_move.src()), r_to, flag)
         } else {
             lookup_move
         };

         let idx = crate::lc0_mapping::get_lc0_index(&lookup_move);
         let logit = if let Some(idx) = idx {
             policy_probs[idx]
         } else {
             f32::NEG_INFINITY
         };
         
         if logit > max_legal_logit {
             max_legal_logit = logit;
         }
         legal_logits.push(logit);
    }
    

    
    // Softmax (Temp=1) for Storage
    let mut sum_exp = 0.0;
    let mut probs = Vec::with_capacity(moves.len());
    let mut greedy_best_move_idx = 0;
    let mut max_prob = -1.0;
    
    for logit in &legal_logits {
        if *logit > f32::NEG_INFINITY {
            let p = (*logit - max_legal_logit).exp();
            sum_exp += p;
            probs.push(p);
        } else {
            probs.push(0.0);
        }
    }
    
    // Normalize and Store in Dist
    let scale = 1.0 / sum_exp;
    for (i, mov) in moves.iter().enumerate() {
        let p = probs[i] * scale;
        let mf_move = montyformat::chess::Move::from(u16::from(*mov));
        let visits = (p * 65535.0) as u32;
        dist.push((mf_move, visits));
        
        if p > max_prob {
            max_prob = p;
            greedy_best_move_idx = i;
        }
    }
    
    // Select Played Move (Temp decay)
    let played_move_idx = if game.temp > 0.0 {
        // Sample with temperature
        let mut sum_exp_temp = 0.0;
        let mut probs_temp = Vec::with_capacity(moves.len());
        
        // Reuse max_legal_logit for stability: (l - max)/T
        for logit in &legal_logits {
             if *logit > f32::NEG_INFINITY {
                 let val = (*logit - max_legal_logit) / game.temp;
                 let p = val.exp();
                 sum_exp_temp += p;
                 probs_temp.push(p);
             } else {
                 probs_temp.push(0.0);
             }
        }
        
        // Sample
        let mut r = rng.rand_float() * sum_exp_temp;
        let mut selected = 0;
        // Robust sampling loop
        for (i, &p) in probs_temp.iter().enumerate() {
            if p > 0.0 {
                r -= p;
                if r <= 0.0 {
                    selected = i;
                    break;
                }
            }
        }
        // Correct float drift edge case
        if r > 0.0 { selected = probs_temp.len().saturating_sub(1); }
        selected
    } else {
        greedy_best_move_idx
    };
    
    let best_move =  moves[played_move_idx];
    
    // Decay Temperature
    game.temp *= 0.9;
    if game.temp < 0.2 {
        game.temp = 0.0f32;
    }

    // Use LC0 Value (Q is typically -1.0 to 1.0 from perspective of STM)
    // Monty expects score 0.0 (Loss) to 1.0 (Win).
    // So map: (q + 1.0) / 2.0
    let score = (lc0_value + 1.0) / 2.0;

    let mf_best_move = montyformat::chess::Move::from(u16::from(best_move));

    // VERIFICATION: Check Policy Integrity
    /* if game.iters < 3 {
        println!("--- VERIFICATION [Game {} Iter {}] ---", game.searches / BATCH_SIZE, game.iters);
        println!("FEN: {}", game.position.board().as_fen());
        println!("LC0 Value: {:.6} -> Score: {:.6}", lc0_value, score);
        
        println!("Max Legal Logit: {:.4}", max_legal_logit);
        let mut dist_sum = 0.0;
        for (m, v) in dist.iter().take(5) {
             let prob = (*v as f32) / 65535.0;
             println!("Move: {:?} | Visits: {} | Prob: {:.6}", m, v, prob);
        }
        for (_, v) in &dist {
            dist_sum += (*v as f32) / 65535.0;
        }
        println!("Stored Policy Sum: {:.6}", dist_sum);
        println!("Best Move: {}", best_move);
    }*/

    if output_policy {
        let search_data = SearchData::new(mf_best_move, score, Some(dist));
        game.policy_game.push(search_data);
    } else {
        game.value_game.push(game.position.stm(), mf_best_move, score);
    }

    game.searches += 1;
    game.iters += 1;

    game.position.make_move(best_move);

    let state = game.position.game_state();
    let over = match state {
         GameState::Ongoing => false,
         _ => true,
    };
    
    if over {
        let result = match state {
            GameState::Lost(_) => if game.position.stm() == 0 { 0.0 } else { 1.0 },
            GameState::Won(_) => if game.position.stm() == 0 { 1.0 } else { 0.0 }, 
            _ => 0.5,
        };
        
        if output_policy {
            game.policy_game.result = result;
            dest.lock().unwrap().push_policy(&game.policy_game, stop, game.searches, game.iters);
        } else {
             game.value_game.result = result;
             dest.lock().unwrap().push(&game.value_game, stop, game.searches, game.iters);
        }
        
        game.reset(book, rng.rand_int());
    }
}
