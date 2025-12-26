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

pub fn run_policy_datagen(
    opts: RunOptions,
) {
    println!("Starting LC0 Datagen with BATCH_SIZE={}", BATCH_SIZE);
    println!("Using Network: {}", LC0_NETWORK_PATH);

    // Ensure lc0_inference exists relative to CWD or use absolute path if simpler
    // Just using "./lc0_inference/lc0_inference.exe" for Windows
    let exe_path = if cfg!(target_os = "windows") {
        "./lc0_inference_standalone/lc0_inference.exe"
    } else {
        "./lc0_inference_standalone/lc0_inference"
    };

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

        // 1. Send FENs
        for game in &games {
            let fen = game.position.board().as_fen();
            writeln!(stdin, "{}", fen).unwrap();
        }
        stdin.flush().unwrap();

        // 2. Read Results
        let mut game_idx = 0;
        let mut current_policy = [0.0f32; 1858]; // Reset per game? Yes.
        let mut current_value = 0.0f32;
        
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
                 // Reset policy buffer to NEG_INFINITY (missing logits are effectively 0 prob)
                 current_policy = [f32::NEG_INFINITY; 1858];
                 current_value = 0.0;
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
                    process_game(game, &current_policy, current_value, &dest, &stop, &mut rng, opts.policy_data, book_ref);
                    game_idx += 1;
                }
            } else if line.starts_with("Policy (Top > 1%):") {
                 // Legacy ignore
            }
        }
    }
    
    let _ = child.kill();
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
