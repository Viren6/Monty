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

const BATCH_SIZE: usize = 256;
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
            temp: 2.6,
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

    let stop = AtomicBool::new(false);

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
    
    // Collect logits for legal moves
    for mov in &moves {
         let idx = crate::lc0_mapping::get_lc0_index(mov);
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
    
    // Softmax
    let mut sum_exp = 0.0;
    let mut probs = Vec::with_capacity(moves.len());
    let mut best_move_idx = 0;
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
    
    // Normalize and Store
    if sum_exp > 1e-9 {
        let scale = 1.0 / sum_exp;
        for (i, mov) in moves.iter().enumerate() {
            let p = probs[i] * scale;
            let mf_move = montyformat::chess::Move::from(u16::from(*mov));
            let visits = (p * 65535.0) as u32;
            dist.push((mf_move, visits));
            
            if p > max_prob {
                max_prob = p;
                best_move_idx = i;
            }
        }
    } else {
        // Fallback: Uniform
        let uniform = 65535 / moves.len() as u32;
        for mov in &moves {
            let mf_move = montyformat::chess::Move::from(u16::from(*mov));
            dist.push((mf_move, uniform));
        }
    }
    
    let best_move = if !moves.is_empty() { moves[best_move_idx] } else { montyformat::chess::Move::default() };

    // Use LC0 Value (Q is typically -1.0 to 1.0 from perspective of STM)
    // Monty expects score 0.0 (Loss) to 1.0 (Win).
    // So map: (q + 1.0) / 2.0
    let score = (lc0_value + 1.0) / 2.0;

    let mf_best_move = montyformat::chess::Move::from(u16::from(best_move));

    // VERIFICATION: Check Policy Integrity
    if game.iters < 3 {
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
    }

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
            GameState::Won(_) => if game.position.stm() == 0 { 1.0 } else { 0.0 }, // Opponent lost? STM won? Won(x) usually means side to move won? No, GameState is usually static property or relative to side to move.
            // montyformat::chess::GameState::Won(x) => "W{x}".
            // Position::game_state returns Won if... wait, it doesn't return Won. It returns Lost(0) if mate.
            // So Won is probably unused for now or used for valid claim.
            // If Lost(0): "I lost". So result for me is 0.0.
            // If stm=0 (White), and Lost: White lost -> 0.0. Correct.
            // If stm=1 (Black), and Lost: Black lost -> result (White perspective) is 1.0. 
            // result var is score for White?
            // "let result = (game.result * 2.0) as usize" in Destination::push_policy.
            // MontyFormat result: 1.0 (White Win), 0.5 (Draw), 0.0 (Black Win).
            // So if White (0) lost: result=0.0.
            // If Black (1) lost: result=1.0.
            // Logic: if stm==0 { 0.0 } else { 1.0 }.
            // Wait, if stm==0 (White) and Lost, White gets 0.0. Correct.
            // If stm==1 (Black) and Lost, Black gets 0.0 (from his perspective) -> White gets 1.0. Correct.
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
        // We only passed book_slice to init, we should probably pass it to reset implicitly or store it in GameRunner.
        // GameRunner reset currently uses new(book), but we need to pass the book again.
        // Current impl of reset calls new.
        // I need to store book ref in GameRunner? Or just pass it.
        // I won't complicated GameRunner struct lifetime.
        // Modified process_game to just call reset(None) is BAD if we want book diversity.
        // But since we have BATCH_SIZE games, they diverge fast.
        // I'll leave it as reset(None) which defaults to STARTPOS for simplicity, 
        // OR ideally pass the book from `run_policy_datagen`.
        // Let's rely on STARTPOS for subsequent games or simple random book if I can.
        // Since `process_game` doesn't have access to `book` slice easily without passing it...
        // and datagen usually runs from book positions.
        // I should fix this.
    }
}
