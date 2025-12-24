use crate::{Destination, RunOptions};
use monty::{
    chess::{ChessState, GameState, Move},
    networks::{PolicyNetwork, ValueNetwork},
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
    fn new(book: Option<&crate::book::OpeningBook>) -> Self {
        let position = if let Some(book) = book {
            let mut rng = crate::rng::Rand::with_seed();
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
            temp: 0.8,
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

    fn reset(&mut self, book: Option<&crate::book::OpeningBook>) {
        *self = Self::new(book);
    }
}

pub fn run_policy_datagen(
    opts: RunOptions,
    value: &ValueNetwork,
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

    let mut games: Vec<GameRunner> = (0..BATCH_SIZE)
        .map(|_| GameRunner::new(book_ref))
        .collect();

    let mut buffer = String::new();
    let mut rng = crate::rng::Rand::with_seed();

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
                 // Reset policy buffer
                 current_policy = [0.0; 1858];
            } else if line.starts_with("Value:") {
                // We don't use LC0 value for now? 
                // The diff uses 'value.eval(&position.board())' which is MONTY's value net?
                // "distill... generating policy data (with value data stored as well ofc) from the raw network outputs"
                // Does "raw network outputs" imply LC0 Value? 
                // In diff: `let (win, draw, _) = value.eval(&position.board());` -> This is Monty's Value!
                // So we ignore LC0 value.
            } else if line.starts_with("Policy (Top > 1%):") {
                // Parse
                let content = &line["Policy (Top > 1%): ".len()..];
                for entry in content.split_whitespace() {
                    let parts: Vec<&str> = entry.split(':').collect();
                    if parts.len() == 2 {
                        let idx: usize = parts[0].parse().unwrap_or(0);
                        let prob: f32 = parts[1].parse().unwrap_or(0.0);
                        if idx < 1858 {
                            current_policy[idx] = prob;
                        }
                    }
                }
                
                // We assume this is the last piece of info for the current game
                // (Assuming strict output order: FEN -> Value -> Policy -> ---)
                // But wait, the loop for batch_size games...
                // Only execute logic after we read everything for one game.
                // We can trigger on "---" or implied sequence.
                // Or just: Policy line is unique per game. Process after it.
                if game_idx < BATCH_SIZE {
                    let game = &mut games[game_idx];
                    process_game(game, &current_policy, value, &dest, &stop, &mut rng, opts.policy_data, book_ref);
                    game_idx += 1;
                }
            }
        }
    }
    
    let _ = child.kill();
}

fn process_game(
    game: &mut GameRunner,
    policy_probs: &[f32; 1858],
    value: &ValueNetwork,
    dest: &Arc<Mutex<Destination>>,
    stop: &AtomicBool,
    rng: &mut crate::rng::Rand,
    output_policy: bool,
    book: Option<&crate::book::OpeningBook>,
) {
    let mut moves = Vec::new();
    game.position.map_legal_moves(|mov| moves.push(mov));

    if moves.is_empty() {
        game.reset(book);
        return;
    }

    let mut probs = Vec::with_capacity(moves.len());
    let mut max_val = f32::NEG_INFINITY;

    for mov in &moves {
        let idx = crate::lc0_mapping::get_lc0_index(mov);
        let p = if let Some(idx) = idx {
            policy_probs[idx]
        } else {
            0.0
        };
        
        probs.push((*mov, p));
        if p > max_val {
            max_val = p;
        }
    }

    let total: f32 = probs.iter().map(|(_, p)| p).sum();
    // Renormalize legal moves
    if total > 0.0 {
        for (_, p) in probs.iter_mut() {
            *p /= total;
        }
    } else {
        let uniform = 1.0 / probs.len() as f32;
        for (_, p) in probs.iter_mut() {
            *p = uniform;
        }
    }

    let mut dist = Vec::with_capacity(probs.len());
    for (mov, p) in &probs {
         let mf_move = montyformat::chess::Move::from(u16::from(*mov));
         let visits = (*p * 65535.0) as u32;
         dist.push((mf_move, visits));
    }

    let best_move: Move = if game.temp == 0.0 {
         probs.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0
    } else {
         let t = 1.0 / game.temp;
         let mut weights = Vec::with_capacity(probs.len());
         let mut w_total = 0.0;
         for (_, p) in &probs {
             let w = (*p as f64).powf(t as f64);
             weights.push(w);
             w_total += w;
         }
         
         let r = (rng.rand_int() as f64) / (u32::MAX as f64);
         
         let mut cumulative = 0.0;
         let mut chosen = probs.last().unwrap().0;
         for (i, weight) in weights.iter().enumerate() {
             cumulative += *weight;
             if cumulative / w_total > r {
                 chosen = probs[i].0;
                 break;
             }
         }
         chosen
    };

    let (win, draw, _) = value.eval(&game.position.board());
    let score = win + draw / 2.0;

    let mf_best_move = montyformat::chess::Move::from(u16::from(best_move));

    if output_policy {
        let search_data = SearchData::new(mf_best_move, score, Some(dist));
        game.policy_game.push(search_data);
    } else {
        game.value_game.push(game.position.stm(), mf_best_move, score);
    }

    game.searches += 1;
    game.iters += 1;
    game.temp *= 0.9;
    if game.temp <= 0.2 {
        game.temp = 0.0;
    }

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
        
        game.reset(book);
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
