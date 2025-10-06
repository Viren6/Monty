use std::{env, fmt, process, sync::atomic::AtomicBool};

use monty::{
    chess::{ChessState, GameState, Move},
    mcts::{Limits, MctsParams, Searcher},
    networks::{self, PolicyNetwork, ValueNetwork},
    read_into_struct_unchecked,
    tree::Tree,
};

#[derive(Debug, Clone)]
struct Config {
    games: usize,
    nodes: usize,
    hash_mb: usize,
    threads: usize,
    random_plies: usize,
    max_game_plies: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            games: 10,
            nodes: 2_000,
            hash_mb: 1,
            threads: 1,
            random_plies: 8,
            max_game_plies: 1024,
        }
    }
}

impl Config {
    fn parse() -> Self {
        let mut cfg = Self::default();
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--games" => {
                    cfg.games = parse_required_value(arg.as_str(), args.next());
                }
                "--nodes" => {
                    cfg.nodes = parse_required_value(arg.as_str(), args.next());
                }
                "--hash-mb" => {
                    cfg.hash_mb = parse_required_value(arg.as_str(), args.next());
                    if cfg.hash_mb == 0 {
                        eprintln!("--hash-mb must be at least 1");
                        process::exit(1);
                    }
                }
                "--threads" => {
                    cfg.threads = parse_required_value(arg.as_str(), args.next());
                    if cfg.threads == 0 {
                        eprintln!("--threads must be at least 1");
                        process::exit(1);
                    }
                }
                "--random-plies" => {
                    cfg.random_plies = parse_required_value(arg.as_str(), args.next());
                }
                "--max-plies" => {
                    cfg.max_game_plies = parse_required_value(arg.as_str(), args.next());
                    if cfg.max_game_plies == 0 {
                        eprintln!("--max-plies must be at least 1");
                        process::exit(1);
                    }
                }
                "--help" | "-h" => {
                    print_usage();
                    process::exit(0);
                }
                unknown => {
                    eprintln!("Unknown argument: {unknown}\n");
                    print_usage();
                    process::exit(1);
                }
            }
        }

        cfg
    }
}

fn parse_required_value<T: std::str::FromStr>(flag: &str, value: Option<String>) -> T {
    match value {
        Some(v) => v.parse().unwrap_or_else(|_| {
            eprintln!("Invalid value for {flag}: {v}");
            process::exit(1);
        }),
        None => {
            eprintln!("Missing value for {flag}");
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("Usage: cargo run --release --bin internal_match -- [options]\n");
    eprintln!("Options:");
    eprintln!("  --games <n>          Number of games to play (default: 20)");
    eprintln!("  --nodes <n>          Node budget per move (default: 2000)");
    eprintln!("  --hash-mb <n>        Hash/table size per engine in MiB (default: 1)");
    eprintln!("  --threads <n>        Threads per search (default: 1)");
    eprintln!("  --random-plies <n>   Random plies before each game (default: 8)");
    eprintln!("  --max-plies <n>      Maximum plies before declaring a draw (default: 1024)");
}

struct EngineState {
    tree: Tree,
    params: MctsParams,
    nodes: usize,
}

impl EngineState {
    fn new(hash_mb: usize, threads: usize) -> Self {
        Self {
            tree: Tree::new_mb(hash_mb, threads),
            params: MctsParams::default(),
            nodes: 0,
        }
    }

    fn reset(&mut self, threads: usize) {
        self.tree.clear(threads);
        self.nodes = 0;
    }
}

enum GameOutcome {
    White,
    Black,
    Draw,
}

impl fmt::Display for GameOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameOutcome::White => write!(f, "1-0"),
            GameOutcome::Black => write!(f, "0-1"),
            GameOutcome::Draw => write!(f, "1/2-1/2"),
        }
    }
}

struct SimpleRng(u64);

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let seed = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
        self.0 = x;
        x
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        assert!(upper > 0);
        (self.next_u64() % upper as u64) as usize
    }
}

fn main() {
    let cfg = Config::parse();

    let policy_mapped = unsafe { read_into_struct_unchecked(networks::PolicyFileDefaultName) };
    let value_mapped = unsafe { read_into_struct_unchecked(networks::ValueFileDefaultName) };

    let policy = policy_mapped.data;
    let value = value_mapped.data;

    println!(
        "Playing {} games | nodes={} hash={}MiB threads={} random_plies={}",
        cfg.games, cfg.nodes, cfg.hash_mb, cfg.threads, cfg.random_plies
    );

    let mut engines = [
        EngineState::new(cfg.hash_mb, cfg.threads),
        EngineState::new(cfg.hash_mb, cfg.threads),
    ];

    let mut score = [0.0f32; 2];

    for game_idx in 0..cfg.games {
        engines[0].reset(cfg.threads);
        engines[1].reset(cfg.threads);

        let white_idx = if game_idx % 2 == 0 { 0 } else { 1 };
        let black_idx = 1 - white_idx;

        let (outcome, plies, opening_moves) = play_single_game(
            game_idx,
            &cfg,
            &mut engines,
            [white_idx, black_idx],
            policy,
            value,
        );

        match outcome {
            GameOutcome::White => {
                score[white_idx] += 1.0;
            }
            GameOutcome::Black => {
                score[black_idx] += 1.0;
            }
            GameOutcome::Draw => {
                score[white_idx] += 0.5;
                score[black_idx] += 0.5;
            }
        }

        let opening_desc = if opening_moves.is_empty() {
            "-".to_string()
        } else {
            opening_moves.join(" ")
        };

        println!(
            "Game {:>3}: {:<7} | plies={:<3} | opening={opening_desc}",
            game_idx + 1,
            outcome,
            plies,
        );
    }

    println!(
        "Final score: Engine A {:.1} - Engine B {:.1}",
        score[0], score[1]
    );
}

fn play_single_game(
    game_idx: usize,
    cfg: &Config,
    engines: &mut [EngineState; 2],
    color_to_engine: [usize; 2],
    policy: &PolicyNetwork,
    value: &ValueNetwork,
) -> (GameOutcome, usize, Vec<String>) {
    let mut pos = ChessState::default();
    let mut rng = SimpleRng::new(0x9e37_79b9_7f4a_7c15u64.wrapping_mul((game_idx as u64) + 1));

    let mut opening_moves = Vec::new();
    for _ in 0..cfg.random_plies {
        if !matches!(pos.game_state(), GameState::Ongoing) {
            break;
        }

        let legal = legal_moves(&pos);
        if legal.is_empty() {
            break;
        }

        let choice = legal[rng.gen_range(legal.len())];
        opening_moves.push(pos.conv_mov_to_str(choice));
        pos.make_move(choice);
    }

    let mut plies_played = opening_moves.len();

    loop {
        if let Some(result) = terminal_result(&pos) {
            return (result, plies_played, opening_moves);
        }

        if plies_played >= cfg.max_game_plies {
            return (GameOutcome::Draw, plies_played, opening_moves);
        }

        let side = pos.stm();
        let engine_idx = color_to_engine[side];

        let best_move = match search_best_move(&mut engines[engine_idx], &pos, cfg, policy, value) {
            Some(m) => m,
            None => {
                return (GameOutcome::Draw, plies_played, opening_moves);
            }
        };

        if plies_played < cfg.random_plies {
            opening_moves.push(pos.conv_mov_to_str(best_move));
        }

        pos.make_move(best_move);
        plies_played += 1;
    }
}

fn legal_moves(pos: &ChessState) -> Vec<Move> {
    let mut moves = Vec::new();
    pos.map_legal_moves(|m| moves.push(m));
    moves
}

fn terminal_result(pos: &ChessState) -> Option<GameOutcome> {
    match pos.game_state() {
        GameState::Draw => Some(GameOutcome::Draw),
        GameState::Lost(_) => {
            if pos.stm() == 0 {
                Some(GameOutcome::Black)
            } else {
                Some(GameOutcome::White)
            }
        }
        GameState::Ongoing | GameState::Won(_) => None,
    }
}

fn search_best_move(
    engine: &mut EngineState,
    pos: &ChessState,
    cfg: &Config,
    policy: &PolicyNetwork,
    value: &ValueNetwork,
) -> Option<Move> {
    engine.tree.set_root_position(pos);

    let limits = Limits {
        max_time: None,
        opt_time: None,
        max_depth: 256,
        max_nodes: cfg.nodes,
        #[cfg(feature = "datagen")]
        kld_min_gain: None,
    };

    let abort = AtomicBool::new(false);
    let searcher = Searcher::new(&engine.tree, &engine.params, policy, value, &abort);

    let (best_move, _) = searcher.search(
        cfg.threads,
        limits,
        false,
        &mut engine.nodes,
        #[cfg(feature = "datagen")]
        false,
        #[cfg(feature = "datagen")]
        1.0,
    );

    if best_move == Move::NULL {
        None
    } else {
        Some(best_move)
    }
}