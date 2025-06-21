use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use once_cell::sync::Lazy;
use std::sync::Mutex;

use crate::chess::{ChessState, Move};
use crate::tree::Tree;

pub struct UciEngine {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl UciEngine {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;

        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());

        Ok(Self { _child: child, stdin, stdout })
    }

    fn send(&mut self, cmd: &str) -> std::io::Result<()> {
        self.stdin.write_all(cmd.as_bytes())?;
        self.stdin.flush()
    }

    /// Request policies for the given FEN position.
    pub fn root_policies(&mut self, fen: &str) -> std::io::Result<Vec<(String, f32)>> {
        self.send(&format!("position fen {}\n", fen))?;
        self.send("go nodes 1\n")?;

        let mut line = String::new();
        let mut policies = Vec::new();
        loop {
            line.clear();
            if self.stdout.read_line(&mut line)? == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.starts_with("bestmove") {
                break;
            }
            if let Some(rest) = trimmed.strip_prefix("info string") {
                let mut parts = rest.trim().split_whitespace();
                if let Some(mov) = parts.next() {
                    if let Some(pidx) = trimmed.find("(P:") {
                        if let Some(val) = trimmed[pidx + 3..].split('%').next() {
                            if let Ok(p) = val.trim().trim_start_matches(':').trim().parse::<f32>() {
                                policies.push((mov.to_string(), p / 100.0));
                            }
                        }
                    }
                }
            }
        }
        Ok(policies)
    }
}

pub static ENGINE: Lazy<Mutex<Option<UciEngine>>> = Lazy::new(|| {
    if let Ok(path) = std::env::var("UCI_ENGINE_PATH") {
        match UciEngine::new(&path) {
            Ok(mut eng) => {
                // initialise engine
                let _ = eng.send("uci\n");
                let mut line = String::new();
                while eng.stdout.read_line(&mut line).ok().filter(|&n| n > 0).is_some() {
                    if line.trim() == "uciok" {
                        break;
                    }
                    line.clear();
                }
                Mutex::new(Some(eng))
            }
            Err(e) => {
                eprintln!("failed to launch engine: {e}");
                Mutex::new(None)
            }
        }
    } else {
        Mutex::new(None)
    }
});

fn parse_move(pos: &ChessState, mv: &str) -> Option<Move> {
    let mut res = None;
    pos.map_legal_moves(|m| {
        if mv == pos.conv_mov_to_str(m) {
            res = Some(m);
        }
    });
    res
}

pub fn apply_root_policy(tree: &Tree, pos: &ChessState) {
    let mut guard = ENGINE.lock().unwrap();
    let engine = match guard.as_mut() {
        Some(e) => e,
        None => return,
    };

    let fen = pos.as_fen();
    let Ok(policies) = engine.root_policies(&fen) else { return; };

    if policies.is_empty() {
        return;
    }

    let mut map: HashMap<u16, f32> = HashMap::new();
    for (mstr, p) in policies {
        if let Some(mv) = parse_move(pos, &mstr) {
            map.insert(u16::from(mv), p);
        }
    }

    if map.is_empty() {
        return;
    }

    let node = tree.root_node();
    if !tree[node].has_children() {
        return;
    }

    let first = tree[node].actions();
    let num = tree[node].num_actions();
    let mut total = 0.0;
    for i in 0..num {
        let mv = tree[first + i].parent_move();
        if let Some(&p) = map.get(&u16::from(mv)) {
            total += p;
        }
    }
    if total == 0.0 {
        return;
    }
    let mut sum_sq: f32 = 0.0;
    for i in 0..num {
        let mv = tree[first + i].parent_move();
        let mut p = map.get(&u16::from(mv)).copied().unwrap_or(0.0);
        p /= total;
        tree[first + i].set_policy(p);
        sum_sq += p * p;
    }
    let gini = (1.0f32 - sum_sq).clamp(0.0, 1.0);
    tree[node].set_gini_impurity(gini);
}