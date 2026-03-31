use std::{
    io::{BufRead, BufReader, Write},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
};

#[derive(Debug)]
pub struct Lc0Result {
    pub value: f32,
    pub draw: f32,
    pub policy_logits: Vec<(usize, f32)>,
}

pub struct Lc0Worker {
    child: Child,
    stdin: std::process::ChildStdin,
    result_slot: Arc<Mutex<Option<Vec<Lc0Result>>>>,
    _reader_thread: JoinHandle<()>,
    batch_size: usize,
    pub busy: bool,
}

impl Lc0Worker {
    pub fn spawn(
        network_path: &str,
        backend: &str,
        batch_size: usize,
        chess960: bool,
    ) -> Self {
        let exe_path = if cfg!(target_os = "windows") {
            "./lc0_inference_standalone/lc0_inference.exe"
        } else {
            "./lc0_inference_standalone/build/release/lc0_inference"
        };

        let mut command = Command::new(exe_path);
        command.arg(network_path).arg(batch_size.to_string());

        if chess960 {
            command.arg("--chess960");
        }

        if !backend.is_empty() {
            command.arg("--backend").arg(backend);
        }

        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("Failed to spawn lc0_inference. Make sure it is compiled.");

        let stdin = child.stdin.take().expect("Failed to open stdin");
        let stdout = child.stdout.take().expect("Failed to open stdout");

        let result_slot: Arc<Mutex<Option<Vec<Lc0Result>>>> = Arc::new(Mutex::new(None));
        let slot_clone = result_slot.clone();

        let reader_thread = thread::spawn(move || {
            reader_loop(stdout, slot_clone);
        });

        Self {
            child,
            stdin,
            result_slot,
            _reader_thread: reader_thread,
            batch_size,
            busy: false,
        }
    }

    pub fn send_batch(&mut self, fens: &[String]) {
        assert!(fens.len() <= self.batch_size);

        // Pad to batch_size with startpos if needed (lc0_inference expects exactly batch_size lines)
        for fen in fens {
            writeln!(self.stdin, "{}", fen).expect("Failed to write to lc0 stdin");
        }
        for _ in fens.len()..self.batch_size {
            writeln!(self.stdin, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                .expect("Failed to write padding to lc0 stdin");
        }
        self.stdin.flush().expect("Failed to flush lc0 stdin");
        self.busy = true;
    }

    pub fn try_recv(&self) -> Option<Vec<Lc0Result>> {
        let mut slot = self.result_slot.try_lock().ok()?;
        slot.take()
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
    }
}

impl Drop for Lc0Worker {
    fn drop(&mut self) {
        self.kill();
    }
}

fn reader_loop(
    stdout: std::process::ChildStdout,
    result_slot: Arc<Mutex<Option<Vec<Lc0Result>>>>,
) {
    let mut reader = BufReader::new(stdout);
    let mut buffer = String::new();

    loop {
        let mut batch_results: Vec<Lc0Result> = Vec::new();
        let mut current_value = 0.0f32;
        let mut current_draw = 0.0f32;
        let mut current_logits: Vec<(usize, f32)> = Vec::new();
        let mut in_position = false;

        loop {
            buffer.clear();
            let bytes = reader.read_line(&mut buffer).unwrap_or(0);
            if bytes == 0 {
                // EOF - process died
                return;
            }

            let line = buffer.trim();

            if line == "BATCH_DONE" {
                // Finalize last position if any
                if in_position {
                    batch_results.push(Lc0Result {
                        value: current_value,
                        draw: current_draw,
                        policy_logits: std::mem::take(&mut current_logits),
                    });
                }
                break;
            }

            if line.starts_with("FEN:") {
                // Save previous position if any
                if in_position {
                    batch_results.push(Lc0Result {
                        value: current_value,
                        draw: current_draw,
                        policy_logits: std::mem::take(&mut current_logits),
                    });
                }
                current_value = 0.0;
                current_draw = 0.0;
                current_logits.clear();
                in_position = true;
            } else if line.starts_with("Value:") {
                if let Some(val_str) = line.split_whitespace().nth(1) {
                    current_value = val_str.parse().unwrap_or(0.0);
                }
            } else if line.starts_with("Draw:") {
                if let Some(val_str) = line.split_whitespace().nth(1) {
                    current_draw = val_str.parse().unwrap_or(0.0);
                }
            } else if line.starts_with("Policy (Logits):") {
                let content = line.trim_start_matches("Policy (Logits):").trim();
                for token in content.split_whitespace() {
                    if let Some((idx_str, val_str)) = token.split_once(':') {
                        if let (Ok(idx), Ok(val)) =
                            (idx_str.parse::<usize>(), val_str.parse::<f32>())
                        {
                            if idx < 1858 {
                                current_logits.push((idx, val));
                            }
                        }
                    }
                }
            }
            // Ignore separator lines and other output
        }

        if !batch_results.is_empty() {
            // Spin until the slot is empty (previous batch consumed)
            loop {
                if let Ok(mut slot) = result_slot.lock() {
                    if slot.is_none() {
                        *slot = Some(batch_results);
                        break;
                    }
                    // Previous batch not yet consumed, drop lock and retry
                    drop(slot);
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
            }
        }
    }
}
