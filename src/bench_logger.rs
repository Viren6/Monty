use once_cell::sync::OnceCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Mutex;

static LOGGER: OnceCell<Mutex<BufWriter<File>>> = OnceCell::new();

pub fn init() {
    if LOGGER.get().is_none() {
        let file = File::create("bench_policy.csv").expect("failed to create bench_policy.csv");
        let mut writer = BufWriter::new(file);
        writeln!(writer, "good see policy %,bad see policy %").unwrap();
        LOGGER.set(Mutex::new(writer)).ok();
    }
}

pub fn log(good: f32, bad: f32) {
    if let Some(m) = LOGGER.get() {
        let mut writer = m.lock().unwrap();
        writeln!(writer, "{:.6},{:.6}", good * 100.0, bad * 100.0).ok();
    }
}
