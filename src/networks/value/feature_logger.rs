use once_cell::sync::Lazy;
use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::Write,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use super::threats;

static LOGGER: Lazy<ValueFeatureLogger> = Lazy::new(ValueFeatureLogger::new);

pub fn begin_search_logging() {
    LOGGER.begin();
}

pub fn finalize_logging_request() {
    LOGGER.finish();
}

pub fn request_feature_log() {
    LOGGER.request_log();
}

pub fn record_features(features: &[usize]) {
    LOGGER.record(features);
}

struct ValueFeatureLogger {
    enabled: AtomicBool,
    log_requested: AtomicBool,
    eval_counter: AtomicU64,
    global_xor: AtomicU64,
    global_sum: AtomicU64,
    counts: Vec<AtomicU64>,
    xor_fingerprints: Vec<AtomicU64>,
    sum_fingerprints: Vec<AtomicU64>,
}

impl ValueFeatureLogger {
    fn new() -> Self {
        let len = threats::TOTAL;
        Self {
            enabled: AtomicBool::new(false),
            log_requested: AtomicBool::new(false),
            eval_counter: AtomicU64::new(0),
            global_xor: AtomicU64::new(0),
            global_sum: AtomicU64::new(0),
            counts: (0..len).map(|_| AtomicU64::new(0)).collect(),
            xor_fingerprints: (0..len).map(|_| AtomicU64::new(0)).collect(),
            sum_fingerprints: (0..len).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    fn begin(&self) {
        self.enabled.store(true, Ordering::Relaxed);
        self.log_requested.store(false, Ordering::Relaxed);
        self.eval_counter.store(0, Ordering::Relaxed);
        self.global_xor.store(0, Ordering::Relaxed);
        self.global_sum.store(0, Ordering::Relaxed);

        for atomic in self
            .counts
            .iter()
            .chain(self.xor_fingerprints.iter())
            .chain(self.sum_fingerprints.iter())
        {
            atomic.store(0, Ordering::Relaxed);
        }
    }

    fn record(&self, features: &[usize]) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        let eval_id = self.eval_counter.fetch_add(1, Ordering::Relaxed);
        let hash = splitmix64(eval_id);
        let sum_hash = splitmix64(hash);

        self.global_xor.fetch_xor(hash, Ordering::Relaxed);
        self.global_sum.fetch_add(sum_hash, Ordering::Relaxed);

        for &feature in features {
            if let Some((count, xor_fp, sum_fp)) = self.get_feature_entries(feature) {
                count.fetch_add(1, Ordering::Relaxed);
                xor_fp.fetch_xor(hash, Ordering::Relaxed);
                sum_fp.fetch_add(sum_hash, Ordering::Relaxed);
            }
        }
    }

    fn get_feature_entries(&self, feature: usize) -> Option<(&AtomicU64, &AtomicU64, &AtomicU64)> {
        self.counts.get(feature).and_then(|count| {
            Some((
                count,
                &self.xor_fingerprints[feature],
                &self.sum_fingerprints[feature],
            ))
        })
    }

    fn request_log(&self) {
        self.log_requested.store(true, Ordering::Relaxed);
    }

    fn finish(&self) {
        if !self.enabled.swap(false, Ordering::Relaxed) {
            return;
        }

        if !self.log_requested.swap(false, Ordering::Relaxed) {
            return;
        }

        let total_evals = self.eval_counter.load(Ordering::Relaxed);
        if total_evals == 0 {
            return;
        }

        let global_xor = self.global_xor.load(Ordering::Relaxed);
        let global_sum = self.global_sum.load(Ordering::Relaxed);

        let mut active_features = Vec::new();

        for idx in 0..self.counts.len() {
            let count = self.counts[idx].load(Ordering::Relaxed);
            if count == 0 {
                continue;
            }

            active_features.push(FeatureRecord {
                idx,
                count,
                xor: self.xor_fingerprints[idx].load(Ordering::Relaxed),
                sum: self.sum_fingerprints[idx].load(Ordering::Relaxed),
            });
        }

        if active_features.is_empty() {
            return;
        }

        let correlated = find_correlated(&active_features);
        let anti_correlated =
            find_anti_correlated(&active_features, total_evals, global_xor, global_sum);

        if let Err(err) = write_log(total_evals, &active_features, &correlated, &anti_correlated) {
            eprintln!("Failed to write value feature log: {err}");
        }
    }
}

struct FeatureRecord {
    idx: usize,
    count: u64,
    xor: u64,
    sum: u64,
}

fn find_correlated(features: &[FeatureRecord]) -> Vec<Vec<usize>> {
    let mut map: HashMap<(u64, u64, u64), Vec<usize>> = HashMap::new();

    for record in features {
        map.entry((record.count, record.xor, record.sum))
            .or_default()
            .push(record.idx);
    }

    map.into_values().filter(|group| group.len() > 1).collect()
}

fn find_anti_correlated(
    features: &[FeatureRecord],
    total_evals: u64,
    total_xor: u64,
    total_sum: u64,
) -> Vec<(usize, usize)> {
    let mut map: HashMap<(u64, u64, u64), Vec<usize>> = HashMap::new();
    for record in features {
        map.entry((record.count, record.xor, record.sum))
            .or_default()
            .push(record.idx);
    }

    let mut pairs = Vec::new();

    for record in features {
        let complement_key = (
            total_evals - record.count,
            total_xor ^ record.xor,
            total_sum.wrapping_sub(record.sum),
        );

        if let Some(candidates) = map.get(&complement_key) {
            for &other in candidates {
                if other > record.idx {
                    pairs.push((record.idx, other));
                }
            }
        }
    }

    pairs
}

fn write_log(
    total_evals: u64,
    features: &[FeatureRecord],
    correlated: &[Vec<usize>],
    anti_correlated: &[(usize, usize)],
) -> std::io::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("value_feature_stats.log")?;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    writeln!(
        file,
        "==== Value feature statistics (timestamp: {timestamp}, evaluations: {total_evals}) ===="
    )?;

    writeln!(file, "Feature activation counts:")?;
    for record in features {
        writeln!(file, "feature {0}: {1}", record.idx, record.count)?;
    }

    writeln!(file, "Perfectly correlated feature groups:")?;
    if correlated.is_empty() {
        writeln!(file, "  (none)")?;
    } else {
        for group in correlated {
            writeln!(file, "  {:?}", group)?;
        }
    }

    writeln!(file, "Perfectly anti-correlated feature pairs:")?;
    if anti_correlated.is_empty() {
        writeln!(file, "  (none)")?;
    } else {
        for (a, b) in anti_correlated {
            writeln!(file, "  ({a}, {b})")?;
        }
    }

    writeln!(file)?;
    Ok(())
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
