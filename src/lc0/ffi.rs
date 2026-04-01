use std::ffi::{CString, c_char, c_float, c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::chess::Position;
use super::worker::Lc0Result;

type Lc0HandleRaw = *mut c_void;
type Lc0BatchHandleRaw = *mut c_void;

/// Wrapper to make the raw pointer Send+Sync.
/// Safety: lc0 Network objects are thread-safe for creating new computations.
#[derive(Clone, Copy)]
pub struct Lc0Handle(Lc0HandleRaw);
unsafe impl Send for Lc0Handle {}
unsafe impl Sync for Lc0Handle {}

impl Lc0Handle {
    fn raw(self) -> Lc0HandleRaw { self.0 }
    pub fn is_null(self) -> bool { self.0.is_null() }
}

extern "C" {
    fn lc0_init(weights_path: *const c_char, backend_name: *const c_char, chess960: c_int) -> Lc0HandleRaw;
    #[allow(dead_code)]
    fn lc0_destroy(handle: Lc0HandleRaw);
    fn lc0_new_batch(handle: Lc0HandleRaw) -> Lc0BatchHandleRaw;
    fn lc0_batch_add_fen(batch: Lc0BatchHandleRaw, fen: *const c_char) -> c_int;
    fn lc0_batch_compute(batch: Lc0BatchHandleRaw);
    fn lc0_batch_get_q(batch: Lc0BatchHandleRaw, sample_idx: c_int) -> c_float;
    fn lc0_batch_get_d(batch: Lc0BatchHandleRaw, sample_idx: c_int) -> c_float;
    fn lc0_batch_get_num_moves(batch: Lc0BatchHandleRaw, sample_idx: c_int) -> c_int;
    fn lc0_batch_get_moves(batch: Lc0BatchHandleRaw, sample_idx: c_int, out_indices: *mut c_int, out_logits: *mut c_float);
    fn lc0_free_batch(batch: Lc0BatchHandleRaw);
}

/// Initialize a single lc0 network. Returns a shared handle that can be
/// used by multiple workers concurrently (one network copy on GPU).
pub fn lc0_init_shared(network_path: &str, backend: &str, chess960: bool) -> Lc0Handle {
    let weights = CString::new(network_path).expect("invalid network path");
    let backend_c = CString::new(backend).expect("invalid backend name");

    let raw = unsafe {
        lc0_init(weights.as_ptr(), backend_c.as_ptr(), chess960 as c_int)
    };

    if raw.is_null() {
        panic!("Failed to initialize lc0 via FFI. Check network path and backend.");
    }

    Lc0Handle(raw)
}

fn run_batch(context: Lc0Handle, fens: &[String]) -> Vec<Lc0Result> {
    let batch = unsafe { lc0_new_batch(context.raw()) };

    for fen in fens {
        let fen_c = CString::new(fen.as_str()).unwrap();
        unsafe { lc0_batch_add_fen(batch, fen_c.as_ptr()) };
    }

    unsafe { lc0_batch_compute(batch) };

    let mut results = Vec::with_capacity(fens.len());
    for i in 0..fens.len() {
        let idx = i as c_int;
        let value = unsafe { lc0_batch_get_q(batch, idx) };
        let draw = unsafe { lc0_batch_get_d(batch, idx) };

        let num_moves = unsafe { lc0_batch_get_num_moves(batch, idx) } as usize;
        let mut indices = vec![0i32; num_moves];
        let mut logits = vec![0.0f32; num_moves];
        unsafe {
            lc0_batch_get_moves(batch, idx, indices.as_mut_ptr(), logits.as_mut_ptr());
        }
        let policy_logits: Vec<(usize, f32)> = indices
            .iter()
            .zip(logits.iter())
            .map(|(&i, &l)| (i as usize, l))
            .collect();

        results.push(Lc0Result {
            value,
            draw,
            policy_logits,
        });
    }

    unsafe { lc0_free_batch(batch) };

    results
}

/// FFI-based lc0 worker with a persistent background thread.
/// Multiple workers share a single Lc0Handle (one network copy on GPU).
pub struct Lc0FfiWorker {
    work_tx: mpsc::Sender<Vec<Position>>,
    result_cache: Arc<Mutex<Option<Vec<Lc0Result>>>>,
    busy: AtomicBool,
    _thread: JoinHandle<()>,
}

impl Lc0FfiWorker {
    /// Create a worker with a persistent thread that shares the given network handle.
    pub fn new(shared_context: Lc0Handle) -> Self {
        let (work_tx, work_rx) = mpsc::channel::<Vec<Position>>();
        let result_cache: Arc<Mutex<Option<Vec<Lc0Result>>>> = Arc::new(Mutex::new(None));
        let cache_clone = result_cache.clone();

        let handle = thread::spawn(move || {
            while let Ok(positions) = work_rx.recv() {
                let fens: Vec<String> = positions.iter().map(|p| p.as_fen()).collect();
                let results = run_batch(shared_context, &fens);
                *cache_clone.lock().unwrap() = Some(results);
            }
        });

        Self {
            work_tx,
            result_cache,
            busy: AtomicBool::new(false),
            _thread: handle,
        }
    }

    pub fn wait_ready(&self) {
        // FFI init is synchronous - if we got here, we're ready
    }

    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Relaxed)
    }

    pub fn set_busy(&self, val: bool) {
        self.busy.store(val, Ordering::Relaxed);
    }

    /// Send a batch of positions to the persistent worker thread.
    pub fn send_batch(&self, positions: &[Position]) {
        self.busy.store(true, Ordering::Relaxed);
        let _ = self.work_tx.send(positions.to_vec());
    }

    pub fn try_recv(&self) -> Option<Vec<Lc0Result>> {
        self.result_cache.try_lock().ok()?.take()
    }

    pub fn kill(&self) {
        // Thread exits when work_tx is dropped
    }
}
