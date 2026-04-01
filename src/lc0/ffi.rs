use std::ffi::{CString, c_char, c_float, c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

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

#[repr(C)]
struct Lc0ResultRaw {
    value: c_float,
    draw: c_float,
    num_moves: c_int,
    move_indices: *mut c_int,
    move_logits: *mut c_float,
}

extern "C" {
    fn lc0_init(weights_path: *const c_char, backend_name: *const c_char, chess960: c_int) -> Lc0HandleRaw;
    #[allow(dead_code)]
    fn lc0_destroy(handle: Lc0HandleRaw);
    fn lc0_new_batch(handle: Lc0HandleRaw) -> Lc0BatchHandleRaw;
    fn lc0_batch_add_fen(batch: Lc0BatchHandleRaw, fen: *const c_char) -> c_int;
    fn lc0_batch_compute(batch: Lc0BatchHandleRaw);
    fn lc0_batch_get_result(batch: Lc0BatchHandleRaw, sample_idx: c_int) -> Lc0ResultRaw;
    fn lc0_free_result(result: *mut Lc0ResultRaw);
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

/// FFI-based lc0 worker. Multiple workers share a single Lc0Handle
/// (one network copy on GPU), each creating independent batch computations.
pub struct Lc0FfiWorker {
    context: Lc0Handle,
    busy: AtomicBool,
    result_cache: Arc<Mutex<Option<Vec<Lc0Result>>>>,
}

impl Lc0FfiWorker {
    /// Create a worker that shares the given network handle.
    pub fn new(shared_context: Lc0Handle) -> Self {
        Self {
            context: shared_context,
            busy: AtomicBool::new(false),
            result_cache: Arc::new(Mutex::new(None)),
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

    /// Send a batch of FENs for inference on a background thread.
    /// Results are cached and retrievable via try_recv().
    pub fn send_batch(&self, fens: &[String]) {
        self.busy.store(true, Ordering::Relaxed);

        let context = self.context;
        let fens_owned: Vec<String> = fens.to_vec();
        let result_cache = self.result_cache.clone();

        std::thread::spawn(move || {
            let batch = unsafe { lc0_new_batch(context.raw()) };

            for fen in &fens_owned {
                let fen_c = CString::new(fen.as_str()).unwrap();
                unsafe { lc0_batch_add_fen(batch, fen_c.as_ptr()) };
            }

            unsafe { lc0_batch_compute(batch) };

            let mut results = Vec::with_capacity(fens_owned.len());
            for i in 0..fens_owned.len() {
                let mut raw = unsafe { lc0_batch_get_result(batch, i as c_int) };

                let mut policy_logits = Vec::with_capacity(raw.num_moves as usize);
                if raw.num_moves > 0 && !raw.move_indices.is_null() && !raw.move_logits.is_null() {
                    for j in 0..raw.num_moves as usize {
                        unsafe {
                            let idx = *raw.move_indices.add(j) as usize;
                            let logit = *raw.move_logits.add(j);
                            policy_logits.push((idx, logit));
                        }
                    }
                }

                unsafe { lc0_free_result(&mut raw) };

                results.push(Lc0Result {
                    value: raw.value,
                    draw: raw.draw,
                    policy_logits,
                });
            }

            unsafe { lc0_free_batch(batch) };

            *result_cache.lock().unwrap() = Some(results);
        });
    }

    pub fn try_recv(&self) -> Option<Vec<Lc0Result>> {
        self.result_cache.try_lock().ok()?.take()
    }

    pub fn kill(&self) {
        // No-op for FFI
    }
}
