use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::cell::UnsafeCell;
use std::hint::spin_loop;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU8, Ordering};

const STATE_UNINITIALIZED: u8 = 0;
const STATE_INITIALIZING: u8 = 1;
const STATE_INITIALIZED: u8 = 2;

struct InitGuard<'a> {
    state: &'a AtomicU8,
}

impl<'a> InitGuard<'a> {
    fn new(state: &'a AtomicU8) -> Self {
        Self { state }
    }
}

impl<'a> Drop for InitGuard<'a> {
    fn drop(&mut self) {
        if self.state.load(Ordering::Acquire) == STATE_INITIALIZING {
            self.state.store(STATE_UNINITIALIZED, Ordering::Release);
        }
    }
}

pub struct NumaBuffer<T> {
    ptr: NonNull<UnsafeCell<MaybeUninit<T>>>,
    states: Box<[AtomicU8]>,
    len: usize,
    layout: Option<Layout>,
}

unsafe impl<T: Send> Send for NumaBuffer<T> {}
unsafe impl<T: Send> Sync for NumaBuffer<T> {}

impl<T> NumaBuffer<T> {
    pub fn new(len: usize) -> Self {
        let states = (0..len)
            .map(|_| AtomicU8::new(STATE_UNINITIALIZED))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                states,
                len,
                layout: None,
            };
        }

        let layout = Layout::array::<UnsafeCell<MaybeUninit<T>>>(len).unwrap();
        let raw = unsafe { alloc(layout) };
        if raw.is_null() {
            handle_alloc_error(layout);
        }

        Self {
            ptr: unsafe { NonNull::new_unchecked(raw.cast()) },
            states,
            len,
            layout: Some(layout),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    fn cell(&self, idx: usize) -> &UnsafeCell<MaybeUninit<T>> {
        debug_assert!(idx < self.len);
        unsafe { &*self.ptr.as_ptr().add(idx) }
    }

    unsafe fn assume_init_ref(&self, idx: usize) -> &T {
        (*self.cell(idx).get()).assume_init_ref()
    }

    /// Ensure that the entry at `idx` is initialized using `init` and return a shared reference to it.
    ///
    /// The initialization closure is executed on the calling thread, enabling first-touch
    /// NUMA placement semantics.
    pub fn init_with<F>(&self, idx: usize, init: F) -> &T
    where
        F: FnOnce() -> T,
    {
        match self.states[idx].load(Ordering::Acquire) {
            STATE_INITIALIZED => unsafe { return self.assume_init_ref(idx) },
            STATE_INITIALIZING => {
                while self.states[idx].load(Ordering::Acquire) != STATE_INITIALIZED {
                    spin_loop();
                }
                unsafe { return self.assume_init_ref(idx) };
            }
            _ => {}
        }

        if self.states[idx]
            .compare_exchange(
                STATE_UNINITIALIZED,
                STATE_INITIALIZING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            let guard = InitGuard::new(&self.states[idx]);
            let value = init();
            unsafe {
                (*self.cell(idx).get()).write(value);
            }
            self.states[idx].store(STATE_INITIALIZED, Ordering::Release);
            std::mem::forget(guard);
            unsafe { return self.assume_init_ref(idx) };
        }

        while self.states[idx].load(Ordering::Acquire) != STATE_INITIALIZED {
            spin_loop();
        }

        unsafe { self.assume_init_ref(idx) }
    }

    pub fn try_get(&self, idx: usize) -> Option<&T> {
        if idx >= self.len {
            return None;
        }

        if self.states[idx].load(Ordering::Acquire) == STATE_INITIALIZED {
            Some(unsafe { self.assume_init_ref(idx) })
        } else {
            None
        }
    }
}

impl<T> Drop for NumaBuffer<T> {
    fn drop(&mut self) {
        if self.len == 0 {
            return;
        }

        for idx in 0..self.len {
            if self.states[idx].load(Ordering::Acquire) == STATE_INITIALIZED {
                unsafe {
                    (*self.cell(idx).get()).assume_init_drop();
                }
            }
        }

        if let Some(layout) = self.layout {
            unsafe {
                dealloc(self.ptr.as_ptr().cast(), layout);
            }
        }
    }
}