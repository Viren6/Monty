use parking_lot::RawMutex;
use parking_lot::lock_api::RawMutex as RawMutexApi;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use super::NodePtr;

pub struct CustomLock {
    value: AtomicU32,
    write_locked: AtomicBool,
    lock: RawMutex,
}

pub struct WriteGuard<'a> {
    lock: &'a CustomLock,
}

impl Drop for WriteGuard<'_> {
    fn drop(&mut self) {
        // Release the write lock so readers see any writes performed
        self.lock.write_locked.store(false, Ordering::Release);
        unsafe { <RawMutex as RawMutexApi>::unlock(&self.lock.lock) };
    }
}

impl WriteGuard<'_> {
    pub fn val(&self) -> NodePtr {
        NodePtr::from_raw(self.lock.value.load(Ordering::Acquire))
    }

    pub fn store(&self, val: NodePtr) {
        self.lock.value.store(val.inner(), Ordering::Relaxed);
    }
}

impl CustomLock {
    pub fn new(val: NodePtr) -> Self {
        Self {
            value: AtomicU32::new(val.inner()),
            write_locked: AtomicBool::new(false),
            lock: <RawMutex as RawMutexApi>::INIT,
        }
    }

    pub fn read(&self) -> NodePtr {
        while self.write_locked.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        NodePtr::from_raw(self.value.load(Ordering::Acquire))
    }

    pub fn write(&self) -> WriteGuard<'_> {
        RawMutexApi::lock(&self.lock);
        self.write_locked.store(true, Ordering::Relaxed);
        WriteGuard { lock: self }
    }
}
