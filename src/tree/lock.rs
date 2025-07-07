use parking_lot::{Mutex, MutexGuard};
use std::sync::atomic::{AtomicU32, Ordering};

use super::NodePtr;

#[derive(Debug)]
pub struct CustomLock {
    value: AtomicU32,
    lock: Mutex<()>,
}

pub struct WriteGuard<'a> {
    lock: &'a CustomLock,
    _guard: MutexGuard<'a, ()>,
}

impl<'a> WriteGuard<'a> {
    pub fn val(&self) -> NodePtr {
        NodePtr::from_raw(self.lock.value.load(Ordering::Acquire))
    }

    pub fn store(&mut self, val: NodePtr) {
        self.lock.value.store(val.inner(), Ordering::Release);
    }
}

impl CustomLock {
    pub fn new(val: NodePtr) -> Self {
        Self {
            value: AtomicU32::new(val.inner()),
            lock: Mutex::new(()),
        }
    }

    pub fn read(&self) -> NodePtr {
        NodePtr::from_raw(self.value.load(Ordering::Acquire))
    }

    pub fn write(&self) -> WriteGuard<'_> {
        WriteGuard {
            lock: self,
            _guard: self.lock.lock(),
        }
    }
}
