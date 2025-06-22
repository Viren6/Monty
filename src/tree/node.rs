use std::{
    ops::Add,
    sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering},
};

use crate::chess::{GameState, Move};

use super::lock::{CustomLock, WriteGuard};

const QUANT: i32 = 16384 * 4;

#[cfg(debug_assertions)]
const CANARY: u64 = 0xDEAD_CAFE_F00D_FACE;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodePtr(u32);

impl NodePtr {
    pub const NULL: Self = Self(u32::MAX);

    pub fn is_null(self) -> bool {
        self == Self::NULL
    }

    pub fn new(half: bool, idx: u32) -> Self {
        #[cfg(debug_assertions)]
        {
            const IDX_MASK: u32 = (1 << 8) - 1;
            Self((u32::from(half) << 31) | (idx & IDX_MASK))
        }
        #[cfg(not(debug_assertions))]
        {
            Self((u32::from(half) << 31) | idx)
        }
    }

    pub fn half(self) -> bool {
        self.0 & (1 << 31) > 0
    }

    pub fn idx(self) -> usize {
        (self.0 & 0x7FFFFFFF) as usize
    }

    pub fn inner(self) -> u32 {
        self.0
    }

    pub fn from_raw(inner: u32) -> Self {
        Self(inner)
    }
}

impl Add<usize> for NodePtr {
    type Output = NodePtr;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs as u32)
    }
}

#[derive(Debug)]
pub struct Node {
    #[cfg(debug_assertions)]
    canary: AtomicU64,
    #[cfg(debug_assertions)]
    debug_ptr: AtomicU32,
    #[cfg(debug_assertions)]
    debug_gen: AtomicU32,
    actions: CustomLock,
    num_actions: AtomicU8,
    state: AtomicU16,
    threads: AtomicU16,
    mov: AtomicU16,
    policy: AtomicU16,
    visits: AtomicU32,
    sum_q: AtomicU64,
    sum_sq_q: AtomicU64,
    gini_impurity: AtomicU8,
}

impl Node {
    pub fn new(state: GameState) -> Self {
        Node {
            #[cfg(debug_assertions)]
            canary: AtomicU64::new(CANARY),
            #[cfg(debug_assertions)]
            debug_ptr: AtomicU32::new(NodePtr::NULL.inner()),
            #[cfg(debug_assertions)]
            debug_gen: AtomicU32::new(0),
            actions: CustomLock::new(NodePtr::NULL),
            num_actions: AtomicU8::new(0),
            state: AtomicU16::new(u16::from(state)),
            threads: AtomicU16::new(0),
            mov: AtomicU16::new(0),
            policy: AtomicU16::new(0),
            visits: AtomicU32::new(0),
            sum_q: AtomicU64::new(0),
            sum_sq_q: AtomicU64::new(0),
            gini_impurity: AtomicU8::new(0),
        }
    }

    pub fn set_new(&self, mov: Move, policy: f32) {
        self.clear();
        self.mov.store(u16::from(mov), Ordering::Relaxed);
        self.set_policy(policy);
    }

    pub fn is_terminal(&self) -> bool {
        self.state() != GameState::Ongoing
    }

    pub fn num_actions(&self) -> usize {
        usize::from(self.num_actions.load(Ordering::Relaxed))
    }

    pub fn set_num_actions(&self, num: usize) {
        self.num_actions.store(num as u8, Ordering::Relaxed);
    }

    pub fn threads(&self) -> u16 {
        self.threads.load(Ordering::Relaxed)
    }

    pub fn visits(&self) -> u32 {
        self.visits.load(Ordering::Relaxed)
    }

    fn q64(&self) -> f64 {
        let visits = self.visits.load(Ordering::Relaxed);

        if visits == 0 {
            return 0.0;
        }

        let sum_q = self.sum_q.load(Ordering::Relaxed);

        (sum_q / u64::from(visits)) as f64 / f64::from(QUANT)
    }

    pub fn q(&self) -> f32 {
        self.q64() as f32
    }

    pub fn sq_q(&self) -> f64 {
        let sum_sq_q = self.sum_sq_q.load(Ordering::Relaxed);
        let visits = self.visits.load(Ordering::Relaxed);
        (sum_sq_q / u64::from(visits)) as f64 / f64::from(QUANT).powi(2)
    }

    pub fn var(&self) -> f32 {
        (self.sq_q() - self.q64().powi(2)).max(0.0) as f32
    }

    pub fn inc_threads(&self) {
        self.threads.fetch_add(1, Ordering::Relaxed);
    }

    pub fn dec_threads(&self) {
        self.threads.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn actions(&self) -> NodePtr {
        self.check_canary();
        self.actions.read()
    }

    pub fn actions_mut(&self) -> WriteGuard {
        self.check_canary();
        self.actions.write()
    }

    pub fn state(&self) -> GameState {
        self.check_canary();
        GameState::from(self.state.load(Ordering::Relaxed))
    }

    pub fn set_state(&self, state: GameState) {
        self.state.store(u16::from(state), Ordering::Relaxed);
    }

    pub fn policy(&self) -> f32 {
        self.check_canary();
        f32::from(self.policy.load(Ordering::Relaxed)) / f32::from(u16::MAX)
    }

    pub fn set_policy(&self, policy: f32) {
        self.policy
            .store((policy * f32::from(u16::MAX)) as u16, Ordering::Relaxed);
    }

    pub fn has_children(&self) -> bool {
        self.num_actions() != 0
    }

    pub fn is_not_expanded(&self) -> bool {
        self.state() == GameState::Ongoing && self.num_actions() == 0
    }

    pub fn gini_impurity(&self) -> f32 {
        f32::from(self.gini_impurity.load(Ordering::Relaxed)) / 255.0
    }

    pub fn set_gini_impurity(&self, gini_impurity: f32) {
        self.gini_impurity.store(
            (gini_impurity.clamp(0.0, 1.0) * 255.0) as u8,
            Ordering::Relaxed,
        );
    }

    pub fn clear_actions(&self) {
        self.actions.write().store(NodePtr::NULL);
        self.num_actions.store(0, Ordering::Relaxed);
    }

    pub fn parent_move(&self) -> Move {
        self.check_canary();
        Move::from(self.mov.load(Ordering::Relaxed))
    }

    pub fn copy_from(&self, other: &Self) {
        use std::sync::atomic::Ordering::Relaxed;

        self.threads.store(other.threads.load(Relaxed), Relaxed);
        self.mov.store(other.mov.load(Relaxed), Relaxed);
        self.policy.store(other.policy.load(Relaxed), Relaxed);
        self.state.store(other.state.load(Relaxed), Relaxed);
        self.gini_impurity
            .store(other.gini_impurity.load(Relaxed), Relaxed);
        self.visits.store(other.visits.load(Relaxed), Relaxed);
        self.sum_q.store(other.sum_q.load(Relaxed), Relaxed);
        self.sum_sq_q.store(other.sum_sq_q.load(Relaxed), Relaxed);
    }

    pub fn check_canary(&self) {
        #[cfg(debug_assertions)]
        {
            if self.canary.load(Ordering::Relaxed) != CANARY {
                panic!("Node at {:?} overwritten!", self as *const _);
            }
        }
    }

    #[cfg(debug_assertions)]
    pub fn mark_ptr(&self, ptr: NodePtr) {
        self.debug_ptr.store(ptr.inner(), Ordering::Relaxed);
    }

    #[cfg(debug_assertions)]
    pub fn mark_generation(&self, gen: u32) {
        self.debug_gen.store(gen, Ordering::Relaxed);
    }

    #[cfg(debug_assertions)]
    pub fn check_generation(&self, gen: u32) {
        let stored = self.debug_gen.load(Ordering::Relaxed);
        if stored != gen {
            panic!("Node generation mismatch: expected {} got {}", gen, stored);
        }
    }

    #[cfg(debug_assertions)]
    pub fn check_ptr(&self, ptr: NodePtr) {
        let stored = self.debug_ptr.load(Ordering::Relaxed);
        if stored != NodePtr::NULL.inner() && stored != ptr.inner() {
            panic!(
                "Node pointer mismatch: expected {} got {}",
                stored,
                ptr.inner()
            );
        }
    }

    pub fn clear(&self) {
        #[cfg(debug_assertions)]
        {
            self.canary.store(CANARY, Ordering::Relaxed);
            self.debug_ptr
                .store(NodePtr::NULL.inner(), Ordering::Relaxed);
            self.debug_gen.store(0, Ordering::Relaxed);
        }
        self.clear_actions();
        self.set_state(GameState::Ongoing);
        self.set_gini_impurity(0.0);
        self.visits.store(0, Ordering::Relaxed);
        self.sum_q.store(0, Ordering::Relaxed);
        self.sum_sq_q.store(0, Ordering::Relaxed);
        self.threads.store(0, Ordering::Relaxed);
    }

    pub fn update(&self, q: f32) -> f32 {
        self.check_canary();
        let q = (f64::from(q) * f64::from(QUANT)) as u64;
        let old_v = self.visits.fetch_add(1, Ordering::Relaxed);
        let old_q = self.sum_q.fetch_add(q, Ordering::Relaxed);
        self.sum_sq_q.fetch_add(q * q, Ordering::Relaxed);

        (((q + old_q) / u64::from(1 + old_v)) as f64 / f64::from(QUANT)) as f32
    }
}
