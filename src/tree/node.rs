use std::{
    ops::Add,
    sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, Ordering},
};

use crate::chess::{GameState, Move};

use super::lock::{CustomLock, WriteGuard};

const QUANT: i32 = 16384 * 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodePtr([u8; 6]);

impl NodePtr {
    const HALF_MASK: u64 = 1_u64 << 47;
    const INDEX_MASK: u64 = Self::HALF_MASK - 1;
    const STORED_MASK: u64 = (1_u64 << 48) - 1;

    const fn pack(bits: u64) -> [u8; 6] {
        [
            (bits & 0xFF) as u8,
            ((bits >> 8) & 0xFF) as u8,
            ((bits >> 16) & 0xFF) as u8,
            ((bits >> 24) & 0xFF) as u8,
            ((bits >> 32) & 0xFF) as u8,
            ((bits >> 40) & 0xFF) as u8,
        ]
    }

    fn bits(self) -> u64 {
        (self.0[0] as u64)
            | ((self.0[1] as u64) << 8)
            | ((self.0[2] as u64) << 16)
            | ((self.0[3] as u64) << 24)
            | ((self.0[4] as u64) << 32)
            | ((self.0[5] as u64) << 40)
    }

    pub const NULL: Self = Self(Self::pack(Self::STORED_MASK));

    pub fn is_null(self) -> bool {
        self == Self::NULL
    }

    pub fn new(half: bool, idx: usize) -> Self {
        let idx = idx as u64;
        assert!(
            idx <= Self::INDEX_MASK,
            "node index exceeds representable range"
        );
        let bits = (if half { Self::HALF_MASK } else { 0 }) | idx;
        Self(Self::pack(bits))
    }

    pub fn half(self) -> bool {
        self.bits() & Self::HALF_MASK > 0
    }

    pub fn idx(self) -> usize {
        debug_assert!(!self.is_null());
        (self.bits() & Self::INDEX_MASK) as usize
    }

    pub fn inner(self) -> u64 {
        self.bits()
    }

    pub fn from_raw(inner: u64) -> Self {
        assert!(
            inner & !Self::STORED_MASK == 0,
            "node pointer contains bits outside the supported range",
        );
        Self(Self::pack(inner))
    }
}

impl Add<usize> for NodePtr {
    type Output = NodePtr;

    fn add(self, rhs: usize) -> Self::Output {
        let new_idx = (self.idx() as u64) + rhs as u64;
        assert!(
            new_idx <= Self::INDEX_MASK,
            "node index exceeds representable range"
        );
        Self::new(self.half(), new_idx as usize)
    }
}

#[derive(Debug)]
pub struct Node {
    sum_q: AtomicU64,
    sum_sq_q: AtomicU64,
    actions: CustomLock,
    visits: AtomicU32,
    threads: AtomicU16,
    mov: AtomicU16,
    policy: AtomicU16,
    state: AtomicU16,
    num_actions: AtomicU8,
    gini_impurity: AtomicU8,
}

impl Node {
    pub fn new(state: GameState) -> Self {
        Node {
            sum_q: AtomicU64::new(0),
            sum_sq_q: AtomicU64::new(0),
            actions: CustomLock::new(NodePtr::NULL),
            visits: AtomicU32::new(0),
            threads: AtomicU16::new(0),
            mov: AtomicU16::new(0),
            policy: AtomicU16::new(0),
            state: AtomicU16::new(u16::from(state)),
            num_actions: AtomicU8::new(0),
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
        self.actions.read()
    }

    pub fn actions_mut(&self) -> WriteGuard<'_> {
        self.actions.write()
    }

    pub fn state(&self) -> GameState {
        GameState::from(self.state.load(Ordering::Relaxed))
    }

    pub fn set_state(&self, state: GameState) {
        self.state.store(u16::from(state), Ordering::Relaxed);
    }

    pub fn policy(&self) -> f32 {
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

    pub fn clear(&self) {
        self.clear_actions();
        self.set_state(GameState::Ongoing);
        self.set_gini_impurity(0.0);
        self.visits.store(0, Ordering::Relaxed);
        self.sum_q.store(0, Ordering::Relaxed);
        self.sum_sq_q.store(0, Ordering::Relaxed);
        self.threads.store(0, Ordering::Relaxed);
    }

    pub fn update(&self, q: f32) -> f32 {
        let q = (f64::from(q) * f64::from(QUANT)) as u64;
        let old_v = self.visits.fetch_add(1, Ordering::Relaxed);
        let old_q = self.sum_q.fetch_add(q, Ordering::Relaxed);
        self.sum_sq_q.fetch_add(q * q, Ordering::Relaxed);

        (((q + old_q) / u64::from(1 + old_v)) as f64 / f64::from(QUANT)) as f32
    }

    #[cfg(feature = "datagen")]
    pub fn kld_gain(new_visit_dist: &[i32], old_visit_dist: &[i32]) -> Option<f64> {
        let new_parent_visits = new_visit_dist.iter().sum::<i32>();
        let old_parent_visits = old_visit_dist.iter().sum::<i32>();

        if old_parent_visits == 0 {
            return None;
        }

        let mut kld_gain = 0.0;

        for (&new_visits, &old_visits) in new_visit_dist.iter().zip(old_visit_dist.iter()) {
            if old_visits == 0 {
                return None;
            }

            let q = f64::from(new_visits) / f64::from(new_parent_visits);
            let p = f64::from(old_visits) / f64::from(old_parent_visits);

            kld_gain += p * (p / q).ln();
        }

        Some(kld_gain / f64::from(new_parent_visits - old_parent_visits))
    }
}
