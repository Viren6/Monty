use crate::tree::{Node, Tree, QUANT};

const FLUSH_THRESHOLD: u32 = 128;

pub struct RootBuffer {
    visits: u32,
    sum_q: u64,
    sum_sq_q: u64,
}

impl RootBuffer {
    pub fn new() -> Self {
        Self {
            visits: 0,
            sum_q: 0,
            sum_sq_q: 0,
        }
    }

    pub fn add(&mut self, q: f32) {
        let q = (f64::from(q) * f64::from(QUANT)) as u64;
        self.visits += 1;
        self.sum_q += q;
        self.sum_sq_q += q * q;
    }

    pub fn should_flush(&self) -> bool {
        self.visits >= FLUSH_THRESHOLD
    }

    pub fn flush(&mut self, node: &Node, hash: u64, tree: &Tree) {
        if self.visits == 0 {
            return;
        }
        let new_q = node.bulk_update(self.visits, self.sum_q, self.sum_sq_q);
        tree.push_hash(hash, 1.0 - new_q);
        self.visits = 0;
        self.sum_q = 0;
        self.sum_sq_q = 0;
    }
}

pub struct RootBuffers {
    buffers: Vec<RootBuffer>,
}

impl RootBuffers {
    pub fn new(threads: usize) -> Self {
        Self {
            buffers: (0..threads).map(|_| RootBuffer::new()).collect(),
        }
    }

    pub fn thread(&mut self, tid: usize) -> &mut RootBuffer {
        &mut self.buffers[tid]
    }

    pub fn flush_all(&mut self, node: &Node, hash: u64, tree: &Tree) {
        for buf in &mut self.buffers {
            buf.flush(node, hash, tree);
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [RootBuffer] {
        &mut self.buffers
    }
}
