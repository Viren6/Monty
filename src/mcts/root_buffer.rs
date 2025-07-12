use crate::tree::{Node, Tree, QUANT};

const FLUSH_THRESHOLD: u32 = 16;

pub struct RootBuffer {
    visits: u32,
    sum_q: u64,
    sum_sq_q: u64,
    local_visits: u32,
    local_sum_q: u64,
    local_sum_sq_q: u64,
}

impl RootBuffer {
    pub fn new() -> Self {
        Self {
            visits: 0,
            sum_q: 0,
            sum_sq_q: 0,
            local_visits: 0,
            local_sum_q: 0,
            local_sum_sq_q: 0,
        }
    }

    pub fn sync(&mut self, node: &Node) {
        self.local_visits = node.visits();
        self.local_sum_q = node.sum_q();
        self.local_sum_sq_q = node.sum_sq_q();
    }

    pub fn visits_total(&self) -> u32 {
        self.local_visits
    }

    pub fn q(&self) -> f32 {
        if self.local_visits == 0 {
            0.0
        } else {
            (self.local_sum_q as f64 / self.local_visits as f64 / f64::from(QUANT)) as f32
        }
    }

    pub fn var(&self) -> f32 {
        if self.local_visits == 0 {
            0.0
        } else {
            let q = self.local_sum_q as f64 / self.local_visits as f64 / f64::from(QUANT);
            let sq_q = self.local_sum_sq_q as f64
                / self.local_visits as f64
                / f64::from(QUANT).powi(2);
            (sq_q - q.powi(2)).max(0.0) as f32
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
        self.local_visits += self.visits;
        self.local_sum_q += self.sum_q;
        self.local_sum_sq_q += self.sum_sq_q;
        self.visits = 0;
        self.sum_q = 0;
        self.sum_sq_q = 0;
    }
}

pub struct RootBuffers {
    buffers: Vec<RootBuffer>,
}

impl RootBuffers {
    pub fn new(threads: usize, root: &Node) -> Self {
        let mut buffers: Vec<RootBuffer> = (0..threads).map(|_| RootBuffer::new()).collect();
        for buf in &mut buffers {
            buf.sync(root);
        }
        Self { buffers }
    }

    pub fn sync_all(&mut self, root: &Node) {
        for buf in &mut self.buffers {
            buf.sync(root);
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
