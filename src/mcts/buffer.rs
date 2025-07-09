use crate::tree::{Node, NodePtr, Tree};

const QUANT: u64 = (16384 * 4) as u64;

#[derive(Default, Clone)]
pub struct NodeBuffer {
    visits: u32,
    sum_q: u64,
    sum_sq_q: u64,
}

impl NodeBuffer {
    pub fn add(&mut self, q: f32) {
        let q = (q * QUANT as f32) as u64;
        self.visits += 1;
        self.sum_q += q;
        self.sum_sq_q += q * q;
    }

    pub fn needs_flush(&self, thresh: u32) -> bool {
        self.visits >= thresh
    }

    fn flush_internal(&mut self, node: &Node) {
        node.add_stats(self.visits, self.sum_q, self.sum_sq_q);
        self.visits = 0;
        self.sum_q = 0;
        self.sum_sq_q = 0;
    }

    pub fn flush(&mut self, tree: &Tree, ptr: NodePtr, hash: u64) {
        if self.visits == 0 {
            return;
        }
        self.flush_internal(&tree[ptr]);
        tree.push_hash(hash, 1.0 - tree[ptr].q());
    }
}

pub struct RootContext {
    pub root_ptr: NodePtr,
    pub first_child_ptr: NodePtr,
    pub root_hash: u64,
    pub child_hashes: Vec<u64>,
}

pub struct RootBuffer {
    pub root: NodeBuffer,
    pub children: Vec<NodeBuffer>,
}

impl RootBuffer {
    pub fn new(num_children: usize) -> Self {
        Self {
            root: NodeBuffer::default(),
            children: vec![NodeBuffer::default(); num_children],
        }
    }

    pub fn maybe_flush(&mut self, tree: &Tree, ctx: &RootContext, thresh: u32) {
        if self.root.needs_flush(thresh) {
            self.root.flush(tree, ctx.root_ptr, ctx.root_hash);
        }
        for (i, buf) in self.children.iter_mut().enumerate() {
            if buf.needs_flush(thresh) {
                let ptr = ctx.first_child_ptr + i;
                buf.flush(tree, ptr, ctx.child_hashes[i]);
            }
        }
    }

    pub fn flush_all(&mut self, tree: &Tree, ctx: &RootContext) {
        self.root.flush(tree, ctx.root_ptr, ctx.root_hash);
        for (i, buf) in self.children.iter_mut().enumerate() {
            let ptr = ctx.first_child_ptr + i;
            buf.flush(tree, ptr, ctx.child_hashes[i]);
        }
    }
}