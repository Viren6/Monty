use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use super::{Node, NodePtr};
use crate::chess::GameState;

const CACHE_SIZE: usize = 1024;

pub struct TreeHalf {
    pub(super) nodes: Vec<Node>,
    used: AtomicUsize,
    next: Vec<AtomicUsize>,
    end: Vec<AtomicUsize>,
    half: bool,
    #[cfg(debug_assertions)]
    generation: AtomicU32,
}

impl std::ops::Index<NodePtr> for TreeHalf {
    type Output = Node;

    fn index(&self, index: NodePtr) -> &Self::Output {
        &self.nodes[index.idx()]
    }
}

impl TreeHalf {
    pub fn new(size: usize, half: bool, threads: usize) -> Self {
        let mut res = Self {
            nodes: Vec::new(),
            used: AtomicUsize::new(0),
            next: (0..threads).map(|_| AtomicUsize::new(0)).collect(),
            end: (0..threads).map(|_| AtomicUsize::new(0)).collect(),
            half,
            #[cfg(debug_assertions)]
            generation: AtomicU32::new(0),
        };

        res.nodes.reserve_exact(size);

        unsafe {
            use std::mem::MaybeUninit;
            let chunk_size = size.div_ceil(threads);
            let ptr = res.nodes.as_mut_ptr().cast();
            let uninit: &mut [MaybeUninit<Node>] = std::slice::from_raw_parts_mut(ptr, size);

            std::thread::scope(|s| {
                for chunk in uninit.chunks_mut(chunk_size) {
                    s.spawn(|| {
                        for node in chunk {
                            node.write(Node::new(GameState::Ongoing));
                        }
                    });
                }
            });

            res.nodes.set_len(size);
        }

        #[cfg(debug_assertions)]
        for (idx, node) in res.nodes.iter().enumerate() {
            node.scrub();
            node.mark_ptr(NodePtr::new(half, idx as u32));
            node.mark_generation(0);
            node.mark_depth(0);
        }

        res
    }

    pub fn reserve_nodes_thread(&self, num: usize, thread: usize) -> Option<NodePtr> {
        let mut next = self.next[thread].load(Ordering::Relaxed);
        let mut end = self.end[thread].load(Ordering::Relaxed);

        if next + num > end {
            let block = CACHE_SIZE.max(num);
            let start = self.used.fetch_add(block, Ordering::Relaxed);
            if start + block > self.nodes.len() {
                return None;
            }
            next = start;
            end = start + block;
            self.next[thread].store(next + num, Ordering::Relaxed);
            self.end[thread].store(end, Ordering::Relaxed);
            let ptr = NodePtr::new(self.half, start as u32);
            #[cfg(debug_assertions)]
            {
                let gen = self.generation.load(Ordering::Relaxed);
                for i in 0..num {
                    self.nodes[(ptr.idx() + i) as usize].mark_generation(gen);
                }
            }
            Some(ptr)
        } else {
            self.next[thread].store(next + num, Ordering::Relaxed);
            let ptr = NodePtr::new(self.half, next as u32);
            #[cfg(debug_assertions)]
            {
                let gen = self.generation.load(Ordering::Relaxed);
                for i in 0..num {
                    self.nodes[(ptr.idx() + i) as usize].mark_generation(gen);
                }
            }
            Some(ptr)
        }
    }

    pub fn clear(&self) {
        self.used.store(0, Ordering::Relaxed);
        for (n, e) in self.next.iter().zip(&self.end) {
            n.store(0, Ordering::Relaxed);
            e.store(0, Ordering::Relaxed);
        }
        #[cfg(debug_assertions)]
        {
            self.generation.fetch_add(1, Ordering::Relaxed);
            for node in &self.nodes {
                node.scrub();
                node.mark_generation(self.generation.load(Ordering::Relaxed));
                node.mark_depth(0);
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.used.load(Ordering::Relaxed) == 0
    }

    pub fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    #[cfg(debug_assertions)]
    pub fn generation(&self) -> u32 {
        self.generation.load(Ordering::Relaxed)
    }

    pub fn is_full(&self) -> bool {
        self.used() >= self.nodes.len()
    }
}
