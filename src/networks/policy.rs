use std::simd::prelude::*; 

use crate::{
    boxed_and_zeroed,
    chess::{Attacks, Board, Move},
};

use super::{
    accumulator::Accumulator,
    layer::{Layer, TransposedLayer},
};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const PolicyFileDefaultName: &str = "nn-658ca1d47406.network";

const QA: i16 = 128;
const QB: i16 = 128;
const FACTOR: i16 = 32;

pub const L1: usize = 12288;
const L1_SIMD_LANES: usize = L1 / 16;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    l1: Layer<i8, { 768 * 4 }, L1>,
    l2: TransposedLayer<i8, { L1 / 2 }, { 1880 * 2 }>,
}

impl PolicyNetwork {
    pub fn hl(&self, pos: &Board) -> Accumulator<i16, { L1 / 2 }> {
        let mut l1_accumulators = [Simd::<i16, 16>::splat(0); L1_SIMD_LANES];

        for (chunk_idx, chunk) in l1_accumulators.iter_mut().enumerate() {
            let slice = &self.l1.biases.0[chunk_idx * 16..(chunk_idx + 1) * 16];
            let bias_i16 = Simd::<i8, 16>::from_slice(slice).cast::<i16>();
            *chunk += bias_i16;
        }

        pos.map_features(|feat| {
            for (chunk_idx, chunk) in l1_accumulators.iter_mut().enumerate() {
                let slice = &self.l1.weights[feat].0[chunk_idx * 16..(chunk_idx + 1) * 16];
                let weight_i16 = Simd::<i8, 16>::from_slice(slice).cast::<i16>();
                *chunk += weight_i16;
            }
        });

        let min = Simd::<i16, 16>::splat(0);
        let max = Simd::<i16, 16>::splat(QA);
        let l1_clipped: [Simd<i16, 16>; L1_SIMD_LANES] = l1_accumulators
            .map(|acc| acc.simd_max(min).simd_min(max));

        let mut res = Accumulator([0; L1 / 2]);

        for sim_index in 0..(L1_SIMD_LANES / 2) {
            let i_simd = l1_clipped[sim_index];
            let j_simd = l1_clipped[L1_SIMD_LANES / 2 + sim_index];

            let scaling_factor = QA / FACTOR;
            let scaled_simd = (i_simd * j_simd) / Simd::<i16, 16>::splat(scaling_factor);

            res.0[sim_index * 16..(sim_index + 1) * 16]
                .copy_from_slice(&scaled_simd.to_array());
        }

        res
    }

    pub fn get(&self, pos: &Board, mov: &Move, hl: &Accumulator<i16, { L1 / 2 }>) -> f32 {
        let idx = map_move_to_index(pos, *mov);
        let weights = &self.l2.weights[idx];

        let mut res = 0i32;

        const SIMD_WIDTH: usize = 16;
        let chunks = hl.0.len() / SIMD_WIDTH;
        for chunk_idx in 0..chunks {
            let w_simd = Simd::<i8, SIMD_WIDTH>::from_slice(&weights.0[chunk_idx * SIMD_WIDTH..(chunk_idx + 1) * SIMD_WIDTH]);
            let v_simd = Simd::<i16, SIMD_WIDTH>::from_slice(&hl.0[chunk_idx * SIMD_WIDTH..(chunk_idx + 1) * SIMD_WIDTH]);

            let w_i16 = w_simd.cast::<i16>();

            let prod = w_i16 * v_simd;

            let array: [i16; 16] = prod.to_array();

            let sum: i32 = array.iter().map(|x| *x as i32).sum();

            res = res.wrapping_add(sum);
        }

        (res as f32 / f32::from(QA * FACTOR) + f32::from(self.l2.biases.0[idx])) / f32::from(QB)
    }
}

const PROMOS: usize = 4 * 22;

fn map_move_to_index(pos: &Board, mov: Move) -> usize {
    let hm = if pos.king_index() % 8 > 3 { 7 } else { 0 };
    let good_see = (OFFSETS[64] + PROMOS) * usize::from(pos.see(&mov, -108));

    let idx = if mov.is_promo() {
        let ffile = (mov.src() ^ hm) % 8;
        let tfile = (mov.to() ^ hm) % 8;
        let promo_id = 2 * ffile + tfile;

        OFFSETS[64] + 22 * (mov.promo_pc() - 3) + usize::from(promo_id)
    } else {
        let flip = if pos.stm() == 1 { 56 } else { 0 };
        let from = usize::from(mov.src() ^ flip ^ hm);
        let dest = usize::from(mov.to() ^ flip ^ hm);

        let below = Attacks::ALL_DESTINATIONS[from] & ((1 << dest) - 1);

        OFFSETS[from] + below.count_ones() as usize
    };

    good_see + idx
}

const OFFSETS: [usize; 65] = {
    let mut offsets = [0; 65];

    let mut curr = 0;
    let mut sq = 0;

    while sq < 64 {
        offsets[sq] = curr;
        curr += Attacks::ALL_DESTINATIONS[sq].count_ones() as usize;
        sq += 1;
    }

    offsets[64] = curr;

    offsets
};

#[repr(C)]
pub struct UnquantisedPolicyNetwork {
    l1: Layer<f32, { 768 * 4 }, L1>,
    l2: Layer<f32, { L1 / 2 }, { 1880 * 2 }>,
}

impl UnquantisedPolicyNetwork {
    pub fn quantise(&self) -> Box<PolicyNetwork> {
        let mut quantised: Box<PolicyNetwork> = unsafe { boxed_and_zeroed() };

        self.l1.quantise_into_i8(&mut quantised.l1, QA, 0.99);
        self.l2
            .quantise_transpose_into_i8(&mut quantised.l2, QB, 0.99);

        quantised
    }
}

