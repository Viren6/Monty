use std::simd::prelude::SimdInt;
use std::simd::cmp::SimdOrd;
use std::simd::Simd;

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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PolicyNetwork {
    l1: Layer<i8, { 768 * 4 }, L1>,
    l2: TransposedLayer<i8, { L1 / 2 }, { 1880 * 2 }>,
}

impl PolicyNetwork {
    pub fn hl(&self, pos: &Board) -> Accumulator<i16, { L1 / 2 }> {
        let mut l1 = Accumulator([0; L1]);

        // Initialize l1 with biases
        {
            let bias_slice = &self.l1.biases.0;
            let l1_slice = &mut l1.0;

            const CHUNK: usize = 16;
            let mut i = 0;
            while i + CHUNK <= L1 {
                let mut temp = [0i8; CHUNK];
                temp.copy_from_slice(&bias_slice[i..i + CHUNK]);
                let b_vec = Simd::from_array(temp);
                let b_i16 = b_vec.cast::<i16>();
                let b_arr = b_i16.to_array();
                l1_slice[i..i + CHUNK].copy_from_slice(&b_arr);
                i += CHUNK;
            }
            // Remainder
            for j in i..L1 {
                l1_slice[j] = i16::from(bias_slice[j]);
            }
        }

        // Add sparse features
        pos.map_features(|feat| {
            let weights_slice = &self.l1.weights[feat].0;
            let l1_slice = &mut l1.0;

            const CHUNK: usize = 16;
            let mut i = 0;
            while i + CHUNK <= L1 {
                let mut w_temp = [0i8; CHUNK];
                w_temp.copy_from_slice(&weights_slice[i..i + CHUNK]);
                let w_vec = Simd::from_array(w_temp);
                let w_i16 = w_vec.cast::<i16>();

                let mut orig_temp = [0i16; CHUNK];
                orig_temp.copy_from_slice(&l1_slice[i..i + CHUNK]);
                let orig_vec = Simd::from_array(orig_temp);

                let result = orig_vec + w_i16;
                let res_arr = result.to_array();
                l1_slice[i..i + CHUNK].copy_from_slice(&res_arr);
                i += CHUNK;
            }

            for j in i..L1 {
                l1_slice[j] += i16::from(weights_slice[j]);
            }
        });

        // Now compute the half-layer transformation with clamping and multiplication
        let mut res = Accumulator([0; L1 / 2]);
        const CHUNK: usize = 16; // vector width for final step
        let half = L1 / 2;
        let divisor = i32::from(QA / FACTOR);

        let l1_first = &l1.0[..half];
        let l1_second = &l1.0[half..];

        let mut i = 0;
        while i + CHUNK <= half {
            let mut i_temp = [0i16; CHUNK];
            i_temp.copy_from_slice(&l1_first[i..i + CHUNK]);
            let i_vec = Simd::from_array(i_temp);

            let mut j_temp = [0i16; CHUNK];
            j_temp.copy_from_slice(&l1_second[i..i + CHUNK]);
            let j_vec = Simd::from_array(j_temp);

            let zero = Simd::<i16, CHUNK>::splat(0);
            let max_qa = Simd::<i16, CHUNK>::splat(QA);
            let i_clamped = i_vec.simd_max(zero).simd_min(max_qa);
            let j_clamped = j_vec.simd_max(zero).simd_min(max_qa);

            let i_i32 = i_clamped.cast::<i32>();
            let j_i32 = j_clamped.cast::<i32>();
            let product = i_i32 * j_i32;

            let div = product / Simd::<i32, CHUNK>::splat(divisor);
            let result_i16 = div.cast::<i16>();

            let out_arr = result_i16.to_array();
            res.0[i..i + CHUNK].copy_from_slice(&out_arr);
            i += CHUNK;
        }

        // Remainder for final step
        for j in i..half {
            let ii = i32::from(l1.0[j].clamp(0, QA));
            let jj = i32::from(l1.0[j + half].clamp(0, QA));
            res.0[j] = ((ii * jj) / divisor) as i16;
        }

        res
    }

    pub fn get(&self, pos: &Board, mov: &Move, hl: &Accumulator<i16, { L1 / 2 }>) -> f32 {
        let idx = map_move_to_index(pos, *mov);
        let weights = &self.l2.weights[idx];

        // Vectorized dot product
        let w_slice = &weights.0;
        let v_slice = &hl.0;
        let len = w_slice.len();

        const CHUNK: usize = 16;
        let mut sum_vec = Simd::<i32, CHUNK>::splat(0);
        let mut i = 0;

        while i + CHUNK <= len {
            let mut w_temp = [0i8; CHUNK];
            w_temp.copy_from_slice(&w_slice[i..i + CHUNK]);
            let w_chunk = Simd::from_array(w_temp);

            let mut v_temp = [0i16; CHUNK];
            v_temp.copy_from_slice(&v_slice[i..i + CHUNK]);
            let v_chunk = Simd::from_array(v_temp);

            let w_i32 = w_chunk.cast::<i32>();
            let v_i32 = v_chunk.cast::<i32>();
            sum_vec = sum_vec + (w_i32 * v_i32);
            i += CHUNK;
        }

        // Manual horizontal sum
        let sum_arr = sum_vec.to_array();
        let mut res = 0;
        for val in sum_arr {
            res += val;
        }

        for j in i..len {
            res += i32::from(w_slice[j]) * i32::from(v_slice[j]);
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
