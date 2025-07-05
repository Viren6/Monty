use crate::{
    boxed_and_zeroed,
    chess::{Attacks, Board, Move},
};

use super::{
    accumulator::Accumulator,
    layer::{Layer, TransposedLayer},
};

#[inline]
fn dot_i8_i16_fallback(lhs: &[i8], rhs: &[i16]) -> i32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&w, &h)| i32::from(w) * i32::from(h))
        .sum()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_i16_avx2(lhs: &[i8], rhs: &[i16]) -> i32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_si256();
    let chunks = lhs.len() / 16;
    for i in 0..chunks {
        let off = i * 16;
        let a = _mm_loadu_si128(lhs.as_ptr().add(off) as *const __m128i);
        let b = _mm256_loadu_si256(rhs.as_ptr().add(off) as *const __m256i);
        let a16 = _mm256_cvtepi8_epi16(a);
        let prod = _mm256_madd_epi16(a16, b);
        sum = _mm256_add_epi32(sum, prod);
    }
    let mut out = [0i32; 8];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, sum);
    let mut total: i32 = out.iter().sum();
    for i in (chunks * 16)..lhs.len() {
        total += (*lhs.get_unchecked(i) as i32) * (*rhs.get_unchecked(i) as i32);
    }
    total
}

#[inline]
fn dot_i8_i16(lhs: &[i8], rhs: &[i16]) -> i32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        return dot_i8_i16_avx2(lhs, rhs);
    }
    #[allow(unreachable_code)]
    {
        dot_i8_i16_fallback(lhs, rhs)
    }
}

// DO NOT MOVE
#[allow(non_upper_case_globals, dead_code)]
pub const PolicyFileDefaultName: &str = "nn-658ca1d47406.network";
#[allow(non_upper_case_globals, dead_code)]
pub const CompressedPolicyName: &str = "nn-4b70c6924179.network";

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

        for (r, &b) in l1.0.iter_mut().zip(self.l1.biases.0.iter()) {
            *r = i16::from(b);
        }

        let mut feats = [0usize; 256];
        let mut count = 0;
        pos.map_features(|feat| {
            feats[count] = feat;
            count += 1;
        });

        l1.add_multi_i8(&feats[..count], &self.l1.weights);

        let mut res = Accumulator([0; L1 / 2]);

        for (elem, (&i, &j)) in res
            .0
            .iter_mut()
            .zip(l1.0.iter().take(L1 / 2).zip(l1.0.iter().skip(L1 / 2)))
        {
            let i = i32::from(i).clamp(0, i32::from(QA));
            let j = i32::from(j).clamp(0, i32::from(QA));
            *elem = ((i * j) / i32::from(QA / FACTOR)) as i16;
        }

        res
    }

    pub fn get(&self, pos: &Board, mov: &Move, hl: &Accumulator<i16, { L1 / 2 }>) -> f32 {
        let idx = map_move_to_index(pos, *mov);
        let weights = &self.l2.weights[idx];

        let res = dot_i8_i16(&weights.0, &hl.0);
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
