#[cfg(not(target_feature = "avx2"))]
mod autovec;
#[cfg(not(target_feature = "avx2"))]
use autovec as backend;
#[cfg(target_feature = "avx2")]
mod avx2;
#[cfg(target_feature = "avx2")]
use avx2 as backend;

use crate::chess::Board;

use super::{activation::SCReLU, layer::Layer, threats, Accumulator};

// DO NOT MOVE
#[allow(non_upper_case_globals, dead_code)]
pub const ValueFileDefaultName: &str = "nn-ecb40419d127.network";
#[allow(non_upper_case_globals, dead_code)]
pub const CompressedValueName: &str = "nn-c1dd869bed55.network";
#[allow(non_upper_case_globals, dead_code)]
pub const DatagenValueFileName: &str = "nn-5601bb8c241d.network";

const QA: i16 = 128;
const QB: i16 = 128;

const L1: usize = 8192;

#[repr(C, align(64))]
pub struct Align64<T, const N: usize>([T; N]);

#[repr(C, align(64))]
pub struct ValueNetwork {
    pst: [Accumulator<f32, 3>; threats::TOTAL],
    l1: Layer<i8, { threats::TOTAL }, L1>,
    l2w: Align64<i8, { L1 * 8 }>,
    l2b: [f32; 16],
    l3: Layer<f32, 16, 128>,
    l4: Layer<f32, 128, 3>,
}

impl ValueNetwork {
    pub fn eval(&self, board: &Board) -> (f32, f32, f32) {
        let mut pst = Accumulator([0.0; 3]);

        let mut count = 0;
        let mut feats = [0; 160];
        threats::map_features(board, |feat| {
            feats[count] = feat;
            pst.add(&self.pst[feat]);
            count += 1;
        });

        let mut l2 = Accumulator([0; L1]);

        for (i, &j) in l2.0.iter_mut().zip(self.l1.biases.0.iter()) {
            *i = i16::from(j);
        }

        l2.add_multi_i8(&feats[..count], &self.l1.weights);

        let mut act = Align64([0; L1 / 2]);

        for (a, (&i, &j)) in act
            .0
            .iter_mut()
            .zip(l2.0.iter().take(L1 / 2).zip(l2.0.iter().skip(L1 / 2)))
        {
            let i = i.clamp(0, QA);
            let j = j.clamp(0, QA);
            *a = ((i * j) >> 7) as u8;
        }

        // QA * QA * QB / (1 << SHIFT)
        let l3_ = unsafe { backend::l2(self, &act) };

        let mut l3 = Accumulator([0.0; 16]);

        let k = f32::from(1i16 << 7) / f32::from(QA) / (f32::from(QA) * f32::from(QB));

        for (r, (&f, &b)) in l3.0.iter_mut().zip(l3_.0.iter().zip(self.l2b.iter())) {
            *r = f as f32 * k + b;
        }

        let l4 = self.l3.forward::<SCReLU>(&l3);
        let mut out = self.l4.forward::<SCReLU>(&l4);
        out.add(&pst);

        let mut win = out.0[2];
        let mut draw = out.0[1];
        let mut loss = out.0[0];

        let max = win.max(draw).max(loss);

        win = (win - max).exp();
        draw = (draw - max).exp();
        loss = (loss - max).exp();

        let sum = win + draw + loss;

        (win / sum, draw / sum, loss / sum)
    }
}
