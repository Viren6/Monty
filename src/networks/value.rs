use crate::chess::Board;

use super::{activation::{Activation, SCReLU}, layer::Layer, threats, Accumulator};

// DO NOT MOVE
#[allow(non_upper_case_globals, dead_code)]
pub const ValueFileDefaultName: &str = "sb1000-single-layer.network";
#[allow(non_upper_case_globals, dead_code)]
pub const CompressedValueName: &str = "nn-f004da0ebf25.network";
#[allow(non_upper_case_globals, dead_code)]
pub const DatagenValueFileName: &str = "nn-5601bb8c241d.network";

const QA: i16 = 255;
const QB: i16 = 128;

const L1: usize = 3072;

#[repr(C, align(64))]
pub struct ValueNetwork {
    pst: [Accumulator<f32, 3>; threats::TOTAL],
    l1: Layer<i16, { threats::TOTAL }, L1>,
    l2: Layer<i16, L1, 3>,
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

        let mut l1 = self.l1.biases;

        l1.add_multi(&feats[..count], &self.l1.weights);

        let mut act = Accumulator([0.0; L1]);

        for (a, &i) in act.0.iter_mut().zip(l1.0.iter()) {
            *a = f32::from(i) / f32::from(QA);
        }

        let mut out = Accumulator([0.0; 3]);

        for (o, &b) in out.0.iter_mut().zip(self.l2.biases.0.iter()) {
            *o = f32::from(b) / f32::from(QB);
        }

        for (i, weights) in act.0.iter().zip(self.l2.weights.iter()) {
            let act_i = SCReLU::activate(*i);
            out.madd_i16(act_i / f32::from(QB), weights);
        }

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
