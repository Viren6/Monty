use crate::{boxed_and_zeroed, Board};

use super::layer::{Layer, TransposedLayer};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const ValueFileDefaultName: &str = "nn-a0a560b3ac29.network";

const QA: i16 = 512;
const QB: i16 = 1024;
const FACTOR: i16 = 32;

const L1: usize = 12288;

#[repr(C)]
pub struct ValueNetwork {
    l1: Layer<i16, { 768 * 4 }, L1>,
    l2: TransposedLayer<i16, L1, 3>,
}

impl ValueNetwork {
    pub fn eval(&self, board: &Board) -> (f32, f32, f32) {
        let mut count = 0;
        let mut feats = [0; 32];
        board.map_features(|feat| {
            feats[count] = feat;
            count += 1;
        });

        let mut l2 = self.l1.biases;

        l2.add_multi(&feats[..count], &self.l1.weights);

        let out = self.l2.screlu_affine::<QA, QB, FACTOR>(&l2);

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

#[repr(C)]
pub struct UnquantisedValueNetwork {
    l1: Layer<f32, { 768 * 4 }, L1>,
    l2: Layer<f32, L1, 3>,
}

impl UnquantisedValueNetwork {
    pub fn quantise(&self) -> Box<ValueNetwork> {
        let mut quantised: Box<ValueNetwork> = unsafe { boxed_and_zeroed() };

        self.l1.quantise_into_i16(&mut quantised.l1, QA, 0.99);
        self.l2
            .quantise_transpose_into_i16(&mut quantised.l2, QB, 0.99);

        quantised
    }
}
