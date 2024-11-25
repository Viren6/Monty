use crate::Board;

use super::{accumulator::Accumulator, activation::SCReLU, layer::Layer};

// DO NOT MOVE
#[allow(non_upper_case_globals)]
pub const ValueFileDefaultName: &str = "nn-7582517b793b.network";

const INPUT_CHANNELS: usize = 768;
const INPUT_DIM: usize = 8;
const INPUT_SIZE: usize = INPUT_DIM * INPUT_DIM;
const INPUT_EMBED: usize = INPUT_CHANNELS * INPUT_SIZE;

const KERNEL_DIM: usize = 5;

const OUTPUT_CHANNELS: usize = 8;
const OUTPUT_DIM: usize = 4;
const OUTPUT_SIZE: usize = OUTPUT_DIM * OUTPUT_DIM;
const OUTPUT_EMBED: usize = OUTPUT_CHANNELS * OUTPUT_SIZE;

const L3: usize = 1024;
const L4: usize = 32;

#[repr(C)]
pub struct ValueNetwork {
    l1: Layer<f32, 3072, INPUT_EMBED>,
    l2: ConvLayer,
    l3: Layer<f32, OUTPUT_EMBED, L3>,
    l4: Layer<f32, L3, L4>,
    l5: Layer<f32, L4, 3>,
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

        l2.screlu();

        let l3 = self.l2.forward(&l2);
        let l4 = self.l3.forward::<SCReLU>(&l3);
        let l5 = self.l4.forward::<SCReLU>(&l4);
        let out = self.l5.forward::<SCReLU>(&l5);

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
#[derive(Clone, Copy)]
pub struct ConvKernel([[f32; KERNEL_DIM]; KERNEL_DIM]);

#[repr(C)]
pub struct ConvLayer {
    weights: [[ConvKernel; INPUT_CHANNELS]; OUTPUT_CHANNELS],
    biases: [[f32; OUTPUT_SIZE]; OUTPUT_CHANNELS],
}

impl ConvLayer {
    pub fn forward(&self, input: &Accumulator<f32, INPUT_EMBED>) -> Accumulator<f32, OUTPUT_EMBED> {
        let mut out = Accumulator([0.0; OUTPUT_EMBED]);

        for i in 0..OUTPUT_CHANNELS {
            for j in 0..OUTPUT_SIZE {
                out.0[OUTPUT_SIZE * i + j] = self.biases[i][j];
            }
        }

        for i in 0..OUTPUT_CHANNELS {
            for j in 0..INPUT_CHANNELS {
                let kernel = self.weights[i][j];

                for x in 0..OUTPUT_DIM {
                    for y in 0..OUTPUT_DIM {
                        let base_idx = INPUT_SIZE * j + INPUT_DIM * x + y;

                        let mut elem = 0.0;
                        for a in 0..KERNEL_DIM {
                            for b in 0..KERNEL_DIM {
                                elem += kernel.0[a][b] * input.0[base_idx + INPUT_DIM * a + b];
                            }
                        }

                        out.0[OUTPUT_SIZE * i + OUTPUT_DIM * x + y] += elem;
                    }
                }
            }
        }

        out
    }
}
