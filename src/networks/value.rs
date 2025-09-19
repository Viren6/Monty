use crate::chess::{consts::ValueOffsets, Board};

use super::{
    activation::{Activation, SCReLU},
    layer::{Layer, TransposedLayer},
    threats, Accumulator,
};

// DO NOT MOVE
#[allow(non_upper_case_globals, dead_code)]
pub const ValueFileDefaultName: &str = "nn-58274aa39e13.network";
#[allow(non_upper_case_globals, dead_code)]
pub const CompressedValueName: &str = "nn-fa1a8afd872c.network";
#[allow(non_upper_case_globals, dead_code)]
pub const DatagenValueFileName: &str = "nn-5601bb8c241d.network";

const QA: i16 = 128;
const QB: i16 = 1024;

const L1: usize = 3072;
const VALUE_BUCKETS: usize = 37;
const BUCKET_OUTPUTS: usize = 3;
const BUCKET_HIDDEN: usize = 16;
const TOTAL_THREATS: usize = 2 * ValueOffsets::END;

#[derive(Clone, Copy)]
struct Bucket {
    idx: u8,
    min_piece: u8,
    max_piece: Option<u8>,
    min_threat: u16,
    max_threat: Option<u16>,
}

const PIECE_THREAT_BUCKETS: [Bucket; VALUE_BUCKETS] = [
    Bucket {
        idx: 0,
        min_piece: 0,
        max_piece: Some(5),
        min_threat: 0,
        max_threat: Some(2),
    },
    Bucket {
        idx: 1,
        min_piece: 0,
        max_piece: Some(5),
        min_threat: 3,
        max_threat: None,
    },
    Bucket {
        idx: 2,
        min_piece: 6,
        max_piece: Some(8),
        min_threat: 0,
        max_threat: Some(2),
    },
    Bucket {
        idx: 3,
        min_piece: 6,
        max_piece: Some(8),
        min_threat: 3,
        max_threat: Some(5),
    },
    Bucket {
        idx: 4,
        min_piece: 6,
        max_piece: Some(8),
        min_threat: 6,
        max_threat: None,
    },
    Bucket {
        idx: 5,
        min_piece: 9,
        max_piece: Some(11),
        min_threat: 0,
        max_threat: Some(2),
    },
    Bucket {
        idx: 6,
        min_piece: 9,
        max_piece: Some(11),
        min_threat: 3,
        max_threat: Some(5),
    },
    Bucket {
        idx: 7,
        min_piece: 9,
        max_piece: Some(11),
        min_threat: 6,
        max_threat: Some(8),
    },
    Bucket {
        idx: 8,
        min_piece: 9,
        max_piece: Some(11),
        min_threat: 9,
        max_threat: None,
    },
    Bucket {
        idx: 9,
        min_piece: 12,
        max_piece: Some(14),
        min_threat: 0,
        max_threat: Some(5),
    },
    Bucket {
        idx: 10,
        min_piece: 12,
        max_piece: Some(14),
        min_threat: 6,
        max_threat: Some(9),
    },
    Bucket {
        idx: 11,
        min_piece: 12,
        max_piece: Some(14),
        min_threat: 10,
        max_threat: Some(13),
    },
    Bucket {
        idx: 12,
        min_piece: 12,
        max_piece: Some(14),
        min_threat: 14,
        max_threat: None,
    },
    Bucket {
        idx: 13,
        min_piece: 15,
        max_piece: Some(17),
        min_threat: 0,
        max_threat: Some(9),
    },
    Bucket {
        idx: 14,
        min_piece: 15,
        max_piece: Some(17),
        min_threat: 10,
        max_threat: Some(14),
    },
    Bucket {
        idx: 15,
        min_piece: 15,
        max_piece: Some(17),
        min_threat: 15,
        max_threat: Some(19),
    },
    Bucket {
        idx: 16,
        min_piece: 15,
        max_piece: Some(17),
        min_threat: 20,
        max_threat: None,
    },
    Bucket {
        idx: 17,
        min_piece: 18,
        max_piece: Some(20),
        min_threat: 0,
        max_threat: Some(14),
    },
    Bucket {
        idx: 18,
        min_piece: 18,
        max_piece: Some(20),
        min_threat: 15,
        max_threat: Some(20),
    },
    Bucket {
        idx: 19,
        min_piece: 18,
        max_piece: Some(20),
        min_threat: 21,
        max_threat: Some(26),
    },
    Bucket {
        idx: 20,
        min_piece: 18,
        max_piece: Some(20),
        min_threat: 26,
        max_threat: None,
    },
    Bucket {
        idx: 21,
        min_piece: 21,
        max_piece: Some(23),
        min_threat: 0,
        max_threat: Some(20),
    },
    Bucket {
        idx: 22,
        min_piece: 21,
        max_piece: Some(23),
        min_threat: 21,
        max_threat: Some(26),
    },
    Bucket {
        idx: 23,
        min_piece: 21,
        max_piece: Some(23),
        min_threat: 27,
        max_threat: Some(32),
    },
    Bucket {
        idx: 24,
        min_piece: 21,
        max_piece: Some(23),
        min_threat: 33,
        max_threat: None,
    },
    Bucket {
        idx: 25,
        min_piece: 24,
        max_piece: Some(26),
        min_threat: 0,
        max_threat: Some(26),
    },
    Bucket {
        idx: 26,
        min_piece: 24,
        max_piece: Some(26),
        min_threat: 27,
        max_threat: Some(33),
    },
    Bucket {
        idx: 27,
        min_piece: 24,
        max_piece: Some(26),
        min_threat: 34,
        max_threat: Some(40),
    },
    Bucket {
        idx: 28,
        min_piece: 24,
        max_piece: Some(26),
        min_threat: 41,
        max_threat: None,
    },
    Bucket {
        idx: 29,
        min_piece: 27,
        max_piece: Some(29),
        min_threat: 0,
        max_threat: Some(32),
    },
    Bucket {
        idx: 30,
        min_piece: 27,
        max_piece: Some(29),
        min_threat: 33,
        max_threat: Some(39),
    },
    Bucket {
        idx: 31,
        min_piece: 27,
        max_piece: Some(29),
        min_threat: 40,
        max_threat: Some(46),
    },
    Bucket {
        idx: 32,
        min_piece: 27,
        max_piece: Some(29),
        min_threat: 47,
        max_threat: None,
    },
    Bucket {
        idx: 33,
        min_piece: 30,
        max_piece: None,
        min_threat: 0,
        max_threat: Some(37),
    },
    Bucket {
        idx: 34,
        min_piece: 30,
        max_piece: None,
        min_threat: 38,
        max_threat: Some(44),
    },
    Bucket {
        idx: 35,
        min_piece: 30,
        max_piece: None,
        min_threat: 45,
        max_threat: Some(51),
    },
    Bucket {
        idx: 36,
        min_piece: 30,
        max_piece: None,
        min_threat: 52,
        max_threat: None,
    },
];

#[repr(C, align(64))]
pub struct ValueNetwork {
    pst: [Accumulator<f32, { threats::TOTAL }>; VALUE_BUCKETS * BUCKET_OUTPUTS],
    l0: Layer<i8, { threats::TOTAL }, L1>,
    l1: TransposedLayer<i16, { L1 / 2 }, { VALUE_BUCKETS * BUCKET_HIDDEN }>,
    l2: Layer<f32, BUCKET_HIDDEN, { VALUE_BUCKETS * BUCKET_HIDDEN }>,
    l3: Layer<f32, BUCKET_HIDDEN, { VALUE_BUCKETS * BUCKET_OUTPUTS }>,
}

impl ValueNetwork {
    pub fn eval(&self, board: &Board) -> (f32, f32, f32) {
        let mut count = 0;
        let mut feats = [0; 160];
        let mut piece_count = 0u16;
        let mut threat_count = 0u16;
        threats::map_features(board, |feat| {
            feats[count] = feat;
            count += 1;
            if feat < TOTAL_THREATS {
                threat_count += 1;
            } else {
                piece_count += 1;
            }
        });

        let bucket = select_bucket(piece_count as u8, threat_count);

        let mut pst = Accumulator([0.0; BUCKET_OUTPUTS]);
        let pst_rows = &self.pst[bucket * BUCKET_OUTPUTS..(bucket + 1) * BUCKET_OUTPUTS];
        for &feat in feats[..count].iter() {
            for (acc, row) in pst.0.iter_mut().zip(pst_rows.iter()) {
                *acc += row.0[feat];
            }
        }

        let mut l0 = Accumulator([0; L1]);

        for (r, &b) in l0.0.iter_mut().zip(self.l0.biases.0.iter()) {
            *r = i16::from(b);
        }

        l0.add_multi_i8(&feats[..count], &self.l0.weights);

        let mut act = [0; L1 / 2];

        for (a, (&i, &j)) in act
            .iter_mut()
            .zip(l0.0.iter().take(L1 / 2).zip(l0.0.iter().skip(L1 / 2)))
        {
            let i = i.clamp(0, QA);
            let j = j.clamp(0, QA);
            *a = i * j;
        }

        let l1 = self.forward_bucket_l1(bucket, &act);
        let l2 = self.forward_bucket_dense::<
            SCReLU,
            BUCKET_HIDDEN,
            BUCKET_HIDDEN,
            { VALUE_BUCKETS * BUCKET_HIDDEN },
        >(&self.l2, bucket, &l1);
        let mut out = self.forward_bucket_dense::<
            SCReLU,
            BUCKET_HIDDEN,
            BUCKET_OUTPUTS,
            { VALUE_BUCKETS * BUCKET_OUTPUTS },
        >(&self.l3, bucket, &l2);

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

    fn forward_bucket_l1(
        &self,
        bucket: usize,
        input: &[i16; L1 / 2],
    ) -> Accumulator<f32, BUCKET_HIDDEN> {
        let mut res = Accumulator([0.0; BUCKET_HIDDEN]);
        let start = bucket * BUCKET_HIDDEN;
        for (out, (weights, &bias)) in res.0.iter_mut().zip(
            self.l1.weights[start..start + BUCKET_HIDDEN]
                .iter()
                .zip(self.l1.biases.0[start..start + BUCKET_HIDDEN].iter()),
        ) {
            let mut acc = 0i32;
            for (&inp, &w) in input.iter().zip(weights.0.iter()) {
                acc += i32::from(inp) * i32::from(w);
            }

            *out = (acc as f32 / f32::from(QA * QA) + f32::from(bias)) / f32::from(QB);
        }

        res
    }

    fn forward_bucket_dense<
        A: Activation,
        const IN: usize,
        const OUT: usize,
        const TOTAL_OUT: usize,
    >(
        &self,
        layer: &Layer<f32, IN, TOTAL_OUT>,
        bucket: usize,
        input: &Accumulator<f32, IN>,
    ) -> Accumulator<f32, OUT> {
        let mut res = Accumulator([0.0; OUT]);
        let start = bucket * OUT;

        for (dst, &bias) in res
            .0
            .iter_mut()
            .zip(layer.biases.0[start..start + OUT].iter())
        {
            *dst = bias;
        }

        for (inp, weights) in input.0.iter().zip(layer.weights.iter()) {
            let act = A::activate(*inp);
            let bucket_weights = &weights.0[start..start + OUT];
            for (dst, &weight) in res.0.iter_mut().zip(bucket_weights.iter()) {
                *dst += act * weight;
            }
        }

        res
    }
}

fn select_bucket(pieces: u8, threats: u16) -> usize {
    for bucket in PIECE_THREAT_BUCKETS {
        let piece_ok =
            pieces >= bucket.min_piece && bucket.max_piece.map(|max| pieces <= max).unwrap_or(true);
        let threat_ok = threats >= bucket.min_threat
            && bucket.max_threat.map(|max| threats <= max).unwrap_or(true);

        if piece_ok && threat_ok {
            return bucket.idx as usize;
        }
    }

    VALUE_BUCKETS - 1
}
