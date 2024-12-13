use crate::{
    chess::{
        consts::{Piece, Side, ValueAttacks, ValueOffsets},
        Attacks,
    },
    Board,
};

const TOTAL_THREATS: usize = 2 * 12 * ValueOffsets::END;
pub const TOTAL: usize = TOTAL_THREATS + 768;

pub fn map_features<F: FnMut(usize)>(pos: &Board, mut f: F) {
    let mut bbs = pos.bbs();

    // flip to stm perspective
    if pos.stm() == Side::WHITE {
        bbs.swap(0, 1);
        for bb in bbs.iter_mut() {
            *bb = bb.swap_bytes()
        }
    }

    // horiontal mirror
    let ksq = (bbs[0] & bbs[Piece::KING]).trailing_zeros();
    if ksq % 8 > 3 {
        for bb in bbs.iter_mut() {
            *bb = flip_horizontal(*bb);
        }
    };

    let mut pieces = [13; 64];
    for side in [Side::WHITE, Side::BLACK] {
        for piece in Piece::PAWN..=Piece::KING {
            let pc = 6 * side + piece - 2;
            map_bb(bbs[side] & bbs[piece], |sq| pieces[sq] = pc);
        }
    }

    let occ = bbs[0] | bbs[1];

    for side in [Side::WHITE, Side::BLACK] {
        let side_offset = 12 * ValueOffsets::END * side;

        for piece in Piece::PAWN..=Piece::KING {
            map_bb(bbs[side] & bbs[piece], |sq| {
                let threats = match piece {
                    Piece::PAWN => Attacks::pawn(sq, side),
                    Piece::KNIGHT => Attacks::knight(sq),
                    Piece::BISHOP => ValueAttacks::BISHOP[sq],
                    Piece::ROOK => ValueAttacks::ROOK[sq],
                    Piece::QUEEN => ValueAttacks::QUEEN[sq],
                    Piece::KING => Attacks::king(sq),
                    _ => unreachable!(),
                } & occ;

                f(TOTAL_THREATS + [0, 384][side] + 64 * (piece - 2) + sq);
                map_bb(threats, |dest| {
                    let idx = map_piece_threat(piece, sq, dest);
                    f(side_offset + pieces[dest] * ValueOffsets::END + idx)
                });
            });
        }
    }
}

fn map_bb<F: FnMut(usize)>(mut bb: u64, mut f: F) {
    while bb > 0 {
        let sq = bb.trailing_zeros() as usize;
        f(sq);
        bb &= bb - 1;
    }
}

fn flip_horizontal(mut bb: u64) -> u64 {
    const K1: u64 = 0x5555555555555555;
    const K2: u64 = 0x3333333333333333;
    const K4: u64 = 0x0f0f0f0f0f0f0f0f;
    bb = ((bb >> 1) & K1) | ((bb & K1) << 1);
    bb = ((bb >> 2) & K2) | ((bb & K2) << 2);
    ((bb >> 4) & K4) | ((bb & K4) << 4)
}

fn map_piece_threat(piece: usize, src: usize, dest: usize) -> usize {
    match piece {
        Piece::PAWN => map_pawn_threat(src, dest),
        Piece::KNIGHT => map_knight_threat(src, dest),
        Piece::BISHOP => map_bishop_threat(src, dest),
        Piece::ROOK => map_rook_threat(src, dest),
        Piece::QUEEN => map_queen_threat(src, dest),
        Piece::KING => map_king_threat(src, dest),
        _ => unreachable!(),
    }
}

fn below(src: usize, dest: usize, table: &[u64; 64]) -> usize {
    (table[src] & ((1 << dest) - 1)).count_ones() as usize
}

fn map_pawn_threat(src: usize, dest: usize) -> usize {
    let diff = if dest > src { dest - src } else { src - dest };
    let attack = if diff == 7 { 0 } else { 1 } + 2 * (src % 8) - 1;
    (src / 8) * 14 + attack
}

fn map_knight_threat(src: usize, dest: usize) -> usize {
    ValueOffsets::KNIGHT[src] + below(src, dest, &ValueAttacks::KNIGHT)
}

fn map_bishop_threat(src: usize, dest: usize) -> usize {
    ValueOffsets::BISHOP[src] + below(src, dest, &ValueAttacks::BISHOP)
}

fn map_rook_threat(src: usize, dest: usize) -> usize {
    ValueOffsets::ROOK[src] + below(src, dest, &ValueAttacks::ROOK)
}

fn map_queen_threat(src: usize, dest: usize) -> usize {
    ValueOffsets::QUEEN[src] + below(src, dest, &ValueAttacks::QUEEN)
}

fn map_king_threat(src: usize, dest: usize) -> usize {
    ValueOffsets::KING[src] + below(src, dest, &ValueAttacks::KING)
}
