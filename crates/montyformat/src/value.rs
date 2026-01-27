use crate::{
    chess::{Castling, Move, Position},
    format::CompressedChessBoard,
    interleave::{interleave, FastDeserialise},
    read_into_primitive, read_primitive_into_vec,
};

pub struct SearchResult {
    pub best_move: Move,
    pub q_value: u16,
    pub d_value: u16,
}

pub struct MontyValueFormat {
    pub startpos: Position,
    pub castling: Castling,
    pub result: f32,
    pub moves: Vec<SearchResult>,
}

impl MontyValueFormat {
    pub fn push(&mut self, _stm: usize, best_move: Move, q_value: u16, d_value: u16) {
        self.moves.push(SearchResult { best_move, q_value, d_value });
    }

    pub fn serialise_into(&self, writer: &mut impl std::io::Write) -> std::io::Result<()> {
        let compressed = CompressedChessBoard::from(self.startpos);

        for bb in compressed.bbs {
            writer.write_all(&bb.to_le_bytes())?;
        }

        writer.write_all(&compressed.stm.to_le_bytes())?;
        writer.write_all(&compressed.enp_sq.to_le_bytes())?;
        writer.write_all(&compressed.rights.to_le_bytes())?;
        writer.write_all(&compressed.halfm.to_le_bytes())?;
        writer.write_all(&compressed.fullm.to_le_bytes())?;

        for side in self.castling.rook_files() {
            for rook in side {
                writer.write_all(&rook.to_le_bytes())?;
            }
        }

        let result = (self.result * 2.0) as u8;
        writer.write_all(&result.to_le_bytes())?;

        for SearchResult { best_move, q_value, d_value } in &self.moves {
            writer.write_all(&u16::from(*best_move).to_le_bytes())?;
            writer.write_all(&q_value.to_le_bytes())?;
            writer.write_all(&d_value.to_le_bytes())?;
        }

        writer.write_all(&[0; 6])?;
        Ok(())
    }

    pub fn deserialise_from(
        reader: &mut impl std::io::BufRead,
        buffer: Vec<SearchResult>,
    ) -> std::io::Result<Self> {
        let mut bbs = [0u64; 4];
        for bb in &mut bbs {
            *bb = read_into_primitive!(reader, u64);
        }

        let stm = read_into_primitive!(reader, u8);
        let enp_sq = read_into_primitive!(reader, u8);
        let rights = read_into_primitive!(reader, u8);
        let halfm = read_into_primitive!(reader, u8);
        let fullm = read_into_primitive!(reader, u16);

        let compressed = CompressedChessBoard {
            bbs,
            stm,
            enp_sq,
            rights,
            halfm,
            fullm,
        };

        let startpos = Position::from(compressed);

        let mut rook_files = [[0; 2]; 2];
        for side in &mut rook_files {
            for rook in side {
                *rook = read_into_primitive!(reader, u8);
            }
        }

        let castling = Castling::from_raw(&startpos, rook_files);

        let result = read_into_primitive!(reader, u8) as f32 / 2.0;

        let mut moves = buffer;
        moves.clear();

        loop {
            let mut buf = [0; 6];
            reader.read_exact(&mut buf)?;

            if buf == [0; 6] {
                break;
            }

            let best_move = u16::from_le_bytes([buf[0], buf[1]]);
            let q_value = u16::from_le_bytes([buf[2], buf[3]]);
            let d_value = u16::from_le_bytes([buf[4], buf[5]]);

            moves.push(SearchResult {
                best_move: best_move.into(),
                q_value,
                d_value,
            });
        }

        Ok(Self {
            startpos,
            castling,
            result,
            moves,
        })
    }

    pub fn interleave(input_paths: &[String], output_path: &str, seed: u64) -> std::io::Result<()> {
        interleave::<Self>(input_paths, output_path, seed)
    }
}

impl FastDeserialise for MontyValueFormat {
    fn deserialise_fast_into_buffer(
        reader: &mut impl std::io::BufRead,
        buffer: &mut Vec<u8>,
    ) -> std::io::Result<()> {
        buffer.clear();

        let mut buf = [0u8; 43];
        reader.read_exact(&mut buf)?;
        buffer.extend_from_slice(&buf);

        while read_primitive_into_vec!(reader, buffer, u32) != 0 {}

        Ok(())
    }
}
