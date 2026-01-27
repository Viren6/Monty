use bullet_lib::{
    acyclib::{
        device::tensor::Shape,
        trainer::{dataloader::{DataLoader, HostDenseMatrix, HostMatrix, HostSparseMatrix, PreparedBatchHost}, DataLoadingError},
    },
    game::inputs::SparseInputType,
};

use crate::structs::WdlPosition;

pub struct SoftWdlDataLoader<L, I> {
    loader: L,
    input_getter: I,
    threads: usize,
}

impl<L, I> SoftWdlDataLoader<L, I>
where
    L: bullet_lib::value::loader::DataLoader<WdlPosition>,
    I: SparseInputType<RequiredDataType = WdlPosition> + Clone + Send + Sync + 'static,
{
    pub fn new(
        loader: L,
        input_getter: I,
        threads: usize,
    ) -> Self {
        Self {
            loader,
            input_getter,
            threads,
        }
    }

    fn prepare(&self, data: &[WdlPosition]) -> PreparedBatchHost {
        let batch_size = data.len();
        let max_active = self.input_getter.max_active();
        let input_size = self.input_getter.num_inputs();
        let output_size = 3; // WDL soft targets

        let sparse_size = max_active * batch_size;
        
        // Accumulators
        let mut stm_vals = vec![0i32; sparse_size];
        let mut nstm_vals = vec![0i32; sparse_size];
        let mut target_vals = vec![0.0f32; output_size * batch_size];

        let chunk_size = batch_size.div_ceil(self.threads);

        std::thread::scope(|s| {
            data.chunks(chunk_size)
                .zip(stm_vals.chunks_mut(max_active * chunk_size))
                .zip(nstm_vals.chunks_mut(max_active * chunk_size))
                .zip(target_vals.chunks_mut(output_size * chunk_size))
                .for_each(|(((data_chunk, stm_chunk), nstm_chunk), target_chunk)| {
                    let inp = &self.input_getter;
                    s.spawn(move || {
                        for (i, pos) in data_chunk.iter().enumerate() {
                            // Sparse Inputs
                            let sparse_offset = max_active * i;
                            let mut j = 0;
                            inp.map_features(pos, |our, opp| {
                                stm_chunk[sparse_offset + j] = our as i32;
                                nstm_chunk[sparse_offset + j] = opp as i32;
                                j += 1;
                            });
                            for k in j..max_active {
                                stm_chunk[sparse_offset + k] = -1;
                                nstm_chunk[sparse_offset + k] = -1;
                            }
                            
                            // Targets (Structure is now [L, D, W])
                            let l = pos.wdl[0];
                            let d = pos.wdl[1];
                            let w = pos.wdl[2];
                            
                            target_chunk[output_size * i + 0] = l;
                            target_chunk[output_size * i + 1] = d;
                            target_chunk[output_size * i + 2] = w;
                        }
                    });
                });
        });

        let mut inputs = std::collections::HashMap::new();
        
        unsafe {
            inputs.insert(
                "stm".to_string(),
                HostMatrix::Sparse(HostSparseMatrix::new(
                    stm_vals,
                    Some(batch_size),
                    Shape::new(input_size, 1),
                    max_active,
                )),
            );
            inputs.insert(
                "nstm".to_string(),
                HostMatrix::Sparse(HostSparseMatrix::new(
                    nstm_vals,
                    Some(batch_size),
                    Shape::new(input_size, 1),
                    max_active,
                )),
            );
            inputs.insert(
                "targets".to_string(),
                HostMatrix::Dense(HostDenseMatrix::new(
                    target_vals,
                    Some(batch_size),
                    Shape::new(output_size, 1),
                )),
            );
        }

        PreparedBatchHost {
            batch_size,
            inputs,
        }
    }
}

impl<L, I> DataLoader for SoftWdlDataLoader<L, I>
where
    L: bullet_lib::value::loader::DataLoader<WdlPosition>,
    I: SparseInputType<RequiredDataType = WdlPosition> + Clone + Send + Sync + 'static,
{
    type Error = DataLoadingError;

    fn map_batches<F: FnMut(PreparedBatchHost) -> bool>(
        self,
        batch_size: usize,
        mut f: F,
    ) -> Result<(), Self::Error> {

        self.loader.map_batches(0, batch_size, |batch| {
            let prepared = self.prepare(batch);
            f(prepared)
        });
        
        Ok(())
    }
}
