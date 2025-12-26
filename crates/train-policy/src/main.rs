pub mod data;
pub mod model;

use acyclib::{
    device::{
        tensor::{Shape, Tensor, TensorRef},
        Device,
    },
    trainer::{
        optimiser::{
            adam::{AdamW, AdamWParams},
            Optimiser,
        },
        schedule::{TrainingSchedule, TrainingSteps},
        dataloader::PreparedBatchDevice,
        Trainer,
    },
};
use bullet_cuda_backend::{
    kernel::{Expr, Kernel, KernelArgs, KernelInput},
    CudaDevice,
};

const REDUCE_KERNEL: &str = r#"
extern "C" __global__ void reduce_sq_norm(const float* inputs, float* output, int index, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        float val = inputs[i];
        local_sum += val * val;
    }
    atomicAdd(output + index, local_sum);
}
"#;

const SCALE_KERNEL: &str = r#"
extern "C" __global__ void scale_tensor(float* gradients, const float* scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float s = *scale;
    for (int i = idx; i < n; i += stride) {
        gradients[i] *= s;
    }
}
"#;

use data::MontyDataLoader;

fn main() {
    let hl = 16384;
    let dataloader = MontyDataLoader::new(
        "/home/privateclient/monty_value_training/interleaved.binpack",
        96000,
        4,
        8,
    );

    let device = CudaDevice::new(0).unwrap();

    let (graph, node) = model::make(device, hl);

    let params = AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    };
    let mut optimiser = Optimiser::<_, _, AdamW<_>>::new(graph, params).unwrap();

    let mut running_error = 0.0;

    let scratch = TensorRef::new(
        Tensor::new(device.clone(), Shape::new(4, 1), None, false).unwrap(),
    );
    let scale_buf = TensorRef::new(
        Tensor::new(device.clone(), Shape::new(1, 1), None, false).unwrap(),
    );

    let weight_ids = ["l0w", "l0b", "l1w", "l1b"];
    let reduce_kernels: Vec<_> = weight_ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let node_id = optimiser.graph.weight_idx(id).unwrap();
            let grad_tensor = optimiser
                .graph
                .get_ref(node_id, acyclib::graph::GraphNodeIdTy::Gradients);
            let n = grad_tensor.single_size();

            unsafe {
                Kernel::new(
                    "reduce_sq_norm".to_string(),
                    REDUCE_KERNEL.to_string(),
                    KernelArgs {
                        inputs: vec![
                            KernelInput::Slice {
                                slice: grad_tensor,
                                layout: None,
                                mutable: false,
                                batched: false,
                                shape: Shape::new(n, 1),
                            },
                            KernelInput::Slice {
                                slice: scratch.clone(),
                                layout: None,
                                mutable: true,
                                batched: false,
                                shape: Shape::new(4, 1),
                            },
                            KernelInput::Size(Expr::Literal(i as i32)),
                            KernelInput::Size(Expr::Literal(n as i32)),
                        ],
                        grid_dim: [Expr::Literal(1024), Expr::Literal(1), Expr::Literal(1)],
                        block_dim: [Expr::Literal(256), Expr::Literal(1), Expr::Literal(1)],
                        shared_mem_bytes: Expr::Literal(0),
                    },
                )
                .unwrap()
            }
        })
        .collect();

    let scale_kernels: Vec<_> = weight_ids
        .iter()
        .map(|id| {
            let node_id = optimiser.graph.weight_idx(id).unwrap();
            let grad_tensor = optimiser
                .graph
                .get_ref(node_id, acyclib::graph::GraphNodeIdTy::Gradients);
            let n = grad_tensor.single_size();

            unsafe {
                Kernel::new(
                    "scale_tensor".to_string(),
                    SCALE_KERNEL.to_string(),
                    KernelArgs {
                        inputs: vec![
                            KernelInput::Slice {
                                slice: grad_tensor,
                                layout: None,
                                mutable: true,
                                batched: false,
                                shape: Shape::new(n, 1),
                            },
                            KernelInput::Slice {
                                slice: scale_buf.clone(),
                                layout: None,
                                mutable: false,
                                batched: false,
                                shape: Shape::new(1, 1),
                            },
                            KernelInput::Size(Expr::Literal(n as i32)),
                        ],
                        grid_dim: [Expr::Literal(1024), Expr::Literal(1), Expr::Literal(1)],
                        block_dim: [Expr::Literal(256), Expr::Literal(1), Expr::Literal(1)],
                        shared_mem_bytes: Expr::Literal(0),
                    },
                )
                .unwrap()
            }
        })
        .collect();

    let save_rate = 40;
    let end_superbatch = 800;
    let initial_lr = 0.001;
    let final_lr = 0.00001;
    let max_grad_norm = 2.0;

    let steps = TrainingSteps {
        batch_size: 16384,
        batches_per_superbatch: 6104,
        start_superbatch: 1,
        end_superbatch,
    };

    let schedule = TrainingSchedule {
        steps,
        log_rate: 64,
        lr_schedule: Box::new(|_, sb| {
            if sb >= end_superbatch {
                return final_lr;
            }

            let lambda = sb as f32 / end_superbatch as f32;
            initial_lr * (final_lr / initial_lr).powf(lambda)
        }),
    };

    let mut trainer = Trainer {
        optimiser,
        state: (),
    };

    let mut clipped_updates = 0;
    let mut total_updates = 0;

    let mut superbatch = steps.start_superbatch;
    let mut curr_batch = 0;

    dataloader
        .map_batches(steps.batch_size, |batch| {
            let device = trainer.optimiser.graph.device();
            let mut batch = PreparedBatchDevice::new(vec![device], &batch).unwrap();
            batch.load_into_graph(&mut trainer.optimiser.graph).unwrap();

            trainer.optimiser.graph.zero_grads().unwrap();
            let loss = trainer.optimiser.graph.forward().unwrap();
            trainer.optimiser.graph.backward().unwrap();

            running_error += loss;

            scratch.borrow_mut().zero().unwrap();

            for k in &reduce_kernels {
                k.execute().unwrap();
            }

            let sums = scratch.get_dense_vals().unwrap();
            let mut sq_norm = 0.0;
            for x in sums {
                sq_norm += x;
            }

            let norm = sq_norm.sqrt();
            if norm > max_grad_norm {
                clipped_updates += 1;
                let scale = max_grad_norm / norm;

                scale_buf
                    .dense_mut()
                    .load_from_slice(None, &[scale])
                    .unwrap();

                for k in &scale_kernels {
                    k.execute().unwrap();
                }
            }

            total_updates += 1;

            let lr = (schedule.lr_schedule)(0, superbatch);
            trainer.optimiser.update(1.0 / steps.batch_size as f32, lr).unwrap();

            curr_batch += 1;

            if curr_batch % 100 == 0 {
                 println!("Batch {curr_batch} | Loss {:.5}", running_error / curr_batch as f32);
            }

            if curr_batch >= steps.batches_per_superbatch {
                let avg_loss = running_error / steps.batches_per_superbatch as f32;
                running_error = 0.0;

                println!(
                    "Superbatch {superbatch} - Loss {avg_loss:.5} - Clipped {:.2}%",
                    clipped_updates as f32 / total_updates as f32 * 100.0
                );
                clipped_updates = 0;
                total_updates = 0;

                if superbatch % save_rate == 0 || superbatch == steps.end_superbatch {
                    println!("Saving Checkpoint");
                    let dir = format!("checkpoints/policy-{superbatch}");
                    let _ = std::fs::create_dir(&dir);
                    trainer.optimiser.write_to_checkpoint(&dir).unwrap();
                    model::save_quantised(&trainer.optimiser.graph, &format!("{dir}/quantised.bin"))
                        .unwrap();
                }

                curr_batch = 0;
                superbatch += 1;
            }

            superbatch >= steps.end_superbatch
        })
        .unwrap();

    model::eval(
        &mut trainer.optimiser.graph,
        node,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    );
}
