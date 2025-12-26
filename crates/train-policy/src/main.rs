pub mod data;
pub mod model;

use acyclib::{
    device::Device,
    trainer::{
        dataloader::{DataLoader, PreparedBatchDevice},
        optimiser::{
            adam::{AdamW, AdamWParams},
            Optimiser,
        },
        schedule::{TrainingSchedule, TrainingSteps},
        Trainer,
    },
    graph::GraphNodeIdTy,
};
use std::sync::Arc;
use bullet_cuda_backend::CudaDevice;

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
    let optimiser = Optimiser::<_, _, AdamW<_>>::new(graph, params).unwrap();

    let mut trainer = Trainer {
        optimiser,
        state: (),
    };

    let save_rate = 40;
    let end_superbatch = 800;
    let initial_lr = 0.001;
    let final_lr = 0.00001;

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

    let mut total_updates = 0;
    let mut clipped_updates = 0;
    let mut superbatch = steps.start_superbatch;
    let mut curr_batch = 0;
    let mut running_error = 0.0;

    dataloader
        .map_batches(steps.batch_size, |batch| {
            let batch = PreparedBatchDevice::new(vec![Arc::new(device)], &batch).unwrap();
            batch.load_into_graph(&mut trainer.optimiser.graph).unwrap();

            trainer.optimiser.graph.zero_grads().unwrap();
            let loss = trainer.optimiser.graph.forward().unwrap();
            trainer.optimiser.graph.backward().unwrap();

            running_error += loss;

            let mut sq_norm = 0.0f32;
            let mut grads = Vec::new();

            for id in ["l0w", "l0b", "l1w", "l1b"] {
                let node_id = trainer
                    .optimiser
                    .graph
                    .weight_idx(id)
                    .unwrap();

                let grad_tensor = trainer.optimiser.graph.get_ref(node_id, GraphNodeIdTy::Gradients);
                let g = grad_tensor.get_dense_vals().unwrap();
                for x in &g {
                    sq_norm += x * x;
                }
                grads.push((id, g));
            }

            let norm = sq_norm.sqrt();
            if norm > 0.25 {
                clipped_updates += 1;
                let scale = 0.25 / norm;

                for (id, mut g) in grads {
                    for x in g.iter_mut() {
                        *x *= scale;
                    }

                    let node_id = trainer
                        .optimiser
                        .graph
                        .weight_idx(id)
                        .unwrap();

                    trainer
                        .optimiser
                        .graph
                        .get_ref(node_id, GraphNodeIdTy::Gradients)
                        .dense_mut()
                        .load_from_slice(None, &g)
                        .unwrap();
                }
            }

            total_updates += 1;

            let lr = (schedule.lr_schedule)(0, superbatch);
            trainer.optimiser.update(1.0 / steps.batch_size as f32, lr).unwrap();

            curr_batch += 1;

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

            superbatch < steps.end_superbatch
        })
        .unwrap();

    model::eval(
        &mut trainer.optimiser.graph,
        node,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    );
}
