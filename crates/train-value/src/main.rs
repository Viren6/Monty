mod dataloader;
mod input;
mod structs;
mod soft_loader;

use dataloader::MontyBinpackLoader;
use input::ThreatInputs;

use bullet_lib::{
    game::inputs::SparseInputType,
    nn::{
        optimiser::{AdamW, AdamWParams},
        InitSettings, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr::ExponentialDecayLR, wdl::ConstantWDL, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::ValueTrainerBuilder,
};

use bullet_lib::acyclib::graph::save::GraphWeights;

use montyformat::chess::{Move, Position};

fn main() {
    let experiment_name = "3072T".to_string();

    // architecture
    let input_features = ThreatInputs;
    let l1 = 3072;
    let l2 = 16;
    let l3 = 128;

    // training schedule
    let initial_lr = 0.001;
    let final_lr = 0.0000001;
    let superbatches = 4000;

    // data
    let data_path = "/home/privateclient/monty_value_training/interleaved-value.binpack";
    let dataloader_buffer_size_mb = 96000;
    let dataloader_threads = 8;

    let quantisations = vec![
        ("pst", SavedFormat::id("pst")),
        ("l0w", SavedFormat::id("l0w").quantise::<i8>(128).round()),
        ("l0b", SavedFormat::id("l0b").quantise::<i8>(128).round()),
        ("l1w", SavedFormat::id("l1w").quantise::<i16>(1024).transpose().round()),
        ("l1b", SavedFormat::id("l1b").quantise::<i16>(1024).round()),
        ("l2w", SavedFormat::id("l2w")),
        ("l2b", SavedFormat::id("l2b")),
        ("l3w", SavedFormat::id("l3w")),
        ("l3b", SavedFormat::id("l3b")),
    ];

    let trainer_formats: Vec<SavedFormat> = quantisations.iter().map(|(_, f)| f.clone()).collect();

    let mut trainer = ValueTrainerBuilder::default()
        .wdl_output()
        .inputs(input_features)
        .optimiser(AdamW)
        .save_format(&trainer_formats)
        .build_custom(|builder, inputs, targets| {
            let num_inputs = input_features.num_inputs();

            let pst = builder.new_weights("pst", Shape::new(3, num_inputs), InitSettings::Zeroed);
            let l0 = builder.new_affine("l0", num_inputs, l1);
            let l1 = builder.new_affine("l1", l1 / 2, l2);
            let l2 = builder.new_affine("l2", l2, l3);
            let l3 = builder.new_affine("l3", l3, 3);

            l0.init_with_effective_input_size(input_features.max_active());

            let l0 = l0.forward(inputs).crelu().pairwise_mul();
            let l1 = l1.forward(l0).screlu();
            let l2 = l2.forward(l1).screlu();
            let l3 = l3.forward(l2);
            let out = l3 + pst.matmul(inputs);

            let ones = builder.new_constant(Shape::new(1, 3), &[1.0; 3]);
            let loss = ones.matmul(out.softmax_crossentropy_loss(targets));

            (out, loss)
        });

    let optimiser_params = AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    };

    trainer.optimiser.set_params(optimiser_params);

    let schedule = TrainingSchedule {
        net_id: experiment_name,
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 65_536,
            batches_per_superbatch: 1526,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: ConstantWDL { value: 1.0 },
        lr_scheduler: ExponentialDecayLR {
            initial_lr,
            final_lr,
            final_superbatch: superbatches,
        },
        save_rate: 200,
        // note: removed save_rate/etc from builder context, handled manually
    };

    let settings = LocalSettings {
        threads: 2,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 32,
    };

    fn filter(_: &Position, _: Move, _: u16, _: u16) -> bool {
        true
    }

    let data_loader = MontyBinpackLoader::new(
        data_path,
        dataloader_buffer_size_mb,
        dataloader_threads,
        filter,
    );

    let loader = soft_loader::SoftWdlDataLoader::new(data_loader, input_features, 2);


    
    let steps = schedule.steps;
    let lr_scheduler = schedule.lr_scheduler;
    let save_rate = schedule.save_rate;

    trainer
        .train_custom(
            bullet_lib::acyclib::trainer::schedule::TrainingSchedule {
                steps,
                log_rate: 128,
                lr_schedule: Box::new(move |a, b| {
                    use bullet_lib::trainer::schedule::lr::LrScheduler;
                    lr_scheduler.lr(a, b)
                }),
            },
            loader,
            |_trainer, _superbatch, _batch, _loss| {},
            move |trainer, superbatch| {
                if superbatch % save_rate == 0 || superbatch == steps.end_superbatch {
                    let path = format!("{}/checkpoint-{}", settings.output_directory, superbatch);
                    std::fs::create_dir_all(&path).unwrap();
                    
                    println!("Saving Checkpoint");
                    let graph = &trainer.optimiser.graph;
                    let weights = GraphWeights::from(graph);

                    for (name, fmt) in &quantisations {
                        let bytes = fmt.write_to_byte_buffer(&weights).unwrap();
                        std::fs::write(format!("{}/{}.bin", path, name), bytes).unwrap();
                    }
                    println!("Saved checkpoint to {}", path);
                }
            },
        )
        .unwrap();

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}
