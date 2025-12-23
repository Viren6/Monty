# Helper Tool: Batched LC0 Inference

This folder contains a standalone, stripped-down version of Leela Chess Zero designed specifically for **batched inference** on CPU or GPU.
It allows you to pass a list of FEN strings and receive Value and Policy predictions from a `.pb.gz` network file.

## Features
- **Standalone**: Does not require the full LC0 search logic.
- **Batched Inference**: Processes multiple positions in parallel for higher throughput.
- **Auto-Backend**: Automatically selects the best available backend (CUDA/cudnn if available, else BLAS/Reference).
- **Simple Interface**: Reads FENs from stdin (or piped input), outputs results to stdout.

## Building
To build the tool, run the `build.cmd` script in this directory:
```cmd
.\build.cmd
```
This will generate `build\lc0_inference.exe`.

## Usage
Run the tool with the network path and batch size:
```cmd
build\lc0_inference.exe <path_to_network.pb.gz> <batch_size>
```

### Example
Create a file `fens.txt` with FEN strings (one per line):
```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
```

Run inference:
```cmd
type fens.txt | build\lc0_inference.exe ..\BT4-1024x15x32h-swa-6147500.pb.gz 4
```

## Output Format
The tool prints logs to stderr and predictions to stdout.
Stdout format:
```
policy: move_index:probability ...
value: q: ... d: ... m: ...
```
