<div align="center">

# Monty
#### "MCTS is cool."

</div>

## Compiling
To compile, run `make`. The required networks will be downloaded automatically (and validated).
This requires `make` and a recent enough rust version (see the [MSRV](Cargo.toml)) installed via `rustup` (the official way).

## Development & Project Structure

### Internal match reproducer

The repository ships with an `internal_match` binary that mirrors how we
typically use [fastchess](https://github.com/Disservin/fastchess) for branch
vs. branch testing. It plays both sides of a match, starts each game from the
initial position, and then makes eight random plies so that every game begins
from a unique opening.

```bash
cargo run --release --bin internal_match -- --games 10 --nodes 2000 --hash-mb 1
```

`--games` controls the match length while `--nodes` and `--hash-mb` reproduce
the problematic settings (`nodes=2000`, hash table = 1 MiB). The script also
supports `--threads`, `--random-plies`, and `--max-plies` if you need to tweak
the search configuration.

The binary loads the same policy/value networks as the main engine. Building
with `make` downloads the correct files automatically. If you ever need to
fetch them manually, the expected filenames are recorded in
[`src/networks/policy.rs`](src/networks/policy.rs) and
[`src/networks/value.rs`](src/networks/value.rs); they can be downloaded from
`https://tests.montychess.org/api/nn/<filename>`.

#### Testing

Development of Monty is facilitated by [montytest](https://tests.montychess.org/tests).
Functional patches are required to pass on montytest, with an STC followed by an LTC test.
Nonfunctional patches may be required to pass non-regression test(s) if there are any concerns.

#### Source Code

The main engine code is found in [src/](src/), containing all the search code and network inference code.

There are a number of other crates found in [crates/](crates/):
- [`montyformat`](crates/montyformat/)
    - Core chess implementation
    - Policy/value data formats
    - All other crates depend on this
- [`datagen`](crates/datagen/)
    - Intended to be ran on montytest, there is no need to run it locally (unless testing changes)
- [`train-value`](crates/train-value/)
    - Uses [bullet](https://github.com/jw1912/bullet)
- [`train-policy`](crates/train-policy/)
    - Uses [bullet](https://github.com/jw1912/bullet) & extends it with custom operations

## Terms of use

Monty is free and distributed under the [**GNU Affero General Public License**][license-link] (AGPL v3). Essentially,
this means you are free to do almost exactly what you want with the program, including distributing it among your friends, 
making it available for download from your website, selling it (either by itself or as part of some bigger software package), 
or using it as the starting point for a software project of your own.

The only real limitation is that whenever you distribute Monty in some way, including distribution over a network (such as providing 
access to Monty via a web application or service), you MUST always include the license and the full source code (or a pointer to where 
the source code can be found) to generate the exact binary you are distributing. If you make any changes to the source code, these 
changes must also be made available under AGPL v3.

[license-link]:       https://github.com/official-monty/Monty/blob/master/Copying.txt
