#!/usr/bin/env bash
set -euo pipefail

# ---- Config (override via env vars) ----
MONTY_BIN="${MONTY_BIN:-./monty}"
BOOK="${BOOK:-datagenbook_UHO_cdb_82_102.epd}"
GAMES="${GAMES:-100000}"
OUT_PREFIX="${OUT_PREFIX:-output-value-stockfish}"
NODES="${NODES:-10000}" # 10k nodes per move for Stockfish
THREADS="${THREADS:-1}" # Stockfish processes to spawn from this instance (usually 1 per screen session)
BATCH_SIZE="${BATCH_SIZE:-128}" # Games per batch

DFRC_MODE="${DFRC_MODE:-0}"         # Default Off for Value? User said "Remember the dfrc option... same behaviour"

# ---- Checks ----
if ! command -v screen >/dev/null 2>&1; then
  echo "error: 'screen' is not installed." >&2
  exit 1
fi

if [[ ! -x "$MONTY_BIN" ]]; then
  echo "error: monty binary not found/executable at: $MONTY_BIN" >&2
  exit 1
fi

# ---- Launch ----
# For value datagen, we might want multiple screen sessions, each running monty with -t 1 (1 SF process)
# Or one monty with -t N (N SF processes).
# User said "The number of games played in parallel is set by the batch_size variable in the main script" -> Wait.
# User said "Each game will be on 1 thread again" ... "Make sure the game hash is set to 64MB".
# "I think the best way is to spawn all the stockfish processes... batch size... send the number of games equal to the batch size to each stockfish process".
# So: 1 Stockfish Process (managed by 1 Monty Thread) handles BATCH_SIZE games sequentially (or in parallel internally? No, SF is single threaded per game usually).
# User said "This will be done in parallel, 1 game per thread." -> implies many threads?
# "The number of games played in parallel is set by the batch_size variable".
# This logic is a bit conflicting.
# "Send the number of games equal to the batch size to each stockfish process at once, and then values are returned after all games finish".
# This implies Stockfish plays the batch_size games *sequentially* (if it is 1 process) or *in parallel* (if it spawns threads)?
# Stockfish `datagen` implementation I wrote loops `batch_size` times. So it is sequential.
# But we can run multiple Stockfish processes.
# Let's say we want 4 parallel streams. We run Monty with `-t 4`. 
# Each thread spawns 1 SF process. That SF process runs batches of 128 games.
# Total parallel = 4 * 1 = 4 games/processes actively running?
# Or does BATCH_SIZE imply something else?
# "This send the number of games equal to the batch size to each stockfish process at once"
# This suggests IO optimization.

# I will use 4 sessions, each with threads=1 (1 SF process).
NUM_SESSIONS="${NUM_SESSIONS:-4}"

for idx in $(seq 1 "$NUM_SESSIONS"); do
    session="val_gen_${idx}"
    outfile="${OUT_PREFIX}-${idx}.binpack"
    
    if screen -ls | grep -qE "[[:space:]]*[0-9]+\.${session}[[:space:]]"; then
      echo "session ${session} already exists, skipping"
      continue
    fi
    
    # We pass --features value via `make gen-value` which produced the binary?
    # No, `monty` is the binary. `make gen-value` produced `monty` (or `datagen` renamed?).
    # user said "The make target for this pipeline will be make gen-value ... And make gen-value for the Monty side."
    # Makefile shows `gen-value` calls `cargo ... --bin datagen`.
    # So the binary is likely `target/release/datagen` (or `datagen.exe`).
    # I should check where it is put. Makefile: `LINK := -- --emit link=$(NAME)` where NAME=monty.exe?
    # `gen-value`: `invokes ... --bin datagen ... $(LINK)`.
    # So it overwrites `monty.exe`? Yes.
    
    cmd=(env "$MONTY_BIN" -b "$BOOK" -g "$GAMES" -o "$outfile" -t "$THREADS" -n "$NODES")
    
    case "${DFRC_MODE,,}" in
      1|true|yes|y|on) cmd+=(--dfrc) ;;
    esac
    
    printf -v cmd_str '%q ' "${cmd[@]}"
    
    echo "Starting ${session}: ${cmd[*]}"
    screen -dmS "$session" bash -lc "$cmd_str"
done

echo "Done. Sessions:"
screen -ls | grep "val_gen" || true
