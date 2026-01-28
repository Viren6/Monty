#!/usr/bin/env bash
set -euo pipefail

# ---- Config (override via env vars) ----
MONTY_BIN="${MONTY_BIN:-./monty}"
BOOK="${BOOK:-DFRC_openings.epd}"
GAMES="${GAMES:-3750000}"
OUT_PREFIX="${OUT_PREFIX:-output-policy-dfrc-big}"

NUM_GPUS="${NUM_GPUS:-4}"           # gpus 0..3
PROCS_PER_GPU="${PROCS_PER_GPU:-2}" # 2 processes per gpu

DFRC_MODE="${DFRC_MODE:-1}"         # set to 1/true/yes to append --dfrc
ONNX_MODE="${ONNX_MODE:-0}"         # set to 1/true/yes to append --onnx

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
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  for idx in $(seq 1 "$PROCS_PER_GPU"); do
    session="gpu${gpu}_${idx}"
    outfile="${OUT_PREFIX}-gpu${gpu}-${idx}.binpack"

    # Skip if session already exists
    if screen -ls | grep -qE "[[:space:]]*[0-9]+\.${session}[[:space:]]"; then
      echo "session ${session} already exists, skipping"
      continue
    fi

    cmd=(env CUDA_VISIBLE_DEVICES="$gpu" "$MONTY_BIN" -b "$BOOK" -g "$GAMES" -o "$outfile")

    # If DFRC_MODE enabled, append --dfrc
    case "${DFRC_MODE,,}" in
      1|true|yes|y|on) cmd+=(--dfrc) ;;
    esac

    # If ONNX_MODE enabled, append --onnx
    case "${ONNX_MODE,,}" in
      1|true|yes|y|on) cmd+=(--onnx) ;;
    esac

    printf -v cmd_str '%q ' "${cmd[@]}"

    echo "Starting ${session}: ${cmd[*]}"
    screen -dmS "$session" bash -lc "$cmd_str"
  done
done

echo
echo "Done. Sessions:"
screen -ls | grep -E "gpu[0-9]+_[0-9]+" || true
echo
echo "Attach with: screen -r gpu3_1"
echo "Kill one with: screen -S gpu3_1 -X quit"
