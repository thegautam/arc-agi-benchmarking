#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cli/run_task.sh <task_id> [<config> [<data_dir> [<work_dir>]]] \
#       [--config NAME] [--data-dir PATH] [--work-dir PATH] \
#       [--print-submission] [--log-level LEVEL]
#
# Runs a single ARC task end-to-end by invoking:
#   1) python main.py (prediction)
#   2) python -m src.arc_agi_benchmarking.scoring.scoring (scoring)
#   3) python -m src.arc_agi_benchmarking.scoring.visualize_all (visualization)
#
# Example:
#   cli/run_task.sh a85d4709 --print-submission --log-level INFO

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <task_id> [<config> [<data_dir> [<work_dir>]]] [--config NAME] [--data-dir PATH] [--work-dir PATH] [--print-submission] [--log-level LEVEL]" >&2
  exit 1
fi

TASK_ID="$1"; shift

# Defaults
DEFAULT_MODEL_CONFIG="gpt-5-mini-2025-08-07-low"
DEFAULT_DATA_DIR="data/arc-agi/data/training"
DEFAULT_WORK_BASE="results/training"

# Positional overrides (optional). If a token starts with --, treat it as a flag.
MODEL_CONFIG="$DEFAULT_MODEL_CONFIG"
if [ "${1-}" != "" ] && [[ ! "${1-}" =~ ^-- ]]; then
  MODEL_CONFIG="$1"; shift
fi

DATA_DIR="$DEFAULT_DATA_DIR"
if [ "${1-}" != "" ] && [[ ! "${1-}" =~ ^-- ]]; then
  DATA_DIR="$1"; shift
fi

if [ "${1-}" != "" ] && [[ ! "${1-}" =~ ^-- ]]; then
  WORK_DIR="$1"; shift
else
  WORK_DIR="$DEFAULT_WORK_BASE/$TASK_ID"
fi

PRINT_SUBMISSION_FLAG=""
LOG_LEVEL="INFO"

while (( "$#" )); do
  case "$1" in
    --print-submission)
      PRINT_SUBMISSION_FLAG="--print_submission"
      shift
      ;;
    --config)
      if [ -n "${2-}" ]; then
        MODEL_CONFIG="$2"; shift 2
      else
        echo "--config requires a value" >&2; exit 1
      fi
      ;;
    --data-dir)
      if [ -n "${2-}" ]; then
        DATA_DIR="$2"; shift 2
      else
        echo "--data-dir requires a value" >&2; exit 1
      fi
      ;;
    --work-dir)
      if [ -n "${2-}" ]; then
        WORK_DIR="$2"; shift 2
      else
        echo "--work-dir requires a value" >&2; exit 1
      fi
      ;;
    --log-level)
      if [ -n "${2-}" ]; then
        LOG_LEVEL="$2"; shift 2
      else
        echo "--log-level requires a value" >&2; exit 1
      fi
      ;;
    *)
      echo "Unknown option: $1" >&2; exit 1
      ;;
  esac
done

mkdir -p "$WORK_DIR"

echo "[run_task.sh] Predicting: task=$TASK_ID, config=$MODEL_CONFIG"
python main.py \
  --task_id "$TASK_ID" \
  --config "$MODEL_CONFIG" \
  --data_dir "$DATA_DIR" \
  --save_submission_dir "$WORK_DIR" \
  $PRINT_SUBMISSION_FLAG \
  --log-level "$LOG_LEVEL"

echo "[run_task.sh] Scoring: task=$TASK_ID"
python -m src.arc_agi_benchmarking.scoring.scoring \
  --task_dir "$DATA_DIR" \
  --submission_dir "$WORK_DIR" \
  --results_dir "$WORK_DIR" \
  --print_logs

echo "[run_task.sh] Visualizing: task=$TASK_ID"
python -m src.arc_agi_benchmarking.scoring.visualize_all \
  --task_id "$TASK_ID" \
  --data_dir "$DATA_DIR" \
  --submission_dir "$WORK_DIR" \
  --output_dir "$WORK_DIR"

echo "[run_task.sh] Done: $TASK_ID"
