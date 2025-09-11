#!/bin/bash

# This script runs the ARC training tasks using the asynchronous orchestrator
# cli/run_all.py instead of invoking each task sequentially. It mirrors the
# behavior of cli/run_training_tasks.sh but leverages run_all.py to process
# tasks concurrently.

# Configuration
MODEL_CONFIG="gpt-5-2025-08-07-medium"
DATA_DIR="data/arc-agi/data/training"
OUTPUT_DIR="results/training"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_run_all_${TIMESTAMP}.log"
TASK_LIMIT=""

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit)
            TASK_LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

# Ensure jq is available for summary extraction
if ! command -v jq &> /dev/null; then
    echo "jq not found. Installing jq..." | tee -a "$LOG_FILE"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install jq
    else
        sudo apt-get update && sudo apt-get install -y jq
    fi
fi

# Prepare optional task list if a limit is specified
TASK_LIST_ARG=""
if [[ -n "$TASK_LIMIT" ]]; then
    TASK_LIST_FILE=$(mktemp)
    find "$DATA_DIR" -maxdepth 1 -name '*.json' | sort | head -n "$TASK_LIMIT" \
        | xargs -n1 basename | sed 's/.json$//' > "$TASK_LIST_FILE"
    TASK_LIST_ARG="--task_list_file $TASK_LIST_FILE"
fi

# Run all training tasks concurrently
uv run python cli/run_all.py \
    --model_configs "$MODEL_CONFIG" \
    --data_dir "$DATA_DIR" \
    --submissions-root "$OUTPUT_DIR" \
    --print_submission \
    $TASK_LIST_ARG \
    --log-level INFO 2>&1 | tee -a "$LOG_FILE"

# Score all submissions and generate results.json
uv run python -m src.arc_agi_benchmarking.scoring.scoring \
    --task_dir "$DATA_DIR" \
    --submission_dir "$OUTPUT_DIR" \
    --results_dir "$OUTPUT_DIR" \
    --print_logs 2>&1 | tee -a "$LOG_FILE"

# Build a CSV summary from results.json and visualize outputs
RESULTS_JSON="${OUTPUT_DIR}/results.json"
SUMMARY_CSV="${OUTPUT_DIR}/results_summary.csv"
if [ -f "$RESULTS_JSON" ]; then
    echo "Task ID,Score,Cost,Attempts,Output Tokens,Duration" > "$SUMMARY_CSV"
    jq -r '.task_results | to_entries[] | ["\"" + .key + "\"", .value.score, .value.cost, .value.attempts, .value.output_tokens, .value.duration] | @csv' "$RESULTS_JSON" >> "$SUMMARY_CSV"

    # Visualize each task's submission
    for task_id in $(jq -r '.task_results | keys[]' "$RESULTS_JSON"); do
        uv run python -m src.arc_agi_benchmarking.scoring.visualize_all \
            --task_id "$task_id" \
            --data_dir "$DATA_DIR" \
            --submission_dir "$OUTPUT_DIR" \
            --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"
    done
fi

# Clean up temporary task list file
if [[ -n "$TASK_LIMIT" ]]; then
    rm -f "$TASK_LIST_FILE"
fi

# Final summary message
echo "" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "Training run (run_all) completed" | tee -a "$LOG_FILE"
if [[ -n "$TASK_LIMIT" ]]; then
    echo "- Tasks processed: $TASK_LIMIT" | tee -a "$LOG_FILE"
fi
echo "- Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "- Results summary: $SUMMARY_CSV" | tee -a "$LOG_FILE"
echo "- Submissions directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
