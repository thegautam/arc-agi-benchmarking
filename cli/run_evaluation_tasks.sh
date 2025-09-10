#!/bin/bash

# Configuration
TASKS=(
    # Add your evaluation task IDs here
    # Example: "task1" "task2" "task3"
    # These should be the task IDs from the evaluation set
)
MODEL_CONFIG="gpt-5-mini-2025-08-07-low"  # Or your preferred model config
PROMPT_NAME="agent_coding_prompt"  # Prompt template to use
DATA_DIR="data/arc-agi/data/evaluation"
OUTPUT_DIR="results/evaluation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/evaluation_run_${TIMESTAMP}.log"

# Create output and logs directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

# Print chosen model and prompt once
echo "Using model config: $MODEL_CONFIG | Prompt: $PROMPT_NAME" | tee -a "$LOG_FILE"

# Initialize results file
echo "Task ID,Score,Cost,Attempts,Output Tokens,Duration" > "${OUTPUT_DIR}/results_summary.csv"

# Function to run a single task
run_task() {
    local task_id=$1
    local task_dir="${OUTPUT_DIR}/${task_id}"
    
    echo "==================================================" | tee -a "$LOG_FILE"
    echo "Running evaluation task: $task_id" | tee -a "$LOG_FILE"
    echo "==================================================" | tee -a "$LOG_FILE"
    
    # Create task directory
    mkdir -p "${task_dir}"
    
    # Run the task
    python main.py \
        --task_id "$task_id" \
        --config "$MODEL_CONFIG" \
        --prompt_name "$PROMPT_NAME" \
        --data_dir "$DATA_DIR" \
        --save_submission_dir "${task_dir}" \
        --print_submission \
        --log-level INFO 2>&1 | tee -a "$LOG_FILE"
    
    # Score the submission
    python -m src.arc_agi_benchmarking.scoring.scoring \
        --task_dir "$DATA_DIR" \
        --submission_dir "${task_dir}" \
        --results_dir "$task_dir" \
        --print_logs 2>&1 | tee -a "$LOG_FILE"
    
    # Generate visualizations
    python src.arc_agi_benchmarking.scoring.visualize_all.py \
        --task_id "$task_id" \
        --data_dir "$DATA_DIR" \
        --submission_dir "${task_dir}" \
        --output_dir "${task_dir}" 2>&1 | tee -a "$LOG_FILE"
       
    # Extract results
    local score_file="${task_dir}/results.json"
    if [ -f "$score_file" ]; then
        local score=$(jq -r '.score' "$score_file" 2>/dev/null || echo "0")
        local cost=$(jq -r '.total_cost' "$score_file" 2>/dev/null || echo "0")
        local attempts=$(jq -r '.total_attempts' "$score_file" 2>/dev/null || echo "0")
        local tokens=$(jq -r '.total_output_tokens' "$score_file" 2>/dev/null || echo "0")
        local duration=$(jq -r '.avg_duration_per_task' "$score_file" 2>/dev/null || echo "0")
        
        # Append to summary
        echo "\"$task_id\",$score,$cost,$attempts,$tokens,$duration" >> "${OUTPUT_DIR}/results_summary.csv"
    fi
}

# Check if specific task IDs are provided as arguments
if [ $# -gt 0 ]; then
    TASKS=("$@")
    echo "Running specific tasks: ${TASKS[*]}" | tee -a "$LOG_FILE"
    # Ensure task IDs don't include .json extension
    TASKS=("${TASKS[@]%.json}")
elif [ ${#TASKS[@]} -eq 0 ]; then
    # If no tasks are specified, find all JSON files in the evaluation directory
    echo "No tasks specified, finding all tasks in $DATA_DIR" | tee -a "$LOG_FILE"
    TASKS=($(find "$DATA_DIR" -maxdepth 1 -type f -name "*.json" -exec basename {} .json \; | sort))
    
    if [ ${#TASKS[@]} -eq 0 ]; then
        echo "No task files found in $DATA_DIR" | tee -a "$LOG_FILE"
        exit 1
    fi
    
    echo "Found ${#TASKS[@]} tasks to run" | tee -a "$LOG_FILE"
fi

# Run all tasks
for task_id in "${TASKS[@]}"; do
    run_task "$task_id"
done

# Print summary
echo "" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "Evaluation completed!" | tee -a "$LOG_FILE"
echo "- Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "- Results summary: ${OUTPUT_DIR}/results_summary.csv" | tee -a "$LOG_FILE"
echo "- Task directories: ${OUTPUT_DIR}/<task_id>/" | tee -a "$LOG_FILE"
