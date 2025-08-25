#!/bin/bash

# Configuration
TASKS=(
    "a85d4709" "c8cbb738" "8e1813be" "a699fb00" "5c2c9af4"
    "44f52bb0" "23581191" "94f9d214" "f9012d9b" "4258a5f9"
)
MODEL_CONFIG="gpt-5-mini-2025-08-07-low"
DATA_DIR="data/arc-agi/data/training"
OUTPUT_DIR="training_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_run_${TIMESTAMP}.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

# Initialize results file
echo "Task ID,Score,Cost,Attempts,Output Tokens,Duration" > "${OUTPUT_DIR}/results_summary.csv"

# Function to run a single task
run_task() {
    local task_id=$1
    local task_dir="${OUTPUT_DIR}/${task_id}"
    
    echo "==================================================" | tee -a "$LOG_FILE"
    echo "Running task: $task_id" | tee -a "$LOG_FILE"
    echo "==================================================" | tee -a "$LOG_FILE"
    
    # Create task directory
    mkdir -p "${task_dir}"
    
    # Run the task
    python main.py \
        --task_id "$task_id" \
        --config "$MODEL_CONFIG" \
        --data_dir "$DATA_DIR" \
        --save_submission_dir "${task_dir}" \
        --print_submission \
        --log-level INFO 2>&1 | tee -a "$LOG_FILE"
       
    # Score the task
    python -m src.arc_agi_benchmarking.scoring.scoring \
        --task_dir "$DATA_DIR" \
        --submission_dir "${task_dir}" \
        --results_dir "$task_dir" \
        --print_logs 2>&1 | tee -a "$LOG_FILE" 
   
    # Visualize the results
    python visualize_all.py \
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
        local tokens=$(jq -r '.avg_output_tokens_per_task' "$score_file" 2>/dev/null || echo "0")
        local duration=$(jq -r '.avg_duration_per_task' "$score_file" 2>/dev/null || echo "0")
        
        # Append to summary
        echo "\"$task_id\",$score,$cost,$attempts,$tokens,$duration" >> "${OUTPUT_DIR}/results_summary.csv"
    fi
}

# Install jq if not present
if ! command -v jq &> /dev/null; then
    echo "jq not found. Installing jq..." | tee -a "$LOG_FILE"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install jq
    else
        sudo apt-get update && sudo apt-get install -y jq
    fi
fi

# Run all tasks
for task in "${TASKS[@]}"; do
    run_task "$task"
done

# Score everything
python -m src.arc_agi_benchmarking.scoring.scoring \
    --task_dir "$DATA_DIR" \
    --submission_dir "$OUTPUT_DIR" \
    --results_dir "$task_dir" \
    --print_logs 2>&1 | tee -a "$LOG_FILE"

# Print summary
echo ""
echo "==================================================" | tee -a "$LOG_FILE"
echo "Training Run Summary" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "Model: $MODEL_CONFIG" | tee -a "$LOG_FILE"
echo "Tasks completed: $(($(wc -l < "${OUTPUT_DIR}/results_summary.csv") - 1))" | tee -a "$LOG_FILE"
echo "Total cost: $(awk -F, 'NR>1 {sum+=$3} END {print sum}' "${OUTPUT_DIR}/results_summary.csv" 2>/dev/null || echo "0")" | tee -a "$LOG_FILE"
echo "Average score: $(awk -F, 'NR>1 {sum+=$2; count++} END {if(count>0) printf "%.2f%%", sum/count*100; else printf "0%%"}' "${OUTPUT_DIR}/results_summary.csv" 2>/dev/null)" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "Detailed results saved to:" | tee -a "$LOG_FILE"
echo "- Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "- Results summary: ${OUTPUT_DIR}/results_summary.csv" | tee -a "$LOG_FILE"
echo "- Task directories: ${OUTPUT_DIR}/<task_id>/" | tee -a "$LOG_FILE"
