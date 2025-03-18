#!/bin/bash

set -e  # Exit on any error

# Run test for a provider
run_test() {
    local config_name=$1
    local task_id=$2
    
    
    echo "🔄 Testing task: ${task_id}, config: ${config_name:-default}"
    
    # Build the command with model as the config
    local cmd="python3 -m main --data_dir data/arc-agi/data/evaluation --config \"${config_name}\" --task_id \"${task_id}\" --save_submission_dir . --print_logs"
    
    # Add config_name if provided (for future use if needed)
    # if [ -n "$config_name" ]; then
    #     cmd="$cmd --config_name \"${config_name}\""
    # fi
    
    # Execute the command
    if eval $cmd; then
        echo "✅ Test completed successfully for ${config_name:+/}${config_name}"
        return 0
    else
        echo "❌ Test failed for ${config_name:+/}${config_name}"
        return 1
    fi
}

export -f run_test

echo "Starting provider tests..."

# Create a temporary file with all configurations
if cat << EOF | parallel --halt now,fail=1 --colsep ' ' -j4 "run_test {1} {2}"
o3_mini e7639916
EOF
then
    echo "✨ All tests completed successfully"
    exit 0
else
    echo "💥 Some tests failed"
    exit 1
fi 