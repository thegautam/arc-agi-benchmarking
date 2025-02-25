#!/bin/bash

set -e  # Exit on any error

# Run test for a provider
run_test() {
    local provider=$1
    local model=$2
    local task_id=$3
    local config_name=$4
    
    echo "🔄 Testing provider: ${provider}, model: ${model}, task: ${task_id}, config: ${config_name:-default}"
    
    # Build the command with optional config_name
    local cmd="python3 -m main --data_dir data/arc-agi/data/evaluation --provider \"${provider}\" --config \"${config_name}\" --task_id \"${task_id}\" --save_submission_dir . --print_logs"
    
    # Add config_name if provided
    if [ -n "$config_name" ]; then
        cmd="$cmd --config_name \"${config_name}\""
    fi
    
    # Execute the command
    if eval $cmd; then
        echo "✅ Test completed successfully for ${provider}/${model}${config_name:+/}${config_name}"
        return 0
    else
        echo "❌ Test failed for ${provider}/${model}${config_name:+/}${config_name}"
        return 1
    fi
}

export -f run_test

echo "Starting provider tests..."

# Create a temporary file with all configurations
if cat << EOF | parallel --halt now,fail=1 --colsep ' ' -j4 "run_test {1} {2} {3} {4}"
anthropic claude_sonnet e7639916
openai gpt4o_standard 66f2d22f
openai o1_short_response 0b17323b
openai o1_long_response 0b17323b
openai o3_mini 85b81ff1
deepseek deepseek_chat d4b1c2b1
gemini gemini_pro e57337a4
openai gpt4o_mini 639f5a19
openai o1_mini 551d5bf1
anthropic claude_haiku 55059096
anthropic claude_opus 5783df64
deepseek deepseek_reasoner ca8f78db
gemini gemini_flash_8b e9bb6954
gemini gemini_flash e57337a4
gemini gemini_flash_2 fafd9572
EOF
then
    echo "✨ All tests completed successfully"
    exit 0
else
    echo "💥 Some tests failed"
    exit 1
fi 