#!/bin/bash

set -e  # Exit on any error

# Run test for a provider
run_test() {
    local provider=$1
    local model=$2
    local task_id=$3
    
    echo "🔄 Testing provider: ${provider}, model: ${model}, task: ${task_id}"
    if python3 -m main \
        --data_dir data/arc-agi/data/evaluation \
        --provider "${provider}" \
        --model "${model}" \
        --task_id "${task_id}" \
        --save_submission_dir . \
        --print_logs; then
        echo "✅ Test completed successfully for ${provider}/${model}"
        return 0
    else
        echo "❌ Test failed for ${provider}/${model}"
        return 1
    fi
}

export -f run_test

echo "Starting provider tests..."

# Create a temporary file with all configurations
if cat << EOF | parallel --halt now,fail=1 --colsep ' ' -j4 "run_test {1} {2} {3}"
anthropic claude-3-5-sonnet-20241022 e7639916
openai gpt-4o 66f2d22f
openai o1 0b17323b
openai o3-mini 85b81ff1
deepseek deepseek-chat d4b1c2b1
gemini gemini-1.5-pro e57337a4
openai gpt-4o-mini 639f5a19
openai o1-mini 551d5bf1
anthropic claude-3-5-haiku-latest 55059096
anthropic claude-3-opus-latest 5783df64
deepseek deepseek-reasoner ca8f78db
gemini gemini-1.5-flash-8b e9bb6954
gemini gemini-1.5-flash e57337a4
gemini gemini-2.0-flash fafd9572
EOF
then
    echo "✨ All tests completed successfully"
    exit 0
else
    echo "💥 Some tests failed"
    exit 1
fi 