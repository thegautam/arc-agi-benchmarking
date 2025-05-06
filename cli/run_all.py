import asyncio
import os
import argparse # For potential future enhancements, not used initially
import time # Import time module for benchmarking
from typing import List, Tuple, Dict, Any

# Imports from your project
from main import ARCTester # The main class that runs a single test
from src.utils.task_utils import read_models_config, read_provider_rate_limits
from src.utils.rate_limiter import AsyncRequestRateLimiter

# --- Configuration ---
# List of (config_name, task_id) tuples to run
# These should match the entries from your test_providers.sh script
CONFIGS_AND_TASKS: List[Tuple[str, str]] = [
    ("gpt-4-5-preview-2025-02-27", "14754a24"),
    ("gpt-4-1-2025-04-14", "7bb29440"),
    ("gpt-4o-2024-11-20", "f0afb749"),
    ("claude-3-7-sonnet-20250219", "dc2e9a9d"),
    ("claude_opus", "f83cb3f6"),
    ("gemini-1-5-pro-002", "baf41dbf"),
    ("deepseek_chat", "93b4f4b3"),
    ("o3-mini-2025-01-31-high", "94414823"),
    ("QwQ-32B-Fireworks", "e57337a4"),
    ("grok-3-beta", "d4b1c2b1"),
]

# Default parameters for ARCTester, matching test_providers.sh and main.py defaults
# These could be made configurable via argparse if needed later
DEFAULT_DATA_DIR = "data/arc-agi/data/evaluation"
DEFAULT_SAVE_SUBMISSION_DIR = "." # Saves in the current directory where run_all.py is executed
DEFAULT_OVERWRITE_SUBMISSION = False
DEFAULT_PRINT_SUBMISSION = False # As per main.py, can be enabled by ARCTester's print_logs
DEFAULT_NUM_ATTEMPTS = 2
DEFAULT_RETRY_ATTEMPTS = 2
DEFAULT_PRINT_LOGS = True # To see output from ARCTester

# --- Globals for Orchestrator ---
PROVIDER_RATE_LIMITERS: Dict[str, AsyncRequestRateLimiter] = {}
MODEL_CONFIG_CACHE: Dict[str, Any] = {} # Cache for read_models_config results

def get_model_config(config_name: str):
    if config_name not in MODEL_CONFIG_CACHE:
        MODEL_CONFIG_CACHE[config_name] = read_models_config(config_name)
    return MODEL_CONFIG_CACHE[config_name]

def get_or_create_rate_limiter(provider_name: str, all_provider_limits: Dict) -> AsyncRequestRateLimiter:
    if provider_name not in PROVIDER_RATE_LIMITERS:
        if provider_name not in all_provider_limits:
            print(f"Warning: No rate limit configuration found for provider '{provider_name}' in provider_config.yml. Using default (1 req/sec).")
            # Default fallback if a provider is in models.yml but not provider_config.yml
            rate = 1
            period = 1
        else:
            limits = all_provider_limits[provider_name]
            rate = limits['rate']
            period = limits['period']
        
        print(f"Initializing rate limiter for provider '{provider_name}' with rate={rate} requests per {period} seconds.")
        PROVIDER_RATE_LIMITERS[provider_name] = AsyncRequestRateLimiter(rate=rate, period=period)
    return PROVIDER_RATE_LIMITERS[provider_name]

async def run_single_test_wrapper(config_name: str, task_id: str, limiter: AsyncRequestRateLimiter) -> bool:
    """
    Wrapper to run a single ARCTester task in a thread, respecting the rate limiter.
    Returns True on success, False on failure.
    """
    print(f"[Orchestrator] 🔄 Queuing task: {task_id}, config: {config_name}")

    def synchronous_task_execution():
        """This function will be run in a separate thread."""
        try:
            print(f"[Thread-{task_id}-{config_name}] Spawning ARCTester...")
            arc_solver = ARCTester(
                config=config_name,
                save_submission_dir=DEFAULT_SAVE_SUBMISSION_DIR,
                overwrite_submission=DEFAULT_OVERWRITE_SUBMISSION,
                print_submission=DEFAULT_PRINT_SUBMISSION,
                num_attempts=DEFAULT_NUM_ATTEMPTS,
                retry_attempts=DEFAULT_RETRY_ATTEMPTS,
                print_logs=DEFAULT_PRINT_LOGS
            )
            print(f"[Thread-{task_id}-{config_name}] Starting generate_task_solution...")
            # generate_task_solution returns the attempts or None
            result = arc_solver.generate_task_solution(
                data_dir=DEFAULT_DATA_DIR,
                task_id=task_id
            )
            # Consider a task successful if generate_task_solution doesn't raise an exception
            # and potentially if it returns a non-None result (if that implies success)
            # For now, any non-exception completion is a success from the orchestrator's view.
            print(f"[Thread-{task_id}-{config_name}] ✅ Task completed.")
            return True # Indicates successful execution of the method
        except Exception as e:
            print(f"[Thread-{task_id}-{config_name}] ❌ Exception during ARCTester execution: {e}")
            # import traceback
            # print(traceback.format_exc()) # For more detailed debugging
            return False # Indicates failure

    try:
        # Acquire the limiter before starting the thread-based execution
        async with limiter:
            print(f"[Orchestrator] Rate limiter acquired for: {config_name} (Provider: {limiter._rate}req/{limiter._period}s). Executing task...")
            success = await asyncio.to_thread(synchronous_task_execution)
        
        if success:
            print(f"[Orchestrator] ✅ Successfully processed: {config_name} / {task_id}")
        else:
            print(f"[Orchestrator] ❌ Failed to process: {config_name} / {task_id} (see thread log for details)")
        return success
    except Exception as e:
        # This would catch issues with asyncio.to_thread or the limiter itself
        print(f"[Orchestrator] 💥 Critical error for {config_name} / {task_id}: {e}")
        return False

async def main():
    start_time = time.perf_counter() # Record start time
    print("Starting ARC Test Orchestrator...")
    
    # 1. Load all provider rate limits
    try:
        all_provider_limits = read_provider_rate_limits()
        print(f"Loaded rate limits from provider_config.yml: {list(all_provider_limits.keys())}")
    except FileNotFoundError:
        print("Error: provider_config.yml not found. Please create it with rate limit settings.")
        print("See speedup_plan.md or previous messages for an example structure.")
        print("Proceeding with default rate limits (1 req/sec per provider if encountered)." )
        all_provider_limits = {}
    except Exception as e: # Catch other parsing errors from read_provider_rate_limits
        print(f"Error reading or parsing provider_config.yml: {e}")
        print("Proceeding with default rate limits (1 req/sec per provider if encountered)." )
        all_provider_limits = {}

    # 2. Prepare tasks
    async_tasks = []
    for config_name, task_id in CONFIGS_AND_TASKS:
        try:
            model_config = get_model_config(config_name)
            provider_name = model_config.provider
            limiter = get_or_create_rate_limiter(provider_name, all_provider_limits)
            async_tasks.append(run_single_test_wrapper(config_name, task_id, limiter))
        except ValueError as e: # Catch errors from read_models_config (e.g. model not found)
            print(f"Skipping config '{config_name}' due to error: {e}")
        except Exception as e:
            print(f"Unexpected error setting up task for '{config_name}', '{task_id}': {e}")

    if not async_tasks:
        print("No tasks to run. Exiting.")
        return

    # 3. Run tasks concurrently
    print(f"\nExecuting {len(async_tasks)} tasks concurrently...")
    results = await asyncio.gather(*async_tasks)

    # 4. Report summary
    successful_runs = sum(1 for r in results if r is True)
    failed_runs = len(results) - successful_runs

    print("\n--- Orchestrator Summary ---")
    if failed_runs == 0:
        print(f"✨ All {successful_runs} test configurations completed successfully by the orchestrator.")
        exit_code = 0
    else:
        print(f"💥 {failed_runs} out of {len(results)} test configurations failed or encountered errors during orchestration.")
        exit_code = 1
    
    print("Note: Individual task success/failure is logged by ARCTester within the thread logs.")
    print("Orchestrator failure indicates an issue with running the ARCTester task itself.")
    
    if exit_code != 0:
        print("\nCheck logs above for specific errors.") # Reminder to check logs
    
    end_time = time.perf_counter() # Record end time
    total_duration = end_time - start_time
    print(f"\n--- Orchestrator Timing ---")
    print(f"Total execution time for cli/run_all.py: {total_duration:.2f} seconds")
    
    exit(exit_code)

if __name__ == "__main__":
    # Ensure the current working directory allows imports from src and main.py
    # This usually means running from the root of the model_baseline project.
    # Example: python cli/run_all.py
    asyncio.run(main()) 