import asyncio
import os
import argparse # For command-line arguments
import time # Import time module for benchmarking
from typing import List, Tuple, Dict, Any

import sys # Add sys for path manipulation
import logging # Added import for logger

# Add the project root directory to sys.path
# This allows cli/run_all.py to import 'main' and 'src' from the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports from your project
from main import ARCTester # The main class that runs a single test
from src.utils.task_utils import read_models_config, read_provider_rate_limits
from src.utils.rate_limiter import AsyncRequestRateLimiter

# Tenacity imports
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log

# Attempt to import provider-specific exceptions for retrying
try:
    from anthropic import RateLimitError as AnthropicRateLimitError
except ImportError:
    AnthropicRateLimitError = None
    logging.getLogger(__name__).warning("Anthropic SDK not installed or RateLimitError not found. Retries for Anthropic rate limits will not be specific.")

try:
    from openai import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIRateLimitError = None
    logging.getLogger(__name__).warning("OpenAI SDK not installed or RateLimitError not found. Retries for OpenAI rate limits will not be specific.")

try:
    from google.api_core.exceptions import ResourceExhausted as GoogleResourceExhausted
except ImportError:
    GoogleResourceExhausted = None
    logging.getLogger(__name__).warning("Google API Core SDK not installed or ResourceExhausted not found. Retries for Google rate limits will not be specific.")

# Define RETRYABLE_EXCEPTIONS, filtering out any that couldn't be imported
_RETRYABLE_EXCEPTIONS_CLASSES = tuple(
    exc for exc in (AnthropicRateLimitError, OpenAIRateLimitError, GoogleResourceExhausted) if exc is not None
)

if not _RETRYABLE_EXCEPTIONS_CLASSES:
    logging.getLogger(__name__).warning(
        "No specific retryable exception classes were successfully imported. "
        "Retries might not trigger as expected or might catch too broadly if fallback to general Exception is used."
    )
    # Fallback if no specific exceptions are available - you might want to make this stricter
    # For example, by raising an error or using a very limited set of generic exceptions.
    # For now, if none are defined, tenacity will retry on *any* exception by default unless
    # retry_if_exception_type is very carefully used.
    # We will explicitly use retry_if_exception_type with what we have.
    # If _RETRYABLE_EXCEPTIONS_CLASSES is empty, tenacity might not retry as expected unless we provide a default.
    # Let's make it retry on Exception if nothing specific is found, and log a warning.
    EFFECTIVE_RETRYABLE_EXCEPTIONS = (Exception,) if not _RETRYABLE_EXCEPTIONS_CLASSES else _RETRYABLE_EXCEPTIONS_CLASSES
else:
    EFFECTIVE_RETRYABLE_EXCEPTIONS = _RETRYABLE_EXCEPTIONS_CLASSES

# --- Configuration ---
# Define which model configurations to test against the task list.
# These are names from your models.yml file.
MODEL_CONFIGS_TO_TEST: List[str] = [
    "gpt-4o-2024-11-20",
    # "claude_opus", # Commented out claude_opus
    # Add other model config names here as desired, e.g.:
    # "gemini-1-5-pro-002",
    # "deepseek_chat",
]

# Default parameters for ARCTester - these can remain as they were
DEFAULT_DATA_DIR = "data/arc-agi/data/evaluation"
DEFAULT_SAVE_SUBMISSION_DIR_BASE = "submissions_run_all" # Base dir for submissions
DEFAULT_OVERWRITE_SUBMISSION = False
DEFAULT_PRINT_SUBMISSION = False
DEFAULT_NUM_ATTEMPTS = 2
DEFAULT_RETRY_ATTEMPTS = 2
DEFAULT_PRINT_LOGS = False

# --- Globals for Orchestrator ---
PROVIDER_RATE_LIMITERS: Dict[str, AsyncRequestRateLimiter] = {}
MODEL_CONFIG_CACHE: Dict[str, Any] = {}

def get_model_config(config_name: str):
    if config_name not in MODEL_CONFIG_CACHE:
        MODEL_CONFIG_CACHE[config_name] = read_models_config(config_name)
    return MODEL_CONFIG_CACHE[config_name]

def get_or_create_rate_limiter(provider_name: str, all_provider_limits: Dict) -> AsyncRequestRateLimiter:
    if provider_name not in PROVIDER_RATE_LIMITERS:
        if provider_name not in all_provider_limits:
            print(f"Warning: No rate limit configuration found for provider '{provider_name}' in provider_config.yml. Using default (400 req/60s).")
            # Default fallback: 400 requests per 60 seconds
            default_config_rate = 400
            default_config_period = 60
            actual_rate_for_limiter = default_config_rate / default_config_period
            actual_capacity_for_limiter = max(1.0, actual_rate_for_limiter)
        else:
            limits = all_provider_limits[provider_name]
            config_rate = limits['rate']
            config_period = limits['period']
            if config_period <= 0:
                actual_rate_for_limiter = float('inf')
                actual_capacity_for_limiter = float('inf')
                print(f"Warning: Provider '{provider_name}' has period <= 0 in config. Treating as unconstrained.")
            else:
                calculated_rps = config_rate / config_period
                actual_rate_for_limiter = calculated_rps
                actual_capacity_for_limiter = max(1.0, calculated_rps)
        print(f"Initializing rate limiter for provider '{provider_name}' with rate={actual_rate_for_limiter:.2f} req/s, capacity={actual_capacity_for_limiter:.2f}.")
        PROVIDER_RATE_LIMITERS[provider_name] = AsyncRequestRateLimiter(rate=actual_rate_for_limiter, capacity=actual_capacity_for_limiter)
    return PROVIDER_RATE_LIMITERS[provider_name]

async def run_single_test_wrapper(config_name: str, task_id: str, limiter: AsyncRequestRateLimiter,
                                  data_dir: str, save_submission_dir_base: str,
                                  overwrite_submission: bool, print_submission: bool,
                                  num_attempts: int, retry_attempts: int, print_logs: bool) -> bool:
    logger = logging.getLogger(__name__) # Get a logger instance
    print(f"[Orchestrator] 🔄 Queuing task: {task_id}, config: {config_name}")
    save_submission_dir_for_config = os.path.join(save_submission_dir_base, config_name)

    # Apply tenacity retry decorator directly to the synchronous function
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60), # Exponential backoff: 1s, 2s, 4s, 8s... up to 60s
        stop=stop_after_attempt(4), # Max 3 retries (1 initial + 3 retries = 4 attempts)
        retry=retry_if_exception_type(EFFECTIVE_RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING) # Log before sleeping on retry
    )
    def _synchronous_task_execution_attempt_with_tenacity():
        print(f"[Thread-{task_id}-{config_name}] Spawning ARCTester (Executing attempt)...")
        arc_solver = ARCTester(
            config=config_name,
            save_submission_dir=save_submission_dir_for_config,
            overwrite_submission=overwrite_submission,
            print_submission=print_submission,
            num_attempts=num_attempts,
            retry_attempts=retry_attempts, # This is ARCTester's internal retries, distinct from tenacity
            print_logs=print_logs
        )
        print(f"[Thread-{task_id}-{config_name}] Starting generate_task_solution...")
        arc_solver.generate_task_solution(
            data_dir=data_dir,
            task_id=task_id
        )
        print(f"[Thread-{task_id}-{config_name}] ✅ Task attempt completed successfully.")

    try:
        async with limiter:
            logger.info(f"[Orchestrator] Rate limiter acquired for: {config_name}. Executing task with tenacity retries...")
            # Run the decorated synchronous function in a thread
            await asyncio.to_thread(_synchronous_task_execution_attempt_with_tenacity)
        
        logger.info(f"[Orchestrator] ✅ Successfully processed (with tenacity retries if any): {config_name} / {task_id}")
        return True
    except Exception as e:
        # This catches exceptions if all tenacity retries fail, or non-retryable exceptions from ARCTester/to_thread itself.
        # Tenacity's before_sleep_log would have logged retry attempts.
        # The final error from tenacity (RetryError) or other exceptions will be caught here.
        logger.error(f"[Orchestrator] ❌ Failed to process (after all tenacity retries or due to non-retryable error): {config_name} / {task_id}. Error: {type(e).__name__} - {e}", exc_info=True)
        return False

async def main(task_list_file: str, model_configs_to_test: List[str],
               data_dir: str, save_submission_dir_base: str,
               overwrite_submission: bool, print_submission: bool,
               num_attempts: int, retry_attempts: int, print_logs: bool):
    # Basic logging setup - consider moving to a dedicated logging config function
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # Ensure main logger also uses this config.

    start_time = time.perf_counter()
    print(f"Starting ARC Test Orchestrator...")
    print(f"Using task list: {task_list_file}")
    print(f"Testing with model configurations: {model_configs_to_test}")

    # 1. Load task IDs from the file
    task_ids: List[str] = []
    try:
        with open(task_list_file, 'r') as f:
            task_ids = [line.strip() for line in f if line.strip()]
        if not task_ids:
            print(f"Error: No task IDs found in {task_list_file}. Exiting.")
            return
        print(f"Loaded {len(task_ids)} task IDs from {task_list_file}.")
    except FileNotFoundError:
        print(f"Error: Task list file not found: {task_list_file}. Exiting.")
        return

    # 2. Generate all (config_name, task_id) pairs
    all_jobs_to_run: List[Tuple[str, str]] = []
    for config_name in model_configs_to_test:
        for task_id in task_ids:
            all_jobs_to_run.append((config_name, task_id))
    
    if not all_jobs_to_run:
        print("No jobs to run (check model_configs_to_test and task list file). Exiting.")
        return
    
    print(f"Total jobs to process: {len(all_jobs_to_run)}")

    # 3. Load provider rate limits
    try:
        all_provider_limits = read_provider_rate_limits()
        print(f"Loaded rate limits from provider_config.yml: {list(all_provider_limits.keys())}")
    except FileNotFoundError:
        print("Warning: provider_config.yml not found. Using default rate limits (100 req/60s per provider).")
        all_provider_limits = {}
    except Exception as e:
        print(f"Warning: Error reading or parsing provider_config.yml: {e}. Using default rate limits.")
        all_provider_limits = {}

    # 4. Prepare async tasks
    async_tasks_to_execute = []
    for config_name, task_id in all_jobs_to_run:
        try:
            model_config_obj = get_model_config(config_name) # Renamed for clarity
            provider_name = model_config_obj.provider
            limiter = get_or_create_rate_limiter(provider_name, all_provider_limits)
            async_tasks_to_execute.append(run_single_test_wrapper(
                config_name, task_id, limiter,
                data_dir, save_submission_dir_base,
                overwrite_submission, print_submission,
                num_attempts, retry_attempts, print_logs
            ))
        except ValueError as e:
            print(f"Skipping config '{config_name}' for task '{task_id}' due to model config error: {e}")
        except Exception as e:
            print(f"Unexpected error setting up task for '{config_name}', '{task_id}': {e}")

    if not async_tasks_to_execute:
        print("No tasks could be prepared for execution. Exiting.")
        return

    # 5. Run tasks concurrently
    print(f"\nExecuting {len(async_tasks_to_execute)} tasks concurrently...")
    results = await asyncio.gather(*async_tasks_to_execute, return_exceptions=True) # Added return_exceptions

    # 6. Report summary
    successful_runs = sum(1 for r in results if r is True)
    # exceptions_caught = sum(1 for r in results if isinstance(r, Exception))
    # failed_wrapper_calls = sum(1 for r in results if r is False and not isinstance(r, Exception))
    # total_processed = len(results)

    # A result is False if synchronous_task_execution returned False (exception in ARCTester)
    # or if run_single_test_wrapper itself had a critical error.
    # A result is an Exception if asyncio.gather caught it from one of the coroutines.
    orchestrator_level_failures = sum(1 for r in results if r is False or isinstance(r, Exception))

    print("\n--- Orchestrator Summary ---")
    if orchestrator_level_failures == 0:
        print(f"✨ All {successful_runs} test configurations completed successfully by the orchestrator.")
        exit_code = 0
    else:
        print(f"💥 {orchestrator_level_failures} out of {len(results)} test configurations failed or encountered errors during orchestration.")
        # Detailed error logging for exceptions caught by gather
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                original_job = all_jobs_to_run[i] # Assuming results are in order
                print(f"  - Error for {original_job[0]}/{original_job[1]}: {type(res).__name__} - {str(res)}")
            elif res is False:
                original_job = all_jobs_to_run[i]
                print(f"  - Failure reported by wrapper for {original_job[0]}/{original_job[1]} (check thread log)")

        exit_code = 1
    
    print("Note: Individual task success/failure is logged by ARCTester within the thread logs.")
    print("Orchestrator failure indicates an issue with running the ARCTester task itself or an unhandled exception in the wrapper.")
    
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    print(f"\n--- Orchestrator Timing ---")
    print(f"Total execution time for cli/run_all.py: {total_duration:.2f} seconds")
    
    # Ensure metrics are dumped if atexit doesn't run due to early/error exit
    # This might be redundant if atexit always fires, but can be a safeguard.
    # from src.utils.metrics import _dump_all as dump_metrics_now
    # dump_metrics_now() # Consider if this is needed or if atexit is reliable enough

    exit(exit_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARC tasks concurrently using specified model configurations and a task list file.")
    parser.add_argument(
        "--task_list_file", 
        type=str, 
        default="data/task_lists/public_evaluation_v1.txt", 
        help="Path to the .txt file containing task IDs, one per line. Defaults to data/task_lists/public_evaluation_v1.txt"
    )
    parser.add_argument(
        "--model_configs",
        type=str,
        default=",".join(MODEL_CONFIGS_TO_TEST),
        help="Comma-separated list of model configuration names to test (from models.yml). Defaults to pre-defined list."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Data set directory to run. Defaults to {DEFAULT_DATA_DIR}"
    )
    parser.add_argument(
        "--save_submission_dir_base",
        type=str,
        default=DEFAULT_SAVE_SUBMISSION_DIR_BASE,
        help=f"Base folder name to save submissions under. Subfolders per config will be created. Defaults to {DEFAULT_SAVE_SUBMISSION_DIR_BASE}"
    )
    parser.add_argument(
        "--overwrite_submission",
        action="store_true",
        default=DEFAULT_OVERWRITE_SUBMISSION,
        help=f"Overwrite submissions if they already exist. Defaults to {DEFAULT_OVERWRITE_SUBMISSION}"
    )
    parser.add_argument(
        "--print_submission",
        action="store_true",
        default=DEFAULT_PRINT_SUBMISSION,
        help=f"Print submissions to console after each task. Defaults to {DEFAULT_PRINT_SUBMISSION}"
    )
    parser.add_argument(
        "--num_attempts",
        type=int,
        default=DEFAULT_NUM_ATTEMPTS,
        help=f"Number of attempts for each prediction by ARCTester. Defaults to {DEFAULT_NUM_ATTEMPTS}"
    )
    parser.add_argument(
        "--retry_attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help=f"Number of internal retry attempts by ARCTester for failed predictions. Defaults to {DEFAULT_RETRY_ATTEMPTS}"
    )
    parser.add_argument(
        "--print_logs",
        action="store_true",
        default=DEFAULT_PRINT_LOGS, 
        help=f"Enable ARCTester's per-step logging. Defaults to {DEFAULT_PRINT_LOGS}"
    )

    args = parser.parse_args()

    # Post-process model_configs from comma-separated string to list
    model_configs_list = [m.strip() for m in args.model_configs.split(',') if m.strip()]
    if not model_configs_list: # Fallback if parsing results in empty (e.g. only commas/whitespace)
        model_configs_list = MODEL_CONFIGS_TO_TEST


    asyncio.run(main(
        task_list_file=args.task_list_file,
        model_configs_to_test=model_configs_list,
        data_dir=args.data_dir,
        save_submission_dir_base=args.save_submission_dir_base,
        overwrite_submission=args.overwrite_submission,
        print_submission=args.print_submission,
        num_attempts=args.num_attempts,
        retry_attempts=args.retry_attempts,
        print_logs=args.print_logs
    )) 