import sys
import os

from arc_agi_benchmarking.prompts.scene_builder import compare_grids

# Added: Add the src directory to sys.path to allow direct execution of main.py
# This assumes main.py is in the project root and 'src' is a subdirectory.
_project_root = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_project_root, 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import json
from arc_agi_benchmarking.adapters import ProviderAdapter, AnthropicAdapter, OpenAIAdapter, DeepseekAdapter, GeminiAdapter, HuggingFaceFireworksAdapter, FireworksAdapter, GrokAdapter, OpenRouterAdapter, XAIAdapter
from arc_agi_benchmarking.cache.cached_adapter import CachedAdapter
from dotenv import load_dotenv
import arc_agi_benchmarking.utils as utils
from arc_agi_benchmarking.utils.metrics import timeit, set_metrics_enabled
from arc_agi_benchmarking.schemas import ARCTaskOutput, ARCPair, Attempt
from arc_agi_benchmarking.prompts.prompt_manager import convert_task_pairs_to_prompt, _load_prompt
from typing import List, Any, Optional
import argparse
import logging
from arc_agi_benchmarking.scoring.execute_llm_code import run_code_attempt, is_grid, run_on_training_and_compare

logger = logging.getLogger(__name__)

load_dotenv()

class ARCTester:
    def __init__(self, config: str, save_submission_dir: str, overwrite_submission: bool, print_submission: bool, num_attempts: int, retry_attempts: int, prompt_name: str = "simple_coding_prompt"):
        self.config = config
        self.model_config = utils.read_models_config(config)
        self.provider = self.init_provider(self.model_config.provider)
        self.save_submission_dir = save_submission_dir
        self.overwrite_submission = overwrite_submission
        self.print_submission = print_submission
        self.num_attempts = num_attempts
        self.retry_attempts = retry_attempts
        self.prompt_name = prompt_name

    def init_provider(self, provider_name: str) -> ProviderAdapter:
        if provider_name == "anthropic":
            provider = AnthropicAdapter(self.config)
        elif provider_name == "openai":
            provider = OpenAIAdapter(self.config)
        elif provider_name == "deepseek":
            provider = DeepseekAdapter(self.config)
        elif provider_name == "gemini":
            provider = GeminiAdapter(self.config)
        elif provider_name == "huggingfacefireworks":
            provider = HuggingFaceFireworksAdapter(self.config)
        elif provider_name == "fireworks":
            provider = FireworksAdapter(self.config)
        elif provider_name == "grok":
            provider = GrokAdapter(self.config)
        elif provider_name == "openrouter":
            provider = OpenRouterAdapter(self.config)
        elif provider_name == "xai":
            provider = XAIAdapter(self.config)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        # Optional caching layer (enabled via env var ARC_AGI_CACHE_ENABLED)
        cache_enabled = os.getenv("ARC_AGI_CACHE_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
        if cache_enabled:
            cache_dir = os.getenv("ARC_AGI_CACHE_DIR")
            zero_cost_on_hit = os.getenv("ARC_AGI_CACHE_ZERO_COST_ON_HIT", "1").lower() in {"1", "true", "yes", "on"}
            provider = CachedAdapter(provider, cache_dir=cache_dir, enabled=True, zero_cost_on_hit=zero_cost_on_hit)
            logger.debug(f"Provider caching enabled. Cache dir: {cache_dir or 'logs/provider_cache'}; zero_cost_on_hit={zero_cost_on_hit}")
        return provider
        
    def predict_task_output(self, training_pairs: List[ARCPair], test_input: ARCPair, task_id: str, test_id: str, pair_index: int, bypass_cache: bool = False, prompt_feedback: Optional[str] = None, messages: Optional[List[dict]] = None):
        """
        Given a task, predict the test output. This reponse may need parsing.

        Convert the training pairs and test pairs into a prompt
        Give the prompt to the LLM
        return the response
        """

        # Build either a single prompt string (legacy) or use provided messages for chat
        prompt = None
        if messages is None:
            # Convert the training pairs and test pairs into a prompt
            prompt = convert_task_pairs_to_prompt(training_pairs, test_input, prompt_name=self.prompt_name, prompt_feedback=prompt_feedback)

        logger.debug(f"Using model config: {self.model_config.name} ({self.model_config.provider})")
        if prompt is not None:
            logger.debug(f"Prompt length: {len(prompt)} characters")
        
        try:
            logger.debug("Waiting for model response...")
            # If provider is wrapped with cache, allow bypass for fresh calls
            try:
                if isinstance(self.provider, CachedAdapter):
                    if messages is not None:
                        response: Attempt = self.provider.make_prediction(
                            prompt, task_id=task_id, test_id=test_id, pair_index=pair_index, bypass_cache=bypass_cache, messages=messages
                        )
                    else:
                        response: Attempt = self.provider.make_prediction(
                            prompt, task_id=task_id, test_id=test_id, pair_index=pair_index, bypass_cache=bypass_cache
                        )
                else:
                    if messages is not None:
                        response: Attempt = self.provider.make_prediction(
                            prompt, task_id=task_id, test_id=test_id, pair_index=pair_index, messages=messages
                        )
                    else:
                        response: Attempt = self.provider.make_prediction(
                            prompt, task_id=task_id, test_id=test_id, pair_index=pair_index
                        )
            except TypeError:
                # Fallback for providers that don't support bypass_cache kwarg
                if messages is not None:
                    response: Attempt = self.provider.make_prediction(
                        prompt, task_id=task_id, test_id=test_id, pair_index=pair_index, messages=messages
                    )
                else:
                    response: Attempt = self.provider.make_prediction(
                        prompt, task_id=task_id, test_id=test_id, pair_index=pair_index
                    )
        
            # Save the code if present
            code_str = getattr(response, "code", None)
            if code_str:
                with open(os.path.join(self.save_submission_dir, f"{task_id}.py"), "w") as f:
                    f.write(code_str)

                # Run on training
                train_outputs, stdout_pieces, error = run_on_training_and_compare(
                    code_str,
                    training_pairs,
                    timeout=10,
                )

                # Build feedback for the next attempt based on training comparisons
                feedback_lines = []
                if error:
                    feedback_lines.append("Error occurred while running code on training examples:")
                    feedback_lines.append(str(error))
                else:
                    feedback_lines.append("Training comparison results (per example):")
                    for idx, (input_pair, actual_output) in enumerate(zip(training_pairs, train_outputs)):
                        score, message = compare_grids(input_pair.output, actual_output)
                        feedback_lines.append(f"- Example {idx}: score={score:.3f}. {message}")
                    if stdout_pieces:
                        # Join all captured stdout entries from training runs and keep concise
                        combined_stdout = "\n".join(s for s in stdout_pieces if isinstance(s, str))
                        trimmed_stdout = combined_stdout.strip()
                        if len(trimmed_stdout) > 800:
                            trimmed_stdout = trimmed_stdout[:800] + "... [truncated]"
                        feedback_lines.append("Auxiliary analysis stdout (truncated if long):")
                        feedback_lines.append(trimmed_stdout)

                # Attach feedback to the Attempt for use in subsequent attempts
                try:
                    response.prompt_feedback = "\n".join(feedback_lines) if feedback_lines else None
                except Exception:
                    # Be resilient if provider returned a non-standard object
                    pass

                # Determine if ALL training examples passed exactly
                try:
                    training_all_passed = False
                    if not error and isinstance(train_outputs, list) and len(train_outputs) == len(training_pairs):
                        training_all_passed = all(
                            compare_grids(inp.output, out)[0] == 1.0
                            for inp, out in zip(training_pairs, train_outputs)
                        )
                    # Propagate this signal via metadata.kwargs for outer-loop early stopping
                    if getattr(response, 'metadata', None) and getattr(response.metadata, 'kwargs', None) is not None:
                        response.metadata.kwargs["training_all_passed"] = training_all_passed
                except Exception:
                    # Non-fatal if anything goes wrong computing this
                    pass

                # Preserve previous prints for visibility
                for idx, (input_pair, actual_output) in enumerate(zip(training_pairs, train_outputs)):
                    score, message = compare_grids(input_pair.output, actual_output)
                    if error:
                        logger.error(error)
                    elif score < 1.0:
                        per_example_stdout = ""
                        if isinstance(stdout_pieces, list) and idx < len(stdout_pieces):
                            s = stdout_pieces[idx]
                            per_example_stdout = s if isinstance(s, str) else ""
                        detail = f"{message} {per_example_stdout}".strip()
                        logger.debug(detail)
                        logger.debug(score)

            # Execute simple transform(grid) code safely and use correct grid attribute
            test_output = None
            if code_str:
                # Prefer sandboxed execution
                try:
                    test_output, _, _ = run_code_attempt(
                        code_str,
                        test_input.input,
                        timeout=10,
                    )
                except Exception:
                    test_output = None

            # Update the attempt answer only if we produced an output
            if test_output is not None:
                response.answer = test_output
            else:
                logger.info("Code execution failed, missing, or returned invalid output; keeping parsed answer.")                      
            
            logger.debug(f"Response received - Cost: ${response.metadata.cost.total_cost:.6f}, Usage: {response.metadata.usage.total_tokens} tokens")
            
            # In verbose mode, show more detailed response info
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Response details - Model: {response.metadata.model}")
                if hasattr(response.metadata.usage, 'completion_tokens_details') and response.metadata.usage.completion_tokens_details:
                    reasoning_tokens = response.metadata.usage.completion_tokens_details.reasoning_tokens
                    if reasoning_tokens > 0:
                        logger.debug(f"Reasoning tokens used: {reasoning_tokens}")
                
        except Exception as e:
            logger.error(f"Provider prediction failed for task {task_id}, test {test_id}, pair_index {pair_index}: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            raise

        return response

    def get_task_prediction(self, training_pairs: List[ARCPair], test_input: ARCPair, task_id: str, test_id: str, pair_index: int, bypass_cache: bool = False, prompt_feedback: Optional[str] = None, messages: Optional[List[dict]] = None) -> Attempt:
        """
        Modified to return the full Attempt object instead of just the parsed answer
        Uses the refactored parsing logic from arc_agi_benchmarking.parsing
        """
        # Get the initial response as an Attempt object
        attempt: Attempt = self.predict_task_output(training_pairs, test_input, task_id, test_id, pair_index, bypass_cache=bypass_cache, prompt_feedback=prompt_feedback, messages=messages)

        try:
            # If the validator couldn't parse the answer, fall back to the provider extractor
            if isinstance(attempt.answer, str):
                parsed = self.provider.extract_json_from_response(attempt.answer)
                if parsed is None:
                    raise ValueError("Failed to parse answer")
                attempt.answer = parsed
            return attempt
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                f"Parsing/Validation failed for task {task_id}, test {test_id}, pair_index {pair_index}: {e}",
                exc_info=True,
            )
            raise

    @timeit
    def generate_task_solution(self, data_dir, task_id):
        """
        Generates and saves the solution for a specific ARC task.

        Args:
            data_dir: The directory containing the ARC task data (e.g., 'data/arc-agi/data/evaluation').
            task_id: The ID of the specific task to solve (e.g., '0a1d4ef5').

        Instance attributes used:
            self.config: The model configuration name being used.
            self.num_attempts: The number of prediction attempts per test pair.
            self.retry_attempts: The number of internal retries if a prediction attempt fails.
            self.save_submission_dir: Directory to save the final submission JSON.
            self.overwrite_submission: Whether to overwrite existing submission files.
            self.print_submission: Whether to log the final submission JSON content.
            self.provider: The initialized provider adapter.

        Returns:
            A list representing the submission structure if successful and saving is enabled,
            or None if no valid predictions were made or saving is disabled but run completes.
            Returns None immediately if submission exists and overwrite is False.
        """
        
        utils.validate_data(data_dir, task_id)

        # Use the config name as the test_id
        test_id = self.config

        # Logic for overwrite. If save_submission_dir is provided, check if the submission already exists
        if self.save_submission_dir:
            submission_file = os.path.join(self.save_submission_dir, f"{task_id}.json")
            bypass_cache_for_task = False
            if os.path.exists(submission_file):
                with open(submission_file, "r") as f:
                    existing_submission = json.load(f)

                # Check if the existing submission is correct
                if utils.is_submission_correct(existing_submission, data_dir, task_id):
                    logger.info("cached-correct |")
                    if not self.overwrite_submission:
                        logger.info(f" complete |")
                        return
                    else:
                        logger.info(f" overwriting |")
                else:
                    logger.info("cached-incorrect | bypass |")
                    bypass_cache_for_task = False
        else:
            bypass_cache_for_task = False
        
        task_attempts = []

        train_pairs = utils.get_train_pairs_from_task(data_dir, task_id)
        test_input_pairs = utils.get_test_input_from_task(data_dir, task_id)

        # Preload system prompt once
        try:
            system_prompt_text = _load_prompt("system_prompt")
        except Exception:
            system_prompt_text = "You are an ARC solver. Return only the final answer as a JSON array of arrays of integers."

        # Go through each test pair to get a prediction. 96% of challenges have 1 pair.
        for t, pair_input_obj in enumerate(test_input_pairs):
            pair_index = t
            logger.debug(f"Starting task {task_id}, ModelConfig: {test_id}, Test Pair Index: {pair_index+1}/{len(test_input_pairs)}")
            
            pair_submission_attempts = {}
            previous_prompt_feedback: Optional[str] = None
            # Maintain a running message index so subsequent attempts don't reuse indices
            next_choice_index: int = 0
            # Initialize per-pair conversation messages
            messages: List[dict] = [
                {"role": "system", "content": system_prompt_text}
            ]

            # Run through each prediction attempt
            stop_attempts = False
            for attempt_num in range(1, self.num_attempts + 1):
                attempt_key = f"attempt_{attempt_num}"
                pair_submission_attempts[attempt_key] = None

                for retry_num in range(self.retry_attempts):
                    try:
                        logger.debug(f"    Task {task_id}, ModelConfig {test_id}, Pair {pair_index+1}, Predicting attempt #{attempt_num}, retry #{retry_num + 1}")
                        # Build the user message for attempt 1 only (training + test input)
                        if attempt_num == 1 and retry_num == 0 and not any(m.get("role") == "user" for m in messages):
                            first_user_content = convert_task_pairs_to_prompt(train_pairs, pair_input_obj, prompt_name=self.prompt_name, prompt_feedback=None)
                            messages.append({"role": "user", "content": first_user_content})

                        # For attempt > 1, append assistant prior reply and user feedback
                        if attempt_num > 1 and retry_num == 0:
                            # Append assistant last answer from previous attempt if available
                            if isinstance(pair_submission_attempts.get(f"attempt_{attempt_num-1}"), dict):
                                try:
                                    prev_choices = pair_submission_attempts[f"attempt_{attempt_num-1}"]["metadata"]["choices"]
                                    if prev_choices:
                                        last_msg = prev_choices[-1]["message"]["content"]
                                        if last_msg:
                                            messages.append({"role": "assistant", "content": last_msg})
                                except Exception:
                                    pass
                            # Append feedback (if any)
                            if previous_prompt_feedback:
                                messages.append({"role": "user", "content": f"Feedback from previous attempt:\n{previous_prompt_feedback}"})

                        # Now storing the full attempt object with task_id and test_id
                        attempt_obj = self.get_task_prediction(
                            training_pairs=train_pairs,
                            test_input=pair_input_obj,
                            task_id=task_id,
                            test_id=test_id,
                            pair_index=pair_index,
                            bypass_cache=bypass_cache_for_task,
                            prompt_feedback=None,  # feedback now lives as a separate user message
                            messages=messages,
                        )

                        if attempt_obj is not None:
                            logger.debug(f"    Task {task_id}, ModelConfig {test_id}, Pair {pair_index+1}, Attempt #{attempt_num} successful. Prediction: {attempt_obj.answer}")
                            # Renumber choices so we don't repeat indices across attempts
                            try:
                                if getattr(attempt_obj, 'metadata', None) and getattr(attempt_obj.metadata, 'choices', None):
                                    for ch in attempt_obj.metadata.choices:
                                        # Assign new index and advance the counter
                                        ch.index = next_choice_index
                                        next_choice_index += 1
                            except Exception:
                                # Be resilient; if anything goes wrong, proceed without renumbering
                                pass

                            pair_submission_attempts[attempt_key] = attempt_obj.model_dump(mode='json')
                            # Capture feedback for the next attempt (if any)
                            if getattr(attempt_obj, 'prompt_feedback', None):
                                previous_prompt_feedback = attempt_obj.prompt_feedback

                            # If all training examples passed, write the test output and stop further attempts for this pair
                            try:
                                training_all_passed = bool(getattr(attempt_obj.metadata, 'kwargs', {}).get('training_all_passed', False))
                            except Exception:
                                training_all_passed = False

                            if training_all_passed:
                                try:
                                    if self.save_submission_dir:
                                        # Ensure directory exists
                                        os.makedirs(self.save_submission_dir, exist_ok=True)
                                        # Write per-pair test output with a non-.json extension so scoring won't pick it up
                                        test_output_path = os.path.join(self.save_submission_dir, f"{task_id}.pair{pair_index}.test_output.jsonl")
                                        with open(test_output_path, "w") as outf:
                                            json.dump(attempt_obj.answer, outf, indent=4)
                                        logger.debug(f"Wrote test output for task {task_id}, pair {pair_index} to {test_output_path}")
                                except Exception as e:
                                    logger.warning(f"    Failed to write test output for task {task_id}, pair {pair_index}: {e}")

                                stop_attempts = True
                            break 
                    except Exception as e:
                        error_msg = f"    Task {task_id}, ModelConfig {test_id}, Pair {pair_index+1}, Attempt #{attempt_num}, Retry #{retry_num + 1} failed. Error: {e}"
                        logger.warning(error_msg)
                        
                        # In verbose mode, show full traceback for debugging
                        if logger.isEnabledFor(logging.DEBUG):
                            import traceback
                            logger.debug(f"Full traceback for the above error:\n{traceback.format_exc()}")

                    if retry_num == self.retry_attempts - 1:
                        logger.warning(f"    Task {task_id}, ModelConfig {test_id}, Pair {pair_index+1}, All {self.retry_attempts} retries failed for attempt #{attempt_num}")

                # After finishing retries for this attempt, if we've flagged to stop, break out of attempts loop
                if stop_attempts:
                    break

            # Only append non-None attempts for this pair
            if any(v is not None for v in pair_submission_attempts.values()):
                task_attempts.append(pair_submission_attempts)

        if task_attempts:
            if self.print_submission:
                # Log the submission content; use json.dumps for potentially large structures
                logger.debug(f"Final submission for task {task_id}, ModelConfig {test_id}:\n{json.dumps(task_attempts, indent=4)}")

            if self.save_submission_dir:
                utils.save_submission(self.save_submission_dir, task_id, task_attempts)
                logger.debug(f"Submission for task {task_id}, ModelConfig {test_id} saved to {self.save_submission_dir}")
        else:
            logger.warning(f"No valid predictions for task {task_id}, ModelConfig {test_id} after all attempts. Skipping submission.")

        return task_attempts if task_attempts else None

def main_cli(cli_args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Run ARC Tester")
    parser.add_argument("--data_dir", type=str, help="Data set to run. Configure in config/config.json")
    parser.add_argument("--task_id", type=str, help="Specific task ID to run")
    parser.add_argument("--config", type=str, required=True, help="Configuration name (e.g., 'o1_high', 'gemini_short_response')")
    parser.add_argument(
        "--save_submission_dir",
        type=str,
        metavar="FOLDER_NAME",
        help="Folder name to save the submissions under Ex: 'submissions/o1_high'"
    )
    parser.add_argument("--overwrite_submission", action="store_true", help="Overwrite the submission if it already exists")
    parser.add_argument("--print_submission", action="store_true", help="Print the submission to the console after each task")
    parser.add_argument("--task_set", type=str, default="public_eval", choices=["public_eval", "public_training"], help="Task set to run")
    parser.add_argument("--num_attempts", type=int, default=2, help="Number of attempts for each prediction")
    parser.add_argument("--retry_attempts", type=int, default=2, help="Number of retry attempts for failed predictions")
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        default=False,
        help="Enable metrics collection and dumping (disabled by default)."
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        default="agent_coding_prompt",
        help="Prompt template name in prompts/ without extension (e.g., 'simple_coding_prompt' or 'agent_coding_prompt')"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output (shows debug info for arc_agi_benchmarking only, keeps libraries quiet)"
    )
    args = parser.parse_args(cli_args)

    # Set metrics enabled status based on CLI arg first
    set_metrics_enabled(args.enable_metrics)

    # Configure logging
    if args.verbose:
        # Verbose mode: Show DEBUG for our code, WARNING+ for libraries
        handler = logging.StreamHandler()
        handler.terminator = ""
        handler.setFormatter(logging.Formatter('%(message)s'))
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[handler],
            force=True
        )
        
        # Set library loggers to WARNING to reduce noise
        library_loggers = [
            'openai', 'httpx', 'httpcore', 'urllib3', 'requests', 
            'anthropic', 'google', 'pydantic', 'transformers'
        ]
        for lib_logger in library_loggers:
            logging.getLogger(lib_logger).setLevel(logging.WARNING)
        
        # Keep our application loggers at DEBUG
        logging.getLogger('arc_agi_benchmarking').setLevel(logging.DEBUG)
        logging.getLogger('__main__').setLevel(logging.DEBUG)
        
        logger.debug("Verbose mode enabled - showing debug output for arc_agi_benchmarking only")
    else:
        # Normal mode: Use the specified log level, no timestamps or newlines
        handler = logging.StreamHandler()
        handler.terminator = ""
        handler.setFormatter(logging.Formatter('%(message)s'))
        logging.basicConfig(
            level=getattr(logging, args.log_level.upper()),
            handlers=[handler],
            force=True
        )

        # Silence noisy third-party loggers
        library_loggers = [
            'openai', 'httpx', 'httpcore', 'urllib3', 'requests',
            'anthropic', 'google', 'pydantic', 'transformers'
        ]
        for lib_logger in library_loggers:
            logging.getLogger(lib_logger).setLevel(logging.WARNING)

    arc_solver = ARCTester(
        config=args.config,
        save_submission_dir=args.save_submission_dir, 
        overwrite_submission=args.overwrite_submission,
        print_submission=args.print_submission,
        num_attempts=args.num_attempts,
        retry_attempts=args.retry_attempts,
        prompt_name=args.prompt_name
    )
   
    arc_solver.generate_task_solution(
        data_dir=args.data_dir,
        task_id=args.task_id
    )
    # Optionally return the solver or a status for more detailed testing if needed
    # For this test, we'll just ensure it runs without error.

if __name__ == "__main__":
    main_cli()
