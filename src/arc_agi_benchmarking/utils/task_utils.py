import os
from arc_agi_benchmarking.schemas import ARCPair, ModelConfig
from typing import List
import json
import re
import yaml

def get_train_pairs_from_task(data_dir, task_id) -> List[ARCPair]:
    """
    Loads up task train pairs from task json file
    """

    task_file = os.path.join(data_dir, f"{task_id}.json")
    with open(task_file, 'r') as f:
        task_data = json.load(f)

    pairs = []
    for pair in task_data['train']:
        pairs.append(ARCPair(input=pair['input'], output=pair['output']))

    return pairs

def get_test_input_from_task(data_dir, task_id) -> List[ARCPair]:
    task_file = os.path.join(data_dir, f"{task_id}.json")
    with open(task_file, 'r') as f:
        task_data = json.load(f)

    pairs = []
    for pair in task_data['test']:
        pairs.append(ARCPair(input=pair['input']))

    return pairs


def save_submission(save_submission_dir: str, task_id: str, task_attempts) -> None:
    """
    Save the submission to a file with full attempt metadata.
    
    The save_submission_dir should be a directory path that includes the config name,
    e.g., 'submissions/o1_short_response' or 'submissions/gemini_pro'.
    """
    os.makedirs(save_submission_dir, exist_ok=True)
    submission_file = os.path.join(save_submission_dir, f"{task_id}.json")
    
    with open(submission_file, "w") as f:
        json.dump(task_attempts, f, indent=4)

    return submission_file

def normalize_model_name(name: str) -> str:
    """
    Normalize model name for comparison by:
    1. Converting dots to dashes
    2. Removing any date suffixes
    3. Removing 'latest' suffix
    4. Removing duplicate dashes
    
    Examples:
        claude-3.5-sonnet -> claude-3-5-sonnet
        claude-3-5-sonnet-20240315 -> claude-3-5-sonnet
        claude-3-5-sonnet-latest -> claude-3-5-sonnet
    """
    # Remove any date suffix (assuming YYYYMMDD format)
    name = re.sub(r'-\d{8}$', '', name)
    
    # Remove 'latest' suffix
    name = re.sub(r'-latest$', '', name)
    
    # Convert dots to dashes
    name = name.replace('.', '-')
    
    # Clean up multiple dashes
    name = re.sub(r'-+', '-', name)
    
    return name

def read_models_config(config: str) -> ModelConfig:
    """
    Reads and parses both models.yml and models_private.yml configuration files 
    for a specific configuration.
    
    Args:
        config (str): The configuration name to look up (e.g., 'o1_high', 'gemini_short_response')
        
    Returns:
        ModelConfig: The configuration for the specified model
        
    Raises:
        ValueError: If no matching configuration is found
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_file = os.path.join(base_dir, "models.yml")
    models_private_file = os.path.join(base_dir, "models_private.yml")
    
    # Initialize with models from the main config file
    with open(models_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Add models from private config if it exists
    if os.path.exists(models_private_file):
        with open(models_private_file, 'r') as f:
            private_config_data = yaml.safe_load(f)
            # Merge the models lists
            if 'models' in private_config_data:
                config_data['models'].extend(private_config_data['models'])
    
    # Look for a model with the name matching the config parameter
    for model in config_data['models']:
        if model.get('name') == config:
            return ModelConfig(**model)
            
    raise ValueError(f"No matching configuration found for '{config}'")

def is_submission_correct(submission: list, data_dir: str, task_id: str) -> bool:
    """
    Check if a submission matches the expected output for a task.
    
    Args:
        submission: The submission to check (list of attempts)
        data_dir: Directory containing the task data
        task_id: ID of the task
        
    Returns:
        bool: True if the submission is correct, False otherwise
    """
    # Validate submission structure
    if not submission or not isinstance(submission, list):
        return False

    # Load task JSON directly to access expected outputs for test pairs
    task_file = os.path.join(data_dir, f"{task_id}.json")
    if not os.path.exists(task_file):
        return False

    with open(task_file, 'r') as f:
        task_data = json.load(f)

    test_data = task_data.get('test', [])
    if not isinstance(test_data, list) or len(test_data) == 0:
        return False

    # If expected outputs are not present (e.g., evaluation set), we cannot verify
    # Treat as not verifiable -> return False (do not assert correctness)
    if not all(isinstance(item, dict) and 'output' in item for item in test_data):
        return False

    # Compare each test pair's latest non-null attempt's answer with expected output
    # Require all available test outputs to match
    for idx, test_item in enumerate(test_data):
        expected_output = test_item.get('output')
        # If we don't have a corresponding submission entry, it's incorrect
        if idx >= len(submission):
            return False

        pair_submission = submission[idx]
        if not isinstance(pair_submission, dict):
            return False

        # Find the highest-numbered non-null attempt for this pair
        latest_answer = None
        max_attempt_num = -1
        for k, v in pair_submission.items():
            if isinstance(k, str) and k.startswith('attempt_') and v is not None:
                try:
                    num = int(k.split('_', 1)[1])
                except (IndexError, ValueError):
                    continue
                if num > max_attempt_num:
                    max_attempt_num = num
                    if isinstance(v, dict):
                        latest_answer = v.get('answer')
                    else:
                        # If the structure is not a dict, cannot verify
                        latest_answer = None

        # Must have a latest non-null answer
        if latest_answer is None:
            return False

        # Answers should be lists (grids); if not, consider incorrect
        if latest_answer != expected_output:
            return False

    # All test pairs matched expected outputs
    return True


def read_provider_rate_limits() -> dict:
    """
    Reads and parses the provider_config.yml file to get rate limit configurations.

    Assumes provider_config.yml is in the same base directory as models.yml.

    Returns:
        dict: A dictionary where keys are provider names and values are dicts
              containing 'rate' and 'period'.
              Example: {'openai': {'rate': 60, 'period': 60}}

    Raises:
        FileNotFoundError: If provider_config.yml is not found.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    # Go up three levels from src/utils/task_utils.py to get to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    provider_config_file = os.path.join(base_dir, "provider_config.yml")

    if not os.path.exists(provider_config_file):
        raise FileNotFoundError(f"provider_config.yml not found at {provider_config_file}")

    with open(provider_config_file, 'r') as f:
        try:
            rate_limits_data = yaml.safe_load(f)
            if not isinstance(rate_limits_data, dict):
                raise yaml.YAMLError("provider_config.yml root should be a dictionary of providers.")
            # Basic validation for each provider's config
            for provider, limits in rate_limits_data.items():
                if not isinstance(limits, dict) or 'rate' not in limits or 'period' not in limits:
                    raise yaml.YAMLError(
                        f"Provider '{provider}' in provider_config.yml must have 'rate' and 'period' keys."
                    )
                if not isinstance(limits['rate'], int) or not isinstance(limits['period'], int):
                    raise yaml.YAMLError(
                        f"'rate' and 'period' for provider '{provider}' must be integers."
                    )
            return rate_limits_data
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing provider_config.yml: {e}")