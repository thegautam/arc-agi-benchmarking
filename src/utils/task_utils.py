import os
from src.schemas import ARCPair, ModelConfig
from typing import List, Dict, Any
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

def convert_2d_list_to_string(list_of_lists: List[List[int]]) -> str:
    """
    Convert a list of lists to a string
    """

    string_list = ""

    for row in list_of_lists:
        string_list += json.dumps(row) + "\n"

    return string_list

def regex_extract_json(response: str) -> List[List[int]]:
    """
    Extract JSON from various possible formats in the response.
    Returns None if extraction or parsing fails.
    """
    try:
        json_str_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_str_match:
            return json.loads(json_str_match.group(0))
    except json.JSONDecodeError:
        return None
    except Exception:
        return None
    
    return None

def extract_json_from_code_block(response: str) -> List[List[int]]:
    """
    Extract JSON from a code block in the response.
    Returns None if extraction or parsing fails.
    """
    try:
        code_block_start = response.find("```json")
        code_block_end = response.find("```", code_block_start + 6)
        if code_block_start != -1 and code_block_end != -1:
            json_str = response[code_block_start + 6:code_block_end].strip()
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError, Exception):
        return None
    
    return None

def save_submission(save_submission_dir: str, task_id: str, task_attempts) -> None:
    """
    Save the submission to a file with full attempt metadata
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
    3. Removing duplicate dashes
    
    Examples:
        claude-3.5-sonnet -> claude-3-5-sonnet
        claude-3-5-sonnet-20240315 -> claude-3-5-sonnet
    """
    # Remove any date suffix (assuming YYYYMMDD format)
    name = re.sub(r'-\d{8}$', '', name)
    
    # Convert dots to dashes
    name = name.replace('.', '-')
    
    # Clean up multiple dashes
    name = re.sub(r'-+', '-', name)
    
    return name

def read_models_config(model_name: str) -> ModelConfig:
    """
    Reads and parses the models.yml configuration file for a specific model.
    Uses normalized name matching to find the base model configuration.
    
    Args:
        model_name (str): The name of the model to get configuration for. Can include version/date suffixes.
        
    Returns:
        ModelConfig: The configuration for the specified model
        
    Raises:
        ValueError: If no matching base model is found in the configuration
    """
    models_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.yml")
    
    with open(models_file, 'r') as f:
        config = yaml.safe_load(f)
    
    normalized_input = normalize_model_name(model_name)
        
    # Find the first model whose normalized name matches
    for model in config['models']:
        if normalize_model_name(model['name']) == normalized_input:
            # Use the provided model name but keep the base config
            model_config = model.copy()
            model_config['name'] = model_name
            return ModelConfig(**model_config)
            
    raise ValueError(f"No matching base model found in configuration for '{model_name}' (normalized: '{normalized_input}')")