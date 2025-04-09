import json
from typing import List, Optional, Any, Callable
import src.utils as utils

# Type hint for the provider's JSON extraction function
ProviderJsonExtractor = Callable[[str], Optional[List[List[int]]]]

def convert_single_integer_to_2d_list(data: str) -> Optional[List[List[int]]]:
    """
    If the input string represents a single integer, return it as a nested list.
    Otherwise, return None.
    """
    try:
        parsed_data = int(data)
        result = [[parsed_data]]
        return result
    except ValueError:
        pass
    return None

def convert_1d_list_to_2d_list(data: str) -> Optional[List[List[int]]]:
    """
    If the input string represents a single-item list containing one or more integers,
    return it as a nested list. Otherwise, return None.
    """
    try:
        # Remove whitespace and parse the string as JSON
        parsed_data = json.loads(data.strip())
        if isinstance(parsed_data, list) and 1 <= len(parsed_data) <= 30 and all(isinstance(item, int) for item in parsed_data):
            result = [[item] for item in parsed_data]
            return result
    except json.JSONDecodeError:
        pass
    return None

def extract_json_from_response(response: str, provider_extractor: ProviderJsonExtractor) -> List[List[int]]:
    """
    Extract JSON from various possible formats in the response.
    Requires a function `provider_extractor` to handle provider-specific LLM extraction as a fallback.
    """
    # 1. Try to extract JSON from code block (most precise method)
    json_code_block_match = utils.extract_json_from_code_block(response)
    if json_code_block_match:
        return json_code_block_match
    
    # 2. Try to extract JSON grid from end of response (specialized for grid formats)
    json_grid_match = utils.extract_json_grid_from_end(response)
    if json_grid_match:
        return json_grid_match
    
    # 3. Try to extract JSON array using regex (more general approach)
    json_str_match = utils.regex_extract_json(response)
    if json_str_match:
        return json_str_match

    # 4. Finally, use the provided provider extractor function (last resort)
    json_llm_match = provider_extractor(response)
    if json_llm_match:
        return json_llm_match

    # If all extraction methods fail, raise an exception
    raise json.JSONDecodeError("Failed to extract valid JSON from the response", response, 0)

def parse_and_validate_json(response: str, provider_extractor: ProviderJsonExtractor) -> List[List[int]]:
    """
    Parse the response string into JSON (List[List[int]]) and validate its structure.

    Requires a function `provider_extractor` for the fallback mechanism in extract_json_from_response.
    """
    # Check for edge cases first
    single_integer_match = convert_single_integer_to_2d_list(response)
    if single_integer_match:
        # Consider adding logging here instead of print if needed
        # print(f"Extracted single integer: {single_integer_match}") 
        return single_integer_match

    one_d_match = convert_1d_list_to_2d_list(response)
    if one_d_match:
        # print(f"Extracted 1d list: {one_d_match}")
        return one_d_match

    # Try direct JSON parsing
    try:
        parsed_json = json.loads(response)
        # print(f"Extracted raw JSON: {parsed_json}")
        # Proceed to validation
    except json.JSONDecodeError:
        # If raw parsing fails, try the advanced extraction methods
        # print(f"Raw JSON parsing failed, trying extraction methods...")
        parsed_json = extract_json_from_response(response, provider_extractor)

    # --- Validation --- 
    # 1. Validate Structure: Must be a list of lists
    if not isinstance(parsed_json, list) or not all(isinstance(row, list) for row in parsed_json):
        raise ValueError("Invalid JSON structure: expected a list of lists")

    # 2. Validate Content: Must contain only integers within the lists
    # This check only runs if the structure validation passed.
    if not all(isinstance(item, int) for row in parsed_json for item in row):
         raise ValueError("Invalid JSON content: all items in sub-lists must be integers")

    return parsed_json 