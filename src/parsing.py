import json
from typing import List, Optional, Callable

# Type hint for the provider's JSON extraction function (can be refined later)
ProviderJsonExtractor = Callable[[str], Optional[List[List[int]]]]

# --- Helper Parsing Functions (Stubs) ---

def _parse_json_code_block(response: str) -> Optional[List[List[int]]]:
    """Attempts to extract and parse JSON from a ```json code block."""
    # TODO: Implement logic to find and parse ```json blocks
    print(f"DEBUG: Trying _parse_json_code_block on {response!r}") # Debug
    return None

def _parse_direct_json(response: str) -> Optional[List[List[int]]]:
    """Attempts to parse the entire response string as JSON."""
    # TODO: Implement direct json.loads parsing and validation
    print(f"DEBUG: Trying _parse_direct_json on {response!r}") # Debug
    try:
        # Basic check - doesn't validate structure/content yet
        parsed = json.loads(response.strip())
        if isinstance(parsed, list): # Basic validation placeholder
             # Needs proper validation like before
             # return parsed # Temporarily returning raw parse for structure
             pass
    except json.JSONDecodeError:
        pass
    return None

def _parse_single_int(response: str) -> Optional[List[List[int]]]:
    """Attempts to parse the response as a single integer."""
    # TODO: Implement single integer parsing
    print(f"DEBUG: Trying _parse_single_int on {response!r}") # Debug
    try:
        num = int(response.strip())
        return [[num]]
    except ValueError:
        return None

def _parse_1d_list_as_2d(response: str) -> Optional[List[List[int]]]:
    """Attempts to parse a 1D list like [1, 2] into [[1], [2]]. """
    # TODO: Implement 1D list parsing and conversion
    print(f"DEBUG: Trying _parse_1d_list_as_2d on {response!r}") # Debug
    try:
        parsed = json.loads(response.strip())
        if isinstance(parsed, list) and all(isinstance(item, int) for item in parsed):
             return [[item] for item in parsed]
    except json.JSONDecodeError:
        pass
    return None


# --- Main Parsing Orchestrator ---

def parse_and_validate_json(response: str, provider_extractor: ProviderJsonExtractor) -> Optional[List[List[int]]]:
    """
    Orchestrates parsing by trying various helper functions.
    Falls back to the provider_extractor if all helpers fail.
    Returns the parsed List[List[int]] or None if all methods fail.
    """
    parsing_attempts = [
        _parse_single_int,
        _parse_1d_list_as_2d,
        _parse_json_code_block,
        _parse_direct_json,
        # Add other parsing functions here in desired order
    ]

    for parser in parsing_attempts:
        result = parser(response)
        if result is not None:
            # TODO: Add validation step here later
            print(f"DEBUG: Parser {parser.__name__} succeeded: {result!r}") # Debug
            return result # Return immediately on first success

    # If all deterministic parsers fail, use the provider's extractor
    print(f"DEBUG: All deterministic parsers failed, calling provider_extractor.") # Debug
    provider_result = provider_extractor(response)
    if provider_result is not None:
        # TODO: Add validation step here later
         print(f"DEBUG: provider_extractor succeeded: {provider_result!r}") # Debug
         return provider_result

    print(f"DEBUG: All parsing methods failed for response: {response!r}") # Debug
    # Consider raising an error here if no method works, or returning None
    # raise ValueError(f"Failed to parse response after all attempts: {response!r}")
    return None # Or raise error
