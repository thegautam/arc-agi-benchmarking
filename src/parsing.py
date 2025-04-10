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
    Orchestrates parsing by trying the provider extractor.
    Returns the parsed List[List[int]] or raises ValueError if validation fails.
    """
    parsing_attempts = [
        provider_extractor,
    ]

    for parser in parsing_attempts:
        result = parser(response)
        if result is not None:
            # Validate the structure: must be list of lists
            if isinstance(result, list) and all(isinstance(row, list) for row in result):
                print(f"DEBUG: Parser {parser.__name__} succeeded and validated: {result!r}") # Debug
                return result # Return immediately on first success and validation
            else:
                # Raise error if structure is invalid, triggering retry in main loop
                raise ValueError(f"Parser {parser.__name__} produced invalid structure: {result!r}")

    print(f"DEBUG: All parsing methods failed for response: {response!r}") # Debug
    # Raise an error here if no method works, triggering retry in main loop
    raise ValueError(f"Failed to parse response after all attempts: {response!r}")
