import json
from typing import List, Optional, Callable, Tuple
import re

# Type hint for the provider's JSON extraction function (can be refined later)
ProviderJsonExtractor = Callable[[str], Optional[List[List[int]]]]

# --- Helper Parsing Functions: Can be added to as needed ---

def backscan_json_parser(log_str: str) -> Optional[List[List[int]]]:
    """
    Extract the last valid JSON substring that matches the List[List] structure
    from the given log string by scanning backwards from the end.

    Parameters:
        log_str (str): The full log output text.

    Returns:
        The parsed List[List] object if found and valid, otherwise None.
    """
    last_bracket_idx = -1
    closing_bracket = None
    for i in range(len(log_str) - 1, -1, -1):
        char = log_str[i]
        if char in (']', '}'):
            last_bracket_idx = i
            closing_bracket = char
            break

    if last_bracket_idx == -1:
        return None

    opening_bracket = '[' if closing_bracket == ']' else '{'

    bracket_counter = 1 # Start at 1 to account for the found closing bracket
    start_idx = -1

    for i in range(last_bracket_idx - 1, -1, -1):
        char = log_str[i]
        if char == closing_bracket:
            bracket_counter += 1
        elif char == opening_bracket:
            bracket_counter -= 1
            if bracket_counter == 0:
                start_idx = i
                break

    if start_idx == -1:
        return None

    json_candidate = log_str[start_idx:last_bracket_idx+1]

    try:
        parsed_json = json.loads(json_candidate)

        # Validate the structure: must be a non-empty list of lists.
        if isinstance(parsed_json, list) and parsed_json and all(isinstance(row, list) for row in parsed_json):
            return parsed_json
        else:
            return None

    except json.JSONDecodeError:
        return None

def extract_from_boxed(log_str: str) -> Optional[List[List[int]]]:
    """
    Extracts JSON from a LaTeX-style \boxed{} command in a string.
    """
    match = re.search(r"\\boxed\{(.*?)\}\s*$", log_str, re.DOTALL | re.MULTILINE)
    if match:
        content = match.group(1).strip()
        try:
            # The content inside boxed is often a list of lists
            parsed_json = json.loads(content)
            if isinstance(parsed_json, list) and all(isinstance(i, list) for i in parsed_json):
                return parsed_json
        except json.JSONDecodeError:
            # If json.loads fails, it's not the JSON we're looking for.
            pass
    return None

def extract_from_json_response(log_str: str) -> Optional[List[List[int]]]:
    """
    Extracts the output from a JSON response that contains explanation, code, and output fields.
    
    Example format:
    {
        "explanation": "...",
        "code": "...",
        "output": [[...], [...], ...]
    }
    """
    # Clean up the string first - remove any potential trailing commas or invalid JSON
    clean_str = log_str.strip()
    
    # Try to parse the entire string as JSON first
    try:
        data = json.loads(clean_str)
        if isinstance(data, dict) and "output" in data:
            return backscan_json_parser(data["output"])
    except json.JSONDecodeError:
        # If that fails, try to find a JSON object in the string
        try:
            # Look for the output field specifically
            output_match = re.search(r'"output"\s*:\s*(\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\])', clean_str, re.DOTALL)
            if output_match:
                output_str = output_match.group(1)
                output = json.loads(output_str)
                if isinstance(output, list) and all(isinstance(row, list) for row in output):
                    return output
            
            # If that doesn't work, try to find any valid JSON array of arrays
            array_match = re.search(r'(\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\])', clean_str, re.DOTALL)
            if array_match:
                array_str = array_match.group(1)
                output = json.loads(array_str)
                if isinstance(output, list) and all(isinstance(row, list) for row in output):
                    return output
                    
        except (json.JSONDecodeError, AttributeError):
            pass
    
    return None


# --- Main Parsing Orchestrator ---

def parse_and_validate_json(response: str) -> Tuple[List[List[int]], Optional[str]]:
    """
    Orchestrates parsing by trying the provider extractor.
    Returns the parsed List[List[int]] or raises ValueError if validation fails.
    """
    # Clean up the response string first
    cleaned_response = response.strip()
    
    # List of parser functions to try in order
    parsers = [extract_from_json_response, backscan_json_parser, extract_from_boxed]
    
    # Try parsing as JSON first
    # Try to parse the entire string as JSON
    data = json.loads(cleaned_response)
    output = data["output"]
    if isinstance(output, str):
        output = json.loads(output)
    code = data.get("code")

    if isinstance(output, list) and all(isinstance(row, list) for row in output):
        return output, code
    elif isinstance(output, list) and all(isinstance(x, (int, float)) for x in output):
        return [output], code  # Convert single list to list of lists
   
    # If we get here, all parsing attempts failed
    raise ValueError(f"Failed to parse response after all attempts: {response[:200]}...")

if __name__ == "__main__":
    response0 = '{\n    "explanation": "Find the single cell with value 8 and the single cell with value 7. Fill the entire row and entire column of the 8 with 8s, and fill the entire row and entire column of the 7 with 7s. When the row of one number intersects the column of the other, place a 2 at that intersection (overriding 7 or 8). All other cells remain 0.",\n    "code": "def transform(grid):\\n    n = len(grid)\\n    # locate 8 and 7\\n    pos8 = pos7 = None\\n    for i in range(n):\\n        for j in range(n):\\n            if grid[i][j] == 8:\\n                pos8 = (i,j)\\n            if grid[i][j] == 7:\\n                pos7 = (i,j)\\n    out = [[0]*n for _ in range(n)]\\n    if pos8:\\n        r8,c8 = pos8\\n        # fill row and column with 8\\n        for j in range(n): out[r8][j] = 8\\n        for i in range(n): out[i][c8] = 8\\n    if pos7:\\n        r7,c7 = pos7\\n        # fill row and column with 7\\n        for j in range(n): out[r7][j] = 7\\n        for i in range(n): out[i][c7] = 7\\n    # intersections: row of 8 with column of 7, and row of 7 with column of 8 set to 2\\n    if pos8 and pos7:\\n        r8,c8 = pos8\\n        r7,c7 = pos7\\n        out[r8][c7] = 2\\n        out[r7][c8] = 2\\n    return out\\n\\n# Run on training examples to verify\\ntrain0 = [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,8,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,7,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]\\ntrain1 = [[0,0,0,0,0,0,0,0,0],[0,0,0,8,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,7,0,0],[0,0,0,0,0,0,0,0,0]]\\nprint(transform(train0))\\nprint(transform(train1))\\n\\n# Run on test input\\ntest = [[0,0,0,0,0,0,0,0,0],[0,0,0,0,8,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,7,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]]\\nresult = transform(test)\\nprint(result)\\n",\n    "output": [[0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 2, 8, 8, 2, 8, 7, 8, 8], [0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 0, 0, 0], [7, 7, 7, 7, 2, 7, 7, 7, 7], [0, 0, 0, 0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 8, 0, 0, 0, 0]]\n}'
    print(parse_and_validate_json(response0))

    response1 = """### Summary and Final Answer:\nThe pattern transforms the input, which consists of two lists, into six lists in the output. Each of the first four output lists is constructed by repeating the original or reversed input lists three times. The fifth and sixth output lists are repetitions of the first and second output lists, respectively.\n\nFor the test input `[[3, 2], [7, 8]]`:\n1. The first input list `[3, 2]` is repeated three times to form the first output list.\n2. The second input list `[7, 8]` is repeated three times to form the second output list.\n3. The reversed first input list `[2, 3]` is repeated three times to form the third output list.\n4. The reversed second input list `[8, 7]` is repeated three times to form the fourth output list.\n5. The first output list is repeated as the fifth output list.\n6. The second output list is repeated as the sixth output list.\n\nThus, the output is:\n\n```python\n[[3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8], [2, 3, 2, 3, 2, 3], [8, 7, 8, 7, 8, 7], [3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8]]\n```\n\nFinal answer (formatted to match the training output style):\n\n```python\n[[3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8], [2, 3, 2, 3, 2, 3], [8, 7, 8, 7, 8, 7], [3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8]]\n```\n\n\\boxed{[[3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8], [2, 3, 2, 3, 2, 3], [8, 7, 8, 7, 8, 7], [3, 2, 3, 2, 3, 2], [7, 8, 7, 8, 7, 8]]}'
"""
    print(parse_and_validate_json(response1))

    response2 = '{\n    "explanation": "Each input row contains exactly one 5 and two 0s. The output row\'s elements are all equal to a number determined by the position (index) of the 5 in that input row: if the 5 is at index 0 -> output value 2; index 1 -> 4; index 2 -> 3. So we transform each row into a uniform row [v,v,v] where v depends on where the 5 sits in that row.",\n    "code": "def transform(grid):\\n    mapping = {0:2, 1:4, 2:3}\\n    out = []\\n    for row in grid:\\n        # find index of the 5\\n        idx = row.index(5)\\n        val = mapping[idx]\\n        out.append([val, val, val])\\n    return out\\n\\n# Test on training examples\\ntrain = [\\n    ([[0,0,5],[0,5,0],[5,0,0]], [[3,3,3],[4,4,4],[2,2,2]]),\\n    ([[0,0,5],[0,0,5],[0,0,5]], [[3,3,3],[3,3,3],[3,3,3]]),\\n    ([[5,0,0],[0,5,0],[5,0,0]], [[2,2,2],[4,4,4],[2,2,2]]),\\n    ([[0,5,0],[0,0,5],[0,5,0]], [[4,4,4],[3,3,3],[4,4,4]])\\n]\\nfor inp, expected in train:\\n    out = transform(inp)\\n    assert out == expected, (inp, out, expected)\\n\\n# Run on test input\\ntest_input = [[0,0,5],[5,0,0],[0,5,0]]\\nresult = transform(test_input)\\nprint(result)\\n",\n    "output": "[[3, 3, 3], [2, 2, 2], [4, 4, 4]]"\n}'
    print(parse_and_validate_json(response2))

    response3 = '{\n    "explanation": "In the training pairs, every time two 1s appear in the same row with exactly one cell between them (pattern 1,0,1 horizontally), the middle 0 is changed to 2. No vertical or diagonal changes occur. The code scans each row and whenever grid[r][c]==1 and grid[r][c+2]==1 and the middle cell is 0, it sets that middle cell to 2. The same transformation is applied to the test input.",\n    "code": "def transform(grid):\\n    H = len(grid)\\n    W = len(grid[0])\\n    out = [row[:] for row in grid]\\n    for r in range(H):\\n        for c in range(W-2):\\n            if grid[r][c] == 1 and grid[r][c+2] == 1 and grid[r][c+1] == 0:\\n                out[r][c+1] = 2\\n    return out\\n\\n# Run on training examples to verify (omitted here) and on test input:\\ninput_grid = [[0, 1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]\\noutput_grid = transform(input_grid)\\nprint(output_grid)",\n    "output": [[0, 1, 2, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 1, 2, 1, 2, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 2, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 2, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]\n}'
    print(parse_and_validate_json(response3))
