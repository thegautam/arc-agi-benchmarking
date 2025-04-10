import pytest
import json
# Import the functions directly from the new module
from src.parsing import (
    parse_and_validate_json, 
    extract_json_from_response,
    convert_single_integer_to_2d_list,
    convert_1d_list_to_2d_list
)
import src.utils # Keep for potential mocking later
from unittest.mock import MagicMock, patch

# --- Tests for convert_single_integer_to_2d_list ---
def test_convert_single_integer():
    assert convert_single_integer_to_2d_list("5") == [[5]]

def test_convert_single_integer_invalid():
    assert convert_single_integer_to_2d_list("abc") is None
    assert convert_single_integer_to_2d_list("[5]") is None

# --- Tests for convert_1d_list_to_2d_list ---
def test_convert_1d_list():
    assert convert_1d_list_to_2d_list("[1, 2, 3]") == [[1], [2], [3]]

def test_convert_1d_list_single_item():
    assert convert_1d_list_to_2d_list("[7]") == [[7]]

def test_convert_1d_list_with_space():
     assert convert_1d_list_to_2d_list(" [1, 2] ") == [[1], [2]]

def test_convert_1d_list_invalid_json():
    assert convert_1d_list_to_2d_list("[1, 2") is None

def test_convert_1d_list_not_list():
    assert convert_1d_list_to_2d_list("{\"a\": 1}") is None

def test_convert_1d_list_wrong_items():
    assert convert_1d_list_to_2d_list("[[1], [2]]") is None # Nested list
    assert convert_1d_list_to_2d_list("['a', 'b']") is None # List of strings

# --- Tests for parse_and_validate_json ---

# Helper to create a mock provider extractor
@pytest.fixture
def mock_provider_extractor():
    return MagicMock()

def test_parse_valid_standard_json(mock_provider_extractor):
    response = "[[1, 2], [3, 4]]"
    expected = [[1, 2], [3, 4]]
    assert parse_and_validate_json(response, mock_provider_extractor) == expected
    mock_provider_extractor.assert_not_called()

def test_parse_calls_single_integer(mock_provider_extractor):
    response = "5"
    expected = [[5]]
    assert parse_and_validate_json(response, mock_provider_extractor) == expected
    mock_provider_extractor.assert_not_called()

def test_parse_calls_1d_list(mock_provider_extractor):
    response = "[1, 2, 3]"
    expected = [[1], [2], [3]]
    assert parse_and_validate_json(response, mock_provider_extractor) == expected
    mock_provider_extractor.assert_not_called()

# Test case where direct parsing/conversion fails, relying on extract_json_from_response
# We now let the actual extract_json_from_response (and its utils calls) run.
@patch('src.parsing.utils.extract_json_from_code_block') # Still mock utils to control outcome
@patch('src.parsing.utils.extract_json_grid_from_end')
@patch('src.parsing.utils.regex_extract_json')
def test_parse_uses_extraction_success(mock_regex, mock_grid, mock_code_block, mock_provider_extractor):
    response = "Some text [[1]] more text"
    expected_result = [[1]]
    # Simulate that regex_extract_json succeeds
    mock_code_block.return_value = None
    mock_grid.return_value = None
    mock_regex.return_value = expected_result
    # Provider extractor should not be needed/called
    mock_provider_extractor.return_value = None 

    result = parse_and_validate_json(response, mock_provider_extractor)

    assert result == expected_result
    mock_regex.assert_called_once_with(response)
    mock_provider_extractor.assert_not_called() # Ensure fallback wasn't reached

def test_parse_invalid_structure_direct_load(mock_provider_extractor):
    """Test invalid structure directly from json.loads"""
    response = "[1, [2]]" # Invalid: Not List[List]
    mock_provider_extractor.assert_not_called() # Should fail before needing extractor
    with pytest.raises(ValueError, match="Invalid JSON structure"):
        parse_and_validate_json(response, mock_provider_extractor)

def test_parse_invalid_content_direct_load(mock_provider_extractor):
    """Test invalid content directly from json.loads"""
    # Use valid JSON with double quotes for strings
    response = '[["a"], ["b"]]' # Invalid: Not List[List[int]]
    with pytest.raises(ValueError, match="Invalid JSON content"):
        parse_and_validate_json(response, mock_provider_extractor)

def test_parse_all_deterministic_fail(mock_provider_extractor):
    """Test when direct load, conversions, and all utils extractors fail."""
    response = "completely unparseable string"
    # Make the final fallback fail
    mock_provider_extractor.return_value = None
    
    # Patch utils to return None, simulating their failure on this input
    with patch('src.parsing.utils.extract_json_from_code_block', return_value=None), \
         patch('src.parsing.utils.extract_json_grid_from_end', return_value=None), \
         patch('src.parsing.utils.regex_extract_json', return_value=None):
             
        # Expect JSONDecodeError because all extraction methods failed
        with pytest.raises(json.JSONDecodeError):
            parse_and_validate_json(response, mock_provider_extractor)
            
    # Verify the provider extractor (the last step) was called
    mock_provider_extractor.assert_called_once_with(response)


@pytest.mark.xfail(reason="Deterministic parsers cannot yet handle this malformed grid format.")
def test_parse_malformed_grid_expect_success(mock_provider_extractor):
    """Test the malformed grid, expecting success from deterministic parsing eventually."""
    response = (
        "[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8]\n"
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0]\n"
        "[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8]\n"
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 6, 8, 0, 0]\n"
        "[8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]\n"
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n"
        "[8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8]\n"
        "[0, 0, 0, 0, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0]\n"
        "[8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]\n"
        "[0, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n"
        "[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8]\n"
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n"
        "[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8]\n"
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0]\n"
        "[8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8]\n"
        "[0, 0, 0, 0, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0]\n"
        "[8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8]\n"
        "[0, 0, 8, 6, 8, 0, 0, 0, 0, 0, 0, 8, 6, 8, 0, 0, 0]\n"
        "[8, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8]"
    )
    # This is the structure we WANT the parser to extract eventually
    expected_output = [
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 6, 8, 0, 0],
        [8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [0, 0, 0, 0, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [0, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        [0, 0, 0, 0, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8],
        [0, 0, 8, 6, 8, 0, 0, 0, 0, 0, 0, 8, 6, 8, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    ]

    # Make the final fallback fail, forcing reliance on deterministic steps
    mock_provider_extractor.return_value = None 
    
    # Call the function - this is expected to fail currently but should eventually pass
    # We don't patch the utils here, we want to see if the *real* ones can handle it
    result = parse_and_validate_json(response, mock_provider_extractor)
    
    assert result == expected_output
    mock_provider_extractor.assert_not_called() # Ideally, utils parse it, provider isn't called

# --- Tests specifically for extract_json_from_response ---

# These priority tests remain the same as they test internal logic of extract_json_from_response
@patch('src.parsing.utils.extract_json_from_code_block')
@patch('src.parsing.utils.extract_json_grid_from_end')
@patch('src.parsing.utils.regex_extract_json')
def test_extract_priority_code_block(
    mock_regex_extract, mock_grid_extract, mock_code_block_extract, mock_provider_extractor
):
    response = "```json [[1]] ```"
    expected = [[1]]
    mock_code_block_extract.return_value = expected
    mock_grid_extract.return_value = [[99]] # Should not be called
    mock_regex_extract.return_value = [[88]] # Should not be called
    mock_provider_extractor.return_value = [[77]] # Should not be called

    result = extract_json_from_response(response, mock_provider_extractor)
    assert result == expected
    mock_code_block_extract.assert_called_once_with(response)
    mock_grid_extract.assert_not_called()
    mock_regex_extract.assert_not_called()
    mock_provider_extractor.assert_not_called()

@patch('src.parsing.utils.extract_json_from_code_block')
@patch('src.parsing.utils.extract_json_grid_from_end')
@patch('src.parsing.utils.regex_extract_json')
def test_extract_priority_grid_end(
    mock_regex_extract, mock_grid_extract, mock_code_block_extract, mock_provider_extractor
):
    response = "some text [[1]]"
    expected = [[1]]
    mock_code_block_extract.return_value = None # Fails
    mock_grid_extract.return_value = expected # Succeeds
    mock_regex_extract.return_value = [[88]] # Should not be called
    mock_provider_extractor.return_value = [[77]] # Should not be called

    result = extract_json_from_response(response, mock_provider_extractor)
    assert result == expected
    mock_code_block_extract.assert_called_once_with(response)
    mock_grid_extract.assert_called_once_with(response)
    mock_regex_extract.assert_not_called()
    mock_provider_extractor.assert_not_called()

@patch('src.parsing.utils.extract_json_from_code_block')
@patch('src.parsing.utils.extract_json_grid_from_end')
@patch('src.parsing.utils.regex_extract_json')
def test_extract_priority_regex(
    mock_regex_extract, mock_grid_extract, mock_code_block_extract, mock_provider_extractor
):
    response = "some text [[1]] somewhere else"
    expected = [[1]]
    mock_code_block_extract.return_value = None # Fails
    mock_grid_extract.return_value = None # Fails
    mock_regex_extract.return_value = expected # Succeeds
    mock_provider_extractor.return_value = [[77]] # Should not be called

    result = extract_json_from_response(response, mock_provider_extractor)
    assert result == expected
    mock_code_block_extract.assert_called_once_with(response)
    mock_grid_extract.assert_called_once_with(response)
    mock_regex_extract.assert_called_once_with(response)
    mock_provider_extractor.assert_not_called()

@patch('src.parsing.utils.extract_json_from_code_block')
@patch('src.parsing.utils.extract_json_grid_from_end')
@patch('src.parsing.utils.regex_extract_json')
def test_extract_priority_provider(
    mock_regex_extract, mock_grid_extract, mock_code_block_extract, mock_provider_extractor
):
    response = "only provider can parse this"
    expected = [[1]]
    mock_code_block_extract.return_value = None # Fails
    mock_grid_extract.return_value = None # Fails
    mock_regex_extract.return_value = None # Fails
    mock_provider_extractor.return_value = expected # Succeeds

    result = extract_json_from_response(response, mock_provider_extractor)
    assert result == expected
    mock_code_block_extract.assert_called_once_with(response)
    mock_grid_extract.assert_called_once_with(response)
    mock_regex_extract.assert_called_once_with(response)
    mock_provider_extractor.assert_called_once_with(response)

@patch('src.parsing.utils.extract_json_from_code_block')
@patch('src.parsing.utils.extract_json_grid_from_end')
@patch('src.parsing.utils.regex_extract_json')
def test_extract_all_fail(
    mock_regex_extract, mock_grid_extract, mock_code_block_extract, mock_provider_extractor
):
    response = "nothing works"
    mock_code_block_extract.return_value = None # Fails
    mock_grid_extract.return_value = None # Fails
    mock_regex_extract.return_value = None # Fails
    mock_provider_extractor.return_value = None # Fails

    with pytest.raises(json.JSONDecodeError):
        extract_json_from_response(response, mock_provider_extractor)
        
    mock_code_block_extract.assert_called_once_with(response)
    mock_grid_extract.assert_called_once_with(response)
    mock_regex_extract.assert_called_once_with(response)
    mock_provider_extractor.assert_called_once_with(response) 