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

# Test case where direct json.loads fails, and fallback extraction is needed
@patch('src.parsing.extract_json_from_response') # Patch the function in the parsing module
def test_parse_uses_fallback_extraction(mock_extract_json, mock_provider_extractor):
    response = "Some text [[1]] more text"
    expected_fallback_result = [[1]]
    mock_extract_json.return_value = expected_fallback_result

    # This input will fail convert_single_int, convert_1d_list, and json.loads
    # It should then call extract_json_from_response
    result = parse_and_validate_json(response, mock_provider_extractor)

    assert result == expected_fallback_result
    mock_extract_json.assert_called_once_with(response, mock_provider_extractor)
    mock_provider_extractor.assert_not_called() # The provider extractor is called *inside* the mocked extract_json

def test_parse_invalid_structure_after_fallback(mock_provider_extractor):
    response = "invalid structure that needs fallback"
    # Mock the fallback extractor to return something invalid
    mock_provider_extractor.return_value = [1, [2]] # Not List[List[int]]
    
    # Patch the utils methods so extract_json_from_response calls the provider mock
    with patch('src.parsing.utils.extract_json_from_code_block', return_value=None), \
         patch('src.parsing.utils.extract_json_grid_from_end', return_value=None), \
         patch('src.parsing.utils.regex_extract_json', return_value=None):
        
        with pytest.raises(ValueError, match="Invalid JSON structure"):
            # parse_and_validate will call extract_json_from_response, which will call the mock_provider_extractor
            parse_and_validate_json(response, mock_provider_extractor)
            
    mock_provider_extractor.assert_called_once_with(response)

def test_parse_invalid_content_after_fallback(mock_provider_extractor):
    response = "invalid content that needs fallback"
    # Mock the fallback extractor to return list[list[str]]
    mock_provider_extractor.return_value = [['a'], ['b']]
    
    with patch('src.parsing.utils.extract_json_from_code_block', return_value=None), \
         patch('src.parsing.utils.extract_json_grid_from_end', return_value=None), \
         patch('src.parsing.utils.regex_extract_json', return_value=None):

        with pytest.raises(ValueError, match="Invalid JSON content"):
             parse_and_validate_json(response, mock_provider_extractor)
             
    mock_provider_extractor.assert_called_once_with(response)

def test_parse_all_fallbacks_fail(mock_provider_extractor):
    response = "completely unparseable"
    # Mock the provider extractor to also fail (return None or raise)
    mock_provider_extractor.return_value = None # Simulate provider failure
    
    # Patch the utils methods to fail
    with patch('src.parsing.utils.extract_json_from_code_block', return_value=None), \
         patch('src.parsing.utils.extract_json_grid_from_end', return_value=None), \
         patch('src.parsing.utils.regex_extract_json', return_value=None):
             
        with pytest.raises(json.JSONDecodeError):
            parse_and_validate_json(response, mock_provider_extractor)
            
    mock_provider_extractor.assert_called_once_with(response)

def test_parse_malformed_grid_example(mock_provider_extractor):
    # This example is not valid JSON as is.
    # It will fail direct parsing and potentially rely on utils or the provider extractor
    response = (
        "[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8]\n"
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0]\n"
        # ... (rest of the malformed string) ...
        "[8, 8, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8]"
    )
    # Mock the provider extractor to fail
    mock_provider_extractor.return_value = None 
    
    # Assume utils also fail for this input for now
    with patch('src.parsing.utils.extract_json_from_code_block', return_value=None), \
         patch('src.parsing.utils.extract_json_grid_from_end', return_value=None), \
         patch('src.parsing.utils.regex_extract_json', return_value=None):
             
        with pytest.raises(json.JSONDecodeError):
            parse_and_validate_json(response, mock_provider_extractor)
            
    mock_provider_extractor.assert_called_once_with(response)

# --- Tests specifically for extract_json_from_response ---

# We need more tests here to check the priority of utils methods
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