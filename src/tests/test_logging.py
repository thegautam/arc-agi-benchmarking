import pytest
import logging
import subprocess
import sys
import os
import argparse
from unittest.mock import patch, MagicMock

# Add project root to sys.path to allow importing main and cli.run_all
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Imports now possible after path adjustment
from main import ARCTester
from cli.run_all import main, run_single_test_wrapper, AsyncRequestRateLimiter, get_or_create_rate_limiter
from src.schemas import Attempt # Import Attempt schema

# --- Test ARCTester Logging (main.py) ---

@pytest.mark.parametrize(
    "log_level_arg, expected_debug_present, expected_info_present",
    [
        ("DEBUG", True, True),
        ("INFO", False, True),
        ("WARNING", False, False),
        ("ERROR", False, False),
    ]
)
@patch('main.utils')
@patch('main.ARCTester.init_provider')
@patch('main.parse_and_validate_json') # Mock the parsing function directly
def test_arctester_log_levels(
    mock_parse_validate, mock_init_provider, mock_utils, caplog,
    log_level_arg, expected_debug_present, expected_info_present
):
    """
    Tests that ARCTester logs messages correctly based on the configured log level.
    Uses caplog to set the level for the 'main' logger.
    """
    log_level_numeric = getattr(logging, log_level_arg.upper())
    caplog.set_level(log_level_numeric, logger='main')

    # Setup Mocks
    mock_provider_instance = MagicMock()
    
    # Create a mock Attempt object that can be dumped and has necessary attributes
    mock_attempt_obj = MagicMock(spec=Attempt)
    mock_attempt_obj.metadata = MagicMock()
    # Simulate a response that the mocked parse_and_validate_json can handle
    mock_attempt_obj.metadata.choices = [MagicMock(message=MagicMock(content="Valid content for mock parser"))]
    # Define what model_dump should return - a serializable dict
    mock_attempt_obj.model_dump.return_value = {
        "task_id": "mock_task_001", 
        "test_id": "mock_config",
        "pair_index": 0,
        "answer": [[1]], # This should match what mock_parse_validate returns
        "metadata": {"usage": {"total_tokens": 10}} # Example metadata
    }
    # Mock make_prediction to return this mock Attempt
    mock_provider_instance.make_prediction.return_value = mock_attempt_obj
    
    mock_init_provider.return_value = mock_provider_instance

    # Mock parse_and_validate_json to return a valid answer structure
    # This is called within get_task_prediction
    mock_parse_validate.return_value = [[1]] 

    mock_utils.validate_data.return_value = True
    mock_utils.submission_exists.return_value = False
    mock_utils.get_train_pairs_from_task.return_value = [] # Empty training pairs
    mock_utils.get_test_input_from_task.return_value = [MagicMock(input=[[0]])] # Single test pair
    mock_utils.save_submission.return_value = "mock/path/saved.json"
    mock_utils.read_models_config.return_value = MagicMock(provider="mock_provider", model_name="mock_model")

    # Instantiate ARCTester
    arc_tester = ARCTester(
        config="mock_config",
        save_submission_dir="mock_submissions/mock_config",
        overwrite_submission=True,
        print_submission=True, # Enable submission printing for testing INFO logs
        num_attempts=1,
        retry_attempts=1
    )

    # Call the method that logs
    arc_tester.generate_task_solution(data_dir="mock_data", task_id="mock_task_001")

    log_text = caplog.text

    # Assertions based on expected presence for the level
    debug_msg_pattern = "Predicting attempt #1, retry #1"
    info_msg_pattern = "Running task mock_task_001"
    submission_log_pattern = "Final submission for task mock_task_001"

    # Check for message presence based on expected flags
    assert (debug_msg_pattern in log_text) == expected_debug_present
    assert (info_msg_pattern in log_text) == expected_info_present
    # Submission log is INFO level, tied to print_submission=True
    assert (submission_log_pattern in log_text) == expected_info_present

    # Example check: Ensure no DEBUG logs if level is INFO or higher
    if not expected_debug_present:
        assert debug_msg_pattern not in log_text
    # Example check: Ensure no INFO logs if level is WARNING or higher
    if not expected_info_present:
        assert info_msg_pattern not in log_text
        assert submission_log_pattern not in log_text

# --- Test Orchestrator Logging (cli/run_all.py) ---

def test_orchestrator_rate_limiter_log(caplog):
    """Specifically test if the rate limiter init log is captured."""
    caplog.set_level(logging.INFO, logger='cli.run_all')
    
    # Ensure caches are clear for this specific test
    from cli import run_all
    run_all.PROVIDER_RATE_LIMITERS.clear()
    run_all.MODEL_CONFIG_CACHE.clear()
    
    # Call the function that logs
    _ = get_or_create_rate_limiter("test_provider", {})
    
    # Check if the specific log message exists
    found = any(rec.getMessage().startswith("Initializing rate limiter for provider 'test_provider'") 
                for rec in caplog.records if rec.name == 'cli.run_all' and rec.levelno == logging.INFO)
    assert found, f"Rate limiter init log not found. Logs: {caplog.text}"


@pytest.mark.parametrize(
    "log_level_arg, expected_debug_present, expected_info_present",
    [
        ("DEBUG", True, True),
        ("INFO", False, True),
        ("WARNING", False, False),
        ("ERROR", False, False),
    ]
)
@pytest.mark.asyncio
@patch('cli.run_all.run_single_test_wrapper') # Mock the core workhorse function
@patch('cli.run_all.read_provider_rate_limits')
@patch('cli.run_all.get_model_config')
@patch('builtins.open')
@patch('cli.run_all.exit', side_effect=lambda code: None) # Prevent SystemExit
async def test_orchestrator_log_levels(
    mock_exit, mock_open, mock_get_model_config, mock_read_limits, mock_run_wrapper, caplog,
    log_level_arg, expected_debug_present, expected_info_present
):
    """
    Tests that the cli/run_all.py orchestrator logs its own messages correctly
    based on the configured log level. Uses caplog to set the level.
    """
    log_level_numeric = getattr(logging, log_level_arg.upper())
    # Ensure root logger is permissive enough for caplog to capture
    # logging.getLogger().setLevel(logging.DEBUG) # REMOVED - Let caplog handle it
    # Set level specifically for the logger under test
    caplog.set_level(log_level_numeric, logger='cli.run_all')

    # Mock setup
    mock_run_wrapper.return_value = True
    mock_read_limits.return_value = {"mock_provider": {"rate": 100, "period": 60}}
    mock_model_config = MagicMock(provider="mock_provider")
    mock_get_model_config.return_value = mock_model_config
    
    # Make mock_open return an object that supports context management AND iteration
    mock_file_handle = MagicMock()
    file_lines = ["task1\n", "task2\n"]
    mock_file_handle.__enter__.return_value.__iter__.return_value = iter(file_lines)
    mock_open.return_value = mock_file_handle 

    model_configs_list = ['mock_config']

    # Call the main async function
    await main(
        task_list_file="dummy.txt",
        model_configs_to_test=model_configs_list,
        data_dir="dummy_data",
        submissions_root="dummy_submissions",
        overwrite_submission=False,
        print_submission=False,
        num_attempts=1,
        retry_attempts=1
    )

    log_text = caplog.text

    # Define expected log messages at different levels
    # Note: Some DEBUG messages might be inside mocked functions like run_single_test_wrapper
    # We focus on messages logged directly by the `main` orchestrator function.
    info_msg_patterns = [
        "Starting ARC Test Orchestrator...",
        "Loaded 2 task IDs",
        "Total jobs to process: 2",
        "Loaded rate limits from provider_config.yml",
        # Use startswith for the rate limiter message due to variable details
        "Executing 2 tasks concurrently...",
        "Orchestrator Summary"
    ]
    # Add a DEBUG message example if one exists directly in `main` - currently none obvious.
    # debug_msg_pattern = "Example orchestrator debug message"

    # Assertions
    # if expected_debug_present:
    #     assert debug_msg_pattern in log_text
    # else:
    #     assert debug_msg_pattern not in log_text
        
    if expected_info_present:
        for pattern in info_msg_patterns:
            # Simplified assertion: Check if pattern substring exists in the whole log text
            assert pattern in log_text, f"Expected log pattern '{pattern}' not found in logs: {caplog.text}"
    else:
        # Check that none of the INFO patterns are present
        for pattern in info_msg_patterns:
            assert pattern not in log_text, f"Unexpected log pattern '{pattern}' found when level was {log_level_arg}: {caplog.text}"

@pytest.mark.asyncio
@patch('cli.run_all.run_single_test_wrapper', return_value=True)
@patch('cli.run_all.read_provider_rate_limits', return_value={})
@patch('cli.run_all.get_model_config')
@patch('builtins.open')
@patch('cli.run_all.exit', side_effect=lambda code: None) # Prevent SystemExit
async def test_orchestrator_log_level_none(mock_exit, mock_open, mock_get_model_config, mock_read_limits, mock_run_wrapper, caplog):
    """
    Tests that --log-level NONE effectively silences logging output.
    Uses caplog to check for absence of records.
    """
    # Set root logger high first, then specific logger if needed (though NONE handler should stop it)
    # logging.getLogger().setLevel(logging.CRITICAL + 1) # REMOVED - Rely on cli/run_all setup for NONE
    # logging.getLogger('cli.run_all').setLevel(logging.CRITICAL + 1) # REMOVED
    # Caplog captures everything regardless of handler levels, check records
    caplog.set_level(logging.DEBUG) 

    mock_model_config = MagicMock(provider="mock_provider")
    mock_get_model_config.return_value = mock_model_config
    
    # Make mock_open return an object that supports context management AND iteration
    mock_file_handle = MagicMock()
    file_lines = ["task1\n"]
    mock_file_handle.__enter__.return_value.__iter__.return_value = iter(file_lines)
    mock_open.return_value = mock_file_handle

    # Simulate the effect of --log-level NONE on the logger config
    # We already set the levels above, no need to simulate argparse/basicConfig here # REMOVED Comment
    # Setting levels directly as the test runs outside __main__ where basicConfig is called
    logging.getLogger('cli.run_all').setLevel(logging.CRITICAL + 1)
    # Also set root logger high in case of propagation or library logging
    logging.getLogger().setLevel(logging.CRITICAL + 1)
    
    model_configs_list = ['mock_config']

    await main(
        task_list_file="dummy.txt",
        model_configs_to_test=model_configs_list,
        data_dir="dummy_data",
        submissions_root="dummy_submissions",
        overwrite_submission=False,
        print_submission=False,
        num_attempts=1,
        retry_attempts=1
    )

    assert len(caplog.records) == 0, f"Expected no log records, but got {len(caplog.records)}: {caplog.text}"

    # Restore levels after test if needed, though pytest usually isolates
    # logging.getLogger('cli.run_all').setLevel(logging.NOTSET)
    # logging.getLogger().setLevel(logging.NOTSET)