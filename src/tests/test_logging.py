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
from cli.run_all import main, run_single_test_wrapper, AsyncRequestRateLimiter

@pytest.mark.asyncio
@patch('cli.run_all.run_single_test_wrapper', return_value=True)
@patch('cli.run_all.read_provider_rate_limits', return_value={})
@patch('cli.run_all.get_model_config')
@patch('builtins.open')
@patch('cli.run_all.exit', side_effect=lambda code: None) # Prevent SystemExit
async def test_orchestrator_log_level_none(mock_exit, mock_open, mock_get_model_config, mock_read_limits, mock_run_wrapper, caplog):
    """
    Tests that --log-level NONE effectively silences logging output.
    """
    # caplog is set to DEBUG to capture everything; the test then verifies no app logs were made.
    caplog.set_level(logging.DEBUG) 

    mock_model_config = MagicMock(provider="mock_provider")
    mock_get_model_config.return_value = mock_model_config
    mock_file_handle = MagicMock()
    mock_file_handle.__enter__.return_value.readlines.return_value = ["task1\n"]
    mock_open.return_value = mock_file_handle
    
    args_namespace = argparse.Namespace(
        task_list_file="dummy.txt",
        model_configs='mock_config',
        data_dir="dummy_data",
        submissions_root="dummy_submissions",
        overwrite_submission=False,
        print_submission=False,
        num_attempts=1,
        retry_attempts=1,
        log_level="NONE"
    )

    # Simulate the logging setup from cli/run_all.py for log_level="NONE"
    if args_namespace.log_level == "NONE":
        log_level_to_set = logging.CRITICAL + 1
        logging.basicConfig(
            level=log_level_to_set,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True # Ensure re-configuration
        )
    else: # Should not happen in this test
        log_level_to_set = getattr(logging, args_namespace.log_level.upper())
        logging.basicConfig(
            level=log_level_to_set,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)],
            force=True
        )
        
    model_configs_list = [m.strip() for m in args_namespace.model_configs.split(',') if m.strip()]

    await main(
        task_list_file=args_namespace.task_list_file,
        model_configs_to_test=model_configs_list,
        data_dir=args_namespace.data_dir,
        submissions_root=args_namespace.submissions_root,
        overwrite_submission=args_namespace.overwrite_submission,
        print_submission=args_namespace.print_submission,
        num_attempts=args_namespace.num_attempts,
        retry_attempts=args_namespace.retry_attempts
    )

    assert len(caplog.records) == 0, f"Expected no log records, but got {len(caplog.records)}: {caplog.text}"