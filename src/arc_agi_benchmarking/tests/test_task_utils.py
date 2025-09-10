import json
import os
import pytest
from pathlib import Path

from arc_agi_benchmarking.utils.task_utils import (
    get_train_pairs_from_task,
    get_test_input_from_task,
    save_submission,
    normalize_model_name,
    read_models_config,
    is_submission_correct,
    read_provider_rate_limits,
)


def create_sample_task(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    task_id = "task1"
    task_json = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}]
    }
    (data_dir / f"{task_id}.json").write_text(json.dumps(task_json))
    return data_dir, task_id, task_json


def test_get_train_and_test_pairs(tmp_path):
    data_dir, task_id, task_json = create_sample_task(tmp_path)
    train_pairs = get_train_pairs_from_task(str(data_dir), task_id)
    test_pairs = get_test_input_from_task(str(data_dir), task_id)

    assert len(train_pairs) == 1
    assert train_pairs[0].input == task_json["train"][0]["input"]
    assert train_pairs[0].output == task_json["train"][0]["output"]

    assert len(test_pairs) == 1
    assert test_pairs[0].input == task_json["test"][0]["input"]
    assert test_pairs[0].output is None


def test_save_submission(tmp_path):
    submission_dir = tmp_path / "subs"
    task_id = "task1"
    submission = [{"attempt_1": {"answer": [[4]]}}]

    path = save_submission(str(submission_dir), task_id, submission)
    assert Path(path).exists()
    saved = json.loads(Path(path).read_text())
    assert saved == submission


def test_normalize_model_name():
    assert normalize_model_name("claude-3.5-sonnet") == "claude-3-5-sonnet"
    assert normalize_model_name("claude-3-5-sonnet-20240315") == "claude-3-5-sonnet"
    assert normalize_model_name("claude-3-5-sonnet-latest") == "claude-3-5-sonnet"
    assert normalize_model_name("foo..bar--baz") == "foo-bar-baz"


def test_read_models_config():
    config = read_models_config("gpt-5-2025-08-07-high")
    assert config.model_name == "gpt-5-2025-08-07"
    assert config.provider == "openai"
    with pytest.raises(ValueError):
        read_models_config("nonexistent")


def test_is_submission_correct(tmp_path):
    data_dir, task_id, task_json = create_sample_task(tmp_path)
    correct_submission = [{"attempt_1": {"answer": [[4]]}}]
    wrong_submission = [{"attempt_1": {"answer": [[5]]}}]

    assert is_submission_correct(correct_submission, str(data_dir), task_id)
    assert not is_submission_correct(wrong_submission, str(data_dir), task_id)


def test_read_provider_rate_limits():
    limits = read_provider_rate_limits()
    assert "openai" in limits
    assert limits["openai"]["rate"] == 400
    assert limits["openai"]["period"] == 60
