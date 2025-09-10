import json
import pytest

from arc_agi_benchmarking.utils.generate_tasks_list import generate_task_list_from_dir
from arc_agi_benchmarking.utils.submission_exists import submission_exists
from arc_agi_benchmarking.utils.validate_data import validate_data


def test_generate_task_list_from_dir(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "task1.json").write_text("{}")
    (task_dir / "task2.json").write_text("{}")

    output_file = tmp_path / "out" / "task_list"
    tasks = generate_task_list_from_dir(str(task_dir), str(output_file))

    assert sorted(tasks) == ["task1", "task2"]

    output_path = output_file.with_suffix(".txt")
    assert output_path.exists()
    with open(output_path, "r") as f:
        lines = {line.strip() for line in f if line.strip()}
    assert lines == {"task1", "task2"}


def test_submission_exists(tmp_path):
    submission_dir = tmp_path / "subs"
    submission_dir.mkdir()
    task_id = "abc123"

    assert submission_exists(str(submission_dir), task_id) is False

    (submission_dir / f"{task_id}.json").write_text("{}")
    assert submission_exists(str(submission_dir), task_id) is True


def test_validate_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    task_id = "task1"
    task_json = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}]
    }
    (data_dir / f"{task_id}.json").write_text(json.dumps(task_json))

    assert validate_data(str(data_dir), task_id) is True

    with pytest.raises(ValueError):
        validate_data("", task_id)

    with pytest.raises(ValueError):
        validate_data(str(data_dir), "")

    with pytest.raises(ValueError):
        validate_data(str(data_dir / "missing"), task_id)

    bad_json_file = data_dir / "bad.json"
    bad_json_file.write_text("{bad json")
    with pytest.raises(ValueError):
        validate_data(str(data_dir), "bad")

    missing_keys = data_dir / "missing_keys.json"
    missing_keys.write_text(json.dumps({"train": []}))
    with pytest.raises(ValueError):
        validate_data(str(data_dir), "missing_keys")
