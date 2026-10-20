import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest

import record_store


def test_append_preserves_old_lines_and_utf8_multiline_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    first = record_store.new_record(temperature=0.2, max_tokens=512, stream=False)
    first["output_text"] = '中文第一行\n第二行，带有"引号"和\\反斜杠。'
    first["messages"] = [{"role": "user", "content": "问题\n第二行"}]
    first_path = record_store.append_record(first)
    original_bytes = first_path.read_bytes()
    second = record_store.new_record(temperature=0.7, max_tokens=128, stream=True)
    record_store.append_record(second)

    saved_bytes = first_path.read_bytes()
    assert saved_bytes.startswith(original_bytes)
    lines = saved_bytes.decode("utf-8").splitlines()
    assert len(lines) == 2
    assert "中文第一行" in lines[0]
    assert "\\n" in lines[0]
    assert json.loads(lines[0]) == first
    assert json.loads(lines[1]) == second
    assert first["run_id"] != second["run_id"]
    assert UUID(first["run_id"]).version == 4
    assert datetime.fromisoformat(first["created_at"]).utcoffset() is not None
    assert first_path == record_store.RUNS_PATH


def test_unknown_values_are_null_and_actual_zero_is_preserved() -> None:
    record = record_store.new_record(temperature=0.2, max_tokens=512, stream=False)
    record["usage"] = {"input_tokens": 10, "output_tokens": 0, "total_tokens": None}
    record_store.append_record(record)
    saved = json.loads(record_store.RUNS_PATH.read_text(encoding="utf-8"))
    for field in (
        "returned_model",
        "latency_ms",
        "first_text_latency_ms",
        "finish_reason",
    ):
        assert saved[field] is None
    assert saved["usage"] == {
        "input_tokens": 10,
        "output_tokens": 0,
        "total_tokens": None,
    }


def test_secret_is_redacted_without_mutating_the_in_memory_record() -> None:
    secret = 'test-only-key-"quoted"\\value'
    record = record_store.new_record(temperature=0.2, max_tokens=512, stream=False)
    record["messages"] = [{"role": "user", "content": f"测试意外输入：{secret}"}]
    record["output_text"] = f"测试意外回显：{secret}"
    record["finish_reason"] = secret

    record_store.append_record(record, api_key=secret)

    saved = json.loads(record_store.RUNS_PATH.read_text(encoding="utf-8"))
    assert saved["output_text"] == "测试意外回显：[REDACTED]"
    assert saved["messages"][0]["content"] == "测试意外输入：[REDACTED]"
    assert saved["finish_reason"] == "[REDACTED]"
    assert record["output_text"].endswith(secret)


@pytest.mark.parametrize("location", ["top-level", "parameters", "messages", "usage"])
def test_unapproved_fields_are_rejected_before_writing(location: str) -> None:
    record = record_store.new_record(temperature=0.2, max_tokens=512, stream=False)
    if location == "top-level":
        record["api_key"] = "test-only-secret"
    elif location == "parameters":
        record["parameters"]["headers"] = {"Authorization": "test-only-secret"}
    elif location == "messages":
        record["messages"] = [
            {"role": "user", "content": "问题", "api_key": "test-only-secret"}
        ]
    else:
        record["usage"] = {"api_key": "test-only-secret"}

    with pytest.raises(ValueError):
        record_store.append_record(record)
    assert not record_store.RUNS_PATH.exists()


@pytest.mark.parametrize("field", ["temperature", "output_text", "usage"])
def test_nested_objects_cannot_hide_credentials_in_allowed_fields(field: str) -> None:
    record = record_store.new_record(temperature=0.2, max_tokens=512, stream=False)
    if field == "temperature":
        record["parameters"]["temperature"] = {"api_key": "test-only-secret"}
    elif field == "output_text":
        record["output_text"] = {"api_key": "test-only-secret"}
    else:
        record["usage"] = {
            "input_tokens": {"api_key": "test-only-secret"},
            "output_tokens": 0,
            "total_tokens": 0,
        }

    with pytest.raises(ValueError):
        record_store.append_record(record)
    assert not record_store.RUNS_PATH.exists()


def test_serialization_failure_does_not_append_or_create_a_file() -> None:
    record = record_store.new_record(temperature=0.2, max_tokens=512, stream=False)
    record["latency_ms"] = float("nan")
    with pytest.raises(ValueError):
        record_store.append_record(record)
    assert not record_store.RUNS_PATH.exists()


def test_directory_creation_failure_is_reported_to_the_caller(tmp_path: Path) -> None:
    (tmp_path / "project").write_text("测试阻挡目录创建的普通文件", encoding="utf-8")
    record = record_store.new_record(temperature=0.2, max_tokens=512, stream=False)
    with pytest.raises(OSError):
        record_store.append_record(record)
