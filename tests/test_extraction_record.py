# 阅读提示：conftest.py 已把日志路径替换到 tmp_path；这些写入不会污染真实记录。
# json.loads 把 JSON 字符串读成 Python 字典，便于逐字段断言；None 在文件中表示 null。
# read_bytes 按原始字节读取，用于验证旧内容未变；splitlines 按日志的真实换行拆分记录。
# monkeypatch 的修改只对当前测试生效；pytest.raises 验证拒绝不合法数据的行为。

import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage

import extraction_record
import record_store
from extraction_models import AfterSalesExtraction
from extraction_record import (
    append_extraction_record,
    collect_extraction_metadata,
    new_extraction_record,
)


# 写入两条中文多行内容，确认 JSONL 逐行追加、编号不同，且 V01 文件完全不变。
def test_append_utf8_newlines_and_keep_v01_file_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    v01 = record_store.new_record(temperature=0.2, max_tokens=512, stream=False)
    v01_path = record_store.append_record(v01)
    # 保存旧文件字节快照，最后精确比较它是否被抽取日志写入影响。
    v01_bytes = v01_path.read_bytes()
    first = new_extraction_record(max_tokens=1024)
    first["input_text"] = '中文第一行\n第二行"引号"\\反斜杠。'
    first_path = append_extraction_record(first)
    original = first_path.read_bytes()
    second = new_extraction_record(max_tokens=2048)
    append_extraction_record(second)

    # 新内容必须以旧内容开头，证明进行了追加而不是覆盖。
    assert first_path.read_bytes().startswith(original)
    lines = first_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "中文第一行\\n第二行" in lines[0]
    assert json.loads(lines[0]) == first
    assert json.loads(lines[1]) == second
    # 把字符串还原为 UUID 并核对版本，不只是看编号像一串随机字符。
    assert UUID(first["run_id"]).version == 4
    assert first["run_id"] != second["run_id"]
    # 解析 ISO 时间；utcoffset() 不为 None 说明明确携带了时区。
    assert datetime.fromisoformat(first["created_at"]).utcoffset() is not None
    assert v01_path.read_bytes() == v01_bytes
    assert first_path == extraction_record.EXTRACTION_RUNS_PATH


# 让测试密钥出现在嵌套文本里，检查递归脱敏生效且内存中的原始记录不被修改。
def test_redaction_of_allowed_nested_fields_does_not_mutate_original() -> None:
    secret = 'test-secret-"quoted"\\value'
    record = new_extraction_record(max_tokens=1024)
    record["input_text"] = f"意外输入：{secret}"
    record["error_message"] = f"意外回显：{secret}"
    result = AfterSalesExtraction(
        order_id=None,
        product_name=None,
        issue_description=secret,
        requested_action=None,
        received_time_text=None,
        ambiguities=[secret],
        evidence=[{"field": "issue_description", "quote": secret}],
    )
    record["extraction"] = result.model_dump()

    path = append_extraction_record(record, api_key=secret)

    # 检查重新读出的落盘数据，确认实际文件中的嵌套文本已经脱敏。
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["input_text"] == "意外输入：[REDACTED]"
    assert saved["error_message"] == "意外回显：[REDACTED]"
    assert saved["extraction"]["issue_description"] == "[REDACTED]"
    assert saved["extraction"]["ambiguities"] == ["[REDACTED]"]
    assert saved["extraction"]["evidence"][0]["quote"] == "[REDACTED]"
    assert record["extraction"]["issue_description"] == secret
    assert record["input_text"].endswith(secret)


# 在不同层级夹带请求头、对象或错误类型，确认白名单在落盘之前拒绝它们。
@pytest.mark.parametrize(
    "location", ["top", "parameters", "usage", "extraction", "checks", "input", "error"]
)
def test_whitelist_rejects_extra_fields_and_nested_objects_before_writing(
    location: str,
) -> None:
    record = new_extraction_record(max_tokens=1024)
    secret_object = {"headers": {"Authorization": "not-for-log"}}
    if location == "top":
        record["headers"] = secret_object
    elif location == "parameters":
        record["parameters"]["headers"] = secret_object
    elif location == "usage":
        record["usage"] = {
            "input_tokens": secret_object,
            "output_tokens": None,
            "total_tokens": None,
        }
    elif location == "extraction":
        record["extraction"] = secret_object
    elif location == "checks":
        record["checks"] = secret_object
    elif location == "input":
        record["input_text"] = secret_object
    else:
        record["error_message"] = RuntimeError("not-for-log")

    # 如果保存函数接受了危险结构而没抛 ValueError，测试就会失败。
    with pytest.raises(ValueError):
        append_extraction_record(record)
    assert not extraction_record.EXTRACTION_RUNS_PATH.exists()


# NaN 不是标准 JSON 数值，序列化失败时不能创建日志文件。
def test_nan_is_rejected_before_creating_file() -> None:
    record = new_extraction_record(max_tokens=1024)
    # float("nan") 构造非数值标记；Python 能表示它，但严格 JSON 不允许。
    record["latency_ms"] = float("nan")
    with pytest.raises(ValueError):
        append_extraction_record(record)
    assert not extraction_record.EXTRACTION_RUNS_PATH.exists()


# 模拟 SDK 补出的零值，确认日志只信供应商实际用量字段，未知值仍为 null。
@pytest.mark.parametrize("raw_usage", [None, {}, {"completion_tokens": 0}])
def test_missing_usage_is_never_replaced_by_sdk_fallbacks(
    raw_usage: dict | None,
) -> None:
    record = new_extraction_record(max_tokens=1024)
    raw = AIMessage(
        content="",
        response_metadata={"token_usage": raw_usage},
        usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    # 只采集白名单统计，不把 raw 整个响应对象序列化到文件。
    collect_extraction_metadata(record, {"raw": raw})
    path = append_extraction_record(record)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if raw_usage is None:
        assert saved["usage"] is None
    else:
        assert saved["usage"] == {
            "input_tokens": None,
            "output_tokens": 0 if raw_usage else None,
            "total_tokens": None,
        }
