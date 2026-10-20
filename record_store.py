import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

RUNS_PATH = Path(__file__).resolve().parent / "data" / "runs" / "runs.jsonl"


class RunRecord(TypedDict):
    run_id: str
    created_at: str
    requested_model: str | None
    returned_model: str | None
    messages: list[dict[str, str]]
    parameters: dict[str, object]
    output_text: str
    latency_ms: float | None
    first_text_latency_ms: float | None
    usage: dict[str, int | None] | None
    finish_reason: str | None
    request_status: Literal["completed", "error", "cancelled"]
    error_type: str | None


def new_record(*, temperature: float, max_tokens: int, stream: bool) -> RunRecord:
    """运行开始时生成编号和带时区的 UTC 时间；未知值保持 None。"""
    return {
        "run_id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "requested_model": None,
        "returned_model": None,
        "messages": [],
        "parameters": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "thinking": {"type": "disabled"},
        },
        "output_text": "",
        "latency_ms": None,
        "first_text_latency_ms": None,
        "usage": None,
        "finish_reason": None,
        "request_status": "error",
        "error_type": None,
    }


def append_record(record: RunRecord, *, api_key: str = "") -> Path:
    """只追加白名单字段；保存失败时向调用方抛错，不吞掉异常。"""
    if set(record) != set(RunRecord.__annotations__):
        raise ValueError("运行记录字段不符合白名单。")
    if set(record["parameters"]) != {"temperature", "max_tokens", "stream", "thinking"}:
        raise ValueError("参数字段不符合白名单。")
    if record["parameters"]["thinking"] != {"type": "disabled"}:
        raise ValueError("思考模式配置不符合当前调用设置。")
    if any(set(message) != {"role", "content"} for message in record["messages"]):
        raise ValueError("消息字段不符合白名单。")
    if record["usage"] is not None and set(record["usage"]) != {
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }:
        raise ValueError("Token 用量字段不符合白名单。")
    if record["request_status"] not in {"completed", "error", "cancelled"}:
        raise ValueError("无效的请求状态。")

    parameters = record["parameters"]
    if (
        type(parameters["temperature"]) not in (int, float)
        or type(parameters["max_tokens"]) is not int
        or type(parameters["stream"]) is not bool
    ):
        raise ValueError("运行参数类型不正确。")
    for name in ("latency_ms", "first_text_latency_ms"):
        if record[name] is not None and type(record[name]) not in (int, float):
            raise ValueError("耗时类型不正确。")
    if record["usage"] is not None and any(
        value is not None and type(value) is not int
        for value in record["usage"].values()
    ):
        raise ValueError("Token 用量类型不正确。")

    def redact(text: str) -> str:
        return text.replace(api_key, "[REDACTED]") if api_key else text

    payload = dict(record)
    for name in (
        "run_id",
        "created_at",
        "requested_model",
        "returned_model",
        "output_text",
        "finish_reason",
        "error_type",
    ):
        value = record[name]
        if value is not None and not isinstance(value, str):
            raise ValueError("运行记录文本字段类型不正确。")
        payload[name] = redact(value) if value is not None else None
    payload["messages"] = []
    for message in record["messages"]:
        if message["role"] not in {"system", "user"} or not isinstance(
            message["content"], str
        ):
            raise ValueError("消息类型不正确。")
        payload["messages"].append(
            {"role": message["role"], "content": redact(message["content"])}
        )

    # 不使用 default=str，不序列化客户端或任意对象。转义后的换行仍在同一行内。
    line = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_PATH.open("a", encoding="utf-8") as file:
        file.write(line + "\n")
    return RUNS_PATH
