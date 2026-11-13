import json  # 将记录字典序列化为一行 JSON。
from datetime import UTC, datetime  # 生成带 UTC 时区的时间戳。
from pathlib import Path  # 文件路径与文件读写接口。
from typing import Literal, TypedDict  # 描述记录结构；实际校验仍由下面的函数执行。
from uuid import uuid4  # 生成随机的唯一运行编号。

# V01 日志位置仍保持不变；V02 复用写入函数，但使用 extraction.jsonl。
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
        "run_id": str(uuid4()),  # UUID 对象转成字符串，便于存入 JSON。
        "created_at": datetime.now(UTC).isoformat(),  # ISO 格式会保留时区信息。
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
    # __annotations__ 提供 TypedDict 声明的字段；只允许这组确定的键。
    if set(record) != set(RunRecord.__annotations__):
        raise ValueError("运行记录字段不符合白名单。")
    if set(record["parameters"]) != {"temperature", "max_tokens", "stream", "thinking"}:
        raise ValueError("参数字段不符合白名单。")
    if record["parameters"]["thinking"] != {"type": "disabled"}:
        raise ValueError("思考模式配置不符合当前调用设置。")
    # any() 发现任意一条消息包含不允许的字段就返回 True。
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

    # 类型提示不会自动保护日志；下面显式拒绝嵌套对象和错误的值类型。
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

    # 内部函数可以读取外层 api_key，但不会把它放进返回的日志字典。
    def redact(text: str) -> str:
        return text.replace(api_key, "[REDACTED]") if api_key else text

    payload = dict(record)  # 浅复制最外层字典，随后逐项替换文本和消息列表。
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
    payload["messages"] = []  # 为消息创建新列表，避免修改原始输入记录。
    for message in record["messages"]:
        if message["role"] not in {"system", "user"} or not isinstance(
            message["content"], str
        ):
            raise ValueError("消息类型不正确。")
        payload["messages"].append(
            {"role": message["role"], "content": redact(message["content"])}
        )

    return append_jsonl(payload, RUNS_PATH)


def append_jsonl(payload: dict[str, object], path: Path) -> Path:
    """共享追加写入机制；调用方须先完成字段白名单校验和脱敏。"""
    # 本练习只支持单进程串行写入；没有文件锁，不保证多进程/多线程并发安全。
    # 不使用 default=str，不序列化客户端或任意对象。转义后的换行仍在同一行内。
    # ensure_ascii=False 保留中文；allow_nan=False 拒绝 JSON 不支持的 NaN/Infinity。
    # separators 去掉无用空格；回答里的换行被转义，不会变成多条日志。
    line = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    # parent 指目录；parents=True 创建缺少的上级目录，exist_ok=True 允许目录已存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # with 是上下文管理器：退出代码块时自动关闭文件，即使中途出现异常。
    # "a" 表示追加，不能写成会覆盖旧记录的 "w"；UTF-8 用于中文文本。
    with path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")  # 每条完整 JSON 后只追加一个真实换行符。
    return path  # 写入成功后才返回路径，调用方据此显示“已保存”。
