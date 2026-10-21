"""抽取任务专用记录白名单，复用 V01 的 JSONL 追加写入机制。"""

from pathlib import Path  # 生成固定日志路径和创建目录。
from typing import Literal, TypedDict, get_args  # 类型描述，以及读取 Literal 允许的值。

from langchain_core.messages import AIMessage

from extraction_models import SCHEMA_VERSION, AfterSalesExtraction
from extraction_rules import CheckResult
from extractor import OUTPUT_METHODS
from record_store import append_jsonl, new_record

# __file__ 是当前 Python 文件的路径；resolve().parent 得到它所在的绝对目录。
# 抽取日志使用独立文件，因此不改变 V01 的 runs.jsonl 字段格式。
EXTRACTION_RUNS_PATH = (
    Path(__file__).resolve().parent / "data" / "runs" / "extraction.jsonl"
)
# 字面量联合类型记录允许的状态；下面的保存函数还会做运行时白名单校验。
ExtractionStatus = Literal[
    "input_error",
    "request_error",
    "validation_error",
    "evidence_error",
    "needs_clarification",
    "draft_ready",
    "cancelled",
]


class ExtractionRecord(TypedDict):
    task_type: str  # 标识这是一条售后抽取任务记录。
    run_id: str  # 每次运行的唯一编号，连接控制台输出与日志。
    created_at: str  # 带时区的 ISO 时间字符串。
    schema_version: str  # 本次字段规则的版本。
    prompt_sha256: str | None  # 完整系统提示词的哈希；未构建成功则为空。
    requested_model: str | None  # 配置指定的模型名称。
    returned_model: str | None  # 能确认来自响应的实际标识，未知时不补请求值。
    input_text: str | None  # 待抽取用户原文；文件读取失败时可能为空。
    mode: str  # CLI 名称：prompt、json 或 structured。
    output_mode: str  # 实际方式：invoke、json_mode 或 function_calling。
    parameters: dict[str, object]  # 只保存温度、Token 上限等允许的参数。
    extraction: dict[str, object] | None  # Schema 通过后的抽取结果。
    checks: CheckResult | None  # 业务检查结果；未执行时为空。
    status: ExtractionStatus  # 本次处理的最终状态。
    latency_ms: float | None  # 毫秒耗时，float 表示可带小数。
    usage: dict[str, int | None] | None  # 实际 Token 计数，未知项保留 None。
    finish_reason: str | None  # 供应商返回的结束原因，例如 length。
    error_type: str | None  # 异常类别名或本地错误类别，不保存异常对象。
    error_message: str | None  # 固定安全说明，不复制供应商异常全文。


def new_extraction_record(
    *, max_tokens: int, mode: str = "structured"
) -> ExtractionRecord:
    if mode not in OUTPUT_METHODS:
        raise ValueError("不支持的抽取模式。")
    # 复用已有编号、时间和基本参数的生成逻辑，不复制整个客户端或配置。
    base = new_record(temperature=0.2, max_tokens=max_tokens, stream=False)
    # 每次调用都新建一个字典，不会与前一次运行共用可变数据。
    return {
        "task_type": "after_sales_extraction",
        "run_id": base["run_id"],
        "created_at": base["created_at"],
        "schema_version": SCHEMA_VERSION,
        "prompt_sha256": None,
        "requested_model": None,
        "returned_model": None,
        "input_text": None,
        "mode": mode,
        "output_mode": OUTPUT_METHODS[mode],  # 根据 CLI 名称查实际方法。
        # ** 在字典中展开已有键值对，类似 JS 的 {...object, timeout: 30}。
        "parameters": {**base["parameters"], "timeout": 30, "max_retries": 0},
        "extraction": None,
        "checks": None,
        "status": "input_error",  # 初始值；后续完成请求和检查时再更新。
        "latency_ms": None,
        "usage": None,
        "finish_reason": None,
        "error_type": None,
        "error_message": None,
    }


def collect_extraction_metadata(record: ExtractionRecord, response: object) -> None:
    """只提取用量、结束原因和可确认的模型标识，不保存整个响应或推理内容。"""
    # 先检查类型再调用 get，避免畸形响应导致“对象没有 get 方法”的错误。
    raw = response.get("raw") if isinstance(response, dict) else None
    if not isinstance(raw, AIMessage):
        return  # 没有有效响应就保留记录里的未知值，不补造统计。
    metadata = raw.response_metadata  # 框架提供的响应元数据字典。
    returned_model = metadata.get("model_name")  # 键缺失时 get 返回 None。
    # langchain-openai 1.7.2 可能用请求模型回填 model_name。
    # 同名无法证明供应商返回了标识，保守记为 null，绝不拿请求值冒充。
    if (
        isinstance(returned_model, str)
        and returned_model.strip()
        and returned_model != record["requested_model"]
    ):
        record["returned_model"] = returned_model  # 原地更新传入的记录字典。
    if isinstance(metadata.get("finish_reason"), str):
        record["finish_reason"] = metadata["finish_reason"]
    # 当前 SDK 的 usage_metadata 会将缺少的计数补 0、推算 total_tokens。
    # 非流式响应的 token_usage 保留供应商字段，因此只从这里取实际返回值。
    token_usage = metadata.get("token_usage")
    if isinstance(token_usage, dict):
        # 字典推导式按固定映射生成三个字段，不复制供应商元数据的其他键。
        # := 是赋值表达式：读取供应商值的同时绑定到 value，避免重复读取。
        # type(value) is int 还会拒绝 bool；Python 的 bool 是 int 的子类。
        record["usage"] = {
            name: value
            if type(value := token_usage.get(provider_name)) is int
            else None
            for name, provider_name in (
                ("input_tokens", "prompt_tokens"),
                ("output_tokens", "completion_tokens"),
                ("total_tokens", "total_tokens"),
            )
        }


def redact_values(value: object, api_key: str) -> object:
    """仅处理已通过白名单的 JSON 值；不将异常、客户端等对象强转为字符串。"""
    if isinstance(value, str):  # 叶子是文本时，替换已知密钥的完整匹配。
        return value.replace(api_key, "[REDACTED]") if api_key else value
    if isinstance(value, list):  # 递归处理列表，生成新列表而非修改原列表。
        return [redact_values(item, api_key) for item in value]
    if isinstance(value, dict):  # items() 依次提供每个键和值。
        return {key: redact_values(item, api_key) for key, item in value.items()}
    return value  # 数字、布尔值和 None 不需要文字替换，保持原值。


def append_extraction_record(record: ExtractionRecord, *, api_key: str = "") -> Path:
    """校验白名单后追加；错误信息只接收调用方生成的固定说明，不接收 str(exc)。"""
    # set(dict) 取字典的键集合；__annotations__ 存放 TypedDict 声明的字段。
    # 集合必须完全相同：未知字段与缺字段都会被拒绝。
    if set(record) != set(ExtractionRecord.__annotations__):
        raise ValueError("抽取记录字段不符合白名单。")
    # get_args(Literal[...]) 返回允许的字符串值；模式与实际方法也必须匹配。
    if (
        record["task_type"] != "after_sales_extraction"
        or record["mode"] not in OUTPUT_METHODS
        or record["output_mode"] != OUTPUT_METHODS[record["mode"]]
        or record["status"] not in get_args(ExtractionStatus)
    ):
        raise ValueError("抽取任务类型、输出模式或状态无效。")
    # 文本字段可以未知，但不能偷偷放入客户端对象、请求头字典等其他类型。
    for name in (
        "run_id",
        "created_at",
        "schema_version",
        "prompt_sha256",
        "requested_model",
        "returned_model",
        "input_text",
        "finish_reason",
        "error_type",
        "error_message",
    ):
        if record[name] is not None and not isinstance(record[name], str):
            raise ValueError("抽取记录文本字段类型不正确。")

    parameters = record["parameters"]  # 再检查嵌套参数，不能只限制最外层字段。
    if set(parameters) != {
        "temperature",
        "max_tokens",
        "stream",
        "thinking",
        "timeout",
        "max_retries",
    }:
        raise ValueError("抽取参数字段不符合白名单。")
    # 数字、布尔值、思考模式分别验证；这些检查是运行时执行的。
    if (
        type(parameters["temperature"]) not in (int, float)
        or type(parameters["max_tokens"]) is not int
        or parameters["max_tokens"] <= 0
        or parameters["stream"] is not False
        or parameters["thinking"] != {"type": "disabled"}
        or type(parameters["timeout"]) is not int
        or type(parameters["max_retries"]) is not int
    ):
        raise ValueError("抽取参数值不符合当前调用设置。")
    # 耗时只接受数值或未知值；JSON 写入层还会拒绝 NaN 等非法数值。
    if record["latency_ms"] is not None and type(record["latency_ms"]) not in (
        int,
        float,
    ):
        raise ValueError("抽取耗时类型不正确。")
    # any(...) 只要发现一项类型不符就返回 True；未知计数仍可使用 None。
    if record["usage"] is not None and (
        set(record["usage"]) != {"input_tokens", "output_tokens", "total_tokens"}
        or any(
            value is not None and type(value) is not int
            for value in record["usage"].values()
        )
    ):
        raise ValueError("Token 用量字段或类型不符合白名单。")

    if record["extraction"] is not None:
        # 再检查嵌套字段，禁止在合法字段中夹带请求头或任意对象。
        AfterSalesExtraction.model_validate(record["extraction"])
    checks = record["checks"]  # None 表示流程尚未到业务检查阶段。
    if checks is not None:
        if set(checks) != set(CheckResult.__annotations__) or checks["status"] not in {
            "evidence_error",
            "needs_clarification",
            "draft_ready",
        }:
            raise ValueError("检查结果字段不符合白名单。")
        # 每项必须是文本列表；只有 missing_fields 允许用 None 表示未检查。
        for name in ("evidence_errors", "missing_fields", "clarification_questions"):
            value = checks[name]
            if name == "missing_fields" and value is None:
                continue  # 跳过当前字段，继续检查列表中的下一个字段。
            if not isinstance(value, list) or any(
                not isinstance(item, str) for item in value
            ):
                raise ValueError("检查结果必须为文本列表。")
        if checks["status"] != record["status"]:  # 避免日志同时给出矛盾的结论。
            raise ValueError("检查结果与最终状态不一致。")

    payload = redact_values(record, api_key)  # 校验后复制并脱敏，不修改原记录。
    return append_jsonl(payload, EXTRACTION_RUNS_PATH)  # 真正追加文件，并返回保存路径。
