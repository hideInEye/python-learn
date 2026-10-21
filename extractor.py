"""对照三种输出方式；每次只执行一次非流式模型调用。"""

# json 是标准库：在 Python 字典和 JSON 字符串之间转换。
import json

# AIMessage 是 LangChain 的模型响应类型；正文、工具调用和元数据分开存放。
from langchain_core.messages import AIMessage

# ChatDeepSeek 是已安装的模型客户端；这里只导入类型，不会发起请求。
from langchain_deepseek import ChatDeepSeek

# Pydantic 在字段类型、必需字段或 JSON 格式不符合要求时抛出此异常。
from pydantic import ValidationError

# 导入项目唯一的抽取 Schema，提示词和校验都以它为准。
from extraction_models import AfterSalesExtraction

# CLI 名称与实际调用方式显式区分，不使用供应商的 json_schema 方法。
OUTPUT_METHODS = {
    "prompt": "invoke",  # 靠提示词约束格式，普通调用返回正文。
    "json": "json_mode",  # 请求 JSON 模式，应用仍需检查字段和证据。
    "structured": "function_calling",  # 把抽取结果放入工具调用参数。
}


# def 定义函数；str 是参数类型提示，-> str 是返回值类型提示。
# * 后的参数必须写名称，例如 mode="json"，不能只传一个位置值。
def build_extraction_prompt(business_prompt: str, *, mode: str) -> str:
    """共用业务规则、Schema 和完整示例，只改变输出方式指令。"""
    if mode not in OUTPUT_METHODS:  # 对字典用 in / not in，检查的是键名。
        raise ValueError("不支持的抽取模式。")
    # Python 的条件表达式是“值 A if 条件 else 值 B”，类似 JS 的三元表达式。
    # 圆括号让表达式跨行；相邻字符串字面量会自动拼接。
    output_instruction = (
        "只返回一次 AfterSalesExtraction 工具调用，参数必须符合下面的 JSON Schema。"
        if mode == "structured"
        else "仅输出一个符合下面 JSON Schema 的 JSON 对象。"
        "不要输出 Markdown 代码围栏、解释文字或多个 JSON 对象。"
    )
    # model_json_schema() 从 Pydantic 类生成字段规则，返回 Python 字典。
    # dumps() 转成字符串；ensure_ascii=False 保留中文，indent=2 便于阅读。
    schema = json.dumps(
        AfterSalesExtraction.model_json_schema(), ensure_ascii=False, indent=2
    )
    # 这是明确标注的格式示例，绝不作为模型请求失败时的返回值。
    example_text = "订单 DEMO-001 的台灯不亮，前天收到，请帮我维修。"
    # 调用 Schema 类会创建并校验一个数据对象；这里不会调用 LLM。
    example = AfterSalesExtraction(
        order_id="DEMO-001",
        product_name="台灯",
        issue_description="不亮",
        requested_action="repair",
        received_time_text="前天收到",
        ambiguities=[],  # [] 是空列表：这个格式示例没有待确认的歧义。
        evidence=[  # 每个字典表示一个字段及支持它的原文片段。
            {"field": "order_id", "quote": "DEMO-001"},
            {"field": "product_name", "quote": "台灯"},
            {"field": "issue_description", "quote": "不亮"},
            {"field": "requested_action", "quote": "请帮我维修"},
            {"field": "received_time_text", "quote": "前天收到"},
        ],
    )
    # f"...{变量}..." 是格式化字符串，类似 JS 模板字符串。
    # rstrip() 只去掉业务提示词末尾空白；\n 是换行字符，不是实际字符 n。
    # model_dump_json() 把已校验的数据对象转成 JSON 字符串，作为格式示例。
    return (
        f"{business_prompt.rstrip()}\n\n{output_instruction}\n\n"
        f"【JSON Schema】\n{schema}\n\n"
        "【完整格式示例】\n"
        "下面仅演示格式，不是本次用户事实，不要复制示例值。\n"
        f"示例输入：{example_text}\n"
        f"示例 JSON 对象（工具模式下为调用参数）：\n{example.model_dump_json(indent=2)}"
    )


# 自定义异常继承 ValueError，让 CLI 能单独识别“响应校验失败”。
# 这只是错误分类，不是新的模型客户端或自动重试机制。
class ExtractionParseError(ValueError):
    """响应没有包含唯一、有效的售后抽取结果。"""


def extract_after_sales(
    model: ChatDeepSeek, text: str, system_prompt: str, *, mode: str = "structured"
) -> AfterSalesExtraction:
    """保留 V02-A 的接口；CLI 分开调用请求和校验，以保存失败响应的统计。"""
    # 先求值内层 invoke_extraction() 获取响应，再交给外层函数校验。
    # mode 的默认值保证旧调用不传该参数时仍走 structured。
    return parse_extraction_result(
        invoke_extraction(model, text, system_prompt, mode=mode), mode=mode
    )


def invoke_extraction(
    model: ChatDeepSeek, text: str, system_prompt: str, *, mode: str = "structured"
) -> dict[str, object]:
    """沿用已验收的调用方式，只请求一次，不自动修复或切换方法。"""
    if mode not in OUTPUT_METHODS:
        raise ValueError("不支持的抽取模式。")
    # 列表中的两个元组分别表示系统规则和用户输入；元组使用圆括号。
    # LangChain 接受这种简写，并在内部转换成消息对象。
    messages = [("system", system_prompt), ("human", text)]
    if mode == "prompt":
        # 留到校验阶段再解析，避免把 JSON 格式错误归类为请求失败。
        return {"raw": model.invoke(messages)}  # invoke 是真正发起一次模型请求的位置。
    # 创建带结构化输出配置的 Runnable；这一步配置调用方式，不生成回答。
    structured_model = model.with_structured_output(
        AfterSalesExtraction,
        method=OUTPUT_METHODS[mode],  # 用 CLI 模式查字典，得到框架的方法名。
        include_raw=True,  # 同时保留 raw、parsed、parsing_error，便于检查失败。
    )
    # 只有本行执行请求；未追加任何修复调用、第二次请求或模式降级。
    return structured_model.invoke(messages)


def parse_extraction_result(
    result: object, *, mode: str = "structured"
) -> AfterSalesExtraction:
    """只验证 Schema 和工具调用结构；业务检查由 extraction_rules 执行。"""
    if mode not in OUTPUT_METHODS:
        raise ValueError("不支持的抽取模式。")
    # object 类型提示表示输入可能是任意对象，因此先做运行时类型检查。
    if not isinstance(result, dict):
        raise ExtractionParseError("响应不是有效的抽取结果。")
    # and 要求两边条件都成立；prompt 的普通响应没有框架解析状态。
    if mode != "prompt" and "parsing_error" not in result:
        raise ExtractionParseError("结构化响应缺少解析状态。")

    # dict.get() 在键缺失时返回 None；result["raw"] 则会直接抛 KeyError。
    raw = result.get("raw")
    if not isinstance(raw, AIMessage):
        raise ExtractionParseError("结构化响应缺少有效的原始消息。")

    if mode in {"prompt", "json"}:  # 花括号内没有键值对时，这里创建的是集合。
        # 非空列表为真，空列表为假；or 表示任意一种工具调用存在就拒绝。
        if (
            raw.tool_calls
            or raw.invalid_tool_calls
            or raw.additional_kwargs.get("tool_calls")
        ):
            raise ExtractionParseError("文本输出模式不应返回工具调用。")
        if mode == "json":
            # is not None 检查是否真的有错误对象，不能靠对象的真假值替代。
            if result["parsing_error"] is not None:
                raise ExtractionParseError("JSON 响应无法解析或未通过 Schema 校验。")
            # 没有异常并不保证有结果；None 和普通字典都不算有效 Schema 对象。
            if not isinstance(result.get("parsed"), AfterSalesExtraction):
                raise ExtractionParseError("未得到有效的 AfterSalesExtraction 结果。")
        try:
            # .text 只取可见文本，不包含额外推理。直接校验全文，不截取或去围栏。
            # json_mode 的框架解析器可能容忍 Markdown / 截断 JSON；
            # 对照实验仍以原始正文是否为完整、合法的 JSON 为准。
            return AfterSalesExtraction.model_validate_json(raw.text)
        except ValidationError:  # try 内校验失败时走这里，成功时直接 return。
            # from None 隐藏原始异常链，避免原始响应内容出现在错误输出中。
            raise ExtractionParseError(
                "原始正文不是符合 Schema 的完整 JSON 对象。"
            ) from None

    # 当前框架的 Pydantic 解析器使用 first_tool_only=True。
    # 同时检查原始和标准化调用列表，避免悄悄忽略多余或无效的工具调用。
    # additional_kwargs 放供应商额外字段；仅查看工具调用，不输出推理内容。
    original_calls = raw.additional_kwargs.get("tool_calls")
    if original_calls is not None and (
        not isinstance(original_calls, list) or len(original_calls) != 1
    ):
        raise ExtractionParseError("原始响应必须恰好包含一个工具调用。")
    # len() 计算数量；有效和无效的工具调用合计必须恰好一个。
    if len(raw.tool_calls) + len(raw.invalid_tool_calls) != 1:
        raise ExtractionParseError("响应必须恰好包含一个工具调用。")
    if result["parsing_error"] is not None:
        raise ExtractionParseError("工具调用参数无法解析或未通过 Schema 校验。")
    # or 会短路：存在 invalid_tool_calls 时，不再访问可能为空的 tool_calls[0]。
    # [0] 取第一个元素；__name__ 读取 Schema 类名，避免另写一份工具名。
    if (
        raw.invalid_tool_calls
        or raw.tool_calls[0]["name"] != AfterSalesExtraction.__name__
    ):
        raise ExtractionParseError("响应未包含有效的 AfterSalesExtraction 工具调用。")

    # structured 结果可以只在工具参数中；不要求 raw 的可见正文非空。
    parsed = result.get("parsed")
    if not isinstance(parsed, AfterSalesExtraction):
        raise ExtractionParseError("未得到有效的 AfterSalesExtraction 结果。")
    return parsed  # 返回已经通过 Schema 校验的对象，业务规则由后续模块检查。
