"""三种模式的离线对照：所有回答仅来自这里的 Mock，不请求真实模型。"""
# 阅读提示：这里比较结构和流程，不比较真实模型的措辞；所有模型响应由本文件模拟。
# fixture 通过同名参数注入；参数化 fixture 或 parametrize 会让同一函数跑多组离线用例。
# Mock 是可控制的替身；assert_called_once 检查只调用一次，assert_not_called 检查从未调用。
# pytest.raises 用来验证预期异常；异常没发生反而会让测试失败。
# with_structured_output 的真实解析器测试只替换响应来源，网络保护仍然启用。

import hashlib
import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_deepseek import ChatDeepSeek
from pydantic import ValidationError

import config
import extract
import extraction_record
import llm_client
from extraction_models import AfterSalesExtraction
from extraction_record import append_extraction_record, new_extraction_record
from extractor import (
    ExtractionParseError,
    build_extraction_prompt,
    invoke_extraction,
    parse_extraction_result,
)

TEXT = "订单 A10086 的耳机左边没声音，昨天收到，我想换货。"
SECRET = "test-only-mode-key"
METHODS = {"prompt": "invoke", "json": "json_mode", "structured": "function_calling"}


# 参数化 fixture 为每个使用它的测试分别提供 prompt、json、structured 三种模式。
@pytest.fixture(params=["prompt", "json", "structured"])
def mode(request: pytest.FixtureRequest) -> str:
    # 读取 fixture 装饰器 params 列表中当前这一轮的模式。
    return request.param


# 提供完整测试样本；每个测试得到自己的字典，修改它不会影响其他测试。
@pytest.fixture
def payload() -> dict:
    return {
        "order_id": "A10086",
        "product_name": "耳机",
        "issue_description": "左边没声音",
        "requested_action": "exchange",
        "received_time_text": "昨天收到",
        "ambiguities": [],
        "evidence": [
            {"field": "order_id", "quote": "A10086"},
            {"field": "product_name", "quote": "耳机"},
            {"field": "issue_description", "quote": "左边没声音"},
            {"field": "requested_action", "quote": "我想换货"},
            {"field": "received_time_text", "quote": "昨天收到"},
        ],
    }


# 按模式构造测试响应；字段非法时用真实 Pydantic 异常模拟 parsing_error。
def fake_response(mode: str, payload: dict) -> AIMessage | dict:
    """构造明确的测试响应；非法 Schema 用真实 Pydantic 异常表达。"""
    # json.dumps 把测试字典转成响应正文；ensure_ascii=False 保留可读中文。
    raw = AIMessage(content=json.dumps(payload, ensure_ascii=False))
    # 普通 invoke 返回消息；另两种模式返回 raw/parsed/parsing_error 字典。
    if mode == "prompt":
        return raw
    if mode == "structured":
        raw = AIMessage(
            content="",
            tool_calls=[
                {"name": "AfterSalesExtraction", "args": payload, "id": "test-call"}
            ],
        )
    try:
        # 真实 Pydantic 校验测试数据，非法字段会产生真实的 ValidationError。
        parsed = AfterSalesExtraction.model_validate(payload)
    except ValidationError as exc:
        return {"raw": raw, "parsed": None, "parsing_error": exc}
    return {"raw": raw, "parsed": parsed, "parsing_error": None}


# 替换模型构造器并设置假响应；使用 spec 限制 Mock 暴露的属性，减少接口拼写错误。
@pytest.fixture
def fake_model(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    payload: dict,
) -> tuple[Mock, Mock]:
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    monkeypatch.setenv("LLM_MODEL", "test-only-comparison-model")
    model = Mock(spec=ChatDeepSeek)
    # 条件表达式选择该模式对应的 invoke 层级，避免模拟错调用位置。
    operation = (
        model.invoke
        if mode == "prompt"
        else model.with_structured_output.return_value.invoke
    )
    operation.return_value = fake_response(mode, payload)
    # return_value=model 使业务代码调用构造器时拿到这个测试替身。
    constructor = Mock(return_value=model)
    monkeypatch.setattr(llm_client, "ChatDeepSeek", constructor)
    return model, constructor


# 普通模式调用 model.invoke，另两种模式调用结构化包装器的 invoke。
def called_operation(model: Mock, mode: str) -> Mock:
    return (
        model.invoke
        if mode == "prompt"
        else model.with_structured_output.return_value.invoke
    )


# 只从已被 fixture 替换的临时路径读取日志，不访问生产记录。
def read_record() -> dict:
    return json.loads(
        extraction_record.EXTRACTION_RUNS_PATH.read_text(encoding="utf-8")
    )


# 检查各模式使用正确接口、相同预算和参数，并把真实发送的提示词哈希写入日志。
def test_modes_use_the_requested_api_shared_settings_and_record_actual_prompt(
    mode: str,
    payload: dict,
    fake_model: tuple[Mock, Mock],
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, constructor = fake_model

    assert extract.main(["--text", TEXT, "--mode", mode, "--max-tokens", "2048"]) == 0

    operation = called_operation(model, mode)
    operation.assert_called_once()
    if mode == "prompt":
        model.with_structured_output.assert_not_called()
    else:
        model.with_structured_output.assert_called_once_with(
            AfterSalesExtraction,
            method=METHODS[mode],
            include_raw=True,
        )
        model.invoke.assert_not_called()
    model.stream.assert_not_called()
    constructor.assert_called_once_with(
        model="test-only-comparison-model",
        api_key=SECRET,
        temperature=0.2,
        max_tokens=2048,
        timeout=30,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    # call_args 保存最近一次调用；args[0] 是第一个位置参数，也就是消息列表。
    messages = operation.call_args.args[0]
    assert len(messages) == 2
    assert messages[1] == ("human", TEXT)
    assert messages[0][0] == "system"
    assert messages[0][1].startswith(
        extract.PROMPT_PATH.read_text(encoding="utf-8").rstrip()
    )
    record = read_record()
    assert record["mode"] == mode
    assert record["output_mode"] == METHODS[mode]
    assert (
        # 用实际发送给 Mock 的完整提示词重算哈希，核对日志是否准确。
        record["prompt_sha256"]
        == hashlib.sha256(messages[0][1].encode("utf-8")).hexdigest()
    )
    assert record["input_text"] == TEXT
    assert record["status"] == "draft_ready"
    assert record["extraction"] == payload
    assert record["usage"] is None
    assert record["returned_model"] is None
    assert record["finish_reason"] is None
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "draft_ready"
    assert output["extraction"] == payload
    assert output["checks"] == record["checks"]
    config.load_dotenv.assert_not_called()


# 从提示词中读出 Schema 和示例，验证它们与 Pydantic 定义一致且示例证据完整。
def test_prompts_share_generated_schema_and_complete_valid_example(mode: str) -> None:
    business = extract.PROMPT_PATH.read_text(encoding="utf-8")
    prompt = build_extraction_prompt(business, mode=mode)
    # split(..., 1) 只按第一个标记拆分；这里读的是应用生成的提示词，不是修复模型回答。
    schema_text = prompt.split("【JSON Schema】\n", 1)[1].split(
        "\n\n【完整格式示例】", 1
    )[0]
    assert json.loads(schema_text) == AfterSalesExtraction.model_json_schema()
    example_text = prompt.split("示例输入：", 1)[1].split("\n", 1)[0]
    example_json = prompt.split("示例 JSON 对象（工具模式下为调用参数）：\n", 1)[1]
    example = AfterSalesExtraction.model_validate_json(example_json)
    assert set(json.loads(example_json)) == set(AfterSalesExtraction.model_fields)
    assert example.evidence
    # 集合推导式收集证据字段名，确认五个业务字段都被覆盖。
    assert {item.field for item in example.evidence} == {
        "order_id",
        "product_name",
        "issue_description",
        "requested_action",
        "received_time_text",
    }
    assert all(item.quote in example_text for item in example.evidence)
    assert "不是本次用户事实" in prompt
    if mode == "structured":
        assert "只返回一次 AfterSalesExtraction 工具调用" in prompt
    else:
        assert "仅输出一个" in prompt
        assert "不要输出 Markdown" in prompt
        assert "只返回一次 AfterSalesExtraction 工具调用" not in prompt


# 比较三种提示词的共享部分，并验证不传 --mode 时默认仍为 structured。
def test_only_output_instruction_differs_between_prompts() -> None:
    # 字典推导式为每个模式构建提示词，并用模式名作为键保存。
    prompts = {
        mode: build_extraction_prompt("共同业务规则", mode=mode) for mode in METHODS
    }
    assert prompts["prompt"] == prompts["json"]
    assert (
        prompts["prompt"].split("【JSON Schema】", 1)[1]
        == prompts["structured"].split("【JSON Schema】", 1)[1]
    )
    assert extract.parse_args(["--text", TEXT]).mode == "structured"


# 同一组坏字段和坏证据跨三种模式测试，确认解析、证据与追问的判断顺序一致。
@pytest.mark.parametrize(
    "case, status, exit_code",
    [
        ("nulls", "needs_clarification", 0),
        ("insufficient", "needs_clarification", 0),
        ("missing-key", "validation_error", 3),
        ("enum", "validation_error", 3),
        ("extra", "validation_error", 3),
        ("empty-quote", "validation_error", 3),
        ("empty-evidence", "evidence_error", 4),
        ("blank-quote", "evidence_error", 4),
        ("absent-quote", "evidence_error", 4),
        ("missing-field-evidence", "evidence_error", 4),
    ],
)
def test_schema_evidence_and_clarification_share_the_same_flow(
    mode: str,
    case: str,
    status: str,
    exit_code: int,
    payload: dict,
    fake_model: tuple[Mock, Mock],
    capsys: pytest.CaptureFixture[str],
) -> None:
    if case == "nulls":
        for name in (
            "order_id",
            "product_name",
            "issue_description",
            "requested_action",
            "received_time_text",
        ):
            payload[name] = None
        payload["evidence"] = []
    elif case == "insufficient":
        payload["order_id"] = None
        # [1:] 是切片，复制从第二项到末尾的列表，用于删除订单号证据。
        payload["evidence"] = payload["evidence"][1:]
    elif case == "missing-key":
        # 删除键不同于值为 None：此 Schema 允许空值，但不允许省略必需键。
        payload.pop("order_id")
    elif case == "enum":
        payload["requested_action"] = "delete_order"
    elif case == "extra":
        payload["refund_approved"] = True
    elif case == "empty-quote":
        payload["evidence"][0]["quote"] = ""
    elif case == "empty-evidence":
        payload["evidence"] = []
    elif case == "blank-quote":
        payload["evidence"][0]["quote"] = " \n\t"
    elif case == "absent-quote":
        payload["evidence"][0]["quote"] = "原文没有这句话"
    else:
        payload["evidence"] = payload["evidence"][1:]
    operation = called_operation(fake_model[0], mode)
    # 把修改后的数据装进 Mock 响应，再交给真实应用流程处理。
    operation.return_value = fake_response(mode, payload)

    assert extract.main(["--text", TEXT, "--mode", mode]) == exit_code

    operation.assert_called_once()
    record = read_record()
    captured = capsys.readouterr()
    assert record["status"] == status
    assert record["mode"] == mode
    if status == "validation_error":
        assert record["extraction"] is None
        assert record["checks"] is None
        assert captured.out == ""
    else:
        assert record["extraction"] == payload
        assert json.loads(captured.out)["checks"] == record["checks"]
        if status == "needs_clarification":
            expected_missing = (
                ["order_id", "issue_description", "requested_action"]
                if case == "nulls"
                else ["order_id"]
            )
            assert record["checks"]["missing_fields"] == expected_missing
            assert len(record["checks"]["clarification_questions"]) == len(
                expected_missing
            )
        else:
            assert record["checks"]["evidence_errors"]
            assert record["checks"]["missing_fields"] is None


# 给正文加围栏、说明或截断内容，即使 Mock 提供有效 parsed，也不能掩盖原始格式错误。
@pytest.mark.parametrize("mode", ["prompt", "json"])
@pytest.mark.parametrize(
    "decoration",
    ["fenced", "prefix", "suffix", "truncated", "two", "empty", "reasoning-only"],
)
def test_text_modes_do_not_clean_up_or_repair_invalid_json(
    mode: str,
    decoration: str,
    payload: dict,
    fake_model: tuple[Mock, Mock],
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    # 用字典准备多种坏正文，最后的 [decoration] 选取当前用例。
    malformed = {
        "fenced": f"```json\n{text}\n```",
        "prefix": f"这是结果：{text}",
        "suffix": f"{text}\n这是结果。",
        "truncated": text[:-1],
        "two": f"{text}\n{text}",
        "empty": "",
        "reasoning-only": "",
    }[decoration]
    # 故意把合法 JSON 放进额外推理字段，确认它不会被当成最终回答。
    raw = AIMessage(content=malformed, additional_kwargs={"reasoning_content": text})
    if decoration == "reasoning-only":
        raw.content = [{"type": "reasoning", "reasoning": text}]
    operation = called_operation(fake_model[0], mode)
    # 模拟框架已经给出 parsed，仍须严格检查原始正文，不被宽松解析掩盖。
    operation.return_value = (
        raw
        if mode == "prompt"
        else {
            "raw": raw,
            "parsed": AfterSalesExtraction.model_validate(payload),
            "parsing_error": None,
        }
    )

    assert extract.main(["--text", TEXT, "--mode", mode]) == 3

    assert read_record()["status"] == "validation_error"
    assert read_record()["checks"] is None
    operation.assert_called_once()
    assert capsys.readouterr().out == ""


# 分别检查 parsing_error 非空与 parsed 为空；合法正文也不能绕过这两个守卫。
@pytest.mark.parametrize("mode", ["json", "structured"])
@pytest.mark.parametrize("failure", ["parsing-error", "parsed-none"])
def test_structured_wrappers_require_both_successful_parse_and_non_null_result(
    mode: str,
    failure: str,
    fake_model: tuple[Mock, Mock],
    capsys: pytest.CaptureFixture[str],
) -> None:
    operation = called_operation(fake_model[0], mode)
    response = operation.return_value
    if failure == "parsing-error":
        response["parsing_error"] = ValueError(f"{SECRET} other-private-credential")
    else:
        response["parsed"] = None

    assert extract.main(["--text", TEXT, "--mode", mode]) == 3

    record = read_record()
    assert record["status"] == "validation_error"
    assert record["checks"] is None
    operation.assert_called_once()
    captured = capsys.readouterr()
    assert captured.out == ""
    for secret in (SECRET, "other-private-credential"):
        assert secret not in captured.err + json.dumps(record)


# 模拟只有工具参数没有正文的响应：structured 可成功，两个文本模式应拒绝。
def test_empty_body_is_valid_only_for_structured_tool_results(
    mode: str,
    payload: dict,
    fake_model: tuple[Mock, Mock],
) -> None:
    operation = called_operation(fake_model[0], mode)
    response = fake_response("structured", payload)
    assert response["raw"].content == ""
    operation.return_value = response["raw"] if mode == "prompt" else response
    assert extract.main(["--text", TEXT, "--mode", mode]) == (
        0 if mode == "structured" else 3
    )
    assert read_record()["status"] == (
        "draft_ready" if mode == "structured" else "validation_error"
    )


# 模拟请求失败或接口不支持，检查不会改走另一种调用路径，也不会泄露异常全文。
@pytest.mark.parametrize("failure_type", [RuntimeError, NotImplementedError])
def test_request_failure_and_unsupported_interface_never_fall_back(
    mode: str,
    failure_type: type[Exception],
    fake_model: tuple[Mock, Mock],
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, _ = fake_model
    operation = called_operation(model, mode)
    # 不支持通常在绑定接口时报告；普通调用则在 invoke 时报告。
    # 按异常类型选择绑定阶段或 invoke 阶段，覆盖不同失败位置。
    failing = (
        model.with_structured_output
        if failure_type is NotImplementedError and mode != "prompt"
        else operation
    )
    failing.side_effect = failure_type(f"{SECRET} other-private-credential")

    assert extract.main(["--text", TEXT, "--mode", mode]) == 1

    failing.assert_called_once()
    if mode == "prompt":
        model.with_structured_output.assert_not_called()
    else:
        model.invoke.assert_not_called()
        assert operation.call_count == (
            0 if failing is model.with_structured_output else 1
        )
    record = read_record()
    assert record["status"] == "request_error"
    assert record["mode"] == mode
    assert record["error_type"] == failure_type.__name__
    assert record["usage"] is None
    assert record["checks"] is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert mode in captured.err
    assert "未自动降级或重试" in captured.err
    if failure_type is NotImplementedError:
        assert "不支持" in captured.err
    assert SECRET not in captured.err + json.dumps(record)
    assert "other-private-credential" not in captured.err + json.dumps(record)


# 结束原因 length 要如实记录；是否通过校验仍取决于内容，未知 Token 不用 SDK 补零值。
@pytest.mark.parametrize("valid", [True, False])
def test_length_finish_reason_and_unknown_usage_survive_validation(
    mode: str,
    valid: bool,
    fake_model: tuple[Mock, Mock],
) -> None:
    operation = called_operation(fake_model[0], mode)
    response = operation.return_value
    raw = response if mode == "prompt" else response["raw"]
    raw.response_metadata = {"finish_reason": "length", "token_usage": None}
    # 模拟 SDK 补出的零值；日志必须保留供应商用量未知这一事实。
    raw.usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if not valid:
        if mode == "prompt":
            raw.content = raw.content[:-1]
        else:
            response["parsing_error"] = ValueError("测试模拟输出截断")
            response["parsed"] = None

    assert extract.main(["--text", TEXT, "--mode", mode]) == (0 if valid else 3)

    record = read_record()
    assert record["finish_reason"] == "length"
    assert record["usage"] is None
    assert record["mode"] == mode
    assert record["status"] == ("draft_ready" if valid else "validation_error")
    operation.assert_called_once()


# 保留真实 LangChain json_mode 解析器，用假消息验证接口配置及本地严格校验。
@pytest.mark.parametrize(
    "case", ["valid", "nulls", "missing-key", "enum", "extra", "fenced", "truncated"]
)
def test_installed_json_mode_parser_with_mock_response(
    case: str,
    payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if case == "nulls":
        for name in (
            "order_id",
            "product_name",
            "issue_description",
            "requested_action",
            "received_time_text",
        ):
            payload[name] = None
        payload["evidence"] = []
    elif case == "missing-key":
        payload.pop("order_id")
    elif case == "enum":
        payload["requested_action"] = "delete_order"
    elif case == "extra":
        payload["unexpected"] = True
    text = json.dumps(payload)
    if case == "fenced":
        text = f"```json\n{text}\n```"
    elif case == "truncated":
        text = text[:-1]
    source = Mock(return_value=AIMessage(content=text))
    # RunnableLambda 提供固定消息，后面的解析仍由真实框架代码完成。
    bind = Mock(return_value=RunnableLambda(source))
    bind_tools = Mock(side_effect=AssertionError("json 模式不能使用工具调用"))
    beta = Mock(side_effect=AssertionError("不能启用 Beta"))
    monkeypatch.setattr(ChatDeepSeek, "bind", bind)
    monkeypatch.setattr(ChatDeepSeek, "bind_tools", bind_tools)
    monkeypatch.setattr(ChatDeepSeek, "_with_beta_api_base", beta)
    model = llm_client.create_model(model_name="test-json-mode", api_key=SECRET)

    # 响应来源已替换成本地 Mock，本调用不会访问模型服务。
    response = invoke_extraction(model, TEXT, "离线测试：输出 JSON", mode="json")

    if case in ("valid", "nulls"):
        assert parse_extraction_result(response, mode="json").model_dump() == payload
    else:
        with pytest.raises(ExtractionParseError):
            parse_extraction_result(response, mode="json")
    if case in ("fenced", "truncated"):
        # 当前框架容忍了格式问题，但应用不会把它计为原始 JSON 格式正确。
        assert response["parsing_error"] is None
        assert isinstance(response["parsed"], AfterSalesExtraction)
    source.assert_called_once()
    bind.assert_called_once()
    assert bind.call_args.kwargs["response_format"] == {"type": "json_object"}
    assert bind.call_args.kwargs["ls_structured_output_format"]["kwargs"] == {
        "method": "json_mode"
    }
    bind_tools.assert_not_called()
    beta.assert_not_called()


# 连续写三条不同模式的临时记录，确认模式映射正确、编号独立、旧行不被覆盖。
def test_log_keeps_modes_distinct_when_appending() -> None:
    path = None
    for mode in METHODS:
        record = new_extraction_record(max_tokens=1024, mode=mode)
        path = append_extraction_record(record)
    # 列表推导式逐行 json.loads，验证每行对应一条独立运行记录。
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [(record["mode"], record["output_mode"]) for record in records] == list(
        METHODS.items()
    )
    assert len({record["run_id"] for record in records}) == 3
    assert all(record["usage"] is None for record in records)


# 日志中的 CLI 模式和实际方法必须对应，错误值不能写入文件。
@pytest.mark.parametrize("invalid_mode", ["json_schema", "unknown"])
def test_unknown_or_mismatched_modes_are_rejected_before_saving(
    invalid_mode: str,
) -> None:
    record = new_extraction_record(max_tokens=1024)
    record["mode"] = invalid_mode
    with pytest.raises(ValueError):
        append_extraction_record(record)
    record["mode"] = "json"
    with pytest.raises(ValueError):
        append_extraction_record(record)
    assert not extraction_record.EXTRACTION_RUNS_PATH.exists()
