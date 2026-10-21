"""抽取功能的离线测试；所有模型响应均由测试 Mock 提供。"""
# 阅读提示：pytest 会发现 test_ 开头的函数；函数参数中的 fixture 由 pytest 自动注入。
# @pytest.mark.parametrize 把一张输入表展开为多次测试，不是运行真实批量模型实验。
# monkeypatch 临时替换变量/函数，测试结束后还原；tmp_path 是隔离的临时目录。
# capsys 捕获 print 的 stdout/stderr；assert 条件不成立时使测试失败。
# Mock.return_value 设置返回值；side_effect 可设置要抛出的异常；调用断言检查是否误重试。

import hashlib
import json
import runpy
from functools import partial
from pathlib import Path
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
from extractor import (
    ExtractionParseError,
    build_extraction_prompt,
    extract_after_sales,
)

TEST_API_KEY = "test-only-extraction-key"
TEXT = "订单 A10086 的耳机左边没声音，昨天收到，我想换货。"


# 每个测试自动替换配置入口，确保不会读取真实 .env；partial 固定 load_env=False。
@pytest.fixture(autouse=True)
def disable_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        extract, "load_config", partial(config.load_config, load_env=False)
    )


# fixture 构造合法的测试数据对象；测试参数同名时，pytest 会自动传入它。
@pytest.fixture
def extraction() -> AfterSalesExtraction:
    return AfterSalesExtraction(
        order_id="A10086",
        product_name="耳机",
        issue_description="左边没声音",
        requested_action="exchange",
        received_time_text="昨天收到",
        ambiguities=[],
        evidence=[
            {"field": "order_id", "quote": "A10086"},
            {"field": "product_name", "quote": "耳机"},
            {"field": "issue_description", "quote": "左边没声音"},
            {"field": "requested_action", "quote": "我想换货"},
            {"field": "received_time_text", "quote": "昨天收到"},
        ],
    )


# 构造供应商工具调用的 JSON 形状；这只是测试数据，不是调用外部工具。
def tool_call(arguments: str, *, name: str = "AfterSalesExtraction") -> dict:
    """模拟供应商返回的一个工具调用，不发送任何 HTTP 请求。"""
    return {
        "id": "test-call",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


# 用工具列表构造 AIMessage，模拟正文为空、答案放在工具参数中的响应。
def raw_message(calls: list[dict]) -> AIMessage:
    # additional_kwargs 放供应商形式的工具参数；AIMessage 将它们转换为框架可用的消息字段。
    return AIMessage(content="", additional_kwargs={"tool_calls": calls})


# 用 Mock 替换模型构造器，设置固定响应与虚构测试凭据，真实客户端不会被调用。
@pytest.fixture
def mock_model(
    monkeypatch: pytest.MonkeyPatch, extraction: AfterSalesExtraction
) -> tuple[Mock, Mock]:
    # 这里只设置虚构测试凭据；真实环境已经由 conftest.py 隔离。
    monkeypatch.setenv("DEEPSEEK_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("LLM_MODEL", "test-only-extraction-model")
    # spec 限定 Mock 的属性来自 ChatDeepSeek，写错接口名更容易被发现。
    model = Mock(spec=ChatDeepSeek)
    # 链式 return_value 先模拟包装器，再模拟包装器 invoke 返回的响应。
    model.with_structured_output.return_value.invoke.return_value = {
        "raw": raw_message([tool_call(extraction.model_dump_json())]),
        "parsed": extraction,
        "parsing_error": None,
    }
    constructor = Mock(return_value=model)
    monkeypatch.setattr(llm_client, "ChatDeepSeek", constructor)
    return model, constructor


# 分别测试直接文本和文件输入：检查原文、初始化参数、调用次数、JSON 结果及临时日志。
@pytest.mark.parametrize("use_file", [False, True])
def test_cli_success_preserves_input_and_existing_model_settings(
    use_file: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
    extraction: AfterSalesExtraction,
) -> None:
    # 切换到临时目录，验证默认提示词仍然按项目根目录定位。
    monkeypatch.chdir(tmp_path)
    text = f"  {TEXT}\n"
    argv = ["--text", text]
    if use_file:
        Path("description.txt").write_text(text, encoding="utf-8")
        argv = ["--text-file", "description.txt", "--max-tokens", "2048"]

    assert extract.main(argv) == 0

    model, constructor = mock_model
    # 不仅要求调用一次，还逐项核对传给客户端的实际关键字参数。
    constructor.assert_called_once_with(
        model="test-only-extraction-model",
        api_key=TEST_API_KEY,
        temperature=0.2,
        max_tokens=2048 if use_file else 1024,
        timeout=30,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    model.with_structured_output.assert_called_once_with(
        AfterSalesExtraction, method="function_calling", include_raw=True
    )
    model.with_structured_output.return_value.invoke.assert_called_once_with(
        [
            (
                "system",
                build_extraction_prompt(
                    extract.PROMPT_PATH.read_text(encoding="utf-8"), mode="structured"
                ),
            ),
            ("human", text),
        ]
    )
    model.invoke.assert_not_called()
    model.stream.assert_not_called()
    config.load_dotenv.assert_not_called()
    # 读取并清空捕获的输出；out 是业务 JSON，err 是运行提示。
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "draft_ready",
        "extraction": extraction.model_dump(),
        "checks": {
            "status": "draft_ready",
            "evidence_errors": [],
            "missing_fields": [],
            "clarification_questions": [],
        },
    }
    saved = json.loads(
        extraction_record.EXTRACTION_RUNS_PATH.read_text(encoding="utf-8")
    )
    assert saved["extraction"] == extraction.model_dump()
    assert saved["input_text"] == text
    assert saved["status"] == "draft_ready"
    assert captured.err == (
        f"run_id: {saved['run_id']}\n"
        f"记录已保存: {extraction_record.EXTRACTION_RUNS_PATH}\n"
    )
    assert TEST_API_KEY not in captured.out


# 参数表中的每种非法用法都应先被 argparse 拒绝，配置加载和模型构造都不能发生。
@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--text", TEXT, "--text-file", "unused.txt"],
        ["--text", TEXT, "--max-tokens", "0"],
        ["--text", TEXT, "--max-tokens", "-1"],
        ["--text", TEXT, "--max-tokens", "1.5"],
        ["--text", TEXT, "--stream"],
        ["--text", TEXT, "--mode", "json_schema"],
    ],
)
def test_cli_argument_errors_precede_configuration_and_model(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    mock_model: tuple[Mock, Mock],
) -> None:
    loader = Mock()
    monkeypatch.setattr(extract, "load_config", loader)
    # 预期 argparse 抛 SystemExit；exc_info.value 可取得异常对象和退出码。
    with pytest.raises(SystemExit) as exc_info:
        extract.main(argv)
    assert exc_info.value.code == 2
    loader.assert_not_called()
    mock_model[1].assert_not_called()


# 创建空白、缺失、乱码或目录输入，确认本地输入检查先于配置和模型调用。
@pytest.mark.parametrize(
    "input_kind",
    ["empty-text", "blank-text", "blank-file", "missing", "utf8", "directory"],
)
def test_invalid_input_precedes_configuration_and_model(
    input_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    path = tmp_path / "input.txt"
    if input_kind == "blank-file":
        path.write_text(" \n\t", encoding="utf-8")
    elif input_kind == "utf8":
        # 写非法 UTF-8 字节稳定模拟解码失败，不依赖系统默认编码。
        path.write_bytes(b"\xff\xfe")
    elif input_kind == "directory":
        path.mkdir()
    argv = ["--text-file", str(path)]
    if input_kind in ("empty-text", "blank-text"):
        argv = ["--text", "" if input_kind == "empty-text" else " \n\t"]
    loader = Mock()
    monkeypatch.setattr(extract, "load_config", loader)

    assert extract.main(argv) == 2

    loader.assert_not_called()
    mock_model[1].assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "输入或配置错误" in captured.err


# 模拟权限异常，检查错误提示可读，同时不泄露异常文本中放入的测试密钥。
def test_unreadable_file_has_safe_error_before_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    loader = Mock()
    monkeypatch.setattr(extract, "load_config", loader)
    # 显式模拟权限错误，不依赖当前用户是否有能力绕过 chmod 权限。
    monkeypatch.setattr(
        Path, "read_text", Mock(side_effect=PermissionError(TEST_API_KEY))
    )
    assert extract.main(["--text-file", str(tmp_path / "denied.txt")]) == 2
    loader.assert_not_called()
    mock_model[1].assert_not_called()
    captured = capsys.readouterr()
    assert "无法读取文件" in captured.err
    assert captured.out == ""
    assert TEST_API_KEY not in captured.err


# 提示词缺失或只有空白时必须停止，不应带着无效规则请求模型。
@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_empty_prompt_precedes_model(
    exists: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_model: tuple[Mock, Mock],
) -> None:
    prompt = tmp_path / "extract.txt"
    if exists:
        prompt.write_text(" \n", encoding="utf-8")
    monkeypatch.setattr(extract, "PROMPT_PATH", prompt)
    loader = Mock()
    monkeypatch.setattr(extract, "load_config", loader)

    assert extract.main(["--text", TEXT]) == 2

    loader.assert_not_called()
    mock_model[1].assert_not_called()


# 删除一个必要配置项，检查明确报错且模型构造器从未被调用。
@pytest.mark.parametrize("missing", ["DEEPSEEK_API_KEY", "LLM_MODEL"])
def test_missing_configuration_precedes_model(
    missing: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    # 只删除测试环境映射里的配置项，不修改磁盘上的 .env。
    monkeypatch.delenv(missing)
    assert extract.main(["--text", TEXT]) == 2
    mock_model[1].assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert missing in captured.err
    assert TEST_API_KEY not in captured.err


# 在初始化、绑定或请求阶段注入异常，确认每步至多尝试一次并仅输出安全错误。
@pytest.mark.parametrize("stage", ["initialization", "binding", "invoke"])
def test_request_error_is_distinct_and_never_retried(
    stage: str,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    model, constructor = mock_model
    # 建立阶段到 Mock 的映射，再按 stage 选出本次制造异常的位置。
    operation = {
        "initialization": constructor,
        "binding": model.with_structured_output,
        "invoke": model.with_structured_output.return_value.invoke,
    }[stage]
    # 设置 side_effect 后，调用这个 Mock 会抛异常而不是返回默认值。
    operation.side_effect = RuntimeError(TEST_API_KEY)

    assert extract.main(["--text", TEXT]) == 1

    operation.assert_called_once()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "请求错误" in captured.err
    assert "RuntimeError" in captured.err
    assert TEST_API_KEY not in captured.err


# 逐项构造畸形响应，覆盖空结果、多工具、错误工具等；不能把它们当作成功或重试。
@pytest.mark.parametrize(
    "case",
    [
        "not-dict",
        "missing-status",
        "missing-raw",
        "missing-parsed",
        "parsed-none",
        "parsed-dict",
        "parsing-error",
        "no-tools",
        "multiple-tools",
        "valid-and-invalid",
        "invalid-only",
        "wrong-tool",
        "original-extra-tool",
    ],
)
def test_invalid_structured_results_fail_without_json_or_retry(
    case: str,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
    extraction: AfterSalesExtraction,
) -> None:
    model, _ = mock_model
    invoke = model.with_structured_output.return_value.invoke
    result = invoke.return_value
    valid_call = tool_call(extraction.model_dump_json())
    if case == "not-dict":
        result = extraction
    elif case.startswith("missing-"):
        # case[8:] 去掉 missing- 前缀；pop 删除对应键，模拟响应缺字段。
        result.pop({"status": "parsing_error"}.get(case[8:], case[8:]))
    elif case == "parsed-none":
        result["parsed"] = None
    elif case == "parsed-dict":
        result["parsed"] = extraction.model_dump()
    elif case == "parsing-error":
        result["parsing_error"] = ValueError(TEST_API_KEY)
    elif case == "no-tools":
        result["raw"] = AIMessage(content="这是普通文本，不是结构化结果。")
    elif case == "multiple-tools":
        result["raw"] = raw_message([valid_call, {**valid_call, "id": "test-call-2"}])
    elif case == "valid-and-invalid":
        result["raw"] = raw_message([valid_call, tool_call("{")])
    elif case == "invalid-only":
        result["raw"] = raw_message([tool_call("{")])
    elif case == "wrong-tool":
        result["raw"] = raw_message([tool_call("{}", name="UnexpectedTool")])
    elif case == "original-extra-tool":
        # 原始列表含未标准化为 tool_calls / invalid_tool_calls 的异常条目。
        result["raw"] = raw_message([valid_call, {"type": "unexpected"}])
    invoke.return_value = result

    assert extract.main(["--text", TEXT]) == 3

    model.with_structured_output.assert_called_once_with(
        AfterSalesExtraction, method="function_calling", include_raw=True
    )
    invoke.assert_called_once()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "解析校验错误" in captured.err
    assert "请求错误" not in captured.err
    assert TEST_API_KEY not in captured.err


# 保留已安装框架的真实解析器，只把模型响应换成 Mock，检查 Schema 与接口兼容性。
@pytest.mark.parametrize(
    "case", ["valid", "nulls", "enum", "extra", "missing-fields", "json", "none", "two"]
)
def test_installed_framework_parser_with_mock_model_response(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
    extraction: AfterSalesExtraction,
) -> None:
    payload = extraction.model_dump()
    if case == "nulls":
        # fromkeys 给字段统一填 None；字典的 | 运算把右侧键值合并进来。
        payload = dict.fromkeys(
            [
                "order_id",
                "product_name",
                "issue_description",
                "requested_action",
                "received_time_text",
            ]
        ) | {"ambiguities": [], "evidence": []}
    elif case == "enum":
        payload["requested_action"] = "delete_order"
    elif case == "extra":
        payload["refund_approved"] = True
    elif case == "missing-fields":
        payload["missing_fields"] = []
    calls = [tool_call("{" if case == "json" else json.dumps(payload))]
    if case == "none":
        calls = []
    elif case == "two":
        calls.append({**calls[0], "id": "test-call-2"})
    response = raw_message(calls)
    response_source = Mock(return_value=response)
    # RunnableLambda 把普通可调用对象包装成 Runnable，供真实解析链使用。
    binding = Mock(return_value=RunnableLambda(response_source))
    monkeypatch.setattr(ChatDeepSeek, "bind_tools", binding)
    beta_switch = Mock(side_effect=AssertionError("不能启用 Beta"))
    monkeypatch.setattr(ChatDeepSeek, "_with_beta_api_base", beta_switch)
    # 初始化客户端不会请求模型；bind_tools 替换为离线响应 Runnable，
    # 保留真实 with_structured_output 和 Pydantic 解析器以核对当前依赖接口。
    model = llm_client.create_model(
        model_name="test-only-extraction-model", api_key=TEST_API_KEY, max_tokens=1024
    )
    if case in ("valid", "nulls"):
        result = extract_after_sales(model, TEXT, "仅用于离线测试的提示词")
        assert isinstance(result, AfterSalesExtraction)
        assert result.model_dump() == payload
    else:
        with pytest.raises(ExtractionParseError):
            extract_after_sales(model, TEXT, "仅用于离线测试的提示词")
    response_source.assert_called_once()
    binding.assert_called_once()
    assert binding.call_args.args == ([AfterSalesExtraction],)
    assert binding.call_args.kwargs["strict"] is None
    beta_switch.assert_not_called()


# 把字符串订单号替换为整数，确认严格 Schema 会拒绝错误类型。
def test_pydantic_strictness_does_not_coerce_invalid_field_types(
    extraction: AfterSalesExtraction,
) -> None:
    payload = extraction.model_dump() | {"order_id": 10086}
    with pytest.raises(ValidationError):
        AfterSalesExtraction.model_validate(payload)


# runpy 以非入口名称加载模块，确认 import 不会加载配置或构造模型。
def test_imports_do_not_initialize_model_or_load_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = Mock(side_effect=AssertionError("导入时不能加载配置"))
    constructor = Mock(side_effect=AssertionError("导入时不能初始化模型"))
    monkeypatch.setattr(config, "load_config", loader)
    monkeypatch.setattr(llm_client, "create_model", constructor)
    for filename in (
        "extractor.py",
        "extraction_rules.py",
        "extraction_record.py",
        "extract.py",
    ):
        # 用自定义模块名执行定义，不会满足 __name__ == "__main__" 的入口条件。
        runpy.run_path(
            str(config.PROJECT_ROOT / filename), run_name="offline_import_test"
        )
    loader.assert_not_called()
    constructor.assert_not_called()


# 读取测试临时日志并用 json.loads 转回字典，方便后面的断言检查字段。
def saved_record() -> dict:
    return json.loads(
        extraction_record.EXTRACTION_RUNS_PATH.read_text(encoding="utf-8")
    )


# 修改一个业务条件，核对状态、退出码、固定追问及日志；模型仍只能调用一次。
@pytest.mark.parametrize(
    "case, expected_status, expected_exit",
    [
        ("missing", "needs_clarification", 0),
        ("ambiguities", "needs_clarification", 0),
        ("other", "needs_clarification", 0),
        ("evidence", "evidence_error", 4),
    ],
)
def test_business_status_and_log_after_exactly_one_call(
    case: str,
    expected_status: str,
    expected_exit: int,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
    extraction: AfterSalesExtraction,
) -> None:
    model, _ = mock_model
    if case == "missing":
        extraction.order_id = None
        extraction.evidence = [
            item for item in extraction.evidence if item.field != "order_id"
        ]
    elif case == "ambiguities":
        extraction.ambiguities = ["请确认是换货还是维修。"]
    elif case == "other":
        extraction.requested_action = "other"
    else:
        extraction.evidence = []
    response = model.with_structured_output.return_value.invoke.return_value
    response["raw"] = raw_message([tool_call(extraction.model_dump_json())])

    assert extract.main(["--text", TEXT]) == expected_exit

    model.with_structured_output.return_value.invoke.assert_called_once()
    model.with_structured_output.assert_called_once()
    record = saved_record()
    output = json.loads(capsys.readouterr().out)
    assert record["status"] == output["status"] == expected_status
    assert record["checks"] == output["checks"]
    assert record["extraction"] == output["extraction"] == extraction.model_dump()
    assert record["checks"]["status"] == expected_status
    if expected_status == "needs_clarification":
        assert record["checks"]["clarification_questions"]
        assert record["error_type"] is None
        assert record["error_message"] is None
    else:
        assert record["checks"]["missing_fields"] is None
        assert record["checks"]["clarification_questions"] == []
        assert record["error_type"] == "EvidenceCheckError"


# 用可控时钟和模拟元数据验证耗时、哈希、实际用量及脱敏，不使用真实 sleep。
@pytest.mark.parametrize("failure", [False, True])
def test_actual_metadata_versions_parameters_and_controlled_total_latency(
    failure: bool,
    monkeypatch: pytest.MonkeyPatch,
    mock_model: tuple[Mock, Mock],
) -> None:
    # iter 创建迭代器，next 会按顺序取出这两个预设时间点。
    times = iter([10.0, 10.125])
    # lambda 是匿名函数；用 next(times) 代替真实时钟，不依赖速度或 sleep。
    monkeypatch.setattr(extract, "perf_counter", lambda: next(times))
    model, _ = mock_model
    response = model.with_structured_output.return_value.invoke.return_value
    response["raw"].response_metadata = {
        "model_name": "provider-actual-model-version",
        "finish_reason": "length" if failure else "tool_calls",
        "token_usage": {"prompt_tokens": 17, "completion_tokens": 0},
        "headers": {"Authorization": "other-private-credential"},
    }
    # 这些是 SDK 补出来的值，不能把总量 17 当作供应商实际返回的计数。
    response["raw"].usage_metadata = {
        "input_tokens": 17,
        "output_tokens": 0,
        "total_tokens": 17,
    }
    response["raw"].additional_kwargs["reasoning_content"] = "private-reasoning"
    if failure:
        response["parsed"] = None
        response["parsing_error"] = ValueError(
            f"{TEST_API_KEY} other-private-credential"
        )

    assert extract.main(["--text", TEXT]) == (3 if failure else 0)

    record = saved_record()
    assert record["task_type"] == "after_sales_extraction"
    assert record["schema_version"] == "after-sales-v1"
    assert (
        record["prompt_sha256"]
        == hashlib.sha256(
            model.with_structured_output.return_value.invoke.call_args.args[0][0][
                1
            ].encode("utf-8")
        ).hexdigest()
    )
    assert record["requested_model"] == "test-only-extraction-model"
    assert record["returned_model"] == "provider-actual-model-version"
    assert record["output_mode"] == "function_calling"
    assert record["parameters"] == {
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": False,
        "thinking": {"type": "disabled"},
        "timeout": 30,
        "max_retries": 0,
    }
    # 两个时间点相差 0.125 秒，换算后应该严格等于 125 毫秒。
    assert record["latency_ms"] == 125.0
    assert record["usage"] == {
        "input_tokens": 17,
        "output_tokens": 0,
        "total_tokens": None,
    }
    assert record["finish_reason"] == ("length" if failure else "tool_calls")
    if failure:
        assert record["status"] == "validation_error"
        assert record["extraction"] is None
        assert record["checks"] is None
    serialized = json.dumps(record, ensure_ascii=False)
    for disallowed in (
        "other-private-credential",
        "private-reasoning",
        TEST_API_KEY,
        "Authorization",
    ):
        assert disallowed not in serialized


# 供应商信息不足时保留 JSON null，不用请求模型名或推算计数补齐未知值。
def test_unknown_usage_and_unproven_returned_model_remain_null(
    mock_model: tuple[Mock, Mock],
) -> None:
    response = mock_model[0].with_structured_output.return_value.invoke.return_value
    response["raw"].response_metadata = {"model_name": "test-only-extraction-model"}
    assert extract.main(["--text", TEXT]) == 0
    record = saved_record()
    assert record["usage"] is None
    assert record["returned_model"] is None
    assert record["finish_reason"] is None


# 分别模拟请求异常和 Ctrl+C，确认仍尽量留下安全记录，且不会输出假结果。
@pytest.mark.parametrize(
    "failure, status, exit_code",
    [
        (RuntimeError("other-credential"), "request_error", 1),
        (KeyboardInterrupt(), "cancelled", 130),
    ],
)
def test_request_failure_and_cancel_save_safe_records(
    failure: BaseException,
    status: str,
    exit_code: int,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    invoke = mock_model[0].with_structured_output.return_value.invoke
    invoke.side_effect = failure

    assert extract.main(["--text", TEXT]) == exit_code

    invoke.assert_called_once()
    record = saved_record()
    assert record["status"] == status
    assert record["error_type"] == type(failure).__name__
    assert record["error_message"]
    assert record["extraction"] is None
    assert record["checks"] is None
    assert record["usage"] is None
    assert record["returned_model"] is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert record["run_id"] in captured.err
    assert "other-credential" not in captured.err + json.dumps(record)


# 让 parsed 为空，再把业务检查器设成一调用就失败的 Mock，验证流程顺序。
def test_schema_failure_does_not_reach_business_checks(
    monkeypatch: pytest.MonkeyPatch,
    mock_model: tuple[Mock, Mock],
) -> None:
    response = mock_model[0].with_structured_output.return_value.invoke.return_value
    response["parsed"] = None
    # 如果错误地进入业务检查，这个 Mock 会抛 AssertionError 让测试失败。
    checker = Mock(side_effect=AssertionError("Schema 未通过时不能检查业务"))
    monkeypatch.setattr(extract, "check_extraction", checker)

    assert extract.main(["--text", TEXT]) == 3

    checker.assert_not_called()
    assert saved_record()["status"] == "validation_error"


# 模拟磁盘写入失败，核对非零退出码和失败提示，不能误报“记录已保存”。
def test_log_write_failure_is_explicit_without_false_saved_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    monkeypatch.setattr(
        extract,
        "append_extraction_record",
        Mock(side_effect=OSError(f"{TEST_API_KEY} other-credential")),
    )
    assert extract.main(["--text", TEXT]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "draft_ready"
    assert "run_id:" in captured.err
    assert "记录保存失败" in captured.err
    assert "记录已保存" not in captured.err
    assert TEST_API_KEY not in captured.err
    assert "other-credential" not in captured.err
    assert not extraction_record.EXTRACTION_RUNS_PATH.exists()


# 输入错误也应留下本地记录，但不会构造或调用模型。
def test_input_error_also_saves_a_record_without_model(
    mock_model: tuple[Mock, Mock],
) -> None:
    assert extract.main(["--text", " \t"]) == 2
    mock_model[1].assert_not_called()
    record = saved_record()
    assert record["status"] == "input_error"
    assert record["input_text"] == " \t"
    assert record["checks"] is None
    assert record["error_type"] == "ValueError"
