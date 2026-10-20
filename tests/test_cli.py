"""离线测试：配置使用临时文件，模型回答全部来自明确标记的 Mock。"""

import json
import runpy
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import dotenv
import langchain_deepseek
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

import config
import llm_client
import main
import record_store

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_API_KEY = "test-only-api-key"


@pytest.fixture
def mock_model(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    monkeypatch.setenv("DEEPSEEK_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("LLM_MODEL", "test-only-model")
    model = Mock()
    model.invoke.return_value = AIMessage(content="【离线测试模拟回答】")
    constructor = Mock(return_value=model)
    monkeypatch.setattr(llm_client, "ChatDeepSeek", constructor)
    return model, constructor


def test_default_invocation_from_another_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    monkeypatch.chdir(tmp_path)
    model, constructor = mock_model

    assert main.main(["--question", "  什么是大语言模型？\n"]) == 0

    messages = model.invoke.call_args.args[0]
    assert messages == [
        ("system", (PROJECT_ROOT / "prompts" / "base.txt").read_text(encoding="utf-8")),
        ("human", "什么是大语言模型？"),
    ]
    model.invoke.assert_called_once()
    model.stream.assert_not_called()
    constructor.assert_called_once_with(
        model="test-only-model",
        api_key=TEST_API_KEY,
        temperature=0.2,
        max_tokens=512,
        timeout=30,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    captured = capsys.readouterr()
    assert captured.out == "【离线测试模拟回答】\n"
    record = json.loads(record_store.RUNS_PATH.read_text(encoding="utf-8"))
    assert captured.err == (
        f"run_id: {record['run_id']}\n记录已保存: {record_store.RUNS_PATH}\n"
    )


def test_custom_files_and_parameters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mock_model: tuple[Mock, Mock],
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("system.txt").write_text("仅依据给定资料回答。", encoding="utf-8")
    Path("context.txt").write_text("第一行资料。\n第二行资料。", encoding="utf-8")
    model, constructor = mock_model

    assert (
        main.main(
            [
                "--question",
                "总结资料",
                "--system-file",
                "system.txt",
                "--context-file",
                "context.txt",
                "--temperature",
                "0.7",
                "--max-tokens",
                "128",
            ]
        )
        == 0
    )

    model.invoke.assert_called_once_with(
        [
            ("system", "仅依据给定资料回答。"),
            ("human", "【参考资料】\n第一行资料。\n第二行资料。\n\n【问题】\n总结资料"),
        ]
    )
    assert constructor.call_args.kwargs["temperature"] == 0.7
    assert constructor.call_args.kwargs["max_tokens"] == 128


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--question", " \n\t"],
        ["--question", "问题", "--temperature", "not-a-number"],
        ["--question", "问题", "--temperature", "-0.1"],
        ["--question", "问题", "--temperature", "2.1"],
        ["--question", "问题", "--temperature", "nan"],
        ["--question", "问题", "--temperature", "inf"],
        ["--question", "问题", "--max-tokens", "0"],
        ["--question", "问题", "--max-tokens", "-1"],
        ["--question", "问题", "--max-tokens", "1.5"],
        ["--question", "问题", "--max-tokens", "393217"],
    ],
)
def test_invalid_arguments_stop_before_config_or_model(
    argv: list[str], mock_model: tuple[Mock, Mock]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main.main(argv)
    assert exc_info.value.code == 2
    config.load_dotenv.assert_not_called()
    mock_model[1].assert_not_called()


@pytest.mark.parametrize("temperature,max_tokens", [("0", "1"), ("2", "393216")])
def test_numeric_boundaries(temperature: str, max_tokens: str) -> None:
    args = main.parse_args(
        ["--question", "问题", "--temperature", temperature, "--max-tokens", max_tokens]
    )
    assert args.temperature == float(temperature)
    assert args.max_tokens == int(max_tokens)


@pytest.mark.parametrize("option", ["--system-file", "--context-file"])
@pytest.mark.parametrize("file_kind", ["missing", "invalid-utf8", "directory"])
def test_invalid_files_stop_before_config_or_model(
    option: str,
    file_kind: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    path = tmp_path / "input.txt"
    if file_kind == "invalid-utf8":
        path.write_bytes(b"\xff\xfe")
    elif file_kind == "directory":
        path.mkdir()

    assert main.main(["--question", "问题", option, str(path)]) == 2
    config.load_dotenv.assert_not_called()
    mock_model[1].assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "输入或配置错误" in captured.err


@pytest.mark.parametrize("missing_key", ["DEEPSEEK_API_KEY", "LLM_MODEL"])
@pytest.mark.parametrize("missing_value", [None, "", " \t"])
def test_missing_configuration_stops_before_model(
    missing_key: str,
    missing_value: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    if missing_value is None:
        monkeypatch.delenv(missing_key)
    else:
        monkeypatch.setenv(missing_key, missing_value)

    assert main.main(["--question", "问题"]) == 2

    mock_model[1].assert_not_called()
    captured = capsys.readouterr()
    assert missing_key in captured.err
    assert TEST_API_KEY not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("use_environment_override", [False, True])
def test_config_uses_project_root_and_preserves_environment(
    use_environment_override: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text(
        f"DEEPSEEK_API_KEY={TEST_API_KEY}\nLLM_MODEL=test-file-model\n",
        encoding="utf-8",
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text(
        "LLM_MODEL=wrong-directory-model\n", encoding="utf-8"
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", project)
    # 仅此测试启用真实 dotenv 解析器；此时路径已指向测试临时目录。
    monkeypatch.setattr(config, "load_dotenv", dotenv.load_dotenv)
    monkeypatch.chdir(elsewhere)
    if use_environment_override:
        monkeypatch.setenv("LLM_MODEL", "  test-environment-model  ")

    expected_model = (
        "test-environment-model" if use_environment_override else "test-file-model"
    )
    assert config.load_config() == (expected_model, TEST_API_KEY)


def test_imports_have_no_config_or_model_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked_loader = Mock(side_effect=AssertionError("导入模块不应读取 .env"))
    blocked_constructor = Mock(side_effect=AssertionError("导入模块不应初始化模型"))
    monkeypatch.setattr(dotenv, "load_dotenv", blocked_loader)
    monkeypatch.setattr(langchain_deepseek, "ChatDeepSeek", blocked_constructor)
    for filename in ("config.py", "llm_client.py", "record_store.py", "main.py"):
        runpy.run_path(str(PROJECT_ROOT / filename), run_name="offline_import_test")
    blocked_loader.assert_not_called()
    blocked_constructor.assert_not_called()


@pytest.mark.parametrize("stage", ["initialization", "invoke"])
def test_model_failure_does_not_leak_key_or_return_fake_answer(
    stage: str, capsys: pytest.CaptureFixture[str], mock_model: tuple[Mock, Mock]
) -> None:
    model, constructor = mock_model
    error = RuntimeError(f"sensitive error containing {TEST_API_KEY}")
    if stage == "initialization":
        constructor.side_effect = error
    else:
        model.invoke.side_effect = error

    assert main.main(["--question", "问题"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RuntimeError" in captured.err
    assert TEST_API_KEY not in captured.err
    if stage == "initialization":
        model.invoke.assert_not_called()


def test_stream_flushes_text_before_requesting_next_chunk(
    monkeypatch: pytest.MonkeyPatch, mock_model: tuple[Mock, Mock]
) -> None:
    model, _ = mock_model
    clock = Mock(side_effect=[10.0, 10.25, 11.0])
    monkeypatch.setattr(main, "perf_counter", clock)
    output = Mock()
    monkeypatch.setattr(main.sys, "stdout", output)
    stats = Mock()
    monkeypatch.setattr(main.sys, "stderr", stats)

    def chunks() -> Iterator[AIMessageChunk]:
        yield AIMessageChunk(content="")
        yield AIMessageChunk(
            content="", additional_kwargs={"reasoning_content": "【测试推理内容】"}
        )
        clock.assert_called_once()  # 空文本和额外推理不能触发首文本计时。
        output.write.assert_not_called()
        yield AIMessageChunk(content="【离线测试】大语言")
        assert output.write.call_args_list[0].args == ("【离线测试】大语言",)
        output.flush.assert_called_once()  # 下一片段产生前必须已经刷新。
        yield AIMessageChunk(
            content="模型。", response_metadata={"finish_reason": "stop"}
        )
        yield AIMessageChunk(
            content="",
            usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        )
        # 模拟 LangChain 的结尾空片段，确保统计来自聚合结果而非最后一个片段。
        yield AIMessageChunk(content="", chunk_position="last")

    model.stream.return_value = chunks()

    assert main.main(["--question", "问题", "--stream"]) == 0

    model.invoke.assert_not_called()
    model.stream.assert_called_once_with(
        [
            (
                "system",
                (PROJECT_ROOT / "prompts" / "base.txt").read_text(encoding="utf-8"),
            ),
            ("human", "问题"),
        ],
        stream_usage=True,
    )
    answer = "".join(call.args[0] for call in output.write.call_args_list)
    summary = "".join(call.args[0] for call in stats.write.call_args_list)
    assert answer == "【离线测试】大语言模型。\n"
    assert output.flush.call_count == 3  # 两段回答和结束换行各刷新一次。
    assert "latency_ms: 1000.00\n" in summary
    assert "first_text_latency_ms: 250.0\n" in summary
    assert (
        "usage_metadata: {'input_tokens': 10, 'output_tokens': 4, 'total_tokens': 14}"
        in summary
    )
    assert "finish_reason: stop" in summary
    record = json.loads(record_store.RUNS_PATH.read_text(encoding="utf-8"))
    assert record["latency_ms"] == pytest.approx(1000.0)
    assert record["first_text_latency_ms"] == pytest.approx(250.0)


def test_stream_content_blocks_exclude_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    model, _ = mock_model
    clock = Mock(side_effect=[10.0, 10.5, 11.0])
    monkeypatch.setattr(main, "perf_counter", clock)

    def chunks() -> Iterator[AIMessageChunk]:
        yield AIMessageChunk(
            content=[{"type": "reasoning", "reasoning": "【测试推理内容】"}]
        )
        clock.assert_called_once()
        yield AIMessageChunk(
            content=[
                {"type": "text", "text": "【离线测试】回答"},
                {"type": "reasoning", "reasoning": "【更多测试推理】"},
                {"type": "text", "text": "内容。"},
            ]
        )

    model.stream.return_value = chunks()

    assert main.main(["--question", "问题", "--stream"]) == 0

    captured = capsys.readouterr()
    assert captured.out == "【离线测试】回答内容。\n"
    assert "first_text_latency_ms: 500.0" in captured.err
    assert "usage_metadata: None" in captured.err
    assert "finish_reason: None" in captured.err


@pytest.mark.parametrize(
    "chunks",
    [
        [],
        [AIMessageChunk(content="")],
        [AIMessageChunk(content=" \n\t")],
        [
            AIMessageChunk(
                content="", additional_kwargs={"reasoning_content": "【测试推理内容】"}
            )
        ],
        [
            AIMessageChunk(
                content="",
                response_metadata={"finish_reason": "stop"},
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 0,
                    "total_tokens": 10,
                },
            )
        ],
    ],
    ids=["no-chunks", "empty-text", "whitespace", "reasoning-only", "metadata-only"],
)
def test_stream_without_visible_text_has_no_first_text_latency(
    chunks: list[AIMessageChunk],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    model, _ = mock_model
    model.stream.return_value = iter(chunks)
    clock = Mock(side_effect=[10.0, 10.5])
    monkeypatch.setattr(main, "perf_counter", clock)

    assert main.main(["--question", "问题", "--stream"]) == 0

    captured = capsys.readouterr()
    assert "first_text_latency_ms: None" in captured.err
    assert "latency_ms: 500.00" in captured.err
    assert clock.call_count == 2
    assert "测试推理内容" not in captured.out
    if not chunks or chunks[0].usage_metadata is None:
        assert "usage_metadata: None" in captured.err
    model.invoke.assert_not_called()


@pytest.mark.parametrize("has_partial_text", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_stream_failure_preserves_partial_text_and_reports_failure(
    has_partial_text: bool,
    error_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    model, _ = mock_model
    ticks = [10.0, 10.25, 10.75] if has_partial_text else [10.0, 10.75]
    monkeypatch.setattr(main, "perf_counter", Mock(side_effect=ticks))

    def failing_chunks() -> Iterator[AIMessageChunk]:
        if has_partial_text:
            yield AIMessageChunk(content="【离线测试】已收到的部分回答")
        raise error_type(f"sensitive error containing {TEST_API_KEY}")

    model.stream.return_value = failing_chunks()
    expected_code = 130 if error_type is KeyboardInterrupt else 1

    assert main.main(["--question", "问题", "--stream"]) == expected_code

    captured = capsys.readouterr()
    expected_text = "【离线测试】已收到的部分回答\n" if has_partial_text else ""
    assert captured.out == expected_text
    assert "流式调用失败" in captured.err
    assert error_type.__name__ in captured.err
    assert "正常完成" not in captured.err
    assert "latency_ms: 750.00" in captured.err
    expected_first_text = "250.0" if has_partial_text else "None"
    assert f"first_text_latency_ms: {expected_first_text}" in captured.err
    assert "usage_metadata: None" in captured.err
    assert "finish_reason: None" in captured.err
    assert TEST_API_KEY not in captured.out + captured.err
    model.invoke.assert_not_called()
    record = json.loads(record_store.RUNS_PATH.read_text(encoding="utf-8"))
    assert record["output_text"] == expected_text.rstrip("\n")
    assert record["first_text_latency_ms"] == (
        pytest.approx(250.0) if has_partial_text else None
    )


def test_stream_preserves_partial_metadata_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    model, _ = mock_model
    monkeypatch.setattr(main, "perf_counter", Mock(side_effect=[10.0, 10.25, 11.0]))

    def failing_chunks() -> Iterator[AIMessageChunk]:
        yield AIMessageChunk(content="【离线测试】部分回答")
        yield AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        )
        raise RuntimeError("测试尾部连接失败")

    model.stream.return_value = failing_chunks()

    assert main.main(["--question", "问题", "--stream"]) == 1

    captured = capsys.readouterr()
    assert "流式调用失败" in captured.err
    assert "'total_tokens': 14" in captured.err
    assert "finish_reason: stop" in captured.err
    assert "正常完成" not in captured.err


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status", ["completed", "error", "cancelled"])
def test_each_request_outcome_appends_one_record(
    streaming: bool,
    status: str,
    mock_model: tuple[Mock, Mock],
    capsys: pytest.CaptureFixture[str],
) -> None:
    model, _ = mock_model
    usage = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    metadata = {"model_name": "test-returned-model", "finish_reason": "stop"}

    def chunks() -> Iterator[AIMessageChunk]:
        yield AIMessageChunk(content="【离线测试】第一行\n")
        if status == "error":
            raise RuntimeError(f"Authorization: Bearer {TEST_API_KEY}; password=secret")
        if status == "cancelled":
            raise KeyboardInterrupt()
        yield AIMessageChunk(
            content="第二行", response_metadata=metadata, usage_metadata=usage
        )

    if streaming:
        model.stream.return_value = chunks()
    elif status == "completed":
        model.invoke.return_value = AIMessage(
            content="【离线测试】第一行\n第二行",
            response_metadata=metadata,
            usage_metadata=usage,
        )
    else:
        error_type = RuntimeError if status == "error" else KeyboardInterrupt
        model.invoke.side_effect = error_type(
            f"Authorization: Bearer {TEST_API_KEY}; password=secret"
        )

    argv = ["--question", "  测试问题  ", "--temperature", "0.7", "--max-tokens", "128"]
    if streaming:
        argv.append("--stream")
    assert main.main(argv) == {"completed": 0, "error": 1, "cancelled": 130}[status]

    raw = record_store.RUNS_PATH.read_text(encoding="utf-8")
    assert len(raw.splitlines()) == 1
    record = json.loads(raw)
    assert set(record) == set(record_store.RunRecord.__annotations__)
    assert record["request_status"] == status
    assert record["requested_model"] == "test-only-model"
    assert record["messages"][1] == {"role": "user", "content": "测试问题"}
    assert record["parameters"] == {
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": streaming,
        "thinking": {"type": "disabled"},
    }
    assert record["latency_ms"] is not None
    if not streaming:
        assert record["first_text_latency_ms"] is None
    if status == "completed":
        assert record["output_text"] == "【离线测试】第一行\n第二行"
        assert record["usage"] == usage
        assert record["returned_model"] == "test-returned-model"
        assert record["finish_reason"] == "stop"
        assert record["error_type"] is None
    else:
        assert record["output_text"] == ("【离线测试】第一行\n" if streaming else "")
        assert record["error_type"] == (
            "RuntimeError" if status == "error" else "KeyboardInterrupt"
        )
        assert record["returned_model"] is None
        assert record["usage"] is None
        assert record["finish_reason"] is None
    captured = capsys.readouterr()
    assert TEST_API_KEY not in raw + captured.err
    assert "password=secret" not in raw + captured.err
    assert "Authorization" not in raw + captured.err
    assert captured.err.endswith(
        f"run_id: {record['run_id']}\n记录已保存: {record_store.RUNS_PATH}\n"
    )


def test_record_contains_actual_context_and_unknown_metadata(
    tmp_path: Path, mock_model: tuple[Mock, Mock]
) -> None:
    context_path = tmp_path / "context.txt"
    context_path.write_text("【离线测试资料】\n第二行", encoding="utf-8")

    assert (
        main.main(["--question", "总结资料", "--context-file", str(context_path)]) == 0
    )

    record = json.loads(record_store.RUNS_PATH.read_text(encoding="utf-8"))
    sent_messages = mock_model[0].invoke.call_args.args[0]
    assert [message["content"] for message in record["messages"]] == [
        content for _, content in sent_messages
    ]
    assert record["returned_model"] is None
    assert record["usage"] is None
    assert record["first_text_latency_ms"] is None


@pytest.mark.parametrize("streaming", [False, True])
def test_requested_model_is_not_used_as_a_returned_model_fallback(
    streaming: bool, mock_model: tuple[Mock, Mock]
) -> None:
    model, _ = mock_model
    # 普通调用的同名 metadata 无法区分供应商原值与 SDK 回填。
    metadata = {"model_name": "test-only-model"}
    argv = ["--question", "问题"]
    if streaming:
        model.stream.return_value = iter(
            [AIMessageChunk(content="【离线测试】回答", response_metadata=metadata)]
        )
        argv.append("--stream")
    else:
        model.invoke.return_value = AIMessage(
            content="【离线测试】回答", response_metadata=metadata
        )

    assert main.main(argv) == 0

    record = json.loads(record_store.RUNS_PATH.read_text(encoding="utf-8"))
    assert record["returned_model"] == ("test-only-model" if streaming else None)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_initialization_failure_is_recorded_without_request_metrics(
    error_type: type[BaseException], mock_model: tuple[Mock, Mock]
) -> None:
    model, constructor = mock_model
    constructor.side_effect = error_type(TEST_API_KEY)

    assert main.main(["--question", "问题"]) == (
        130 if error_type is KeyboardInterrupt else 1
    )

    record = json.loads(record_store.RUNS_PATH.read_text(encoding="utf-8"))
    assert record["request_status"] == (
        "cancelled" if error_type is KeyboardInterrupt else "error"
    )
    assert record["error_type"] == error_type.__name__
    assert record["latency_ms"] is None
    assert record["usage"] is None
    assert record["output_text"] == ""
    model.invoke.assert_not_called()


@pytest.mark.parametrize("request_fails", [False, True])
def test_save_failure_is_explicit_and_does_not_claim_success(
    request_fails: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    if request_fails:
        mock_model[0].invoke.side_effect = RuntimeError(TEST_API_KEY)
    save = Mock(side_effect=PermissionError(f"{TEST_API_KEY}; password=secret"))
    monkeypatch.setattr(main, "append_record", save)

    assert main.main(["--question", "问题"]) == 1

    captured = capsys.readouterr()
    assert "记录保存失败" in captured.err
    assert "已保存" not in captured.err
    assert "run_id:" in captured.err
    assert str(record_store.RUNS_PATH) in captured.err
    assert TEST_API_KEY not in captured.err
    assert "password=secret" not in captured.err
    save.assert_called_once()
    assert not record_store.RUNS_PATH.exists()


@pytest.mark.parametrize("streaming", [False, True])
def test_length_finish_reason_and_missing_usage_are_preserved(
    streaming: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mock_model: tuple[Mock, Mock],
) -> None:
    model, _ = mock_model
    output = "【离线测试】中文回答\n到输出上限时截断"
    metadata = {"finish_reason": "length"}
    ticks = [10.0, 10.25, 11.0] if streaming else [10.0, 11.0]
    monkeypatch.setattr(main, "perf_counter", Mock(side_effect=ticks))
    argv = ["--question", "问题"]
    if streaming:
        model.stream.return_value = iter(
            [
                AIMessageChunk(content=""),
                AIMessageChunk(content=output),
                AIMessageChunk(content="", response_metadata=metadata),
            ]
        )
        argv.append("--stream")
    else:
        model.invoke.return_value = AIMessage(
            content=output, response_metadata=metadata
        )

    assert main.main(argv) == 0

    raw = record_store.RUNS_PATH.read_text(encoding="utf-8")
    assert len(raw.splitlines()) == 1
    record = json.loads(raw)
    assert record["output_text"] == output
    assert record["finish_reason"] == "length"
    assert record["request_status"] == "completed"
    assert record["error_type"] is None
    assert record["usage"] is None
    assert '"usage":null' in raw
    assert record["latency_ms"] == pytest.approx(1000.0)
    assert record["first_text_latency_ms"] == (
        pytest.approx(250.0) if streaming else None
    )
    captured = capsys.readouterr()
    assert captured.out == output + "\n"
    if streaming:
        model.invoke.assert_not_called()
        assert "finish_reason: length" in captured.err
    else:
        model.stream.assert_not_called()
