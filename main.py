import argparse
import math
import sys
from pathlib import Path
from time import perf_counter

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_deepseek import ChatDeepSeek

from config import PROJECT_ROOT, load_config
from llm_client import create_model
from record_store import RUNS_PATH, RunRecord, append_record, new_record

# DeepSeek Chat Completions 的 max_tokens 上限：
# https://api-docs.deepseek.com/api/create-chat-completion/
MAX_TOKENS_LIMIT = 393216


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="本地 LLM 实验台：发送一次问题并打印回答。"
    )
    parser.add_argument(
        "--question", required=True, help="问题，去除首尾空白后不能为空"
    )
    parser.add_argument(
        "--system-file",
        type=Path,
        default=PROJECT_ROOT / "prompts" / "base.txt",
        help="UTF-8 系统提示词文件，默认使用项目根目录的 prompts/base.txt",
    )
    parser.add_argument("--context-file", type=Path, help="可选的 UTF-8 参考资料文件")
    parser.add_argument(
        "--stream", action="store_true", help="实时输出流式回答及调用统计"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="采样温度，范围 0–2，默认 0.2"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help=f"最大输出 token 数，范围 1–{MAX_TOKENS_LIMIT}，默认 512",
    )
    args = parser.parse_args(argv)
    args.question = args.question.strip()
    if not args.question:
        parser.error("--question 去除空白后不能为空。")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        parser.error("--temperature 必须是 0–2 之间的有限数值。")
    if not 1 <= args.max_tokens <= MAX_TOKENS_LIMIT:
        parser.error(f"--max-tokens 必须是 1–{MAX_TOKENS_LIMIT} 之间的整数。")
    return args


def read_text_file(path: Path) -> str:
    """将常见文件错误转换为清晰的本地校验提示。"""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"文件不存在：{path}") from None
    except UnicodeError:
        raise ValueError(f"文件不是有效的 UTF-8 文本：{path}") from None
    except OSError:
        raise ValueError(f"无法读取文件：{path}，请检查路径和权限。") from None


def build_messages(
    system_prompt: str, question: str, context: str | None = None
) -> list[tuple[str, str]]:
    user_message = question
    if context is not None:
        user_message = f"【参考资料】\n{context}\n\n【问题】\n{question}"
    return [("system", system_prompt), ("human", user_message)]


def collect_response_metadata(
    record: RunRecord, response: AIMessage, *, streaming: bool
) -> None:
    """仅提取允许的响应字段，不记录额外推理、请求头或原始响应对象。"""
    metadata = response.response_metadata
    returned_model = metadata.get("model_name")
    # langchain-openai 1.7.2 普通调用可能回填 self.model_name。
    # 同名时无法证明来自供应商；流式转换没有这项回填。
    if (
        isinstance(returned_model, str)
        and returned_model
        and (streaming or returned_model != record["requested_model"])
    ):
        record["returned_model"] = returned_model
    finish_reason = metadata.get("finish_reason")
    if isinstance(finish_reason, str):
        record["finish_reason"] = finish_reason
    usage = response.usage_metadata
    if usage is not None:
        record["usage"] = {}
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(name)
            record["usage"][name] = value if type(value) is int else None


def stream_response(
    model: ChatDeepSeek, messages: list[tuple[str, str]], record: RunRecord
) -> int:
    """实时输出回答，将流式统计写到 stderr；异常时保留已输出的文本。"""
    aggregated: AIMessageChunk | None = None
    first_text_latency_ms: float | None = None
    has_output = False
    failure: Exception | KeyboardInterrupt | None = None
    started_at = perf_counter()
    try:
        for chunk in model.stream(messages, stream_usage=True):
            # 当前版本的 .text 属性只提取文本，不包含额外的推理内容。
            text = chunk.text
            if text:
                if first_text_latency_ms is None and text.strip():
                    first_text_latency_ms = (perf_counter() - started_at) * 1000
                record["output_text"] += text
                has_output = True
                print(text, end="", flush=True)
            # 包括空文本的元数据片段；用 LangChain 的聚合接口处理用量。
            aggregated = chunk if aggregated is None else aggregated + chunk
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - 避免泄露原始异常中的密钥。
        failure = exc
    latency_ms = (perf_counter() - started_at) * 1000
    record["latency_ms"] = latency_ms
    record["first_text_latency_ms"] = first_text_latency_ms
    if aggregated is not None:
        collect_response_metadata(record, aggregated, streaming=True)
    if isinstance(failure, KeyboardInterrupt):
        record["request_status"] = "cancelled"
    elif failure is not None:
        record["request_status"] = "error"
    else:
        record["request_status"] = "completed"
    record["error_type"] = type(failure).__name__ if failure is not None else None

    if has_output:
        print(flush=True)
    if failure is not None:
        print(
            f"流式调用失败，错误类型：{type(failure).__name__}。"
            "已输出的文本可能不完整；以下统计仅基于已收到的片段。",
            file=sys.stderr,
        )
        status_code = getattr(failure, "status_code", None)
        if isinstance(status_code, int):
            print(f"HTTP 状态码：{status_code}", file=sys.stderr)

    usage_metadata = aggregated.usage_metadata if aggregated is not None else None
    finish_reason = (
        aggregated.response_metadata.get("finish_reason")
        if aggregated is not None
        else None
    )
    print(f"latency_ms: {latency_ms:.2f}", file=sys.stderr)
    first_text_display = (
        round(first_text_latency_ms, 2) if first_text_latency_ms is not None else None
    )
    print(f"first_text_latency_ms: {first_text_display}", file=sys.stderr)
    print(f"usage_metadata: {usage_metadata}", file=sys.stderr)
    print(f"finish_reason: {finish_reason}", file=sys.stderr)

    if isinstance(failure, KeyboardInterrupt):
        return 130
    return 1 if failure is not None else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    record = new_record(
        temperature=args.temperature, max_tokens=args.max_tokens, stream=args.stream
    )
    api_key = ""
    exit_code = 0
    stage = "input"
    try:
        system_prompt = read_text_file(args.system_file)
        context = (
            read_text_file(args.context_file) if args.context_file is not None else None
        )
        messages = build_messages(system_prompt, args.question, context)
        record["messages"] = [
            {"role": "user" if role == "human" else role, "content": content}
            for role, content in messages
        ]
        model_name, api_key = load_config()
        record["requested_model"] = model_name
        stage = "request"
        model = create_model(
            model_name=model_name,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if args.stream:
            exit_code = stream_response(model, messages, record)
        else:
            started_at = perf_counter()
            try:
                response = model.invoke(messages)
            finally:
                record["latency_ms"] = (perf_counter() - started_at) * 1000
            record["output_text"] = response.text
            collect_response_metadata(record, response, streaming=False)
            record["request_status"] = "completed"
            print(response.content)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - 仅保存异常类别。
        record["error_type"] = type(exc).__name__
        if isinstance(exc, KeyboardInterrupt):
            record["request_status"] = "cancelled"
            print("用户取消了本次运行。", file=sys.stderr)
            exit_code = 130
        else:
            record["request_status"] = "error"
            exit_code = 2 if stage == "input" else 1
            if stage == "input" and isinstance(exc, ValueError):
                print(f"输入或配置错误：{exc}", file=sys.stderr)
            elif isinstance(exc, ImportError):
                print(
                    "调用失败：缺少或无法导入 Python 依赖。请使用当前 Python 执行 "
                    "python -m pip install -r requirements.txt；SOCKS 代理需要 socksio。",
                    file=sys.stderr,
                )
            else:
                print(f"调用失败，错误类型：{type(exc).__name__}", file=sys.stderr)
                status_code = getattr(exc, "status_code", None)
                if isinstance(status_code, int):
                    print(f"HTTP 状态码：{status_code}", file=sys.stderr)
                print("请检查账户、网络和请求参数。", file=sys.stderr)
    finally:
        try:
            saved_path = append_record(record, api_key=api_key)
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - 保存错误也不打印原文。
            print(f"记录保存失败，错误类型：{type(exc).__name__}", file=sys.stderr)
            print(f"run_id: {record['run_id']}", file=sys.stderr)
            print(f"目标路径（未确认保存成功）: {RUNS_PATH}", file=sys.stderr)
            if isinstance(exc, KeyboardInterrupt):
                exit_code = 130
            elif exit_code == 0:
                exit_code = 1
        else:
            print(f"run_id: {record['run_id']}", file=sys.stderr)
            print(f"记录已保存: {saved_path}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
