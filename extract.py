"""单个售后请求的结构化抽取命令行入口。"""

import argparse  # 标准库：解析 --text、--mode 等命令行参数。
import hashlib  # 标准库：计算提示词的 SHA-256 哈希，方便比较实验版本。
import json  # 标准库：把 Python 字典转换为终端输出的 JSON。
import sys  # 提供 stderr，单独输出错误和保存提示。
from pathlib import Path  # 用对象拼接路径、读取文件，比手动拼字符串清晰。
from time import perf_counter  # 适合计算耗时的单调时钟，返回单位为秒。

# 以下导入的是已有函数和常量；导入模块本身不会读取 .env 或调用模型。
from config import PROJECT_ROOT, load_config
from extraction_record import (
    EXTRACTION_RUNS_PATH,
    append_extraction_record,
    collect_extraction_metadata,
    new_extraction_record,
    redact_values,
)
from extraction_rules import review_extraction
from extractor import (
    OUTPUT_METHODS,
    ExtractionParseError,
    build_extraction_prompt,
    invoke_extraction,
    parse_extraction_result,
)
from llm_client import create_model

# Path 的 / 运算符表示拼接路径；从项目根目录定位，不依赖终端当前目录。
PROMPT_PATH = PROJECT_ROOT / "prompts" / "extract.txt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # list[str] 表示字符串列表；| None 表示也可以不传参数。
    # argv=None 时 argparse 读取命令行；测试传列表就能模拟命令行。
    parser = argparse.ArgumentParser(description="从一段用户描述中抽取售后信息。")
    # 互斥组要求二选一；required=True 又要求至少选一个。
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="待抽取的用户原始描述")
    source.add_argument(
        "--text-file", type=Path, help="包含待抽取用户描述的 UTF-8 文本文件"
    )
    # choices 限定可选值；传入字典时使用它的键，即三个 CLI 模式名。
    parser.add_argument(
        "--mode",
        choices=OUTPUT_METHODS,
        default="structured",
        help="输出模式，默认 structured（function_calling）；json 使用 json_mode",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="最大输出 token 数，默认 1024"
    )
    # type=int 会将命令行字符串转换为整数；转换失败由 argparse 报错。
    args = parser.parse_args(argv)
    # Namespace 是保存解析结果的对象，例如 args.max_tokens。
    if args.max_tokens <= 0:
        parser.error("--max-tokens 必须是正整数。")  # 打印用法并以退出码 2 结束。
    return args


def read_text_file(path: Path) -> str:
    # try/except 类似 JS 的 try/catch；不同异常给出不同的可读提示。
    try:
        # read_text 会打开、解码并关闭文件；UTF-8 支持中文。
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"文件不存在：{path}") from None
    except UnicodeError:  # 文件存在，但内容不能按 UTF-8 解码。
        raise ValueError(f"文件不是有效的 UTF-8 文本：{path}") from None
    except OSError:  # 例如权限不足，或传入的是目录而不是文件。
        raise ValueError(f"无法读取文件：{path}，请检查路径和权限。") from None


def main(argv: list[str] | None = None) -> int:
    # 返回整数作为进程退出码：0 表示正常业务结果，非 0 表示失败或取消。
    args = parse_args(argv)
    # 先创建运行编号和记录，再逐步填入实际收到的数据；未知值保持 None。
    record = new_extraction_record(max_tokens=args.max_tokens, mode=args.mode)
    api_key = ""  # 尚未加载密钥；只在内存中用于请求和日志脱敏。
    stage = "input"  # 跟踪执行阶段，便于区分输入、请求和校验错误。
    exit_code = 0  # 默认成功；needs_clarification 也是正常业务结果。
    started_at = perf_counter()  # 保存起点，最后用终点减起点得到耗时。
    try:
        # 条件表达式：有文件参数就读取文件，否则使用 --text 原文。
        text = (
            read_text_file(args.text_file) if args.text_file is not None else args.text
        )
        record["input_text"] = text  # 保留原文，后续证据检查必须与原文比较。
        # strip() 去掉首尾空白；空字符串为假，所以 not text.strip() 表示空输入。
        if not text.strip():
            raise ValueError("待抽取的用户描述不能为空或仅包含空白。")
        system_prompt = read_text_file(PROMPT_PATH)  # 读取三种模式共用的业务规则。
        if not system_prompt.strip():
            raise ValueError("抽取提示词文件不能为空。")
        # 在业务规则后添加模式指令、自动生成的 Schema 和完整格式示例。
        system_prompt = build_extraction_prompt(system_prompt, mode=args.mode)
        # encode() 将字符串转成字节；hexdigest() 将哈希转成十六进制字符串。
        # 哈希用于比较提示词是否相同，不是用来加密或隐藏提示词。
        record["prompt_sha256"] = hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest()
        # 元组解包：load_config() 返回的两个值分别赋给模型名和密钥。
        # 只有显式执行到这里才加载配置；离线测试会替换此函数并禁用 .env。
        model_name, api_key = load_config()
        record["requested_model"] = model_name
        stage = "request"  # 从这里开始的初始化或调用异常归类为请求错误。
        record["request_status"] = "error"  # 先设失败；只有实际返回才改成 completed。
        # 关键字参数通过“名字=值”传递；create_model 只初始化已有 SDK 客户端。
        model = create_model(
            model_name=model_name,
            api_key=api_key,
            temperature=0.2,
            max_tokens=args.max_tokens,
        )
        # 根据 mode 发起一次请求；业务检查和追问不会再调用模型。
        response = invoke_extraction(model, text, system_prompt, mode=args.mode)
        record["request_status"] = "completed"  # 请求完成不代表解析或业务检查成功。
        stage = "validation"  # 后续异常不会被误记为模型请求失败。
        # 校验失败也保留实际返回的用量与结束原因，不保存整个原始响应。
        collect_extraction_metadata(record, response)
        result = parse_extraction_result(response, mode=args.mode)
        # mode="json" 生成 JSON 可保存的字典；保留原抽取值，不修改字段或证据。
        record["extraction"] = result.model_dump(mode="json")
        # 只在 Schema 与工具结构通过后检查业务；text 始终是最初的用户原文。
        review = review_extraction(text, result)
        record["review"] = review.model_dump(mode="json")
        record["status"] = review.status
        # 保留旧日志字段；这里只转换键名，不运行旧规则或发起第二次模型请求。
        record["checks"] = {
            "status": review.status,
            "evidence_errors": review.evidence_errors,
            "missing_fields": None
            if review.status == "evidence_error"
            else review.missing_fields,
            "clarification_questions": review.questions,
        }
        if record["status"] == "evidence_error":
            record["error_type"] = "EvidenceCheckError"
            record["error_message"] = "原文证据机械检查失败，请核对检查结果。"
            print(
                "抽取结果存在证据问题（evidence_error），不是用户信息不足。",
                file=sys.stderr,
            )
            exit_code = 4  # 证据不合格，不进入可复核草稿状态。
    except ExtractionParseError as exc:  # as exc 将捕获的异常绑定到变量。
        record["status"] = "validation_error"
        record["error_type"] = type(exc).__name__  # 只读取异常类别名。
        record["error_message"] = str(exc)  # 此类异常仅包含本地固定的安全说明。
        # 只显示本地定义的原因，不输出供应商原文或校验异常中的输入值。
        print(f"解析校验错误（validation_error）：{exc}", file=sys.stderr)
        exit_code = 3
    except KeyboardInterrupt:  # 用户在终端按 Ctrl+C 会触发此异常。
        record["status"] = "cancelled"
        # 请求已返回后取消本地处理，不抹掉“请求完成”这个事实。
        if record["request_status"] != "completed":
            record["request_status"] = "cancelled"
        record["error_type"] = "KeyboardInterrupt"
        record["error_message"] = "用户取消了本次抽取。"
        print("用户取消了本次抽取。", file=sys.stderr)
        exit_code = 130  # 终端中表示被用户中断的常用退出码。
    except Exception as exc:  # noqa: BLE001 - 不泄露请求异常中的凭据。
        # 不记录供应商异常全文：其中可能包含请求内容、请求头或凭据。
        record["error_type"] = type(exc).__name__
        if stage == "input":
            record["status"] = "input_error"
            record["error_message"] = "输入、提示词或配置无效，请检查命令参数与配置。"
            # 输入阶段的 ValueError 由本地函数生成；其他异常仅显示类别名。
            detail = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            print(f"输入或配置错误：{redact_values(detail, api_key)}", file=sys.stderr)
            exit_code = 2
        elif stage == "request":
            record["status"] = "request_error"
            record["error_message"] = (
                f"当前接口不支持 {args.mode} 模式；未自动降级或重试。"
                if isinstance(exc, NotImplementedError)
                else f"{args.mode} 模式请求失败，请检查接口支持情况、账户、网络和参数；"
                "未自动降级或重试。"
            )
            print(
                f"请求错误（request_error），错误类型：{type(exc).__name__}。"
                f"{record['error_message']}",
                file=sys.stderr,
            )
            exit_code = 1
        else:
            record["status"] = "validation_error"
            record["error_message"] = "无法验证抽取结果。"
            print(
                f"解析校验错误（validation_error），错误类型：{type(exc).__name__}",
                file=sys.stderr,
            )
            exit_code = 3
    finally:  # 不论 try 成功、失败还是被中断，这一段都会执行。
        # 整个本次处理的耗时（输入、请求、检查）；不含 JSONL 落盘时间。
        record["latency_ms"] = (perf_counter() - started_at) * 1000  # 秒转换成毫秒。

    # 保留 status 供旧代码读取，同时明确命名为 processing_status。
    record["processing_status"] = record["status"]
    # 只有完成业务检查才打印业务 JSON；解析失败时不拼一个假结果。
    if record["review"] is not None:
        output = {
            "extraction": record["extraction"],
            "review": record["review"],
            "processing_status": record["processing_status"],
            "status": record["status"],  # 兼容原有 JSON 输出字段。
            "checks": record["checks"],
        }
        # 先脱敏，再转 JSON；默认 print 写 stdout，方便调用者重定向保存结果。
        print(json.dumps(redact_values(output, api_key), ensure_ascii=False, indent=2))
        # stdout 继续保持一个可解析的 JSON；中文说明与追问写入 stderr。
        print("抽取结果：extraction；程序检查结果：review。", file=sys.stderr)
        if record["processing_status"] == "needs_clarification":
            print("固定追问：", file=sys.stderr)
            for question in record["review"]["questions"]:
                print(f"- {redact_values(question, api_key)}", file=sys.stderr)
        elif record["processing_status"] == "draft_ready":
            print("当前检查通过，仅供人工复核，不表示退款批准。", file=sys.stderr)

    # stderr 是另一条输出通道，运行信息不会混进 stdout 的 JSON 中。
    print(f"最终处理状态: {record['processing_status']}", file=sys.stderr)
    print(f"run_id: {record['run_id']}", file=sys.stderr)
    try:
        # 白名单校验后追加；只有成功才会返回保存路径。
        saved_path = append_extraction_record(record, api_key=api_key)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - 不显示落盘异常原文。
        print(
            f"记录保存失败：{EXTRACTION_RUNS_PATH}，错误类型：{type(exc).__name__}",
            file=sys.stderr,
        )
        return exit_code or 1  # 已有失败码就保留；此前成功则改为保存失败的退出码 1。
    print(f"记录已保存: {saved_path}", file=sys.stderr)
    return exit_code


# 直接运行 python extract.py 时 __name__ 为 __main__；被 import 时不执行入口。
if __name__ == "__main__":
    # SystemExit 把 main 返回的整数交给终端，其他程序可以据此判断执行结果。
    raise SystemExit(main())
