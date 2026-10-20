import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek


def main() -> None:
    project_root = Path(__file__).resolve().parent
    load_dotenv(project_root / ".env")

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    model_name = os.getenv("LLM_MODEL", "").strip()

    if not api_key:
        raise SystemExit("缺少 DEEPSEEK_API_KEY，请检查本地 .env。")

    if not model_name:
        raise SystemExit("缺少 LLM_MODEL，请检查本地 .env。")

    try:
        model = ChatDeepSeek(
            model=model_name,
            api_key=api_key,
            temperature=0.2,
            max_tokens=512,
            timeout=30,
            max_retries=0,
            extra_body={"thinking": {"type": "disabled"}},
        )

        messages = [
            ("system", "请用中文回答，表达清楚，不知道的信息不要编造。"),
            ("human", "用三句话向新手解释什么是大语言模型。"),
        ]

        started_at = perf_counter()
        response = model.invoke(messages)
        elapsed_ms = (perf_counter() - started_at) * 1000

    except ImportError:
        print("调用失败：缺少或无法导入 Python 依赖，尚未获得 HTTP 响应。")
        print(
            "请使用运行本脚本的 Python 执行：python -m pip install -r requirements.txt"
        )
        print("若使用 SOCKS 代理，请确认已安装 socksio。")
        raise SystemExit(1) from None

    except Exception as exc:  # noqa: BLE001 - 避免原始异常泄露 API Key。
        status_code = getattr(exc, "status_code", None)
        print(f"调用失败，错误类型：{type(exc).__name__}")
        print(f"HTTP 状态码：{status_code}")
        print("请检查账户、网络和请求参数；不要输出 API Key。")
        raise SystemExit(1) from None

    print(response)
    print("\n模型回答：")
    print(response.content)

    print("\n调用信息：")
    print(f"耗时：{elapsed_ms:.0f} ms")
    print(f"Token 用量：{response.usage_metadata}")
    print(f"结束原因：{response.response_metadata.get('finish_reason')}")


if __name__ == "__main__":
    main()
