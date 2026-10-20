from langchain_deepseek import ChatDeepSeek


def create_model(
    *,
    model_name: str,
    api_key: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> ChatDeepSeek:
    """初始化模型客户端；真正的请求由调用方执行 invoke 或 stream 时发出。"""
    return ChatDeepSeek(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
