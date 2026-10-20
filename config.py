import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(
    *, env_file: Path | None = None, load_env: bool = True
) -> tuple[str, str]:
    """返回 (model_name, api_key)；可指定文件，或禁用 .env 只读取环境变量。"""
    if load_env:
        dotenv_path = env_file if env_file is not None else PROJECT_ROOT / ".env"
        try:
            load_dotenv(dotenv_path, override=False, encoding="utf-8")
        except (OSError, UnicodeError):
            raise ValueError("无法读取配置文件，请确认文件可读且为 UTF-8。") from None

    model_name = os.getenv("LLM_MODEL", "").strip()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

    if not api_key:
        raise ValueError("缺少 DEEPSEEK_API_KEY，请检查项目根目录的 .env 或环境变量。")
    if not model_name:
        raise ValueError("缺少 LLM_MODEL，请检查项目根目录的 .env 或环境变量。")

    return model_name, api_key
