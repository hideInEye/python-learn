"""默认测试隔离：在收集测试前阻止联网，运行时只使用临时配置和日志。"""

import builtins
import io
import os
import socket
import subprocess
from collections.abc import Callable, Iterator
from functools import partial, wraps
from pathlib import Path
from unittest.mock import Mock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_RUNS = PROJECT_ROOT / "data" / "runs"
SESSION_PATCHES = pytest.MonkeyPatch()
TEMP_ROOTS: set[Path] = set()


def block_network(*args: object, **kwargs: object) -> None:
    # pytest.fail 不会被生产代码的 except Exception 吞掉。
    pytest.fail("离线测试禁止真实网络访问或调用真实模型。", pytrace=False)


def block_subprocess(*args: object, **kwargs: object) -> None:
    pytest.fail("离线测试禁止启动未模拟的子进程。", pytrace=False)


def check_file_access(file: object) -> None:
    if not isinstance(file, (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(file)).resolve()
    if path == PROJECT_ROOT / ".env" or (
        path.name == ".env"
        and not any(path.is_relative_to(root) for root in TEMP_ROOTS)
    ):
        pytest.fail("离线测试禁止访问真实 .env。", pytrace=False)
    if path.is_relative_to(PRODUCTION_RUNS):
        pytest.fail("离线测试禁止访问生产运行日志。", pytrace=False)


def guarded_file_operation(operation: Callable) -> Callable:
    @wraps(operation)
    def guarded(*args: object, **kwargs: object) -> object:
        file = args[0] if args else kwargs.get("file", kwargs.get("path"))
        check_file_access(file)
        return operation(*args, **kwargs)

    return guarded


def pytest_configure() -> None:
    """早于测试模块导入，避免在收集测试时发生意外请求或读取。"""
    for name in (
        "create_connection",
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
        "gethostbyaddr",
        "getnameinfo",
    ):
        SESSION_PATCHES.setattr(socket, name, block_network)
    for name in ("connect", "connect_ex", "sendto", "sendmsg"):
        if hasattr(socket.socket, name):
            SESSION_PATCHES.setattr(socket.socket, name, block_network)
    SESSION_PATCHES.setattr(subprocess, "Popen", block_subprocess)
    SESSION_PATCHES.setattr(os, "system", block_subprocess)
    for module, name in (
        (builtins, "open"),
        (io, "open"),
        (os, "open"),
        (os, "mkdir"),
    ):
        SESSION_PATCHES.setattr(
            module, name, guarded_file_operation(getattr(module, name))
        )

    # 先安装网络/文件保护，再导入框架；Mock 模型不受此限制。
    from langchain_deepseek import ChatDeepSeek

    for name in ("invoke", "stream", "ainvoke", "astream"):
        SESSION_PATCHES.setattr(ChatDeepSeek, name, block_network)


def pytest_unconfigure() -> None:
    SESSION_PATCHES.undo()


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    import config
    import main
    import record_store

    # 使用独立映射，测试无法继承开发者的凭据或追踪配置。
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(config, "load_dotenv", Mock(wraps=config.load_dotenv))
    monkeypatch.setattr(
        main, "load_config", partial(config.load_config, load_env=False)
    )
    runs_path = tmp_path / "project" / "data" / "runs" / "runs.jsonl"
    monkeypatch.setattr(record_store, "RUNS_PATH", runs_path)
    monkeypatch.setattr(main, "RUNS_PATH", runs_path)
    TEMP_ROOTS.add(tmp_path.resolve())
    try:
        yield
    finally:
        TEMP_ROOTS.discard(tmp_path.resolve())
