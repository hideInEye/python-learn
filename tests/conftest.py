"""默认测试隔离：在收集测试前阻止联网，运行时只使用临时配置和日志。"""

import builtins  # Python 内建函数所在模块，例如 open。
import io  # Path.read_text 等文件操作会用到的 I/O 接口。
import os  # 环境变量映射、底层文件操作等。
import socket  # 网络连接和 DNS 查询入口，测试中要拦截它们。
import subprocess  # 子进程可能绕开本进程的网络限制，因此也要拦截。
from collections.abc import Callable, Iterator  # 函数和迭代器的类型提示。
from functools import partial, wraps  # 预先绑定参数，以及保留被包装函数的信息。
from pathlib import Path  # 统一处理项目文件和测试临时路径。
from unittest.mock import Mock  # 创建可控制返回值、可检查调用次数的替身。

import pytest

# conftest.py 会被 pytest 自动加载，不需要每个测试文件手工 import 它。
# parents[0] 是 tests 目录，parents[1] 是项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_RUNS = PROJECT_ROOT / "data" / "runs"
SESSION_PATCHES = pytest.MonkeyPatch()  # 整场测试共享的替换管理器，结束时统一还原。
TEMP_ROOTS: set[Path] = set()  # 只允许当前测试自己的临时 .env，不允许真实配置文件。


def block_network(*args: object, **kwargs: object) -> None:
    # *args 接收任意位置参数，**kwargs 接收任意关键字参数，方便替换不同接口。
    # pytest.fail 不会被生产代码的 except Exception 吞掉。
    pytest.fail("离线测试禁止真实网络访问或调用真实模型。", pytrace=False)


def block_subprocess(*args: object, **kwargs: object) -> None:
    # pytest.fail 立即把测试标记为失败，提醒出现了未显式模拟的外部操作。
    pytest.fail("离线测试禁止启动未模拟的子进程。", pytrace=False)


def check_file_access(file: object) -> None:
    # 文件操作也可能传文件描述符等非路径对象，这里只检查可表示路径的参数。
    if not isinstance(file, (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(file)).resolve()  # 统一成绝对路径，也解析符号链接。
    if path == PROJECT_ROOT / ".env" or (
        path.name == ".env"
        # is_relative_to 判断路径是否在某个临时根目录之下。
        and not any(path.is_relative_to(root) for root in TEMP_ROOTS)
    ):
        pytest.fail("离线测试禁止访问真实 .env。", pytrace=False)
    if path.is_relative_to(PRODUCTION_RUNS):
        pytest.fail("离线测试禁止访问生产运行日志。", pytrace=False)


def guarded_file_operation(operation: Callable) -> Callable:
    # 返回一个包裹原操作的新函数：先检查文件路径，再执行原操作。
    # @wraps 是装饰器，保留原函数名称等信息，便于查看报错。
    @wraps(operation)
    def guarded(*args: object, **kwargs: object) -> object:
        # 优先取第一个位置参数，否则兼容 file= 或 path= 的关键字传参。
        file = args[0] if args else kwargs.get("file", kwargs.get("path"))
        check_file_access(file)
        return operation(*args, **kwargs)  # 解包参数，原样转发给真实的文件函数。

    return guarded  # 返回函数本身，不是现在就调用它。


def pytest_configure() -> None:
    """早于测试模块导入，避免在收集测试时发生意外请求或读取。"""
    # pytest 的生命周期钩子：在收集测试模块前安装保护，避免导入时意外联网。
    for name in (
        "create_connection",
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
        "gethostbyaddr",
        "getnameinfo",
    ):
        # setattr 暂时把指定属性替换成拦截函数；不改第三方库的源文件。
        SESSION_PATCHES.setattr(socket, name, block_network)
    for name in ("connect", "connect_ex", "sendto", "sendmsg"):
        if hasattr(socket.socket, name):  # 某些平台没有所有 socket 方法，先检查存在性。
            SESSION_PATCHES.setattr(socket.socket, name, block_network)
    SESSION_PATCHES.setattr(subprocess, "Popen", block_subprocess)
    SESSION_PATCHES.setattr(os, "system", block_subprocess)
    # 元组解包：每次循环同时得到“模块”和“需要替换的属性名”。
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
    SESSION_PATCHES.undo()  # 测试结束后还原所有会话级替换，不影响后续正常运行。


# fixture 是 pytest 自动准备的依赖；autouse=True 让每个测试都使用它。
# monkeypatch 和 tmp_path 由 pytest 提供，不是业务代码中的全局变量。
@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    import config
    import extract
    import extraction_record
    import main
    import record_store

    # 使用独立映射，测试无法继承开发者的凭据或追踪配置。
    monkeypatch.setattr(os, "environ", {})
    # wraps 表示仍可调用原函数，但同时记录调用次数，供断言检查。
    monkeypatch.setattr(config, "load_dotenv", Mock(wraps=config.load_dotenv))
    # partial 预先固定 load_env=False；后续调用不再读取真实 .env。
    monkeypatch.setattr(
        main, "load_config", partial(config.load_config, load_env=False)
    )
    monkeypatch.setattr(
        extract, "load_config", partial(config.load_config, load_env=False)
    )
    # tmp_path 是每个测试独有的临时目录，日志写入它不会污染真实运行记录。
    runs_path = tmp_path / "project" / "data" / "runs" / "runs.jsonl"
    monkeypatch.setattr(record_store, "RUNS_PATH", runs_path)
    monkeypatch.setattr(main, "RUNS_PATH", runs_path)
    extraction_path = runs_path.with_name("extraction.jsonl")
    monkeypatch.setattr(extraction_record, "EXTRACTION_RUNS_PATH", extraction_path)
    monkeypatch.setattr(extract, "EXTRACTION_RUNS_PATH", extraction_path)
    TEMP_ROOTS.add(tmp_path.resolve())  # 登记此测试允许使用的临时配置目录。
    try:
        yield  # 暂停 fixture，让测试函数执行；测试结束后从这里继续清理。
    finally:
        TEMP_ROOTS.discard(tmp_path.resolve())  # 即使测试失败也移除临时授权路径。
