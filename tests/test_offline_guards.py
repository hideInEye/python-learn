"""这些测试故意触发拦截器；操作在接触真实网络或文件内容前就失败。"""

import builtins
import io
import os
import socket
import subprocess
from pathlib import Path

import pytest
from langchain_deepseek import ChatDeepSeek

import config
import main
import record_store

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_network_is_blocked_during_test_collection() -> None:
    assert COLLECTION_GUARD_ACTIVE


# 模块顶层在 fixture 运行前执行，验证保护在测试收集阶段已经生效。
with pytest.raises(pytest.fail.Exception, match="禁止真实网络"):
    socket.getaddrinfo("example.invalid", 443)
COLLECTION_GUARD_ACTIVE = True


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_socket_connections_are_blocked(method: str) -> None:
    with (
        socket.socket() as client,
        pytest.raises(pytest.fail.Exception, match="禁止真实网络"),
    ):
        getattr(client, method)(("192.0.2.1", 443))


def test_udp_is_blocked() -> None:
    with (
        socket.socket(type=socket.SOCK_DGRAM) as client,
        pytest.raises(pytest.fail.Exception, match="禁止真实网络"),
    ):
        client.sendto(b"offline-test", ("192.0.2.1", 443))


@pytest.mark.parametrize("method", ["invoke", "stream", "ainvoke", "astream"])
def test_real_model_methods_are_blocked(method: str) -> None:
    # 不构造真实客户端；检查方法入口处的保护。
    with pytest.raises(pytest.fail.Exception, match="禁止真实网络"):
        getattr(ChatDeepSeek, method)(None, [("human", "离线保护测试")])


def test_unsimulated_subprocess_is_blocked() -> None:
    with pytest.raises(pytest.fail.Exception, match="禁止启动未模拟的子进程"):
        subprocess.run(["this-command-must-never-run"], check=True)


@pytest.mark.parametrize("opener", [builtins.open, io.open, os.open])
def test_real_dotenv_is_blocked_before_any_read(opener: object) -> None:
    path = PROJECT_ROOT / ".env"
    with pytest.raises(pytest.fail.Exception, match="禁止访问真实 .env"):
        if opener is os.open:
            opener(path, os.O_RDONLY)
        else:
            with opener(path, encoding="utf-8"):
                pytest.fail("不应成功打开真实 .env")


def test_default_configuration_cannot_read_real_dotenv() -> None:
    with pytest.raises(pytest.fail.Exception, match="禁止访问真实 .env"):
        config.load_config()


def test_production_log_is_blocked_before_writing() -> None:
    path = PROJECT_ROOT / "data" / "runs" / "runs.jsonl"
    with pytest.raises(pytest.fail.Exception, match="禁止访问生产运行日志"):
        path.write_text("不应被写入", encoding="utf-8")


def test_every_test_gets_a_temporary_log_and_empty_environment(tmp_path: Path) -> None:
    assert record_store.RUNS_PATH.is_relative_to(tmp_path)
    assert main.RUNS_PATH == record_store.RUNS_PATH
    assert os.getenv("DEEPSEEK_API_KEY") is None
    assert os.getenv("LLM_MODEL") is None
    assert os.getenv("LANGSMITH_API_KEY") is None
