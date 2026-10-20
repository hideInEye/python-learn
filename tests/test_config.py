from pathlib import Path
from unittest.mock import Mock

import pytest

import config


def test_load_config_from_explicit_temporary_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=test-file-key\nLLM_MODEL=test-file-model\n", encoding="utf-8"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert config.load_config(env_file=env_file) == ("test-file-model", "test-file-key")
    config.load_dotenv.assert_called_once_with(
        env_file, override=False, encoding="utf-8"
    )


def test_disable_dotenv_uses_only_explicit_test_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = Mock(side_effect=AssertionError("禁用 .env 时不得调用加载器"))
    monkeypatch.setattr(config, "load_dotenv", loader)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "  test-environment-key  ")
    monkeypatch.setenv("LLM_MODEL", "  test-environment-model  ")

    assert config.load_config(load_env=False) == (
        "test-environment-model",
        "test-environment-key",
    )
    loader.assert_not_called()


def test_disable_dotenv_also_ignores_an_explicit_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=test-file-key\nLLM_MODEL=test-file-model\n", encoding="utf-8"
    )
    monkeypatch.setenv("LLM_MODEL", "test-environment-model")

    with pytest.raises(ValueError, match="缺少 DEEPSEEK_API_KEY"):
        config.load_config(env_file=env_file, load_env=False)
    config.load_dotenv.assert_not_called()


def test_explicit_file_keeps_existing_environment_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=test-file-key\nLLM_MODEL=test-file-model\n", encoding="utf-8"
    )
    monkeypatch.setenv("LLM_MODEL", "test-environment-model")

    assert config.load_config(env_file=env_file) == (
        "test-environment-model",
        "test-file-key",
    )


def test_invalid_utf8_configuration_fails_safely(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"\xff\xfe")

    with pytest.raises(ValueError, match="UTF-8"):
        config.load_config(env_file=env_file)
