"""Bring-your-own-key behaviour: where the key is read from, and the messages users see."""
import os

import anthropic
import httpx
import pytest

from study_assistant import config
from study_assistant.agent import friendly_api_error


@pytest.fixture(autouse=True)
def isolated_environ(monkeypatch):
    # load_settings() writes into os.environ; keep that from leaking into other tests.
    monkeypatch.setattr(os, "environ", dict(os.environ))


def test_env_file_in_current_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-file\nCLAUDE_MODEL=claude-sonnet-5-5\n")
    monkeypatch.chdir(tmp_path)
    settings = config.load_settings()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-file"
    assert settings.claude_model == "claude-sonnet-5-5"


def test_real_env_var_wins_over_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")
    monkeypatch.chdir(tmp_path)
    config.load_settings()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-env"


def test_empty_value_in_env_file_is_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "__file__", str(tmp_path / "pkg" / "config.py"))  # no project .env
    config.load_settings()
    assert "ANTHROPIC_API_KEY" not in os.environ


def test_fallbacks_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("CLAUDE_FALLBACKS", "off")
    assert config.load_settings().use_fallbacks is False
    monkeypatch.setenv("CLAUDE_FALLBACKS", "on")
    assert config.load_settings().use_fallbacks is True


def _status_error(cls, code):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(code, request=req), body=None)


def test_messages_for_missing_vs_rejected_key():
    missing = TypeError('"Could not resolve authentication method. Expected one of api_key ..."')
    assert "No Anthropic API key found" in friendly_api_error(missing)
    rejected = _status_error(anthropic.AuthenticationError, 401)
    assert "rejected (401)" in friendly_api_error(rejected)
    assert "CLAUDE_MODEL" in friendly_api_error(_status_error(anthropic.NotFoundError, 404))


def test_env_file_inline_comments_and_quotes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k in ("CLAUDE_MODEL", "CLAUDE_EFFORT", "ANTHROPIC_API_KEY"):
        os.environ.pop(k, None)
    (tmp_path / ".env").write_text(
        "CLAUDE_MODEL=claude-sonnet-5-5          # or claude-opus-5-5\n"
        'CLAUDE_EFFORT="high"\n'
        "ANTHROPIC_API_KEY='sk-ant-quoted#not-a-comment'\n"
    )
    settings = config.load_settings()
    assert settings.claude_model == "claude-sonnet-5-5"
    assert settings.claude_effort == "high"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-quoted#not-a-comment"
