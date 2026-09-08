from __future__ import annotations

import os
from compact_chat_engine.config import ChatConfig


def test_default_config():
    config = ChatConfig()
    assert config.max_context_tokens == 250_000
    assert config.token_headroom == 8_000
    assert config.safe_limit == 242_000
    assert config.recent_buffer_count == 4
    assert config.database_path == "sessions.db"


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("ORCAROUTER_API_KEY", "custom_key")
    monkeypatch.setenv("ORCAROUTER_BASE_URL", "https://custom.url/v1")
    monkeypatch.setenv("MODEL_NAME", "custom-model")
    monkeypatch.setenv("MAX_CONTEXT_TOKENS", "100000")
    monkeypatch.setenv("TOKEN_HEADROOM", "5000")
    monkeypatch.setenv("DATABASE_PATH", "custom_sessions.db")

    config = ChatConfig.from_env()
    assert config.api_key == "custom_key"
    assert config.base_url == "https://custom.url/v1"
    assert config.model_name == "custom-model"
    assert config.max_context_tokens == 100000
    assert config.token_headroom == 5000
    assert config.safe_limit == 95000
    assert config.database_path == "custom_sessions.db"
