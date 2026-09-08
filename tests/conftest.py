from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from compact_chat_engine.config import ChatConfig
from compact_chat_engine.storage import SqliteSessionStorage


@pytest.fixture
def temp_db(tmp_path):
    return str(tmp_path / "test_sessions.db")


@pytest.fixture
def storage(temp_db):
    return SqliteSessionStorage(db_path=temp_db)


@pytest.fixture
def test_config(temp_db):
    return ChatConfig(
        api_key="test-key",
        base_url="https://mock.example.com/v1",
        model_name="test-model",
        max_context_tokens=100,
        token_headroom=20,
        recent_buffer_count=2,
        compaction_temperature=0.2,
        database_path=temp_db,
    )


@pytest.fixture
def mock_openai_client():
    client = MagicMock()

    # Mock streaming response
    def create_mock_completion(**kwargs):
        if kwargs.get("stream"):
            chunks = []
            words = ["Hello", " ", "world!"]
            for w in words:
                chunk = MagicMock()
                chunk.usage = None
                choice = MagicMock()
                choice.delta.content = w
                chunk.choices = [choice]
                chunks.append(chunk)

            # Add usage to final chunk
            usage_chunk = MagicMock()
            usage_chunk.usage = MagicMock()
            usage_chunk.usage.prompt_tokens = 10
            usage_chunk.usage.completion_tokens = 5
            usage_chunk.choices = []
            chunks.append(usage_chunk)

            return iter(chunks)
        else:
            # Non-streaming (e.g. compaction)
            resp = MagicMock()
            choice = MagicMock()
            choice.message.content = "## Core Facts & Decisions\n- User says hello."
            resp.choices = [choice]
            return resp

    client.chat.completions.create.side_effect = create_mock_completion
    return client
