from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from rich.console import Console

from compact_chat_engine.cli import (
    display_archive,
    display_help,
    display_scratchpad,
    display_sessions_table,
    display_tokens,
    handle_slash_command,
    render_session_header,
    render_welcome_banner,
    stream_turn,
)
from compact_chat_engine.config import ChatConfig
from compact_chat_engine.engine import ChatSession
from compact_chat_engine.models import (
    CompactionEndEvent,
    CompactionStartEvent,
    ErrorEvent,
    InterruptedEvent,
    Message,
    SessionState,
    TextDeltaEvent,
    UsageReportEvent,
)
from compact_chat_engine.storage import SqliteSessionStorage


@pytest.fixture
def mock_storage(tmp_path):
    db_path = str(tmp_path / "test_sessions.db")
    return SqliteSessionStorage(db_path=db_path)


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.backend = "mock-tiktoken"
    tok.count_messages.return_value = 42
    tok.count_text.return_value = 10
    return tok


@pytest.fixture
def test_console():
    return Console(record=True, width=100)


def test_list_sessions_detailed(mock_storage):
    # Empty initially
    assert mock_storage.list_sessions_detailed() == []

    # Save a session
    state = SessionState(
        session_id="test-session",
        scratchpad="## Core Facts\n- User likes Python",
        recent_messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ],
        archive=[
            Message(role="user", content="old 1", is_archived=True),
        ],
    )
    mock_storage.save_session(state)

    detailed = mock_storage.list_sessions_detailed()
    assert len(detailed) == 1
    item = detailed[0]
    assert item["session_id"] == "test-session"
    assert item["has_scratchpad"] is True
    assert item["active_count"] == 2
    assert item["archived_count"] == 1
    assert item["updated_at"] > 0


def test_render_welcome_banner(test_console, mock_storage):
    config = ChatConfig(api_key="test-key")
    render_welcome_banner(test_console, mock_storage, config)
    output = test_console.export_text()
    assert "Compact Chat Engine" in output
    assert "Safe Limit:" in output


def test_display_sessions_table(test_console, mock_storage):
    state = SessionState(
        session_id="alpha-session",
        scratchpad="",
        recent_messages=[Message(role="user", content="hi")],
    )
    mock_storage.save_session(state)

    display_sessions_table(test_console, mock_storage)
    output = test_console.export_text()
    assert "alpha-session" in output
    assert "Stored Chat Sessions" in output


def test_render_session_header(test_console, mock_storage, mock_tokenizer):
    bot = ChatSession(session_id="beta-session", storage=mock_storage, tokenizer=mock_tokenizer)
    render_session_header(test_console, bot)
    output = test_console.export_text()
    assert "beta-session" in output
    assert "Active Session" in output


def test_display_help(test_console):
    display_help(test_console)
    output = test_console.export_text()
    assert "/help" in output
    assert "/scratchpad" in output
    assert "/tokens" in output
    assert "/switch" in output
    assert "/clear" in output


def test_display_scratchpad(test_console, mock_storage, mock_tokenizer):
    bot = ChatSession(session_id="scratch-session", storage=mock_storage, tokenizer=mock_tokenizer)
    # Empty scratchpad
    display_scratchpad(test_console, bot)
    output = test_console.export_text()
    assert "empty" in output.lower()

    # Populated scratchpad
    test_console.clear()
    bot.scratchpad = "## Core Facts\n- Item A"
    display_scratchpad(test_console, bot)
    output = test_console.export_text()
    assert "Core Facts" in output


def test_display_tokens(test_console, mock_storage, mock_tokenizer):
    bot = ChatSession(session_id="tokens-session", storage=mock_storage, tokenizer=mock_tokenizer)
    display_tokens(test_console, bot)
    output = test_console.export_text()
    assert "Token Telemetry" in output
    assert "Active Context Tokens" in output


def test_display_archive(test_console, mock_storage, mock_tokenizer):
    bot = ChatSession(session_id="archive-session", storage=mock_storage, tokenizer=mock_tokenizer)
    display_archive(test_console, bot)
    output = test_console.export_text()
    assert "Turn Distribution" in output
    assert "Active Buffer" in output


def test_handle_slash_commands(test_console, mock_storage, mock_tokenizer):
    bot = ChatSession(session_id="cmd-session", storage=mock_storage, tokenizer=mock_tokenizer)
    bot.recent_messages.append(Message(role="user", content="to clear"))

    # Test /help
    handled, switch = handle_slash_command("/help", bot, test_console)
    assert handled is True and switch is None

    # Test /switch
    handled, switch = handle_slash_command("/switch next-session", bot, test_console)
    assert handled is True and switch == "next-session"

    handled, switch = handle_slash_command("/switch", bot, test_console)
    assert handled is True and switch is None

    # Test /clear
    assert len(bot.recent_messages) == 1
    handled, switch = handle_slash_command("/clear", bot, test_console)
    assert handled is True and switch is None
    assert len(bot.recent_messages) == 0

    # Test /exit and /quit
    handled, switch = handle_slash_command("/exit", bot, test_console)
    assert handled is False and switch == "__EXIT__"

    handled, switch = handle_slash_command("/quit", bot, test_console)
    assert handled is False and switch == "__EXIT__"

    # Test unknown command
    handled, switch = handle_slash_command("/unknowncmd", bot, test_console)
    assert handled is True and switch is None


def test_stream_turn_success(test_console, mock_storage, mock_tokenizer):
    bot = ChatSession(session_id="stream-session", storage=mock_storage, tokenizer=mock_tokenizer)
    bot.stream_chat = MagicMock(
        return_value=[
            TextDeltaEvent(delta="Hello "),
            TextDeltaEvent(delta="**world**!"),
            UsageReportEvent(prompt_tokens=15, completion_tokens=5),
        ]
    )

    stream_turn(bot, "hi", test_console)
    output = test_console.export_text()
    assert "Assistant" in output
    assert "world" in output
    assert "Prompt: 15" in output


def test_stream_turn_with_compaction(test_console, mock_storage, mock_tokenizer):
    bot = ChatSession(
        session_id="compaction-session", storage=mock_storage, tokenizer=mock_tokenizer
    )
    bot.stream_chat = MagicMock(
        return_value=[
            CompactionStartEvent(),
            CompactionEndEvent(scratchpad="## Updated Facts"),
            TextDeltaEvent(delta="Post compaction response"),
            UsageReportEvent(prompt_tokens=100, completion_tokens=10),
        ]
    )

    stream_turn(bot, "hi", test_console)
    output = test_console.export_text()
    assert "Memory compacted" in output
    assert "Post compaction response" in output


def test_stream_turn_interrupted(test_console, mock_storage, mock_tokenizer):
    bot = ChatSession(
        session_id="interrupt-session", storage=mock_storage, tokenizer=mock_tokenizer
    )
    bot.stream_chat = MagicMock(
        return_value=[
            TextDeltaEvent(delta="Partial"),
            InterruptedEvent(),
        ]
    )

    stream_turn(bot, "hi", test_console)
    output = test_console.export_text()
    assert "interrupted" in output.lower()


def test_stream_turn_error(test_console, mock_storage, mock_tokenizer):
    bot = ChatSession(session_id="error-session", storage=mock_storage, tokenizer=mock_tokenizer)
    bot.stream_chat = MagicMock(
        return_value=[
            ErrorEvent(error=RuntimeError("Connection timeout")),
        ]
    )

    stream_turn(bot, "hi", test_console)
    output = test_console.export_text()
    assert "Connection timeout" in output
