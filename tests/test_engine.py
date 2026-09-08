from __future__ import annotations

from compact_chat_engine.engine import ChatSession
from compact_chat_engine.models import (
    CompactionEndEvent,
    CompactionStartEvent,
    Message,
    TextDeltaEvent,
    UsageReportEvent,
)


def test_engine_stream_chat(test_config, storage, mock_openai_client):
    session = ChatSession(
        session_id="test_stream",
        config=test_config,
        storage=storage,
        client=mock_openai_client,
    )

    events = list(session.stream_chat("What is the capital of France?"))

    # Assert events emitted
    delta_events = [e for e in events if isinstance(e, TextDeltaEvent)]
    assert len(delta_events) == 3
    full_text = "".join(e.delta for e in delta_events)
    assert full_text == "Hello world!"

    usage_events = [e for e in events if isinstance(e, UsageReportEvent)]
    assert len(usage_events) == 1
    assert usage_events[0].prompt_tokens == 10
    assert usage_events[0].completion_tokens == 5

    # Assert state updated
    assert len(session.recent_messages) == 2
    assert session.recent_messages[0].role == "user"
    assert session.recent_messages[1].role == "assistant"
    assert session.recent_messages[1].content == "Hello world!"

    # Assert state persisted in storage
    reloaded = storage.load_session("test_stream")
    assert len(reloaded.recent_messages) == 2
    assert reloaded.recent_messages[1].content == "Hello world!"


def test_engine_compaction_trigger(test_config, storage, mock_openai_client):
    # Set low limits to force compaction
    test_config.max_context_tokens = 50
    test_config.token_headroom = 10
    # Safe limit = 40
    test_config.recent_buffer_count = 2

    session = ChatSession(
        session_id="test_compact",
        config=test_config,
        storage=storage,
        client=mock_openai_client,
    )

    # Prepopulate with 4 turns
    session.recent_messages.extend(
        [
            Message(role="user", content="Turn 1 " * 10),
            Message(role="assistant", content="Turn 2 " * 10),
            Message(role="user", content="Turn 3 " * 10),
            Message(role="assistant", content="Turn 4 " * 10),
        ]
    )
    session.save_session()

    events = list(session.stream_chat("Trigger compaction now!"))

    # Verify compaction events yielded
    compaction_starts = [e for e in events if isinstance(e, CompactionStartEvent)]
    compaction_ends = [e for e in events if isinstance(e, CompactionEndEvent)]
    assert len(compaction_starts) == 1
    assert len(compaction_ends) == 1
    assert "Core Facts & Decisions" in compaction_ends[0].scratchpad

    # Verify recent_messages trimmed to recent_buffer_count (2) + newly generated assistant turn (1) = 3
    assert len(session.recent_messages) == 3
    # Verify evicted turns were moved to archive
    assert len(session.archive) >= 2


def test_engine_chat_convenience_method(test_config, storage, mock_openai_client):
    session = ChatSession(
        session_id="test_convenience",
        config=test_config,
        storage=storage,
        client=mock_openai_client,
    )

    result = session.chat("Hello there")
    assert result == "Hello world!"
