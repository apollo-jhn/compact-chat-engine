from __future__ import annotations

import json
import os
from compact_chat_engine.models import Message, SessionState
from compact_chat_engine.storage import SqliteSessionStorage


def test_sqlite_save_and_load(storage: SqliteSessionStorage):
    state = SessionState(
        session_id="test_session",
        scratchpad="Some scratchpad notes",
        recent_messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ],
        archive=[
            Message(role="user", content="Old message", is_archived=True),
        ],
    )

    storage.save_session(state)
    loaded = storage.load_session("test_session")

    assert loaded.session_id == "test_session"
    assert loaded.scratchpad == "Some scratchpad notes"
    assert len(loaded.recent_messages) == 2
    assert loaded.recent_messages[0].content == "Hello"
    assert loaded.recent_messages[1].content == "Hi there!"
    assert len(loaded.archive) == 1
    assert loaded.archive[0].content == "Old message"
    assert loaded.archive[0].is_archived is True


def test_sqlite_list_and_delete(storage: SqliteSessionStorage):
    s1 = SessionState(session_id="session_alpha")
    s2 = SessionState(session_id="session_beta")

    storage.save_session(s1)
    storage.save_session(s2)

    sessions = storage.list_sessions()
    assert "session_alpha" in sessions
    assert "session_beta" in sessions

    storage.delete_session("session_alpha")
    sessions_after = storage.list_sessions()
    assert "session_alpha" not in sessions_after
    assert "session_beta" in sessions_after


def test_auto_migrate_legacy_flat_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_file = str(tmp_path / "test.db")
    storage = SqliteSessionStorage(db_path=db_file)

    # Write a legacy flat JSON file
    legacy_file = "session_legacy1.json"
    legacy_turns = [
        {"role": "user", "content": "Turn 1"},
        {"role": "assistant", "content": "Reply 1"},
        {"role": "user", "content": "Turn 2"},
        {"role": "assistant", "content": "Reply 2"},
        {"role": "user", "content": "Turn 3"},
        {"role": "assistant", "content": "Reply 3"},
    ]
    with open(legacy_file, "w", encoding="utf-8") as f:
        json.dump(legacy_turns, f)

    loaded = storage.load_session("legacy1", recent_buffer_count=2)
    assert loaded.session_id == "legacy1"
    assert len(loaded.recent_messages) == 2
    assert loaded.recent_messages[-1].content == "Reply 3"
    assert len(loaded.archive) == 4
    assert loaded.archive[0].content == "Turn 1"

    # Verify legacy file was marked as .migrated
    assert not os.path.exists(legacy_file)
    assert os.path.exists(f"{legacy_file}.migrated")


def test_auto_migrate_legacy_dual_layer_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db_file = str(tmp_path / "test.db")
    storage = SqliteSessionStorage(db_path=db_file)

    legacy_file = "session_dual.json"
    legacy_payload = {
        "version": 2,
        "scratchpad": "Existing facts",
        "recent_messages": [{"role": "user", "content": "Recent"}],
        "archive": [{"role": "user", "content": "Archived"}],
    }
    with open(legacy_file, "w", encoding="utf-8") as f:
        json.dump(legacy_payload, f)

    loaded = storage.load_session("dual")
    assert loaded.session_id == "dual"
    assert loaded.scratchpad == "Existing facts"
    assert len(loaded.recent_messages) == 1
    assert loaded.recent_messages[0].content == "Recent"
    assert len(loaded.archive) == 1
    assert loaded.archive[0].content == "Archived"

    assert os.path.exists(f"{legacy_file}.migrated")
