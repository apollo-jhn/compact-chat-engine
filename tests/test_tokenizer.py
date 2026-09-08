from __future__ import annotations

from compact_chat_engine.models import Message
from compact_chat_engine.tokenizer import TokenizerManager


def test_tokenizer_heuristic_fallback(monkeypatch):
    # Force heuristic backend
    tm = TokenizerManager(repo_id="invalid/repo")
    tm.backend = "heuristic"
    tm.hf_tokenizer = None
    tm.tiktoken_enc = None

    text = "Hello world! This is a test."
    tokens = tm.count_text(text)
    assert tokens == max(1, len(text) // 4)

    empty_tokens = tm.count_text("")
    assert empty_tokens == 0


def test_tokenizer_count_messages():
    tm = TokenizerManager(repo_id="invalid/repo")
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
    ]
    total = tm.count_messages(messages)
    # Check that message counting returns a positive non-zero integer with overhead
    assert total > 0
    assert total > (tm.count_text("Hello") + tm.count_text("Hi"))
