from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str
    is_archived: bool = False
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict) -> Message:
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            is_archived=bool(data.get("is_archived", False)),
            created_at=float(data.get("created_at", time.time())),
        )


@dataclass
class SessionState:
    session_id: str
    scratchpad: str = ""
    recent_messages: list[Message] = field(default_factory=list)
    archive: list[Message] = field(default_factory=list)


# Streaming Event Types
class ChatEvent:
    """Base class for all engine streaming events."""

    pass


@dataclass
class TextDeltaEvent(ChatEvent):
    """Fired when a token or text chunk is streamed from the model."""

    delta: str


@dataclass
class CompactionStartEvent(ChatEvent):
    """Fired when working memory synthesis begins."""

    pass


@dataclass
class CompactionEndEvent(ChatEvent):
    """Fired when working memory synthesis finishes."""

    scratchpad: str


@dataclass
class UsageReportEvent(ChatEvent):
    """Fired when generation completes, reporting token usage."""

    prompt_tokens: int
    completion_tokens: int


@dataclass
class InterruptedEvent(ChatEvent):
    """Fired when stream is canceled or interrupted."""

    pass


@dataclass
class ErrorEvent(ChatEvent):
    """Fired if an unrecoverable error occurs during streaming."""

    error: Exception
