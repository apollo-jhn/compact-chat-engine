from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from .config import ChatConfig
from .models import (
    ChatEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    ErrorEvent,
    InterruptedEvent,
    Message,
    SessionState,
    TextDeltaEvent,
    UsageReportEvent,
)
from .storage import SqliteSessionStorage
from .tokenizer import TokenizerManager


class ChatSession:
    """
    Core conversation engine featuring working memory synthesis,
    tiered token limits, and event-driven decoupled streaming.
    """

    def __init__(
        self,
        session_id: str = "default",
        config: ChatConfig | None = None,
        storage: SqliteSessionStorage | None = None,
        tokenizer: TokenizerManager | None = None,
        client: OpenAI | None = None,
    ):
        self.session_id = session_id
        self.config = config or ChatConfig.from_env()
        self.storage = storage or SqliteSessionStorage(self.config.database_path)
        self.tokenizer = tokenizer or TokenizerManager(self.config.hf_tokenizer_repo)
        self.client = client or OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )

        self.supports_stream_options = True
        self.state: SessionState = self.storage.load_session(
            self.session_id, recent_buffer_count=self.config.recent_buffer_count
        )

    @property
    def scratchpad(self) -> str:
        return self.state.scratchpad

    @scratchpad.setter
    def scratchpad(self, value: str) -> None:
        self.state.scratchpad = value

    @property
    def recent_messages(self) -> list[Message]:
        return self.state.recent_messages

    @property
    def archive(self) -> list[Message]:
        return self.state.archive

    def save_session(self) -> None:
        """Persists current state to storage."""
        self.storage.save_session(self.state)

    def build_system_message(self) -> dict[str, str]:
        """Consolidates base instructions and scratchpad memory into index 0."""
        if not self.scratchpad.strip():
            return {"role": "system", "content": self.config.base_system_prompt}

        content = (
            f"{self.config.base_system_prompt}\n\n"
            f"--- WORKING MEMORY & CONTEXT SCRATCHPAD ---\n"
            f"{self.scratchpad.strip()}\n"
            f"------------------------------------------"
        )
        return {"role": "system", "content": content}

    def get_active_payload(self) -> list[dict[str, str]]:
        payload = [self.build_system_message()]
        payload.extend(msg.to_dict() for msg in self.recent_messages)
        return payload

    def merge_to_scratchpad(self, evicted_turns: list[Message]) -> str:
        """Synthesizes evicted turns into the structured markdown scratchpad."""
        transcript_lines = [f"{msg.role.upper()}: {msg.content}" for msg in evicted_turns]
        transcript_text = "\n\n".join(transcript_lines)

        merge_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert memory synthesis assistant. Update the existing memory scratchpad "
                    "by integrating the provided conversation turns.\n\n"
                    "Output strictly using these Markdown sections:\n"
                    "## Core Facts & Decisions\n"
                    "## User Preferences\n"
                    "## In-Progress Tasks\n\n"
                    "Retain all enduring facts, resolve conflicting information with newer updates, "
                    "and keep descriptions concise and dense."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Existing Scratchpad:\n{self.scratchpad or 'None'}\n\n"
                    f"New Turns to Merge:\n{transcript_text}"
                ),
            },
        ]

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=merge_messages,
            temperature=self.config.compaction_temperature,
        )

        new_scratchpad = response.choices[0].message.content.strip()
        self.scratchpad = new_scratchpad
        for turn in evicted_turns:
            turn.is_archived = True
        self.archive.extend(evicted_turns)
        self.save_session()
        return new_scratchpad

    def compact_context_if_needed(self) -> str | None:
        """Monitors token usage and compacts historical turns when crossing the limit."""
        active_messages = self.get_active_payload()
        current_tokens = self.tokenizer.count_messages(active_messages)

        if current_tokens <= self.config.safe_limit:
            return None

        if len(self.recent_messages) <= self.config.recent_buffer_count:
            return None

        evicted_turns = self.recent_messages[: -self.config.recent_buffer_count]
        self.state.recent_messages = self.recent_messages[-self.config.recent_buffer_count :]
        return self.merge_to_scratchpad(evicted_turns)

    def stream_chat(self, user_input: str) -> Iterator[ChatEvent]:
        """
        Processes a user turn and yields streaming events as tokens arrive.
        Decoupled from CLI/stdout presentation.
        """
        user_message = Message(role="user", content=user_input)
        self.recent_messages.append(user_message)

        # Check token usage & compact if necessary
        active_messages = self.get_active_payload()
        current_tokens = self.tokenizer.count_messages(active_messages)
        if (
            current_tokens > self.config.safe_limit
            and len(self.recent_messages) > self.config.recent_buffer_count
        ):
            yield CompactionStartEvent()
            evicted = self.recent_messages[: -self.config.recent_buffer_count]
            self.state.recent_messages = self.recent_messages[-self.config.recent_buffer_count :]
            new_scratchpad = self.merge_to_scratchpad(evicted)
            yield CompactionEndEvent(scratchpad=new_scratchpad)

        active_payload = self.get_active_payload()
        collected_chunks: list[str] = []
        reported_usage = None

        request_kwargs = {
            "model": self.config.model_name,
            "messages": active_payload,
            "stream": True,
        }
        if self.supports_stream_options:
            request_kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = self.client.chat.completions.create(**request_kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            if self.supports_stream_options and any(
                k in err_msg for k in ["stream_options", "extra", "unknown", "400", "422"]
            ):
                self.supports_stream_options = False
                request_kwargs.pop("stream_options", None)
                stream = self.client.chat.completions.create(**request_kwargs)
            else:
                self.recent_messages.pop()  # Rollback user turn on hard failure
                yield ErrorEvent(error=e)
                return

        try:
            for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    reported_usage = chunk.usage

                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content or ""
                    if delta_content:
                        collected_chunks.append(delta_content)
                        yield TextDeltaEvent(delta=delta_content)

        except KeyboardInterrupt:
            if self.recent_messages and self.recent_messages[-1].role == "user":
                self.recent_messages.pop()
            yield InterruptedEvent()
            return

        assistant_response = "".join(collected_chunks)
        self.recent_messages.append(Message(role="assistant", content=assistant_response))
        self.save_session()

        # Telemetry accounting
        if reported_usage:
            prompt_tokens = reported_usage.prompt_tokens
            comp_tokens = reported_usage.completion_tokens
        else:
            prompt_tokens = self.tokenizer.count_messages(active_payload)
            comp_tokens = self.tokenizer.count_text(assistant_response)

        yield UsageReportEvent(prompt_tokens=prompt_tokens, completion_tokens=comp_tokens)

    def chat(self, user_input: str) -> str | None:
        """Convenience method that runs stream_chat and returns the full assistant text."""
        chunks = []
        for event in self.stream_chat(user_input):
            if isinstance(event, TextDeltaEvent):
                chunks.append(event.delta)
            elif isinstance(event, InterruptedEvent):
                return None
            elif isinstance(event, ErrorEvent):
                raise event.error
        return "".join(chunks)
