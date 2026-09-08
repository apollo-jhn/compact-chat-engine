from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class ChatConfig:
    api_key: str = "YOUR_ORCAROUTER_API_KEY_HERE"
    base_url: str = "https://api.orcarouter.ai/v1"
    model_name: str = "z-ai/glm-5.3-flash-free"
    hf_tokenizer_repo: str = "THUDM/glm-4-9b-chat"
    base_system_prompt: str = "You are a helpful and intelligent AI assistant."
    max_context_tokens: int = 250_000
    token_headroom: int = 8_000
    recent_buffer_count: int = 4
    compaction_temperature: float = 0.2
    database_path: str = "sessions.db"

    @property
    def safe_limit(self) -> int:
        return self.max_context_tokens - self.token_headroom

    @classmethod
    def from_env(cls) -> ChatConfig:
        load_dotenv()
        return cls(
            api_key=os.environ.get("ORCAROUTER_API_KEY", "YOUR_ORCAROUTER_API_KEY_HERE"),
            base_url=os.environ.get("ORCAROUTER_BASE_URL")
            or os.environ.get("BASE_URL", "https://api.orcarouter.ai/v1"),
            model_name=os.environ.get("MODEL_NAME", "z-ai/glm-5.3-flash-free"),
            hf_tokenizer_repo=os.environ.get("HF_TOKENIZER_REPO", "THUDM/glm-4-9b-chat"),
            base_system_prompt=os.environ.get(
                "BASE_SYSTEM_PROMPT", "You are a helpful and intelligent AI assistant."
            ),
            max_context_tokens=int(os.environ.get("MAX_CONTEXT_TOKENS", "250000")),
            token_headroom=int(os.environ.get("TOKEN_HEADROOM", "8000")),
            recent_buffer_count=int(os.environ.get("RECENT_BUFFER_COUNT", "4")),
            compaction_temperature=float(os.environ.get("COMPACTION_TEMPERATURE", "0.2")),
            database_path=os.environ.get("DATABASE_PATH", "sessions.db"),
        )
