from __future__ import annotations

import math
from typing import Any


class TokenizerManager:
    """
    Tiered token counting:
      1. Hugging Face GLM tokenizer (exact match)
      2. tiktoken cl100k_base with +5% margin (fast fallback)
      3. Character heuristic (~4 chars/token safeguard)
    """

    def __init__(self, repo_id: str = "THUDM/glm-4-9b-chat"):
        self.repo_id = repo_id
        self.hf_tokenizer = None
        self.tiktoken_enc = None
        self.backend = "heuristic"

        # Tier 1: Attempt Hugging Face AutoTokenizer (cache-first, suppressed warnings)
        try:
            import io
            import warnings
            from contextlib import redirect_stderr, redirect_stdout

            f_err = io.StringIO()
            f_out = io.StringIO()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with redirect_stderr(f_err), redirect_stdout(f_out):
                    from transformers import AutoTokenizer, logging as hf_logging

                    hf_logging.set_verbosity_error()

                    # Step 1: Attempt local cache first to avoid network checks & Hub warnings
                    try:
                        self.hf_tokenizer = AutoTokenizer.from_pretrained(
                            repo_id, trust_remote_code=True, local_files_only=True
                        )
                    except Exception:
                        # Step 2: Attempt network download if not cached
                        self.hf_tokenizer = AutoTokenizer.from_pretrained(
                            repo_id, trust_remote_code=True, local_files_only=False
                        )
            self.backend = "huggingface"
        except Exception:
            # Tier 2: Attempt tiktoken
            try:
                import tiktoken

                self.tiktoken_enc = tiktoken.get_encoding("cl100k_base")
                self.backend = "tiktoken"
            except Exception:
                self.backend = "heuristic"

    def count_text(self, text: str) -> int:
        if not text:
            return 0

        if self.backend == "huggingface" and self.hf_tokenizer:
            try:
                return len(self.hf_tokenizer.encode(text))
            except Exception:
                pass

        if self.backend == "tiktoken" and self.tiktoken_enc:
            try:
                tokens = len(self.tiktoken_enc.encode(text))
                return math.ceil(tokens * 1.05)  # 5% safety margin
            except Exception:
                pass

        return max(1, len(text) // 4)

    def count_messages(self, messages: list[dict | Any]) -> int:
        total = 0
        for msg in messages:
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            total += self.count_text(content) + 4  # Turn overhead
        return total + 3  # Assistant reply priming overhead
