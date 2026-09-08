# Compact Chat Engine

A lightweight, robust AI chat engine featuring working memory synthesis, tiered tokenization, and SQLite session persistence.

## Features
- **Working Memory & Scratchpad**: Automatically compacts historical conversation turns when context thresholds are crossed.
- **Tiered Tokenizer**: Hugging Face GLM tokenizer (Tier 1), `tiktoken` (Tier 2 fast fallback), and heuristic safeguard (Tier 3).
- **SQLite Persistence**: Stores session state, active turns, and archived history in a relational SQLite database with automatic migration from legacy JSON files.
- **Decoupled Architecture**: Generator-based streaming events (`ChatEvent`) decoupled from the CLI interface.

## Quickstart

```bash
# Run via CLI script
uv run compact-chat

# Or run via main.py
uv run python main.py
```
