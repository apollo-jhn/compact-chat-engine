# Compact Chat Engine

Compact Chat Engine is a Python library and command line tool for long AI conversations. When a conversation has many messages, large language models can run out of context space. This engine solves that problem. It automatically summarizes older messages into a compact working memory scratchpad and keeps recent messages intact. It also stores your chat sessions in a local SQLite database and monitors token usage carefully.

## Features

- Working Memory Scratchpad: Summarizes older conversation turns into structured notes when token limits are reached.
- Tiered Token Counting: Uses a three-level system to count tokens accurately and quickly. It can use Hugging Face tokenizers, tiktoken, or a character-based fallback.
- SQLite Database Storage: Saves chat sessions, active messages, and archived messages in a local SQLite database. It can also import old JSON session files automatically.
- Event Streaming: Delivers streaming responses through clean Python events, so you can build your own user interfaces or use the built-in terminal interface.

## Project Structure

```text
compact-chat-engine/
├── src/
│   └── compact_chat_engine/
│       ├── __init__.py       # Package exports
│       ├── cli.py            # Terminal chat interface
│       ├── config.py         # Configuration settings and environment variables
│       ├── engine.py         # Main ChatSession logic and compaction cycle
│       ├── models.py         # Data models and stream events
│       ├── storage.py        # SQLite database operations and migration
│       └── tokenizer.py      # Tiered token counter
├── tests/                    # Automated test suite
├── .env.example              # Example environment settings
├── main.py                   # Simple entry point script
├── pyproject.toml            # Project metadata and dependencies
└── README.md                 # Project documentation
```

## Requirements

- Python 3.12 or higher
- An OrcaRouter API key

## Installation

You can install and run this project using uv or standard pip.

### Option 1: Using uv (Recommended)

First, clone the repository and enter the directory:

```bash
git clone https://github.com/your-username/compact-chat-engine.git
cd compact-chat-engine
```

Install dependencies:

```bash
uv sync
```

If you want exact token counting with Hugging Face transformers, install the optional `hf` group:

```bash
uv sync --extra hf
```

### Option 2: Using pip and a Virtual Environment

Create and activate a virtual environment:

```bash
# On Linux and macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the package in editable mode:

```bash
pip install -e .
```

To install the optional Hugging Face tokenizer support:

```bash
pip install -e ".[hf]"
```

## Configuration

The project reads settings from a `.env` file in the root folder. Copy the example configuration to get started:

```bash
# On Linux and macOS:
cp .env.example .env

# On Windows (PowerShell):
Copy-Item .env.example .env
```

Open the `.env` file and set your OrcaRouter credentials. Here is the list of available options:

| Variable | Default Value | Description |
| --- | --- | --- |
| `ORCAROUTER_API_KEY` | `your_orcarouter_api_key_here` | Your secret API key for OrcaRouter. |
| `ORCAROUTER_BASE_URL` | `https://api.orcarouter.ai/v1` | The base URL for the OrcaRouter API. |
| `MODEL_NAME` | `z-ai/glm-5.3-flash-free` | The language model used for conversation and memory compaction. |
| `HF_TOKENIZER_REPO` | `THUDM/glm-4-9b-chat` | The Hugging Face model repository used for Tier 1 token counting. |
| `BASE_SYSTEM_PROMPT` | `You are a helpful and intelligent AI assistant.` | The base instruction given to the model at the start of every chat. |
| `MAX_CONTEXT_TOKENS` | `250000` | The total context token limit for the model. |
| `TOKEN_HEADROOM` | `8000` | Safety token buffer. Compaction triggers when tokens cross safe limit. |
| `RECENT_BUFFER_COUNT` | `4` | Number of recent messages kept active and uncompacted. |
| `COMPACTION_TEMPERATURE` | `0.2` | Temperature setting used when the model writes memory summaries. |
| `DATABASE_PATH` | `sessions.db` | File path where SQLite stores chat history. |

## Quickstart (CLI)

You can start an interactive chat session in your terminal with either `uv` or standard Python:

```bash
# Run with uv
uv run compact-chat

# Or run the main script directly
uv run python main.py
```

If you installed the package with `pip` inside an active virtual environment, you can run:

```bash
compact-chat
# or
python main.py
```

### CLI Features

- Custom Sessions: When the program starts, enter a name to load an existing session or start a new one. Press Enter to use the default session.
- Safe Interrupts: Press `Ctrl+C` while the model is typing to stop generation safely without losing your session.
- Exit: Type `exit` or `quit` to save your session and exit the program.

## Python API Usage

You can also use `compact-chat-engine` directly inside your own Python programs.

### Streaming Responses

Streaming is the recommended way to interact with the engine. It emits events as text arrives from the model:

```python
from compact_chat_engine import ChatSession
from compact_chat_engine.models import (
    CompactionStartEvent,
    CompactionEndEvent,
    TextDeltaEvent,
    UsageReportEvent,
    ErrorEvent,
)

# Initialize a chat session
session = ChatSession(session_id="my-first-session")

# Stream a conversation turn
user_input = "Hello! Can you explain how solar panels work?"

for event in session.stream_chat(user_input):
    if isinstance(event, CompactionStartEvent):
        print("\n[System: Compacting older memory...]")
    elif isinstance(event, CompactionEndEvent):
        print("[System: Memory compacted successfully.]\n")
    elif isinstance(event, TextDeltaEvent):
        print(event.delta, end="", flush=True)
    elif isinstance(event, UsageReportEvent):
        print(f"\n\n[Tokens: {event.prompt_tokens} prompt, {event.completion_tokens} completion]")
    elif isinstance(event, ErrorEvent):
        print(f"\n[Error: {event.error}]")
```

### Simple Synchronous Call

If you do not need real-time streaming, use the `chat` helper method to get the full reply string:

```python
from compact_chat_engine import ChatSession

session = ChatSession(session_id="simple-session")
response = session.chat("What are three main benefits of exercise?")
print(response)
```

## How Memory Compaction Works

When you have long conversations, sending every past message wastes tokens and eventually exceeds the model context window. Compact Chat Engine handles this automatically:

1. Token Monitoring: Before each turn, the engine counts the tokens of the system prompt and active messages.
2. Threshold Check: If the total token count exceeds the safe limit (`MAX_CONTEXT_TOKENS - TOKEN_HEADROOM`), compaction begins.
3. Message Eviction: The engine separates messages into two groups. The most recent messages (defined by `RECENT_BUFFER_COUNT`) stay active. The older messages are selected for compaction.
4. Memory Synthesis: The engine sends the older messages and the current scratchpad to the model. The model updates a structured markdown scratchpad with three sections:
   - Core Facts and Decisions
   - User Preferences
   - In-Progress Tasks
5. Context Update: The new scratchpad is attached to the system prompt. The compacted messages move to the archive in SQLite.
6. Persistence: The updated scratchpad, active messages, and archived messages are saved to `sessions.db`.

## Tiered Tokenization

The engine uses a tiered tokenizer to measure tokens safely under different environments:

- Tier 1 (Hugging Face): If `transformers` is installed, it loads the model tokenizer for exact token counts.
- Tier 2 (tiktoken): If `transformers` is not installed, it falls back to OpenAI `tiktoken` with a 5% safety margin.
- Tier 3 (Character Heuristic): If neither package is available, it estimates tokens using an average of 4 characters per token.

## Running Tests

The project includes an automated test suite written with `pytest`. You can run all tests using `uv`:

```bash
uv run pytest
```

If you are using a standard virtual environment with `pip`, run:

```bash
pytest
```
