from __future__ import annotations

import sys
from .engine import ChatSession
from .models import (
    CompactionEndEvent,
    CompactionStartEvent,
    ErrorEvent,
    InterruptedEvent,
    TextDeltaEvent,
    UsageReportEvent,
)


def run_session(session_id: str = "default") -> None:
    bot = ChatSession(session_id=session_id)

    print(f"\n--- Session '{session_id}' Active (Backend: {bot.tokenizer.backend}) ---")
    print(
        f"Memory: {len(bot.archive)} archived turns | {len(bot.recent_messages)} active turns"
    )
    if bot.scratchpad:
        print("Scratchpad loaded.")
    print("Type 'exit' or 'quit' to close. Press Ctrl+C mid-stream to halt output safely.\n")

    while True:
        try:
            user_text = input("User: ").strip()
            if not user_text:
                continue
            if user_text.lower() in ("exit", "quit"):
                print("Session stored. Exiting.")
                break

            assistant_started = False
            for event in bot.stream_chat(user_text):
                if isinstance(event, CompactionStartEvent):
                    print("[Compacting context scratchpad...]", flush=True)
                elif isinstance(event, CompactionEndEvent):
                    print("[Compaction complete.]\n", flush=True)
                elif isinstance(event, TextDeltaEvent):
                    if not assistant_started:
                        print("\nAssistant: ", end="", flush=True)
                        assistant_started = True
                    sys.stdout.write(event.delta)
                    sys.stdout.flush()
                elif isinstance(event, InterruptedEvent):
                    print("\n\n[Generation interrupted. Turn discarded cleanly.]\n", flush=True)
                elif isinstance(event, UsageReportEvent):
                    if assistant_started:
                        print()
                    print(
                        f"[Usage: ~{event.prompt_tokens} prompt | ~{event.completion_tokens} completion]\n",
                        flush=True,
                    )
                elif isinstance(event, ErrorEvent):
                    print(f"\n[Error] {event.error}\n", flush=True)

        except KeyboardInterrupt:
            print("\nSession stored. Exiting.")
            break
        except Exception as e:
            print(f"\n[Error] {e}\n")


def main() -> None:
    try:
        session_id = (
            input("Enter session name (press Enter for 'default'): ").strip()
            or "default"
        )
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return

    run_session(session_id=session_id)


if __name__ == "__main__":
    main()
