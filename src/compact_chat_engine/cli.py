from __future__ import annotations

import sys
import time
from typing import Any

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .config import ChatConfig
from .engine import ChatSession
from .models import (
    CompactionEndEvent,
    CompactionStartEvent,
    ErrorEvent,
    InterruptedEvent,
    TextDeltaEvent,
    UsageReportEvent,
)
from .storage import SqliteSessionStorage

# Configure UTF-8 encoding on Windows to ensure box and symbol rendering
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_console() -> Console:
    return Console()


def render_welcome_banner(
    console: Console, storage: SqliteSessionStorage, config: ChatConfig
) -> None:
    """Renders the branded welcome banner and existing sessions table."""
    banner_content = Text()
    banner_content.append("Compact Chat Engine\n", style="bold cyan")
    banner_content.append(
        "Tiered Tokenization & Autonomous Context Compaction\n\n", style="dim italic"
    )
    banner_content.append("Model: ", style="bold white")
    banner_content.append(f"{config.model_name}\n", style="green")
    banner_content.append("Context Limits: ", style="bold white")
    banner_content.append(
        f"Safe Limit: ~{config.safe_limit:,} tokens | Max: {config.max_context_tokens:,} tokens\n",
        style="yellow",
    )
    banner_content.append("Database: ", style="bold white")
    banner_content.append(f"{config.database_path}", style="dim")

    console.print(
        Panel(
            banner_content,
            title="[bold blue]System Initialized[/bold blue]",
            border_style="bright_blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    display_sessions_table(console, storage)


def display_sessions_table(console: Console, storage: SqliteSessionStorage) -> None:
    """Displays a formatted table of all saved sessions and their metrics."""
    sessions = storage.list_sessions_detailed()
    if not sessions:
        return

    table = Table(
        title="Stored Chat Sessions",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_header=True,
    )
    table.add_column("Session ID", style="bold green", min_width=16)
    table.add_column("Active Turns", justify="right", style="cyan")
    table.add_column("Archived Turns", justify="right", style="magenta")
    table.add_column("Scratchpad", justify="center")
    table.add_column("Last Active", style="dim")

    for s in sessions:
        updated_time = (
            time.strftime("%Y-%m-%d %H:%M", time.localtime(s["updated_at"]))
            if s["updated_at"]
            else "N/A"
        )
        has_sp = "[green]Loaded[/green]" if s["has_scratchpad"] else "[dim]Empty[/dim]"
        table.add_row(
            s["session_id"],
            str(s["active_count"]),
            str(s["archived_count"]),
            has_sp,
            updated_time,
        )

    console.print(table)
    console.print()


def render_session_header(console: Console, bot: ChatSession) -> None:
    """Displays an active session status card and available commands."""
    header = Text()
    header.append("Active Session: ", style="bold white")
    header.append(f"'{bot.session_id}'\n", style="bold cyan")

    header.append("Backend: ", style="bold white")
    header.append(f"{bot.tokenizer.backend}  |  ", style="magenta")

    header.append("Active: ", style="bold white")
    header.append(f"{len(bot.recent_messages)} turns  |  ", style="cyan")

    header.append("Archived: ", style="bold white")
    header.append(f"{len(bot.archive)} turns  |  ", style="dim")

    header.append("Memory: ", style="bold white")
    if bot.scratchpad.strip():
        header.append("Scratchpad Active\n", style="bold green")
    else:
        header.append("Scratchpad Empty\n", style="dim")

    header.append(
        "Commands: /help · /scratchpad · /tokens · /archive · /sessions · /switch · /clear · /exit",
        style="dim",
    )

    console.print(
        Panel(
            header,
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 2),
        )
    )


def display_help(console: Console) -> None:
    """Renders the command quick reference table."""
    table = Table(
        title="Compact Chat Engine - Command Reference",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Command", style="bold green", min_width=16)
    table.add_column("Description", style="white")

    commands = [
        ("/help", "Show this interactive command cheat sheet"),
        ("/scratchpad", "Render the current working memory scratchpad in Markdown"),
        ("/tokens", "Display live token count, safety buffer, and context usage breakdown"),
        ("/archive", "Inspect turn breakdown between active buffer and SQLite archive"),
        ("/sessions", "List all persistent chat sessions stored in the SQLite database"),
        ("/switch <name>", "Switch immediately to an existing session or start a new one"),
        ("/clear", "Clear recent conversation turns from the active buffer for this session"),
        ("/exit or /quit", "Persist current session state to disk and safely exit"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(table)


def display_scratchpad(console: Console, bot: ChatSession) -> None:
    """Renders the bot's working memory scratchpad in a styled Markdown panel."""
    if bot.scratchpad.strip():
        console.print(
            Panel(
                Markdown(bot.scratchpad.strip()),
                title=f"[bold cyan]Memory Scratchpad: {bot.session_id}[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )
    else:
        console.print(
            "[yellow]Working memory scratchpad is currently empty. Compaction activates automatically when context tokens cross the safety threshold.[/yellow]"
        )


def display_tokens(console: Console, bot: ChatSession) -> None:
    """Displays real-time token telemetry and context headroom breakdown."""
    active_payload = bot.get_active_payload()
    active_tokens = bot.tokenizer.count_messages(active_payload)
    max_tokens = bot.config.max_context_tokens
    safe_limit = bot.config.safe_limit
    headroom = bot.config.token_headroom
    pct = (active_tokens / max_tokens * 100) if max_tokens else 0.0

    table = Table(
        title=f"Token Telemetry - Session '{bot.session_id}'",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="cyan", justify="right")
    table.add_column("Details", style="dim")

    table.add_row("Tokenizer Backend", bot.tokenizer.backend, "Tiered counting system")
    table.add_row(
        "Active Messages",
        str(len(bot.recent_messages)),
        f"Preserved buffer: {bot.config.recent_buffer_count} turns",
    )
    table.add_row(
        "Active Context Tokens",
        f"{active_tokens:,}",
        f"{pct:.2f}% of total context capacity",
    )
    table.add_row(
        "Compaction Threshold",
        f"{safe_limit:,}",
        "Tokens crossing this boundary trigger synthesis",
    )
    table.add_row("Safety Headroom", f"{headroom:,}", "Buffer reserved for completion")
    table.add_row(
        "Max Context Tokens",
        f"{max_tokens:,}",
        f"Configured limit for {bot.config.model_name}",
    )

    console.print(table)


def display_archive(console: Console, bot: ChatSession) -> None:
    """Displays active and archived turns count and summary."""
    active_count = len(bot.recent_messages)
    archive_count = len(bot.archive)
    total = active_count + archive_count

    panel_text = Text()
    panel_text.append(f"Session '{bot.session_id}' History\n\n", style="bold white")
    panel_text.append(f"  Active Buffer:   {active_count} turns\n", style="cyan")
    panel_text.append(f"  Archived Memory: {archive_count} turns\n", style="magenta")
    panel_text.append(f"  Total Turns:     {total} turns\n", style="bold green")
    if bot.archive:
        panel_text.append(
            "\nArchived turns are synthesized inside the working memory scratchpad and stored in SQLite.",
            style="dim",
        )

    console.print(
        Panel(
            panel_text,
            title="[bold magenta]Turn Distribution[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def handle_slash_command(
    cmd_line: str, bot: ChatSession, console: Console
) -> tuple[bool, str | None]:
    """
    Evaluates in-chat slash commands.
    Returns (handled, optional_new_session_id_to_switch).
    """
    parts = cmd_line.strip().split()
    cmd = parts[0].lower()

    if cmd == "/help":
        display_help(console)
        return True, None
    elif cmd in ("/scratchpad", "/memory"):
        display_scratchpad(console, bot)
        return True, None
    elif cmd in ("/tokens", "/stats"):
        display_tokens(console, bot)
        return True, None
    elif cmd == "/archive":
        display_archive(console, bot)
        return True, None
    elif cmd == "/sessions":
        display_sessions_table(console, bot.storage)
        return True, None
    elif cmd == "/switch":
        if len(parts) < 2:
            console.print("[yellow]Usage: /switch <session_name>[/yellow]")
            return True, None
        new_session = parts[1].strip()
        if not new_session:
            console.print("[yellow]Please specify a valid session name.[/yellow]")
            return True, None
        return True, new_session
    elif cmd == "/clear":
        bot.state.recent_messages.clear()
        bot.save_session()
        console.print(
            f"[bold green][Active conversation buffer cleared for session '{bot.session_id}'.][/bold green]"
        )
        return True, None
    elif cmd in ("/exit", "/quit"):
        return False, "__EXIT__"

    console.print(f"[yellow]Unknown command '{cmd}'. Type /help for available commands.[/yellow]")
    return True, None


SLASH_COMMANDS = [
    "/help",
    "/scratchpad",
    "/memory",
    "/tokens",
    "/stats",
    "/archive",
    "/sessions",
    "/switch",
    "/clear",
    "/exit",
    "/quit",
]


def create_prompt_session() -> tuple[Any | None, Any | None]:
    """Creates a PromptSession with persistent history and tab completion."""
    try:
        import os

        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory

        history_path = os.path.expanduser("~/.compact_chat_history")
        completer = WordCompleter(SLASH_COMMANDS, ignore_case=True, sentence=True)
        return PromptSession(history=FileHistory(history_path)), completer
    except Exception:
        return None, None


def read_user_input(
    console: Console,
    prompt_session: Any | None,
    completer: Any | None,
) -> str:
    """Reads user input via prompt_toolkit when available in an interactive tty, falling back to console.input."""
    if prompt_session is not None and completer is not None and sys.stdin.isatty():
        try:
            from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
            from prompt_toolkit.formatted_text import HTML

            return prompt_session.prompt(
                HTML("<ansicyan><b>You &gt; </b></ansicyan>"),
                completer=completer,
                auto_suggest=AutoSuggestFromHistory(),
            ).strip()
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception:
            pass

    return console.input("[bold cyan]You > [/bold cyan]").strip()


def stream_turn(bot: ChatSession, user_text: str, console: Console) -> None:
    """
    Executes turn generation with a live token counter spinner,
    followed by fully rendered Markdown with tables, code blocks, and headers.
    """
    status = console.status("[dim]Thinking...[/dim]", spinner="dots")
    status.start()
    accumulated_chunks: list[str] = []
    approx_tokens = 0
    usage_info: UsageReportEvent | None = None
    interrupted = False
    has_error = False

    try:
        for event in bot.stream_chat(user_text):
            if isinstance(event, CompactionStartEvent):
                status.update(
                    status="[bold yellow]Compacting older conversation turns into scratchpad...[/bold yellow]",
                    spinner="dots",
                )
            elif isinstance(event, CompactionEndEvent):
                status.stop()
                console.print(
                    "[bold green][Memory compacted and scratchpad synthesized.][/bold green]\n"
                )
                status.update(status="[dim]Thinking...[/dim]", spinner="dots")
                status.start()
            elif isinstance(event, TextDeltaEvent):
                accumulated_chunks.append(event.delta)
                approx_tokens += max(1, len(event.delta) // 4)
                status.update(
                    status=f"[dim]Generating response... (~{approx_tokens} tokens)[/dim]",
                    spinner="dots",
                )
            elif isinstance(event, InterruptedEvent):
                interrupted = True
                status.stop()
                console.print(
                    "\n\n[bold yellow][Generation interrupted. Turn discarded cleanly.][/bold yellow]\n"
                )
            elif isinstance(event, UsageReportEvent):
                usage_info = event
            elif isinstance(event, ErrorEvent):
                has_error = True
                status.stop()
                console.print(f"\n[bold red]Error:[/bold red] {event.error}\n")

    finally:
        status.stop()

    if accumulated_chunks and not interrupted and not has_error:
        full_markdown = "".join(accumulated_chunks)
        console.print("\n[bold green]Assistant >[/bold green]\n")
        console.print(Markdown(full_markdown))
        if usage_info:
            active_payload = bot.get_active_payload()
            active_tokens = bot.tokenizer.count_messages(active_payload)
            max_tokens = bot.config.max_context_tokens
            pct = (active_tokens / max_tokens * 100) if max_tokens else 0.0
            console.print(
                f"\n[dim][Prompt: {usage_info.prompt_tokens:,} | "
                f"Completion: {usage_info.completion_tokens:,} | "
                f"Context: ~{active_tokens:,}/{max_tokens:,} ({pct:.1f}%)] [/dim]\n"
            )
        else:
            console.print()


def run_session(session_id: str = "default", console: Console | None = None) -> None:
    if console is None:
        console = get_console()

    current_session_id = session_id
    bot = ChatSession(session_id=current_session_id)
    render_session_header(console, bot)
    prompt_session, completer = create_prompt_session()

    while True:
        try:
            user_text = read_user_input(console, prompt_session, completer)
            if not user_text:
                continue

            # Check for regular exit
            if user_text.lower() in ("exit", "quit"):
                console.print(f"[bold cyan]Session '{bot.session_id}' stored. Goodbye![/bold cyan]")
                break

            # Handle slash commands
            if user_text.startswith("/"):
                handled, new_session = handle_slash_command(user_text, bot, console)
                if new_session == "__EXIT__":
                    console.print(
                        f"[bold cyan]Session '{bot.session_id}' stored. Goodbye![/bold cyan]"
                    )
                    break
                elif new_session:
                    bot.save_session()
                    current_session_id = new_session
                    bot = ChatSession(
                        session_id=current_session_id,
                        config=bot.config,
                        storage=bot.storage,
                        tokenizer=bot.tokenizer,
                    )
                    console.print(
                        f"[bold green][Switched to session '{current_session_id}'.][/bold green]"
                    )
                    render_session_header(console, bot)
                continue

            # Stream normal assistant turn
            stream_turn(bot, user_text, console)

        except (KeyboardInterrupt, EOFError):
            console.print(f"\n[bold cyan]Session '{bot.session_id}' stored. Goodbye![/bold cyan]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Unexpected error:[/bold red] {e}\n")


def main() -> None:
    console = get_console()
    config = ChatConfig.from_env()
    storage = SqliteSessionStorage(config.database_path)

    render_welcome_banner(console, storage, config)

    try:
        session_id = (
            Prompt.ask(
                "[bold cyan]Enter session name[/bold cyan]",
                default="default",
                console=console,
            ).strip()
            or "default"
        )
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold cyan]Exiting.[/bold cyan]")
        return

    run_session(session_id=session_id, console=console)


if __name__ == "__main__":
    main()
