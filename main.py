import sys

import ollama
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from chatbot import NightWatch
from config import CHAT_MODEL, EMBED_MODEL

console = Console()

BANNER = r"""
 _   _  _       _     _    __        __    _       _     _
| \ | |(_) __ _| |__ | |_  \ \      / /_ _| |_ ___| |__ | |
|  \| || |/ _` | '_ \| __|  \ \ /\ / / _` | __/ __| '_ \| |
| |\  || | (_| | | | | |_    \ V  V / (_| | || (__| | | |_|
|_| \_||_|\__, |_| |_|\__|    \_/\_/ \__,_|\__\___|_| |_(_)
          |___/

              https://github.com/tidbocode/NightWatch-LLM
"""

HELP_TEXT = """
[bold]Commands[/bold]
  [cyan]/stats[/cyan]      Show token usage and memory statistics
  [cyan]/memories[/cyan]   List all stored vector memories
  [cyan]/clear[/cyan]      Clear conversation (vector memories persist)
  [cyan]/help[/cyan]       Show this message
  [cyan]/quit[/cyan]       Exit
"""


def check_ollama() -> bool:
    """Verify Ollama is reachable and required models are available."""
    try:
        available = {m.model for m in ollama.list().models}
    except Exception:
        console.print(
            "[red]Cannot reach Ollama.[/red] Start it with: [yellow]ollama serve[/yellow]"
        )
        return False

    missing = [m for m in (CHAT_MODEL, EMBED_MODEL) if not any(a.startswith(m) for a in available)]
    if missing:
        console.print(f"[red]Missing models:[/red] {', '.join(missing)}")
        for m in missing:
            console.print(f"  [yellow]ollama pull {m}[/yellow]")
        return False
    return True


def show_stats(bot: NightWatch) -> None:
    s = bot.stats()
    pct = s["pct_used"]
    bar_filled = pct // 5
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    color = "green" if pct < 60 else "yellow" if pct < 85 else "red"

    table = Table(box=box.ROUNDED, title="Session Stats", title_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Turns", str(s["turns"]))
    table.add_row("Messages in context", str(s["messages_in_context"]))
    table.add_row("Tokens used", f"{s['tokens_used']} / {s['token_budget']} ({pct}%)")
    table.add_row("Context bar", f"[{color}]{bar}[/{color}]")
    table.add_row("Has rolling summary", "yes" if s["has_summary"] else "no")
    table.add_row("Vector memories total", str(s["vector_memories"]))
    table.add_row("Memories retrieved (last)", str(s["last_memories_retrieved"]))
    table.add_row("Facts stored (last)", str(s["last_facts_stored"]))

    console.print(table)


def show_memories(bot: NightWatch) -> None:
    memories = bot.list_memories()
    if not memories:
        console.print("[dim]No memories stored yet.[/dim]")
        return
    lines = "\n".join(f"[dim]{i + 1}.[/dim] {m}" for i, m in enumerate(memories))
    console.print(Panel(lines, title=f"[bold cyan]Vector Memories ({len(memories)})[/bold cyan]", border_style="cyan"))


def main() -> None:
    console.print(f"[bold cyan]{BANNER}[/bold cyan]")
    console.print("[dim]Checking Ollama...[/dim]")

    if not check_ollama():
        sys.exit(1)

    console.print("[dim]Loading NightWatch...[/dim]")
    try:
        bot = NightWatch()
    except Exception as exc:
        console.print(f"[red]Init failed:[/red] {exc}")
        sys.exit(1)

    console.print("[green]Ready.[/green] Type [cyan]/help[/cyan] for commands.\n")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        text = user_input.strip()
        if not text:
            continue

        cmd = text.lower()
        if cmd in ("/quit", "/exit"):
            console.print("[dim]Goodbye.[/dim]")
            break
        elif cmd == "/stats":
            show_stats(bot)
        elif cmd == "/memories":
            show_memories(bot)
        elif cmd == "/clear":
            bot.clear()
            console.print("[green]Conversation cleared.[/green]")
        elif cmd == "/help":
            console.print(HELP_TEXT)
        else:
            console.print("[bold magenta]NightWatch[/bold magenta]: ", end="")
            try:
                for token in bot.chat(text):
                    console.print(token, end="")
            except Exception as exc:
                console.print(f"\n[red]Error:[/red] {exc}")
            console.print()


if __name__ == "__main__":
    main()
