import argparse
import json
import sys
import time
from pathlib import Path

import ollama
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from analyzer import ThreatAnalyzer
from config import ALERT_DB_PATH, CHAT_MODEL, FAST_MODEL
from memory.alert_store import AlertStore
from models.alert import Alert, Severity
from models.log_entry import LogFormat
from parsers.base import FormatDetector

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

_SEV_COLOR = {
    "CRITICAL": "red",
    "HIGH":     "yellow",
    "MEDIUM":   "dark_orange",
    "LOW":      "blue",
    "INFO":     "dim",
}

_QUERY_HELP = """
[bold]Commands[/bold]
  [cyan]/findings[/cyan]              Show 20 most recent alerts
  [cyan]/top-iocs[/cyan]              Show most frequently seen IOCs
  [cyan]/stats[/cyan]                 Database summary
  [cyan]/severity HIGH[/cyan]         Filter alerts by minimum severity
  [cyan]/ip 1.2.3.4[/cyan]            Find alerts involving an IP address
  [cyan]/source /path/to/file[/cyan]  Find alerts from a specific log file
  [cyan]/clear[/cyan]                 Delete all stored alerts and IOCs
  [cyan]/help[/cyan]                  Show this message
  [cyan]/quit[/cyan]                  Exit

  [dim]Or type any text to search alert titles and descriptions.[/dim]
"""

# Lines buffered before sending to analyzer in watch mode
_WATCH_BUFFER_SIZE = 50


# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

def check_ollama(fast: bool = False) -> bool:
    model = FAST_MODEL if fast else CHAT_MODEL
    try:
        available = {m.model for m in ollama.list().models}
    except Exception:
        console.print(
            "[red]Cannot reach Ollama.[/red] "
            "Start it with: [yellow]ollama serve[/yellow]"
        )
        return False

    if not any(a.startswith(model) for a in available):
        console.print(f"[red]Model not found:[/red] {model}")
        console.print(f"  [yellow]ollama pull {model}[/yellow]")
        return False
    return True


# ---------------------------------------------------------------------------
# Alert rendering
# ---------------------------------------------------------------------------

def render_alert(alert: Alert, min_severity: str = "LOW", show_lines: bool = True) -> None:
    if not alert.meets_minimum(min_severity):
        return

    color = _SEV_COLOR.get(alert.severity.value, "white")
    parts = [alert.description]

    if alert.iocs:
        parts += ["", "[dim]IOCs[/dim]"]
        parts += [f"  • {ioc}" for ioc in alert.iocs]

    parts += ["", f"[dim]Recommendation:[/dim] {alert.recommendation}"]

    if show_lines and alert.affected_lines:
        parts += ["", "[dim]Log lines:[/dim]"]
        for line in alert.affected_lines[:3]:
            parts.append(f"  [dim]{line[:120]}[/dim]")
        if len(alert.affected_lines) > 3:
            parts.append(f"  [dim]... {len(alert.affected_lines) - 3} more[/dim]")

    ts = f" · {alert.timestamp_first}" if alert.timestamp_first else ""
    console.print(Panel(
        "\n".join(parts),
        title=f"[{color}][{alert.severity.value}] {alert.title}[/{color}]{ts}",
        border_style=color,
        expand=False,
    ))


def _render_alert_row(row: dict) -> None:
    color = _SEV_COLOR.get(row["severity"], "white")
    iocs = row.get("iocs", [])
    ioc_str = ", ".join(iocs[:3]) + ("…" if len(iocs) > 3 else "")
    ts = str(row.get("generated_at", ""))[:19]
    console.print(
        f"[{color}]{row['severity']:10}[/{color}]"
        f"[bold]{row['title']}[/bold]\n"
        f"           [dim]{str(row.get('description', ''))[:100]}[/dim]\n"
        f"           [dim]IOCs: {ioc_str or 'none'} · {ts}[/dim]"
    )


def print_summary(alerts: list[Alert], analyzer: ThreatAnalyzer) -> None:
    console.print(Rule("[bold]Analysis Complete[/bold]"))
    stats = analyzer.stats()
    counts = stats["by_severity"]

    if not any(counts.values()):
        console.print("[green]No alerts found.[/green]")
    else:
        table = Table(box=box.ROUNDED, title="Alert Summary", title_style="bold cyan")
        table.add_column("Severity", style="bold")
        table.add_column("Count", justify="right")
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
            n = counts.get(sev, 0)
            if n:
                c = _SEV_COLOR[sev]
                table.add_row(f"[{c}]{sev}[/{c}]", str(n))
        console.print(table)

    console.print(
        f"[dim]Total stored: {stats['total_alerts']} alerts · "
        f"{stats['chunks_analyzed']} chunks · "
        f"{stats['ioc_count']} unique IOCs[/dim]"
    )


# ---------------------------------------------------------------------------
# Batch command
# ---------------------------------------------------------------------------

def cmd_batch(args) -> None:
    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f"[red]File not found:[/red] {args.file}")
        sys.exit(1)

    if not check_ollama(fast=args.fast):
        sys.exit(1)

    if args.format == "auto":
        fmt = FormatDetector.detect_format(str(file_path))
        console.print(f"[dim]Detected format:[/dim] [cyan]{fmt.value}[/cyan]")
    else:
        fmt = LogFormat(args.format)

    parser = FormatDetector.get_parser(fmt)
    analyzer = ThreatAnalyzer(db_path=args.db, fast_mode=args.fast)
    model_label = FAST_MODEL if args.fast else CHAT_MODEL

    console.print(Rule(
        f"[bold]{file_path.name}[/bold] · format: [cyan]{fmt.value}[/cyan] · "
        f"model: [cyan]{model_label}[/cyan]"
    ))

    all_alerts: list[Alert] = []
    try:
        with file_path.open(encoding="utf-8", errors="replace") as f:
            for alert in analyzer.analyze_stream(
                parser.parse_lines(f), source_file=str(file_path)
            ):
                render_alert(alert, args.min_severity)
                all_alerts.append(alert)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — partial results below.[/yellow]")

    print_summary(all_alerts, analyzer)

    if args.output:
        _write_json(all_alerts, args.output)
        console.print(f"[dim]Alerts written to {args.output}[/dim]")


# ---------------------------------------------------------------------------
# Watch command
# ---------------------------------------------------------------------------

def cmd_watch(args) -> None:
    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f"[red]File not found:[/red] {args.file}")
        sys.exit(1)

    if not check_ollama(fast=True):
        sys.exit(1)

    if args.format == "auto":
        fmt = FormatDetector.detect_format(str(file_path))
    else:
        fmt = LogFormat(args.format)

    log_parser = FormatDetector.get_parser(fmt)
    analyzer = ThreatAnalyzer(db_path=args.db, fast_mode=True)

    console.print(Rule(
        f"[bold]Watching[/bold] {file_path.name} · "
        f"format: [cyan]{fmt.value}[/cyan] · "
        f"model: [cyan]{FAST_MODEL}[/cyan] · "
        f"[dim]Ctrl-C to stop[/dim]"
    ))

    buffer = []
    try:
        with file_path.open(encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)  # start at end — only analyze new lines
            while True:
                line = f.readline()
                if line:
                    buffer.append(log_parser.parse_line(line.rstrip("\n")))
                    if len(buffer) >= _WATCH_BUFFER_SIZE:
                        _flush_buffer(buffer, analyzer, args.min_severity)
                        buffer = []
                else:
                    if buffer:
                        _flush_buffer(buffer, analyzer, args.min_severity)
                        buffer = []
                    time.sleep(args.interval)
    except KeyboardInterrupt:
        if buffer:
            _flush_buffer(buffer, analyzer, args.min_severity)
        console.print("\n[dim]Watch stopped.[/dim]")


def _flush_buffer(buffer: list, analyzer: ThreatAnalyzer, min_severity: str) -> None:
    for alert in analyzer.analyze_stream(iter(buffer)):
        render_alert(alert, min_severity)


# ---------------------------------------------------------------------------
# Query command
# ---------------------------------------------------------------------------

def cmd_query(args) -> None:
    store = AlertStore(args.db)
    console.print(Rule("[bold]NightWatch Query Mode[/bold]"))
    console.print("[dim]Type /help for commands, or search text to find alerts.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]nightwatch[/bold cyan]")
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
        elif cmd == "/help":
            console.print(_QUERY_HELP)
        elif cmd == "/stats":
            _show_db_stats(store)
        elif cmd == "/findings":
            _show_alerts(store.query_recent(20), "Recent Alerts")
        elif cmd == "/top-iocs":
            _show_iocs(store)
        elif cmd.startswith("/severity "):
            sev = text.split(" ", 1)[1].strip().upper()
            _show_alerts(store.query_by_severity(sev), f"Alerts ≥ {sev}")
        elif cmd.startswith("/ip "):
            ip = text.split(" ", 1)[1].strip()
            _show_alerts(store.query_by_ip(ip), f"Alerts for {ip}")
        elif cmd.startswith("/source "):
            src = text.split(" ", 1)[1].strip()
            _show_alerts(store.query_by_source(src), f"Alerts from {src}")
        elif cmd == "/clear":
            _confirm_clear(store)
        else:
            _show_alerts(store.query_by_text(text), f'Search: "{text}"')


def _show_alerts(alerts: list[dict], title: str) -> None:
    if not alerts:
        console.print("[dim]No alerts found.[/dim]\n")
        return
    console.print(f"\n[bold cyan]{title}[/bold cyan] ({len(alerts)})\n")
    for row in alerts:
        _render_alert_row(row)
        console.print()


def _show_db_stats(store: AlertStore) -> None:
    counts = store.severity_counts()
    table = Table(box=box.SIMPLE, title="Database Stats", title_style="bold cyan")
    table.add_column("Severity", style="bold")
    table.add_column("Count", justify="right")
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
        n = counts.get(sev, 0)
        if n:
            c = _SEV_COLOR[sev]
            table.add_row(f"[{c}]{sev}[/{c}]", str(n))
    console.print(table)
    console.print(
        f"Total alerts: [bold]{store.count()}[/bold] · "
        f"Unique IOCs: [bold]{store.ioc_count()}[/bold]\n"
    )


def _show_iocs(store: AlertStore) -> None:
    iocs = store.top_iocs(20)
    if not iocs:
        console.print("[dim]No IOCs stored yet.[/dim]\n")
        return
    table = Table(box=box.SIMPLE, title="Top IOCs by Alert Count", title_style="bold cyan")
    table.add_column("Value")
    table.add_column("Type", style="dim")
    table.add_column("Alerts", justify="right")
    for row in iocs:
        table.add_row(str(row["value"]), str(row["type"]), str(row["alert_count"]))
    console.print(table)


def _confirm_clear(store: AlertStore) -> None:
    confirm = Prompt.ask("[yellow]Delete all stored alerts and IOCs?[/yellow] (yes/no)")
    if confirm.strip().lower() == "yes":
        store.clear()
        console.print("[green]Database cleared.[/green]")
    else:
        console.print("[dim]Cancelled.[/dim]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(alerts: list[Alert], output_path: str) -> None:
    data = [
        {
            "id": a.id,
            "severity": a.severity.value,
            "title": a.title,
            "description": a.description,
            "recommendation": a.recommendation,
            "iocs": a.iocs,
            "log_format": a.log_format,
            "source_file": a.source_file,
            "chunk_index": a.chunk_index,
            "timestamp_first": a.timestamp_first.isoformat() if a.timestamp_first else None,
            "timestamp_last": a.timestamp_last.isoformat() if a.timestamp_last else None,
            "generated_at": a.generated_at.isoformat(),
        }
        for a in alerts
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nightwatch",
        description="NightWatch — Local LLM Log Analyzer for Threat Detection",
    )
    p.add_argument(
        "--db", default=ALERT_DB_PATH, metavar="PATH",
        help=f"Alert database path (default: {ALERT_DB_PATH})",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # -- batch ---------------------------------------------------------------
    b = sub.add_parser("batch", help="Analyze a log file")
    b.add_argument("--file", "-f", required=True, metavar="PATH", help="Log file to analyze")
    b.add_argument(
        "--format", default="auto",
        choices=["auto", "syslog", "clf", "json", "windows_csv"],
        help="Log format (default: auto-detect)",
    )
    b.add_argument("--output", "-o", metavar="PATH", help="Write alerts to a JSON file")
    b.add_argument(
        "--min-severity", dest="min_severity", default="LOW",
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
        help="Minimum severity to display (default: LOW)",
    )
    b.add_argument(
        "--fast", action="store_true",
        help=f"Use {FAST_MODEL} for faster (less accurate) analysis",
    )

    # -- watch ---------------------------------------------------------------
    w = sub.add_parser("watch", help="Tail a log file and alert on new entries")
    w.add_argument("--file", "-f", required=True, metavar="PATH", help="Log file to monitor")
    w.add_argument(
        "--format", default="auto",
        choices=["auto", "syslog", "clf", "json", "windows_csv"],
        help="Log format (default: auto-detect)",
    )
    w.add_argument(
        "--interval", type=float, default=5.0, metavar="SECS",
        help="Seconds to wait between tail checks when idle (default: 5)",
    )
    w.add_argument(
        "--min-severity", dest="min_severity", default="MEDIUM",
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"],
        help="Minimum severity to display (default: MEDIUM)",
    )

    # -- query ---------------------------------------------------------------
    sub.add_parser("query", help="Interactively query stored alerts")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    console.print(f"[bold cyan]{BANNER}[/bold cyan]")
    args = build_parser().parse_args()

    if args.command == "batch":
        cmd_batch(args)
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "query":
        cmd_query(args)


if __name__ == "__main__":
    main()
