# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Prerequisites & setup

Ollama must be running locally (`ollama serve`) with both models pulled:

```bash
ollama pull mistral:7b
ollama pull llama3.2
pip install -r requirements.txt
```

## Running

```bash
python main.py batch --file /var/log/auth.log
python main.py watch --file /var/log/nginx/access.log
python main.py query
```

`main.py` calls `check_ollama()` on startup, which verifies Ollama is reachable and the required model is present before proceeding.

## Architecture

NightWatch is a local LLM log analyzer for threat detection. `main.py` is the CLI entry point with three subcommands: `batch`, `watch`, `query`.

### Per-analysis flow

1. **`FormatDetector.detect_format()`** — samples the first 20 lines and scores each parser's confidence. Returns the best-fit `LogFormat`.
2. **`LogParser.parse_lines()`** — streams `LogEntry` objects from the file. Never raises; malformed lines return `format=UNKNOWN`.
3. **`ThreatAnalyzer.analyze_stream()`** — consumes `LogEntry` objects and yields `Alert` objects:
   - Groups entries into chunks capped at `CHUNK_TOKEN_BUDGET` tokens
   - Builds a layered prompt: system prompt → rolling context summary → log chunk
   - Streams response from Ollama; parses JSON → `list[Alert]`
   - Persists each `Alert` to SQLite via `AlertStore`
   - Updates the rolling summary for the next chunk

### Key design decisions

- **JSON output from LLM** — the system prompt requires JSON-only responses. `_parse_response()` strips markdown fences, scans for the first `{`, and uses `raw_decode()` to tolerate trailing prose (mistral:7b sometimes adds it).
- **SQLite over ChromaDB** — structured queries (by IP, severity, time, source file) fit log analysis better than cosine similarity. `AlertStore` uses a single persistent connection for `:memory:` mode (used in tests).
- **Two models** — `mistral:7b` (CHAT_MODEL) for batch analysis where accuracy matters; `llama3.2` (FAST_MODEL) for watch mode where speed matters.

### File map

```
main.py                  CLI — batch / watch / query subcommands
analyzer.py              ThreatAnalyzer — chunking, prompting, parsing, persistence
config.py                All constants

models/
  log_entry.py           LogEntry dataclass + LogFormat enum
  alert.py               Alert dataclass + Severity enum + SEVERITY_RANK

parsers/
  base.py                LogParser ABC + FormatDetector
  syslog.py              BSD syslog, RFC 5424, systemd/journald
  clf.py                 Nginx / Apache Combined Log Format
  json_log.py            Generic JSON logs (handles many key-name variants)
  windows_csv.py         Windows Event Log CSV (Event Viewer + PowerShell)

memory/
  alert_store.py         SQLite persistence — alerts + IOC tracking
  session.py             AnalysisSession — per-run metadata and counters

utils/
  token_budget.py        TokenBudget — 1 token ≈ 4 chars heuristic

tests/
  test_parsers.py        46 tests covering all four parsers + FormatDetector
  test_analyzer.py       24 tests covering chunking, JSON parsing, persistence
  samples/auth.log       Sample syslog file for smoke testing
```

### Key config knobs (`config.py`)

| Variable | Default | Effect |
|---|---|---|
| `CHAT_MODEL` | `mistral:7b` | Model for batch analysis |
| `FAST_MODEL` | `llama3.2` | Model for watch mode |
| `CHUNK_TOKEN_BUDGET` | `1500` | Token cap per log chunk |
| `MIN_SEVERITY` | `LOW` | Default display threshold |
| `MAX_AFFECTED_LINES` | `10` | Max raw lines stored per Alert |
| `ALERT_DB_PATH` | `./nightwatch.db` | SQLite database path |
