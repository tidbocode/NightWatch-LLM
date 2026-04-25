import json
import re
from typing import Iterator

import ollama

from config import (
    ALERT_DB_PATH,
    CHAT_MODEL,
    CHUNK_TOKEN_BUDGET,
    FAST_MODEL,
    MAX_AFFECTED_LINES,
)
from memory.alert_store import AlertStore
from memory.session import AnalysisSession
from models.alert import Alert, Severity
from models.log_entry import LogEntry
from utils.token_budget import TokenBudget

_SYSTEM_PROMPT = """\
You are NightWatch, a cybersecurity log analysis engine running entirely locally.
Analyze the log entries provided and identify security threats, anomalies, and suspicious patterns.

You MUST respond with ONLY valid JSON — no explanation, no markdown fences, no extra text.

Response schema:
{
  "alerts": [
    {
      "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
      "title": "Short title (max 60 chars)",
      "description": "What is happening and why it is suspicious",
      "recommendation": "Concrete remediation step",
      "iocs": ["ip", "username", "path", "hash", ...],
      "affected_lines": ["exact raw log line 1", ...]
    }
  ],
  "chunk_summary": "One sentence describing the overall activity in this batch"
}

Severity guide:
  CRITICAL — Active exploitation, confirmed breach, RCE, ransomware
  HIGH     — Brute force, SQLi/XSS attempts, port scanning, privilege escalation
  MEDIUM   — Repeated failures, suspicious user agents, unusual access patterns
  LOW      — Single failed login, minor anomaly
  INFO     — Notable but non-threatening event

Threat categories to detect (not exhaustive):
  - Brute force / credential stuffing (many failed logins from one source)
  - SQL injection, XSS, path traversal, command injection in URLs or fields
  - Remote code execution attempts
  - Privilege escalation (sudo, su, unusual group changes)
  - Lateral movement (unusual internal connections)
  - Port / service scanning
  - Suspicious user agents (scanners, exploit frameworks)
  - Malware / C2 beacon patterns (beaconing intervals, known C2 paths)
  - Data exfiltration (large outbound transfers, unusual destinations)
  - Account manipulation (new users, group changes, permission changes)

If no threats are found return: {"alerts": [], "chunk_summary": "..."}"""

# Rolling summary length cap — older context is trimmed to stay within budget
_MAX_SUMMARY_CHARS = 1500


class ThreatAnalyzer:
    """
    Orchestrates log chunk analysis:
      1. Chunk log entries to fit the token budget
      2. Build a layered prompt (system → rolling context → log chunk)
      3. Stream response from the local LLM
      4. Parse JSON → Alert objects
      5. Persist each alert to SQLite
      6. Update the rolling session summary
    """

    def __init__(self, db_path: str = ALERT_DB_PATH, fast_mode: bool = False):
        self.model = FAST_MODEL if fast_mode else CHAT_MODEL
        self.alert_store = AlertStore(db_path)
        self._rolling_summary: str | None = None
        self._chunk_index: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_stream(
        self,
        entries: Iterator[LogEntry],
        source_file: str = "",
    ) -> Iterator[Alert]:
        """
        Consume an iterator of LogEntry objects, yield Alert objects as each
        chunk is analyzed. Fully lazy — suitable for large or live log streams.
        """
        session = AnalysisSession(
            source_file=source_file,
            log_format="",  # set on first entry
        )

        chunk: list[LogEntry] = []
        chunk_tokens = 0

        for entry in entries:
            session.total_lines += 1
            if not session.log_format and entry.format.value != "unknown":
                session.log_format = entry.format.value

            entry_tokens = TokenBudget.estimate(entry.raw)

            if chunk_tokens + entry_tokens > CHUNK_TOKEN_BUDGET and chunk:
                alerts = self._analyze_chunk(chunk, source_file)
                session.total_chunks += 1
                for alert in alerts:
                    session.record_alert(alert.severity.value)
                    yield alert
                chunk = []
                chunk_tokens = 0

            chunk.append(entry)
            chunk_tokens += entry_tokens

        # Flush the final partial chunk
        if chunk:
            alerts = self._analyze_chunk(chunk, source_file)
            session.total_chunks += 1
            for alert in alerts:
                session.record_alert(alert.severity.value)
                yield alert

    def stats(self) -> dict:
        return {
            "total_alerts": self.alert_store.count(),
            "by_severity": self.alert_store.severity_counts(),
            "ioc_count": self.alert_store.ioc_count(),
            "chunks_analyzed": self._chunk_index,
        }

    # ------------------------------------------------------------------
    # Chunk analysis
    # ------------------------------------------------------------------

    def _analyze_chunk(self, entries: list[LogEntry], source_file: str) -> list[Alert]:
        chunk_text = "\n".join(e.raw for e in entries)
        messages = self._build_messages(chunk_text)

        full_response = ""
        try:
            for part in ollama.chat(model=self.model, messages=messages, stream=True):
                full_response += part.message.content
        except Exception as exc:
            return [self._error_alert(f"Ollama error: {exc}", entries, source_file)]

        alerts, chunk_summary = self._parse_response(full_response, entries, source_file)

        for alert in alerts:
            self.alert_store.store(alert)

        self._update_summary(alerts, chunk_summary)
        self._chunk_index += 1

        return alerts

    def _build_messages(self, chunk_text: str) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if self._rolling_summary:
            messages.append({
                "role": "system",
                "content": f"[Context from previously analyzed batches]\n{self._rolling_summary}",
            })

        messages.append({
            "role": "user",
            "content": f"Analyze these log entries for threats:\n\n{chunk_text}",
        })
        return messages

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        raw: str,
        entries: list[LogEntry],
        source_file: str,
    ) -> tuple[list[Alert], str]:
        # Strip markdown fences and whitespace
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        # Scan for the first { in case the LLM prefixed prose before the JSON
        brace = cleaned.find("{")
        if brace > 0:
            cleaned = cleaned[brace:]

        data = None
        try:
            # raw_decode stops at the end of the first valid JSON object,
            # so trailing prose or newlines after the closing } are ignored
            data, _ = json.JSONDecoder().raw_decode(cleaned)
        except json.JSONDecodeError:
            pass

        if data is None:
            return [self._error_alert(
                f"LLM response was not valid JSON: {raw[:300]}",
                entries,
                source_file,
            )], ""

        chunk_summary: str = data.get("chunk_summary", "")
        alerts: list[Alert] = []

        timestamps = [e.timestamp for e in entries if e.timestamp]
        ts_first = min(timestamps) if timestamps else None
        ts_last  = max(timestamps) if timestamps else None
        log_format = entries[0].format.value if entries else "unknown"

        for item in data.get("alerts", []):
            try:
                severity = Severity(item.get("severity", "INFO").upper())
            except ValueError:
                severity = Severity.INFO

            alerts.append(Alert(
                severity=severity,
                title=str(item.get("title", "Unknown threat"))[:80],
                description=str(item.get("description", "")),
                recommendation=str(item.get("recommendation", "")),
                iocs=list(item.get("iocs", [])),
                affected_lines=list(item.get("affected_lines", []))[:MAX_AFFECTED_LINES],
                log_format=log_format,
                chunk_index=self._chunk_index,
                timestamp_first=ts_first,
                timestamp_last=ts_last,
                source_file=source_file,
            ))

        return alerts, chunk_summary

    # ------------------------------------------------------------------
    # Rolling summary
    # ------------------------------------------------------------------

    def _update_summary(self, alerts: list[Alert], chunk_summary: str) -> None:
        if not alerts and not chunk_summary:
            return

        if alerts:
            findings = "; ".join(f"{a.severity.value}: {a.title}" for a in alerts)
            line = f"[chunk {self._chunk_index}] {findings}"
        else:
            line = f"[chunk {self._chunk_index}] {chunk_summary}"

        self._rolling_summary = (
            f"{self._rolling_summary}\n{line}" if self._rolling_summary else line
        )

        if len(self._rolling_summary) > _MAX_SUMMARY_CHARS:
            self._rolling_summary = "...\n" + self._rolling_summary[-_MAX_SUMMARY_CHARS:]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _error_alert(self, description: str, entries: list[LogEntry], source_file: str) -> Alert:
        return Alert(
            severity=Severity.INFO,
            title="Analysis error",
            description=description,
            recommendation="Check that Ollama is running and the model is available.",
            iocs=[],
            affected_lines=[e.raw for e in entries[:MAX_AFFECTED_LINES]],
            log_format=entries[0].format.value if entries else "unknown",
            chunk_index=self._chunk_index,
            source_file=source_file,
        )
