"""
Tests for ThreatAnalyzer. Ollama is mocked; AlertStore uses an in-memory SQLite DB.
"""
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from analyzer import ThreatAnalyzer, _MAX_SUMMARY_CHARS
from models.alert import Severity
from models.log_entry import LogEntry, LogFormat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(raw: str, ts: datetime | None = None, ip: str | None = None) -> LogEntry:
    return LogEntry(
        raw=raw, format=LogFormat.SYSLOG, message=raw,
        timestamp=ts, source_ip=ip,
    )


def _stream(text: str) -> list[MagicMock]:
    """Simulate ollama streaming a single text response as three chunks."""
    third = len(text) // 3 or 1
    chunks = [text[:third], text[third:2*third], text[2*third:]]
    return [MagicMock(message=MagicMock(content=c)) for c in chunks]


def _json_response(alerts: list[dict], summary: str = "test chunk") -> list[MagicMock]:
    payload = json.dumps({"alerts": alerts, "chunk_summary": summary})
    return _stream(payload)


def _make_analyzer() -> ThreatAnalyzer:
    """ThreatAnalyzer backed by in-memory SQLite."""
    return ThreatAnalyzer(db_path=":memory:")


# ---------------------------------------------------------------------------
# analyze_stream — chunking behaviour
# ---------------------------------------------------------------------------

class TestChunking:

    def test_single_chunk_for_small_input(self):
        az = _make_analyzer()
        entries = [_entry("Jan  5 12:34:56 host sshd[1]: Failed password")]

        with patch("analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = iter(_json_response([]))
            list(az.analyze_stream(iter(entries), "auth.log"))

        assert mock_ollama.chat.call_count == 1

    def test_large_entries_split_into_multiple_chunks(self):
        az = _make_analyzer()
        # Each entry is ~400 chars → ~100 tokens; budget is 1500 → expect splits
        big_line = "A" * 400
        entries = [_entry(f"Jan  5 12:34:56 host sshd[1]: {big_line} {i}") for i in range(20)]

        with patch("analyzer.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = lambda **kw: iter(_json_response([]))
            list(az.analyze_stream(iter(entries), "auth.log"))

        assert mock_ollama.chat.call_count > 1

    def test_empty_stream_produces_no_alerts(self):
        az = _make_analyzer()
        with patch("analyzer.ollama"):
            alerts = list(az.analyze_stream(iter([]), "auth.log"))
        assert alerts == []

    def test_chunk_index_increments(self):
        az = _make_analyzer()
        big_line = "B" * 400
        entries = [_entry(f"Jan  5 12:34:56 host sshd[1]: {big_line} {i}") for i in range(20)]

        with patch("analyzer.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = lambda **kw: iter(_json_response([]))
            list(az.analyze_stream(iter(entries)))

        assert az._chunk_index == mock_ollama.chat.call_count


# ---------------------------------------------------------------------------
# _parse_response — JSON parsing
# ---------------------------------------------------------------------------

class TestParseResponse:

    def setup_method(self):
        self.az = _make_analyzer()
        self.entries = [_entry("test line")]

    def test_valid_json_produces_alerts(self):
        raw = json.dumps({"alerts": [
            {
                "severity": "HIGH",
                "title": "SSH Brute Force",
                "description": "Many failed logins",
                "recommendation": "Block IP",
                "iocs": ["192.168.1.1"],
                "affected_lines": ["test line"],
            }
        ], "chunk_summary": "brute force detected"})

        alerts, summary = self.az._parse_response(raw, self.entries, "auth.log")

        assert len(alerts) == 1
        assert alerts[0].severity == Severity.HIGH
        assert alerts[0].title == "SSH Brute Force"
        assert "192.168.1.1" in alerts[0].iocs
        assert summary == "brute force detected"

    def test_markdown_fences_stripped(self):
        raw = "```json\n" + json.dumps({"alerts": [], "chunk_summary": "ok"}) + "\n```"
        alerts, _ = self.az._parse_response(raw, self.entries, "auth.log")
        assert alerts == []

    def test_invalid_json_returns_error_alert(self):
        alerts, summary = self.az._parse_response("not json at all", self.entries, "auth.log")
        assert len(alerts) == 1
        assert alerts[0].severity == Severity.INFO
        assert "not valid JSON" in alerts[0].description
        assert summary == ""

    def test_empty_alerts_array(self):
        raw = json.dumps({"alerts": [], "chunk_summary": "nothing suspicious"})
        alerts, summary = self.az._parse_response(raw, self.entries, "auth.log")
        assert alerts == []
        assert summary == "nothing suspicious"

    def test_invalid_severity_defaults_to_info(self):
        raw = json.dumps({"alerts": [
            {"severity": "ULTRA_HIGH", "title": "t", "description": "d",
             "recommendation": "r", "iocs": [], "affected_lines": []}
        ], "chunk_summary": ""})
        alerts, _ = self.az._parse_response(raw, self.entries, "auth.log")
        assert alerts[0].severity == Severity.INFO

    def test_all_severity_levels_parsed(self):
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
            raw = json.dumps({"alerts": [
                {"severity": sev, "title": "t", "description": "d",
                 "recommendation": "r", "iocs": [], "affected_lines": []}
            ], "chunk_summary": ""})
            alerts, _ = self.az._parse_response(raw, self.entries, "auth.log")
            assert alerts[0].severity == Severity(sev)

    def test_affected_lines_capped(self):
        from config import MAX_AFFECTED_LINES
        raw_lines = [f"line {i}" for i in range(MAX_AFFECTED_LINES + 5)]
        raw = json.dumps({"alerts": [
            {"severity": "LOW", "title": "t", "description": "d",
             "recommendation": "r", "iocs": [], "affected_lines": raw_lines}
        ], "chunk_summary": ""})
        alerts, _ = self.az._parse_response(raw, self.entries, "auth.log")
        assert len(alerts[0].affected_lines) == MAX_AFFECTED_LINES

    def test_timestamps_from_entries(self):
        ts1 = datetime(2024, 1, 5, 10, 0, 0)
        ts2 = datetime(2024, 1, 5, 12, 0, 0)
        entries = [_entry("a", ts=ts1), _entry("b", ts=ts2)]
        raw = json.dumps({"alerts": [
            {"severity": "LOW", "title": "t", "description": "d",
             "recommendation": "r", "iocs": [], "affected_lines": []}
        ], "chunk_summary": ""})
        alerts, _ = self.az._parse_response(raw, entries, "auth.log")
        assert alerts[0].timestamp_first == ts1
        assert alerts[0].timestamp_last == ts2

    def test_source_file_set_on_alert(self):
        raw = json.dumps({"alerts": [
            {"severity": "LOW", "title": "t", "description": "d",
             "recommendation": "r", "iocs": [], "affected_lines": []}
        ], "chunk_summary": ""})
        alerts, _ = self.az._parse_response(raw, self.entries, "/var/log/auth.log")
        assert alerts[0].source_file == "/var/log/auth.log"


# ---------------------------------------------------------------------------
# analyze_stream — end-to-end with mocked Ollama
# ---------------------------------------------------------------------------

class TestAnalyzeStreamE2E:

    def test_alerts_yielded(self):
        az = _make_analyzer()
        entries = [_entry("Jan  5 12:34:56 host sshd[1]: Failed password for root from 1.2.3.4")]

        response = _json_response([{
            "severity": "HIGH", "title": "Brute Force", "description": "many failures",
            "recommendation": "block", "iocs": ["1.2.3.4"], "affected_lines": ["line"],
        }])

        with patch("analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = iter(response)
            alerts = list(az.analyze_stream(iter(entries), "auth.log"))

        assert len(alerts) == 1
        assert alerts[0].severity == Severity.HIGH
        assert alerts[0].title == "Brute Force"

    def test_alerts_stored_in_db(self):
        az = _make_analyzer()
        entries = [_entry("Jan  5 12:34:56 host sshd[1]: msg")]

        response = _json_response([{
            "severity": "MEDIUM", "title": "Suspicious", "description": "x",
            "recommendation": "y", "iocs": ["10.0.0.1"], "affected_lines": [],
        }])

        with patch("analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = iter(response)
            list(az.analyze_stream(iter(entries)))

        assert az.alert_store.count() == 1
        assert az.alert_store.ioc_count() == 1

    def test_ollama_error_yields_error_alert(self):
        az = _make_analyzer()
        entries = [_entry("some log line")]

        with patch("analyzer.ollama") as mock_ollama:
            mock_ollama.chat.side_effect = ConnectionError("Ollama unreachable")
            alerts = list(az.analyze_stream(iter(entries)))

        assert len(alerts) == 1
        assert "Ollama error" in alerts[0].description

    def test_multiple_alerts_per_chunk(self):
        az = _make_analyzer()
        entries = [_entry(f"line {i}") for i in range(3)]

        response = _json_response([
            {"severity": "HIGH", "title": "A", "description": "x", "recommendation": "y",
             "iocs": [], "affected_lines": []},
            {"severity": "LOW",  "title": "B", "description": "x", "recommendation": "y",
             "iocs": [], "affected_lines": []},
        ])

        with patch("analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = iter(response)
            alerts = list(az.analyze_stream(iter(entries)))

        assert len(alerts) == 2


# ---------------------------------------------------------------------------
# Rolling summary
# ---------------------------------------------------------------------------

class TestRollingSummary:

    def test_summary_set_after_first_chunk_with_alerts(self):
        az = _make_analyzer()
        assert az._rolling_summary is None

        entries = [_entry("line")]
        response = _json_response([
            {"severity": "HIGH", "title": "Test", "description": "d",
             "recommendation": "r", "iocs": [], "affected_lines": []},
        ], summary="test summary")

        with patch("analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = iter(response)
            list(az.analyze_stream(iter(entries)))

        assert az._rolling_summary is not None
        assert "HIGH" in az._rolling_summary
        assert "Test" in az._rolling_summary

    def test_summary_trimmed_when_too_long(self):
        az = _make_analyzer()
        az._rolling_summary = "x" * (_MAX_SUMMARY_CHARS + 500)
        az._update_summary([], "some chunk summary")
        assert len(az._rolling_summary) <= _MAX_SUMMARY_CHARS + 10  # +10 for "...\n" prefix

    def test_summary_injected_into_subsequent_prompts(self):
        az = _make_analyzer()
        az._rolling_summary = "previous findings"

        messages = az._build_messages("log chunk text")

        system_contents = [m["content"] for m in messages if m["role"] == "system"]
        assert any("previous findings" in c for c in system_contents)

    def test_no_summary_when_alerts_and_summary_both_empty(self):
        az = _make_analyzer()
        entries = [_entry("line")]

        with patch("analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = iter(_json_response([], summary=""))
            list(az.analyze_stream(iter(entries)))

        assert az._rolling_summary is None


# ---------------------------------------------------------------------------
# AlertStore integration
# ---------------------------------------------------------------------------

class TestAlertStore:

    def test_query_by_severity(self):
        az = _make_analyzer()
        entries = [_entry("line")]

        response = _json_response([
            {"severity": "CRITICAL", "title": "C", "description": "d",
             "recommendation": "r", "iocs": ["1.1.1.1"], "affected_lines": []},
            {"severity": "LOW", "title": "L", "description": "d",
             "recommendation": "r", "iocs": [], "affected_lines": []},
        ])

        with patch("analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = iter(response)
            list(az.analyze_stream(iter(entries)))

        critical_only = az.alert_store.query_by_severity("CRITICAL")
        assert len(critical_only) == 1
        assert critical_only[0]["severity"] == "CRITICAL"

        high_and_up = az.alert_store.query_by_severity("HIGH")
        assert len(high_and_up) == 1

        all_alerts = az.alert_store.query_by_severity("INFO")
        assert len(all_alerts) == 2

    def test_query_by_ip(self):
        az = _make_analyzer()
        entries = [_entry("line")]

        response = _json_response([
            {"severity": "HIGH", "title": "T", "description": "d",
             "recommendation": "r", "iocs": ["192.168.1.99"], "affected_lines": []},
        ])

        with patch("analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = iter(response)
            list(az.analyze_stream(iter(entries)))

        hits = az.alert_store.query_by_ip("192.168.1.99")
        assert len(hits) == 1
        assert "192.168.1.99" in hits[0]["iocs"]

        no_hits = az.alert_store.query_by_ip("10.0.0.1")
        assert no_hits == []

    def test_stats_reflect_stored_alerts(self):
        az = _make_analyzer()
        entries = [_entry("line")]

        response = _json_response([
            {"severity": "HIGH", "title": "T", "description": "d",
             "recommendation": "r", "iocs": ["1.2.3.4", "admin"], "affected_lines": []},
        ])

        with patch("analyzer.ollama") as mock_ollama:
            mock_ollama.chat.return_value = iter(response)
            list(az.analyze_stream(iter(entries)))

        s = az.stats()
        assert s["total_alerts"] == 1
        assert s["by_severity"].get("HIGH") == 1
        assert s["ioc_count"] == 2
