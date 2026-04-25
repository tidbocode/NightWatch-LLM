import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from models.alert import Alert, Severity, SEVERITY_RANK


_DDL = """
CREATE TABLE IF NOT EXISTS alerts (
    id              TEXT PRIMARY KEY,
    severity        TEXT NOT NULL,
    title           TEXT NOT NULL,
    description     TEXT,
    recommendation  TEXT,
    iocs            TEXT,           -- JSON array
    affected_lines  TEXT,           -- JSON array
    log_format      TEXT,
    source_file     TEXT,
    chunk_index     INTEGER,
    timestamp_first TEXT,
    timestamp_last  TEXT,
    generated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS iocs (
    value       TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    alert_count INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_alerts_severity    ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_source      ON alerts(source_file);
CREATE INDEX IF NOT EXISTS idx_alerts_generated   ON alerts(generated_at);
CREATE INDEX IF NOT EXISTS idx_iocs_alert_count   ON iocs(alert_count DESC);
"""


def _classify_ioc(value: str) -> str:
    if re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", value):
        return "ip"
    if re.match(r"^[a-fA-F0-9]{32,64}$", value):
        return "hash"
    if "@" in value and "." in value:
        return "email"
    if value.startswith("/") or re.match(r"^[A-Za-z]:\\", value):
        return "path"
    return "string"


class AlertStore:

    def __init__(self, db_path: str):
        self.db_path = db_path
        # In-memory databases are destroyed when the connection closes, so we
        # keep a single persistent connection for :memory: mode (used in tests).
        self._shared_conn: sqlite3.Connection | None = None
        if db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(self, alert: Alert) -> None:
        now = alert.generated_at.isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alerts
                    (id, severity, title, description, recommendation,
                     iocs, affected_lines, log_format, source_file,
                     chunk_index, timestamp_first, timestamp_last, generated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    alert.id,
                    alert.severity.value,
                    alert.title,
                    alert.description,
                    alert.recommendation,
                    json.dumps(alert.iocs),
                    json.dumps(alert.affected_lines),
                    alert.log_format,
                    alert.source_file,
                    alert.chunk_index,
                    alert.timestamp_first.isoformat() if alert.timestamp_first else None,
                    alert.timestamp_last.isoformat() if alert.timestamp_last else None,
                    now,
                ),
            )
            for ioc in alert.iocs:
                conn.execute(
                    """
                    INSERT INTO iocs (value, type, first_seen, last_seen, alert_count)
                    VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(value) DO UPDATE SET
                        last_seen   = excluded.last_seen,
                        alert_count = iocs.alert_count + 1
                    """,
                    (ioc, _classify_ioc(ioc), now, now),
                )

    def clear(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM alerts")
            conn.execute("DELETE FROM iocs")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    def severity_counts(self) -> dict[str, int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT severity, COUNT(*) FROM alerts GROUP BY severity"
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def query_by_severity(self, min_severity: str) -> list[dict]:
        min_rank = SEVERITY_RANK.get(min_severity, 99)
        qualifying = [s for s, r in SEVERITY_RANK.items() if r <= min_rank]
        placeholders = ",".join("?" * len(qualifying))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM alerts WHERE severity IN ({placeholders})"
                " ORDER BY generated_at DESC",
                qualifying,
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def query_by_ip(self, ip: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE iocs LIKE ? ORDER BY generated_at DESC",
                (f'%"{ip}"%',),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def query_by_source(self, source_file: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts WHERE source_file = ? ORDER BY generated_at DESC",
                (source_file,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def query_recent(self, n: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY generated_at DESC LIMIT ?", (n,)
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def top_iocs(self, n: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM iocs ORDER BY alert_count DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(row) for row in rows]

    def ioc_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM iocs").fetchone()[0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        if self._shared_conn is not None:
            try:
                yield self._shared_conn
                self._shared_conn.commit()
            except Exception:
                self._shared_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["iocs"] = json.loads(d["iocs"] or "[]")
        d["affected_lines"] = json.loads(d["affected_lines"] or "[]")
        return d
