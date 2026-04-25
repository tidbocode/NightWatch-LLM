from dataclasses import dataclass, field
from datetime import datetime

from models.alert import Severity


@dataclass
class AnalysisSession:
    source_file: str
    log_format: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    total_lines: int = 0
    total_chunks: int = 0
    alerts_by_severity: dict[str, int] = field(
        default_factory=lambda: {s.value: 0 for s in Severity}
    )

    @property
    def total_alerts(self) -> int:
        return sum(self.alerts_by_severity.values())

    def record_alert(self, severity: str) -> None:
        self.alerts_by_severity[severity] = self.alerts_by_severity.get(severity, 0) + 1

    def summary_line(self) -> str:
        parts = [
            f"{count} {sev.lower()}"
            for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")
            if (count := self.alerts_by_severity.get(sev, 0)) > 0
        ]
        findings = ", ".join(parts) if parts else "no alerts"
        elapsed = (datetime.utcnow() - self.start_time).seconds
        return (
            f"{self.source_file} | {self.log_format} | "
            f"{self.total_lines} lines | {self.total_chunks} chunks | "
            f"{findings} | {elapsed}s"
        )
