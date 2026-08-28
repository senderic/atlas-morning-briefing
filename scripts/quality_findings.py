#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Shared vocabulary for the daily quality check.

Every layer of the quality check (source health, report invariants, the LLM
judge) reports the same ``Finding`` type, so the orchestrator can sort, route,
and deduplicate findings without knowing which layer produced them.

See references/quality_monitoring_design.md for the design this implements.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Severity ladder. CRITICAL means the reader got a broken, wrong, or missing
# briefing and someone should look now. WARN means something degraded but the
# briefing still shipped. INFO is worth knowing and needs no action.
CRITICAL = "CRITICAL"
WARN = "WARN"
INFO = "INFO"

SEVERITY_ORDER = [CRITICAL, WARN, INFO]


@dataclass
class Finding:
    """One thing the quality check noticed."""

    severity: str
    code: str
    message: str
    source: str = ""
    pipeline: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def dedupe_key(self) -> str:
        """
        Stable identity for a recurring finding.

        A dead feed should page once and then sit in the digest as a standing
        item; alert history is keyed on this.
        """
        return f"{self.pipeline}:{self.code}:{self.source}".strip(":")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "pipeline": self.pipeline,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Finding":
        return cls(
            severity=data.get("severity", INFO),
            code=data.get("code", ""),
            message=data.get("message", ""),
            source=data.get("source", ""),
            pipeline=data.get("pipeline", ""),
            detail=data.get("detail", {}) or {},
        )


def severity_rank(severity: str) -> int:
    """Sort key: CRITICAL first, unknown severities last."""
    try:
        return SEVERITY_ORDER.index(severity)
    except ValueError:
        return len(SEVERITY_ORDER)


def sort_findings(findings: List[Finding]) -> List[Finding]:
    """Most severe first, then by code, then by source. Stable and total."""
    return sorted(
        findings, key=lambda f: (severity_rank(f.severity), f.code, f.source)
    )


def worst_severity(findings: List[Finding]) -> Optional[str]:
    """The most severe severity present, or None for an empty list."""
    if not findings:
        return None
    return sort_findings(findings)[0].severity


def counts_by_severity(findings: List[Finding]) -> Dict[str, int]:
    """Count per severity, including zeros, in ladder order."""
    return {s: sum(1 for f in findings if f.severity == s) for s in SEVERITY_ORDER}
