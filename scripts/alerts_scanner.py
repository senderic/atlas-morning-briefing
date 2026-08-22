#!/usr/bin/env python3
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""
Public-safety alert scanner (National Weather Service).

News search cannot answer "is there a heat warning where I live today" -- it
returns whatever storm story is trending nationally. The NWS alerts API can:
it is free, needs no key, and is scoped to the exact forecast/county/fire
zones the reader lives in.

Zones are config-driven. Find yours with:
    curl 'https://api.weather.gov/points/<lat>,<lon>'
and read forecastZone / county / fireWeatherZone from the response.

Usage:
    python3 scripts/alerts_scanner.py --zones CAZ043,CAC073 --max-alerts 5
"""

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Sequence

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Most urgent first; anything unrecognized sorts last.
SEVERITY_ORDER = ["Extreme", "Severe", "Moderate", "Minor", "Unknown"]


class NWSAlertsScanner:
    """Fetches active NWS alerts for a fixed set of zones."""

    API_BASE = "https://api.weather.gov/alerts/active"

    def __init__(
        self,
        zones: Sequence[str],
        user_agent: str = "atlas-morning-briefing",
        max_alerts: int = 5,
        timeout: int = 20,
        api_base: Optional[str] = None,
    ):
        """
        Args:
            zones: NWS zone codes (forecast, county, and/or fire zones).
            user_agent: weather.gov requires a descriptive User-Agent.
            max_alerts: Maximum alerts to return.
            timeout: HTTP timeout in seconds.
            api_base: Override for the alerts endpoint (tests).
        """
        self.zones = [z.strip() for z in zones if str(z).strip()]
        self.user_agent = user_agent
        self.max_alerts = max_alerts
        self.timeout = timeout
        self.api_base = api_base or self.API_BASE

    @staticmethod
    def _severity_rank(severity: str) -> int:
        try:
            return SEVERITY_ORDER.index(severity)
        except ValueError:
            return len(SEVERITY_ORDER)

    def _parse_alert(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        props = feature.get("properties", {}) or {}
        return {
            "event": props.get("event", "") or "",
            "headline": props.get("headline", "") or "",
            "severity": props.get("severity", "Unknown") or "Unknown",
            "urgency": props.get("urgency", "") or "",
            "certainty": props.get("certainty", "") or "",
            "area": props.get("areaDesc", "") or "",
            "onset": props.get("onset") or props.get("effective") or "",
            "expires": props.get("ends") or props.get("expires") or "",
            "instruction": (props.get("instruction") or "").strip(),
            "description": (props.get("description") or "").strip(),
            "url": props.get("@id", "") or "",
            "source": "National Weather Service",
        }

    def fetch(self) -> List[Dict[str, Any]]:
        """
        Fetch active alerts, most severe first.

        Returns an empty list (never raises) when the API is unreachable, so
        the pipeline degrades gracefully like every other scanner.
        """
        if not self.zones:
            logger.info("No alert zones configured, skipping")
            return []

        try:
            response = requests.get(
                self.api_base,
                params={"zone": ",".join(self.zones)},
                headers={"User-Agent": self.user_agent, "Accept": "application/geo+json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            features = response.json().get("features", []) or []
        except (requests.RequestException, ValueError) as e:
            logger.error(f"NWS alerts fetch failed: {e}")
            raise

        alerts = [self._parse_alert(f) for f in features]
        # Drop duplicates: the same event often covers several of our zones.
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for alert in alerts:
            key = (alert["event"], alert["onset"], alert["expires"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(alert)

        deduped.sort(key=lambda a: (self._severity_rank(a["severity"]), a["onset"]))
        logger.info(
            f"Found {len(deduped)} active alerts for zones {','.join(self.zones)}"
        )
        return deduped[: self.max_alerts]


def create_scanner(config: Dict[str, Any]) -> Optional[NWSAlertsScanner]:
    """Build a scanner from the ``alerts`` config block, or None when disabled."""
    cfg = config.get("alerts") or {}
    if not cfg.get("enabled"):
        return None
    provider = cfg.get("provider", "nws")
    if provider != "nws":
        logger.warning(f"Unknown alerts provider '{provider}', skipping")
        return None
    return NWSAlertsScanner(
        zones=cfg.get("zones", []),
        user_agent=cfg.get("user_agent", "atlas-morning-briefing"),
        max_alerts=int(cfg.get("max_alerts", 5)),
        timeout=int(cfg.get("timeout", 20)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch active NWS alerts")
    parser.add_argument("--zones", required=True, help="Comma-separated NWS zone codes")
    parser.add_argument("--max-alerts", type=int, default=5)
    parser.add_argument(
        "--user-agent", default="atlas-morning-briefing (local test)"
    )
    args = parser.parse_args()

    scanner = NWSAlertsScanner(
        zones=args.zones.split(","),
        user_agent=args.user_agent,
        max_alerts=args.max_alerts,
    )
    try:
        alerts = scanner.fetch()
    except requests.RequestException as e:
        print(f"fetch failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps(alerts, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
