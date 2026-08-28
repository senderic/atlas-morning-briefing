# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
"""Tests for the NWS public-safety alert scanner (scripts/alerts_scanner.py)."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from scripts.alerts_scanner import NWSAlertsScanner, create_scanner


def _feature(event="Heat Advisory", severity="Moderate", onset="2026-08-22T10:00:00-07:00",
             expires="2026-08-24T20:00:00-07:00", area="San Diego County Coastal Areas"):
    return {
        "properties": {
            "event": event,
            "headline": f"{event} issued August 21",
            "severity": severity,
            "urgency": "Expected",
            "certainty": "Likely",
            "areaDesc": area,
            "onset": onset,
            "ends": expires,
            "instruction": "Drink plenty of fluids and stay out of the sun.",
            "description": "A long NWS description.",
            "@id": "https://api.weather.gov/alerts/urn:oid:1.2.3",
        }
    }


def _response(features):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"features": features}
    resp.raise_for_status.return_value = None
    return resp


class TestFetch:
    def test_no_zones_returns_empty(self):
        assert NWSAlertsScanner(zones=[]).fetch() == []

    def test_parses_alert_fields(self):
        with patch("scripts.alerts_scanner.requests.get", return_value=_response([_feature()])):
            alerts = NWSAlertsScanner(zones=["CAZ043"]).fetch()
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["event"] == "Heat Advisory"
        assert alert["severity"] == "Moderate"
        assert alert["area"] == "San Diego County Coastal Areas"
        assert alert["instruction"].startswith("Drink plenty")
        assert alert["source"] == "National Weather Service"

    def test_zones_are_joined_into_one_request(self):
        with patch("scripts.alerts_scanner.requests.get", return_value=_response([])) as get:
            NWSAlertsScanner(zones=["CAZ043", " CAZ243 "]).fetch()
        assert get.call_args.kwargs["params"] == {"zone": "CAZ043,CAZ243"}

    def test_sends_user_agent(self):
        with patch("scripts.alerts_scanner.requests.get", return_value=_response([])) as get:
            NWSAlertsScanner(zones=["CAZ043"], user_agent="atlas (me@example.com)").fetch()
        assert get.call_args.kwargs["headers"]["User-Agent"] == "atlas (me@example.com)"

    def test_sorts_most_severe_first(self):
        features = [
            _feature(event="Heat Advisory", severity="Moderate"),
            _feature(event="Extreme Heat Warning", severity="Severe"),
            _feature(event="Beach Hazards", severity="Minor"),
        ]
        with patch("scripts.alerts_scanner.requests.get", return_value=_response(features)):
            alerts = NWSAlertsScanner(zones=["CAZ043"]).fetch()
        assert [a["event"] for a in alerts] == [
            "Extreme Heat Warning", "Heat Advisory", "Beach Hazards",
        ]

    def test_dedupes_same_event_across_zones(self):
        features = [_feature(), _feature(area="San Diego County Coastal Areas (fire)")]
        with patch("scripts.alerts_scanner.requests.get", return_value=_response(features)):
            alerts = NWSAlertsScanner(zones=["CAZ043", "CAZ243"]).fetch()
        assert len(alerts) == 1

    def test_respects_max_alerts(self):
        features = [_feature(event=f"Event {i}", onset=f"2026-08-2{i}T10:00:00-07:00")
                    for i in range(5)]
        with patch("scripts.alerts_scanner.requests.get", return_value=_response(features)):
            alerts = NWSAlertsScanner(zones=["CAZ043"], max_alerts=2).fetch()
        assert len(alerts) == 2

    def test_network_error_propagates_for_caller_to_log(self):
        with patch("scripts.alerts_scanner.requests.get",
                   side_effect=requests.ConnectionError("boom")):
            with pytest.raises(requests.RequestException):
                NWSAlertsScanner(zones=["CAZ043"]).fetch()

    def test_unknown_severity_sorts_last(self):
        features = [_feature(event="Mystery", severity="Bogus"),
                    _feature(event="Heat Advisory", severity="Moderate")]
        with patch("scripts.alerts_scanner.requests.get", return_value=_response(features)):
            alerts = NWSAlertsScanner(zones=["CAZ043"]).fetch()
        assert [a["event"] for a in alerts] == ["Heat Advisory", "Mystery"]


class TestCreateScanner:
    def test_returns_none_when_absent_or_disabled(self):
        assert create_scanner({}) is None
        assert create_scanner({"alerts": {"enabled": False}}) is None

    def test_returns_none_for_unknown_provider(self):
        assert create_scanner({"alerts": {"enabled": True, "provider": "acme"}}) is None

    def test_builds_from_config(self):
        scanner = create_scanner({
            "alerts": {
                "enabled": True,
                "zones": ["CAZ043", "CAZ243"],
                "max_alerts": 3,
                "user_agent": "atlas (me@example.com)",
            }
        })
        assert scanner is not None
        assert scanner.zones == ["CAZ043", "CAZ243"]
        assert scanner.max_alerts == 3
