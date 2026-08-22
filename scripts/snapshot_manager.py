"""Snapshot manager for saving raw fetched data into dated snapshot directories."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SnapshotManager:
    """Saves raw fetched data into dated snapshot directories."""

    def __init__(self, snapshot_dir: str = "snapshots", enabled: bool = True):
        self.enabled = enabled
        self.base_dir = Path(snapshot_dir)
        self.date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.run_dir = self.base_dir / self.date_str
        self.saved_files: List[Path] = []

    def _write(self, name: str, data: Any) -> Optional[Path]:
        if not self.enabled:
            return None
        if not data:
            logger.info(f"snapshot_skipped source={name} reason=no_data")
            return None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / name
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            self.saved_files.append(path)
            logger.info(f"snapshot_saved path={path} size={path.stat().st_size}")
            return path
        except (IOError, TypeError) as e:
            logger.warning(f"snapshot_failed name={name} error={e}")
            return None

    def save_stocks(self, stocks: List[Dict[str, Any]]):
        self._write("finnhub_data.json", stocks)

    def save_news(self, news: List[Dict[str, Any]]):
        self._write("brave_news.json", news)

    def save_happenings(self, happenings: List[Dict[str, Any]]):
        self._write("happenings.json", happenings)

    def save_alerts(self, alerts: List[Dict[str, Any]]):
        self._write("alerts.json", alerts)

    def save_blogs(self, blogs: List[Dict[str, Any]]):
        self._write("rss_feeds.json", blogs)

    def save_papers(self, papers: List[Dict[str, Any]]):
        self._write("arxiv_papers.json", papers)

    def save_manifest(self):
        manifest = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "date": self.date_str,
            "files": [str(p.relative_to(self.base_dir)) for p in self.saved_files],
            "file_count": len(self.saved_files),
        }
        self._write("snapshot_manifest.json", manifest)
