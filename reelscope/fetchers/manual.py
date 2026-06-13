"""Manual fetcher: load reels you've gathered yourself.

Zero dependencies, works forever. Use this when automated scraping is blocked
or you just want to analyze a hand-picked set of reels. Accepts either CSV or
JSON.

CSV columns (header row required, only ``url`` is mandatory)::

    url,caption,view_count,like_count,comment_count,duration_sec,timestamp

``timestamp`` may be a unix int or an ISO date like ``2026-01-31``.

JSON: either a list of reel objects, or a full ChannelData dump.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

from ..models import ChannelData, Reel
from .base import Fetcher, FetcherError, normalize_handle


def _to_int(v: object) -> int | None:
    if v in (None, "", "-"):
        return None
    try:
        return int(float(str(v).replace(",", "").strip()))
    except ValueError:
        return None


def _to_ts(v: object) -> int | None:
    if v in (None, "", "-"):
        return None
    s = str(v).strip()
    if s.isdigit():
        return int(s)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


class ManualFetcher(Fetcher):
    name = "manual"

    def __init__(self, path: str | None = None, **_: object) -> None:
        self.path = path

    def fetch(self, handle_or_url: str, limit: int = 30) -> ChannelData:
        if not self.path:
            raise FetcherError(
                "manual fetcher needs a file. Pass --input path/to/reels.csv (or .json)."
            )
        handle = normalize_handle(handle_or_url) if handle_or_url else "manual"

        if self.path.lower().endswith(".json"):
            reels = self._from_json()
        else:
            reels = self._from_csv()

        if not reels:
            raise FetcherError(f"No reels found in {self.path}.")

        return ChannelData(
            handle=handle,
            source=self.name,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            reels=reels[:limit],
        )

    def _from_csv(self) -> list[Reel]:
        reels: list[Reel] = []
        with open(self.path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "url" not in reader.fieldnames:
                raise FetcherError("CSV must have a header row including a 'url' column.")
            for row in reader:
                url = (row.get("url") or "").strip()
                if not url:
                    continue
                reels.append(
                    Reel(
                        url=url,
                        caption=(row.get("caption") or "").strip(),
                        view_count=_to_int(row.get("view_count")),
                        like_count=_to_int(row.get("like_count")),
                        comment_count=_to_int(row.get("comment_count")),
                        duration_sec=_to_int(row.get("duration_sec")),
                        timestamp=_to_ts(row.get("timestamp")),
                    )
                )
        return reels

    def _from_json(self) -> list[Reel]:
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "reels" in data:
            data = data["reels"]
        if not isinstance(data, list):
            raise FetcherError("JSON must be a list of reels or a ChannelData object.")
        return [Reel.from_dict(d) for d in data if d.get("url")]
