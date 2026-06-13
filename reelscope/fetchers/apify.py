"""Apify fetcher: most robust automated source, but a paid hosted actor.

Runs the public ``apify/instagram-scraper`` actor and reads back the results.
Enable it by setting ``APIFY_TOKEN`` in your environment (or passing token=...).
No extra Python package is required — it talks to the Apify REST API directly.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from ..models import ChannelData, Reel
from .base import Fetcher, FetcherError, normalize_handle

_ACTOR = "apify~instagram-scraper"
_BASE = "https://api.apify.com/v2"


class ApifyFetcher(Fetcher):
    name = "apify"

    def __init__(self, token: str | None = None, **_: object) -> None:
        self.token = token or os.environ.get("APIFY_TOKEN")

    def fetch(self, handle_or_url: str, limit: int = 30) -> ChannelData:
        if not self.token:
            raise FetcherError(
                "Apify needs a token. Set APIFY_TOKEN in your environment "
                "(get one free at apify.com), then re-run."
            )
        handle = normalize_handle(handle_or_url)

        run_input = {
            "directUrls": [f"https://www.instagram.com/{handle}/"],
            "resultsType": "posts",
            "resultsLimit": limit,
            "onlyPostsNewerThan": "",
            "addParentData": False,
        }
        items = self._run_actor_sync(run_input)
        reels = [self._item_to_reel(it) for it in items]
        reels = [r for r in reels if r is not None][:limit]

        return ChannelData(
            handle=handle,
            source=self.name,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            reels=reels,
        )

    def _run_actor_sync(self, run_input: dict) -> list[dict]:
        # run-sync-get-dataset-items blocks until the run finishes and returns items.
        url = (
            f"{_BASE}/acts/{_ACTOR}/run-sync-get-dataset-items"
            f"?token={self.token}"
        )
        data = json.dumps(run_input).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            raise FetcherError(f"Apify run failed ({e.code}): {body[:300]}") from e
        except urllib.error.URLError as e:
            raise FetcherError(f"Could not reach Apify: {e}") from e

    @staticmethod
    def _item_to_reel(it: dict) -> Reel | None:
        url = it.get("url") or ""
        if not url:
            return None
        ts = None
        if it.get("timestamp"):
            try:
                ts = int(
                    datetime.fromisoformat(
                        it["timestamp"].replace("Z", "+00:00")
                    ).timestamp()
                )
            except (ValueError, AttributeError):
                ts = None
        return Reel(
            url=url,
            caption=it.get("caption") or "",
            view_count=it.get("videoViewCount") or it.get("videoPlayCount"),
            like_count=it.get("likesCount"),
            comment_count=it.get("commentsCount"),
            duration_sec=it.get("videoDuration"),
            timestamp=ts,
            thumbnail=it.get("displayUrl") or "",
        )
