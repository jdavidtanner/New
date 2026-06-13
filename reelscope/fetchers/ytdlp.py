"""Default fetcher: yt-dlp driven by your logged-in browser cookies.

This is the recommended free path. yt-dlp reads the session cookies from a
browser you're already logged into Instagram with, so it can read public
channels without a separate API key.

Requires the ``yt-dlp`` command on PATH (``pip install yt-dlp``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone

from ..models import ChannelData, Reel
from .base import Fetcher, FetcherError, normalize_handle, reels_url


class YtDlpFetcher(Fetcher):
    name = "yt-dlp"

    def __init__(self, cookies_from_browser: str | None = "chrome",
                 cookies_file: str | None = None, **_: object) -> None:
        # cookies_from_browser: chrome | firefox | safari | edge | brave | None
        # cookies_file: path to an exported cookies.txt (overrides browser).
        self.cookies_from_browser = cookies_from_browser
        self.cookies_file = cookies_file

    def _binary(self) -> str:
        exe = shutil.which("yt-dlp")
        if not exe:
            raise FetcherError(
                "yt-dlp is not installed. Install it with:  pip install yt-dlp"
            )
        return exe

    def _build_cmd(self, url: str, limit: int) -> list[str]:
        cmd = [
            self._binary(),
            "-J",                       # dump full metadata as one JSON blob
            "--no-warnings",
            "--ignore-errors",
            "--playlist-end", str(limit),
        ]
        if self.cookies_file:
            cmd += ["--cookies", self.cookies_file]
        elif self.cookies_from_browser:
            cmd += ["--cookies-from-browser", self.cookies_from_browser]
        cmd.append(url)
        return cmd

    def fetch(self, handle_or_url: str, limit: int = 30) -> ChannelData:
        handle = normalize_handle(handle_or_url)
        url = reels_url(handle)
        cmd = self._build_cmd(url, limit)

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600
            )
        except subprocess.TimeoutExpired as e:  # pragma: no cover - env dependent
            raise FetcherError("yt-dlp timed out after 10 minutes.") from e

        if not proc.stdout.strip():
            hint = proc.stderr.strip().splitlines()[-3:] if proc.stderr else []
            raise FetcherError(
                "yt-dlp returned no data. Common causes: not logged into the "
                "browser you pointed --cookies-from-browser at, the account is "
                "private, or Instagram changed its API.\n"
                + ("\n".join(hint) if hint else "")
            )

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise FetcherError(f"Could not parse yt-dlp output as JSON: {e}") from e

        entries = payload.get("entries") or [payload]
        reels = [self._entry_to_reel(e) for e in entries if e]
        reels = [r for r in reels if r is not None]

        return ChannelData(
            handle=handle,
            source=self.name,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            reels=reels,
            follower_count=payload.get("channel_follower_count"),
            bio=payload.get("description", "") if payload.get("_type") else "",
        )

    @staticmethod
    def _entry_to_reel(e: dict) -> Reel | None:
        url = e.get("webpage_url") or e.get("url") or ""
        if not url:
            return None
        return Reel(
            url=url,
            caption=e.get("description") or e.get("title") or "",
            view_count=e.get("view_count"),
            like_count=e.get("like_count"),
            comment_count=e.get("comment_count"),
            duration_sec=e.get("duration"),
            timestamp=e.get("timestamp"),
            title=e.get("title") or "",
            thumbnail=e.get("thumbnail") or "",
        )
