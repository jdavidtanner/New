"""Base class + helpers shared by all fetchers."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ..models import ChannelData


class FetcherError(RuntimeError):
    """Raised when a data source fails or is misconfigured."""


_HANDLE_RE = re.compile(r"^[A-Za-z0-9_.]+$")


def normalize_handle(handle_or_url: str) -> str:
    """Accept '@name', 'name', or a full instagram.com URL -> 'name'."""
    s = handle_or_url.strip()
    # If a full/partial URL was pasted, take the first path segment after the host.
    m = re.search(r"instagram\.com/([A-Za-z0-9_.]+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # Otherwise treat it as a bare handle.
    s = s.lstrip("@").rstrip("/")
    s = re.sub(r"/(reels?|tagged|feed)/?$", "", s, flags=re.IGNORECASE)
    if not _HANDLE_RE.match(s):
        raise FetcherError(f"Could not parse an Instagram handle from: {handle_or_url!r}")
    return s


def reels_url(handle: str) -> str:
    return f"https://www.instagram.com/{handle}/reels/"


class Fetcher(ABC):
    """A source of channel reel data."""

    name: str = "base"

    @abstractmethod
    def fetch(self, handle_or_url: str, limit: int = 30) -> ChannelData:
        """Return up to ``limit`` reels for the given channel."""
        raise NotImplementedError
