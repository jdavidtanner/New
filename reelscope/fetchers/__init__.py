"""Pluggable data sources for pulling a channel's reels.

Every fetcher returns a ``ChannelData`` so the analyze/brief stages don't care
where the data came from. Pick a source with ``get_fetcher(name)``.
"""

from .base import Fetcher, FetcherError
from .ytdlp import YtDlpFetcher
from .instaloader_fetcher import InstaloaderFetcher
from .manual import ManualFetcher
from .apify import ApifyFetcher

_REGISTRY: dict[str, type[Fetcher]] = {
    "yt-dlp": YtDlpFetcher,
    "ytdlp": YtDlpFetcher,
    "instaloader": InstaloaderFetcher,
    "manual": ManualFetcher,
    "apify": ApifyFetcher,
}


def get_fetcher(name: str, **kwargs) -> Fetcher:
    key = name.lower().strip()
    if key not in _REGISTRY:
        avail = ", ".join(sorted({k for k in _REGISTRY}))
        raise FetcherError(f"Unknown fetcher '{name}'. Available: {avail}")
    return _REGISTRY[key](**kwargs)


def available_fetchers() -> list[str]:
    return ["yt-dlp", "instaloader", "manual", "apify"]


__all__ = [
    "Fetcher",
    "FetcherError",
    "get_fetcher",
    "available_fetchers",
    "YtDlpFetcher",
    "InstaloaderFetcher",
    "ManualFetcher",
    "ApifyFetcher",
]
