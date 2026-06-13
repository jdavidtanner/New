"""Data models shared across the fetch → analyze → brief pipeline.

Kept dependency-free (pure stdlib dataclasses) so every stage can import these
without dragging in scraping or LLM packages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


@dataclass
class Reel:
    """A single Instagram reel and the metrics we can analyze."""

    url: str
    caption: str = ""
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    duration_sec: Optional[float] = None
    timestamp: Optional[int] = None  # unix seconds, when it was posted
    title: str = ""
    thumbnail: str = ""

    # Filled in during analysis.
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    hook: str = ""  # first line / opening of the caption

    @property
    def posted_at(self) -> Optional[datetime]:
        if self.timestamp is None:
            return None
        return datetime.utcfromtimestamp(self.timestamp)

    @property
    def engagement(self) -> int:
        """Likes + comments; a rough proxy for resonance."""
        return (self.like_count or 0) + (self.comment_count or 0)

    @property
    def engagement_rate(self) -> Optional[float]:
        """Engagement per view — normalizes for reach. None if views unknown."""
        if not self.view_count:
            return None
        return self.engagement / self.view_count

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Reel":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ChannelData:
    """Raw fetched data for a channel, before analysis."""

    handle: str
    source: str  # which fetcher produced this
    fetched_at: str
    reels: list[Reel] = field(default_factory=list)
    follower_count: Optional[int] = None
    bio: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ChannelData":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        reels = [Reel.from_dict(r) for r in d.pop("reels", [])]
        return cls(reels=reels, **d)


@dataclass
class ChannelAnalysis:
    """The output of the analyze stage — the 'winning formula'."""

    handle: str
    reel_count: int
    median_views: Optional[float]
    top_reels: list[Reel]
    patterns: dict[str, Any]  # see analyze.py for the keys

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d
