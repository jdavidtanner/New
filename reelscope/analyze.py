"""Turn raw channel data into a 'winning formula' — the repeatable patterns
behind the top-performing reels.

Everything here is heuristic and runs with no API key. The optional LLM layer
(llm.py) can enrich these findings with qualitative reasoning, but the numbers
and structure come from here.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Optional

from .models import ChannelAnalysis, ChannelData, Reel

_HASHTAG_RE = re.compile(r"#(\w+)")
_MENTION_RE = re.compile(r"@(\w+)")

# Common call-to-action phrasings worth flagging.
_CTA_PATTERNS = [
    "follow", "comment", "save this", "share", "tag a", "link in bio",
    "dm me", "drop a", "let me know", "swipe", "check out", "sign up",
    "subscribe", "click", "try", "download", "join",
]

# Hook archetypes detected from the opening of a caption.
_HOOK_RULES = [
    ("question", re.compile(r"^\s*[^.?!]*\?", re.IGNORECASE)),
    ("how-to", re.compile(r"\bhow to\b|\bhow i\b", re.IGNORECASE)),
    ("listicle", re.compile(r"^\s*\d+\s+(ways|things|tips|reasons|secrets|steps|mistakes)",
                            re.IGNORECASE)),
    ("number-led", re.compile(r"^\s*\d+", re.IGNORECASE)),
    ("you-address", re.compile(r"\byou\b|\byour\b", re.IGNORECASE)),
    ("stop/warning", re.compile(r"^\s*(stop|don'?t|never|warning|avoid)\b", re.IGNORECASE)),
    ("secret/reveal", re.compile(r"\b(secret|nobody tells you|truth about|what no one)\b",
                                 re.IGNORECASE)),
    ("pov", re.compile(r"^\s*pov\b", re.IGNORECASE)),
    ("bold-claim", re.compile(r"^\s*(this|the)\b.*\b(best|worst|only|fastest|easiest)\b",
                              re.IGNORECASE)),
]


def _enrich(reel: Reel) -> None:
    """Populate hashtags/mentions/hook on a reel in place."""
    text = reel.caption or reel.title or ""
    reel.hashtags = [h.lower() for h in _HASHTAG_RE.findall(text)]
    reel.mentions = [m.lower() for m in _MENTION_RE.findall(text)]
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    reel.hook = first_line[:120]


def _classify_hook(hook: str) -> list[str]:
    return [name for name, rx in _HOOK_RULES if rx.search(hook)]


def _median(values: list[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def _duration_bucket(d: Optional[float]) -> str:
    if d is None:
        return "unknown"
    if d < 8:
        return "0-7s"
    if d < 15:
        return "8-14s"
    if d < 30:
        return "15-29s"
    if d < 60:
        return "30-59s"
    return "60s+"


def _cadence(reels: list[Reel]) -> Optional[float]:
    ts = sorted(r.timestamp for r in reels if r.timestamp)
    if len(ts) < 2:
        return None
    gaps = [(b - a) / 86400 for a, b in zip(ts, ts[1:])]
    return round(statistics.median(gaps), 1)


def analyze(data: ChannelData, top_n: int = 5) -> ChannelAnalysis:
    reels = list(data.reels)
    for r in reels:
        _enrich(r)

    views = [r.view_count for r in reels if r.view_count is not None]
    median_views = _median(views)

    # Rank by views when available, otherwise by engagement.
    def sort_key(r: Reel) -> float:
        return float(r.view_count if r.view_count is not None else r.engagement)

    ranked = sorted(reels, key=sort_key, reverse=True)
    top = ranked[:top_n]

    # --- Pattern extraction, weighted toward the top performers ---------------
    overperformers = (
        [r for r in reels if r.view_count and median_views and r.view_count >= median_views]
        if median_views
        else top
    )

    hook_counter: Counter[str] = Counter()
    for r in overperformers:
        for h in _classify_hook(r.hook):
            hook_counter[h] += 1

    hashtag_counter: Counter[str] = Counter()
    for r in overperformers:
        hashtag_counter.update(r.hashtags)

    duration_perf: dict[str, list[int]] = {}
    for r in reels:
        if r.view_count is not None:
            duration_perf.setdefault(_duration_bucket(r.duration_sec), []).append(r.view_count)
    best_duration = max(
        ((b, statistics.median(v)) for b, v in duration_perf.items() if v),
        key=lambda kv: kv[1],
        default=(None, None),
    )

    cta_counter: Counter[str] = Counter()
    for r in overperformers:
        low = (r.caption or "").lower()
        for cta in _CTA_PATTERNS:
            if cta in low:
                cta_counter[cta] += 1

    caption_lengths = [len(r.caption) for r in overperformers if r.caption]

    patterns = {
        "top_hooks": hook_counter.most_common(5),
        "top_hashtags": hashtag_counter.most_common(12),
        "best_duration_bucket": best_duration[0],
        "best_duration_median_views": best_duration[1],
        "common_ctas": cta_counter.most_common(6),
        "median_caption_chars": int(_median(caption_lengths)) if caption_lengths else None,
        "posting_cadence_days": _cadence(reels),
        "overperformer_count": len(overperformers),
        "example_winning_hooks": [r.hook for r in top if r.hook][:5],
        "view_range": (min(views), max(views)) if views else None,
    }

    return ChannelAnalysis(
        handle=data.handle,
        reel_count=len(reels),
        median_views=median_views,
        top_reels=top,
        patterns=patterns,
    )
