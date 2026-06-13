"""Tests for the ReelScope pipeline (fetch → analyze → brief).

These avoid any network access — they exercise the manual fetcher, the
analyzer, and the brief generator against fixed data.
"""

import os
import tempfile

from reelscope.analyze import analyze
from reelscope.brief import generate_briefs, _topic_from_business
from reelscope.fetchers import get_fetcher
from reelscope.fetchers.base import normalize_handle
from reelscope.models import ChannelData, Reel
from reelscope.report import render_briefs, render_report


SAMPLE_CSV = """url,caption,view_count,like_count,comment_count,duration_sec,timestamp
https://instagram.com/reel/a,"3 mistakes you're making #candles #diy",1200000,45000,800,12,2026-05-01
https://instagram.com/reel/b,"How to make your home smell amazing #candles",890000,30000,500,18,2026-05-04
https://instagram.com/reel/c,"Stop buying cheap candles #candles",2100000,90000,1500,9,2026-05-08
https://instagram.com/reel/d,"POV: you found your scent #candles",430000,12000,200,22,2026-05-12
"""


def _sample_data() -> ChannelData:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "reels.csv")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(SAMPLE_CSV)
        return get_fetcher("manual", path=path).fetch("candlebrand", limit=50)


def test_normalize_handle_variants():
    assert normalize_handle("natgeo") == "natgeo"
    assert normalize_handle("@natgeo") == "natgeo"
    assert normalize_handle("https://www.instagram.com/natgeo/reels/") == "natgeo"
    assert normalize_handle("instagram.com/natgeo/") == "natgeo"


def test_manual_fetcher_parses_csv():
    data = _sample_data()
    assert data.source == "manual"
    assert len(data.reels) == 4
    assert data.reels[0].view_count == 1200000
    assert data.reels[0].timestamp is not None


def test_reel_engagement_and_rate():
    r = Reel(url="x", view_count=1000, like_count=90, comment_count=10)
    assert r.engagement == 100
    assert r.engagement_rate == 0.1
    assert Reel(url="x").engagement_rate is None


def test_analyze_extracts_patterns():
    analysis = analyze(_sample_data())
    assert analysis.reel_count == 4
    assert analysis.median_views is not None
    # Top reel should be the 2.1M-view one.
    assert analysis.top_reels[0].view_count == 2100000
    p = analysis.patterns
    assert p["best_duration_bucket"] is not None
    assert isinstance(p["top_hooks"], list)
    # 'candles' hashtag appears across overperformers.
    tags = dict(p["top_hashtags"])
    assert "candles" in tags


def test_topic_extraction_is_clean():
    assert _topic_from_business("my handmade soy candle shop for cozy homes") == "handmade soy candle shop"
    assert _topic_from_business("our app that helps you budget") == "app"
    assert _topic_from_business("") == "your product"


def test_generate_briefs_uses_topic():
    analysis = analyze(_sample_data())
    briefs = generate_briefs(analysis, business="x", n=3, topic="soy candles")
    assert len(briefs) == 3
    assert all("soy candles" in b.hook_line for b in briefs)
    assert all(b.higgsfield_prompt for b in briefs)
    assert all(b.settings["aspect_ratio"] == "9:16" for b in briefs)


def test_reports_render_without_error():
    analysis = analyze(_sample_data())
    report = render_report(analysis)
    assert "winning formula" in report.lower()
    briefs = generate_briefs(analysis, business="candles", n=2, topic="candles")
    briefs_md = render_briefs(briefs, "candlebrand", "candles")
    assert "Higgsfield prompt" in briefs_md


def test_channel_data_roundtrip(tmp_path):
    data = _sample_data()
    path = tmp_path / "data.json"
    data.save(str(path))
    loaded = ChannelData.load(str(path))
    assert loaded.handle == data.handle
    assert len(loaded.reels) == len(data.reels)
    assert loaded.reels[0].url == data.reels[0].url
