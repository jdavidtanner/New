"""Command-line interface for ReelScope.

Typical use on your laptop::

    # 1) point at a channel, analyze, and generate Higgsfield briefs (free path)
    python -m reelscope analyze natgeo --business "my handmade candle shop" --limit 30

    # 2) use a CSV you gathered by hand instead of scraping
    python -m reelscope analyze natgeo --source manual --input reels.csv \
        --business "my candle shop"

    # 3) re-run analysis/briefs on already-fetched data (no scraping)
    python -m reelscope brief out/natgeo/data.json --business "my candle shop"
"""

from __future__ import annotations

import argparse
import os
import sys

from .analyze import analyze
from .brief import generate_briefs
from .fetchers import available_fetchers, get_fetcher
from .fetchers.base import FetcherError
from .llm import enrich_analysis
from .models import ChannelData
from .report import render_briefs, render_report


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _run_analyze(args: argparse.Namespace) -> int:
    fetcher_kwargs = {}
    if args.source in ("manual",):
        fetcher_kwargs["path"] = args.input
    if args.cookies_from_browser:
        fetcher_kwargs["cookies_from_browser"] = args.cookies_from_browser
    if args.cookies_file:
        fetcher_kwargs["cookies_file"] = args.cookies_file

    try:
        fetcher = get_fetcher(args.source, **fetcher_kwargs)
        print(f"→ Fetching up to {args.limit} reels from @{args.handle} via {args.source}…")
        data = fetcher.fetch(args.handle, limit=args.limit)
    except FetcherError as e:
        print(f"\n✗ Could not fetch reels:\n  {e}\n", file=sys.stderr)
        print("Tip: try '--source manual --input reels.csv', or '--source apify' "
              "with APIFY_TOKEN set.", file=sys.stderr)
        return 1

    if not data.reels:
        print("✗ No reels returned. Nothing to analyze.", file=sys.stderr)
        return 1
    print(f"  got {len(data.reels)} reels.")

    outdir = os.path.join(args.out, data.handle)
    os.makedirs(outdir, exist_ok=True)
    data.save(os.path.join(outdir, "data.json"))

    return _analyze_and_brief(data, args, outdir)


def _run_brief(args: argparse.Namespace) -> int:
    try:
        data = ChannelData.load(args.data)
    except (OSError, ValueError) as e:
        print(f"✗ Could not load {args.data}: {e}", file=sys.stderr)
        return 1
    outdir = os.path.dirname(os.path.abspath(args.data))
    return _analyze_and_brief(data, args, outdir)


def _analyze_and_brief(data: ChannelData, args: argparse.Namespace, outdir: str) -> int:
    print("→ Analyzing patterns…")
    analysis = analyze(data, top_n=args.top)

    llm_notes = ""
    if not args.no_ai:
        notes = enrich_analysis(analysis, args.business)
        if notes:
            print("  added AI analysis (Claude).")
            llm_notes = notes

    report_md = render_report(analysis, llm_notes=llm_notes)
    report_path = os.path.join(outdir, "report.md")
    _write(report_path, report_md)

    print("→ Generating Higgsfield briefs…")
    briefs = generate_briefs(analysis, args.business, n=args.concepts, topic=args.topic)
    briefs_md = render_briefs(briefs, analysis.handle, args.business)
    briefs_path = os.path.join(outdir, "higgsfield_briefs.md")
    _write(briefs_path, briefs_md)

    print("\n✓ Done.")
    print(f"  Analysis : {report_path}")
    print(f"  Briefs   : {briefs_path}")
    print(f"  Raw data : {os.path.join(outdir, 'data.json')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reelscope",
        description="Research an Instagram channel's reels and turn the winning "
                    "formula into Higgsfield briefs for your business.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--business", default="my business",
                        help="Short description of YOUR business (used to reskin the briefs).")
    common.add_argument("--topic", default=None,
                        help="Short subject phrase for hook lines, e.g. 'soy candles'. "
                             "If omitted it's derived from --business.")
    common.add_argument("--out", default="out", help="Output directory (default: out/).")
    common.add_argument("--top", type=int, default=5, help="How many top reels to highlight.")
    common.add_argument("--concepts", type=int, default=3,
                        help="How many Higgsfield concepts to generate.")
    common.add_argument("--no-ai", action="store_true",
                        help="Skip the optional Claude enrichment even if a key is set.")

    a = sub.add_parser("analyze", parents=[common],
                       help="Fetch a channel, analyze it, and generate briefs.")
    a.add_argument("handle", help="Instagram handle, @handle, or profile URL.")
    a.add_argument("--source", default="yt-dlp", choices=available_fetchers(),
                   help="Where to pull reels from (default: yt-dlp).")
    a.add_argument("--limit", type=int, default=30, help="Max reels to fetch.")
    a.add_argument("--input", help="Path to CSV/JSON when --source manual.")
    a.add_argument("--cookies-from-browser", default="chrome",
                   help="Browser to read IG login cookies from for yt-dlp "
                        "(chrome/firefox/safari/edge/brave). Use '' to disable.")
    a.add_argument("--cookies-file", help="Path to a cookies.txt for yt-dlp (overrides browser).")
    a.set_defaults(func=_run_analyze)

    b = sub.add_parser("brief", parents=[common],
                       help="Re-run analysis + briefs on an existing data.json (no scraping).")
    b.add_argument("data", help="Path to a previously saved data.json.")
    b.set_defaults(func=_run_brief)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Normalize: yt-dlp '' means "no browser cookies".
    if getattr(args, "cookies_from_browser", None) == "":
        args.cookies_from_browser = None
    if getattr(args, "command", None) == "analyze":
        # Stash a clean handle for messaging; fetchers normalize again.
        from .fetchers.base import normalize_handle
        try:
            args.handle = normalize_handle(args.handle) if args.source != "manual" or args.handle else args.handle
        except FetcherError:
            pass
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
