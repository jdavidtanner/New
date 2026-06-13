# ReelScope

Research an Instagram channel's reels, learn what makes them successful, and
turn that winning formula into ready-to-use **Higgsfield** briefs for *your*
business — so you duplicate the structure that's working, reskinned to your own
product (not copying anyone's content).

```
  point at a channel  ──▶  analyze what works  ──▶  Higgsfield briefs for you
   (yt-dlp/manual/…)        (hooks, length,           (prompts + shot lists,
                             cadence, CTAs)             reskinned to your biz)
```

## Why it runs on your laptop

Instagram blocks server-side scraping from cloud machines. On **your** laptop,
where a browser is already logged into Instagram, the free `yt-dlp` path can
read public channels using your existing session — no paid API needed.

## Install

```bash
pip install yt-dlp        # the default (free) data source
# optional extras:
pip install instaloader   # alternate scraper
pip install anthropic     # optional AI analysis layer (needs ANTHROPIC_API_KEY)
```

## Quick start

```bash
# Analyze a channel and generate Higgsfield briefs (free yt-dlp path)
python -m reelscope analyze natgeo \
    --business "my handmade soy candle shop for cozy homes" \
    --topic "soy candles" \
    --limit 30
```

Outputs land in `out/<handle>/`:

| File | What it is |
|------|------------|
| `report.md` | The winning formula: top hooks, best length, cadence, CTAs, hashtags |
| `higgsfield_briefs.md` | Paste-ready Higgsfield prompts + shot lists for your business |
| `data.json` | Raw fetched reels (re-run analysis without re-scraping) |

## Data sources (`--source`)

| Source | Cost | Notes |
|--------|------|-------|
| `yt-dlp` *(default)* | Free | Uses your browser's IG login cookies. `--cookies-from-browser chrome` (or firefox/safari/edge/brave). |
| `instaloader` | Free | Pure-Python; can break when IG changes. |
| `manual` | Free | `--input reels.csv` (or `.json`) — data you gathered by hand. Never breaks. |
| `apify` | Paid | Most robust automation. Set `APIFY_TOKEN`. |

**Manual CSV format** (only `url` is required):

```csv
url,caption,view_count,like_count,comment_count,duration_sec,timestamp
https://instagram.com/reel/xxx,"my caption #tag",120000,5400,80,12,2026-05-01
```

## Connecting to Higgsfield

By default ReelScope **generates prompts/shot-lists** you paste into the
Higgsfield app — works today, no key. When you have a Higgsfield Creator plan:

```bash
export HIGGSFIELD_API_KEY=...   # optional secret: HIGGSFIELD_API_SECRET
```

Then in Python:

```python
from reelscope.higgsfield import HiggsfieldClient
from reelscope.brief import generate_briefs
# ... build `analysis`, then:
briefs = generate_briefs(analysis, business="my candle shop", topic="soy candles")
job = HiggsfieldClient().generate_video(briefs[0])
```

## Re-run without re-scraping

```bash
python -m reelscope brief out/natgeo/data.json --business "my candle shop" --topic "candles"
```

## Optional: AI analysis layer

If `ANTHROPIC_API_KEY` is set and `anthropic` is installed, ReelScope adds a
qualitative "why it works / how to adapt it" section to the report. Disable with
`--no-ai`.

## Notes & limits

- Respect Instagram's Terms of Service and only analyze public data.
- View counts/captions are metadata-level; the briefs reproduce *format and
  structure*, which is what's portable across niches.
- IG changes its internals periodically; if `yt-dlp` stalls, update it
  (`pip install -U yt-dlp`) or fall back to `--source manual`.
```
