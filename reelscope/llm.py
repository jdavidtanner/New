"""Optional AI enrichment of the analysis using the Claude API.

This is strictly optional. With no key, ReelScope still produces the full
heuristic report and Higgsfield briefs. If ANTHROPIC_API_KEY is set, we ask
Claude to add a qualitative read of *why* the channel's reels work and how to
adapt the formula — the judgement layer on top of the numbers.
"""

from __future__ import annotations

import os

from .analyze import ChannelAnalysis

_MODEL = os.environ.get("REELSCOPE_LLM_MODEL", "claude-opus-4-8")


def enrich_analysis(analysis: ChannelAnalysis, business: str) -> str:
    """Return Markdown notes from Claude, or '' if unavailable."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ""
    try:
        import anthropic  # noqa: PLC0415 (optional dependency)
    except ImportError:
        return ""

    client = anthropic.Anthropic(api_key=api_key)
    prompt = _build_prompt(analysis, business)
    try:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:  # network/auth/etc — degrade gracefully
        return f"_(AI enrichment skipped: {e})_"

    return "".join(block.text for block in resp.content if block.type == "text")


def _build_prompt(analysis: ChannelAnalysis, business: str) -> str:
    p = analysis.patterns
    top_hooks = "\n".join(f'- "{h}"' for h in p.get("example_winning_hooks", []))
    return f"""You are a short-form video strategist. Below is data extracted from the
Instagram reels of @{analysis.handle}.

Reels analyzed: {analysis.reel_count}
Median views: {analysis.median_views}
Overperforming hook styles: {p.get('top_hooks')}
Best duration bucket: {p.get('best_duration_bucket')}
Common CTAs: {p.get('common_ctas')}
Posting cadence (days): {p.get('posting_cadence_days')}
Their actual top hooks:
{top_hooks}

The user runs this business: "{business}"

Write a concise, practical analysis (markdown, ~300 words) covering:
1. Why these reels likely work (psychology of the hooks + format).
2. The 3-4 most important things to replicate.
3. Specific ways to adapt this formula to the user's business without copying content.
Be concrete and tactical. No preamble."""
