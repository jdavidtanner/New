"""Render analysis + briefs into readable Markdown files."""

from __future__ import annotations

from .analyze import ChannelAnalysis
from .brief import HiggsfieldBrief


def _fmt_int(n: object) -> str:
    if n is None:
        return "—"
    try:
        return f"{int(n):,}"
    except (ValueError, TypeError):
        return str(n)


def render_report(analysis: ChannelAnalysis, llm_notes: str = "") -> str:
    p = analysis.patterns
    lines: list[str] = []
    lines.append(f"# What's working on @{analysis.handle}")
    lines.append("")
    lines.append(f"- Reels analyzed: **{analysis.reel_count}**")
    lines.append(f"- Median views: **{_fmt_int(analysis.median_views)}**")
    if p.get("view_range"):
        lo, hi = p["view_range"]
        lines.append(f"- View range: {_fmt_int(lo)} – {_fmt_int(hi)}")
    if p.get("posting_cadence_days") is not None:
        lines.append(f"- Posting cadence: a reel roughly every **{p['posting_cadence_days']} days**")
    lines.append("")

    lines.append("## The winning formula")
    lines.append("")
    if p.get("top_hooks"):
        lines.append("**Hook styles that overperform:**")
        for name, count in p["top_hooks"]:
            lines.append(f"- `{name}` — used in {count} of their above-median reels")
        lines.append("")
    if p.get("best_duration_bucket"):
        lines.append(
            f"**Best length:** reels in the **{p['best_duration_bucket']}** range pull the most views "
            f"(median {_fmt_int(p.get('best_duration_median_views'))})."
        )
        lines.append("")
    if p.get("median_caption_chars") is not None:
        lines.append(f"**Caption length:** ~{p['median_caption_chars']} characters on winning reels.")
        lines.append("")
    if p.get("common_ctas"):
        ctas = ", ".join(f"`{c}` ({n})" for c, n in p["common_ctas"])
        lines.append(f"**Calls-to-action they lean on:** {ctas}")
        lines.append("")
    if p.get("top_hashtags"):
        tags = " ".join(f"#{h}" for h, _ in p["top_hashtags"])
        lines.append(f"**Recurring hashtags:** {tags}")
        lines.append("")

    if p.get("example_winning_hooks"):
        lines.append("## Their actual top hooks (steal the structure, not the words)")
        lines.append("")
        for h in p["example_winning_hooks"]:
            lines.append(f"- \"{h}\"")
        lines.append("")

    lines.append("## Top reels by performance")
    lines.append("")
    lines.append("| Views | Likes | Length | Hook |")
    lines.append("|------:|------:|:------:|:-----|")
    for r in analysis.top_reels:
        dur = f"{int(r.duration_sec)}s" if r.duration_sec else "—"
        hook = (r.hook or "").replace("|", "\\|")[:70]
        lines.append(f"| {_fmt_int(r.view_count)} | {_fmt_int(r.like_count)} | {dur} | {hook} |")
    lines.append("")

    if llm_notes:
        lines.append("## Deeper read (AI analysis)")
        lines.append("")
        lines.append(llm_notes)
        lines.append("")

    return "\n".join(lines)


def render_briefs(briefs: list[HiggsfieldBrief], handle: str, business: str) -> str:
    lines: list[str] = []
    lines.append(f"# Higgsfield briefs — your version of @{handle}'s formula")
    lines.append("")
    lines.append(f"**Your business:** {business}")
    lines.append("")
    lines.append(
        "Each concept below copies a *structure* that's working on the target "
        "channel and reskins it to your business. Paste the **Higgsfield prompt** "
        "into the Higgsfield app (Text-to-Video or Image-to-Video), film/generate "
        "the shots in order, and use the suggested caption."
    )
    lines.append("")

    for b in briefs:
        lines.append(f"## {b.concept}")
        lines.append("")
        lines.append(f"**Hook line:** {b.hook_line}")
        lines.append("")
        lines.append("**Shot list:**")
        for beat, direction in b.shot_list:
            lines.append(f"- **{beat}** — {direction}")
        lines.append("")
        lines.append("**Higgsfield prompt (paste this):**")
        lines.append("")
        lines.append("```")
        lines.append(b.higgsfield_prompt)
        lines.append("```")
        lines.append("")
        lines.append("**Suggested caption:**")
        lines.append("")
        lines.append("```")
        lines.append(b.suggested_caption)
        lines.append("```")
        lines.append("")
        s = b.settings
        lines.append(
            f"**Settings:** {s.get('aspect_ratio')} · {s.get('target_duration')} · "
            f"{s.get('motion')} · {s.get('style')}"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
