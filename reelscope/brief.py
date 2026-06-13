"""Turn the analyzed 'winning formula' into Higgsfield-ready briefs for YOUR
business.

The goal is to duplicate the *structure* that's working (hook style, pacing,
shot rhythm, CTA placement) while swapping in your own subject matter — so you
inherit the proven format without copying anyone's content.

Output is plain text you can paste straight into Higgsfield, plus structured
data the API client (higgsfield.py) can consume if you wire it up later.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .models import ChannelAnalysis

# Reusable shot-rhythm templates keyed by the hook archetype that performed
# best. Each is a list of (beat_label, direction) tuples.
_SHOT_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "question": [
        ("Hook (0-2s)", "Tight close-up, direct-to-camera, pose the question on-screen as bold text"),
        ("Tension (2-6s)", "Quick cut to the problem visual; punch-in motion to build curiosity"),
        ("Payoff (6-18s)", "Reveal the answer step by step with snappy cuts and motion captions"),
        ("CTA (18-22s)", "Return to subject, hold gaze, overlay the call-to-action"),
    ],
    "listicle": [
        ("Hook (0-2s)", "Bold number + promise as full-frame text over a striking visual"),
        ("Items (2-18s)", "One fast cut per item, each with a kinetic text label, consistent rhythm"),
        ("Recap (18-22s)", "Rapid montage of all items, then the CTA"),
    ],
    "how-to": [
        ("Hook (0-3s)", "Show the finished result first; promise 'here's how'"),
        ("Steps (3-20s)", "Sequential steps, each a clean shot with a step label, smooth transitions"),
        ("CTA (20-24s)", "Result again + save/follow CTA"),
    ],
    "stop/warning": [
        ("Hook (0-2s)", "Hard cut to subject saying 'Stop' — pattern interrupt, freeze-frame"),
        ("Stakes (2-8s)", "Show the mistake/consequence with dramatic motion"),
        ("Fix (8-18s)", "Show the correct way; before/after split"),
        ("CTA (18-22s)", "Direct address + CTA"),
    ],
    "pov": [
        ("Hook (0-2s)", "POV text overlay; immersive first-person framing"),
        ("Scene (2-16s)", "Play out the relatable scenario with cinematic motion"),
        ("Turn (16-22s)", "Emotional or surprising beat, then CTA"),
    ],
    "default": [
        ("Hook (0-2s)", "Strong opening visual + bold on-screen hook text, direct to camera"),
        ("Build (2-8s)", "Punch-in motion, introduce the core idea with a kinetic caption"),
        ("Deliver (8-18s)", "Main content in fast, rhythmic cuts; keep one idea per cut"),
        ("CTA (18-22s)", "Close on the subject with a clear call-to-action overlay"),
    ],
}

# Phrasing scaffolds per hook archetype; {topic} is filled from your business.
_HOOK_SCAFFOLDS: dict[str, list[str]] = {
    "question": ["Did you know {topic}?", "What if {topic}?", "Why does nobody talk about {topic}?"],
    "listicle": ["3 {topic} mistakes you're making", "5 {topic} secrets in 20 seconds",
                 "3 things about {topic} I wish I knew sooner"],
    "how-to": ["How to {topic} (the fast way)", "How I {topic} in under a minute"],
    "stop/warning": ["Stop doing {topic} like this", "Never {topic} until you watch this"],
    "secret/reveal": ["The truth about {topic}", "What no one tells you about {topic}"],
    "number-led": ["1 {topic} tip that changed everything", "30 seconds to better {topic}"],
    "you-address": ["You're doing {topic} wrong", "Your {topic} could be this good"],
    "pov": ["POV: you finally figured out {topic}", "POV: {topic} just got easy"],
    "bold-claim": ["The only {topic} guide you need", "The fastest way to {topic}"],
}


@dataclass
class HiggsfieldBrief:
    concept: str
    hook_archetype: str
    hook_line: str
    shot_list: list[tuple[str, str]]
    higgsfield_prompt: str
    suggested_caption: str
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_LEADING_RE = __import__("re").compile(r"^(my|our|the|a|an)\s+", __import__("re").IGNORECASE)
# Prepositions/conjunctions where a business description usually stops being a
# clean noun phrase ("candle shop FOR cozy homes" -> "candle shop").
_CUT_WORDS = {"for", "that", "to", "in", "with", "of", "about", "where", "which", "selling"}


def _topic_from_business(business: str) -> str:
    """Compress a business description into a short, readable topic phrase that
    drops cleanly into hook scaffolds like 'You're doing {topic} wrong'."""
    business = (business or "your product").strip().rstrip(".")
    business = _LEADING_RE.sub("", business)
    words = business.split()
    out: list[str] = []
    for w in words:
        if w.lower() in _CUT_WORDS:
            break
        out.append(w)
        if len(out) >= 5:
            break
    return " ".join(out) if out else "your product"


def _best_duration_hint(analysis: ChannelAnalysis) -> str:
    bucket = analysis.patterns.get("best_duration_bucket")
    return bucket or "8-14s"


def generate_briefs(
    analysis: ChannelAnalysis,
    business: str,
    n: int = 3,
    topic: str | None = None,
) -> list[HiggsfieldBrief]:
    """Create ``n`` Higgsfield briefs from the channel's winning patterns.

    ``topic`` is the short subject phrase used inside hook lines (e.g.
    "soy candles"). If omitted, it's derived from ``business``.
    """
    topic = (topic or "").strip() or _topic_from_business(business)
    duration = _best_duration_hint(analysis)

    top_hooks = [h for h, _ in analysis.patterns.get("top_hooks", [])]
    if not top_hooks:
        top_hooks = ["default"]

    ctas = [c for c, _ in analysis.patterns.get("common_ctas", [])] or ["follow for more"]
    hashtags = [f"#{h}" for h, _ in analysis.patterns.get("top_hashtags", [])][:8]

    briefs: list[HiggsfieldBrief] = []
    for i in range(n):
        archetype = top_hooks[i % len(top_hooks)]
        scaffolds = _HOOK_SCAFFOLDS.get(archetype, ["{topic}: here's what works"])
        hook_line = scaffolds[i % len(scaffolds)].format(topic=topic)
        shots = _SHOT_TEMPLATES.get(archetype, _SHOT_TEMPLATES["default"])
        cta = ctas[i % len(ctas)]

        prompt = _render_higgsfield_prompt(hook_line, shots, topic, duration)
        caption = f"{hook_line}\n\n{cta.capitalize()}.\n\n" + " ".join(hashtags)

        briefs.append(
            HiggsfieldBrief(
                concept=f"Concept {i + 1}: {archetype} hook",
                hook_archetype=archetype,
                hook_line=hook_line,
                shot_list=shots,
                higgsfield_prompt=prompt,
                suggested_caption=caption,
                settings={
                    "aspect_ratio": "9:16",
                    "target_duration": duration,
                    "motion": "high energy, snappy cuts",
                    "style": "modern social, bold kinetic captions",
                },
            )
        )
    return briefs


def _render_higgsfield_prompt(
    hook_line: str, shots: list[tuple[str, str]], topic: str, duration: str
) -> str:
    """A single paste-ready prompt block for the Higgsfield app."""
    lines = [
        f"Vertical 9:16 short-form reel about {topic}. Target length {duration}.",
        f'Opening hook line on screen: "{hook_line}".',
        "Style: modern, high-energy social video with bold kinetic captions, "
        "punch-in camera motion, fast rhythmic cuts, clean lighting.",
        "Shot sequence:",
    ]
    for beat, direction in shots:
        lines.append(f"  - {beat}: {direction}.")
    lines.append(
        "End on a confident direct-to-camera moment with the call-to-action overlaid."
    )
    return "\n".join(lines)
