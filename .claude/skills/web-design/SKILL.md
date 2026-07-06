---
name: web-design
description: >-
  Orchestrated web-design workflow that chains all five pro web-design tools —
  shadcn MCP (component library), Magic MCP by 21st.dev (animated add-ons),
  the Front-End Design skill (fonts/colors/motion), Chrome DevTools MCP (drive a
  real browser and self-check), and the Web Design Guidelines skill from Vercel
  (final pro-design audit). Use when the user wants to design, build, restyle,
  polish, or review a website or web UI and wants Claude to produce
  professional, non-generic results rather than the default "AI website" look.
  Trigger on: "design a site", "build a landing page", "make this look good",
  "restyle", "web design", "/web-design".
---

# Web Design Workflow

You are running a professional web-design pipeline. Instead of building a UI
blind and shipping the generic AI-website look, you drive five specialist tools
in a deliberate order: **direction → build → see → audit → fix**.

## Preflight: confirm the toolbelt is installed

This skill *uses* the five tools; it does not install them. Before starting,
verify they are available in the session. If any are missing, tell the user
which ones and point them at `references/install.md`, then continue with
whatever is available (degrade gracefully — never fake a step you can't run).

| # | Tool | Kind | What it gives you | How to detect |
|---|------|------|-------------------|---------------|
| 1 | **shadcn** | MCP | Real, good-looking component blocks (buttons, nav, pricing, auth) | `ToolSearch "shadcn"` returns tools |
| 2 | **Web Design Guidelines** (Vercel) | Skill | Audits a built site against pro design rules | listed in available skills |
| 3 | **Front-End Design** | Skill | Good fonts, colors, spacing, motion — kills the generic look | listed in available skills |
| 4 | **Chrome DevTools** | MCP | Opens the site in a real browser; screenshots, console, perf | `ToolSearch "chrome devtools"` returns tools |
| 5 | **Magic** (21st.dev) | MCP | Fancy add-ons: hero/button animations, cohesive full-site styles | `ToolSearch "magic 21st"` returns tools |

Tool names vary by how they were installed. Use `ToolSearch` to load the real
schemas before calling any MCP tool — never call an MCP tool from memory.

## Ask two questions before designing (only if unstated)

Do NOT interrogate the user. If the request already answers these, skip. Only
ask when you genuinely can't proceed:

1. **What is it?** — product/landing/dashboard/blog/app, and the one job the
   page must do.
2. **What vibe?** — brand feel (bold, minimal, playful, editorial, corporate),
   any existing brand colors/fonts, and a reference site they like.

## The pipeline

Run these phases in order. Each phase names the tool that owns it.

### Phase 1 — Direction (Front-End Design skill)

Invoke the **Front-End Design** skill first, before writing any markup. Use it
to lock the aesthetic system up front so everything downstream is consistent:

- Typography: a real type pairing (not system-ui default), scale, weights.
- Color: a committed palette with proper contrast for light **and** dark.
- Spacing & layout rhythm, radius, shadow/elevation language.
- Motion: where animation belongs and where it doesn't.

Output a short **design brief** (the tokens above) and reuse it as the source
of truth for every later phase. If the project already has design tokens /
Tailwind config, read them and extend rather than reinvent.

### Phase 2 — Build the skeleton (shadcn MCP)

Pull real components instead of hand-rolling janky ones. Use the **shadcn** MCP
to search the registry and add the blocks the page needs — nav, hero shell,
feature grid, pricing, testimonials, footer, forms, auth. Then restyle each to
the Phase-1 brief (colors, radius, type). shadcn gives you correct, accessible
structure; you own the look.

Prefer composing existing blocks over generating novel layout from scratch.

### Phase 3 — Elevate with add-ons (Magic MCP)

For the pieces that carry the "wow" — hero, primary CTAs, section reveals — use
**Magic** by 21st.dev to bring in polished animated components and cohesive
full-site styling. Keep motion tasteful and on-brand per the Phase-1 brief;
don't let every element move. Magic supplies the flourish, Phase 1 sets the
budget for it.

### Phase 4 — See it for real (Chrome DevTools MCP)

Stop building blind. Start the dev server, then use **Chrome DevTools** MCP to
open the actual page in a real browser and inspect what you built:

- Screenshot desktop **and** mobile widths — check the responsive story.
- Read the console for errors/warnings; check network for broken assets.
- Verify interactive states (hover, focus, open menus, form validation).
- Sanity-check performance (layout shift, oversized images, jank).

Fix what you find, then re-screenshot. Loop until it renders clean. This is the
step that turns "looks right in the code" into "looks right in the browser."

### Phase 5 — Pro audit (Web Design Guidelines skill)

With a working, real-in-browser page, invoke the **Web Design Guidelines**
(Vercel) skill to grade the whole thing against professional design rules. It
catches the small stuff your eye skips: inconsistent spacing, weak hierarchy,
contrast failures, misaligned optical rhythm, off type scale, a11y gaps.

Turn its findings into a concrete fix list.

### Phase 6 — Fix & re-verify (loop)

Apply the audit fixes, then go back to **Phase 4** (Chrome DevTools) and
re-screenshot to confirm each fix landed and nothing regressed. Repeat Phase
5→6 until the guidelines pass clean and the browser render matches the brief.

## Operating rules

- **Order matters.** Direction before build, build before polish, see before
  audit, audit before "done." Skipping the browser/audit steps is how the
  generic look survives — don't.
- **One source of truth.** The Phase-1 design brief governs colors, type,
  spacing, and motion everywhere. If Magic or shadcn defaults fight the brief,
  the brief wins.
- **Degrade gracefully.** If a tool isn't installed, say so, skip its phase,
  and do the closest thing you can by hand — but never claim you ran a tool you
  didn't.
- **Show, don't tell.** End with browser screenshots (desktop + mobile) and a
  short summary of what each phase changed, plus any audit items intentionally
  left open.

## Reference

- `references/install.md` — one-shot install commands for all five tools.
