---
name: ai-3d-animated-website
description: A repeatable 6-stage production workflow for building premium 3D/animated websites using AI tools — from a visual reference to a live URL. Use when the user wants to build an animated or 3D hero-section website, turn a design reference into a working animated site, create a landing page with a looping video background, or asks about the "AI animated website" / "AI 3D website" workflow. Covers sourcing references, generating visuals, animating them into video loops, scaffolding site code, polishing the UI, and deploying.
---

# AI 3D Animated Websites Workflow

A practical production system for turning a visual reference into a live, animated website using AI generation and code tools. It is **repeatable**: run it once and you get a working site; run it again with a different reference and you get a completely different result. Each stage has a clear **input → actions → output**, so you can always tell whether a stage is done before moving on.

## When to use this skill

Reach for this workflow when the user wants to:
- Build a premium landing page or hero section with a 3D character, motion background, or animated style.
- Turn a screenshot / design reference into a working animated site.
- Create a site with a seamless looping video background.
- Understand or run the end-to-end "idea → live URL" AI website pipeline.

## How to run it

Work through the six stages in order. **Do not skip ahead** — each stage consumes the previous stage's output. Before advancing, confirm the stage's defined output actually exists (the reference screenshot, the clean PNG, the MP4 loop, etc.).

Claude Code's own strongest role is **Stage 4–6** (assembling, polishing, and deploying the code). Stages 1–3 produce media assets using external AI tools; when those tools aren't available in the environment, guide the user to produce each asset and hand it back, then take over the code.

At each stage, state which stage you're on, what its input is, and what output you're producing.

---

## Stage 1 — Ideas & References

Pick a visual direction **before touching any tool**. The reference image becomes the brief for everything that follows.

**Input:** the project's tone / goal.

**Actions:**
- Browse design galleries for layouts with 3D or animated elements.
- Pick one reference that fits the project's tone.
- Take a clean screenshot (full-page or hero section).

**Tools:** Pinterest · Landbook · Dribbble

**Output:** one reference screenshot ready to feed into AI.

---

## Stage 2 — Visual Generation

Strip the reference down to a clean image — no text, no UI chrome — and regenerate it in the same style as a unique visual.

**Input:** the reference screenshot from Stage 1.

**Actions:**
- Upload the reference screenshot to an image AI.
- Use a prompt like: *"Recreate this image, remove all text, keep the style."*
- Set the format to **16:9**.
- Export as PNG or JPG.

**Tools:** ChatGPT / DALL·E 3 · Nano Banana · Google AI Studio

**Output:** a clean visual file (PNG / JPG) with no text overlays.

---

## Stage 3 — Animation

Turn the static image into a looping video background with subtle motion.

**Input:** the clean visual from Stage 2.

**Actions:**
- Upload the generated image to a video AI.
- Set the motion type: slow zoom, gentle rotation, or ambient drift.
- Enable **loop / seamless loop** mode so the video plays endlessly without a visible cut.
- Export as MP4.

**Tools:** Kling AI · Seedance

**Output:** an MP4 video loop ready to use as a background.

---

## Stage 4 — AI Website Builder Assembly

Use an AI builder to scaffold the site's code automatically.

**Input:** the reference screenshot (Stage 1) + the MP4 loop (Stage 3).

**Actions:**
- Open an AI website builder (or scaffold the code directly here in Claude Code).
- Provide both the reference screenshot and the MP4 video.
- Prompt: *"Build a hero section like this, use this video as background."*
- Review the generated code output.

**Tools:** AntiGravity · Bolt.new · Lovable · (or Claude Code directly)

**Output:** working site code with a video background.

> **Doing this in Claude Code:** build a hero section that plays the MP4 as a full-bleed background — `<video autoplay muted loop playsinline>` covering the viewport with `object-fit: cover`, overlaid content on top, and a subtle dark gradient scrim for text contrast. Keep the markup semantic and the CSS minimal.

---

## Stage 5 — Integration & UI Polish

Refine what the builder generated so the layout feels intentional. Small changes make a large difference in perceived quality.

**Input:** the working site code from Stage 4.

**Actions:**
- Add a mouse **parallax** effect to the hero section.
- Swap fonts to Helvetica (or the project's chosen typeface).
- Check spacing, alignment, and text contrast.
- Fix any layout issues on mobile.

**Tools:** Claude Code · Google AI Studio

**Output:** a polished, interactive site ready for deployment.

---

## Stage 6 — Deployment

Push the project live.

**Input:** the polished site from Stage 5.

**Actions:**
- Connect the GitHub repo to a hosting platform, or upload a ZIP archive.
- Configure build settings if needed.
- Attach a custom domain.
- Run a final check on the live URL.

**Tools:** Vercel · Netlify · Hostinger

**Output:** a live URL — the site is accessible and ready to share.

---

## Full workflow at a glance

| # | Stage | Output | Tools |
|---|-------|--------|-------|
| 01 | Ideas & References | Reference screenshot | Pinterest · Landbook · Dribbble |
| 02 | Visual Generation | Clean PNG / JPG (16:9, no text) | ChatGPT/DALL·E 3 · Nano Banana · Google AI Studio |
| 03 | Animation | Seamless MP4 loop | Kling AI · Seedance |
| 04 | Builder Assembly | Working site code w/ video bg | AntiGravity · Bolt.new · Lovable |
| 05 | Integration & Polish | Polished interactive site | Claude Code · Google AI Studio |
| 06 | Deployment | Live URL | Vercel · Netlify · Hostinger |

**Definition of done:** a live, accessible URL showing an animated hero section with a seamless looping video background, mouse parallax, intentional typography, and clean mobile layout.
