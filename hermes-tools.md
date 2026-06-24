# Hermes — Tools & Environment

> One of three composable files: **voice** (`hermes-voice.md`), **policy**
> (`hermes-policy.md`), and **tools** (this file). This file holds the
> environment-specific plumbing — tool schemas, harness conventions, output channels —
> kept out of the voice file because it changes per deployment and would otherwise
> drown the behavioral guidance.

This is a scaffold. The behavioral *principles* for tool use live in the SPIDER
section of `hermes-voice.md` (surface failures, gate preconditions, isolate context,
delegate minimally, escalate clearly, retry with structure). Those are model- and
environment-agnostic and should not be duplicated here. This file holds only the
concrete, swappable specifics.

---

## Available tools

For each tool Hermes can call, document:

- **Name** — exact identifier.
- **Purpose** — one line: when to reach for it.
- **Schema** — parameters, types, required vs optional. (Paste the real schema; do not
  paraphrase it — the model needs exact parameter names.)
- **When to use / when not** — a short trigger and a short anti-trigger, ideally with
  one example each. Anti-triggers prevent over-calling.

> Example stub:
> **`web_search`** — find current or post-cutoff information.
> Use for: prices, recent events, current office-holders, anything that may have
> changed. Don't use for: stable facts Hermes already knows, or pure reasoning tasks.

## Environment conventions

- **Output channels** — where text goes, how files are surfaced to the user, any
  rendering rules (markdown, artifacts, etc.). [Define for your harness.]
- **Working directories / scratch space** — where temporary vs final outputs belong.
- **Permissions / confirmation** — which actions need the user's explicit go-ahead
  (anything destructive, irreversible, or outward-facing).
- **Citation / attribution format** — how tool-derived claims are attributed.

---

> Maintenance note: this file is expected to change most often. Keep schemas pasted
> verbatim and current — a stale parameter name here causes silent tool failures that
> the voice and policy files can't compensate for.
