# Hermes — System Prompt (Index)

Hermes's system prompt is split into three composable files so each can be edited,
audited, and swapped independently. Assemble them in this order:

1. **`hermes-voice.md`** — character, tone, working method. Structured on PRECISE
   (spec), SPIDER (tools & failure handling), and CALM (length & context discipline).
   Model-agnostic. Changes rarely.
2. **`hermes-policy.md`** — content boundaries and refusals. The "gates and
   restrictions." Kept separate so voice stays readable and policy can be enforced or
   swapped per deployment. Changes when jurisdiction or enforcement layer changes.
3. **`hermes-tools.md`** — environment-specific plumbing: tool schemas, output
   channels, harness conventions. Changes most often.

## Why the split

The original single prompt mixed three jobs — voice, policy, and tool wiring — into one
document, which is why it ballooned and contradicted itself. Separating them means:

- editing Hermes's *character* never risks touching a tool schema or a safety gate;
- *policy* can live where it's actually enforced rather than as prose a user could talk
  around;
- *tool* details (which churn constantly) don't drown the behavioral guidance.

## Assembly

Concatenate in order (voice → policy → tools), or load them as separate system-prompt
segments if your harness supports it. Voice is authoritative for *how* Hermes behaves;
policy is authoritative for *what* it will and won't do; tools is authoritative for
*what it can call and how*. Where they appear to overlap (e.g. the medical/legal
information-not-advice posture), voice states the behavior and policy holds the
binding version.

## One thing to verify before shipping

`hermes-policy.md` is a scaffold. Empty sections are not "no restriction" — decide each
one deliberately. Child safety in particular should be enforced at a real layer (not
prose alone) before Hermes is deployed.
