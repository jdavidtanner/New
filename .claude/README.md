# Claude Code tooling

This directory and the project's `.mcp.json` install a Claude Code toolkit
(website-design servers plus a set of workflow skills). Everything here is
**project-scoped** (committed to the repo) so it persists across web sessions
and ephemeral containers.

## MCP servers (`.mcp.json`)

| Server | Package | Purpose |
| --- | --- | --- |
| `github` | remote HTTP (`api.githubcopilot.com/mcp/`) | GitHub's official MCP server — manage repos/issues/PRs, monitor Actions runs, analyze build failures, manage releases, and review security findings. Needs `GITHUB_PAT`. |
| `shadcn` | `shadcn@latest mcp` | Pull production-ready, customizable UI components (buttons, menus, pricing sections, login forms) from the shadcn registry. |
| `chrome-devtools` | `chrome-devtools-mcp@latest` | Let Claude open the site in a real Chrome browser, inspect the DOM/console/network, and self-correct what it builds. |
| `magic` | `@21st-dev/magic@latest` | 21st.dev component generator — hero animations, button animations, and cohesive full-site styles. |

After pulling these changes, restart Claude Code and run `/mcp` to confirm each
server shows **Connected**.

### Required setup

- **Magic MCP** needs an API key. Get one at https://21st.dev/magic/console and
  export it as `MAGIC_API_KEY` (see `.env.example`). The `.mcp.json` reads it via
  `${MAGIC_API_KEY}`.
- **shadcn** and **chrome-devtools** run via `npx` with no key. Chrome DevTools
  MCP requires a Chrome/Chromium install available to the runtime.
- **GitHub** uses the remote (HTTP) mode with a Personal Access Token. Create a
  PAT at https://github.com/settings/personal-access-tokens and export it as
  `GITHUB_PAT`; the `.mcp.json` sends it as `Authorization: Bearer ${GITHUB_PAT}`.
  If your MCP host can't reach remote servers, swap in GitHub's local Docker
  image (`ghcr.io/github/github-mcp-server`) instead.

## Skills (`.claude/skills/`)

### Website design

| Skill | Source | Purpose |
| --- | --- | --- |
| `frontend-design` | Anthropic (`anthropics/claude-code`) | Design-lead guidance — distinctive typography, palettes, and motion so output doesn't read as a templated "AI website". Activates automatically when building UI. |
| `web-design-guidelines` | Vercel (`vercel-labs/agent-skills`) | Audits finished UI against Vercel's Web Interface Guidelines (accessibility, keyboard support, forms, animation, performance) and reports `file:line` fixes. |

Invoke explicitly with `/frontend-design` or `/web-design-guidelines`, or just
ask Claude to build or review UI and the relevant skill engages.

### Design quality (motion, polish, taste)

| Skill | Source | Purpose | External deps |
| --- | --- | --- | --- |
| `emil-design-eng` | `emilkowalski/skills` | Emil Kowalski's craft sensibility — purposeful motion, easing curves, and the invisible details that make UI feel smooth and alive (incl. knowing when *not* to animate). | None |
| `review-animations` | `emilkowalski/skills` | Audit existing animations against the same motion principles. | None |
| `impeccable` | `pbakaus/impeccable` | Shared design vocabulary that fixes spacing, typography, layout, alignment, color, and motion. Ships 44 deterministic anti-pattern detectors and a live-browser iteration loop. | The committed skill is self-contained. For the full 23-command suite + agent + hooks across tools, run `npx impeccable install` then `/impeccable init`. |
| `design-taste-frontend` | `Leonxlnx/taste-skill` | The "Taste" skill — anti-slop landing pages/portfolios/redesigns. Reads the brief, infers a design direction, and ships interfaces that don't look templated. | None. The source repo also has 12 optional variants (minimalist, brutalist, soft, redesign, brandkit, image-to-code, imagegen, etc.) — not installed to avoid clutter. |

### Workflow & productivity (part one)

| Skill | Source | Purpose | External deps |
| --- | --- | --- | --- |
| `skill-creator` | Anthropic (`anthropics/skills`) | Create, edit, and optimize skills; run evals and benchmark skill performance. | None |
| `autoresearch` | `uditgoenka/autoresearch` (Karpathy-inspired) | Autonomous improvement loop: set a goal + a mechanical metric, then modify → verify → keep/discard until it converges. | None for the skill. Full 14-command + 9-hook suite needs the plugin (below). |
| `obsidian-markdown`, `obsidian-bases`, `json-canvas`, `obsidian-cli`, `defuddle` | `kepano/obsidian-skills` (Obsidian's CEO) | Teach Claude to work with Obsidian vaults: Obsidian-flavored Markdown, `.base` databases, JSON Canvas, vault automation via the Obsidian CLI, and clean web→markdown extraction. | `obsidian-cli` needs the [Obsidian CLI](https://help.obsidian.md/cli); `defuddle` needs the Defuddle CLI. The format skills (markdown/bases/canvas) work standalone. |
| `notebooklm` | `teng-lin/notebooklm-py` (unofficial) | Drive Google NotebookLM from Claude Code — create notebooks, add sources, query, generate podcasts/artifacts. | Needs `pip install "notebooklm-py[browser]"`, `playwright install chromium`, then `notebooklm login`. Uses **undocumented** Google APIs — may break without notice. |

## Not committed — interactive / credentialed installs

These were requested but can't live in the repo as files; run them in your own
Claude Code session:

- **Codex plugin** (`openai/codex-plugin-cc`) — adversarial second-opinion
  reviews from OpenAI Codex. Requires Node 18.18+ and a ChatGPT/OpenAI account.
  Install:
  ```
  /plugin marketplace add openai/codex-plugin-cc
  /plugin install codex@openai-codex
  /codex:setup
  ```
  Then use `/codex:review` or `/codex:adversarial-review`.
- **AutoResearch full plugin** — the committed `autoresearch` skill gives the
  core loop, but the complete 14-command set and 9 safety hooks install as a
  plugin: `/plugin marketplace add uditgoenka/autoresearch`.
- **NotebookLM auth** — after `pip install "notebooklm-py[browser]"` and
  `playwright install chromium`, run `notebooklm login` once to authenticate the
  browser session the `notebooklm` skill drives.
