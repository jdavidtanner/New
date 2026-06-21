# Website-designer tooling

This directory and the project's `.mcp.json` install a website-design toolkit
for Claude Code. Everything is **project-scoped** (committed to the repo) so it
persists across web sessions and ephemeral containers.

## MCP servers (`.mcp.json`)

| Server | Package | Purpose |
| --- | --- | --- |
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

## Skills (`.claude/skills/`)

| Skill | Source | Purpose |
| --- | --- | --- |
| `frontend-design` | Anthropic (`anthropics/claude-code`) | Design-lead guidance — distinctive typography, palettes, and motion so output doesn't read as a templated "AI website". Activates automatically when building UI. |
| `web-design-guidelines` | Vercel (`vercel-labs/agent-skills`) | Audits finished UI against Vercel's Web Interface Guidelines (accessibility, keyboard support, forms, animation, performance) and reports `file:line` fixes. |

Invoke a skill explicitly with `/frontend-design` or `/web-design-guidelines`,
or just ask Claude to build or review UI and the relevant skill engages.
