# Installing the five web-design tools

Run these once. Two are **skills** (dropped into `.claude/skills/`), three are
**MCP servers** (registered with `claude mcp add`). After installing, restart
Claude Code (or start a new session) so the tools load, then run `/web-design`.

> Names and exact install commands drift over time. If a command 404s or a
> package moved, search for the current source before improvising, and prefer
> the tool's official README.

## MCP servers

### 1. shadcn MCP — component library

```bash
claude mcp add shadcn -- npx -y shadcn@latest mcp
```

Requires a `components.json` in the project (run `npx shadcn@latest init` in the
web project first if it doesn't exist).

### 4. Chrome DevTools MCP — drive a real browser

```bash
claude mcp add chrome-devtools -- npx -y chrome-devtools-mcp@latest
```

Needs a Chrome/Chromium install. In this environment Chromium is preinstalled.

### 5. Magic MCP by 21st.dev — animated UI add-ons

```bash
claude mcp add magic -- npx -y @21st-dev/magic@latest
```

Requires a free API key from https://21st.dev — set it when prompted or via the
`API_KEY` env the package documents.

## Skills

Skills are folders containing a `SKILL.md`. Install by placing them under
`.claude/skills/` (project) or `~/.claude/skills/` (global).

### 2. Web Design Guidelines skill (Vercel)

Vercel's web interface / design guidelines, packaged as a skill. Get it from
Vercel's published guidelines repo and copy the skill folder into
`.claude/skills/web-design-guidelines/`.

### 3. Front-End Design skill

A front-end aesthetic skill (fonts, color, spacing, motion). Copy its folder
into `.claude/skills/front-end-design/`.

## Verify the install

```bash
claude mcp list      # should show: shadcn, chrome-devtools, magic
```

In a session, `/help` or the skills list should show `web-design-guidelines`
and `front-end-design`. Then run `/web-design` to start the pipeline.
