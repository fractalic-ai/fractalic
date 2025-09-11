<p align="center">
  <img src="https://raw.githubusercontent.com/fractalic-ai/fractalic/main/docs/images/fractalic_hero.png" alt="Fractalic Hero Image">
</p>

<h1 align="center">Fractalic</h1>

<p align="center">
  <strong>The Agentic Development Environment.</strong>
  <br>
  Turn simple documents into powerful, AI-native applications.
</p>

<p align="center">
    <a href="https://github.com/fractalic-ai/fractalic/blob/main/LICENSE.txt"><img alt="License" src="https://img.shields.io/github/license/fractalic-ai/fractalic?style=flat-square&color=blue"></a>
    <a href="https://fractalic.ai/docs/"><img alt="Documentation" src="https://img.shields.io/website?url=https%3A%2F%2Ffractalic.ai%2Fdocs&label=docs&style=flat-square&up_message=online&color=brightgreen"></a>
    <a href="https://discord.gg/DHbnvxAT"><img alt="Discord" src="https://img.shields.io/badge/discord-join%20chat-5865F2?style=flat-square&logo=discord&logoColor=white"></a>
</p>

---

**Fractalic** is a new kind of development environment where your document *is* the program. Instead of writing complex scripts, you build AI-powered workflows using plain Markdown and simple YAML commands. The document evolves as operations run, creating a transparent, readable, and powerful way to orchestrate AI.

## Why Fractalic

- Build agents by writing Markdown + minimal YAML. No frameworks to wire.
- Deploy the same documents as services with a REST API.
- Use your tools via MCP or local shell/Python.
- Git-backed history, token/cost tracking, and diffs by default.

### Core hooks (what’s uniquely valuable)
1) Deterministic execution with embedded operations — runs top-to-bottom with explicit append/replace semantics.
2) Complete execution trace — every turn, tool call, and token cost recorded; easy to diff and audit.
3) Agentic tool use — governed tool loops with MCP services or local tools.
4) Documents are agents — call docs with parameters, return structured results, isolate context.
5) Structured tree context — documents are block trees with IDs; select with `blocks:` and CRUD them.

See also: `docs/core-concepts.md` (§ Blocks and IDs), `docs/advanced-llm-features.md` (§ Tool loops), `docs/context-management.md`.

## Quick Start

Run with Docker:

```bash
docker run -d --name fractalic \
  --network bridge \
  -p 3000:3000 -p 8000:8000 -p 8001:8001 -p 5859:5859 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --env HOST=0.0.0.0 \
  ghcr.io/fractalic-ai/fractalic:main
```
Then open http://localhost:3000. More options: `docs/quick-start.md`.

### Your first program (`hello-fractalic.md`)

```markdown
# Goal: Create a simple greeting {id=goal}
I want to see how Fractalic can use an LLM to generate text based on a prompt and context from this document.

@llm
prompt: Create a friendly, one-sentence greeting that mentions how Fractalic turns documents into programs.
blocks: goal
use-header: "# AI-Generated Greeting"
```

Run it in the UI or call the API:

```bash
curl -X POST http://localhost:8001/execute \
  -H "Content-Type: application/json" \
  -d '{"filename": "./hello-fractalic.md"}'
```

## What you can build (real use cases)

All examples assume the service is enabled in `mcp_servers.json` and authenticated if needed. Learn more: `docs/mcp-integration.md`.

### Memory notes with MCP Memory

```markdown
# Task {id=task}
Store a fact, list keys, then summarize changes.

@llm
prompt: |
  1) Store the note: "Fractalic launched an MCP-powered agent".
  2) List current memory keys.
  3) Summarize the change in one sentence.
blocks: task
use-header: "# Memory Result {id=memory-result}"
tools:
  - mcp/memory-stdio-server
tools-turns-max: 2

@return
block: memory-result
```

### Create a Notion page (MCP Notion)

```markdown
# Task {id=task}
Create a project note in Notion with today’s date and a short status.

@llm
prompt: |
  Create a Notion page titled "AI Weekly Status — {{today}}" with a bullet list of 3 highlights.
  If a database is required, use a reasonable default the server exposes.
blocks: task
use-header: "# Notion Result {id=notion-result}"
tools:
  - mcp/notion
tools-turns-max: 3

@return
block: notion-result
```

### Add a Google Calendar event (MCP Zapier)

```markdown
# Task {id=task}
Schedule a 30-minute sync tomorrow at 10:00 for the AI team.

@llm
prompt: |
  Create a calendar event titled "AI Team Sync" for tomorrow 10:00–10:30.
  Include Meet link if available.
blocks: task
use-header: "# Calendar Result {id=calendar-result}"
tools:
  - mcp/Zapier
tools-turns-max: 3

@return
block: calendar-result
```

### When a tool is enough (local shell_tool)

```markdown
# Task {id=task}
List current directory and summarize files.

@llm
prompt: |
  1) List files in the current directory.
  2) Summarize by type and count.
blocks: task
use-header: "# Shell Summary {id=shell-summary}"
tools:
  - shell_tool
tools-turns-max: 2

@return
block: shell-summary
```

More patterns: `docs/advanced-llm-features.md` (§ Tool loops, JSON-only outputs).

## Documents as agents (call docs with docs)

- Use `@run` for static, synchronous sub-workflows. See `docs/core-concepts.md` (§ @run).
- Use the `fractalic_run` tool to let an LLM dynamically choose and execute another document. See `docs/agent-modular-workflows.md` (§ Dynamic Agent Calling).

Example: router calls a specialized agent with parameters and returns a clean result.

```markdown
# Router Task {id=router-task}
Decide which agent to call and return only the agent’s final result.

@llm
prompt: |
  If the request mentions "notion", call the notion agent.
  Otherwise call the memory agent.
  Return only the callee’s final result.
blocks: router-task
use-header: "# Router Result {id=router-result}"
tools:
  - fractalic_run
tools-turns-max: 2
```

Learn: `docs/operations-reference.md` (§ Calling another agent via fractalic_run).

## Integrations and credits

- Model routing via LiteLLM — one OpenAI-compatible interface to 100+ providers. Project: https://github.com/BerriAI/litellm
- MCP client and manager built on FastMCP — high-level, Pythonic MCP tooling. Project: https://github.com/jlowin/fastmcp

Fractalic integrates with and credits these upstream projects. See licenses in their repos.

## Out-of-the-box tools (packs)

Explore the tools marketplace: https://github.com/fractalic-ai/fractalic-tools
- HubSpot automation suite (CRM): 40+ tools incl. contacts, deals, tickets, process discovery
- Discovery & Process Mining: event discovery, analytics
- OS pack: file, shell, text
- Telegram automation
- Web scraping/search

Add packs as MCP services or as local tools per repo instructions.

## Screenshots

<table>
  <tr>
    <td width="50%">
      <img src="docs/images/editor.png" alt="Fractalic Editor - Notebook-style UI with Markdown and YAML operations" />
      <p align="center"><em>Main Editor Interface</em></p>
    </td>
    <td width="50%">
      <img src="docs/images/notebook.png" alt="Notebook View - Interactive document execution with live results" />
      <p align="center"><em>Notebook Execution View</em></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="docs/images/tools.png" alt="MCP Tools Integration - Access external services via Model Context Protocol" />
      <p align="center"><em>MCP Tools Integration</em></p>
    </td>
    <td width="50%">
      <img src="docs/images/mcp.png" alt="MCP Manager - Unified tool and service management interface" />
      <p align="center"><em>MCP Manager Interface</em></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="docs/images/diff.png" alt="Git-backed Diffs - Complete execution trace with version control" />
      <p align="center"><em>Git-backed Execution Diffs</em></p>
    </td>
    <td width="50%">
      <img src="docs/images/inspector.png" alt="Debug Inspector - Deep inspection of execution state and variables" />
      <p align="center"><em>Debug Inspector</em></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="docs/images/inspector-messages.png" alt="Message Inspector - Detailed view of AI conversation turns and tool calls" />
      <p align="center"><em>Message Inspector</em></p>
    </td>
    <td width="50%">
      <img src="docs/images/markdown.png" alt="Markdown Editor - Clean document editing with syntax highlighting" />
      <p align="center"><em>Markdown Editor</em></p>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="docs/images/deploy.png" alt="Deployment Dashboard - One-click containerization and service deployment" width="50%" />
      <p align="center"><em>Deployment Dashboard</em></p>
    </td>
  </tr>
</table>

## Architecture (short)

- ADE UI (port 8000) — notebook-like UI with streaming, diffs, debug
- AI Server (port 8001+) — FastAPI `/execute` with Swagger `/docs`
- Unified MCP Manager (port 5859) — OAuth, caching, tool catalog
- Git-backed sessions, token/cost tracker, Docker-first deploy

Details: `docs/ui-server-api.md`, `docs/mcp-integration.md`.

## See also

- Introduction: `docs/introduction.md`
- Core Concepts: `docs/core-concepts.md`
- Operations Reference: `docs/operations-reference.md`
- Advanced LLM Features: `docs/advanced-llm-features.md`
- MCP Integration: `docs/mcp-integration.md`
- Context Management: `docs/context-management.md`
- Deployment Guide: `docs/deployment-guide.md`



