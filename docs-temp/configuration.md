---
title: Configuration
description: Configure Fractalic settings, LLM providers, and environment variables
outline: deep
---

# Configuration

Status: Stable

## Purpose
Configuration tells Fractalic which model to use, where to find API keys, what extra tools (MCP servers) are available, and which runtime switches are on. This page explains the single file you touch most: `settings.toml`. After reading you should be able to (a) start with one model, (b) add a second model for polishing, (c) inject environment variables safely, and (d) enable a debugging flag then turn it back off.

## How to Think About It
You have one layered system:
1. `settings.toml` (your baseline defaults)
2. Environment variables (quick overrides / secret injection)
3. Programmatic / future CLI overrides (highest)
If a value appears in more than one place the higher layer wins.

Keep it simple early: one model table + a default. Expand only when you feel a real need (cheaper drafts, higher quality review, special proxy, etc.).

## First Run (Minimal)
Create or copy a starter file:
```toml
defaultProvider = "openrouter/openai/gpt-5-mini"

[settings."openrouter/openai/gpt-5-mini"]
apiKey = "sk-REPLACE"   # Put your real key here (never commit it)
model = "openrouter/openai/gpt-5-mini"
```
That is enough. Fractalic will use this model whenever an `@llm` block does not specify `model:`.

## Full File Anatomy
Top sections you may see:
- `defaultProvider` – name of the model table to use by default.
- `environment` – list of key/value pairs exported to the process (used by tools or shell steps).
- `[runtime]` – small execution toggles (currently only one that matters for most users).
- `[mcp]` – optional list of MCP server URLs (see limitation below).
- `[settings.<alias>]` – one table per logical model configuration.

Model table fields (you only need `apiKey` + `model` at first):
- `apiKey`: provider key.
- `model`: canonical model string (sometimes same as table name).
- `temperature`, `topP`, `topK`: sampling controls (leave defaults unless you know why to change them).
- `contextSize`: informational (lets you remember the intended max window size; not enforced here).
- `base_url`: set only if using a proxy or alternative gateway.

## Adding a Second Model (Draft + Polish Pattern)
Reason: save cost by drafting with a cheaper model then refining with a higher quality one.
```toml
defaultProvider = "draft"

[settings.draft]
apiKey = "sk-fast"
model = "openrouter/openai/gpt-5-mini"
temperature = 0.9   # a little creative

[settings.review]
apiKey = "sk-precise"
model = "openrouter/anthropic/claude-sonnet-4"
temperature = 0     # stable output
```
Usage in a document (final pass only):
```markdown
@llm
prompt: "Tighten wording. Do not add new facts."
model: review
block:
  - draft-section/*
```
You leave earlier calls model‑free so they use the default (`draft`).

## How Model Names Are Resolved (Practical View)
When you specify `model: review` Fractalic looks for a table named `review` under `[settings]`. If you pass a raw provider string (like `openrouter/openai/gpt-5-mini`) it tries:
1. Direct table name match
2. Table whose internal `model` field matches (with minor normalization of `.` / `-` / `_`)
If none match it lists available keys. Fix by either renaming the table or using the table’s exact alias. Keep aliases short and memorable.

## Environment Variable Injection
You can set variables for tools (search APIs, etc.) without exporting them globally.
```toml
[[environment]]
key = "TAVILY_API_KEY"
value = "tvly-REPLACE"

[[environment]]
key = "FRACTALIC_TRACE_TOKENS"
value = "1"  # enables token event logging
```
At startup these are exported just like normal environment variables for processes invoked by tools or `@shell`.
Guidelines:
- Do not overload this with dozens of unused keys.
- Do not commit real secrets to git (add file to `.gitignore` if necessary).
- For production: inject via deployment secrets or mount a file at runtime.

## Runtime Flag: `enableOperationsVisibility`
Default: `false`.
When `true` more internal operation nodes may appear implicitly in model context (beyond headings). This can help debugging “why did it answer that?” but it increases token cost and noise. Leave it off unless investigating a selection issue. Prefer explicit `block:` references instead of relying on implicit expansion.

## MCP Manager (Tool Extension)
Fractalic currently expects exactly one running MCP Manager instance. It listens (by default) on port `5859` and aggregates all underlying MCP services for you. You do NOT list multiple raw MCP servers in `settings.toml`—keep a single entry pointing at the manager.

Minimal configuration:
```toml
[mcp]
mcpServers = ["http://127.0.0.1:5859"]
```
Array form is retained for historical / forward compatibility; treat it as a single‑entry list.

Individual tool service definitions live in `mcp_servers.json` (persisted by the manager and always present). The manager reads that JSON and exposes a unified tool catalog through the single endpoint above. You normally do not edit the `mcpServers` array when adding or removing services—update (or let the UI / manager update) `mcp_servers.json` instead.

Practical workflow:
1. Start (or let the platform start) the MCP Manager (port 5859).
2. Ensure `settings.toml` has exactly the one line above.
3. Add / modify services via the manager (which persists changes into `mcp_servers.json`).
4. Restart or trigger a rescan if you need to reflect new tools immediately.

If the manager is down: tools will not load and the log will show connection attempts to `http://127.0.0.1:5859`.

Keep it simple: one endpoint, manager handles aggregation.

## Typical Scenarios
Scenario | What You Do
-------- | -----------
Just installed / single model | Set `defaultProvider`, add one model table with key
Need cheaper exploration | Make current model table the default, add a second “review” model for final pass
Need search tool auth | Add Tavily (or similar) key via `[[environment]]`
Debugging context selection | Temporarily set `runtime.enableOperationsVisibility = true` then revert
Add internal proxy | Set `base_url` in the relevant model table

## Pitfalls (And Fixes)
Problem | Cause | Fix
------- | ----- | ---
“model not found” error | Alias mismatch | Use table alias not raw marketing name or add correct table
Accidentally committed key | Secrets tracked | Rotate key, purge history if required, move real keys to untracked file
Huge token usage jump | Visibility flag left on | Turn off `enableOperationsVisibility`
Inconsistent tone across passes | Different temperatures | Set polishing model `temperature = 0`
Hard to reproduce output | Drift of defaultProvider | Pin explicit `model:` in final deterministic steps

## Cost & Performance Habits
- Keep only active model tables (delete unused experiments).
- Use a cheaper default draft model + explicit high‑quality review stage.
- Turn off debugging flags (visibility, verbose tracing) after issue resolved.
- Summarize or replace large blocks rather than keeping long append chains (see Optimization sections elsewhere).
- Enable token tracing only while diagnosing budget.

## Secret Handling Essentials
Do | Why
-- | ---
Keep real keys out of git | Prevent leak
Separate dev and prod files | Principle of least privilege
Rotate after exposure | Limit blast window
Use environment / secret store in CI | Central management
Limit environment entries | Smaller attack surface

## Quick Checklist Before Committing
- [ ] `defaultProvider` set and works
- [ ] All required model tables have `apiKey`
- [ ] No real secrets appear in diff
- [ ] Visibility flag off (unless intentionally on)
- [ ] Extra experimental models removed or commented
- [ ] Single MCP manager entry present (`http://127.0.0.1:5859` or your deployed host)

## See Also
- [Syntax Reference](syntax-reference.md)
- [Operations Reference](operations-reference.md)
- [Advanced LLM Features](advanced-llm-features.md)
- [Context Management](context-management.md)

Revision: v1.3 (removed question phrasing in MCP section)
