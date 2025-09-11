# Fractalic

[![PyPI Version](https://img.shields.io/pypi/v/fractalic.svg)](https://pypi.org/project/fractalic/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt) ![Status](https://img.shields.io/badge/status-early--adopter-orange) ![Python](https://img.shields.io/badge/python-3.11+-blue.svg) [![Docs](https://img.shields.io/badge/docs-reference-purple)](docs/)

> Build & evolve AI workflows as executable Markdown. No brittle chains, no node graphs — just living documents you can read, diff and rerun.

![Fractalic Hero UI](docs/images/hero.png)
<!-- Replace with an actual screenshot: main editor (left: source.md, right: evolving context) -->

## One‑Line
Design, run and evolve multi‑model AI workflows in one executable Markdown file—no glue code—with precise context control, tool integration and git‑traceable reproducibility.

## The Core User Pains (Narrative)
Modern AI workflow building is still slowed by:
- Scattered scripts and fragile glue layers that break when intent changes.
- Hidden or overstuffed prompts causing irrelevant context leakage and inconsistent outputs.
- Multiple, inconsistent ways to call tools (shell, Python, APIs, MCP) → duplication & drift.
- Manual orchestration of different models or agent roles (planner / executor / validator).
- Poor auditability: logs exist but lack semantic structure; hard to answer “why this answer?”.
- Steep onboarding curve for non‑engineering stakeholders.

## How Fractalic Addresses These
Fractalic collapses specification, execution, evolution and audit into a single artifact:
- One document = intent + structured knowledge + operations + historical evolution.
- Deterministic context selection: only referenced blocks enter a model call.
- Unified tool surface (MCP, Python tools, deterministic shell) feeding results back as addressable blocks.
- Multi‑model neutrality: assign any provider per operation, combine roles without extra orchestration code.
- Git‑native semantic history: every run emits a `.ctx` snapshot for diff / rollback / analysis.
- Any document can become an agent simply by returning a result (`@return`).
- Progressive elaboration: start as rough notes, organically layer structure, agents, tools.

## Unified Context Architecture
In Fractalic there is no separation between “data context”, “execution context” and “LLM context” — they are the SAME evolving structure:
1. The Markdown document is parsed into a hierarchical knowledge tree (blocks + operations).
2. Operations mutate or extend that tree (append / replace / create / import).
3. `@llm` calls see only the explicitly referenced subset of that tree.
4. Results (LLM output, tool output, sub‑run results) are re‑inserted as new or modified blocks, instantly addressable.
5. The evolving tree at each execution is persisted (.md source + .ctx after run) → reproducible state transition history.

This unified model enables:
- Precise, minimal prompt payloads.
- Deterministic reproducibility (semantic diffing of knowledge state).
- Composable agent layers (each in isolated or shared view of the tree).
- Future advanced features (auto‑compression, staged memory, validators) without changing authoring ergonomics.

## Execution Control: How the LLM Drives the Runtime
Fractalic exposes a minimal set of orthogonal primitives that let LLMs *govern* execution safely:

### 1. `@shell`
Deterministic execution of OS / CLI / API commands. Output is captured and inserted as a structured block (addressable via its auto or explicit ID). Enables runtime acquisition of specs (`--help`, Swagger, code introspection) that become immediate context for subsequent reasoning.

### 2. `@run`
Invokes another Markdown document (module / agent) with optional parameters (prompt text and/or block references). It executes in an isolated context (fresh tree), produces its internal evolution, and can return selected blocks using `@return`. Returned material is merged back into the caller’s context.

### 3. `@llm` with `tools: fractalic_run`
Provides an agentic loop where the model can decide (under given constraints you define in the surrounding prompt) WHEN to invoke subordinate agents and WHICH data blocks to pass. Fractalic mediates each invocation, preserves the full message trail, and re‑integrates outputs.

### 4. `@llm` with `tools: fractalic_opgen`
Allows the model to *generate future operations* (including conditional branches, validation passes, retry logic, or chained `@shell` / `@run` steps). Generated operations are appended to the execution queue. With `run-once: true` you can stage an execution stack where the model seeds the subsequent deterministic pipeline.

Together these create a controllable boundary between freeform reasoning and governed execution: the model can propose structure, but Fractalic enforces deterministic application and traceability.

## What Each Claim Means (Value Breakdown)
- **No glue code**: 3–6 lines of YAML replace custom wrapper classes / pipeline scripts.
- **Multi‑model workflows**: Assign specialized models per step (analysis, creative, verifier) without building a bespoke controller layer.
- **Tool unification**: MCP, Python tool specs, and shell become uniform producers of structured knowledge blocks.
- **Precise context control**: Block URIs and wildcards (`block_uri: research/*`) prevent accidental inclusion of unrelated sections.
- **Git‑traceable**: Each run’s `.ctx` file is a semantic snapshot → diff shows conceptual evolution, not raw log noise.
- **Executable document**: Specification and behavior co‑evolve; no separate “doc vs code” divergence.
- **Composable agents**: Any file + `@return` = reusable semantic module callable from others.
- **Iterative universality**: The same document can be: a prototype → a reusable agent → an internal API layer → an orchestrator of lower agents.

## Core Primitives (Mental Model)
| Primitive | Purpose | Mini Form |
|----------|---------|-----------|
| Knowledge Block | Heading + its content | `# Plan {id=plan}` |
| `@llm` | Model call over selected blocks | `@llm\nprompt: "Refine plan"\nblock: { block_uri: plan }` |
| `@import` | Bring another file / artefact | `@import\nfile: templates/identity.md` |
| `@run` | Execute another markdown as isolated agent | `@run\nfile: agents/research.md\nprompt: "Trends"` |
| `@shell` | Shell / CLI / API access | `@shell\ncommand: curl https://api/...` |
| `@return` | Stop & emit result | `@return\nblock: { block_uri: final/* }` |

## Quick Glimpse
```markdown
# Idea {id=idea}
AI workflow IDE concept.

@llm
prompt: "Generate 3 tagline options."
block: { block_uri: idea }
mode: append
```
Run → output is appended beneath `# Idea`. No framework code. No hidden state.

## End‑to‑End Scenario (Single File, ~8 Lines)
```markdown
# Launch Brief {id=brief}
We need a concise launch narrative for our new AI workflow feature.

@llm
prompt: "List 6 concrete user pains from the brief."
block: { block_uri: brief }
mode: append

@llm
prompt: "Cluster the pains above and name each cluster with a 2-word label."
block: { block_uri: brief/* }
mode: append

@shell
command: curl -s https://api.example.com/market/trends > trends.json

@llm
prompt: "Using clusters + trends.json (summarize it), draft a 150-word launch narrative."
block: { block_uri: brief/* }
use-header: "# Draft Narrative"

@return
block: { block_uri: draft-narrative }
```
Result: Narrative returned; full evolution diffable (`.md` vs `.ctx`).

## Common Uses
- Rapid prototyping of AI & SaaS automation flows
- Research → synthesis → reporting pipelines
- Planner / executor / validator multi‑agent chains
- Tool‑augmented reasoning (API spec ingestion, code gen + shell verification)
- Self‑refining documents (reflection, validation, repair loops)
- Living internal playbooks / procedural knowledge stores

## Key Advantages (at a glance)
- Unified knowledge + execution + LLM context
- Deterministic & auditable evolution
- Zero framework tax (text → runtime)
- Composable agent modules (`@run` + `@return`)
- Integrated tool outputs as structured knowledge
- Multi‑provider model orchestration
- Git native provenance

## Getting Started (60s)
```bash
pip install fractalic
fractalic init demo
cd demo
# create workflow.md then:
fractalic run workflow.md
```
Minimal file:
```markdown
# Goal {id=goal}
Generate 5 focus areas for launch.

@llm
prompt: "List 5 focus areas."
block: { block_uri: goal }
```

## Design Principles
- Explicit over implicit
- Human‑readable over clever abstractions
- Declarative context control
- Progressive elaboration (start messy → structure later)
- Everything diffable

## Roadmap (Condensed)
Current: multi‑model, MCP, shell, agents, git sessions.
Next: knowledge graph memory, validators for `@llm`, tool call loop enrichment, artefact tagging, cost/telemetry panel.
Future: visual notebook overlay, adaptive context compression, multi‑language authoring, agent marketplace.

## When It’s Not a Fit
- Ultra‑low latency microservices
- Heavy numerical / data engineering pipelines (use Spark / SQL)
- Pure real‑time streaming scenarios

## Screenshots (Optional)
| View | Placeholder |
|------|-------------|
| Editor (source vs context) | `docs/images/editor.png` |
| Diff (.md vs .ctx) | `docs/images/diff.png` |
| Multi‑model run trace | `docs/images/models.png` |
| Tool execution panel | `docs/images/tools.png` |
<!-- Add real images in /docs/images and update paths -->

## License
MIT

## Status
Early but production‑experimented. Building with real multi‑agent + tooling scenarios.

## Vision
Executable documentation as the universal interface between humans and AI.

---
Ready to replace brittle chains with living documents? Start a file and run it.
