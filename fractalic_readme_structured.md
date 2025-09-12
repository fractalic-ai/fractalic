# Fractalic

[![PyPI Version](https://img.shields.io/pypi/v/fractalic.svg)](https://pypi.org/project/fractalic/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt) ![Status](https://img.shields.io/badge/status-early--adopter-orange) ![Python](https://img.shields.io/badge/python-3.11+-blue.svg) [![Docs](https://img.shields.io/badge/docs-reference-purple)](docs/)

> Build & evolve AI workflows as executable Markdown. No brittle chains, no node graphs — just living documents you can read, diff and rerun.

![Fractalic Hero UI](docs/images/hero.png)
<!-- Replace with an actual screenshot: main editor (left: source.md, right: evolving context) -->

## One‑Line
Design, run and evolve multi‑model AI workflows in one executable Markdown file—no glue code—with precise context control, tool integration and git‑traceable reproducibility.

## The Core User Pains (Narrative)
Modern AI workflow stacks slow you down because they fragment intent (prompts in one place, scripts in another), hide which data actually reaches the model, require brittle orchestration code, and produce outputs that are hard to audit or reproduce. Non‑technical stakeholders cannot safely adjust logic; iterating small changes means touching multiple layers; adding tools or another model feels like surgery. Context leaks or bloat silently reduce quality. Debugging answers becomes guesswork.

## How Fractalic Shifts the Model
Fractalic collapses specification, execution, evolution and provenance into a single artifact: a Markdown document. You write what you want, reference exactly the knowledge the model should see, run it, watch deterministic changes appended or replaced, and immediately diff the result. No hidden stuffing, no accidental inclusion of future sections, no implicit waterfalls.

Key shifts:
- One medium (Markdown + tiny YAML) instead of code + config + chains
- Explicit, selectable context instead of implicit global stuffing
- Deterministic merge semantics (append / replace / isolate) instead of ad‑hoc string munging
- Tool outputs become first‑class knowledge blocks instead of raw logs
- Every execution produces a semantic snapshot (`.ctx`) you can diff & roll back
- Any document can be promoted to an agent simply by adding `@return`

## Context Architecture
Fractalic treats “context” as layered and programmable:
1. Data / Knowledge Context – Hierarchical tree of headings (blocks). This is your structured knowledge base inside the document.
2. Execution Context – Ordered sequence of operations (`@llm`, `@shell`, `@run`, generated operations) plus their merge effects; acts as a deterministic state machine.
3. Model (LLM) Context Window – Precisely the subset you explicitly reference (single block, branch with `/*`, or assembled fragments). Nothing else is silently injected.

Because the full document is parsed into an addressable tree, you can perform CRUD on blocks: create, append, replace, target sub‑trees, or feed only curated slices to a model. That gives surgical control over what the AI reasons about while keeping a human‑readable narrative.

## Universal Executable Documents
Every Fractalic document is simultaneously:
- Specification (what you intend)
- Runtime (operations mutate context deterministically)
- Module / Agent (it can be invoked via `@run` with parameters)
- Reusable interface (it can expose a contract using `@return`)
- Audit log (each run materializes into a diffable `.ctx` state)

You can:
- Invoke it from another document (`@run file: path/to/file.md prompt: ...`)
- Pass parameters as headings or prompts
- Return structured blocks or generated operations back to the parent
- Export artefacts to external systems (via shell / API calls) while retaining semantic provenance

## Execution Control Surface
Fractalic gives you a minimal, orthogonal set of operation primitives:
- `@llm` – Perform a model call on explicitly selected blocks; can specify model/provider, validators (planned), tools, merge mode, header behavior.
- `@shell` – Execute deterministic OS / CLI / API commands; captured stdout becomes an addressable block immediately available to subsequent operations.
- `@run` – Invoke another Markdown file in an isolated context; parameters (prompt or block injections) are placed at the top; returned blocks merge back according to specified mode.
- `@import` – Bring in reusable identities, templates, prior results or knowledge modules without executing their operations (or with, depending on mode).
- `@return` – Stop execution of the current document/agent and emit the selected block(s) or prompt content to the caller.

## LLM‑Driven Dynamic Orchestration
Beyond static operations you can hand control loops to the model itself using internal tooling (examples):
- `@llm\ntools: fractalic_run` – The model operates in an agent loop where it can decide—under your explicit rules—when to spawn sub‑documents (`@run`), which parameters to pass, and when to terminate.
- `@llm\ntools: fractalic_opgen` – The model generates future operations (including conditional branches, additional `@llm`, `@shell`, `@run` steps) that are inserted into the execution stream and then executed deterministically. With `run-once: true` you can produce a guarded single execution frame or allow recursive unfolding.

This pattern lets you prototype self‑extending workflows (context‑aware reflection loops, adaptive planning, consensus or validator passes) without custom framework code.

## Tool Integration (Unified)
Out of the box you can integrate:
- MCP tools – Structured external capability surface (knowledge bases, SaaS, internal APIs)
- Python tools – Lightweight wrappers or simple CLI commands promoted to operations
- Deterministic `@shell` – System utilities, REST calls (`curl`), data processing scripts
All results flow *back* into the knowledge tree, enriching subsequent reasoning. You are not reading logs—you are evolving structured context.

## Precise Context Control (Why It Matters)
Accidental prompt over‑inclusion leads to cost spikes, degraded answer quality and leakage of future steps. Fractalic’s block addressing (`block_uri: id`, wildcards for branches, explicit lists) ensures models only see curated scope. Refining a workflow becomes: reorganize headings, narrow selection, re‑run—no refactor of orchestration code.

## Reproducibility & Audit Trail
Each run emits:
- Source document (unchanged)
- Context snapshot `.ctx` (post‑execution structure)
- Deterministic ordering of operations and their merge effects
This makes differences semantic (“risk section expanded” / “tool output appended”) instead of opaque diff noise.

## Faster Evolution (Beyond “Fast Iteration”)
Iteration speed comes from eliminating translation layers: the document you think in *is* the one you run. Evolve structure only when needed; start with a single `@llm`, later isolate blocks, then extract a reusable agent—without changing mediums.

## What Each Claim Means (Value Breakdown)
No glue code – 3–6 line YAML ops instead of orchestration classes.
Multi‑model workflows – Assign different providers/models per `@llm` or per generated step.
Tool integration – MCP, Python and shell unify external capability; outputs re‑enter reasoning automatically.
Precise context – Only referenced blocks; no hidden stuffing.
Git‑traceable – Diff human semantic evolution, not just raw text churn.
Executable document – Requirements, operations, artefacts and returns co‑exist; no drift.
Auditable agents – Plain files with explicit inputs/outputs; easy to review.
Adaptive orchestration – Let models propose next steps safely within deterministic execution rules.

## Quick Glimpse
```markdown
# Idea {id=idea}
AI workflow IDE concept.

@llm
prompt: "Generate 3 tagline options."
block: { block_uri: idea }
mode: append
```
Run → output is appended beneath `# Idea`.

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
Result: Narrative returned; full evolution diffable.

## Common Uses
- Research → planning → synthesis → reporting chains
- Multi‑role / multi‑model agent orchestration (planner / executor / validator)
- Context‑aware tool calling (data fetch → transform → reasoning)
- Self‑refining documents (reflection passes, quality validators)
- Internal knowledge playbooks that stay executable
- Rapid prototyping of AI product features & SaaS flows

## Key Advantages (Snapshot)
- Context as a programmable tree
- Deterministic evolution & merge semantics
- Zero framework overhead
- Reusable agent modules (`@run` / `@return`)
- Unified tool surface (MCP, Python, shell)
- Multi‑provider neutrality
- Git native provenance
- Extensible via model‑generated operations

## Getting Started (60s)
```bash
pip install fractalic
fractalic init demo
cd demo
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
- Progressive elaboration
- Everything diffable & auditable

## Roadmap (Condensed)
Current: multi‑model, MCP, shell, agents, git sessions.
Next: knowledge graph memory, validators for `@llm`, tool call loop enrichment, artefact tagging, cost/telemetry panel.
Future: visual notebook overlay, adaptive context compression, multi‑language authoring, agent marketplace.

## When It’s Not a Fit
- Ultra‑low latency microservices
- Heavy numerical / data engineering pipelines (use Spark / SQL)
- Pure real‑time streaming scenarios

## Screenshots (Optional)
Editor (source vs evolving context) – `docs/images/editor.png`
Diff view (.md vs .ctx) – `docs/images/diff.png`
Multi‑model run trace – `docs/images/models.png`
Tool execution panel – `docs/images/tools.png`

## License
MIT

## Status
Early but production‑experimented. Used in real multi‑agent + tooling scenarios.

## Vision
Executable documentation as the universal interface between humans and AI.

---
Start with one heading and one `@llm`. Let the document grow into a system.
