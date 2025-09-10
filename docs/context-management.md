---
title: Context Management
description: Master context control with block selection, wildcards, composition patterns, and token optimization
outline: deep
---

# Context Management

Purpose: How Fractalic builds and evolves the Context Graph (structured markdown blocks) so each operation sees exactly what it needs—while you keep size, cost, and clarity under control.

Why this matters (even if you are new to LLMs):
- Better answers: The model can only reason over what you give it. Clean, focused context = higher quality.
- Lower cost: Unpruned blocks (especially tool results) raise token usage every turn.
- Speed: Lean context shortens round‑trip time.
- Reliability: Stable IDs + controlled merges make flows reproducible.
- Safety: Smaller curated context reduces accidental leakage of irrelevant or sensitive text.

In short: Good context management is the difference between a drifting, expensive session and a precise, cheap, reliable one.

---
### Internal Table of Contents
- [Context Graph & Folder Analogy](#context-graph--folder-analogy)
- [Blocks, IDs & Paths](#blocks-ids--paths)
- [Selecting Content (`block_uri`)](#selecting-content-block_uri-mechanics)
- [Merge Strategies](#merge-strategies-mode--target-toblock_uri)
- [Execution Lifecycle](#execution-lifecycle-what-happens-internally--simplified)
- [Context Scoping (`context` field)](#context-scoping-in-llm)
- [Tool & Agentic Loop Context](#tool--agentic-loop-context)
  - [Dual AST (Immediate Tool Block Availability)](#dual-ast-why-new-blocks-are-instantly-selectable)
  - [Growth Control Checklist](#growth-control-checklist)
- [Managing Size & Cost (Compression Strategies)](#managing-size--cost-compression-strategies)
- [Patterns & Best Practices](#patterns--best-practices)
- [Common Pitfalls](#common-pitfalls)
- [Handy Examples](#handy-examples)
- [Quick Checklist (Before Big Run)](#quick-checklist-before-big-run)
- [Quick Reference Table](#quick-reference-table)
- [Zero → Hero Tool Loop Recap](#zero--hero-tool-loop-recap)
- [Dual Tool Layer (Dual AST) – How Tool Output Becomes Usable Instantly](#dual-tool-layer-dual-ast--how-tool-output-becomes-usable-instantly)
- [Visibility Control (`enableOperationsVisibility`)](#visibility-control-enableoperationsvisibility)
- [Token Mechanics & Growth Dynamics](#token-mechanics--growth-dynamics)
- [Distillation Workflow (From Raw → Final)](#distillation-workflow-from-raw--final)
- [Cross References](#cross-references)

---
## Context Graph & Folder Analogy
Fractalic turns your markdown into a Context Graph:
- Each heading = a node (block)
- Its body + nested headings = content / children
- Siblings = parallel branches
- The graph grows as you refine, import, run tools, or generate with `@llm`.

Folder Analogy:
- Heading = folder
- Text under heading (until next peer) = files
- Subheading = subfolder
- Add stable labels (IDs) to reuse.

Both views describe an ordered, hierarchical set of addressable blocks.

## Blocks, IDs & Paths
Add an explicit ID when you intend to reuse or refine:
```
# User Research Summary {id=research-summary}
```
No ID → auto slug (changes if you rename). Stable reuse needs explicit IDs early.

Path = slash of ancestor IDs/slugs: `research-summary/findings`. Wildcards:
- `section/*` = all direct children of `section`
- Use explicit IDs once structure stabilizes (reduces accidental over-selection).

## Selecting Content (`block_uri` Mechanics)
Ways to specify:
```yaml
block:
  block_uri: research-summary          # single
```
```yaml
block:
  block_uri:                           # multiple
    - research-summary
    - competitor-notes
```
```yaml
block:
  block_uri: research/*                # wildcard children
```
Resolution Order:
1. Expand wildcards → concrete block paths
2. De‑duplicate (first mention wins order)
3. Fetch blocks in order
4. Concatenate their bodies (becomes part of prompt input)
Missing path → error (explicit feedback, no silent skipping).

Guideline: Start broad while exploring → tighten to explicit IDs to reduce noise & cost.

## Merge Strategies (`mode`) + Target (`to.block_uri`)
After an operation produces output blocks:
- append (default): add after existing children
- prepend: insert before existing children
- replace: substitute target block body (preferred for iterative refinement)

Target defaults to the block containing the operation unless you set:
```yaml
to:
  block_uri: results
```
Use `replace` once a draft shape is stable to avoid unbounded growth.

## Execution Lifecycle (What Happens Internally — Simplified)
1. Parse markdown → current Context Graph (AST)
2. Parse operation YAML
3. Resolve `block_uri` selections (and optional ambient lineage)
4. Assemble prompt (selected blocks + literal prompt)
5. Execute action (`@llm`, `@shell`, `@import`, `@run`)
6. Create new block(s)
7. Merge via `mode`
8. Graph updated (now selectable for future steps)

## Context Scoping in `@llm`
Field: `context`
- auto (default): If only a prompt is given, Fractalic may include preceding headings for light grounding
- none: Strict isolation—only explicitly selected blocks + prompt
- Explicit `block:` selection always wins
Use `context: none` for tests, JSON generation, or deterministic evaluation phases.

## Tool & Agentic Loop Context
When `@llm` uses `tools:`, each tool call adds new information that the next turn will see. Without pruning, context keeps enlarging.

Plain Flow:
1. Model decides to call a tool
2. Tool returns data
3. Data appears as new blocks right after the operation
4. Next turn: model sees original + all still-present tool blocks
5. You may summarize/replace/delete at any time

Core Effects:
- Every surviving tool block is resent each turn → token cost stacks
- Large MCP / API responses (issues lists, JSON payloads, schema-rich arrays) should be summarized early
- You control growth by compression and turn limits (`tools-turns-max: 2–4` typical)

Inline Summarization Pattern:
```markdown
@llm
prompt: |
  Use tools only if data missing.
  After each tool result: update '# Data Summary {id=data-summary}' (<=80 tokens). Discard raw bulk.
tools:
  - repo_stats
  - issue_search
tools-turns-max: 2
```
Result: One compact summary instead of accumulating raw clutter.

### Dual AST (Why New Blocks Are Instantly Selectable)
Internally a temporary tool layer is maintained while tools run. Useful tool output is inserted into the main graph immediately; you can reference it in the SAME or subsequent steps with normal `block_uri` values. No extra commands. You never manage two graphs manually—this just explains immediacy.

Guiding Idea: “Tool outputs are transient scratch results that appear as normal blocks—curate them early.”

### Growth Control Checklist
- Summarize long outputs → replace raw
- Cap tool turns
- Give summaries stable IDs
- Remove obsolete exploration blocks
- Run a clean isolated pass (`context: none`) for final structured outputs

## Managing Size & Cost (Compression Strategies)
Goal | Action
---- | ------
Large raw tool output | Summarize → `mode: replace`
Many similar tool blocks | Merge into single “Consolidated Data” block
Exploration done | Delete/archive noisy intermediates
Repeated wildcards | Switch to explicit IDs
Need deterministic final pass | Summaries only + `context: none`
Token spike | Audit biggest blocks → compress or remove

## Patterns & Best Practices
Need | Pattern
---- | -------
Refine safely | Draft (append) → converge (replace)
Summarize research | Raw import → compression → ultra‑summary
Parallel sub-work | Separate headings; later synthesis references only summaries
Reusable artifact | Add `{id=...}` before first cross-file reuse
Tool loop control | Provide explicit “Call a tool only if …” rule + turn cap

## Common Pitfalls
Pitfall | Why It Hurts | Fix
------- | ------------ | ---
Endless append cycles | File & token bloat | Switch to replace after shape stabilizes
Wildcards left in late phase | Pulls stale noise | Replace with explicit IDs
Huge untouched tool dumps | Token cost snowball | Summarize + replace
Lost references after rename | Auto slug changed | Use explicit IDs early
Bloated final model input | Unpruned exploration | Curate only needed summaries

## Handy Examples
Compare two options:
```markdown
@llm
prompt: "List key differences then recommend one."
block:
  - option-a
  - option-b
```
Structured JSON (isolated):
```markdown
@llm
prompt: "Return ONLY JSON with keys: title, risks[]"
block:
  block_uri: research-summary
use-header: none
context: none
```
Refine while preserving structure:
```markdown
@llm
prompt: "Improve tone. Keep bullet order & counts."
block:
  exec-summary
mode: replace
to: exec-summary
```

## Quick Checklist (Before Big Run)
- IDs assigned to key sections?
- Wildcards minimized?
- Tool outputs summarized?
- Turn cap set?
- Redundant drafts removed?
- Need isolation (`context: none`) now?

## Quick Reference Table
Concept | Meaning
------- | -------
Block | Heading + body (+ subtree)
ID | Stable label `{id=...}` for reuse
Path | Slash chain (`parent/child`)
Wildcard | `block/*` = direct children
Selection | Ordered concatenation of chosen blocks
Modes | append / prepend / replace
Context Modes | auto (implicit lineage) / none (strict)
Tool Growth | Each surviving tool block re-sent next turn
Compression Loop | Raw → summary → final artifact

## Zero → Hero Tool Loop Recap
1. Start `@llm` with tools
2. Tool(s) run → blocks appear
3. Summarize/replace raw outputs
4. Next turns reuse only curated summaries
5. Isolated final pass generates clean deliverable

If you never prune, cost + noise climb. Curate early, not later.

## Dual Tool Layer (Dual AST) – How Tool Output Becomes Usable Instantly
When you run an `@llm` with tools, Fractalic maintains a temporary “tool layer” while the model is calling tools. As soon as a tool returns something useful, Fractalic immediately inserts that content into the main Context Graph right after the operation. You don’t do anything special—this is automatic.

Why it matters:
- Immediate Reuse: You can reference a freshly created tool block in the very next (or same chain’s) `block_uri` selection.
- Uniform Handling: Tool-created blocks behave just like any other heading/body block (summarize, replace, delete, give them an ID, etc.).
- No Manual Sync: There is no separate editing step to “import” tool results—they are already merged.

Guiding Idea: “Tool outputs are transient scratch results that appear as normal blocks—curate them early.”

## Visibility Control (`enableOperationsVisibility`)
Fractalic normally feeds the model only heading blocks (and what you explicitly select). A setting `enableOperationsVisibility` can broaden this by letting additional operation nodes appear in the implicit context (e.g. non-heading structural nodes). This gives more granular historical trace—but also increases tokens.

Guidelines:
- Leave OFF for lean, predictable prompts.
- Turn ON only when you need the model to see inline non-heading operational traces (debugging / auditing complex chains).
- Regardless of this flag, inserted tool result headings remain selectable.

Summary: Visibility flag = wider historical trace vs. cost. Keep lean unless you have a reason.

## Token Mechanics & Growth Dynamics
Each model turn re-sends the selected + ambient blocks. Tool loops amplify this because every new tool output that you keep alive is re-included unless you compress or remove it.

Drivers of Growth:
- Raw multi-hundred-line tool outputs (logs, JSON, issue lists)
- Repeated wildcards pulling unchanged large sections
- Accumulated append refinements instead of final replace

Control Levers:
- Summarize & Replace: Convert bulky output into a compact summary block (same ID) → future turns shrink.
- Turn Budget: `tools-turns-max` prevents runaway iterative calls.
- Isolation Pass: After gathering, run a clean `@llm` with only curated summaries + `context: none`.
- Prune Obsolete: Delete exploratory blocks no longer referenced.

Quick Heuristic:
If a block won’t change again AND it’s > ~150–200 lines or very repetitive → summarize & replace.

## Distillation Workflow (From Raw → Final)
Stage | Purpose | Your Action
----- | ------- | -----------
Raw Retrieval | Gather unfiltered data via tools/import | Allow brief accumulation
First Compression | Preserve all facts in shorter form | Summarize (bullets, structured list)
Analytical Synthesis | Derive insights / decisions | Reference only compressed blocks
Final Artifact | Clean deliverable for sharing | Replace draft with polished output
Archival (Optional) | Retain audit trail minimalistically | Keep one “Source Notes” summary block; remove noise

Prompt Pattern (embedded distillation):
```markdown
@llm
prompt: |
  Use tools only if a fact is missing.
  After each tool call: update '# Source Notes {id=source-notes}' with concise bullet facts (no duplicates). Remove the raw bulk.
tools:
  - repo_stats
  - issue_search
tools-turns-max: 3
```
Outcome: A single evolving curated block instead of a pile of raw fragments.

Design Principle: Treat raw tool output as disposable. The value is in the distilled, named, stable blocks you keep.

---
See also:
- [Advanced LLM Features](advanced-llm-features.md)
- [Operations Reference](operations-reference.md)
- [Syntax Reference](syntax-reference.md)
