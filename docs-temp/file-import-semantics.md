---
title: File & Import Semantics
description: Understanding file imports, merging strategies, and workspace management in Fractalic
outline: deep
---

# File & Import Semantics

Purpose: How Fractalic loads external markdown, selectively imports blocks, and merges them into the current context without bloat.

Why it matters:
- Reuse: Single source of truth (research, specs, glossary).
- Modularity: Assemble larger workflows from smaller files.
- Determinism: Explicit IDs / paths give reproducible runs.
- Cost: Import only what is needed.
- Refactoring: Stable IDs keep dependent flows working.

---
### Internal Table of Contents
- [Concept Overview](#concept-overview)
- [Import Syntax (`@import`)](#import-syntax-import)
- [Source Path & Block Resolution](#source-path--block-resolution)
- [Full vs Partial Imports](#full-vs-partial-imports)
- [Descendant Selection with Wildcards (`/*`)](#descendant-selection-with-wildcards-)
- [Merge Target & Modes (`to`, `mode`)](#merge-target--modes-to-mode)
- [ID Stability](#id-stability)
- [Patterns & Examples](#patterns--examples)
- [Optimization & Cost Control](#optimization--cost-control)
- [Common Pitfalls](#common-pitfalls)
- [Quick Reference](#quick-reference)
- [Cross References](#cross-references)

---
## Concept Overview
`@import` copies content from another markdown file into the current file. You control:
- Which file (`file` path)
- Optional starting block inside that file (`block`)
- Whether you want just that block or the block plus its descendants (use wildcard `/*`)
- Where the imported content is merged (`to`)
- How it merges (`mode`: append | prepend | replace)

Result: Imported content becomes normal blocks you can reference later.

## Import Syntax (`@import`)
Minimal (whole file content appended in place):
```markdown
@import
file: docs-temp/core-concepts.md
```
Import one section only:
```markdown
@import
file: docs-temp/core-concepts.md
block: context-graph
```
Import a section plus all its children (wildcard):
```markdown
@import
file: docs-temp/core-concepts.md
block: context-graph/*
```
Import into a specific aggregation block:
```markdown
@import
file: docs-temp/context-management.md
block: 713-quick-reference-table
to: quick-reference-hub
mode: append
```
Controlled refresh (overwrite prior synthesized block once stable):
```markdown
@import
file: research/raw-user-interviews.md
block: interview-summaries
to: research-summary
mode: replace
```
Guidelines:
- Use `append` while still evolving structure.
- Switch to `replace` after the destination shape stabilizes.

## Source Path & Block Resolution
Steps:
1. Read the source file path from `file`.
2. Parse source file.
3. If `block` provided: locate that block.
4. If wildcard form `something/*` used: include that block + all its descendants.
5. Determine destination: `to` if present else the current location of the operation.
6. Merge per `mode`.
Missing file or block triggers an error (no silent skip).

## Full vs Partial Imports
Case | Use
---- | ----
Entire file | Central shared glossary / spec reused widely.
Single block | Only one section needed.
Block with descendants (`id/*`) | Need full structured subtree (children used later).

## Descendant Selection with Wildcards (`/*`)
Add `/*` after a block path or ID to include that block and all its descendant blocks.
Examples:
Single block + descendants:
```
block: architecture/*
```
Multiple selections (order preserved):
```
block:
  - risks/*
  - mitigations
```
Combined (one subtree plus others):
```
block:
  - architecture/*
  - risks/*
  - mitigations
```
Rule: Use a single `block:` key. Provide one value (string) or a YAML list. Do not repeat the key.

Use plain `block: section-id` when only the body of that block is required and not the nested sections.

## Merge Target & Modes (`to`, `mode`)
`to` = ID (or path) of the destination anchor. If omitted, import merges at the operation position.

Mode | Effect
---- | ------
append (default) | Add after existing children / content.
prepend | Insert before existing children.
replace | Overwrite destination block body with imported result.

Use `replace` once you have a stable curated summary that should supersede earlier verbose content.

## ID Stability
- Keep reusable blocks labeled with explicit `{id=...}` in the source file early.
- Avoid renaming published IDs (imports relying on them will fail or shift).
- For major rework: create a new ID, migrate dependents, then retire the old one.

## Patterns & Examples
Central knowledge hub assembly:
```markdown
@import
file: docs-temp/context-management.md
block: 713-quick-reference-table
to: knowledge-hub
mode: append
```
Selective synthesis (child tree required):
```markdown
@import
file: docs-temp/core-concepts.md
block: context-graph/*
mode: append
```
Periodic refresh (stable target):
```markdown
@import
file: research/raw-user-interviews.md
block: interview-summaries
to: research-summary
mode: replace
```

## Optimization & Cost Control
Goal | Action
---- | ------
Limit noise | Import only the block(s) you need.
Prevent duplication | Reference canonical sources instead of copy/paste.
Control size | Use single block import when descendants not required.
Prune growth | Replace verbose historical imports with distilled summaries.

## Common Pitfalls
Pitfall | Impact | Fix
------- | ------ | ---
Importing whole large file repeatedly | Token bloat | Narrow to block or block/*
Using replace too early | Lose evolution trail | Start with append
Relying on auto slug for reused content | Breaks after heading rename | Add explicit ID
Forgetting wildcard when children needed | Missing downstream references | Use `block-id/*`
Over‑importing overlapping sections | Confusion / duplicates | Centralize under one hub block

## Quick Reference
Need | Field / Form
---- | ------------
Whole file | (no `block`)
Single block | block: id-or-path
Block + descendants | block: id-or-path/*
Target anchor | to: some-block-id
Replace existing | mode: replace
Progressive growth | mode: append

---
## Cross References
- [Syntax Reference](syntax-reference.md)
- [Operations Reference](operations-reference.md)
- [Advanced LLM Features](advanced-llm-features.md)
- [Context Management](context-management.md)
