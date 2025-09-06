# 8. File & Import Semantics

Purpose: How Fractalic loads external markdown, selectively imports blocks, and merges them into the current context without bloat.

Why it matters:
- Reuse: Single source of truth (research, specs, glossary).
- Modularity: Assemble larger workflows from smaller files.
- Determinism: Explicit IDs / paths give reproducible runs.
- Cost: Import only what is needed.
- Refactoring: Stable IDs keep dependent flows working.

---
### Internal Table of Contents
- [8.1 Concept Overview](#81-concept-overview)
- [8.2 Import Syntax (`@import`)](#82-import-syntax-import)
- [8.3 Source Path & Block Resolution](#83-source-path--block-resolution)
- [8.4 Full vs Partial Imports](#84-full-vs-partial-imports)
- [8.5 Descendant Selection with Wildcards (`/*`)](#85-descendant-selection-with-wildcards-)
- [8.6 Merge Target & Modes (`to`, `mode`)](#86-merge-target--modes-to-mode)
- [8.7 ID Stability](#87-id-stability)
- [8.8 Patterns & Examples](#88-patterns--examples)
- [8.9 Optimization & Cost Control](#89-optimization--cost-control)
- [8.10 Common Pitfalls](#810-common-pitfalls)
- [8.11 Quick Reference](#811-quick-reference)
- [Cross References](#cross-references)

---
## 8.1 Concept Overview
`@import` copies content from another markdown file into the current file. You control:
- Which file (`file` path)
- Optional starting block inside that file (`block`)
- Whether you want just that block or the block plus its descendants (use wildcard `/*`)
- Where the imported content is merged (`to`)
- How it merges (`mode`: append | prepend | replace)

Result: Imported content becomes normal blocks you can reference later.

## 8.2 Import Syntax (`@import`)
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

## 8.3 Source Path & Block Resolution
Steps:
1. Read the source file path from `file`.
2. Parse source file.
3. If `block` provided: locate that block.
4. If wildcard form `something/*` used: include that block + all its descendants.
5. Determine destination: `to` if present else the current location of the operation.
6. Merge per `mode`.
Missing file or block triggers an error (no silent skip).

## 8.4 Full vs Partial Imports
Case | Use
---- | ----
Entire file | Central shared glossary / spec reused widely.
Single block | Only one section needed.
Block with descendants (`id/*`) | Need full structured subtree (children used later).

## 8.5 Descendant Selection with Wildcards (`/*`)
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

## 8.6 Merge Target & Modes (`to`, `mode`)
`to` = ID (or path) of the destination anchor. If omitted, import merges at the operation position.

Mode | Effect
---- | ------
append (default) | Add after existing children / content.
prepend | Insert before existing children.
replace | Overwrite destination block body with imported result.

Use `replace` once you have a stable curated summary that should supersede earlier verbose content.

## 8.7 ID Stability
- Keep reusable blocks labeled with explicit `{id=...}` in the source file early.
- Avoid renaming published IDs (imports relying on them will fail or shift).
- For major rework: create a new ID, migrate dependents, then retire the old one.

## 8.8 Patterns & Examples
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

## 8.9 Optimization & Cost Control
Goal | Action
---- | ------
Limit noise | Import only the block(s) you need.
Prevent duplication | Reference canonical sources instead of copy/paste.
Control size | Use single block import when descendants not required.
Prune growth | Replace verbose historical imports with distilled summaries.

## 8.10 Common Pitfalls
Pitfall | Impact | Fix
------- | ------ | ---
Importing whole large file repeatedly | Token bloat | Narrow to block or block/*
Using replace too early | Lose evolution trail | Start with append
Relying on auto slug for reused content | Breaks after heading rename | Add explicit ID
Forgetting wildcard when children needed | Missing downstream references | Use `block-id/*`
Over‑importing overlapping sections | Confusion / duplicates | Centralize under one hub block

## 8.11 Quick Reference
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
