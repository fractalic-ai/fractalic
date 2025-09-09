---
title: Core Concepts
description: Understanding the fundamental concepts of Fractalic - documents as programs, execution context, and block types
outline: deep
---

# Core Concepts

## Documents as Programs
Markdown file = ordered set of knowledge + operation blocks. Execution walks top → bottom mutating an internal AST.

## Execution Context & AST
AST nodes: heading, paragraph, operation, generated block. Diffs show evolution (planned UI features may surface more granular diffs).

## Block Types
- Knowledge Block: heading + its content
- Operation Block: `@name` + YAML params creating side-effect (new nodes)

## Block IDs & Paths
Explicit `{id=custom}` or derived kebab-case. Nested path: `parent/child`. Wildcard: `section/*` adds all descendants.

## Context Selection Strategies
- Narrow: specific IDs for precision
- Branch: `/*` when holistic branch reasoning needed
- Composed arrays: sequence defines concatenation order

## Incremental Context Growth
Each op merges via `append`, `prepend`, or `replace`. Early prototyping: append to track evolution; later refine using replace.

## Isolation vs Sharing (@run)
`@run` executes a separate file as an isolated micro-context; only injected prompt/blocks + internal content; returns explicit output.

## Determinism Principles
- Explicit block selection over implicit context (use `context: none` when needed)
- Merge mode semantics stable and predictable
- Header auto-alignment prevents structural drift

## Tool-Generated Context
Tool loop inserts extracted tool outputs as normal user-role blocks. Attribution metadata preserved internally (future surfacing planned).

## Returns as Semantic Boundaries
Every agent should end with `@return` for clarity. Missing returns lead to ambiguous upstream merges.

## Progressive Elaboration Workflow Pattern
1. Start with single prompt
2. Add headings to segment knowledge
3. Import templates for repeated scaffold
4. Abstract repeatable logic into agents
5. Introduce shell / tools for verification & grounding
6. Compress large results to slimmer reference blocks

## Anti-Patterns
| Anti-Pattern | Why | Fix |
|--------------|-----|-----|
| Huge wildcard contexts | Wastes tokens / noise | Curate smaller block sets |
| No explicit IDs for reused sections | Fragile references | Add `{id=...}` |
| Repeated append refinements | Context bloat | Switch to replace once stable |
| Tool loop with no cap | Runaway cost | Set `tools-turns-max` |
| JSON generation with wrapper heading | Breaks parsers | `use-header: none` |

## Glossary Quick Anchor
See full Glossary section later for formal definitions.
