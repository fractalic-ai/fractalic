# 3. Core Concepts

## 3.1 Documents as Programs
Markdown file = ordered set of knowledge + operation blocks. Execution walks top → bottom mutating an internal AST.

## 3.2 Execution Context & AST
AST nodes: heading, paragraph, operation, generated block. Diffs show evolution (planned UI features may surface more granular diffs).

## 3.3 Block Types
- Knowledge Block: heading + its content
- Operation Block: `@name` + YAML params creating side-effect (new nodes)

## 3.4 Block IDs & Paths
Explicit `{id=custom}` or derived kebab-case. Nested path: `parent/child`. Wildcard: `section/*` adds all descendants.

## 3.5 Context Selection Strategies
- Narrow: specific IDs for precision
- Branch: `/*` when holistic branch reasoning needed
- Composed arrays: sequence defines concatenation order

## 3.6 Incremental Context Growth
Each op merges via `append`, `prepend`, or `replace`. Early prototyping: append to track evolution; later refine using replace.

## 3.7 Isolation vs Sharing (@run)
`@run` executes a separate file as an isolated micro-context; only injected prompt/blocks + internal content; returns explicit output.

## 3.8 Determinism Principles
- Explicit block selection over implicit context (use `context: none` when needed)
- Merge mode semantics stable and predictable
- Header auto-alignment prevents structural drift

## 3.9 Tool-Generated Context
Tool loop inserts extracted tool outputs as normal user-role blocks. Attribution metadata preserved internally (future surfacing planned).

## 3.10 Returns as Semantic Boundaries
Every agent should end with `@return` for clarity. Missing returns lead to ambiguous upstream merges.

## 3.11 Progressive Elaboration Workflow Pattern
1. Start with single prompt
2. Add headings to segment knowledge
3. Import templates for repeated scaffold
4. Abstract repeatable logic into agents
5. Introduce shell / tools for verification & grounding
6. Compress large results to slimmer reference blocks

## 3.12 Anti-Patterns
| Anti-Pattern | Why | Fix |
|--------------|-----|-----|
| Huge wildcard contexts | Wastes tokens / noise | Curate smaller block sets |
| No explicit IDs for reused sections | Fragile references | Add `{id=...}` |
| Repeated append refinements | Context bloat | Switch to replace once stable |
| Tool loop with no cap | Runaway cost | Set `tools-turns-max` |
| JSON generation with wrapper heading | Breaks parsers | `use-header: none` |

## 3.13 Glossary Quick Anchor
See full Glossary section later for formal definitions.
