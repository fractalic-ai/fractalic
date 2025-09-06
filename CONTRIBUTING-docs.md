# Fractalic Documentation Authoring Guide

This guide defines how to write, structure, and format Fractalic documentation.

## 1. Purpose of These Standards
Consistent docs = faster learning, lower misuse risk, easier maintenance, cheaper AI execution (via good context hygiene). This guide applies to: core docs, tutorials, specs, examples.

## 2. Structure Pattern
Order for substantial docs (>8 sections):
1. Title & Purpose (what + why it matters)
2. Internal TOC (anchor links)
3. Core Concepts / Model (neutral wording; avoid “mental”)
4. How It Works (flow / lifecycle)
5. Usage Patterns & Examples
6. Edge Cases / Pitfalls
7. Optimization / Performance / Cost Control
8. Safety & Constraints (if relevant)
9. Quick Reference Table / Checklist
10. Cross References (linked docs)

Short docs may collapse 3–7.

## 3. Tone & Language
- Plain, precise, neutral. Avoid marketing language.
- Prefer imperative for actions: “Add an ID…”, “Summarize large block…”.
- Avoid “mental model”; use: “concept”, “idea”, “graph”, “structure”.
- Limit adjectives; clarity over persuasion.

## 4. Key Concepts Representation
When introducing a concept always cover:
- What it is
- Why it matters (quality, cost, reliability, safety, reuse)
- How to use it
- When NOT to use it (if applicable)

## 5. Block & Selection Syntax
Canonical forms:
- Single block (explicit key form):
  ```yaml
  block:
    block_uri: summary
  ```
- Single block (shorthand):
  ```yaml
  blocks: summary
  ```
- Multiple blocks:
  ```yaml
  blocks:
    - summary
    - decisions
  ```
- Wildcard children:
  ```yaml
  blocks: research/*
  ```
Use ONE style per example; do not mix. Use `blocks:` form when showing arrays. Use `block:` + `block_uri:` only when teaching the parameter explicitly.

## 6. Operation Examples Formatting
Order inside an example (where applicable):
```markdown
@llm
prompt: |
  Multi-line or long instruction.
blocks: summary
mode: append  # Only if diverging from default
to: report-draft  # Omit if target is implicit
use-header: "# Optional Heading {id=result}"
tools:
  - repo_stats
  - issue_search
tools-turns-max: 2
```
Guidelines:
- Always use `|` for multi-line prompts, structured text, JSON, or instructions with colons.
- Prefer unquoted one-line simple prompts (no special chars).
- Do not use comma lists for multiple tools—use YAML list form.
- If `mode: replace` appears, include explicit `to:` unless the current block is unmistakably the target (prefer explicit anyway).

## 7. Append vs Replace Policy
Default teaching: `append` grows history; prefer during exploration.
Use `replace` only when:
- Compressing / summarizing stabilized large content
- In-place refinement where previous version is no longer needed
- Pruning before a clean final synthesis or evaluation pass
- Controlled iterative loop where diff noise is not required
Warn: replace discards prior content body.

## 8. Tool / Agent Loop Documentation
Always explain:
- Each surviving tool block is resent next LLM turn (token cost growth)
- Summarize early; replace only after confirming fidelity
- Provide suggested `tools-turns-max`
- Dual layer (tool layer) auto-inserts outputs; user never manually merges
Pattern snippet:
```markdown
@llm
prompt: |
  Use tools only if new facts required. Maintain '# Source Notes {id=source-notes}' (concise, no duplicates). Stop when all KPIs present.
tools:
  - repo_stats
  - issue_search
tools-turns-max: 3
blocks: context-intro
```

## 9. Compression & Distillation Workflow
Stages:
Raw → First Compression → Synthesis → Final Artifact → Optional Archival Summary.
Heuristic: If a block will not materially change and is long (>150–200 lines / highly repetitive) → compress.

## 10. Token & Cost Hygiene Checklist
- Large raw tool outputs summarized?
- Wildcards narrowed?
- Redundant append histories replaced?
- Obsolete exploratory blocks removed?
- Isolation (`context: none`) used for formal JSON / evaluation passes?

## 11. JSON / Structured Output
Recipe:
```markdown
@llm
prompt: |
  Return ONLY JSON with keys: title, risks[] (each {id, severity}). No prose.
blocks: risk-source
use-header: none
```
Add stop sequences sparingly only if hallucinated trailers appear.

## 12. Cross-Linking Rules
End every major doc with a “See also” block linking relative paths:
- `[Syntax Reference](syntax-reference.md)`
- `[Operations Reference](operations-reference.md)`
- `[Advanced LLM Features](advanced-llm-features.md)`
- `[Context Management](context-management.md)` (where relevant)
Use section symbols (§) only if pairing with a link, not alone.

## 13. IDs & Naming
- Use lowercase-dash or underscore (analysis-summary, exec_report)
- Add IDs before first reuse
- Do not rename IDs after publishing examples

## 14. Safety & Reliability Notes
- Shell examples minimal; suggest external scripts for complexity.
- For evaluation / scoring passes, isolate context (`context: none`) + only required blocks.
- Avoid showing destructive or network-modifying shell commands.

## 15. Style Anti-Patterns to Avoid
- Comma-separated tool lists (wrong): `tools: repo_stats, issue_search`
- Mixed selection syntax in one snippet
- Unnecessary quotes around plain scalars
- Using `replace` in early exploratory examples without rationale
- Leaving large raw tool dumps without a follow-up compression example

## 16. Pre-Publish Validation
Before merging a doc:
- [ ] Examples are syntactically valid (YAML indentation, list forms)
- [ ] All `replace` examples show `to:`
- [ ] Multi-line prompts use `|`
- [ ] Cross links present
- [ ] No banned terms (e.g., “mental model”)
- [ ] Tool lists use YAML list form
- [ ] Block selection arrays use `blocks:` not nested `block_uri` unless explicitly teaching

## 17. Revision Discipline
Do not add arbitrary “Revision:” lines in user docs. Track changes via git history. Only architectural specs may include a version header (e.g., `Spec Version: 1.1`).

## 18. Example Library Pattern Tags (Optional)
Tag examples inline via comments for future indexing:
`# pattern: compression`, `# pattern: tool-loop`, `# pattern: replace-inplace`.
Do not overuse; max 1–2 tags per snippet.

---
Questions or proposals for changes: open an issue titled "Docs Guideline Change: <short description>".
