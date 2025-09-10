---
title: Syntax Reference
description: Complete syntax reference for Fractalic operations, blocks, and YAML configuration
outline: deep
---

# Syntax Reference

Fractalic syntax = Standard Markdown (structure, prose) + Lightweight YAML operation blocks (actions). The interpreter walks the file linearly, builds an AST (headings, operations, generated nodes), and applies operations in order.

## Markdown + YAML Hybrid Structure
- Headings (`#`, `##`, `###`, …) define knowledge blocks.
- Paragraphs, lists, code fences belong to the nearest preceding heading.
- An operation block: a line starting with `@operationName` followed by YAML.
- Content up to next heading OR next `@operation` lines belongs to the current block.
```markdown
# Research Plan {id=plan}
We will investigate three competitors.

@llm
prompt: "List the competitors in a table."
use-header: "# Competitor Table"
```

## Operation Block Anatomy
```markdown
@llm                                 # 1) Operation identifier
prompt: |                            # 2) Literal prompt (multi-line YAML scalar)
  Produce a structured outline.
block:                               # 3) Block references
  block_uri: context/*
model: claude-3-5-sonnet             # 4) Logical model alias (provider inferred)
mode: append                         # 5) Merge strategy
use-header: "# Outline"             # 6) Wrapper heading (or 'none')
header-auto-align: true              # 7) Re-level headings
save-to-file: outputs/outline.md     # 8) Persist raw output
```

## Headers, Auto-Alignment, and use-header Strategies
Default wrapper headings:
| Operation | Default Wrapper |
|-----------|-----------------|
| @llm | `# LLM Response block` |
| @shell | `# OS Shell Tool response block` |
| @return | `# Return block` |
| @import | none |
| @run | none |

Rules & patterns:
- Quote any header starting with `#` (YAML comment otherwise).
- `use-header: none` → raw content only.
- `header-auto-align: true` shifts top-level heading depth relative to current structural depth.
```markdown
## Section
@llm
prompt: "Generate a 2-level outline."
header-auto-align: true
```
Raw insertion (no extra heading):
```markdown
@llm
prompt: "Return ONLY a JSON object."
use-header: none
```

## Block Reference Syntax
Forms:
- Direct id: `vision`
- Hierarchical path: `research/findings`
- Wildcard branch: `research/*` (parent + descendants)
- Multiple via array:
```yaml
block:
  - research/*
  - decisions
```
Notes: Order preserved; forward references allowed; missing block → error.

## Special Values & Merge Modes
`use-header: none` suppresses wrapper.
Merge `mode`:
- `append` (default) add after target / op
- `prepend` insert before target
- `replace` swap target content atomically
If `to:` absent → anchor is operation position.

## Prompt Assembly Order (@llm)
1. Referenced block contents (ordered)
2. Literal `prompt`
3. Media attachments (provider-handled)
4. System prompts (implicit)
Missing both `prompt` & `block` → error.

## Parameter Normalization & Model Inference
### model
Alias looked up in `[settings]` of `settings.toml`. Dots/hyphens/underscores normalized.
### stop-sequences
Substrings that terminate generation early (prevent spill‑over / unwanted sections).
```yaml
stop-sequences:
  - "\n```"
  - "\n# "
```
### tools-turns-max
Upper bound on model ↔ tool reasoning iterations.
### use-header
Default per table above; set to `none` for structured outputs (JSON, code).
### header-auto-align
Re-level generated heading hierarchy under current depth.
### save-to-file
Store raw model output outside context to reduce tree bloat.
### media
Attach files for multimodal models (images, pages, etc.).
### context
`auto` (default) may include ambient prior headings when only `prompt` provided.
`none` ensures isolation (only `prompt` + system). (If `block` used, ambient context is never auto-attached.)
### provider
Optional explicit override; normally inferred from model.
### Unrecognized fields
Ignored safely (future-proofing).

Beginner checklist:
- Need raw JSON? `use-header: none` + consider `stop-sequences`.
- Tools enabled? Set `tools-turns-max`.
- Large output? Use `save-to-file`.
- Deep nesting? `header-auto-align: true`.

## Arrays & Multi-Line Scalars
```markdown
@shell
prompt: |
  set -e
  echo "Building"
  ls -1
```
Multiple blocks:
```markdown
@llm
prompt: "Compare the two sections."
block:
  - old-version/*
  - new-version/*
```

## Common Pitfalls
| Issue | Cause | Fix |
|-------|------|-----|
| Block not found | Typo / missing id | Verify kebab-case or add explicit `{id=...}` |
| Duplicate IDs | Auto-generated collisions | Add explicit IDs |
| Wrong heading depth | Forgot `header-auto-align` | Enable or edit prompt |
| Unexpected wrapper | Default header applied | `use-header: none` |
| Empty response | Provider/tool error | Check logs, keys, tool outputs |

## Style Recommendations
- Put large reference material in earlier knowledge blocks, keep prompts terse.
- Prefer explicit IDs for frequently reused anchors.
- Use wildcards sparingly—token cost scales.
- Early iteration: `append` for provenance; later: `replace` for cleanliness.
- Label important generated sections with stable headings for downstream reuse.

## Supplemental Clarifications (Added, Not Altering Original Text Above)
- Clarified that when `block` is provided, ambient context auto-inclusion does not occur regardless of `context` value.
- Reinforced quoting rule for `use-header` values beginning with `#`.
- Added beginner checklist grouping advanced parameters.
