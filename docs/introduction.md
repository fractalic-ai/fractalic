---
title: Introduction
description: Learn about Fractalic - a plain-language programming environment for building AI workflows and agents using Markdown + YAML
outline: deep
---

# Introduction

## What is Fractalic?
Fractalic is a plain‑language programming environment for building AI workflows and agents using ordinary Markdown plus lightweight YAML operation blocks. Instead of wiring nodes in a UI or writing imperative Python scripts, you express intent in structured text. Fractalic interprets your document as a living execution context: headings become knowledge blocks; special `@operation` blocks perform actions (LLM calls, shell commands, imports, executing sub‑workflows, returning results). The document evolves as operations run—new blocks are appended, replaced, or refined—so you can literally “grow” an AI system the way you draft a document.

Key idea: A Fractalic file = (Ordered Knowledge Blocks) + (Operation Blocks that transform / extend context) → Final Returned Result.

## Core Philosophy (Plain-Language Programming, Context as a First-Class Object)
Below each guiding principle you get a tiny, copyable example.

#### 1. Plain-Language as Code
Write intent directly; ops are minimal YAML.
Example:
```markdown
# Product Tagline {id=tagline}
We help teams build AI workflows faster.

@llm
prompt: "Rewrite the tagline above to be 12 words, more energetic"
block:
  block_uri: tagline
use-header: "# Revised Tagline"
```
Result: A new heading inserted – no Python required.

#### 2. Context is the Program State
Only referenced (or auto-included) blocks enter a model call.
```markdown
# Requirements {id=reqs}
List the core must‑have features.

# Draft Output {id=draft}
(Empty initially)

@llm
prompt: "Create an outline using only the requirements."
block:
  block_uri: reqs
to: draft
mode: replace
```
The model never sees future sections you haven’t referenced.

#### 3. Deterministic Evolution
You choose how new content merges: append / prepend / replace.
```markdown
# Analysis {id=analysis}
Initial placeholder.

@llm
prompt: "Summarize key risks."
use-header: "# Risk Summary"
mode: append
to: analysis
```
Append keeps the original; `replace` would overwrite `# Analysis` content.

#### 4. Composable Agents
Any markdown file with a `@return` becomes an agent.
`agent/plan.md`:
```markdown
# Planning Input {id=input}
(Will be filled by caller)

@llm
prompt: "Turn the input into a 3‑step actionable plan."
block:
  block_uri: input
use-header: none

@return
block:
  block_uri: input
```
Caller:
```markdown
@run
file: agent/plan.md
prompt: "Research competitive landscape for lightweight AI orchestration tools."
```

#### 5. Progressive Elaboration
Start simple → grow structure without refactor.
Stage 1:
```markdown
@llm
prompt: "List 5 ways to speed up AI workflow iteration."
```
Stage 2 (extra structure added after seeing first output):
```markdown
# Ideas {id=ideas}
(Placed content from first run here)

@llm
prompt: "Cluster the ideas above and name each cluster."
block:
  block_uri: ideas
```

#### 6. Transparency over Magic
Prompt = (explicit literal + selected blocks). No hidden stuffing.
```markdown
# Background {id=bg}
Details.

@llm
prompt: "Summarize background in 2 sentences."
block:
  block_uri: bg
```
You always know exactly what was sent.

## Key Concepts at a Glance
Each concept below includes a mini example.

#### Knowledge Block
A heading plus everything until the next heading/operation.
```markdown
# Context {id=context}
Problem statement and constraints here.
```
Use it later via `block_uri: context`.

#### Operation Block
Line starting with `@name` + YAML.
```markdown
@llm
prompt: "Generate 3 title options."
```
Creates new content nodes.

#### Block ID
Explicit or auto (kebab-case of heading).
```markdown
# Product Vision {id=vision}
...
@llm
prompt: "Shorten the vision."
block:
  block_uri: vision
```

#### Branch / Wildcard Selection
Include a section and all nested children.
```markdown
# Research {id=research}
## Sources {id=sources}
List...
## Findings {id=findings}
...

@llm
prompt: "Synthesize research."
block:
  block_uri: research/*
```

#### Context Tree (AST)
Internal ordered structure; you see it as headings + generated blocks. Conceptually:
```
root
 ├─ # Context
 ├─ @llm (operation)
 └─ # Generated Summary
```

#### Merge Modes
```markdown
# Notes {id=notes}
Initial.

@llm
prompt: "Add more."
mode: append

@llm
prompt: "Provide final polished version only."
mode: replace
```
Second op overwrites; first appends.

#### Agent (Module)
Reusable file executed with `@run`.
```markdown
# agent/refine.md
@llm
prompt: "Improve clarity."
block:
  block_uri: input
use-header: none

@return
prompt: "Refinement complete"
```
Caller supplies `prompt` or blocks; receives returned content.

#### Return Semantics
`@return` ends workflow and yields value.
```markdown
@return
block:
  block_uri: final-report/*
```
Caller can merge it into its own document.

#### Tool Loop
LLM + external tools (via MCP) iteratively enrich context.
```markdown
@llm
prompt: "Fetch repo stats then summarize most active areas."
tools: github_repo_stats
```
Tool outputs become additional context blocks before final answer.

#### Header Auto-Alignment
Align generated headings relative to current depth.
```markdown
## Section
@llm
prompt: "Produce a 2-level outline."
header-auto-align: true
```
Generated top header becomes level 3 (###) inside this section.

---

## Comparison: Fractalic vs Traditional Scripting vs Agent Frameworks
| Aspect | Fractalic | Traditional Python Script | Typical Agent Framework |
|--------|-----------|---------------------------|--------------------------|
| Authoring Medium | Markdown + YAML ops | Imperative code | Config + code + orchestration logic |
| Context Control | Declarative block selection | Manual string assembly | Often opaque prompt chains |
| Incremental Evolution | Native (diffable tree) | Requires refactoring | Possible but verbose |
| Reusability | Any file can be an agent | Requires function/module design | Requires framework patterns |
| Transparency | Full AST + diffs | Depends on logging | Often hidden internal state |
| Learning Curve | Low for markdown users | Requires programming skill | Medium–High |
| Shell / External Tools | First-class via `@shell` | subprocess glue code | Plugin abstractions |
| Extensibility | New ops / MCP tools | Write more code | Conform to framework APIs |

Interpretation: Use Fractalic when you want rapid, explainable, auditable AI workflows without building a code-heavy infrastructure.

## When (and When Not) to Use Fractalic
Use Fractalic when:
- You prototype or iterate AI workflows faster than traditional refactors allow.
- Non‑developers or mixed teams must read and adjust logic directly.
- You need controlled context windows—only explicitly referenced knowledge enters prompts.
- You want modular “agent” documents that chain together (e.g. planning → research → synthesis → deployment).
- You integrate shell / system tooling (data fetching, transformations, compilation) directly into the AI reasoning loop.
- You require provenance: diff-based history of how output was produced.

Consider other approaches when:
- You need millisecond‑level latency or highly concurrent microservice scaling (Fractalic favors clarity over raw speed in its current form).
- Your logic demands complex branching, tight loops, or numeric computation better served by a programming language runtime.
- You already maintain a mature codebase where adding markdown orchestration would duplicate existing robust pipelines.
- You require strict transactional guarantees or ACID persistence semantics beyond contextual evolution.

Pattern Suitability Examples:
- ✅ Research + summarization pipeline with staged refining agents.
- ✅ Code generation with shell verification loops and test execution.
- ✅ Data ingestion (curl / parse / summarize) feeding an analysis agent.
- ❌ High-frequency trading logic requiring sub-second decisioning.
- ❌ Heavy data engineering transformations better done in Spark / SQL.

Getting Started Mindset:
1. Start a single markdown file: define a problem heading, add `@llm` with a prompt.
2. Add headings to segment emerging knowledge (requirements, plan, outputs).
3. Replace ad-hoc copy/paste with `@import` for reusable templates.
4. Extract a repeatable portion into its own file and invoke via `@run`.
5. Finalize with `@return` referencing the synthesized block(s).
6. Iterate: adjust block references to tighten context, add shell calls for verification.

Outcome: You transition naturally from idea → structured workflow → reusable agent library without switching mediums.
