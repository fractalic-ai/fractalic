---
title: Advanced LLM Features
description: Advanced features for LLM operations including tool loops, streaming, JSON output, and optimization patterns
outline: deep
---

# Advanced LLM Features

Status: Stable (supplements `operations-reference.md`)
Scope: Deep patterns for controlling, structuring, optimizing, and validating `@llm` driven workflows beyond the core syntax.

---
## Overview
Fractalic treats every `@llm` invocation as a deterministic transformation over an evolving document tree. Advanced usage focuses on:
- Precision context shaping (selective block referencing, staged compression)
- Tool / agentic loop governance (bounded reasoning, explicit decision frames)
- Structured + schema‑validated outputs
- Multi-agent orchestration (planner / executor / reviewer / router)
- Performance, token, and cost optimization
- Regression safety (golden outputs, diff contracts)
- Guardrails (content policies, redaction, JSON validation)

Use this document as a pattern catalog; operational field definitions remain canonical in the Operations & Syntax references.

---
## Prompt Engineering Patterns
Pattern | Goal | Technique
------- | ---- | ---------
Context Window Sculpting | Minimize noise | Reference only needed leaf blocks; avoid wide `/*` early
Section Tagging | Enable selective refinement | Embed stable IDs `{id=analysis-raw}` then later target them
Role Decomposition | Increase reasoning quality | "You are acting as: (1) Planner (2) Analyst (3) Synthesizer" with explicit deliverables
Progressive Compression | Preserve signal | Raw → bullet summary → prioritized list → canonical spec
Delta Prompting | Focus revision | "ONLY adjust tone; keep structure & bullet counts identical"
Explicit Termination | Avoid runaway tool loops | "Call a tool ONLY if factual gap remains. Otherwise answer FINAL." / add stop criteria list
Reflection Injection | Quality gating | After draft: second `@llm` with rubric to critique and patch

Inline Example (planning then refining selected block):
```markdown
# Investigation Plan {id=plan-v1}
... initial content ...

@llm
prompt: "Refine ONLY the plan keeping step count fixed. Improve clarity only."
block:
  - plan-v1/*
mode: replace
to: plan-v1  # In-place refinement
```

---
## Structured Output Strategies
Technique | Notes
--------- | -----
`use-header: none` | Removes wrapper heading for pure machine output
Strict JSON Envelope | Instruct: "Return ONLY valid JSON matching schema" + stop sequence guards
Schema Echo | Have model first restate expected keys (short) then produce final JSON (higher adherence)
Dual Phase | Generate reasoning (hidden / discarded) then final JSON (by instructing: "Output ONLY final JSON after thinking internally")
Failure Detection | Post-parse validator: if invalid, re-run with appended error explanation block

Example (schema guard):
```markdown
@llm
prompt: |
  Produce ONLY JSON: {"title": str, "risks": [{"id": str, "severity": "low|med|high"}]}
  No prose.
use-header: none
stop-sequences:
  - "\n# "
```

---
## Tool / Agentic Loop Advanced Controls
Aspect | Pattern
------ | -------
Decision Framing | Provide explicit checklist: "Call tool if ANY of: missing metric, unresolved ID, >1 conflicting claim"
Turn Budgeting | `tools-turns-max: 2` (typical). Raise only with measured benefit
Selective Tool Set | Narrow `tools:` list per phase (gather vs synthesize)
Result Canonicalization | Immediately summarize verbose tool outputs into compact blocks to avoid snowballing context
Attribution IDs | Instruct tool loop to emit headings with stable `{id=...}` for reuse
Abort Clause | "If tool returns no actionable data, respond FINAL directly"

---
## Multi-Agent Composition Patterns
Pattern | Flow | Notes
Planner / Executor | planner.md (@return spec) → executor consumes spec | Keeps large raw exploration isolated
Executor / Reviewer | executor output → reviewer critiques → patch pass | Ensures quality gating
Router | lightweight router agent decides which specialized agent file to invoke via `fractalic_run` | Reduces average cost
Self-Refinement Loop | generation → critique rubric → constrained rewrite | Limit to 1–2 cycles to avoid drift

Router Decision Example:
```markdown
@llm
prompt: |
  Decide: if task mentions "deploy" run deploy agent else run default agent. Respond with JUST the agent call arguments.
tools:
  - fractalic_run
```

---
## Determinism & Reproducibility
Goal | Technique
---- | ---------
Minimize nondeterminism | Stable model versions, explicit temperature (if provider supports), avoid ambiguous instructions
Traceability | Commit intermediate blocks; use `replace` instead of endless `append` growth
Stable References | Add `{id=...}` to reusable blocks before externalizing
Change Auditing | Pair large replacements with a summarized diff block for human review
Idempotent Replays | Avoid date / transient calls inside reasoning; fetch externally then import snapshot

---
## Token & Cost Optimization
Strategy | Detail
-------- | ------
Layered Summaries | Raw import → compressed → ultra-compact semantic index
Context Bloom Control | Replace large exploratory text once distilled
Media Pruning | Only attach images that materially change outcome
Selective Wildcards | Use explicit IDs after first pass, not `section/*` repeatedly
External File Parking | `save-to-file` for bulky logs; re-import narrow extracts
Compression Prompt | "Rewrite the following preserving all numbers & entities; max 120 tokens"

---
## Safety & Guardrails
Concern | Mitigation
------- | ---------
Sensitive Data Leakage | Redact via preprocessing shell step before `@llm`
Unbounded Tool Actions | Provide explicit negative instructions ("Never execute destructive shell commands")
Prompt Injection from Imports | Sanitize or summarize third-party text before inclusion
Toxic Output | Post-generation validator agent that scans and blocks policy violations
Schema Poisoning | Validate JSON; on failure, re-run with appended validator error explanation

Validator Pattern:
```markdown
@llm
prompt: "Check the prior block for PII; respond PASS or list redactions needed."
block:
  block_uri: draft-output/*
```

---
## Evaluation & Regression Testing
Method | Implementation
------ | --------------
Golden Output Snapshots | Store canonical JSON / text; diff on regeneration
Metric Extraction | Secondary `@llm` producing structured rubric scores (clarity, completeness, risk)
Factual Consistency | Compare entity lists across versions; flag additions / deletions
Test Harness | Shell step running unit tests produced by model before acceptance
Drift Detection | Hash compressed summaries; change triggers deeper review

---
## Retrieval & External Knowledge
Need | Pattern
---- | -------
Static Knowledge | `@import` curated handbook sections, not raw dumps
Dynamic Fetch | Shell / MCP retrieval tool → immediate summarization block
Lightweight RAG | Precompute embedding clusters (external), import only matched cluster summaries
Cache Layer | Store previous tool responses; fallback if rate-limited

---
## Model Routing & Fallback
Scenario | Approach
-------- | --------
Cheap Draft, Expensive Polish | First pass small model; second pass high-quality model referencing draft
Structured Extraction | Smaller deterministic model often adequate; escalate only on complexity detection
Automatic Fallback | Router agent interprets error block & retries with alternate model

Routing Skeleton:
```markdown
@llm
prompt: |
  If task complexity (steps > 5 or domain specialized) -> choose model "deep" else "fast". Output ONLY: model=`<id>`.
use-header: none
```

---
## Strict JSON Mode Recipe
1. Provide minimal schema in natural language
2. Set `use-header: none`
3. Provide explicit invalid examples (optional) to steer away
4. Add narrow stop sequences that terminate at first blank line or heading
5. Post-parse; if error → re-run with appended parser message under `# JSON Parse Error {id=json-err}`

Snippet:
```markdown
@llm
prompt: |
  Return ONLY JSON with keys: items (array of {id, label, priority:int 1-5}), summary:str.
  No code fences, no comments.
use-header: none
stop-sequences:
  - "\n# "
```

---
## Parallelization Patterns
Use separate files / agents for independent subtasks then import results for synthesis.
Workflow:
1. Split tasks (A, B, C) each as `@run` or direct `@llm`
2. Each produces compact summary with stable ID
3. Synthesis call references only `*/summary` blocks

Benefits: reduces cross-contamination, keeps token usage bounded.

---
## Debugging Tool Loops
Symptom | Debug Step
------- | ----------
Repeated Tool Calls | Add explicit termination rule; inspect last tool result content
Empty Final Answer | Check raw tool JSON for missing `return_content`
Lost Context | Confirm correct block IDs passed (wildcard vs leaf mismatch)
Heading Explosion | Enable `header-auto-align` or constrain model to avoid re-heading tool outputs
Unexpected Import Size | Instrument with a shell token count before invoking @llm

Tracing Tip: Centralize debug mode that sets `tools-turns-max: 1` + adds a diagnostic block summarizing selected blocks & token counts.

---
## Comprehensive Checklist
Category | Questions
-------- | ---------
Context | Are only necessary blocks referenced? Any stale large blocks replaceable?
Structure | Stable IDs assigned where downstream reuse expected?
Tools | Clear call criteria & bounded turns? Redundant tools removed?
Output | Does it need raw JSON (`use-header: none`)? Schema validated?
Safety | Redaction / policy validation steps present for sensitive domains?
Performance | Summaries replacing verbose intermediates? Large outputs externalized?
Evaluation | Golden outputs or rubric scoring present for critical workflows?
Recovery | Fallback / retry model plan defined?

---
## See Also
- Operations Reference (§5.3.*) for field semantics
- Syntax Reference for YAML field grammar
- (Upcoming) Context Management doc for AST merge & selection internals

---
Revision: v1.0
