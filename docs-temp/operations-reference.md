---
title: Operations Reference
description: Complete reference for all Fractalic operations - @llm, @shell, @import, @run, and @return
outline: deep
---

# Operations Reference

Operations mutate the context tree, consuming inputs (blocks, prompt literals, external sources) and producing new nodes.

## Shared Fields
- `block` → Content selection (single block reference or array of block references).
- `prompt` → Literal instruction / content.
- `to` → Target block path for merge.
- `mode` → `append` | `prepend` | `replace`.
- `use-header` → Wrapper heading or `none`.
- `run-once` → Boolean flag to execute operation only once.
- `header-auto-align` → (@llm) Adjust generated heading levels.
- `model` → Alias; provider inferred via settings.
Priority rule: If both `block` & `prompt` present they concatenate (blocks first). Missing both (where at least one required) → error.

## @import
Purpose: Inject external markdown (entire file or subsection) into current context.

**Parameters:**
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `file` | Yes | string | Source path: folder/file.md or folder/file.ctx |
| `block` | No | string | Source block path: block/subblock/* where /* is optional for nested blocks |
| `mode` | No | string | How to insert content: append, prepend, replace (default: append) |
| `to` | No | string | Target block path where content will be placed |
| `run-once` | No | boolean | Whether this operation should only run once (default: false) |

**Examples:**
```markdown
@import
file: docs/snippet.md
```
Section import:
```markdown
@import
file: docs/reference.md
block: intro/*
```
Params: `file` (req), `block` (optional slice), `mode`, `to`.
Failure: missing file / block → error.
Tips: Centralize reusable templates; prefer narrow block import for token efficiency.

## @llm
Purpose: Core generative / reasoning / transformation operation. It can:
- Consume explicit knowledge blocks and/or a literal prompt
- Orchestrate multi-turn tool-augmented reasoning (agentic loop)
- Invoke MCP tools (search, code introspection, external APIs)
- Dynamically call other Fractalic agents via an MCP tool (e.g. `fractalic_run`)
- Produce structured (JSON / tables / markdown) or free-form output
- Persist raw output to disk to reduce context bloat

### Minimal Examples
Plain literal prompt:
```markdown
@llm
prompt: "List 3 strategic risks for a small AI startup."
```
Blocks only:
```markdown
@llm
block: context/*
use-header: "# Context Summary"
```
Prompt + blocks (blocks concatenated first, then prompt):
```markdown
@llm
prompt: "Compare the two alternatives above and recommend one."
block:
  - option-a/*
  - option-b/*
```
Raw JSON output (no wrapper heading):
```markdown
@llm
prompt: |
  Return ONLY a JSON object with keys: title, bullets (array of strings).
use-header: none
stop-sequences:
  - "\n```"
  - "\n# "
```

### Key Parameters (Extended)
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `prompt` | Conditional | string | Literal input text (multi-line allowed). Either `prompt` or `block` must be present. |
| `block` | Conditional | string or array | One path or array of block paths (supports wildcards `/*`). |
| `media` | No | array | File paths for multimodal context. |
| `save-to-file` | No | string | Persist raw model response pre-merge. |
| `use-header` | No | string | Wrapper heading (default internal value) or `none` to emit raw content. |
| `header-auto-align` | No | boolean | Re-level generated headings to nest under current heading. |
| `mode` | No | string | Merge strategy: append, prepend, replace (default: append). |
| `to` | No | string | Target block path for merge. |
| `provider` | No | string | Explicit provider override (normally inferred from `model`). |
| `model` | No | string | Logical model name (provider inferred via settings). |
| `temperature` | No | number | Temperature setting for LLM call to control randomness (0-1). |
| `run-once` | No | boolean | Whether this operation should only run once (default: false). |
| `stop-sequences` | No | array | Hard stop substrings—truncates generation early. |
| `tools` | No | string or array | MCP tool identifiers; `none` disables tool loop, `all` enables all tools. |
| `tools-turns-max` | No | integer | Upper bound on tool reasoning cycles. |
| `context` | No | string | `auto` (ambient headings when only prompt) or `none` (strict isolation). |

(For semantics of each parameter see Syntax Reference §4.7.)

### Prompt Assembly Internals
Order:
1. Referenced block contents (in listed order)
2. Literal `prompt`
3. Media attachments (provider-handled encodings)
4. System / framework prompts (implicit)
If both `prompt` & `block` absent → validation error.

### Tool / Agentic Loop Lifecycle
When `tools` != `none`:
1. Initial model call issued with current messages.
2. Model may emit tool call instructions.
3. Fractalic executes tools (MCP / internal) and captures raw responses.
4. Responses parsed → textual content extracted (fields: `return_content`, `content`, `output`, etc.).
5. Tool Loop AST built (preserving `return_nodes_attribution`).
6. Tool-generated nodes inserted (flagged `is_tool_generated`).
7. Model re-called with augmented context.
8. Repeat until: a) model emits final response; b) `tools-turns-max` reached; c) provider stops.
9. Final answer merged per `mode` + `to`.

Visualization:
```
[context blocks] → model → tool calls? → execute tools → merge tool outputs → model → ... → final answer
```

### Using MCP Tools
Declare tools:
```markdown
@llm
prompt: "Get open issues mentioning 'latency' then summarize dominant themes."
tools: github_issues_search, text_cluster
```
Representative raw tool JSON:
```json
{
  "tool": "github_issues_search",
  "return_content": "Issue #42: Latency spike...\nIssue #57: Cold start delay...",
  "return_nodes_attribution": [
    {"source": "github_issues_search", "node_id": "issue-42"}
  ]
}
```
Extracted text is appended before the next model turn.

### Calling Another Fractalic Agent via `fractalic_run`
Let the model decide if an agent is required:
```markdown
@llm
prompt: "If planning is needed, call the planning agent. Otherwise provide a summary."
tools: fractalic_run
```
Conceptual tool invocation payload from model:
```json
{
  "name": "fractalic_run",
  "arguments": {
    "file": "agents/plan.md",
    "prompt": "Create a 3-step investigation plan for scaling latency tests"
  }
}
```
Returned agent output is merged as tool-generated context; reasoning continues.

### Referencing Tool-Generated Context
Tool outputs appear as normal user-role blocks with auto IDs. For consistent reuse, instruct the model (in the same call) to emit explicit headers:
```markdown
@llm
prompt: "Fetch stats; put final actionable plan under a heading '# Action Plan {id=action-plan}'."
tools: repo_stats
```
Later:
```markdown
@llm
prompt: "Refine just the plan."
block: action-plan
```

### Ensuring Determinism & Safety
- Constrain loops: set `tools-turns-max` (2–4 typical).
- Add explicit decision criteria: “Call a tool only if data X is missing.”
- For JSON output: `use-header: none` + schema instructions + optional `stop-sequences`.
- Avoid broad wildcards early—curate context.

### Failure & Edge Case Handling
| Symptom | Likely Cause | Mitigation |
|---------|--------------|-----------|
| Empty tool loop result | Tool output lacked extractable fields | Inspect raw JSON; adjust tool or extraction logic |
| Repeated tool calls no progress | Prompt missing termination guidelines | Add explicit stop criteria |
| Agent file not found | Wrong path in tool args | Verify file existence / relative path |
| Mis-leveled headings | Deep nesting + raw `#` output | Set `header-auto-align: true` |
| JSON polluted by headings | Wrapper heading present | `use-header: none` |
| Truncated mid-thought | Overly aggressive stop sequence | Remove / refine sequences |

### Performance & Token Strategy
- Stage: broad gather → compress summary → downstream use summary.
- Use `replace` to keep long evolving artifacts slim once stable.
- Save bulky outputs via `save-to-file` then import trimmed extracts later.
- Prefer narrow block IDs over `/*` across huge trees.

### Design Pattern Examples
Research → Tool Loop → Synthesis:
```markdown
@llm
prompt: "Investigate dependency update risks. Use tools if needed. Provide prioritized risk list."
tools: github_repo_stats, semver_advice
tools-turns-max: 3
```
Dynamic Sub-Agent Decision:
```markdown
@llm
prompt: |
  Determine if we need a remediation plan. If yes invoke remediation agent, else summarize.
tools: fractalic_run
```
Structured Output:
```markdown
@llm
prompt: |
  Return JSON with keys: risks (array of {id, title, severity}), summary.
use-header: none
```
Verification Loop (shell + refine):
```markdown
@shell
prompt: "pytest -q || echo 'TEST FAIL'"
use-header: "# Test Output"

@llm
prompt: "List failing tests (if any) and propose fixes." 
block: test-output/*
```

### Validation Checklist
Before shipping an @llm workflow:
- Minimal necessary blocks selected?
- Tool usage bounded (`tools-turns-max`)?
- Structured output needs wrapper suppression? (`use-header: none`)
- Large outputs stored externally? (`save-to-file`)
- Stable IDs added for downstream references?
- Stop sequences appropriate (not over-broad)?

---
## @shell
Execute shell commands and capture stdout/stderr.

**Parameters:**
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `prompt` | Yes | string | Shell command to execute (single line or multiline) |
| `use-header` | No | string | Header for the block that will contain command output (default: "# OS Shell Tool response block"). Use 'none' to omit header completely |
| `mode` | No | string | How to insert command output: append, prepend, replace (default: append) |
| `to` | No | string | Target block where command output will be placed |
| `run-once` | No | boolean | Whether this operation should only run once (default: false) |

**Examples:**
```markdown
@shell
prompt: "ls -1"
```
Multi-line:
```markdown
@shell
prompt: |
  set -euo pipefail
  curl -s https://example.com/data.json \
    | jq '.items | length'
use-header: "# Data Count"
```
Security: Avoid blindly executing unreviewed model output. Keep commands idempotent.

## @run
Run another markdown workflow (agent) and merge its returned value.

**Parameters:**
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `file` | Yes | string | Path to markdown file to execute: folder/file.md |
| `prompt` | No | string | Optional input text to pass to the executed file |
| `block` | No | string or array | Block reference(s) to use as input |
| `use-header` | No | string | If provided with prompt, header would be appended with prompt content to target file before execution |
| `mode` | No | string | How to insert execution results: append, prepend, replace (default: append) |
| `to` | No | string | Target block where execution results will be placed |
| `run-once` | No | boolean | Whether this operation should only run once (default: false) |

**Examples:**
```markdown
@run
file: agents/refine.md
prompt: "Improve tone."
```
With block inputs:
```markdown
@run
file: agents/merge.md
block:
  - draft-a/*
  - draft-b/*
use-header: "# Unified Draft"
```
Requires callee to contain an `@return` for deterministic output.

## @return
Emit final value and terminate current workflow.

**Parameters:**
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `prompt` | Conditional | string | Literal text to return |
| `block` | Conditional | string or array | Block reference(s) to use as return content |
| `use-header` | No | string | Optional header for returned content. Use 'none' to omit header completely |

Note: At least one of `prompt` or `block` must be provided.

**Examples:**
```markdown
@return
block: final/*
```
Literal:
```markdown
@return
prompt: "DONE"
```
No `mode`/`to` (caller handles merge semantics).

## @goto
Navigate to another block in document.

**Parameters:**
| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `block` | Yes | string | Target block to navigate to (no nested flags allowed) |
| `run-once` | No | boolean | Whether this operation should only run once (default: false) |

**Examples:**
```markdown
@goto
block: decision-node
```

Note: @goto has built-in loop prevention via GOTO_LIMIT configuration.

## Internal Execution Flow
1. Parse YAML
2. Resolve block refs
3. Assemble prompt
4. Execute core action
5. Build output block(s)
6. Merge via `mode` / `to`
7. Update AST & diffs

## Capability Snapshot
| Operation | Reads Blocks | Writes Blocks | External IO | Prompt Required | File Required | Tool Loop | Has run-once |
|-----------|--------------|---------------|-------------|-----------------|--------------|-----------|--------------|
| @import   | Optional     | Yes           | FS          | No              | Yes          | No        | Yes          |
| @llm      | Optional     | Yes           | LLM / Tools | Either prompt or blocks | No | Yes (tools) | Yes          |
| @shell    | No           | Yes           | OS          | Yes             | No           | No        | Yes          |
| @run      | Optional     | Yes (callee return) | FS      | Optional        | Yes          | Indirect  | Yes          |
| @return   | Optional     | (Return value) | None       | Optional        | No           | No        | No           |
| @goto     | Optional     | No            | None        | No              | No           | No        | Yes          |

## Choosing the Right Operation
- Static knowledge: @import
- Generation / reasoning: @llm
- External verification / retrieval: @shell
- Modularity: @run
- Completion: @return
- Experimental flow: @goto

## Error Handling Patterns
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Empty @llm output | Provider auth / tool fail | Check logs & keys |
| Repeated heading clutter | Using append for iterative refinement | Switch to replace when stable |
| Shell hang | Long running process | Add timeouts / simplify |
| No sub-agent output | Missing `@return` in callee | Add return block |
| Duplicate imported structure | Re-import loop | Deduplicate or centralize imports |

## Composition Patterns
Refinement:
```markdown
# Draft {id=draft}
Initial text.

@llm
prompt: "Improve clarity of draft above."
block: draft
mode: replace
to: draft  # In-place refinement target
```
Test then summarize:
```markdown
@shell
prompt: "pytest -q || echo 'TEST FAIL'"
use-header: "# Test Output"

@llm
prompt: "Summarize test output and list failing tests if any."
block: test-output/*
```

## Performance Tips
- Collapse large intermediate blocks into compressed summaries.
- Use `save-to-file` for bulky outputs not needed in future prompts.
- Prefer targeted block refs over broad wildcards.

## Supplemental Clarifications (Added, Not Altering Verbatim Content Above)
- The detailed @llm subsections 5.3.1–5.3.12 expand operational semantics; no paraphrasing of original intent.
- Capability snapshot table aligns with earlier matrix (fields normalized to consistent labels).
- Performance tips consolidated; original guidance preserved under respective subsections.
