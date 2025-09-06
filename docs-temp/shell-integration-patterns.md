# 9. Shell Integration Patterns

Purpose: Show how to run shell commands safely inside Fractalic and merge results without uncontrolled growth.

Why it matters:
- Automate: Capture build steps, scans, generators inline.
- Reproduce: Everything is in markdown history (diffable).
- Compress: Summarize noisy outputs early to control tokens.
- Chain: Feed command output directly into later `@llm` steps.

---
### Internal Table of Contents
- [9.1 Basics (`@shell`)](#91-basics-shell)
- [9.2 Minimal & Multi-Line Forms](#92-minimal--multi-line-forms)
- [9.3 Selecting Where Output Goes (`mode`, `to`)](#93-selecting-where-output-goes-mode-to)
- [9.4 Summarizing Outputs](#94-summarizing-outputs)
- [9.5 Chaining Shell → LLM](#95-chaining-shell--llm)
- [9.6 Repeated Tasks & Refresh Patterns](#96-repeated-tasks--refresh-patterns)
- [9.7 Cost & Noise Control](#97-cost--noise-control)
- [9.8 Common Pitfalls](#98-common-pitfalls)
- [9.9 Quick Reference](#99-quick-reference)
- [Cross References](#cross-references)

---
## 9.1 Basics (`@shell`)
Run a command and capture its stdout + stderr (interleaved) as a new block.
```markdown
@shell
prompt: "echo 'Hello'"
```
Multi-line script (use `|`):
```markdown
@shell
prompt: |
  set -e
  echo "Building"
  python -m pytest -q
```
Result is inserted after the operation (default `append`).

## 9.2 Minimal & Multi-Line Forms
Form | When
---- | ----
Single line (no pipe) | Short one-liner, no special chars needing YAML escaping.
Multi-line (`|`) | Any script with newlines, quotes, or colons.

## 9.3 Selecting Where Output Goes (`mode`, `to`)
Control how the output merges:
- `append` (default) grows history.
- `prepend` positions output before existing content.
- `replace` overwrites a target block's body (after stabilization).
Target another block:
```markdown
@shell
prompt: "ls -1 src"
to: workspace-scan
mode: replace
```
(Requires a heading with `{id=workspace-scan}` already defined.)

## 9.4 Summarizing Outputs
Large logs inflate later prompts. Pattern:
```markdown
@shell
prompt: |
  set -e
  pytest -q --maxfail=1
```
Then compress:
```markdown
@llm
prompt: |
  Summarize the latest test output into <=12 bullet points (failures first). Keep only actionable info.
block: shell-response/*
to: test-summary
mode: replace
```
Give the shell result block an ID (edit its heading) if you plan reuse: `# Test Run Output {id=shell-response}`.

## 9.5 Chaining Shell → LLM
Direct transformation:
```markdown
@shell
prompt: "git diff HEAD~1 --name-only"

@llm
prompt: "Group the changed paths by top-level folder and note probable change types."
block: shell-response
```
Multiple shell outputs combined:
```markdown
@shell
prompt: "git rev-parse --abbrev-ref HEAD"

@shell
prompt: "git log -1 --pretty=%B"

@llm
prompt: "Create a concise release note draft."
block:
  - branch-name
  - last-commit-msg
```
Assign IDs to the generated shell output headings (after first run) so they can be referenced (`branch-name`, `last-commit-msg`).

## 9.6 Repeated Tasks & Refresh Patterns
Use `replace` when refreshing status snapshots:
```markdown
@shell
prompt: "du -sh . | sort -h"
to: size-scan
mode: replace
```
Accumulate progressive evidence first, then prune:
```markdown
@shell
prompt: "grep -R 'TODO' -n src"  # exploratory
@shell
prompt: "grep -R 'FIXME' -n src"
@llm
prompt: "Merge all TODO / FIXME lines into a prioritized action list (dedupe)."
block:
  - shell-todo
  - shell-fixme
mode: replace
to: code-remediation-plan
```

## 9.7 Cost & Noise Control
Technique | Why
--------- | ---
Add IDs early | Makes re-selection explicit; prevents wildcard overreach.
Summarize long logs | Shrinks context for later LLM steps.
Use `replace` on stable snapshots | Prevents endless growth.
External file storage (future pattern) | Keep raw logs out of hot context (then import distilled form). 
Limit breadth of scans | Narrow glob patterns to required subtrees.

## 9.8 Common Pitfalls
Pitfall | Impact | Fix
------- | ------ | ---
Forgetting `|` for multi-line script | YAML parse issues / truncated command | Always use `|` for multi-line.
Overusing append for recurring scans | Token bloat | Switch to `replace` after first stable format.
No IDs on shell outputs | Hard to target later | Edit heading to add `{id=...}`.
Summarizing too late | High cost paid already | Summarize immediately if large.
Blindly capturing huge directories | Irrelevant noise | Scope commands narrowly.

## 9.9 Quick Reference
Need | Pattern
---- | -------
One-liner | prompt: "echo 'Hi'"
Multi-line | prompt: | (then lines)
Snapshot refresh | mode: replace + to: <id>
Transform shell → summary | @llm with block: shell-output-id
Combine multiple runs | block: [ id-a, id-b ]
Prune noise | Summarize + replace

---
## Cross References
- [Syntax Reference](syntax-reference.md)
- [Operations Reference](operations-reference.md)
- [Context Management](context-management.md)
