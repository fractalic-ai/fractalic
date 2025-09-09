---
title: Agent & Modular Workflows
description: Build reusable agents and compose complex workflows using modular patterns and @run operations
outline: deep
---

# Agent & Modular Workflows

Purpose: Learn how to break complex AI workflows into manageable, reusable pieces using separate markdown files that work together. Perfect for newcomers who want to understand Fractalic's modular approach.

## Table of Contents
- [The Basics: Your First Agent](#the-basics-your-first-agent)
- [Understanding @run (Manual Execution)](#understanding-run-manual-execution)
- [How Information Flows Between Files](#how-information-flows-between-files)
- [Using @return to Export Results](#using-return-to-export-results)
- [Dynamic Agent Calling (fractalic_run tool)](#dynamic-agent-calling-fractalic_run-tool)
- [Viewing Results: Session Tree & Artifacts](#viewing-results-session-tree--artifacts)
- [Complete Worked Example](#complete-worked-example)
- [File Organization & Best Practices](#file-organization--best-practices)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Quick Reference](#quick-reference)

---
## What You'll Learn
By the end of this guide, you'll understand how to create agent files (specialized markdown documents), connect them together, control what information flows between them, and troubleshoot common issues. Think of it like building with LEGO blocks—each agent file does one job well and you combine them to build sophisticated workflows.

---
## The Basics: Your First Agent

### What is an Agent File?
An agent file is just a regular markdown file with some operations (like `@llm`, `@shell`) that performs a focused task. Let's create the simplest possible agent:

**File: `agents/simple-summarizer.md`**
```yaml
# Simple Summarizer Agent {id=simple-summarizer}

@llm
prompt: "Summarize the input in 3 bullet points."
block: input-parameters/*

@return
block: llm-response
```

That's it! This agent takes whatever you give it, asks the AI to summarize it, and returns the result.

### How to Use This Agent
From another file (let's call it `main.md`):
```yaml
# My Main Workflow

## Some Data {id=some-data}
Here's information about our Q3 sales performance...
(lots of details)

## Get Summary

@run
file: agents/simple-summarizer.md
block: some-data
```

**What happens:**
1. Fractalic takes your "Some Data" section
2. Runs the `simple-summarizer.md` file in isolation  
3. The agent execution context gets an automatically created `# Input Parameters {id=input-parameters}` section at the top containing what you passed
4. Returns the 3-bullet summary back to your main file

## Understanding @run (Manual Execution)

### The Complete Picture
When you write `@run`, here's the lifecycle:
1. Preparation (load target file)
2. Input injection (`prompt` / `block` merged into a synthetic top section)
3. Execution (operations run in file order)
4. Output capture (any `@return` blocks collected)
5. Integration (returned content merged into caller)

### What the Agent File Actually Sees
Call an agent like:
```yaml
@run
file: agents/risk-eval.md
prompt: |
  Focus on security issues only.
block: commit-analysis/*
```

Execution context (diff view showing injected content):
```js
# Input Parameters {id=input-parameters} // [!code highlight]
Focus on security issues only. // [!code highlight]

## Commit Analysis Summary {id=commit-summary} // [!code highlight]
- Added new authentication method // [!code highlight]
- Updated user database schema // [!code highlight]
- Fixed SQL injection vulnerability // [!code highlight]

# Risk Evaluation Agent {id=risk-eval}
(... rest of the original file content ...)
```

**Key points:**
- Synthetic heading: `# Input Parameters {id=input-parameters}`
- Reference it with: `block: input-parameters/*`
- Temporary (not written to disk)

### @run Parameters Explained
```yaml
@run
file: path/to/agent.md          # Required: which agent to run
block: section-id/*             # Optional: what context to pass in
prompt: |                       # Optional: guidance for the agent
  Additional instructions here
mode: append                    # Optional: append | prepend | replace
to: destination-section         # Optional: where to put returned results
```

**Common patterns:**
- Just run: `@run file: agent.md`
- Pass data: `@run file: agent.md block: my-data/*`
- Pass data + guidance: add `prompt:`
- Control output location: add `to:`

## How Information Flows Between Files

### Isolation Principle
Agents ONLY see what you pass via `block:` and/or `prompt:`.

```yaml
# Main File
## Confidential Notes {id=confidential}
Internal team discussions...

## Public Data {id=public-data}
Customer feedback shows...

@run
file: agents/analyzer.md
block: public-data  # Only this section is visible to the agent
```

Multiple blocks:
```yaml
@run
file: agents/comprehensive-analyzer.md
block:
  - customer-feedback/*
  - market-research/*
  - competitor-data
```

Wildcards:
- `section/*` = section + descendants
- `section` = that section only

## Using @return to Export Results

```yaml
@llm
prompt: "Analyze the data and create a risk assessment."
block: input-parameters/*
to: risk-assessment

@return
block: risk-assessment
```

Multiple:
```yaml
@return
block:
  - risk-assessment
  - recommendations
  - action-items
```

Custom content:
```yaml
@return
prompt: |
  # Analysis Complete
  Processing finished.
```

## Dynamic Agent Calling (fractalic_run tool)

```yaml
@llm
prompt: |
  Analyze commit messages. If security-related changes appear, call
  agents/security-reviewer.md with commit-summary/*; else summarize only.
  Use fractalic_run ONLY if needed.
block: commit-summary
tools:
  - fractalic_run
tools-turns-max: 2
```

Tool result render (diff):
```js
> TOOL RESPONSE, id: call_abc123 // [!code highlight]
response: // [!code highlight]
content: "_IN_CONTEXT_BELOW_" // [!code highlight]

# Security Analysis Results {id=security-analysis} // [!code highlight]
## Critical Issues Found // [!code highlight]
- New authentication bypass // [!code highlight]
- Overly broad DB permissions // [!code highlight]

## Recommendations // [!code highlight]
- Immediate code review // [!code highlight]
- Tighten role policies // [!code highlight]
```

Explanation of the placeholder:
- The string `_IN_CONTEXT_BELOW_` is a deliberate sentinel value. The tool framework replaces the raw tool response body with this placeholder in the tool log so the conversation does not duplicate large markdown output twice.
- The actual generated markdown (sections beginning with `# Security Analysis Results ...`) is injected directly beneath the tool log in the session context.
- Why it matters: When you review traces you can trust that anything after a tool response containing `_IN_CONTEXT_BELOW_` is the real material the model received / produced—not an echo.
- Do not remove or alter this placeholder; it is part of the stable contract for rendering dynamic agent outputs.

## Viewing Results: Session Tree & Artifacts
`.ctx` (execution context) and `.trc` (trace) files are generated per run or tool invocation.

Context file structure:
```markdown
# Input Parameters {id=input-parameters}
(passed prompt + blocks)

# Agent Original Content ...

# Returned / Generated Sections
```

## Complete Worked Example

A newcomer-friendly, non-developer scenario: turning messy meeting notes into a structured weekly update using three small agents.

### Goal
Start with raw meeting notes → extract structured topics → assess risks & action items → produce a polished weekly update.

### Step 1: Create the Agent Files

**File: `agents/meeting-topic-extractor.md`**
```yaml
# Meeting Topic Extractor {id=meeting-topic-extractor}

@llm
prompt: |
  You are an assistant that cleans messy meeting notes.
  1. Identify distinct discussion topics.
  2. For each topic list the key points (max 5 bullets).
  3. Ignore chit-chat or scheduling noise.
block: input-parameters/*
mode: replace
to: analyzed-topics

@return
block: analyzed-topics
```

**File: `agents/risk-action-assessor.md`**
```yaml
# Risk & Action Assessor {id=risk-action-assessor}

@llm
prompt: |
  From the topics and key points:
  - Extract any implicit or explicit risks (label High / Medium / Low)
  - List concrete action items (who + what if names appear; else generic owner)
  - Highlight any blocked items
block: input-parameters/*
mode: replace
to: risks-actions

@return
block: risks-actions
```

**File: `agents/weekly-update-writer.md`**
```yaml
# Weekly Update Writer {id=weekly-update-writer}

@llm
prompt: |
  Produce a clear weekly project update using:
  - Structured topics
  - Risks & action items
  Sections to include:
  1. Overview (2 sentences)
  2. Key Progress
  3. Risks (with severity)
  4. Action Items (checkbox list)
  5. Next Week Focus
block: input-parameters/*
mode: replace
to: weekly-update

@return
block: weekly-update
```

### Step 2: Orchestrator File

**File: `weekly-update-workflow.md`**
```yaml
# Weekly Update Workflow {id=weekly-update-workflow}

## Raw Meeting Notes {id=raw-notes}
Team sync covered onboarding delays for Region A, marketing launch timeline, data quality cleanup, and a potential vendor contract issue. Sarah flagged that the analytics dashboard refresh is still unreliable. We also celebrated early pilot feedback being positive. Next week we expect first draft of the onboarding checklist.

## Extract Topics

@run
file: agents/meeting-topic-extractor.md
block: raw-notes

## Assess Risks & Actions

@run
file: agents/risk-action-assessor.md
block:
  - analyzed-topics

## Create Weekly Update

@run
file: agents/weekly-update-writer.md
block:
  - analyzed-topics
  - risks-actions

@return
block: weekly-update
```

### Step 3: Post-Execution Diff (What You See)
```js
# Weekly Update Workflow {id=weekly-update-workflow}

## Raw Meeting Notes {id=raw-notes}
...original notes...

## Extract Topics
@run file: agents/meeting-topic-extractor.md

# Analyzed Topics {id=analyzed-topics} // [!code highlight]
## Onboarding Delays // [!code highlight]
- Region A behind schedule // [!code highlight]
- Awaiting checklist draft // [!code highlight]
## Marketing Launch Timeline // [!code highlight]
- Launch planning in progress // [!code highlight]
## Data Quality Cleanup // [!code highlight]
- Ongoing cleanup efforts // [!code highlight]
## Vendor Contract Issue // [!code highlight]
- Potential complication flagged // [!code highlight]
## Dashboard Reliability // [!code highlight]
- Refresh still unreliable // [!code highlight]
## Positive Pilot Feedback // [!code highlight]
- Early feedback encouraging // [!code highlight]

## Assess Risks & Actions
@run file: agents/risk-action-assessor.md

# Risks & Actions {id=risks-actions} // [!code highlight]
## Risks // [!code highlight]
- Onboarding delay (Medium) // [!code highlight]
- Vendor contract complication (Medium) // [!code highlight]
- Dashboard reliability (High) // [!code highlight]
## Action Items // [!code highlight]
- Prepare onboarding checklist (Owner: Onboarding Lead) // [!code highlight]
- Review vendor terms (Owner: Ops) // [!code highlight]
- Stabilize dashboard refresh (Owner: Eng) // [!code highlight]
## Blocked / Watch // [!code highlight]
- Dashboard fix awaiting diagnostics // [!code highlight]

## Create Weekly Update
@run file: agents/weekly-update-writer.md

# Weekly Update {id=weekly-update} // [!code highlight]
## Overview // [!code highlight]
Progress on multiple fronts; onboarding and dashboard stability need focus. // [!code highlight]
## Key Progress // [!code highlight]
- Positive pilot feedback // [!code highlight]
- Marketing prep advancing // [!code highlight]
## Risks // [!code highlight]
- Dashboard reliability (High) // [!code highlight]
- Vendor contract (Medium) // [!code highlight]
## Action Items // [!code highlight]
- [ ] Onboarding checklist draft // [!code highlight]
- [ ] Vendor terms review // [!code highlight]
- [ ] Dashboard stability work // [!code highlight]
## Next Week Focus // [!code highlight]
Stabilize dashboard; finalize onboarding materials. // [!code highlight]
```

### Step 4: Artifacts
Generated:
- `weekly-update-workflow.ctx`
- `agents/meeting-topic-extractor.ctx`
- `agents/risk-action-assessor.ctx`
- `agents/weekly-update-writer.ctx`

Why useful:
- Clean separation (extraction → evaluation → publication)
- Non-technical: can be reused for any meeting
- Easy to swap writer agent for different report formats

## File Organization & Best Practices

### Recommended Folder Structure
```bash
your-project/
├── main-workflow.md          # Entry point
├── agents/
│   ├── analyzers/
│   │   ├── commit-analyzer.md
│   │   └── text-analyzer.md
│   ├── evaluators/  
│   │   ├── risk-evaluator.md
│   │   └── quality-evaluator.md
│   └── writers/
│       ├── release-note-writer.md
│       └── summary-writer.md
└── templates/
    ├── report-template.md
    └── analysis-template.md
```

### Naming Conventions
- **Agent files**: Use action verbs (`analyze-commits.md`, `evaluate-risks.md`)
- **IDs**: Use kebab-case (`commit-analysis`, `risk-summary`)  
- **Folders**: Group by function (`analyzers/`, `evaluators/`, `writers/`)

### Agent Design Principles
1. **Single responsibility**: Each agent should do one thing well
2. **Stable IDs**: Use explicit `{id=...}` for sections you'll reference later
3. **Clear inputs**: Document what blocks the agent expects in a comment at the top
4. **Meaningful returns**: Only return what the caller actually needs

### When to Create New Agents vs Expanding Existing
**Create a new agent when:**
- The task is conceptually different (analysis vs writing)
- You want to reuse the logic elsewhere  
- The current agent is getting complex (>5 operations)
- Different people will maintain different parts

**Expand existing agent when:**
- It's a small addition to existing logic
- The tasks are tightly coupled
- Creating separate agents would add unnecessary complexity

## Troubleshooting Guide

### Common Error Messages and Solutions

#### "File not found: agents/my-agent.md"
**Problem**: Wrong file path in `@run`
**Solution**: 
- Check spelling and path relative to your project root
- Use `ls agents/` to verify the file exists
- Ensure you're in the right working directory

#### "Block with URI 'my-block' not found"  
**Problem**: Trying to reference a section that doesn't exist
**Solution**:
- Check the exact ID spelling: `{id=my-block}`
- Verify the section exists before the `@run` operation
- Use wildcard `my-section/*` if you want all subsections

#### "Empty tool output body"
**Problem**: Agent ran but didn't return anything
**Solution**:
- Add `@return` operation to your agent file
- Make sure `@return` references blocks that actually exist
- Check the agent's `.ctx` file to see what was generated

#### Agent seems to ignore your input
**Problem**: Agent doesn't reference the passed blocks
**Solution**:
- Ensure agent uses `block: input-parameters/*` to access your data
- Check that you're actually passing blocks: `@run file: agent.md block: my-data`
- Review the agent's `.ctx` file to confirm input injection

#### "Model ignores fractalic_run tool"
**Problem**: Dynamic calling isn't working
**Solution**:
- Add explicit conditions: "Call fractalic_run ONLY if..."
- Include the tool in `tools:` list
- Give clear guidance about when to use it
- Check that `tools-turns-max` allows enough turns

### Debugging Workflow
1. **Check the main context**: Does your input section have the right data?
2. **Check agent context**: Open the `.ctx` file - what did the agent actually see?
3. **Check return**: Did the agent create the blocks it's trying to return?
4. **Check integration**: Are returned blocks appearing in the right place?

### Performance Issues
#### "Agent taking too long"
- Check if you're passing huge blocks (use summaries instead)
- Reduce wildcard selections (`section/*` → specific IDs)
- Split complex agents into smaller steps

#### "High token costs"
- Pass summaries, not raw data
- Use `mode: replace` to avoid accumulating history  
- Limit `tools-turns-max` on dynamic calling
- Check for redundant agent calls

## Quick Reference

### @run Syntax (What it does and why)
Run another markdown agent file in isolation, injecting only the blocks and/or prompt you specify so it cannot accidentally see unrelated content.
```yaml
@run
file: path/to/agent.md          # Required
block: section-id/*             # Optional: input data
prompt: |                       # Optional: guidance  
  Instructions for agent
mode: append                    # Optional: append/prepend/replace
to: destination                 # Optional: where to put results
```

### @return Syntax (Export only what matters)
Return selected blocks (or custom content) so the caller receives a clean, minimal result.
```yaml
@return
block: section-id               # Single block

@return  
block:                          # Multiple blocks
  - section-1
  - section-2
  
@return                         # Custom content
prompt: |
  Custom return message
```

### fractalic_run Tool Parameters (Dynamic decisions)
Call another agent mid-LLM reasoning only when conditions are met. Keep turns low to control cost.
```json
{
  "file_path": "agents/analyzer.md",
  "block_uri": ["section-1/*", "section-2"],
  "prompt": "Focus on X and Y"
}
```

### Common Patterns (With proper spacing + explanations)
```yaml
# Simple agent call

@run
file: agents/summarizer.md
block: data-section  # Pass one section

# Pass multiple sections  

@run
file: agents/analyzer.md
block:
  - section-a/*      # Include section + its children
  - section-b        # Single section only

# Dynamic calling (only if needed)

@llm
prompt: "If complex analysis needed, use fractalic_run"
tools: [fractalic_run]
tools-turns-max: 2  # Limit tool loop turns

# Return multiple results

@return
block:
  - summary
  - recommendations
  - next-steps
```
Guidance:
- Always leave a blank line before each operation (`@run`, `@llm`, `@return`) to avoid parsing issues.
- Use specific IDs so returned sections are predictable.
- Keep dynamic tool calling conditional; state clearly when to invoke it.

---
## What's Next?
- **Practice**: Start with simple agents and gradually build complexity
- **Explore**: Check out the generated `.ctx` files to understand execution flow  
- **Organize**: Develop your own agent library for common tasks
- **Share**: Well-designed agents can be reused across projects and teams

The key to mastering modular workflows is thinking in terms of focused, reusable components rather than monolithic files. Each agent should solve one problem exceptionally well.
