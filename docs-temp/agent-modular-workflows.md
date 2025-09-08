---
title: Agent & Modular Workflows
description: Build reusable agents and compose complex workflows using modular patterns and @run operations
outline: deep
---

# Agent & Modular Workflows

Purpose: Learn how to break complex AI workflows into manageable, reusable pieces using separate markdown files that work together. Perfect for newcomers who want to understand Fractalic's modular approach.

## Table of Contents
- [10.1 The Basics: Your First Agent](#101-the-basics-your-first-agent)
- [10.2 Understanding @run (Manual Execution)](#102-understanding-run-manual-execution)
- [10.3 How Information Flows Between Files](#103-how-information-flows-between-files)
- [10.4 Using @return to Export Results](#104-using-return-to-export-results)
- [10.5 Dynamic Agent Calling (fractalic_run tool)](#105-dynamic-agent-calling-fractalic_run-tool)
- [10.6 Viewing Results: Session Tree & Artifacts](#106-viewing-results-session-tree--artifacts)
- [10.7 Complete Worked Example](#107-complete-worked-example)
- [10.8 File Organization & Best Practices](#108-file-organization--best-practices)
- [10.9 Troubleshooting Guide](#109-troubleshooting-guide)
- [10.10 Quick Reference](#1010-quick-reference)

---
## What You'll Learn
By the end of this guide, you'll understand how to create agent files (specialized markdown documents), connect them together, control what information flows between them, and troubleshoot common issues. Think of it like building with LEGO blocks—each agent file does one job well and you combine them to build sophisticated workflows.

---
## 10.1 The Basics: Your First Agent

### What is an Agent File?
An agent file is just a regular markdown file with some operations (like `@llm`, `@shell`) that performs a focused task. Let's create the simplest possible agent:

**File: `agents/simple-summarizer.md`**
```markdown
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
```markdown
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

## 10.2 Understanding @run (Manual Execution)

### The Complete Picture
When you write `@run`, here's the lifecycle:
1. Preparation (load target file)
2. Input injection (`prompt` / `block` merged into a synthetic top section)
3. Execution (operations run in file order)
4. Output capture (any `@return` blocks collected)
5. Integration (returned content merged into caller)

### What the Agent File Actually Sees
Call an agent like:
```markdown
@run
file: agents/risk-eval.md
prompt: |
  Focus on security issues only.
block: commit-analysis/*
```

Execution context (diff view showing injected content):
```diff
+ # Input Parameters {id=input-parameters}
+ Focus on security issues only.
+ 
+ ## Commit Analysis Summary {id=commit-summary}
+ - Added new authentication method
+ - Updated user database schema  
+ - Fixed SQL injection vulnerability

# Risk Evaluation Agent {id=risk-eval}
(... rest of the original file content ...)
```

**Key points:**
- Synthetic heading: `# Input Parameters {id=input-parameters}`
- Reference it with: `block: input-parameters/*`
- Temporary (not written to disk)

### @run Parameters Explained
```markdown
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

## 10.3 How Information Flows Between Files

### Isolation Principle
Agents ONLY see what you pass via `block:` and/or `prompt:`.

```markdown
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
```markdown
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

## 10.4 Using @return to Export Results

```markdown
@llm
prompt: "Analyze the data and create a risk assessment."
block: input-parameters/*
to: risk-assessment

@return
block: risk-assessment
```

Multiple:
```markdown
@return
block:
  - risk-assessment
  - recommendations
  - action-items
```

Custom content:
```markdown
@return
prompt: |
  # Analysis Complete
  Processing finished.
```

## 10.5 Dynamic Agent Calling (fractalic_run tool)

```markdown
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
```diff
+ > TOOL RESPONSE, id: call_abc123
+ response:
+ content: "_IN_CONTEXT_BELOW_"
+ 
+ # Security Analysis Results {id=security-analysis}
+ ## Critical Issues Found
+ - New authentication bypass
+ - Overly broad DB permissions
+ 
+ ## Recommendations
+ - Immediate code review
+ - Tighten role policies
```

Explanation of the placeholder:
- The string `_IN_CONTEXT_BELOW_` is a deliberate sentinel value. The tool framework replaces the raw tool response body with this placeholder in the tool log so the conversation does not duplicate large markdown output twice.
- The actual generated markdown (sections beginning with `# Security Analysis Results ...`) is injected directly beneath the tool log in the session context.
- Why it matters: When you review traces you can trust that anything after a tool response containing `_IN_CONTEXT_BELOW_` is the real material the model received / produced—not an echo.
- Do not remove or alter this placeholder; it is part of the stable contract for rendering dynamic agent outputs.

## 10.6 Viewing Results: Session Tree & Artifacts
`.ctx` (execution context) and `.trc` (trace) files are generated per run or tool invocation.

Context file structure:
```markdown
# Input Parameters {id=input-parameters}
(passed prompt + blocks)

# Agent Original Content ...

# Returned / Generated Sections
```

## 10.7 Complete Worked Example

A newcomer-friendly, non-developer scenario: turning messy meeting notes into a structured weekly update using three small agents.

### Goal
Start with raw meeting notes → extract structured topics → assess risks & action items → produce a polished weekly update.

### Step 1: Create the Agent Files

**File: `agents/meeting-topic-extractor.md`**
```markdown
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
```markdown
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
```markdown
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
```markdown
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
```diff
# Weekly Update Workflow {id=weekly-update-workflow}

## Raw Meeting Notes {id=raw-notes}
...original notes...

## Extract Topics
@run file: agents/meeting-topic-extractor.md

+ # Analyzed Topics {id=analyzed-topics}
+ ## Onboarding Delays
+ - Region A behind schedule
+ - Awaiting checklist draft
+ ## Marketing Launch Timeline
+ - Launch planning in progress
+ ## Data Quality Cleanup
+ - Ongoing cleanup efforts
+ ## Vendor Contract Issue
+ - Potential complication flagged
+ ## Dashboard Reliability
+ - Refresh still unreliable
+ ## Positive Pilot Feedback
+ - Early feedback encouraging

## Assess Risks & Actions
@run file: agents/risk-action-assessor.md

+ # Risks & Actions {id=risks-actions}
+ ## Risks
+ - Onboarding delay (Medium)
+ - Vendor contract complication (Medium)
+ - Dashboard reliability (High)
+ ## Action Items
+ - Prepare onboarding checklist (Owner: Onboarding Lead)
+ - Review vendor terms (Owner: Ops)
+ - Stabilize dashboard refresh (Owner: Eng)
+ ## Blocked / Watch
+ - Dashboard fix awaiting diagnostics

## Create Weekly Update
@run file: agents/weekly-update-writer.md

+ # Weekly Update {id=weekly-update}
+ ## Overview
+ Progress on multiple fronts; onboarding and dashboard stability need focus.
+ ## Key Progress
+ - Positive pilot feedback
+ - Marketing prep advancing
+ ## Risks
+ - Dashboard reliability (High)
+ - Vendor contract (Medium)
+ ## Action Items
+ - [ ] Onboarding checklist draft
+ - [ ] Vendor terms review
+ - [ ] Dashboard stability work
+ ## Next Week Focus
+ Stabilize dashboard; finalize onboarding materials.
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

## 10.8 File Organization & Best Practices

### Recommended Folder Structure
```
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

## 10.9 Troubleshooting Guide

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

## 10.10 Quick Reference

### @run Syntax (What it does and why)
Run another markdown agent file in isolation, injecting only the blocks and/or prompt you specify so it cannot accidentally see unrelated content.
```markdown
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
```markdown
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
```markdown
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
