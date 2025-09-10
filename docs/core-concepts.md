---
title: Core Concepts
description: Understanding the fundamental concepts of Fractalic - documents as programs, execution context, and block types
outline: deep
---

# Core Concepts

## Purpose
You use Fractalic to turn plain Markdown into runnable workflows. This page explains the ideas that make that work: what a "block" is, how execution flows, how AI and tools add content, and how to stay in control.

## Big Idea: Documents are Programs
Think of a traditional program: you write code, run it, and see output somewhere else. Fractalic flips this idea. Your document *is* the program, and when you run it, the results get written directly back into the same document. This means you can see your original intent alongside what actually happened.

Here's how it works: You start with a regular Markdown file containing your thoughts, plans, or questions. Then you sprinkle in small "operation" instructions using simple YAML syntax. When Fractalic runs your document, it executes these operations one by one, adding new sections with the results. Your document grows and evolves, becoming a complete record of both your original ideas and what the AI or tools produced.

This approach solves a common problem: losing track of what you asked for and what you got back. With Fractalic, everything stays together in one readable document that tells the complete story.

## Understanding Knowledge Blocks
A knowledge block is simply a Markdown heading and all the content that follows it, up until the next heading or operation. Think of it as a labeled container for information.

For example:
```markdown
# Project Requirements {id=requirements}
We need to build a simple website that showcases our portfolio. 
The site should be responsive and load quickly.
It needs a contact form and gallery section.
```

This entire section—from the heading down to the last sentence—forms one knowledge block. Later in your document, you can reference this block by its ID (`requirements`) and Fractalic will know exactly what content you're talking about.

Knowledge blocks are important because they let you organize information and then reference specific pieces later. Instead of copying and pasting text around your document, you can just point to the block you want to use.

## Understanding Operation Blocks
An operation block tells Fractalic to *do something*. It starts with a line beginning with `@` followed by the operation name, then includes YAML parameters that specify exactly what to do.

Here's a simple example:
```markdown
@llm
prompt: "Create three color scheme options for a modern website"
use-header: "# Color Schemes"
```

When Fractalic encounters this operation, it will send the prompt to an AI language model, wait for the response, then insert a new section called "Color Schemes" containing the AI's suggestions. The operation itself stays in place, but new content appears after it.

Operations are the "action" part of your document. While knowledge blocks store information, operations create new information by calling AI models, running shell commands, importing files, or executing other workflows.

Example (minimal):
```markdown
# Goal
Describe a simple greeting.

@llm
prompt: "Write a cheerful two‑sentence greeting that mentions Fractalic."
blocks: goal
use-header: "# AI Greeting"
```

What happens:
- Fractalic reads the Goal block (via blocks: goal).
- It sends your prompt plus that context to the LLM.
- It inserts a new section "AI Greeting" with the model's response.

## How Block IDs Work
Every knowledge block needs an identifier so you can reference it later. You have two options: let Fractalic create one automatically, or specify your own.

**Automatic IDs:** If you write a heading like `# Project Requirements`, Fractalic automatically creates the ID `project-requirements` (lowercase, with dashes replacing spaces and special characters removed).

**Explicit IDs:** For more control, you can specify your own ID by adding `{id=custom-name}` to any heading:
```markdown
# My Complex Project Title {id=project}
```

Now you can reference this block simply as `project` instead of `my-complex-project-title`.

**Nested Paths:** When you have headings inside other headings, you can reference them using paths:
```markdown
# Research
## Market Analysis
## User Studies
```

You can reference the market analysis section as `research/market-analysis`. This path structure helps you organize complex documents with many sections.

**Wildcards for Whole Branches:** Sometimes you want to reference a section and everything inside it. Use the `/*` syntax:
```markdown
blocks: research/*
```

This selects the "Research" heading plus both "Market Analysis" and "User Studies" subsections. It's useful when you want an AI to consider all related information together.

## How to Select Blocks
When writing operations, you tell Fractalic which blocks to use as context. There are two clean ways to do this:

**Single block:**
```yaml
blocks: summary
```

**Multiple blocks (preserving order):**
```yaml
blocks:
  - research/*
  - decisions
  - timeline
```

The order matters because Fractalic will present the blocks to the AI in the sequence you specify. This lets you control how information flows and builds upon itself.

## How Execution Works Step by Step
Understanding how Fractalic processes your document helps you write more effective workflows. Here's what happens when you run a Fractalic document:

**Step 1: Reading from Top to Bottom**
Fractalic starts at the beginning of your document and works its way down, processing each knowledge block and operation in sequence.

**Step 2: Building Context for Operations**
When Fractalic encounters an operation like `@llm`, it needs to gather the right information to send to the AI. It does this by:
- First, collecting any blocks you specified (like `blocks: research/*`)
- Then, adding your prompt text at the end
- Finally, if you didn't specify any blocks, it might automatically include content from earlier in the document (you can turn this off if needed)

> **Note:** Specifying `blocks:` disables automatic context inclusion.  
> When you explicitly list blocks in an operation, Fractalic will *only* use those blocks as context for the AI or tool. Earlier content in your document will not be included automatically. This gives you precise control, but means you must select all relevant information yourself.

**Step 3: Executing the Operation**
Fractalic runs the operation—sending text to an AI model, executing a shell command, importing a file, or running another workflow.

**Step 4: Adding Results Back to Your Document**
The operation's output gets inserted back into your document under a clear heading. You can control where this happens and what the heading says.

Here's a simple example showing the complete flow:

```js
# Website Ideas
I want to create a portfolio website to showcase my design work.

@llm
prompt: "Based on the website goal above, suggest 5 essential pages"
blocks: website-ideas
use-header: "# Suggested Pages"

# Suggested Pages // [!code highlight]
1. Home - Welcome visitors with your best work preview // [!code highlight]
2. Portfolio - Showcase your design projects with case studies // [!code highlight]
3. About - Tell your story and share your design philosophy // [!code highlight]
4. Services - Explain what you offer to potential clients // [!code highlight]
5. Contact - Make it easy for people to reach out // [!code highlight]
```

Notice how the AI read your original idea (the "Website Ideas" block), processed your prompt, and created a new section with specific suggestions. The document now contains both your original intent and the AI's response, clearly labeled and ready for you to reference in future operations.

## Understanding Merge Modes
When an operation produces output, Fractalic needs to know where to put it and how to combine it with existing content. This is controlled by the "merge mode" and target settings.

**Append Mode (Default)**
This adds new content after the target location. It's perfect for building up information over time:

**Before execution:**
```js
# Research Notes
Initial findings about user behavior.

@llm
prompt: "Add insights about mobile usage patterns"
blocks: research-notes
mode: append
to: research-notes
```

**After execution:**
```js
# Research Notes
Initial findings about user behavior.

# Mobile Usage Insights // [!code highlight]
Mobile users prefer simple navigation with large touch targets. They typically // [!code highlight]
browse during commutes and expect fast loading times. Content should be // [!code highlight]
scannable with clear headings and minimal scrolling required. // [!code highlight]

@llm
prompt: "Add insights about mobile usage patterns"
blocks: research-notes
mode: append
to: research-notes
use-header: "# Mobile Usage Insights"
```

**Replace Mode**
This completely replaces the target content with new content. Use this when you want to refine or update something rather than add to it:

**Before execution:**
```js
# Draft Summary
This is a rough first draft that needs improvement.

@llm
prompt: "Rewrite this summary to be more professional"
blocks: draft-summary
mode: replace
to: draft-summary
```

**After execution:**
```js
# Draft Summary // [!code highlight]
This comprehensive analysis demonstrates key insights and strategic recommendations based on our thorough research findings. // [!code highlight]

@llm
prompt: "Rewrite this summary to be more professional"
blocks: draft-summary
mode: replace
to: draft-summary
use-header: "# Draft Summary"
```

Notice how the original content "This is a rough first draft that needs improvement." was completely replaced with the new, more professional version. The heading stayed the same, but everything under it was rewritten. 

> Note: If you omit use-header in an @llm operation, Fractalic wraps the output in a default section titled "# LLM Response block". Set a custom title with use-header, or disable the wrapper with use-header: none.

**Choosing Where Results Go**
The `to:` parameter specifies where to place results. If you omit it, results appear right after the operation. For clarity and control:

## Controlling What AI Sees (Context Management)
One of Fractalic's strengths is giving you precise control over what information the AI receives. This matters for both quality (focused context produces better results) and cost (less text means lower API bills).

**Tight Control with Block Selection**
The cleanest approach is to explicitly specify which blocks to include:

```yaml
@llm
prompt: "Create a marketing strategy based on this research"
blocks: 
  - market-research/*
  - competitor-analysis
```

The AI will see only the content from those specific blocks, plus your prompt. Nothing else from your document will be included.

**Automatic Context Inclusion**
If you provide only a prompt (no blocks), Fractalic may automatically include content from earlier in your document. This can be convenient but sometimes gives the AI too much irrelevant information:

```yaml
# This block would be included automatically
This section contains information that Fractalic will
add to the AI context by default.

# This block would be included automatically too
This section also gets included automatically, providing
additional details for the AI to consider.

@llm
prompt: "Summarize the key findings"
```

**Disabling Automatic Context**
For maximum control, you can turn off automatic context inclusion:

```yaml
# Now ONLY your prompt goes to the AI
This block will not be included in the next @llm operation
because the context is set to none.

@llm
prompt: "Generate a simple greeting"
context: none
```

**Quote Headers Properly**
Since YAML treats `#` as a comment marker, always quote headers:

```yaml
use-header: "# Analysis Results"  # Correct
use-header: # Analysis Results   # Wrong - YAML ignores this
```

**Branch Selection for Holistic Understanding**
When you need the AI to understand a complete topic with all its subtopics, use wildcard selection:

```yaml
@llm
prompt: "What are the main themes across this research?"
blocks: research/*  # Includes research + all nested sections
```

This gives the AI the full picture when it needs to reason about relationships between different parts of your content.

## The Five Core Operations Explained

> **Note:** This section provides a quick overview of each operation. For complete syntax details, parameters, and advanced usage, see the [Operations Reference](./operations-reference.md).

### **`@llm`: Getting AI to Read and Write**

This operation sends content to an AI language model and inserts the response back into your document. It's the most common operation because it lets you generate text, analyze information, or get creative suggestions.

**Before execution:**
```js
# User Feedback Analysis
Users complained about slow loading times and confusing navigation.
The checkout process has too many steps.

@llm
prompt: |
  Analyze the user feedback above and identify the top 3 pain points.
  Suggest one improvement for each pain point.
blocks: user-feedback-analysis
use-header: "# Feedback Analysis"
```

**After execution:**
```js
# User Feedback Analysis
Users complained about slow loading times and confusing navigation.
The checkout process has too many steps.

@llm
prompt: |
  Analyze the user feedback above and identify the top 3 pain points.
  Suggest one improvement for each pain point.
blocks: user-feedback-analysis
use-header: "# Feedback Analysis"

# Feedback Analysis // [!code highlight]
Based on the user feedback, here are the top 3 pain points and improvements: // [!code highlight]
 // [!code highlight]
1. **Slow loading times** - Implement image compression and lazy loading // [!code highlight]
2. **Confusing navigation** - Redesign menu structure with clear categories // [!code highlight]
3. **Complex checkout** - Reduce to 2-step process with guest checkout option // [!code highlight]
```

The AI reads your feedback, analyzes it, and creates a structured response with specific recommendations.

### **`@shell`: Running Commands and Capturing Output**

This operation executes commands on your computer and adds the output to your document. Use it to run scripts, check file systems, call APIs, or integrate with any command-line tool.

**Before execution:**
```js
# Project Analysis
Let's check how many Python files we have in this project.

@shell
prompt: "find . -name '*.py' | wc -l"
use-header: "# Python File Count"
```

**After execution:**
```js
# Project Analysis
Let's check how many Python files we have in this project.

@shell
prompt: "find . -name '*.py' | wc -l"
use-header: "# Python File Count"

# Python File Count // [!code highlight]
42 // [!code highlight]
```

The shell command runs and captures its output directly in your document, creating a permanent record of what you discovered.

### **`@import`: Bringing in External Content**

This operation reads content from other files and inserts it into your current document. You can import entire files or just specific sections.

**Before execution:**
```js
# Project Setup
We need to include our standard project introduction.

@import
file: templates/project-header.md
block: introduction
```

**After execution (assuming templates/project-header.md contains the introduction section):**
```js
# Project Setup
We need to include our standard project introduction.

@import
file: templates/project-header.md
block: introduction

# Introduction // [!code highlight]
This project follows our standard development workflow with automated testing, // [!code highlight]
continuous integration, and comprehensive documentation. All team members // [!code highlight]
should familiarize themselves with the coding standards and review process // [!code highlight]
outlined in this guide. // [!code highlight]
```

This imports the "introduction" section from your template file and inserts it directly after the `@import` operation.

### **`@run`: Executing Sub-Workflows**

This operation runs another Fractalic document as a mini-workflow, passing it some context and getting back results. Think of it as calling a specialized function that lives in its own file.

**Before execution:**
```js
# Research Data
Survey responses: 850 participants, 73% satisfaction rate
Performance metrics: 2.3s average load time, 94% uptime
User behavior: 65% mobile usage, 45% return visitors

# Methodology
Data collected over 30 days using anonymous tracking and user surveys.

@run
file: agents/research-summarizer.md
blocks: 
  - research-data
  - methodology
use-header: "# Research Summary"
```

**After execution (the research-summarizer.md workflow processes the data and returns a summary):**
```js
# Research Data
Survey responses: 850 participants, 73% satisfaction rate
Performance metrics: 2.3s average load time, 94% uptime
User behavior: 65% mobile usage, 45% return visitors

# Methodology
Data collected over 30 days using anonymous tracking and user surveys.

@run
file: agents/research-summarizer.md
blocks: 
  - research-data
  - methodology
use-header: "# Research Summary"

# Research Summary // [!code highlight]
**Key Findings:** // [!code highlight]
- Strong user satisfaction (73%) with room for improvement // [!code highlight]
- Performance meets standards but load time could be optimized // [!code highlight]
- Mobile-first approach essential given 65% mobile usage // [!code highlight]
 // [!code highlight]
**Recommendations:** // [!code highlight]
- Focus on mobile experience optimization // [!code highlight]
- Investigate load time reduction opportunities // [!code highlight]
- Develop retention strategies for the 55% of new visitors // [!code highlight]
```

The sub-workflow processes your raw data and methodology, then returns a structured summary with findings and recommendations.

#### How the called workflow sees inputs
- Selected `blocks:` are combined (in the listed order) and PREPENDED to the top of the called file.
- If you also pass a `prompt:`, it is appended after those blocks as a small input section.
- The prompt gets a wrapper heading by default ("# Input Parameters"). Set a custom one with `use-header`, or remove it with `use-header: none`.
- These prepended sections become normal blocks the called workflow can reference by their kebab-case slugs (e.g., `research-data`, `input-parameters`). No other caller content is included automatically.

Caller snippet:
```yaml
@run
file: agents/research-summarizer.md
blocks:
  - research-data
prompt: "Summarize concisely for executives"
use-header: "# Run Input"
```

Callee (top of agents/research-summarizer.md) before execution:
```js
# Research Summarizer
... existing content ...
```

Callee after the call (what gets prepended):
```js
# Research Data // [!code highlight]
...copied content from caller... // [!code highlight]

# Run Input // [!code highlight]
Summarize concisely for executives // [!code highlight]

# Research Summarizer
... existing content ...
```

Default header behavior and order
- Blocks only (no prompt): the selected blocks are prepended at the very top.
- Prompt only (no blocks): a default "# Input Parameters" section is prepended with your prompt content.
- Both blocks and prompt: blocks appear first, then the default input section.

Case A — Blocks only
Caller snippet:
```yaml
@run
file: agents/research-summarizer.md
blocks:
  - research-data
```

Callee after injection (top of file):
```js
# Research Data // [!code highlight]
...copied content from caller... // [!code highlight]

# Research Summarizer
... existing content ...
```

Case B — Prompt only
Caller snippet:
```yaml
@run
file: agents/research-summarizer.md
prompt: "Summarize concisely for executives"
```

Callee after injection (top of file):
```js
# Input Parameters // [!code highlight]
Summarize concisely for executives // [!code highlight]

# Research Summarizer
... existing content ...
```

Case C — Blocks + Prompt (no custom header)
Caller snippet:
```yaml
@run
file: agents/research-summarizer.md
blocks:
  - research-data
prompt: "Summarize concisely for executives"
```

Callee after injection (top of file):
```js
# Research Data // [!code highlight]
...copied content from caller... // [!code highlight]

# Input Parameters // [!code highlight]
Summarize concisely for executives // [!code highlight]

# Research Summarizer
... existing content ...
```

Case D — Multiple blocks order
Caller snippet:
```yaml
@run
file: agents/research-summarizer.md
blocks:
  - research-data
  - methodology
```

Callee after injection (top of file):
```js
# Research Data // [!code highlight]
...copied content from caller... // [!code highlight]

# Methodology // [!code highlight]
...copied content from caller... // [!code highlight]

# Research Summarizer
... existing content ...
```

Tip: You can change or remove the wrapper for the prompt section using `use-header` (e.g., `use-header: "# Run Input"` or `use-header: none`).

See also:
- Agentic workflows and composition: [Agent Modular Workflows](./agent-modular-workflows.md)
- Full parameter list and behaviors: [Operations Reference § @run](./operations-reference.md)

### `@return`: Selecting what the caller receives
Use `@return` inside a called workflow to specify exactly what is returned to the caller. Without `@return`, the entire processed document is returned. With `@return`, only the selected content (blocks and/or a prompt) is returned.

Example A — Return a specific block (callee selects the block)
Callee snippet (agents/research-summarizer.md):
```js
# Findings
Top-level insights from analysis...

@return
block: findings
```

Caller after `@run` receives:
```js
# Findings // [!code highlight]
Top-level insights from analysis... // [!code highlight]
```

Example B — Return multiple blocks in order
Callee snippet:
```yaml
@return
blocks:
  - findings
  - recommendations
```

Caller after `@run` receives (order preserved):
```js
# Findings // [!code highlight]
... // [!code highlight]

# Recommendations // [!code highlight]
... // [!code highlight]
```

Example C — Return prompt only (default header)
Callee snippet:
```yaml
@return
prompt: "A concise executive summary"
```

Caller after `@run` receives:
```js
# Return block // [!code highlight]
A concise executive summary // [!code highlight]
```

Example D — Custom or no header for returned prompt
Callee snippet (custom header):
```yaml
@return
prompt: "Three bullet points"
use-header: "# Executive Summary"
```

Caller after `@run` receives:
```js
# Executive Summary // [!code highlight]
Three bullet points // [!code highlight]
```

Callee snippet (no header):
```yaml
@return
prompt: "Three bullet points"
use-header: none
```

Caller after `@run` receives:
```js
Three bullet points // [!code highlight]
```

Notes
- When returning blocks, original headings are preserved.
- When returning a prompt, the default wrapper heading is "# Return block"; customize with `use-header` or remove with `use-header: none`.
- Use `blocks:` to return multiple sections; their order is preserved.

See also:
- Full parameter list and behaviors: [Operations Reference § @return](./operations-reference.md)

## How Tool Output Becomes Part of Your Document
One of the key features of Fractalic is that when AI models or shell commands produce output, that output doesn't just disappear—it becomes a permanent, labeled part of your document that you can reference later.

**Before execution:**
```js
# Directory Analysis
Let's see what files we have in this project.

@shell
prompt: "ls -la"
use-header: "# Directory Contents"
```

**After execution:**
```js
# Directory Analysis
Let's see what files we have in this project.

@shell
prompt: "ls -la"
use-header: "# Directory Contents"

# Directory Contents // [!code highlight]
total 48 // [!code highlight]
drwxr-xr-x  12 user  staff   384 Sep 10 14:30 . // [!code highlight]
drwxr-xr-x   5 user  staff   160 Sep 10 14:25 .. // [!code highlight]
-rw-r--r--   1 user  staff  1234 Sep 10 14:30 README.md // [!code highlight]
-rw-r--r--   1 user  staff   567 Sep 10 14:29 package.json // [!code highlight]
drwxr-xr-x   8 user  staff   256 Sep 10 14:28 src // [!code highlight]
drwxr-xr-x   4 user  staff   128 Sep 10 14:27 docs // [!code highlight]
```

Now you can reference this output in future operations:

```js
@llm
prompt: "Analyze the file structure above and suggest improvements"
blocks: directory-contents
use-header: "# Structure Analysis"

# Structure Analysis // [!code highlight]
Based on the directory listing, this appears to be a well-organized project: // [!code highlight]
 // [!code highlight]
**Strengths:** // [!code highlight]
- Clear separation with `src/` and `docs/` directories // [!code highlight]
- Standard files like README.md and package.json are present // [!code highlight]
 // [!code highlight]
**Suggestions:** // [!code highlight]
- Consider adding a `tests/` directory for test files // [!code highlight]
- Add a `.gitignore` file to exclude build artifacts // [!code highlight]
- Include a `scripts/` directory for automation tools // [!code highlight]
```

This creates a complete audit trail of what happened, when, and what the results were. You can see the entire chain of reasoning and data in one place, which is invaluable for debugging, collaboration, and understanding how you arrived at your final results.

## Staying Safe and Managing Costs
Fractalic gives you powerful capabilities, but with power comes the need for good practices:

**Control AI Tool Usage**
When using AI with tools (like web search or code execution), set limits to prevent runaway costs:

```markdown
@llm
prompt: "Research this topic using web search, but limit your investigation"
tools: web_search
tools-turns-max: 3  # Stop after 3 tool calls
```

## Quick Success Checklist
Before running your Fractalic document, check these items:

- **Clear IDs:** Added `{id=...}` to any sections you'll reference later
- **Minimal blocks:** Selected only necessary content for each operation (`blocks:` not everything)
- **Quoted headers:** Used quotes around any `use-header` starting with `#`
- **Progression strategy:** Started with `append`, planned to switch to `replace` later
- **Tool limits:** Added `tools-turns-max` for any AI operations with tools
- **Context awareness:** Considered whether automatic context inclusion helps or hurts

## See also
- [Syntax Reference](./syntax-reference.md)
- [Operations Reference](./operations-reference.md)
- [Advanced LLM Features](./advanced-llm-features.md)
- [Context Management](./context-management.md)
