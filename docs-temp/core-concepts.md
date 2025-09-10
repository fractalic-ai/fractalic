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
# Goal {id=goal}
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
# Research {id=research}
## Market Analysis {id=market}
## User Studies {id=users}
```

You can reference the market analysis section as `research/market`. This path structure helps you organize complex documents with many sections.

**Wildcards for Whole Branches:** Sometimes you want to reference a section and everything inside it. Use the `/*` syntax:
```markdown
blocks: research/*
```

This selects the "research" heading plus both "market" and "users" subsections. It's useful when you want an AI to consider all related information together.

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

**Step 3: Executing the Operation**
Fractalic runs the operation—sending text to an AI model, executing a shell command, importing a file, or running another workflow.

**Step 4: Adding Results Back to Your Document**
The operation's output gets inserted back into your document under a clear heading. You can control where this happens and what the heading says.

Here's a simple example showing the complete flow:

```js
# Website Ideas {id=ideas}
I want to create a portfolio website to showcase my design work.

@llm
prompt: "Based on the website goal above, suggest 5 essential pages"
blocks: ideas
use-header: "# Suggested Pages"

# Suggested Pages {id=suggested-pages} // [!code highlight]
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

```js
# Research Notes {id=notes}
Initial findings about user behavior.

@llm
prompt: "Add insights about mobile usage patterns"
blocks: notes
mode: append
to: notes

# Research Notes {id=notes}
Initial findings about user behavior.

Additional insights: Mobile users prefer... // [!code highlight]
```

**Replace Mode**
This completely replaces the target content with new content. Use this when you want to refine or update something rather than add to it:

**Before execution:**
```js
# Draft Summary {id=summary}
This is a rough first draft that needs improvement.

@llm
prompt: "Rewrite this summary to be more professional"
blocks: summary
mode: replace
to: summary
```

**After execution:**
```js
# Draft Summary {id=summary} // [!code highlight]
This comprehensive analysis demonstrates key insights and strategic recommendations based on our thorough research findings. // [!code highlight]

@llm
prompt: "Rewrite this summary to be more professional"
blocks: summary
mode: replace
to: summary
```

Notice how the original content "This is a rough first draft that needs improvement." was completely replaced with the new, more professional version. The heading stayed the same, but everything under it was rewritten.

**Choosing Where Results Go**
The `to:` parameter specifies where to place results. If you omit it, results appear right after the operation. For clarity and control:

```yaml
@llm
prompt: "Generate conclusion"
blocks: analysis/*
mode: append
to: final-report
```

## Controlling What AI Sees (Context Management)
One of Fractalic's strengths is giving you precise control over what information the AI receives. This matters for both quality (focused context produces better results) and cost (less text means lower API bills).

**Tight Control with Block Selection**
The cleanest approach is to explicitly specify which blocks to include:

```markdown
@llm
prompt: "Create a marketing strategy based on this research"
blocks: 
  - market-research/*
  - competitor-analysis
```

The AI will see only the content from those specific blocks, plus your prompt. Nothing else from your document will be included.

**Automatic Context Inclusion**
If you provide only a prompt (no blocks), Fractalic may automatically include content from earlier in your document. This can be convenient but sometimes gives the AI too much irrelevant information:

```markdown
@llm
prompt: "Summarize the key findings"
# This might include ALL previous content automatically
```

**Disabling Automatic Context**
For maximum control, you can turn off automatic context inclusion:

```markdown
@llm
prompt: "Generate a simple greeting"
context: none
# Now ONLY your prompt goes to the AI
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

**`@llm`: Getting AI to Read and Write**

This operation sends content to an AI language model and inserts the response back into your document. It's the most common operation because it lets you generate text, analyze information, or get creative suggestions.

Example usage:
```yaml
@llm
prompt: |
  Analyze the user feedback above and identify the top 3 pain points.
  Suggest one improvement for each pain point.
blocks: user-feedback/*
use-header: "# Feedback Analysis"
```

The AI will read all the user feedback sections, process your instructions, and create a new section with its analysis.

**`@shell`: Running Commands and Capturing Output**

This operation executes commands on your computer and adds the output to your document. Use it to run scripts, check file systems, call APIs, or integrate with any command-line tool.

Example usage:
```yaml
@shell
prompt: "find . -name '*.py' | wc -l"
use-header: "# Python File Count"
```

This counts Python files in your project and adds the result to your document.

**`@import`: Bringing in External Content**

This operation reads content from other files and inserts it into your current document. You can import entire files or just specific sections.

Example usage:
```yaml
@import
file: templates/project-header.md
blocks: introduction
use-header: "# Project Overview"
```

This finds the "introduction" section in your template file and brings it into your current document.

**`@run`: Executing Sub-Workflows**

This operation runs another Fractalic document as a mini-workflow, passing it some context and getting back results. Think of it as calling a specialized function that lives in its own file.

Example usage:
```yaml
@run
file: agents/research-summarizer.md
blocks: 
  - raw-data/*
  - methodology
use-header: "# Research Summary"
```

This sends your raw data and methodology to a specialized summarizer workflow and includes its output in your document.

**`@return`: Ending a Workflow with Specific Output**

This operation stops the current workflow and returns specific content. It's essential in sub-workflows (called by @run) to specify what gets sent back to the caller.

Example usage:
```yaml
@return
blocks: final-summary
use-header: "# Results"
```

This ends the workflow and returns only the "final-summary" block to whoever called this workflow.

## How Tool Output Becomes Part of Your Document
One of the key features of Fractalic is that when AI models or shell commands produce output, that output doesn't just disappear—it becomes a permanent, labeled part of your document that you can reference later.

For example, when you run this operation:
```yaml
@shell
prompt: "ls -la"
use-header: "# Directory Contents"
```

Fractalic doesn't just show you the directory listing temporarily. Instead, it creates a new section in your document called "Directory Contents" containing the output. Later, you can reference this section in other operations:

```yaml
@llm
prompt: "Analyze the file structure above and suggest improvements"
blocks: directory-contents
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

**Keep Context Focused**
Sending too much text to AI models wastes money and can reduce response quality. Be selective about which blocks you include:

```markdown
# Good: Focused context
@llm
blocks: user-requirements
prompt: "Create a solution for these specific requirements"

# Avoid: Everything in the document
@llm
blocks: "*"  # This could be expensive and unfocused
```

**Use Replace Mode for Stable Content**
Once content stops changing, use replace mode to keep your document clean and your context focused:

```markdown
@llm
prompt: "Polish this final draft"
blocks: rough-draft
mode: replace
to: rough-draft
```

**Disable Headers for Structured Output**
When you need clean JSON or other structured data, turn off the automatic heading:

```markdown
@llm
prompt: "Return ONLY a JSON object with keys: title, summary, tags"
use-header: none
```

## When NOT to Use Certain Features
Understanding limitations helps you avoid common pitfalls:

**Don't Overuse Wildcards**
While `research/*` is powerful for comprehensive analysis, avoid it when you only need specific pieces:

```markdown
# Good: Specific selection
blocks: research/conclusion

# Avoid when unnecessary: Everything under research
blocks: research/*  # Could include irrelevant subsections
```

**Don't Append Forever**
Repeatedly appending small changes creates clutter and wastes tokens. Switch to replace mode once content stabilizes:

```markdown
# Early exploration: append is fine
mode: append

# Later refinement: use replace
mode: replace
to: target-section
```

**Don't Rely on Implicit Context for Critical Operations**
When precision matters, explicitly control what the AI sees:

```markdown
# Precise control
@llm
prompt: "Generate only a title"
context: none  # No automatic context
# Only your prompt goes to the AI
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
