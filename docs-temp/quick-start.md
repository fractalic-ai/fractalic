---
title: Quick Start
description: Get Fractalic running and execute your first minimal workflow in under 5 minutes
outline: deep
---

# Quick Start

Welcome to Fractalic! Imagine if you could write instructions in plain English and have an AI assistant carry them out automatically, all while keeping a perfect record of what happened. That's exactly what Fractalic does—it turns simple documents into powerful AI programs.

## What You'll Learn
By the end of this guide, you'll understand how to:
- Set up Fractalic and get it running
- Write your first "AI program" using nothing but plain English
- Watch your document come alive as AI executes your instructions
- Understand why this document-based approach changes everything

## How Fractalic Works (The Big Picture)

Think of Fractalic as a smart notebook that can actually *do* things. You write what you want in a regular Markdown document, and Fractalic reads it like a recipe, executing each instruction step by step.

Here's what makes it special:

**Your Document Becomes a Living Program**  
Unlike traditional programming where you write code in one place and see results somewhere else, Fractalic adds AI responses and results directly into your document. You can literally watch your document grow as the AI works.

**Four Types of Instructions You Can Give:**
- **@llm** - "Hey AI, read this section and write me something"
- **@shell** - "Run this command on my computer and show me the output"  
- **@import** - "Bring knowledge from another document into this one"
- **@return** - "Save this result so other documents can use it"

**Everything Builds On Everything**  
Each instruction can reference the results of previous instructions. The AI can read what it just wrote, shell commands can use AI-generated file names, and everything flows together naturally.

## Quick Setup (Choose Your Path)

Getting Fractalic running is straightforward—pick the option that works best for you:

### Option 1: Docker (Easiest - Works Everywhere)
If you have Docker on your computer, this gets everything running with one command:

```bash
docker run -d --name fractalic 
  --network bridge 
  -p 3000:3000 -p 8000:8000 -p 8001:8001 -p 5859:5859 
  -v /var/run/docker.sock:/var/run/docker.sock 
  --env HOST=0.0.0.0 
  ghcr.io/fractalic-ai/fractalic:main
```

Then open your web browser to `http://localhost:3000` and you're ready to go!

### Option 2: GitHub Codespaces (Zero Installation)
Don't want to install anything? Click this button to run Fractalic in your browser:
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/fractalic-ai/fractalic)

### Option 3: Local Development
If you prefer to run everything directly on your computer:

**What you need:**
- Python 3.11 or newer
- Git (for saving your work)

**Setup steps:**
```bash
git clone https://github.com/fractalic-ai/fractalic.git
cd fractalic
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./run_server.sh
```

This starts Fractalic's engine. You can use it from the command line or add the web interface following the instructions in the README.

## Your First AI Program

Let's create a simple Fractalic document that shows you exactly how this works. Create a new file called `my-first-program.md`:

```markdown
# My First AI Assistant {id=intro}
I want to learn how Fractalic works by creating a simple greeting program.

@llm
prompt: "Create a friendly greeting that mentions Fractalic and AI programming. Make it encouraging for a beginner."
blocks: intro
use-header: "# AI Generated Greeting"
```

**Here's what's happening:**
1. **Knowledge Block**: The heading `# My First AI Assistant {id=intro}` creates a "knowledge block" that contains your description
2. **AI Instruction**: `@llm` tells Fractalic "use the AI to do something"
3. **The Task**: The `prompt` explains exactly what you want the AI to create
4. **The Context**: `blocks: intro` tells the AI to read the intro section first
5. **The Output**: `use-header` creates a new section with the AI's response

When you run this document, Fractalic will add the AI's greeting right into your file!

## A More Practical Example

Here's a more interesting example that shows how Fractalic instructions can build on each other:

```markdown
# Website Ideas {id=ideas}
I want to create a personal portfolio website to showcase my projects.

@llm
prompt: |
  Based on the website goal above, create a simple list of 5 essential pages 
  this website should have. Format as a bulleted list.
blocks: ideas
use-header: "# Recommended Pages"

@llm
prompt: |
  For each page listed above, write a one-sentence description of what 
  content should go there.
blocks: recommended-pages
use-header: "# Page Descriptions"

@shell
prompt: "mkdir -p my-portfolio && echo 'Portfolio structure created'"
use-header: "# Setup Confirmation"
```

**Notice the flow:**
1. **You describe your goal** (portfolio website)
2. **First AI call** reads your goal and creates a list of pages
3. **Second AI call** reads the list from step 2 and describes each page
4. **Shell command** creates a folder for your project
5. **Each step builds on the previous ones** - this is the power of Fractalic!

The AI in step 2 doesn't just use your original idea—it actually reads the page list that the first AI created. This is how you can chain together complex workflows using simple English instructions.

## Watching Your Document Come Alive

After running either example, you'll see something magical happen—your document will have grown! The content that was generated gets added right into your file. Here's exactly what you'll see:

### First Example Results
```js
# My First AI Assistant {id=intro}
I want to learn how Fractalic works by creating a simple greeting program.

@llm
prompt: "Create a friendly greeting that mentions Fractalic and AI programming. Make it encouraging for a beginner."
blocks: intro
use-header: "# AI Generated Greeting"

# AI Generated Greeting {id=ai-generated-greeting} // [!code highlight]
Welcome to the exciting world of Fractalic! You're taking your first steps into AI programming, and that's fantastic. Fractalic makes it easy to harness the power of artificial intelligence using simple, readable instructions—you're going to love how intuitive and powerful this approach can be! // [!code highlight]
```

### Portfolio Example Results
```js
# Website Ideas {id=ideas}
I want to create a personal portfolio website to showcase my projects.

@llm
prompt: |
  Based on the website goal above, create a simple list of 5 essential pages 
  this website should have. Format as a bulleted list.
blocks: ideas
use-header: "# Recommended Pages"

# Recommended Pages {id=recommended-pages} // [!code highlight]
• Home - Landing page with brief introduction // [!code highlight]
• About - Personal background and skills // [!code highlight]
• Projects - Showcase of your work and achievements // [!code highlight]
• Blog - Thoughts, tutorials, and updates // [!code highlight]
• Contact - Ways to get in touch // [!code highlight]

@llm
prompt: |
  For each page listed above, write a one-sentence description of what 
  content should go there.
blocks: recommended-pages
use-header: "# Page Descriptions"

# Page Descriptions {id=page-descriptions} // [!code highlight]
• **Home**: A compelling landing page that immediately communicates who you are and what you do, with clear navigation to your best work. // [!code highlight]
• **About**: Your professional story, skills, experience, and what makes you unique as a developer or creator. // [!code highlight]
• **Projects**: Detailed case studies of your best work with screenshots, technologies used, and links to live demos or code repositories. // [!code highlight]
• **Blog**: Industry insights, tutorials, lessons learned, and updates about your current projects to demonstrate your expertise and thought leadership. // [!code highlight]
• **Contact**: Multiple ways for potential employers or collaborators to reach you, including email, social media, and possibly a contact form. // [!code highlight]

@shell
prompt: "mkdir -p my-portfolio && echo 'Portfolio structure created'"
use-header: "# Setup Confirmation"

# Setup Confirmation {id=setup-confirmation} // [!code highlight]
Portfolio structure created // [!code highlight]
```

### The Magic You're Seeing

The highlighted green lines show everything that Fractalic added to your document:
- **AI responses** - exactly what the language model created based on your instructions
- **Command outputs** - results from shell commands (like creating directories)
- **Automatic organization** - everything gets proper headings and IDs without you having to think about it
- **Context chaining** - notice how the second AI instruction automatically used the first AI's output

**This is the key insight**: Your document becomes a living record of both your ideas and what actually happened. You can see the whole process unfold, edit any part, and run it again. Fractalic treats your document like a program that grows and evolves as it runs.

**What makes this different from traditional programming:**
- **No separate files** - your instructions and results live together
- **No complex syntax** - just plain English instructions
- **Visible process** - you can see exactly how the AI reasoned through each step  
- **Iterative refinement** - change any instruction and re-run to see different results

## Using Tools and MCP (Agentic Loop)
Tools let the AI look things up or perform actions. Many tools are provided by MCP servers. When you add `tools:` to an `@llm` operation, you enable an agentic loop: the AI can call those tools across one or more turns (bounded by `tools-turns-max`). Tool outputs are auto-inserted into your document; you never paste them manually.

Tool Example — Web Search with Multiple Tools
**Before execution:**
```js
# Topic {id=topic}
Research Rust async ecosystem highlights.

@llm
prompt: |
  Find 3 recent highlights about the Rust async ecosystem.
  Cite sources inline. Keep it concise.
blocks: topic
tools:
  - web_search
tools-turns-max: 2
use-header: "# Research Highlights"
```

**After execution:**
```js
# Topic {id=topic}
Research Rust async ecosystem highlights.

@llm
prompt: |
  Find 3 recent highlights about the Rust async ecosystem.
  Cite sources inline. Keep it concise.
blocks: topic
tools:
  - web_search
tools-turns-max: 2
use-header: "# Research Highlights"

> TOOL CALL, id: call_web_search_1 // [!code highlight]
tool: web_search // [!code highlight]
args: // [!code highlight]
{ // [!code highlight]
  "query": "Rust async ecosystem highlights", // [!code highlight]
  "limit": 5 // [!code highlight]
} // [!code highlight]

> TOOL RESPONSE, id: call_web_search_1 // [!code highlight]
response: // [!code highlight]
{ // [!code highlight]
  "results": [ // [!code highlight]
    { // [!code highlight]
      "title": "Tokio 1.x Performance Improvements", // [!code highlight]
      "url": "https://tokio.rs/blog/tokio-1-performance", // [!code highlight]
      "content": "Major I/O driver optimizations..." // [!code highlight]
    }, // [!code highlight]
    { // [!code highlight]
      "title": "Async-std Ecosystem Updates", // [!code highlight]
      "url": "https://async.rs/blog/ecosystem-parity", // [!code highlight]
      "content": "Key crate compatibility achieved..." // [!code highlight]
    } // [!code highlight]
  ] // [!code highlight]
} // [!code highlight]

# Research Highlights {id=research-highlights} // [!code highlight]
- Tokio 1.x introduces improved I/O driver performance (source: https://tokio.rs/blog/tokio-1-performance) // [!code highlight]
- Async-std gains ecosystem parity in key crates (source: https://async.rs/blog/ecosystem-parity) // [!code highlight]
- New tracing tools simplify async debugging across the ecosystem // [!code highlight]
```




Notes
- Each surviving tool block is resent next AI turn (token cost grows if you keep long raw outputs).
- Set `tools-turns-max` to bound the loop; summarize early and replace large raw outputs once stable.
- Tool outputs are auto-inserted; you never merge them manually.
- Tool logs show the tool invoked and the arguments used; generated sections appear below. See [Agent Modular Workflows](./agent-modular-workflows.md) for details.

See also: [Agent Modular Workflows](./agent-modular-workflows.md)

## Why This Changes Everything

Most AI tools make you start from scratch every time. With Fractalic:

**Your Work Accumulates**  
Each instruction builds on previous results. The AI can read its own previous responses, refer to command outputs, and continuously refine its understanding.

**Everything Stays Organized**  
No more lost conversations or scattered results. Your entire thought process—from initial idea to final output—lives in one readable document.

**You Stay in Control**  
Unlike black-box AI tools, you can see exactly what the AI is thinking about at each step. Edit any instruction, change the context, or adjust the flow anytime.

**Collaboration Becomes Natural**  
Share your Fractalic document with teammates and they can see your entire reasoning process, modify it, and extend it—all in plain English.

**Real Programs, Real Power**  
Despite the simple syntax, you can build sophisticated workflows: data analysis pipelines, content generation systems, research workflows, or even complex multi-step automation.

The paradigm shift is this: **instead of using AI as a chatbot, you're programming with AI as your execution engine.**

## Troubleshooting Your First Run

Don't worry if something doesn't work perfectly on your first try—here are the most common issues and how to fix them:

| What you see | Why it happens | How to fix |
|--------------|----------------|------------|
| "Missing API key" | Fractalic needs credentials to talk to AI services | Add your OpenAI, Anthropic, or other API key to `settings.toml` |
| "No settings file found" | First-time setup isn't complete | Run the web UI once or copy `settings.toml.sample` to `settings.toml` |
| "Port already in use" | Another program is using the same port | Change ports in config or stop other programs |
| AI gives unexpected responses | The instruction might be unclear or missing context | Be more specific in your prompts or check which blocks you're referencing |
| "Block not found" | Typo in a block reference | Check that your `{id=...}` matches what you're referencing in `blocks:` |

**Pro tip**: When you're just starting out, run each example exactly as written first. Once you see it working, then start experimenting with changes!

## Your Journey Starts Here

Now that you understand the basics, here's how to dive deeper:

### 1. Try the Examples First
Create and run the sample files above exactly as written. Seeing Fractalic in action will make everything click.

### 2. Experiment with Small Changes
Once the examples work, try modifying them:
- Change the prompts to ask for different things
- Reference different sections in your `blocks:` 
- Add new `@llm` instructions that build on the existing results

### 3. Learn the Core Concepts
Read [Core Concepts](./core-concepts.md) to understand how Fractalic thinks about documents, blocks, and context. This will help you build more sophisticated workflows.

### 4. Explore All the Operations
Check out [Operations Reference](./operations-reference.md) to see everything Fractalic can do beyond `@llm` and `@shell`.

### 5. Build Something Real
Start with a task you actually want to accomplish—maybe automating something at work, organizing research, or creating content. Fractalic is designed to handle real-world problems.

**Remember**: Fractalic grows with you. Start simple with natural language instructions, and gradually add sophistication as you learn. The best part? Everything stays readable and modifiable—no mysterious code to maintain, just clear English instructions