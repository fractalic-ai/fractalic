---
title: Quick Start
description: Get Fractalic running and execute your first minimal workflow in under 5 minutes
outline: deep
---

# Quick Start

Welcome to Fractalic! Think of it as a way to write computer programs using plain English in simple documents. Instead of learning complex programming languages, you'll write what you want in regular Markdown files, and Fractalic will make it happen.

## What You'll Learn
By the end of this guide, you'll understand how to:
- Set up Fractalic and get it running
- Write your first "AI program" in plain English
- See how the computer executes your instructions step by step
- Understand why this approach is powerful for working with AI

## How Fractalic Works (The Big Picture)

Imagine writing a recipe, but instead of just instructions for cooking, you're writing instructions for an AI assistant. Each step in your "recipe" can:

- **Ask an AI** to analyze, write, or solve something
- **Run computer commands** to check results or get data  
- **Import knowledge** from other documents or files
- **Return specific results** to share with others

The magic happens because Fractalic treats your document like a living workspace. As each instruction runs, it adds new content to your document. You can see the AI's responses, the output from commands, and how everything builds up step by step.

## Quick Setup (Choose Your Path)

### Option 1: Docker (Easiest - Recommended)
If you have Docker installed, this gets everything running in one command:

```bash
docker run -d --name fractalic \
  --network bridge \
  -p 3000:3000 -p 8000:8000 -p 8001:8001 -p 5859:5859 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --env HOST=0.0.0.0 \
  ghcr.io/fractalic-ai/fractalic:main
```

Then open your web browser to `http://localhost:3000`

### Option 2: GitHub Codespaces (Zero Installation)
Click this button to run Fractalic in your browser with zero setup:
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/fractalic-ai/fractalic)

### Option 3: Local Development
If you want to run Fractalic directly on your computer:

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

This starts Fractalic's backend. You can use it directly from the command line, or add the web interface by following the UI setup in the README.

## Your First AI Program

Let's create a simple Fractalic document that demonstrates the core concepts. Create a new file called `my-first-program.md`:

```markdown
# My First AI Assistant {id=intro}
I want to learn how Fractalic works by creating a simple greeting program.

@llm
prompt: "Create a friendly greeting that mentions Fractalic and AI programming. Make it encouraging for a beginner."
block: intro
use-header: "# AI Generated Greeting"
```

**What this does:**
1. Creates a knowledge block called "intro" with your description
2. Tells the AI (`@llm`) to read that block and create a greeting
3. Puts the AI's response in a new section called "AI Generated Greeting"

When you run this, Fractalic will add the AI's response right into your document!

## A More Practical Example

Here's a slightly more complex example that shows how Fractalic can help you with real tasks:

```markdown
# Website Ideas {id=ideas}
I want to create a personal portfolio website to showcase my projects.

@llm
prompt: |
  Based on the website goal above, create a simple list of 5 essential pages 
  this website should have. Format as a bulleted list.
block: ideas
use-header: "# Recommended Pages"

@llm
prompt: |
  For each page listed above, write a one-sentence description of what 
  content should go there.
block: recommended-pages/*
use-header: "# Page Descriptions"

@shell
prompt: "mkdir -p my-portfolio && echo 'Portfolio structure created'"
use-header: "# Setup Confirmation"
```

**What happens here:**
1. You describe your goal (portfolio website)
2. First AI call creates a list of essential pages
3. Second AI call describes what goes on each page (notice it references the previous AI's output)
4. Shell command creates a folder for your project
5. Each step builds on the previous ones

## Understanding Your Results

After running either example, you'll see your original document has grown! Here's exactly what you'll see (highlighted lines show what Fractalic added):

### First Example Results
```yaml{9-10}
# My First AI Assistant {id=intro}
I want to learn how Fractalic works by creating a simple greeting program.

@llm
prompt: "Create a friendly greeting that mentions Fractalic and AI programming. Make it encouraging for a beginner."
block: intro
use-header: "# AI Generated Greeting"

# AI Generated Greeting {id=ai-generated-greeting}
Welcome to the exciting world of Fractalic! You're taking your first steps into AI programming, and that's fantastic. Fractalic makes it easy to harness the power of artificial intelligence using simple, readable instructions—you're going to love how intuitive and powerful this approach can be!
```

### Portfolio Example Results
```yaml{11-16,25-30,36-37}
# Website Ideas {id=ideas}
I want to create a personal portfolio website to showcase my projects.

@llm
prompt: |
  Based on the website goal above, create a simple list of 5 essential pages 
  this website should have. Format as a bulleted list.
block: ideas
use-header: "# Recommended Pages"

# Recommended Pages {id=recommended-pages}
• Home - Landing page with brief introduction
• About - Personal background and skills
• Projects - Showcase of your work and achievements
• Blog - Thoughts, tutorials, and updates
• Contact - Ways to get in touch

@llm
prompt: |
  For each page listed above, write a one-sentence description of what 
  content should go there.
block: recommended-pages/*
use-header: "# Page Descriptions"

# Page Descriptions {id=page-descriptions}
• **Home**: A compelling landing page that immediately communicates who you are and what you do, with clear navigation to your best work.
• **About**: Your professional story, skills, experience, and what makes you unique as a developer or creator.
• **Projects**: Detailed case studies of your best work with screenshots, technologies used, and links to live demos or code repositories.
• **Blog**: Industry insights, tutorials, lessons learned, and updates about your current projects to demonstrate your expertise and thought leadership.
• **Contact**: Multiple ways for potential employers or collaborators to reach you, including email, social media, and possibly a contact form.

@shell
prompt: "mkdir -p my-portfolio && echo 'Portfolio structure created'"
use-header: "# Setup Confirmation"

# Setup Confirmation {id=setup-confirmation}
Portfolio structure created
```

### What You're Seeing

The highlighted sections show:
- **AI responses** - exactly what the language model generated based on your prompts
- **Command outputs** - results from shell commands (like creating directories)
- **Structured organization** - everything gets proper headings and IDs automatically
- **Context building** - notice how the second AI call referenced the first AI's output

This is the key insight: your document becomes a living record of both your intentions and the AI's work. You can see the whole process, modify it, and run it again. Each time you execute the document, Fractalic will either update existing sections or add new ones, depending on how you configure it.

## Why This Approach Is Powerful

Traditional programming requires learning syntax, managing complex tools, and thinking like a computer. Fractalic lets you:

- **Think in natural language** - write what you want, not how to code it
- **See the process** - watch your document grow as AI works
- **Build incrementally** - start simple, add complexity step by step  
- **Reuse and share** - your documents become reusable "AI programs"
- **Mix AI and automation** - combine AI reasoning with practical commands

## Common First-Time Issues

| What you see | Why it happens | How to fix |
|--------------|----------------|------------|
| "Missing API key" | Fractalic needs credentials to talk to AI services | Add your OpenAI, Anthropic, or other API key to `settings.toml` |
| "No settings file found" | First-time setup not complete | Run the UI once or copy `settings.toml.sample` to `settings.toml` |
| "Port already in use" | Another program using the same port | Change ports in config or stop other programs |
| AI gives weird responses | Context or prompt needs refinement | Be more specific in your prompts or adjust which blocks you reference |

## What's Next?

Now that you've seen the basics:

1. **Try the examples** - Create and run the sample files above
2. **Experiment** - Modify the prompts to see different results  
3. **Learn the concepts** - Read [Core Concepts](./core-concepts.md) to understand how Fractalic works internally
4. **Explore operations** - Check out [Operations Reference](./operations-reference.md) for all available commands
5. **Build something real** - Start with a simple task you actually want to accomplish

Remember: Fractalic is designed to grow with you. Start with simple instructions, and gradually add sophistication as you learn. The beauty is that everything stays readable and modifiable - no complex code to maintain!
