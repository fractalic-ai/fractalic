# Fractalic
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deploy Now](https://img.shields.io/badge/🚀%20Deploy%20Now-One%20Click-brightgreen?style=for-the-badge)](https://fractalic-ai.github.io/fractalic/deploy/)

Program AI in plain language (any language). That's it.

![alt text](<docs/images/fractalic_hero.png>)

## Vision 🚀

Modern AI workflows shouldn’t be harder than they already are. We have powerful LLMs, yet we still end up wrestling with Python scripts and tangled spaghetti-like node-based editors. **Fractalic** aims to fix this by letting you build AI systems as naturally as writing a simple doc.

## What is Fractalic? ✨

Fractalic combines Markdown and YAML to create agentic AI systems using straightforward, human-readable documents. It lets you grow context step by step, control AI knowledge precisely, and orchestrate complex workflows through simple document structure and syntax.

## 🚀 One-Click Cloud Deploy

Deploy Fractalic instantly using our pre-built Docker container `ghcr.io/fractalic-ai/fractalic:main`:

### 🐳 Docker (Local/VPS - Recommended)
```bash
# Pull and run the complete Python + React application with Docker access
docker run -d --name fractalic \
  --network bridge \
  -p 3000:3000 -p 8000:8000 -p 8001:8001 -p 5859:5859 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --env HOST=0.0.0.0 \
  ghcr.io/fractalic-ai/fractalic:main
```

*🎯 **2-3 minutes** → Complete full-stack application with Docker deployment capabilities*

### ☁️ Koyeb (True One-Click Cloud)
[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=docker&image=ghcr.io/fractalic-ai/fractalic:main&name=fractalic&ports=3000;http;/&ports=8000;http;/api&ports=8001;http;/ai&ports=5859;http;/mcp)

*🎯 **3-4 minutes** → Serverless cloud deployment with auto-scaling*

### 🚀 Zeabur (One-Click Template)
> **Note**: Requires creating a Zeabur template first. Visit [Zeabur Dashboard](https://zeabur.com) to create a template with the Docker image `ghcr.io/fractalic-ai/fractalic:main`.

*🎯 **2-3 minutes** → Fast cloud deployment with global CDN*

**Access after deployment:**
- 🖥️ **Frontend UI**: http://localhost:3000
- ⚙️ **Backend API**: http://localhost:8000  
- 🤖 **AI Server**: http://localhost:8001

### 🆕 GitHub Codespaces (Development Environment)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/fractalic-ai/fractalic)

*🎯 **3-4 minutes** → Full development environment in the browser*  
*Perfect for testing, development, and learning Fractalic*

### 📋 What You Get After Deployment:
- 🖥️ **Frontend UI**: http://localhost:3000 (Interactive notebook-style interface)
- ⚙️ **Backend API**: http://localhost:8000 (REST API for integration)  
- 🤖 **AI Server**: http://localhost:8001 (Fractalic script execution engine)
- 🔧 **MCP Manager**: http://localhost:5859 (Model Context Protocol server management)

### 🎯 Choose Your Deployment:
| Method | Best For | Time | Requirements |
|--------|----------|------|--------------|
| 🐳 **Docker** | Local/VPS, production ready | 2-3 min | Docker Desktop |
| ☁️ **Koyeb** | Serverless cloud, auto-scaling | 3-4 min | Koyeb account |
| 🚀 **Zeabur** | Fast cloud deployment | 2-3 min | Zeabur account + template |
| ☁️ **Codespaces** | No-install testing, development | 3-4 min | GitHub account |
| 🏗️ **Manual Build** | Custom setup, advanced users | 5-10 min | Python + Node.js |

> **Note**: Fractalic is a full-stack Python + React application. Platforms like Vercel and Netlify only support static frontends and cannot run the Python backend services required for Fractalic to function properly.

## Tutorial 101 
Watch this video with quick project overview and 101 concepts how to use it (picture is clickable YouTube [link](https://www.youtube.com/watch?v=iRqIzmKE8uw))

[![Watch the video](https://img.youtube.com/vi/iRqIzmKE8uw/0.jpg)](https://www.youtube.com/watch?v=iRqIzmKE8uw)


## Key Features

- 🖥 **Multi-Provider Support: Anthropic, Groq, OpenAI and compatible providers**  
- 🔄 **Integrated Workflow: Collaborate with different models and adjust settings in one workflow**  
- 📁 **Structured Markdown Navigation: Markdown document structure accessible as a tree**  
- ⚙ **Dynamic Knowledge Context: Knowledge context can be changed and updated at runtime**  
- 🔧 **Runtime Flow & Logic Control: Flow and logic control instructions can be generated in runtime**  
- 💻 **Shell Access: Easy shell access with results accessible in context at runtime**  
- 🔑 **External API Token Access: External API tokens are accessible in @shell**  
- 📂 **Git-Backed Sessions: Each session’s results are saved to git**  
- 📝 **Notebook-like UI: Notebook-like UI**  
- 🔍 **Diff-based Session Results: Session results as diffs in UI**  
- 📖 **Integrated Markdown Viewer: Markdown viewer in UI**

## Upcoming Roadmap
- [ ] **LiteLLM Support:** Add any model available on the market.
- [ ] **Context State Player:** Trace and replay each context change.
- [ ] **Session Snapshot Migration:** Migrate from Git to zip format – remove Git library dependencies to resolve Windows 11 issues and provide complete state tracking.
- [ ] **System Context Direct Setting:** Enable direct system context adjustments at runtime.
- [ ] **And Many More...**

# Installation

## 🐳 Quick Start with Docker (Recommended)

### Option 1: Pre-built Container (Fastest)
```bash
# Pull and run from GitHub Container Registry with Docker access
docker run -d --name fractalic \
  --network bridge \
  -p 3000:3000 -p 8000:8000 -p 8001:8001 -p 5859:5859 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --env HOST=0.0.0.0 \
  ghcr.io/fractalic-ai/fractalic:main
```

### Option 2: Build Locally (Development)
```bash
# Clone and build locally
curl -s https://raw.githubusercontent.com/fractalic-ai/fractalic/main/docker_build_run.sh | bash
```

### 🔧 With Your Own Configuration
```bash
# Create config directory
mkdir -p fractalic-config

# Add your config files (copy from .sample files)
cp settings.toml.sample fractalic-config/settings.toml
cp mcp_servers.json.sample fractalic-config/mcp_servers.json

# Edit with your API keys and settings
nano fractalic-config/settings.toml

# Run with your config and Docker access
docker run -d --name fractalic \
  --network bridge \
  -p 3000:3000 -p 8000:8000 -p 8001:8001 -p 5859:5859 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --env HOST=0.0.0.0 \
  -v $(pwd)/fractalic-config:/app \
  ghcr.io/fractalic-ai/fractalic:main
```

### 🌍 With Environment Variables
```bash
# Run with API keys as environment variables and Docker access
docker run -d --name fractalic \
  --network bridge \
  -p 3000:3000 -p 8000:8000 -p 8001:8001 -p 5859:5859 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --env HOST=0.0.0.0 \
  -e OPENAI_API_KEY=sk-your-openai-key \
  -e ANTHROPIC_API_KEY=sk-ant-your-anthropic-key \
  ghcr.io/fractalic-ai/fractalic:main
```

**Access the application:**
- 🎨 **Frontend UI**: http://localhost:3000  
- 🔧 **Backend API**: http://localhost:8000
- 🤖 **AI Server**: http://localhost:8001

---

## Requirements for Local Development
Important: please ensure you have Git installed on your system.
Git dependecy would be removed in future releases. Sessions would be stored on .zip or .tar.gz files.

## Project structure
Fractalic is split into two repositories: `fractalic` and [`fractalic-ui`](https://github.com/fractalic-ai/fractalic-ui). Both repositories need to be installed in the same directory to run the application.

The backend will be launched on port 8000 and the frontend on port 3000.

Currently, the recommended way to use Fractalic is to install both the interpreter with backend server and UI frontend, and run it as a Docker container. If you don't need Docker, you can skip that step and follow the local installation instructions.


# Installation (Local)

1. Backend installation
```bash
git clone https://github.com/fractalic-ai/fractalic.git && \
cd fractalic && \
python3 -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt 
```
2. Run backend
```bash
./run_server.sh
```

3. Frontend installation
```bash
git clone https://github.com/fractalic-ai/fractalic-ui.git  && \
cd fractalic-ui && \
npm install 
```

4. Run frontend
```bash
npm run dev
```

## Running fractalic backend server
Required for UI to work. Please run the following command in the terminal.
```bash
./run_server.sh
```

## Settings
First time you run the UI, settings.toml would be created required for parser (at least while working from UI, if you are using it headless from CLI - you can use script CLI params). You should select default provider and enter env keys for external providers (repicate, tavily and etc).

## Tutorials and Examples (work in progress)

Please check the [`tutorials`](./tutorials) folder for examples. More tutorials and videos will be added soon.

## 01_Basics
- [**Hello World**](./tutorials/hello-world) - Simple introduction demonstrating basic Fractalic operations
- [**Markdown Tree**](./tutorials/01_Basics/markdown-tree) - Demonstrates hierarchical document structure manipulation and block targeting
- [**Markdown Preview**](./tutorials/01_Basics/markdown-preview) - Shows markdown rendering capabilities including tables and Mermaid diagrams
- [**Mermaid Simple**](./tutorials/01_Basics/mermaid-simple) - Focuses on Mermaid diagram rendering with examples of different diagram types
- [**Multimodel Workflow**](./tutorials/01_Basics/multimodel-workflow) - Demonstrates using multiple LLM providers (Claude, OpenAI, Groq) in a single workflow
- [**PDF Summary**](./tutorials/01_Basics/pdf-summary) - Shows how to process PDF documents with the `media` parameter

## 02_Integrations
- [**Yahoo Finance & Tavily**](./tutorials/02_tutorial_yahoofinance_tavily_stocks_news_analytics) - Integrates Yahoo Finance API for stock data and Tavily search for news analysis. **Note:** Requires Tavily API key in settings.toml

# Quick 101
When a Markdown file is executed (either directly or called as an agent/module), the interpreter creates a context tree in memory. This tree consists of two types of blocks:

1. Knowledge blocks: These correspond to Markdown header sections AND their associated content (all text/content under that header until the next header or operation). 
2. Operation blocks: These are YAML instructions starting with a custom `@operation` name. They receive inputs, prompts, and block references as parameters.

Block identifiers can be:
- Explicitly defined in headers using `{id=block-name}`
- Automatically generated by converting header text to kebab-case (lowercase, no spaces or special characters)

Operations can modify the context, return results, and generate new operations within the current execution context. The images referenced show a comparison between an original Markdown file and its resulting context (`.ctx`) file.

### Example 1
![alt text](<docs/images/slide_01.png>)

In this example, we define text blocks and run LLM generations sequentially. Key points:

1. First operation combines all previous blocks with the prompt (default behavior) using global settings (`Claude-3.5-Sonnet`)
2. Results are stored in context under "# LLM Response block" (ID: `llm-response-block`). Headers and nesting levels can be customized via `use-header` parameter
3. Second LLM call (`DeepSeek R-1` via OpenAI-like API) can be restricted to original task block only, preventing access to previous results
4. The `blocks` parameter provides access to all context blocks, including those appearing later in the file
5. The `/*` syntax is used to select entire branch of blocks - both parent (`#`) and child (`##`) blocks at once
6. Temperature can be adjusted per operation (global default is 0.0, recommended for prototyping workflows and processes)

### Example 2
![alt text](<docs/images/slide_02.png>)

As shown in the example above, the `@shell` operation provides LLM access to the external environment through the operating system's command interpreter. This enables universal integration capabilities - any tool, API, or framework accessible via command line becomes available to LLM, from simple curl requests to complex Python scripts and Docker containers.

Key features:
- Environment variables and API keys can be configured in `settings.toml` for each `@shell` session
- Full stdin output available in the execution context
- YAML syntax support for multiline commands (using `|`, `>` and other operators)
- The `use-header` parameter allows customizing block headers, identifiers, and nesting levels for flexible tree structure organization. Please be aware that YAML recognizes '#' as a comment, so it should be escaped with quotes or backslashes if used in the field itself.

### Example 3
![alt text](<docs/images/slide_03.png>)

The system enables modular, reusable workflows and agents that execute in isolated contexts with parameter passing and result return capabilities.

The example shows:
- Left: execution context diff of `main.md`
- Right: context of `shell-agent.md` (in agents folder), which generates zsh/bash commands based on user input, stores them in its context, executes, and returns results to the main module. Relative paths are supported for all files.

Operations:
1. `@run`: passes prompts, blocks, and context branches to the agent's context, concatenating them at the start
2. `@llm`: generates executable operations with parameter semantics. Uses `use-header: none` to prevent markdown header output for proper interpretation. Generated commands can use their own headers, referenced in `@return`
3. `@return`: outputs specified blocks by identifiers (with prompt parameter, returns literal constant)

⚠️ Important: Every module/agent must return a value for workflow semantic completeness. Missing returns cause unpredictable results (validation coming in future releases).

---
# Operations Overview

## Overview

The system maintains an internal "execution context" representing the document structure. Operations modify this context through insertion, replacement, or appending of content.

## Operation Block Structure

An operation block begins with an operation identifier line:

```
@operation_name
```

This is followed by a YAML section containing the operation's parameters.

## Operations Reference

### @import

**Purpose**: Import content from another file or from a specific section within that file.

| Field | Required | Type | Description | Default |
|-------|----------|------|-------------|---------|
| file | Yes | String | File path to import from | - |
| block | No | String | Reference to specific section within source file. Supports nested notation with trailing `*` | - |
| mode | No | String | How content is merged (`"append"`, `"prepend"`, `"replace"`) | `"append"` |
| to | No | String | Target block reference in current document | - |

**Execution Logic**:
1. System reads the specified file
2. If block reference is provided, extracts that portion
   - Using `/*` after a block reference includes all nested blocks
3. Inserts or replaces content at target location according to chosen mode
4. If no target specified, content is inserted at current position

**Examples**:

Import entire file
```yaml
@import
file: instructions/identity.md
```

Import specific section with nested blocks
```yaml
@import
file: docs/other.md
block: section/subsection
```

Import single block and replace target
```yaml
@import
file: "templates/header.md"
block: "main-header"
mode: "replace"
to: "document-header"
```

### @llm

**Purpose**: Send prompts to a language model and insert responses.

| Field | Required | Type | Description | Default |
|-------|----------|------|-------------|---------|
| prompt | See note | String | Literal text string for input prompt | - |
| block | See note | String/Array | Reference(s) to blocks for prompt content | - |
| media | No | Array | File paths for additional media context | - |
| save-to-file | No | String | File path to save raw response | - |
| use-header | No | String | Header for LLM response | `# LLM Response block` |
| header-auto-align | No | Boolean | Auto-adjust header levels in response based on current context | `false` |
| mode | No | String | Merge mode (`"append"`, `"prepend"`, `"replace"`) | Configuration default |
| to | No | String | Target block reference | - |
| provider | No | String | Override for language model provider | Configuration default |
| model | No | String | Override for specific model | Configuration default |

Note: Either `prompt` or `block` must be provided.

**Execution Logic**:
1. System assembles prompt by combining:
   - Referenced block contents (if block specified)
   - Literal prompt text (if provided)
   - Media context (if provided)
2. Sends assembled prompt to language model
3. Captures response
4. Optionally saves raw response to file
5. Adds header (unless set to "none")
6. Merges response into document at specified location

**Examples**:
Simple prompt with default settings
```yaml
@llm
prompt: "Generate a summary of the following text:"
```

# Complex prompt with multiple inputs and specific target
```yaml
@llm
prompt: "Compare and contrast the following sections:"
block: 
   - section1/*
   - section2/*
media: 
   - context.png
   - diagram.svg
use-header: "# Comparison Analysis"
mode: replace
to: analysis-section
```

Auto-align headers to current context level
```yaml
@llm
prompt: |
   Generate structured analysis with headers:
   # Main Topic
   ## Subtopic
   ### Details
header-auto-align: true
use-header: "# Analysis Results"
```

Save response to file with custom model; multiline prompt
```yaml
@llm
prompt: |
   Generate technical documentation
   about specifications above
save-to-file: output/docs.md
provider: custom-provider
model: technical-writer-v2
```

### @shell

**Purpose**: Execute shell commands and capture output.

| Field | Required | Type | Description | Default |
|-------|----------|------|-------------|---------|
| prompt | Yes | String | Shell command to execute | - |
| use-header | No | String | Header for command output | `# OS Shell Tool response block` |
| mode | No | String | Merge mode (`"append"`, `"prepend"`, `"replace"`) | Configuration default |
| to | No | String | Target block reference | - |

**Execution Logic**:
1. System sanitizes command string
2. Executes command in shell environment
3. Captures real-time output
4. Adds header (unless set to "none")
5. Merges output into document at specified location

**Examples**:
```yaml
# Simple command execution
@shell
prompt: "ls -la"
```

Process file and insert at specific location
```yaml
# Error section
This block will be replaced with the error log summary

@shell
prompt: "cat data.txt | grep 'ERROR' | sort -u"
use-header: "# Error Log Summary"
to: error-section
mode: replace
```


### @return

**Purpose**: Produce final output and end current workflow.

| Field | Required | Type | Description | Default |
|-------|----------|------|-------------|---------|
| prompt | See note | String | Literal text to return | - |
| block | See note | String/Array | Reference(s) to blocks to return | - |
| use-header | No | String | Header for returned content | `# Return block` |

Note: Either `prompt` or `block` must be provided.

**Execution Logic**:
1. System gathers content from:
   - Referenced blocks (if specified)
   - Literal prompt text (if provided)
2. Combines all content
3. Adds header (unless set to "none")
4. Returns final block as workflow output
5. Terminates workflow execution

**Examples**:
```yaml
# Return single block
@return
block: final-output
```

Return multiple blocks with custom header
```yaml
@return
block: 
   - summary/*
   - conclusions`
```

Return literal text
```yaml
@return
prompt: Process completed successfully
```

### @run

**Purpose**: Execute another markdown file as a workflow.

| Field | Required | Type | Description | Default |
|-------|----------|------|-------------|---------|
| file | Yes | String | Path to markdown file to execute | - |
| prompt | No | String | Literal text input for workflow | - |
| block | No | String/Array | Reference(s) to blocks for input | - |
| use-header | No | String | Header for workflow output | - |
| mode | No | String | Merge mode (`"append"`, `"prepend"`, `"replace"`) | Configuration default |
| to | No | String | Target block reference | - |

**Execution Logic**:
1. System loads specified markdown file
2. Assembles input from:
   - Referenced blocks (if specified)
   - Literal prompt text (if provided)
3. Executes file as separate workflow with assembled input
4. Captures workflow output
5. Adds header (if specified)
6. Merges results into current document at target location

**Examples**:
Simple workflow execution
```yaml
@run
file: workflows/process.md
```

Multiple inputs and append results
```yaml
# Reports
This root block would be appended with the results of the workflows

@run
file: workflows/compare.md
prompt: Compare performance metrics
block: 
   - metrics-2023/* 
   - metrics-2024/*
mode: append
to: reports
```

## Execution Context

The system processes documents by maintaining an Abstract Syntax Tree (AST) that represents the document structure. Operations modify this AST through three primary actions:

- **Append**: Add new content after the target
- **Prepend**: Insert new content before the target
- **Replace**: Completely replace target content

Each operation:
1. Gathers input (from files, prompts, or command output)
2. Merges content into the document based on target reference and mode
3. Updates the execution context accordingly

The document evolves incrementally as each operation is processed, building toward the final output.

## Block References

Block references (field: `block`) can use several special notations:
- Simple reference: `section-name`
- Nested reference: `parent/child`
- Wildcard nested: `section/*` (includes all nested blocks)
- Multiple blocks are accessabe by using YAML array syntax

## Special Values

For any operation that accepts a `use-header` parameter, the special value `"none"` (case-insensitive) can be used to omit the header entirely.

---


### Operation Field Matrix  
**✓ = Supported** | **– = Not Applicable** | **A** = Array Accepted  

| Field               | @llm | @import | @shell | @run  | @return | @goto | 
|---------------------|------|---------|--------|-------|---------|-------|
| **Core Fields**     |      |         |        |       |         |       |
| `to`                | ✓    | ✓       | ✓      | ✓     | –       | –     | 
| `mode`              | ✓    | ✓       | ✓      | ✓     | –       | –     | 
| `block`             | ✓(A) | ✓       | –      | ✓(A)  | ✓       | ✓     | 
| `file`              | –    | ✓       | –      | ✓     | –       | –     |
| `prompt`            | ✓    | –       | ✓      | ✓     | ✓       | –     |
| `use-header`        | ✓    | –       | ✓      | ✓     | ✓       | –     |
| `header-auto-align` | ✓    | –       | –      | –     | –       | –     |
| **Specialized**     |      |         |        |       |         |       |
| `provider`          | ✓    | –       | –      | –     | –       | –     |
| `model`             | ✓    | –       | –      | –     | –       | –     |
| `save-to-file`      | ✓    | –       | –      | –     | –       | –     |
| `media`             | ✓(A) | –       | –      | –     | –       | –     |




