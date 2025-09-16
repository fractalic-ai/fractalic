[![PyPI Version](https://img.shields.io/pypi/v/fractalic.svg)](https://pypi.org/project/fractalic/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt) ![Status](https://img.shields.io/badge/status-early--adopter-orange) ![Python](https://img.shields.io/badge/python-3.11+-blue.svg) [![Docs](https://img.shields.io/badge/docs-reference-purple)](docs/)

<p align="center">
  <img src="https://raw.githubusercontent.com/fractalic-ai/fractalic/main/docs/images/fractalic_hero.png" alt="Fractalic Hero Image">
</p>

# What is Fractalic?
Design, run and evolve multi‑model AI workflows in one executable Markdown file—no glue code—with precise context control, tool integration and git‑traceable reproducibility.

## What's New in v0.1.x
This update focuses on making Fractalic more practical for everyday use. We added better model options, tool handling, and ways to deploy and debug. Here's a rundown of the changes.

### 🧠 AI & Model Support
- 🤖 LiteLLM integration, supporting over 100 models and providers.
- 🔄 Scripts now work as complete agents, with two-way parameter passing in LLM modes.
- 📊 Basic token tracking and cost analytics (still in early stages).
- 🧠 Improved context diffs (.ctx) for multi-model workflows.

### ⚡ MCP & Tool Ecosystem
- ⚡ Full MCP support, including schema caching.
- 🔐 OAuth 2.0 and token management for MCP services.
- 🛒 MCP marketplace in Fractalic Studio for one-click installs.
- 🔧 Fractalic Tools marketplace with one-click options: Telegram, HubSpot CRM, Tavily web search, MarkItDown, HubSpot process-mining, ultra-fast grep, file patching, and others.
- 🐍 Support for using Python modules as tools.
- 👁️ Tool call tracing, available in context and through the Studio inspector.

### 🚀 Deployment & Publishing
- 🚀 Publisher system with Docker builds and a lightweight server for REST APIs, including Swagger docs.
- 🐳 Automated deployments with process supervision.
- 📦 Fractalic now available as a Python package for standalone use or importing as a module.

### 🎨 Fractalic Studio (IDE)
- 🖥️ Development environment with session views, diff inspector, editor, and deployment tools.
- 📝 Notebook-style editor for building workflows step by step.
- 🛒 Integrated marketplaces for MCP servers and tools.
- 🔍 Debugging features like execution tracing and context inspection.

### 📚 Documentation & Stability
- 📖 Detailed docs covering all features and examples.
- 🛠️ Better stability for tool executions, with improved structured outputs.

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
    - [Method 1: Pre-Built Docker Image (Recommended)](#method-1-pre-built-docker-image-recommended)
    - [Method 2: Build from Source (Full Stack)](#method-2-build-from-source-full-stack)
    - [Method 3: Local Development Setup](#method-3-local-development-setup)
    - [Method 4: Python Package - CLI](#method-4-python-package---cli)
    - [Method 5: Python Package - API](#method-5-python-package---api)
  - [Basic CLI Usage](#basic-cli-usage)
  - [Usage as Python Module](#usage-as-python-module)
- [Basic Principles](#basic-principles)
- [Fractalic Operations](#fractalic-operations)
- [Basic Examples](#basic-examples)
  - [Example: Web Search → Notion Page (MCP Integration)](#example-web-search--notion-page-mcp-integration)
  - [Example: Replicate Image Generation + Download](#example-replicate-image-generation--download)
  - [Example: Context Compression Pattern](#example-context-compression-pattern)
  - [Example: Dynamic Operation Generation](#example-dynamic-operation-generation)
- [Screenshots](#screenshots)
- [Integrations & Credits](#integrations--credits)
- [License](#license)


## Getting Started

### Installation

#### Method 1: Pre-Built Docker Image (Recommended)
Run the published container directly with all services (UI + API + AI server):
```bash
docker run -d --name fractalic --network bridge -p 3000:3000 -p 8000:8000 -p 8001:8001 -p 5859:5859 -v /var/run/docker.sock:/var/run/docker.sock --env HOST=0.0.0.0 ghcr.io/fractalic-ai/fractalic:main
```
Then open: http://localhost:3000

#### Method 2: Build from Source (Full Stack)
Builds latest version from GitHub repositories and runs in Docker:
```bash
curl -s https://raw.githubusercontent.com/fractalic-ai/fractalic/main/deploy/docker-deploy.sh | bash
```
This clones both fractalic + fractalic-ui, builds Docker image locally, and starts all services:
- UI: http://localhost:3000
- API: http://localhost:8000
- AI Server: http://localhost:8001
- MCP Manager: http://localhost:5859

#### Method 3: Local Development Setup
Full source installation with both backend and frontend for development:
```bash
git clone https://github.com/fractalic-ai/fractalic.git
cd fractalic
./local-dev-setup.sh
```
This script will:
- Clone fractalic-ui repository
- Set up Python virtual environment
- Install all dependencies
- Start both backend and frontend servers
- Open http://localhost:3000 automatically

#### Method 4: Python Package - CLI
Install for command-line usage (no UI):
```bash
pip install fractalic
```

Check install:
```bash
fractalic --help
```

Run a workflow file:
```bash
fractalic your_workflow.md
```

#### Method 5: Python Package - API
Install for programmatic usage in Python:
```bash
pip install fractalic
```

Then use in your Python code:
```python
import fractalic

# Run a workflow file
result = fractalic.run_fractalic('workflow.md')

# Run with user input parameters
result = fractalic.run_fractalic('workflow.md', param_input_user_request='Tesla analysis')

# Run with custom model and API key
result = fractalic.run_fractalic(
    'workflow.md', 
    model='openai/gpt-4',
    api_key='your-api-key'
)

# Result is a dictionary with execution details
print(f"Success: {result['success']}")
print(f"Branch: {result['branch_name']}")
print(f"Context file: {result['ctx_file']}")
print(f"Context hash: {result['ctx_hash']}")
```

## Basic Principles
- **One executable Markdown file**: Your workflow specification *is* your runtime. Write what you want in plain Markdown, run it directly. No translation between documentation and code.

- **No glue code**: Replace Python/JS/(any program language) orchestration scripts with 3-6 line YAML plain-text operations. 

- **Multi-model workflows**: Switch between LLM models and providers in the same document. 

- **Precise context control**: Your Markdown becomes a manageable LLM context as an addressable tree. Reference exact sections, branches, or lists. LLMs see only what you specify—no hidden prompt stuffing.

- **Tool integration**: Connect MCP servers, Python functions, and shell commands. All outputs flow back into your document structure for the next operation.

- **Human‑readable audit trail**: Each run outputs a stepwise execution tree plus a complete change log (new blocks, edits, tool calls). Skim it like a focused diff—only actions and their effects, no noise.



## Fractalic Operations
Fractalic is built around a set of key operations. These are deterministic instructions that the Fractalic interpreter executes in sequence to process your workflow. Each operation performs a specific task, like calling an AI model, running a shell command, or manipulating the document's content.

- `@llm` – Sends specified blocks of content to any supported language model, including local models.
- `@shell` – Runs terminal commands, with the output captured as a new block in the document.
- `@run` – Executes another Fractalic Markdown file, allowing you to modularize workflows, pass parameters, and receive results.
- `@import` – Includes content from other files directly into your document.
- `@return` – Sends specified blocks back as a result to a parent workflow that used the `@run` operation.

Each operation can be customized with a variety of parameters. For a detailed reference of all available options, please see the [Operations Reference documentation](docs/operations-reference.md).


## Basic Examples

The following examples demonstrate how you can combine Fractalic's operations (using YAML syntax) with your knowledge (written as Markdown blocks) to create powerful, automated workflows. You'll see how to integrate with external tools, generate and manipulate content, and even create workflows that write themselves.

*Note on examples: The execution results are shown as a `diff`. Green highlighted text (`+`) represents the new content added to the document after the workflow runs. The `+` markers are an artifact of GitHub's diff formatting and won't appear in the actual output file. Some tool outputs have been truncated for brevity.*

## Example: Web Search → Notion Page (MCP Integration)

This example demonstrates a complete workflow: it uses the `tavily_search` tool to find information on the web, then passes the structured results to the `mcp/notion` tool to create a new, formatted page in Notion. This showcases how Fractalic can chain different services together to automate a research and publishing task.
```markdown
# Web search task
Find top-5 world news for today about AI, provide brief summary about each, print them under "# AI news" header (add empty line before it) and suppliment each with direct link

@llm
prompt: Search news 
tools: tavily_search

# Notion task
Based on extracted news, extract important insights, keep for each news a direct link - and save them as newspaper (please format it properly) to my Notion, create new page there - Daily AI news

@llm
prompt: Process news to Notion
block: 
    - notion-task
    - ai-news
tools: mcp/notion
```
Execution result:
```diff
# Web search task
Find top world news for today about AI, provide brief summary about each, print them under "# AI news" header (add empty line before it) and suppliment each with direct link

@llm
prompt: Search news 
tools: tavily_search

+ # LLM response block
+ 
+ > TOOL CALL, id: call_7fl4HiwuAV7crDV9TNJyyCu1
+ tool: tavily_search
+ args:
+ {
+   "task": "search",
+   "query": "AI news today top world news artificial intelligence",
+   "search_depth": "basic",
+   "topic": "news",
+   "days": 1,
+   "max_results": 10
+ }
+ 
+ > TOOL RESPONSE: (search results with 10 AI news articles)
+ 
+ # AI news
+ 
+ 1) OpenAI corporate move: OpenAI announced its nonprofit parent will retain control while the parent also gains an equity stake reported to be worth over $100 billion — a major structural and governance development for the company.  
+ Link: https://www.foxnews.com/tech/ai-newsletter-backlash-over-mystery-companys-data-center
+ 
+ 2) How AI is changing grieving: Reuters reports on growing use of voice cloning and digital avatars to preserve deceased loved ones, exploring benefits for mourning alongside ethical, consent and data‑privacy concerns.  
+ Link: https://www.reuters.com/lifestyle/it-feels-like-almost-hes-here-how-ai-is-changing-way-we-grieve-2025-09-13/
+ 
+ 3) Apple AI leadership exit: Bloomberg reports that Robby Walker, a senior AI and search executive at Apple who oversaw Siri until earlier this year, is leaving the company — a notable leadership change in Apple's AI team.  
+ Link: https://www.bloomberg.com/news/articles/2025-09-12/apple-ai-and-search-executive-robby-walker-to-leave-iphone-maker
+ 
+ 4) Demis Hassabis on future skills: Google's/DeepMind's Demis Hassabis says "learning how to learn" will be one of the most important skills for the next generation as AI reshapes education and work, emphasizing adaptiveness and lifelong learning.  
+ Link: https://www.greenwichtime.com/news/world/article/google-s-top-ai-scientist-says-learning-how-to-21045337.php
+ 
+ 5) xAI workforce cut: Reports say Elon Musk's xAI cut about 500 roles, largely affecting generalist AI tutors, as the company shifts toward a more specialist training model for its Grok chatbot — signaling changes in how AI training workforces are structured.  
+ Link: https://rollingout.com/2025/09/13/musks-xai-reduces-workforce-by-500/

# Notion task
Based on extracted news, extract important insights, keep for each news a direct link - and save them as newspaper (please format it properly) to my Notion, create new page there - Daily AI news

@llm
prompt: Process news to Notion
block: 
    - notion-task
    - ai-news
tools: mcp/notion

+ # LLM response block
+ 
+ > TOOL CALL, id: call_l626mZp42CTRV0bM79avSgei
+ tool: notion_notion-create-pages
+ args:
+ {
+   "pages": [
+     {
+       "content": "# Daily AI news — 2025-09-13\n\nA short, curated \"newspaper\" of important AI developments..."
+     }
+   ]
+ }
+ 
+ > TOOL RESPONSE: {"success": true, "result": {"content": {"pages": [{"id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", "url": "https://www.notion.so/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}]}}}
+ 
+ Done — I created a Notion page titled "Daily AI news — 2025-09-13" containing the curated items and direct links. Link to the new page: https://www.notion.so/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**How this works:**
Think of this workflow as a two-step process. First, we tell an AI model to search the web for news. Then, we tell it to take those search results and save them to a Notion page.

In the first `@llm` step, Fractalic combines the instructions from the `# Web search task` with the `tavily_search` tool. By telling the model exactly which tool to use, we make its job simpler and avoid unnecessary processing costs (tokens). The results, including the raw data from the tool, are neatly stored back into our document under new headings.

For the second `@llm` step, we give the model the instructions from `# Notion task` and the clean `# AI news` results from the previous step. Notice that we don't show it the messy, raw JSON data from the search tool. This is Fractalic's "context isolation" in action: each step gets only the information it needs. This keeps the process efficient and the document clean.

**What happened here:**
- **Hybrid Tool Integration**: This example shows a Fractalic built-in tool (`tavily_search`) working seamlessly in sequence with an external MCP service (`mcp/notion`).
- **Tool Output → Document Flow**: Web search results automatically become structured content in the document tree, which the next operation can reference.
- **Context Control**: Each `@llm` operation sees only the specific blocks it needs—no hidden prompt stuffing or context pollution.
- **Multi-step Automation**: A complex workflow (search → format → save to external service) runs with just two simple operations.
- **Real API Integration**: Shows actual OAuth-authenticated API calls to external services, with real URLs and IDs returned.

## Example: Replicate Image Generation + Download

This workflow demonstrates how Fractalic can orchestrate a multi-step process involving a cloud-based AI service (Replicate) and local shell commands. First, it retrieves the technical specifications (schema) for an image generation model, uses that schema to generate an image, and then downloads and opens the resulting image file on the local machine.
```markdown
# Your goal
1. Please  get on replicate schema of google/nano-banana model, avoid tool usage without pre-filtering outputs
2. Generate image showcasing cute bunny 

@llm
prompt: generate requested image
tools: mcp/replicate

# Image download instructions
1. Check if prediction complete
3. Download image
4. Open image in os (we are at macos, do not use pyhon)

@llm
prompt: get image
tools: shell_tool
```
Execution result:
```diff
# Your goal
1. Please  get on replicate schema of google/nano-banana model, avoid tool usage without pre-filtering outputs
2. Generate image showcasing cute bunny

@llm
prompt: generate requested image
tools: mcp/replicate

+ # LLM response block
+ 
+ > TOOL CALL, id: call_ZKgQ243YPefqWQPvrWbhaiY6
+ tool: Replicate_get_models
+ args:
+ {
+   "model_owner": "google",
+   "model_name": "nano-banana",
+   "jq_filter": ".latest_version.openapi_schema.components.schemas"
+ }
+ 
+ > TOOL RESPONSE: 
+ ...
+ Bunch of lines with model schema with Input/Output specifications
+ ...
+ 
+ > TOOL CALL, id: call_UkFRxoiYmxBCJIAqnNK2hsD2
+ tool: Replicate_create_models_predictions
+ args:
+ {
+   "model_owner": "google",
+   "model_name": "nano-banana",
+   "input": {
+     "prompt": "A very cute fluffy baby bunny sitting on soft grass, pastel color palette, big sparkling eyes...",
+     "output_format": "png"
+   }
+ }
+ 
+ > TOOL RESPONSE: "https://replicate.delivery/xezq/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx/tmp.png"
+ 
+ I retrieved the nano-banana model schema and generated the image you requested.
+ Generated image: PNG URL: https://replicate.delivery/xezq/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx/tmp.png

# Image download instructions
1. Check if prediction complete
3. Download image
4. Open image in os (don't use Python)

@llm
prompt: get image
tools: shell_tool

+ # LLM response block
+ 
+ > TOOL CALL, id: call_Cj7hjShKGepLaU8bucpPqDYl
+ tool: shell_tool
+ args:
+ {
+   "cmd": "curl -I -L \"https://replicate.delivery/...\" && curl -L -o bunny.png \"https://replicate.delivery/...\" && ls -l bunny.png && file bunny.png"
+ }
+ 
+ > TOOL RESPONSE: {"stdout": "HTTP/2 200... -rw-r--r--@ 1 dima staff 1346357 Sep 14 00:28 bunny.png\nbunny.png: PNG image data, 1024 x 1024, 8-bit/color RGB, non-interlaced"}
+ 
+ > TOOL CALL, id: call_nBvHO34MOLtvszwMK7eow0Fo
+ tool: shell_tool
+ args: {"cmd": "open bunny.png && echo 'Opened bunny.png'"}
+ 
+ > TOOL RESPONSE: {"stdout": "Opened bunny.png"}
+ 
+ Done.
+ - Downloaded file: bunny.png (1,346,357 bytes, 1024×1024 PNG)
+ - Image opened in macOS default viewer
```

**How this works:**
This example shows how Fractalic can connect to an online service (Replicate) to create an image, then use your computer's own tools to download and open it.

The first `@llm` step is a two-part conversation with the Replicate service. First, it asks for the "schema" of the image model. Think of this as asking for the instruction manual. With that information, it then makes a second call to actually generate the bunny image.

The second `@llm` step switches gears completely. It uses the `shell_tool` to run commands directly on your computer. It takes the image URL from the first step and uses standard command-line tools like `curl` to download the image and `open` to view it. This demonstrates how easily Fractalic can bridge the gap between cloud services and your local machine.

**What happened here:**
- **Tool Chaining**: MCP Replicate integration followed by shell commands, demonstrating how different tool types work together
- **Cross-Platform Commands**: Shell tools automatically adapt to the OS (macOS `open` command used here)
- **Structured Output Parsing**: Model schemas are retrieved and used programmatically, showing API-first tool integration
- **File System Integration**: Generated content flows from cloud APIs to local files seamlessly

## Example: Context Compression Pattern

This pattern is essential for managing the context window in large language models. LLMs often generate verbose output, especially when using tools, which can quickly consume the available token limit. This example shows how to run an operation, get a large block of text, and then run a second operation that replaces the verbose text with a concise summary, preserving the essential information while freeing up context for subsequent steps.

```markdown
@llm
prompt: generate few parargraphs about GPT arhotecture
use-header: "# Some large output"

@llm
prompt: extract consice detailed info from data above - keep only knowledge, remove noise
mode: replace
to: some-large-output
```

Intermediate state (after the first `@llm` operation):
```markdown
# Some large output

The Generative Pre-trained Transformer (GPT) architecture is a type of neural network model that has revolutionized the field of natural language processing. Developed by OpenAI, it is based on the transformer architecture, which was introduced in the paper "Attention Is All You Need." The key innovation of the transformer is the self-attention mechanism, which allows the model to weigh the importance of different words in the input text when processing and generating language. GPT models are "pre-trained" on vast amounts of text data from the internet, which enables them to learn grammar, facts, reasoning abilities, and a wide range of language patterns.

... (400+ lines of detailed text) ...

This process of autoregressive generation, combined with the vast knowledge encoded during pre-training, allows GPT models to perform a wide array of tasks, from writing essays and code to answering questions and carrying on conversations. The quality and coherence of the generated text are a direct result of the model's ability to predict the most likely next word based on the patterns it has learned from its training data.
```

Execution result:
```diff
@llm
prompt: generate 30 parargraphs about GPT arhotecture
use-header: "# Some large output"

+ # LLM response block
+ - Architecture: decoder-only transformer (GPT family).
+ - Input processing: tokenization → token embeddings; positional information added.
+ - Core stack: repeated transformer blocks with multi-head self-attention and position-wise feed-forward networks; residual connections and layer normalization around sublayers.
+ - Attention: causal (masked) self-attention so each token can attend only to previous tokens.
+ - Multi-head attention: splits representations into multiple heads to capture diverse relationships in parallel.
+ - Feed-forward sublayer: two linear transformations with a nonlinearity applied independently at each position.
+ - Output: final layer projects to token logits → probabilities for next-token prediction.
+ - Training objective: next-token prediction on very large text corpora.
+ - Scaling: capabilities improve with more data, larger model size, and longer training (scaling laws).
+ - Alignment/finetuning: supervised fine-tuning and reinforcement learning from human feedback (RLHF) used to improve usefulness and safety.

@llm
prompt: extract consice detailed info from data above - keep only knowledge, remove noise
mode: replace
to: some-large-output
```

**How this works:**
LLMs, especially when using tools, can generate a lot of output (tokens). A fundamental challenge with LLM architectures is that the entire accumulated context is typically sent with each subsequent call. This can be inefficient and costly.

The first `@llm` step generates a detailed, but verbose, block of text.

The second `@llm` step demonstrates a powerful solution. It uses `mode: replace` to instruct Fractalic: "take the content in the `# Some large output` block, create a concise summary, and then replace the original text with that summary." This allows you to manage the document's size dynamically. You get the benefit of the initial detailed generation but end up with a clean, token-efficient version for use in later steps. The block's address (`some-large-output`) remains stable, ensuring that other parts of your workflow can still reference it without issue.

**What happened here:**
- **Replace Mode**: The `mode: replace` operation shows how content can be compressed in-place to save tokens
- **Block Targeting**: The `to: some-large-output` parameter precisely targets which block to replace
- **Token Cost Control**: Large generated content gets distilled into essential facts, reducing costs for subsequent operations
- **Content Evolution**: Document structure evolves from verbose to concise while preserving knowledge
- **Memory Management**: Demonstrates how to handle large outputs without context overflow

## Example: Dynamic Operation Generation

This example showcases a unique feature of Fractalic: enabling an AI model to dynamically control its own execution flow. This is a form of polymorphism where the AI can generate and execute new instructions based on the context of the document, leading to highly adaptive and powerful workflows. This is crucial for tasks where the exact steps are not known in advance, allowing the AI to reason about the document's structure and build its own plan.
```markdown
# Wiki

## USA
something about USA

## France
something about France

## UK
something about UK

@llm
prompt: use fractalic_opgen to generate for each sub block of wiki, an @llm instructions, to return one paragraph about country, set mode to replace, to = sub-block title in small-case kebab case, and use-header equal to sub-block header. No file field. Print generated instructions only separated and prexeeded by emptylines
tools: fractalic_opgen
```
Execution result:
```diff
# Wiki

## USA
+ The United States of America is a large federal republic in North America made up of 50 states and the federal district of Washington, D.C.; it has a diverse landscape ranging from coasts and plains to mountains and deserts and a population of about 330 million people. As the world's largest economy by nominal GDP, the U.S. is a global leader in technology, finance, entertainment, and higher education, and it hosts a wide mix of cultures and ethnicities due to centuries of immigration. Governed by a constitutional system with separation of powers, the country plays a major role in international politics, trade, and culture.

## France
+ France is a country in Western Europe with a population of about 67 million and its capital in Paris; French is the official language and the government is a semi-presidential republic. A founding member of the European Union and a permanent member of the UN Security Council, France has a large, diversified economy driven by industries such as aerospace, luxury goods, agriculture and tourism, and it uses the euro as its currency. Renowned worldwide for its culture, cuisine and wine, art and fashion, France features iconic landmarks like the Eiffel Tower and the Louvre, and diverse landscapes ranging from Atlantic and Mediterranean coasts to the Alps and Pyrenees.

## UK
+ The United Kingdom (officially the United Kingdom of Great Britain and Northern Ireland) is a sovereign country in northwestern Europe made up of England, Scotland, Wales and Northern Ireland, with a population of about 67 million and its capital in London. It is a constitutional monarchy with a parliamentary system, and its economy is large and service-oriented—notably finance, education, research, creative industries and tourism—using the pound sterling as its currency. A permanent member of the UN Security Council and a G7 country, the UK played a central historical role through its former empire; it exited the European Union in 2020. The UK is renowned for its cultural institutions, universities, and landmarks such as Buckingham Palace, the Houses of Parliament (Big Ben), and Stonehenge, and it features a variety of landscapes from lowland plains to upland highlands and rugged coastlines.

@llm
prompt: use fractalic_opgen to generate for each sub block of wiki, an @llm instructions, to return one paragraph about country, set mode to replace, to = sub-block title in small-case kebab case, and use-header equal to sub-block header. No file field. Print generated instructions only separated and prexeeded by emptylines
tools: fractalic_opgen

+ # LLM response block
+ 
+ > TOOL CALL: fractalic_opgen (3 calls generating operations)
+ 
+ @llm
+ prompt: Return one paragraph about the USA.
+ tools: none
+ temperature: 0.0
+ use-header: '## USA'
+ mode: replace
+ to: usa
+ 
+ @llm
+ prompt: Return one paragraph about France.
+ tools: none
+ temperature: 0.0
+ use-header: '## France'
+ mode: replace
+ to: france
+ 
+ @llm
+ prompt: Return one paragraph about the UK.
+ tools: none
+ temperature: 0.0
+ use-header: '## UK'
+ mode: replace
+ to: uk
```

**How this works:**
This example showcases a unique feature of Fractalic: giving the AI dynamic access to control its own execution. This is a powerful form of meta-programming or polymorphism, where the AI can generate new instructions for itself based on the document's content.

We start with a simple `# Wiki` block containing several country subheadings. We then give the model the `fractalic_opgen` tool and instruct it to create a new operation for each country to write a summary paragraph.

The model analyzes the document structure, identifies the three country subheadings, and uses `fractalic_opgen` to generate three new, perfectly formatted `@llm` operations. Each new operation is tailored to target a specific subheading (`to: usa`, `to: france`, `to: uk`) and replace its placeholder text.

This is critically important for building workflows that can adapt to changing inputs. Instead of hard-coding every step, you can create systems where the AI reasons about the content it's given and builds the appropriate execution plan on the fly.

**What happened here:**
- **Self-Extending Workflows**: The `fractalic_opgen` tool allows models to generate new operations dynamically
- **Block-Aware Generation**: The model analyzed the document structure (`# Wiki` with country subheadings) and created appropriate operations
- **Operation Templates**: Generated operations follow proper YAML structure with parameters like `mode: replace` and `to:` targeting
- **Content Replacement**: Original placeholder text ("something about USA") gets replaced with rich, factual content
- **Meta-Programming**: Shows how Fractalic can modify its own execution flow based on document content


## Screenshots

<table>
  <tr>
    <td width="50%">
      <img src="docs/images/editor.png" alt="Fractalic Editor - Notebook-style UI with Markdown and YAML operations" />
      <p align="center"><em>Main Editor Interface</em></p>
    </td>
    <td width="50%">
      <img src="docs/images/notebook.png" alt="Notebook View - Interactive document execution with live results" />
      <p align="center"><em>Notebook Execution View</em></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="docs/images/tools.png" alt="MCP Tools Integration - Access external services via Model Context Protocol" />
      <p align="center"><em>MCP Tools Integration</em></p>
    </td>
    <td width="50%">
      <img src="docs/images/mcp.png" alt="MCP Manager - Unified tool and service management interface" />
      <p align="center"><em>MCP Manager Interface</em></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="docs/images/diff.png" alt="Git-backed Diffs - Complete execution trace with version control" />
      <p align="center"><em>Git-backed Execution Diffs</em></p>
    </td>
    <td width="50%">
      <img src="docs/images/inspector.png" alt="Debug Inspector - Deep inspection of execution state and variables" />
      <p align="center"><em>Debug Inspector</em></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <img src="docs/images/inspector-messages.png" alt="Message Inspector - Detailed view of AI conversation turns and tool calls" />
      <p align="center"><em>Message Inspector</em></p>
    </td>
    <td width="50%">
      <img src="docs/images/markdown.png" alt="Markdown Editor - Clean document editing with syntax highlighting" />
      <p align="center"><em>Markdown Editor</em></p>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="docs/images/deploy.png" alt="Deployment Dashboard - One-click containerization and service deployment" width="50%" />
      <p align="center"><em>Deployment Dashboard</em></p>
    </td>
  </tr>
</table>

## Integrations & Credits
- LiteLLM (https://github.com/BerriAI/litellm)
- FastMCP (https://github.com/jlowin/fastmcp)

## License
MIT