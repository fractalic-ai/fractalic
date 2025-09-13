[![PyPI Version](https://img.shields.io/pypi/v/fractalic.svg)](https://pypi.org/project/fractalic/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt) ![Status](https://img.shields.io/badge/status-early--adopter-orange) ![Python](https://img.shields.io/badge/python-3.11+-blue.svg) [![Docs](https://img.shields.io/badge/docs-reference-purple)](docs/)

<p align="center">
  <img src="https://raw.githubusercontent.com/fractalic-ai/fractalic/main/docs/images/fractalic_hero.png" alt="Fractalic Hero Image">
</p>

# Fractalic
Design, run and evolve multi‑model AI workflows in one executable Markdown file—no glue code—with precise context control, tool integration and git‑traceable reproducibility.

## What's New in v0.1.0
- ⚡ FastMCP SDK v2 (≈280× tool call throughput) + intelligent caching
- 🔐 OAuth 2.0 integration & token lifecycle management for MCP services
- 🐳 Multi-architecture Docker builds & production process supervision
- 🚀 Publisher system for automated, declarative deployments
- 🧠 Precise context diffs (.ctx) + refined multi-model orchestration
- 🛠 Tool execution stability & structured return improvements
- 🧪 Expanded test coverage + consolidated documentation

## How
**One executable Markdown file**: Your workflow specification *is* your runtime. Write what you want in plain Markdown, run it directly. No translation between documentation and code.

**No glue code**: Replace Python/JS/(any program language) orchestration scripts with 3-6 line YAML plain-text operations. 

**Multi-model workflows**: Switch between LLM models and providers in the same document. 

**Precise context control**: Your Markdown becomes an addressable tree. Reference exact sections, branches, or lists. LLMs see only what you specify—no hidden prompt stuffing.

**Tool integration**: Connect MCP servers, Python functions, and shell commands. All outputs flow back into your document structure for the next operation.

**Git-traceable reproducibility**: Every run produces a `.ctx` file showing exactly what changed. Diff your workflow evolution semantically, not just text changes.

## From Idea to Service in 5 Steps

**1. Minimal workflow (hard‑coded input):**
```markdown
# Market Research {id=task}
Apple quarterly analysis

@llm
prompt: "Search for Apple latest quarterly results"
tools: tavily_search
blocks: task
use-header: "# Financial Data"

@llm
prompt: "Create executive summary of key insights"
blocks: financial-data
use-header: "# Analysis Summary"
```

Run:
```bash
python fractalic.py research.md
```

**2. Add dynamic input (parameter injection):**
Create a parameter file on the fly and inject it as `# Input Parameters`:
```bash
echo '# Input Parameters {id=input-parameters}\nCompany: Tesla' > params.md
python fractalic.py research.md --task_file params.md --param_input_user_request input-parameters
```
Modify workflow to consume injected block:
```markdown
# Market Research {id=task}
@import
blocks: input-parameters
mode: append

@llm
prompt: "Search latest quarterly results for the company in Input Parameters. Extract key metrics."
blocks: input-parameters
use-header: "# Financial Data"

@llm
prompt: "Summarize financial performance in 4 bullet points."
blocks: financial-data
use-header: "# Analysis Summary"
```

**3. Inspect structured diff (.ctx):**
```diff
+ # Input Parameters {id=input-parameters}
+ Company: Tesla
+ # Financial Data {id=financial-data}
+ Tesla Q4 2024 results: Revenue ...
+ # Analysis Summary {id=analysis-summary}
+ • Growth drivers ...
```

**4. Add return for service interface:**
```markdown
@return
blocks: analysis-summary
```
Result: upstream caller (CLI / AI Server) receives only the summary.

**5. Call via AI Server (REST):**
```bash
curl -X POST http://localhost:8001/execute \
  -H 'Content-Type: application/json' \
  -d '{"filename": "research.md", "parameter_text": "Company: Nvidia"}'
```
Note: The server automatically creates an in‑memory `# Input Parameters {id=input-parameters}` block from `parameter_text` (same mechanism works when another document calls this one via @run + prompt). You do not add that block manually before deployment.

**Result:** Same Markdown becomes an on-demand research microservice. No placeholders, no templating engine.

## How It Works
Your Markdown document becomes three things:

**Document tree**: Headings create addressable blocks. Reference exact sections (`block_uri: idea`) or branches (`idea/*`). 

**Operation sequence**: `@llm`, `@shell`, `@run` operations execute in order. Each changes your document structure predictably.

**Context window**: Models see only the blocks you specify. No hidden prompt stuffing.

## Operations
Five operations control your workflow:

- `@llm` – Send specific blocks to any model. GPT-4, Claude, local models.
- `@shell` – Run terminal commands. Output becomes a new block.
- `@run` – Execute another Markdown file. Pass parameters, get results back.
- `@import` – Include content from other files.
- `@return` – Send blocks back to parent workflow.

## Advanced: Let Models Control Flow
Give models tools to extend your workflow:
- `fractalic_run` – Model decides when to spawn sub-workflows
- `fractalic_opgen` – Model generates new operations to execute

This creates self-extending workflows without custom code.

## Example
```markdown
# Launch Brief {id=brief}
We need a launch narrative for our AI workflow feature.

@llm
prompt: "List 6 user pains."
blocks: brief
mode: append

@shell
command: curl api.example.com/trends > trends.json

@llm
prompt: "Draft 150-word narrative using pains + trends."
blocks: brief/*
use-header: "# Draft"

@return
blocks: draft
```

## Extended Examples (MCP & Media)

### Web Search → Notion Page (MCP tavily_search + mcp/notion)
```markdown
# Agent Framework Brief {id=agent-brief}
Track emerging AI agent frameworks (≤4 weeks) and store a concise structured summary into Notion.

@llm
prompt: |
  1. Find up to 6 NEW or fast-rising AI agent frameworks (max 4 weeks old in news / blog chatter).
  2. For each: name, category (orchestration, memory, eval, routing, UI), 1 differentiator.
  3. Write concise bullets (no prose) into '# Source Notes {id=source-notes}'. No duplicates.
blocks: agent-brief
tools:
  - tavily_search
tools-turns-max: 2
use-header: "# Source Notes {id=source-notes}"

@llm
prompt: |
  Create a Notion page titled "Agent Framework Trends — {{today}}" containing:
  - Executive snapshot (2 sentences)
  - Table (framework | category | differentiator | maturity: early/active)
  - 3 strategic implications
  Source ONLY 'source-notes'. Return confirmation (title + page id if any) to '# Notion Result {id=notion-result}'.
blocks: source-notes
tools:
  - mcp/notion
tools-turns-max: 3
use-header: "# Notion Result {id=notion-result}"

@return
blocks: notion-result
```

### Replicate Image (flux-dev) + Optional Animation + Shell Download
```markdown
# Visual Asset Generation {id=visual-goal}
Generate a concept image (agent coordination) + optional short animation. Collect raw URLs only.

@llm
prompt: |
  Tasks:
  1. If available, use flux-dev (or similar) to generate a single image: neon network of cooperating AI agent nodes on dark background.
  2. Extract the direct image URL (png/jpg) -> '# Image URL {id=image-url}' (raw URL only).
  3. If an animation-capable model (wan / wan2 / video diffusion) exists, request ≤4s subtle loop, store raw URL in '# Animation URL {id=animation-url}'. Skip if absent (do not hallucinate).
  4. Keep '# Generation Log {id=generation-log}' minimal: list chosen model(s) + status.
blocks: visual-goal
tools:
  - mcp/replicate
tools-turns-max: 3
use-header: "# Generation Log {id=generation-log}"

@llm
prompt: |
  Use shell_tool to:
  1. Validate that 'image-url' block has a reachable URL (HEAD request with curl -I, 10s timeout). If 4xx/5xx, retry once after 5s.
  2. Download the image to 'agent_asset.png' with curl -L --max-time 60.
  3. If 'animation-url' exists, attempt download to 'agent_animation.bin' (keep original extension if obvious); tolerate failure (log message, continue).
  4. List resulting file sizes (ls -lh) and compute SHA256 hashes (shasum -a 256) for integrity.
  5. (macOS/Linux) Attempt to open the static image (macOS: 'open', Linux: 'xdg-open') but do not fail if command missing.
  Output a concise log (no raw binary, no verbose curl progress) into this block.
blocks:
  - image-url
  - animation-url
  - generation-log
tools:
  - shell_tool
tools-turns-max: 2
use-header: "# Download Log {id=download-log}"

@return
blocks:
  - image-url
  - animation-url
  - download-log
```

## Getting Started
```bash
pip install fractalic
fractalic init demo
cd demo
fractalic run workflow.md
```

Start simple:
```markdown
# Goal {id=goal}
Generate 5 focus areas.

@llm
prompt: "List 5 focus areas."
blocks: goal
```

## Common Uses
- Research → planning → synthesis → reporting
- Multi-model workflows (planner → executor → validator)  
- Tool calling with context (fetch → transform → reason)
- Self-refining documents with quality checks
- Executable playbooks that stay current

## Roadmap
**Current**: Multi-model, MCP, shell, agents, git sessions
**Next**: Knowledge graph memory, validators, cost tracking
**Future**: Visual editor, context compression, agent marketplace

## When Not to Use Fractalic
- Ultra-low latency microservices  
- Heavy data pipelines (use Spark/SQL)
- Real-time streaming scenarios

## License
MIT

---
**Start with one heading and one `@llm`. Let your document grow into a system.**
