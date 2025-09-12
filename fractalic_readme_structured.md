[![PyPI Version](https://img.shields.io/pypi/v/fractalic.svg)](https://pypi.org/project/fractalic/) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt) ![Status](https://img.shields.io/badge/status-early--adopter-orange) ![Python](https://img.shields.io/badge/python-3.11+-blue.svg) [![Docs](https://img.shields.io/badge/docs-reference-purple)](docs/)

<p align="center">
  <img src="https://raw.githubusercontent.com/fractalic-ai/fractalic/main/docs/images/fractalic_hero.png" alt="Fractalic Hero Image">
</p>

# Fractalic
Design, run and evolve multi‑model AI workflows in one executable Markdown file—no glue code—with precise context control, tool integration and git‑traceable reproducibility.

## How

**One executable Markdown file**: Your workflow specification *is* your runtime. Write what you want in plain Markdown, run it directly. No translation between documentation and code.

**No glue code**: Replace Python/JS/(any program language) orchestration scripts with 3-6 line YAML plain-text operations. 

**Multi-model workflows**: Switch between LLM models and providers in the same document. 

**Precise context control**: Your Markdown becomes an addressable tree. Reference exact sections, branches, or lists. LLMs see only what you specify—no hidden prompt stuffing.

**Tool integration**: Connect MCP servers, Python functions, and shell commands. All outputs flow back into your document structure for the next operation.

**Git-traceable reproducibility**: Every run produces a `.ctx` file showing exactly what changed. Diff your workflow evolution semantically, not just text changes.

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
block: { block_uri: brief }
mode: append

@shell
command: curl api.example.com/trends > trends.json

@llm
prompt: "Draft 150-word narrative using pains + trends."
block: { block_uri: brief/* }
use-header: "# Draft"

@return
block: { block_uri: draft }
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
block: { block_uri: goal }
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
