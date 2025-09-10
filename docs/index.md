---
layout: home

hero:
  name: Fractalic
  text: AI Workflows in Markdown
  tagline: Plain-language programming for building AI agents using Markdown + YAML operation blocks
  actions:
    - theme: brand
      text: Get Started
      link: /quick-start
    - theme: alt
      text: Core Concepts
      link: /core-concepts
    - theme: alt
      text: View on GitHub
      link: https://github.com/fractalic-ai/fractalic

features:
  - title: Plain-Language Programming
    details: Write intent directly in Markdown. Operations are minimal YAML blocks that transform your document as they execute.
  - title: Context as First-Class Object
    details: Only referenced blocks enter model calls. Control exactly what context is used in each operation.
  - title: Deterministic Evolution
    details: Choose how content merges with append, prepend, or replace modes. Predictable document growth.
  - title: Composable Agents
    details: Any Markdown file with @return becomes a reusable agent. Chain and combine workflows effortlessly.
  - title: Progressive Elaboration
    details: Start simple and grow structure without refactoring. Add complexity only when needed.
  - title: Transparent Execution
    details: See exactly how your document evolves. Full execution traceability with git-backed sessions.
---

## What is Fractalic?

Fractalic is a plain-language programming environment for building AI workflows and agents using ordinary Markdown plus lightweight YAML operation blocks. Instead of wiring nodes in a UI or writing imperative Python scripts, you express intent in structured text.

**Key idea:** A Fractalic file = (Ordered Knowledge Blocks) + (Operation Blocks that transform context) → Final Returned Result.

## Quick Example

```markdown
# Product Requirements {id=reqs}
- Fast response times
- Easy to use interface
- Reliable performance

@llm
prompt: "Create a technical specification based on these requirements"
blocks: reqs
use-header: "# Technical Specification"
mode: append
```

**After execution, your document grows:**

```js
# Product Requirements {id=reqs}
- Fast response times
- Easy to use interface
- Reliable performance

@llm
prompt: "Create a technical specification based on these requirements"
blocks: reqs
use-header: "# Technical Specification"
mode: append

# Technical Specification {id=technical-specification} // [!code highlight]
## Performance Requirements // [!code highlight]
- Response time: < 200ms for all user interactions // [!code highlight]
- System uptime: 99.9% availability target // [!code highlight]
- Concurrent users: Support up to 10,000 simultaneous connections // [!code highlight]
 // [!code highlight]
## User Interface Specifications // [!code highlight]
- Responsive design for mobile and desktop // [!code highlight]
- Intuitive navigation with maximum 3-click access to any feature // [!code highlight]
- Accessibility compliance (WCAG 2.1 AA standards) // [!code highlight]
 // [!code highlight]
## Reliability Standards // [!code highlight]
- Automated failover mechanisms // [!code highlight]
- Data backup and recovery procedures // [!code highlight]
- Comprehensive error handling and logging // [!code highlight]
```

The document evolves as operations run—new blocks are appended, replaced, or refined—so you can literally "grow" an AI system the way you draft a document. 

## Core Benefits

- **Transparency**: Diffable evolution with full execution history
- **Composability**: Any file can be an agent, workflows chain naturally  
- **Deterministic**: Controlled merge semantics and context windows
- **Integration**: Seamless shell and tool integration via MCP
- **Rapid Development**: Progressive elaboration without refactoring

## Getting Started

Ready to build your first AI workflow? Check out the [Quick Start guide](/quick-start) or dive into [Core Concepts](/core-concepts) to understand the fundamentals.
