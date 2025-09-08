# Fractalic Documentation

This directory contains the Fractalic documentation structured for VitePress.

## Development

### Prerequisites
- Node.js 16+
- npm or yarn

### Setup
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

### Build
```bash
# Build static site
npm run build

# Preview built site
npm run preview
```

## Structure

- `index.md` - Homepage with hero layout
- `.vitepress/config.js` - VitePress configuration
- `*.md` - Documentation pages organized by topic

## Navigation

The documentation is organized into logical sections:
- **Getting Started**: Introduction, Quick Start, Core Concepts
- **Reference**: Syntax, Operations, Configuration
- **Advanced Features**: Advanced LLM features, Context Management, etc.
- **Integration & Workflows**: Shell integration, Agents, MCP, Git sessions
- **Deployment & API**: UI Server and Publisher documentation

## Content Guidelines

This documentation follows the Fractalic documentation authoring constraints defined in `.github/instructions/docs.instructions.md`. Key principles:

- Plain, precise, neutral tone
- Present tense, active voice, second person
- Structured with clear headings and sections
- Consistent YAML examples using `|` for multi-line content
- Cross-references to related documentation

## Deployment

The documentation can be deployed to any static hosting service that supports Node.js builds:
- Vercel
- Netlify
- GitHub Pages
- etc.

Make sure to configure the base URL in `.vitepress/config.js` if deploying to a subdirectory.
