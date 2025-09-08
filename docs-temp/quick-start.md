---
title: Quick Start
description: Get Fractalic running and execute your first minimal workflow in under 5 minutes
outline: deep
---

# Quick Start

## Goal
Get Fractalic running and execute your first minimal workflow in <5 minutes.

## 2.1 Deployment Options
Summaries (see README for full commands):
- Docker (recommended) – one container launches UI + API + AI server + MCP manager
- Codespaces – zero‑install browser dev
- Local Dev – clone both backend + UI repos

## 2.2 Prerequisites (Local Dev)
- Python 3.11+
- Node.js (for UI if using separate repo)
- Git (sessions persistence currently relies on git)

## 2.3 Install (Backend Only)
```bash
git clone https://github.com/fractalic-ai/fractalic.git
cd fractalic
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./run_server.sh
```
Backend default ports: API 8000, AI 8001.

## 2.4 UI Setup (Separate Repo)
```bash
git clone https://github.com/fractalic-ai/fractalic-ui.git
cd fractalic-ui
npm install
npm run dev
```
UI default: `http://localhost:3000`

## 2.5 Minimal Hello World Workflow
Create `hello.md`:
```markdown
# Greeting {id=greeting}
We want a short friendly greeting.

@llm
prompt: "Write a 1-line friendly greeting referencing our initiative."
block:
  block_uri: greeting
```
Run via UI (open file) or headless (future CLI docs TBD). Result: new LLM response block under the operation.

## 2.6 First Workflow with Shell Verification
```markdown
# Task {id=task}
Generate a small Python script that prints 'Hello Fractalic'.

@llm
prompt: "Write python code only (no prose) to print 'Hello Fractalic'"
block:
  block_uri: task
use-header: "# Script"

@shell
prompt: |
  python - <<'PY'
  print('Hello Fractalic')
  PY
use-header: "# Run Output"

@llm
prompt: "Confirm the output was correct and respond DONE."
block:
  block_uri: run-output/*
use-header: "# Verification"
```

## 2.7 Understanding the Result
- `# Task` is a knowledge block
- First `@llm` writes `# Script`
- `@shell` executes code (here inline heredoc) → `# Run Output`
- Final `@llm` inspects prior output → `# Verification`

## 2.8 Common Setup Issues
| Symptom | Cause | Fix |
|---------|-------|-----|
| Model auth errors | Missing API key | Add key to `settings.toml` or env var |
| No settings file | First UI boot not completed | Launch UI once or copy sample manually |
| Port already in use | Existing process | Free port / change config |

## 2.9 Next Steps
Proceed to Core Concepts for deeper foundational understanding.
