---
title: Git-Backed Sessions
description: Persist and version your Fractalic workflows using Git for collaborative development
outline: deep
---

# Git-Backed Sessions

## Purpose
Fractalic automatically versions every workflow execution using git. When you run operations on Markdown documents, Fractalic commits the results, creating an auditable history of document evolution and execution state.

## Internal Table of Contents
- [Purpose](#purpose)
- [Internal Table of Contents](#internal-table-of-contents)
- [Automatic Repository Setup](#automatic-repository-setup)
- [What Gets Committed Automatically](#what-gets-committed-automatically)
- [File Types and Their Roles](#file-types-and-their-roles)
- [Commit Messages and Metadata](#commit-messages-and-metadata)
- [Session Branches](#session-branches)
- [Reviewing Session History](#reviewing-session-history)
- [See Also](#see-also)

## Automatic Repository Setup
If no git repository exists in your working directory, Fractalic automatically:
1. Initializes a new git repository
2. Configures git settings for deterministic operation:
   - `user.email`: "fractalic_core_agent@fractalic_core_agent.com"
   - `user.name`: "fractalic process"  
   - `core.autocrlf`: false
   - `core.safecrlf`: false
   - `core.ignorecase`: false
3. Creates a minimal `.gitignore` file with basic patterns (.DS_Store, *.pyc, __pycache__/, .idea/, .vscode/)
4. Makes an initial commit with the .gitignore

This ensures every Fractalic execution occurs within a versioned environment with consistent git settings.

## What Gets Committed Automatically
Fractalic commits files automatically at key execution points:

**After every operation execution:**
- Source `.md` files (your original documents)
- Generated `.ctx` files (resolved execution context)
- Generated `.trc` files (detailed execution traces)

**At session completion:**
- `call_tree.json` (execution dependency graph)

**Exception handling:**
- Even when operations fail, `.ctx` and `.trc` files are committed with error details

## File Types and Their Roles
- **`.md` files**: Your source documents containing operation blocks
- **`.ctx` files**: Show the complete resolved state after operations execute (what the AI actually saw)
- **`.trc` files**: Detailed execution logs (timing, parameters, intermediate states)
- **`call_tree.json`**: Maps relationships between operations and their execution order

All these files are committed together to maintain execution provenance.

## Commit Messages and Metadata
Fractalic generates commit messages automatically:
- Operation-based: "@return operation", "Final processed files"
- Error handling: "Exception caught: appended traceback"
- Session completion: "Saving call_tree.json with execution state and any pending files"

Each commit can include metadata about trigger files and parent operations for traceability.

## Session Branches
Session branches are created by default for each execution using the format:
```
`<YYYYMMDDHHMMSS>_<hash8>_<sanitized-task-name>`
```
Example: `20250908142310_a1b2c3d4_Testing-git-operations`

Branch names include:
- Timestamp to the second
- 8-character hash to avoid collisions  
- Sanitized task name (only letters, numbers, dashes, underscores)

Note: When Fractalic tools execute other Fractalic files internally (like the `fractalic_run` MCP tool), those executions reuse the current branch rather than creating new ones.

## Reviewing Session History
Use standard git tools to review sessions:
```bash
git log --oneline              # See commit sequence
git show `<commit-hash>`         # View specific changes
git diff HEAD~1 file.ctx       # Compare execution states
```

Since all artifacts are text files, git diffs remain human-readable.

The committed data is also used by Fractalic IDE to provide session analytics and compare `.md` source context versus `.ctx` resulting context, enabling visual analysis of how operations transformed the documents.

## See Also
- Configuration: Git repository settings and paths
- Operations Reference: How individual operations trigger commits
- Context Management: How execution context is built and preserved

---
Focus: Understanding automatic git versioning behavior in Fractalic.
