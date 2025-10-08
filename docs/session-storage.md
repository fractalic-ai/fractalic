---
title: Session Storage and Versioning
description: Understand how Fractalic preserves execution history with isolated session storage
outline: deep
---

# Session Storage and Versioning

## Purpose
Fractalic automatically preserves every workflow execution in isolated session storage. Each run creates a complete snapshot of your workflow, artifacts, and execution metadata, enabling you to review, compare, and understand how your documents evolved.

## Internal Table of Contents
- [Purpose](#purpose)
- [Internal Table of Contents](#internal-table-of-contents)
- [How Session Storage Works](#how-session-storage-works)
  - [Directory Structure](#directory-structure)
  - [Session Lifecycle](#session-lifecycle)
  - [Artifact Organization](#artifact-organization)
  - [Metadata Schema](#metadata-schema)
- [File Types and Their Roles](#file-types-and-their-roles)
- [Reviewing Session History](#reviewing-session-history)
- [See Also](#see-also)

## How Session Storage Works

Fractalic creates isolated session environments within `.fractalic/sessions/` directories. Each execution gets a unique session ID and maintains its own workspace, artifacts, and metadata.

### Directory Structure

```
project_root/
├── .fractalic/
│   └── sessions/
│       └── {execution_id}/
│           ├── workspace/              # Isolated execution environment
│           │   ├── source-file.md      # Snapshot of source file
│           │   └── other-file.md       # Other execution files
│           ├── artifacts/
│           │   ├── nodes/
│           │   │   ├── {node_id_1}/    # First operation node
│           │   │   │   ├── source-file.ctx
│           │   │   │   ├── source-file.trc
│           │   │   │   └── metadata.json
│           │   │   └── {node_id_2}/    # Second operation node
│           │   │       ├── child-agent.ctx
│           │   │       ├── child-agent.trc
│           │   │       └── metadata.json
│           │   └── call_tree.json      # Execution dependency graph
│           └── session.json            # Session metadata
```

**Key Design Principles:**

1. **Session Isolation**: Each execution has its own workspace directory with snapshots of source files
2. **Node-based Artifacts**: Each call tree node gets a unique directory preventing filename conflicts
3. **Original Filenames Preserved**: Context and trace files retain source filenames (e.g., `parent-orchestrator.ctx`)
4. **Hierarchical Metadata**: Both session-level and node-level metadata tracked separately

### Session Lifecycle

**1. Session Creation**

When you run a Fractalic workflow, the system:
- Generates a unique `execution_id` (UUID v4)
- Creates `.fractalic/sessions/{execution_id}/` directory structure
- Initializes `workspace/` and `artifacts/` subdirectories
- Creates `session.json` with metadata

**2. Workspace Preparation**

Before execution:
- Source files are copied to `workspace/` directory
- Fractalic changes working directory to workspace
- All operations execute within isolated environment

**3. Artifact Saving**

After each operation:
- Node artifacts saved to `artifacts/nodes/{node_id}/`
- Context file (`.ctx`) shows complete resolved state
- Trace file (`.trc`) contains execution logs and timing
- `metadata.json` records operation parameters and status

**4. Session Completion**

At workflow end:
- Call tree saved to `artifacts/call_tree.json`
- Session metadata updated with completion status
- Session available for review in Fractalic UI

### Artifact Organization

**Node Artifacts** (`artifacts/nodes/{node_id}/`)

Each call tree node corresponds to one operation execution and contains:

| File | Purpose | Content |
|------|---------|---------|
| `{original_name}.ctx` | Resolved context | Complete document state after operation execution |
| `{original_name}.trc` | Execution trace | Timing, parameters, intermediate states, error details |
| `metadata.json` | Node metadata | Operation type, parent relationships, timestamps |

**Example node metadata:**
```json
{
  "node_id": "baf0e629-fdb",
  "execution_id": "75e42199-f589-4887-8527-91e57eee4dec",
  "operation": "llm",
  "file_path": "parent-orchestrator.md",
  "parent_node_id": null,
  "timestamp": "2025-01-08T14:23:10.123456",
  "status": "completed"
}
```

**Call Tree** (`artifacts/call_tree.json`)

Represents the complete execution graph:

```json
{
  "execution_id": "75e42199-f589-4887-8527-91e57eee4dec",
  "root_file": "parent-orchestrator.md",
  "nodes": [
    {
      "node_id": "baf0e629-fdb",
      "file": "parent-orchestrator.md",
      "operation": "llm",
      "parent_node_id": null,
      "children": ["dd319162-740", "ff892a41-3ce"]
    },
    {
      "node_id": "dd319162-740",
      "file": "child-agent.md",
      "operation": "run",
      "parent_node_id": "baf0e629-fdb",
      "children": []
    }
  ]
}
```

### Metadata Schema

**Session Metadata** (`session.json`)

```json
{
  "execution_id": "75e42199-f589-4887-8527-91e57eee4dec",
  "created_at": "2025-01-08T14:23:10.123456",
  "completed_at": "2025-01-08T14:25:42.789012",
  "status": "completed",
  "root_file": "parent-orchestrator.md",
  "working_directory": "/Users/user/project",
  "session_root": "/Users/user/project",
  "workspace_path": ".fractalic/sessions/75e42199-f589-4887-8527-91e57eee4dec/workspace"
}
```

**Node Metadata** (`artifacts/nodes/{node_id}/metadata.json`)

```json
{
  "node_id": "baf0e629-fdb",
  "execution_id": "75e42199-f589-4887-8527-91e57eee4dec",
  "operation": "llm",
  "file_path": "parent-orchestrator.md",
  "parent_node_id": null,
  "timestamp": "2025-01-08T14:23:10.123456",
  "status": "completed",
  "artifacts": {
    "ctx": "artifacts/nodes/baf0e629-fdb/parent-orchestrator.ctx",
    "trc": "artifacts/nodes/baf0e629-fdb/parent-orchestrator.trc"
  },
  "operation_params": {
    "prompt": "Based on the two analyses above...",
    "blocks": ["research-data", "methodology"]
  }
}
```

## File Types and Their Roles

| File Type | Purpose | Content |
|-----------|---------|---------|
| `.md` | Source documents | Original workflow files with operation blocks |
| `.ctx` | Resolved context | Complete document state after operations execute (what AI saw) |
| `.trc` | Execution trace | Timing, parameters, intermediate states, error details |
| `call_tree.json` | Execution graph | Maps relationships between operations and execution order |
| `session.json` | Session metadata | Execution status, timestamps, file paths |
| `metadata.json` | Node metadata | Per-operation metadata including parameters and status |

All files are plain text (JSON or Markdown) for human readability and diffing.

## Reviewing Session History

**Using the Fractalic UI:**

The Fractalic UI provides visual session browsing with:
- Session list showing execution timestamps and root files
- Hierarchical view of call trees showing parent-child relationships
- Visual diff view comparing `.md` source vs `.ctx` resolved context
- Execution trace viewer for debugging and analysis

**Using the API:**

List all sessions for a project:
```bash
curl http://localhost:8000/branches_and_commits/?repo_path=/path/to/project
```

**Using the filesystem:**

View session artifacts directly:
```bash
# List all sessions
ls -la project/.fractalic/sessions/

# View specific session
ls -la project/.fractalic/sessions/{execution_id}/artifacts/nodes/

# Compare context files between sessions
diff project/.fractalic/sessions/{id1}/artifacts/nodes/{node}/file.ctx \
     project/.fractalic/sessions/{id2}/artifacts/nodes/{node}/file.ctx
```

**Understanding Context Files:**

Context (`.ctx`) files show exactly what content was assembled and sent to each operation. This is invaluable for:
- Understanding why an AI produced specific output
- Debugging unexpected behavior
- Optimizing token usage by seeing actual context size
- Learning how block selection affects results

Trace (`.trc`) files provide detailed execution logs including:
- Operation parameters
- Execution timing
- Error messages and stack traces
- Token usage statistics

## See Also
- [Core Concepts](./core-concepts.md) - Understanding blocks and operations
- [Operations Reference](./operations-reference.md) - How operations trigger artifact saves
- [Context Management](./context-management.md) - How execution context is built and preserved

---
Focus: Understanding session storage and artifact organization in Fractalic.
