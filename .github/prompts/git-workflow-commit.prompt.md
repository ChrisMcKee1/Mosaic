---
description: Git commit workflow with validation and tracking
mode: agent
---

# Git Commit Workflow

Stage all changes and create a validated git commit with Memory MCP logging.

## Input

Commit Message: ${input:message:Enter the commit message}

## Workflow Steps

### 1. Pre-Commit Checks

Run the full local validation suite:

```bash
ruff check . --fix
ruff format .
pytest
```

If any step fails, stop and report the error.

### 2. Review Changes

```bash
git status
git diff --staged
```

Provide a final review of the changes before committing.

### 3. Stage Files

```bash
git add .
```

### 4. Commit

```bash
git commit -m "${message}"
```

### 5. Log to Memory MCP

After successful commit, log the activity:

```
create_entities --entities=[{
    "name": "commit_${commit_hash}",
    "entityType": "commit",
    "observations": [
        "Message: ${message}",
        "Timestamp: ${current_timestamp}",
        "Files changed: ${changed_files}",
        "Validation: All checks passed"
    ]
}]
```

## Validation Rules

- All linting issues must be resolved
- All tests must pass
- No uncommitted changes should remain
- Commit message should follow conventional commit format when possible
