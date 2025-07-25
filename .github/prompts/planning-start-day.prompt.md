---
description: Start daily development session with priority briefing
mode: agent
---

# Daily Priority Briefing & Task Start

Prepare the development session by syncing the repo and identifying the highest-priority tasks from Memory MCP. Then formally start the selected task, setting the active context for all subsequent work.

## Workflow Steps

### 1. Sync Local Environment

Use terminal to run:

```bash
git checkout main
git pull
```

### 2. Identify Critical Path Tasks

Query Memory MCP for the most important work:

- First try: `search_nodes --query="task TODO CRITICAL"` to find critical tasks
- If no critical tasks: `search_nodes --query="task TODO HIGH"` to find high-priority tasks

### 3. Plan the Day's Work

Analyze the prioritized list of tasks from Memory MCP and present them to the user.

Ask: "Based on the critical path, which task should we start today? Please provide the task ID (e.g., 'FR-006-1')."

### 4. Formally Start the Task

Once the user provides the ID, use Memory MCP to officially begin:

```
add_observations --observations=[{
    "entityName": "[user_provided_id]",
    "contents": ["Status: IN_PROGRESS", "Started: {current_date}", "Active context set"]
}]
```

This updates the task with IN_PROGRESS status and sets it as the active context.
