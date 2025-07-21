# Daily Priority Briefing & Task Start Workflow

Prepares the development session by syncing the repo and identifying the highest-priority tasks from Conport. It then formally starts the selected task, setting the active context for all subsequent work.

## Chained Workflow

### 1. Sync Local Environment
- Use `desktop-commander` to run `git checkout main` and `git pull`.

### 2. Identify Critical Path Tasks
- Use `conport` to query for the most important work: `get_tasks --query="status=='TODO' and priority=='CRITICAL'"`
- If no critical tasks are found, query for high-priority tasks: `get_tasks --query="status=='TODO' and priority=='HIGH'"`

### 3. Plan the Day's Work
- Use `sequential-thinking` to analyze the prioritized list of tasks from Conport.
- Present the tasks to the user and ask: "Based on the critical path, which task should we start today? Please provide the task ID (e.g., 'FR-006-1')."

### 4. Formally Start the Task
- Once the user provides the ID, use the `conport` tool to officially begin the task: `start_task --id="[user_provided_id]"`
- This command automatically updates the task status to `IN_PROGRESS` and sets it as the active context in Conport.