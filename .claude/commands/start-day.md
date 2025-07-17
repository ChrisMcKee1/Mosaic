# Daily Start Workflow

Prepare the development environment by syncing the repository and fetching the current project status from Conport. This is the recommended "golden path" to begin any work session.

## Chained Workflow
1.  **Sync Repository:** Use `desktop-commander` to run `git checkout main`, `git pull`, and `git status` to ensure your local environment is up-to-date with the remote `main` branch.
2.  **Fetch Project Status:** Use `conport` to call `get_progress` with `status=TODO` to review all open tasks and priorities.
3.  **Fetch Active Context:** Use `conport` to call `get_active_context` to see what the last focus of work was.
4.  **Summarize & Plan:** Use `sequential-thinking` to analyze the outputs from the previous steps. Provide a concise summary of the project's current state, list the open tasks, and ask "What is our main priority today?".