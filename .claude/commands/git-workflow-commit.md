# Git Commit Workflow

Stages all changes and creates a validated git commit. Conport logging is now handled automatically by a `PostToolUse` hook in `settings.json`.

## Usage
`/git-commit-workflow <commit message>`

## Argument
- `$ARGUMENTS`: The commit message (required).

## Chained Workflow
1.  **Pre-Commit Checks:** Use `desktop-commander` to run the full local validation suite: `ruff check . --fix`, `ruff format .`, and `pytest`. If any step fails, stop and report the error.
2.  **Review Changes:** Use `desktop-commander` to run `git status` and `git diff --staged` so you can provide a final review.
3.  **Stage Files:** Use `desktop-commander` to run `git add .`.
4.  **Commit:** Use `desktop-commander` to run `git commit -m "$ARGUMENTS"`. The automatic hook will handle logging this to Conport.