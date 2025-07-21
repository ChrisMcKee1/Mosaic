# Project Sync & Audit Workflow

Performs a deep, comprehensive audit of the entire Mosaic project by comparing the documentation (the plan), the codebase (the reality), and the project memory (Conport). It then generates specific `conport` commands to correct any drift and re-align the project's recorded state.

## Chained Workflow

### 1. Load the "Planned State" (Documentation)
- **Read the PRD:** Use `desktop-commander` to `read_file` on `docs/Mosaic_MCP_Tool_PRD.md`.
- **Read the TDD:** Use `desktop-commander` to `read_file` on `docs/Mosaic_MCP_Tool_TDD.md`.
- **Read the Gap Analysis:** Use `desktop-commander` to `read_file` on `docs/CODE_INGESTION_ANALYSIS.md`.

### 2. Load the "Reality State" (Codebase)
- **Scan the entire codebase:** Use `desktop-commander` to `ls -R src/`.
- **Get recent commit history:** Use `desktop-commander` to run `git log -n 20 --pretty=oneline`.

### 3. Load the "Remembered State" (Conport)
- **Get all tasks:** Use `conport` to call `get_progress` for all statuses (TODO, IN_PROGRESS, DONE).
- **Get all decisions:** Use `conport` to call `get_decisions`.

### 4. Synthesize, Audit, and Generate Corrections
- **Perform the audit:** Use `sequential-thinking` with the following master prompt:
    "You are a meticulous project auditor for the Mosaic MCP Tool. You have been given the full PRD, TDD, a critical gap analysis document, a complete list of all source code files, the 20 most recent commits, and the entire history of tasks and decisions from our project memory, Conport.

    Your task is to perform a three-way comparison between the **plan** (docs), the **reality** (code), and the **memory** (Conport).

    1.  **Identify Inconsistencies:** Find all discrepancies. Examples to look for:
        * Is there a TODO task in Conport for a feature that already exists in the `src/` directory?
        * Is there a commit for `FR-8` but the Conport task for it is still marked `IN_PROGRESS`?
        * Does the TDD describe a function that doesn't exist in the code?
        * Is there a decision logged in Conport that contradicts the current TDD?

    2.  **Generate a "Sync Report":** Create a brief, bulleted list summarizing your findings. Highlight the most important inconsistencies that need to be addressed.

    3.  **Generate Corrective `conport` Commands:** This is the most important step. Based on your audit, provide a list of the **exact, executable `mcp__conport__...` commands** required to fix the inconsistencies and bring the Conport project memory into perfect alignment with the current reality of the codebase and documentation. For example: `mcp__conport__update_progress --id=123 --status='DONE'`, or `mcp__conport__log_decision --summary='Deprecate old function X as it was replaced by Y'`. "