# Task-Driven Implementation Workflow (with Integrated Research)

Executes a specific development task from Conport. This comprehensive workflow begins by fetching the task's full context, proceeds through a mandatory research and validation phase, guides implementation against acceptance criteria, and concludes with detailed logging and linking in Conport.

## Usage
`/dev-implement <task_id> <brief description of what you're doing>`
Example: `/dev-implement FR-6-1 Implement the GitPython cloning logic`

## Chained Workflow

### 1. Fetch Comprehensive Task Context from Conport
- **Identify Task:** The first argument is the task ID (e.g., "FR-6-1").
- **Get Task Details:** Use `conport` to call `get_task_by_id --id="[task_id]"`. This retrieves the title, description, and acceptance criteria.
- **Get Related Decisions & Notes:** Use `conport` to call `get_related_items --id="[task_id]"`. This retrieves all linked architectural decisions and notes.

### 2. Comprehensive Research & Validation
- **Formulate Research Questions:** Use `sequential-thinking` to analyze the task description and identify the core technologies, libraries, or patterns that require validation (e.g., "What is the latest stable version of GitPython?", "What is the 2025 best practice for handling authentication with it?").
- **Primary Documentation Research (Context7):** Use the **`mcp__context7__get-library-docs`** tool to retrieve official documentation and API references for the core technologies identified.
- **Direct Source Retrieval (WebFetch):** If the documentation references specific articles, blog posts, or GitHub URLs, use the **`WebFetch`** tool to retrieve the full content of those pages.
- **Broader Community Validation (WebSearch):** Use the **`WebSearch`** tool to find real-world context and community sentiment. Good search queries include:
    - "`<topic>` vs `<alternative>` reddit"
    - "`<topic>` common issues stack overflow"
    - "`<topic>` tutorial 2025"
- **Synthesize and Log Findings:** Use `sequential-thinking` to analyze all gathered information and formulate a final recommendation. Then, use `conport` to `log_decision` with this recommendation and `link_conport_items` to link this new decision back to the original task ID.

### 3. Pre-Implementation Planning
- **Create a Checklist:** Use `sequential-thinking` to parse the `acceptance_criteria` from the task details (fetched in Step 1) and create a specific markdown checklist.
- **Review Constraints:** Review all linked architectural decisions from Conport, including the new ones just logged from the research step.
- **Formulate the Plan:** Based on the goal, checklist, and constraints, generate a step-by-step implementation plan and present it for confirmation.

### 4. Implementation with Desktop Commander
- Use `desktop-commander` to perform the file system and code modifications as outlined in the confirmed implementation plan. This includes writing/editing code and running tests.

### 5. Continuous Progress Updates
- As each step in the plan is completed, use `conport` to update or create sub-tasks: `update_progress` or `add_subtask --parent_id="[task_id]"`.
- If a reusable code pattern is created, log it to project memory using `conport`: `log_system_pattern --name="..." --description="..."`.

### 6. Final Validation & Conport Updates
- **Validate Against Checklist:** Use `sequential-thinking` to review the implemented code against the acceptance criteria checklist created in the planning phase.
- **Mark Task Complete:** Once all criteria are met, use `conport` to `update_progress --id="[task_id]" --status='DONE'`.
- **Log Final Decision:** Use `conport` to `log_decision` summarizing the final implementation approach.
- **Link Everything:** Use `conport` to `link_conport_items` to create explicit links between the completed task, the final decision, and any new system patterns that were logged.