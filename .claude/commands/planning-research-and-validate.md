---
description: Performs comprehensive research to validate technologies or implementation patterns before writing code.
argument-hint: <topic>
---
# Research and Validate Workflow

Performs comprehensive research and analysis, producing a structured "Implementation Proposal" that is logged as a decision in Conport. This proposal can then be used to create a formal work task.

## Chained Workflow

### 1. Research Planning
- Use `sequential-thinking` to break down the research topic from the user's prompt (`$ARGUMENTS`) into specific questions.

### 2. Multi-Tool Information Gathering
- **Primary Docs:** Use `context7` to `get-library-docs` for the core technology.
- **Specific URLs:** Use `WebFetch` to get content from any URLs found in the primary docs.
- **Community Context:** Use `WebSearch` to find real-world usage, issues, and best practices on sites like Reddit and Stack Overflow.

### 3. Synthesis & Proposal Generation
- Use `sequential-thinking` to analyze all gathered information. The goal is to synthesize the findings into a structured proposal with the following sections:
    - `## Recommended Approach`: A summary of the best practice.
    - `## Proposed Task Title`: A clear, concise title for a potential work item.
    - `## Proposed Description`: A detailed description for the task.
    - `## Proposed Acceptance Criteria`: A bulleted list of specific, testable criteria.
    - `## Recommended Priority`: Suggest a priority (CRITICAL, HIGH, MEDIUM, LOW) based on the importance of the topic.

### 4. Log Proposal as a Decision in Conport
- Use `conport` to call `log_decision`.
    - `--summary`: Use the "Proposed Task Title" from the previous step.
    - `--rationale`: "Based on comprehensive research into `$ARGUMENTS`."
    - `--implementation_details`: Use the full structured proposal (description, criteria, priority) as the value.

### 5. Handoff to User
- After logging the decision, output the full "Implementation Proposal" to the user and instruct them: "Research complete and logged to Conport. If you agree with this proposal, use the `/planning-create-task` command to create a work item."