---
description: Research and validate technologies before implementation
mode: agent
---

# Research and Validate Workflow

Perform comprehensive research and analysis, producing a structured "Implementation Proposal" that is logged as a decision in Memory MCP.

## Input

Research topic: ${input:topic:What technology or implementation pattern do you want to research?}

## Workflow Steps

### 1. Research Planning

Break down the research topic "${topic}" into specific questions to investigate.

### 2. Multi-Tool Information Gathering

- **Primary Docs**: Get official documentation for the core technology
- **Specific URLs**: Fetch content from any URLs found in the primary docs
- **Community Context**: Search for real-world usage, issues, and best practices on Reddit and Stack Overflow

### 3. Synthesis & Proposal Generation

Analyze all gathered information and create a structured proposal with:

- **Recommended Approach**: Summary of the best practice
- **Proposed Task Title**: Clear, concise title for a potential work item
- **Proposed Description**: Detailed description for the task
- **Proposed Acceptance Criteria**: Specific, testable criteria list
- **Recommended Priority**: Suggest priority (CRITICAL, HIGH, MEDIUM, LOW)

### 4. Log Proposal as Decision in Memory MCP

Use Memory MCP to log the decision:

```
create_entities --entities=[{
    "name": "research_${topic}_${date}",
    "entityType": "decision",
    "observations": [
        "Summary: ${proposed_task_title}",
        "Rationale: Based on comprehensive research into ${topic}",
        "Recommended Approach: ${approach_summary}",
        "Priority: ${recommended_priority}",
        "Implementation Details: ${full_structured_proposal}"
    ]
}]
```

### 5. Handoff to User

Output the full "Implementation Proposal" and instruct: "Research complete and logged to Memory MCP. If you agree with this proposal, use the planning-create-task prompt to create a work item."
