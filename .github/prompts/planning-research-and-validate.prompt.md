---
description: Research and validate technologies before implementation
mode: agent
---

# Research and Validate Workflow

Perform comprehensive research and analysis, producing a structured "Implementation Proposal" that is logged as a decision in Memory MCP.

## Input

Research topic: ${input:topic:What technology or implementation pattern do you want to research?}

## Workflow Steps

### 1. Context-Aware Research Planning

- **Check Existing Knowledge**: Use `search_nodes` to find existing research on "${topic}" and related concepts
- **Analyze Relationships**: Review existing decision entities and their relationships to understand:
  - Previous research findings on similar topics
  - Architectural constraints that might impact this research
  - Related patterns and implementations already validated
- **Identify Research Gaps**: Based on existing knowledge graph, determine specific questions that need investigation
- **Plan Research Approach**: Design targeted research to fill identified gaps and build on existing insights

### 2. Multi-Tool Information Gathering

- **Primary Docs**: Get official documentation for the core technology
- **Specific URLs**: Fetch content from any URLs found in the primary docs
- **Community Context**: Search for real-world usage, issues, and best practices on Reddit and Stack Overflow

### 3. Synthesis & Context-Aware Proposal Generation

Analyze all gathered information alongside existing Memory MCP knowledge to create a structured proposal with:

- **Recommended Approach**: Summary of best practice, building on existing validated patterns from relationships
- **Constraint Alignment**: Verify approach aligns with architectural decisions found in step 1
- **Proposed Task Title**: Clear, concise title for a potential work item
- **Proposed Description**: Detailed description that references related existing work
- **Proposed Acceptance Criteria**: Specific, testable criteria list informed by proven patterns
- **Recommended Priority**: Suggest priority based on urgency and relationship to existing roadmap items
- **Integration Points**: Identify how this connects to existing system components based on relationships

### 4. Log Proposal as Decision with Relationships in Memory MCP

Use Memory MCP to log the decision and establish relationships:

```bash
# Create research decision entity
create_entities --entities=[{
    "name": "research_${topic}_${date}",
    "entityType": "decision",
    "observations": [
        "Summary: ${proposed_task_title}",
        "Rationale: Based on comprehensive research into ${topic}",
        "Recommended Approach: ${approach_summary}",
        "Priority: ${recommended_priority}",
        "Integration Points: ${integration_summary}",
        "Builds On: ${existing_research_summary}",
        "Implementation Details: ${full_structured_proposal}"
    ]
}]

# Link to related existing research and decisions
create_relations --relations=[{
    "from": "research_${topic}_${date}",
    "to": "${related_decision_entity}",
    "relationType": "builds_on"
}, {
    "from": "research_${topic}_${date}",
    "to": "${architectural_constraint_entity}",
    "relationType": "constrained_by"
}]
```

### 5. Handoff to User

Output the full "Implementation Proposal" and instruct: "Research complete and logged to Memory MCP. If you agree with this proposal, use the planning-create-task prompt to create a work item."
