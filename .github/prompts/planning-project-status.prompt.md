---
description: Complete project overview and strategic planning
mode: agent
---

# Project Status and Planning Workflow

Comprehensive project overview with strategic planning and priority assessment.

## Workflow Steps

### 1. Comprehensive Memory MCP Analysis

- **Task Ecosystem Review**: Use `search_nodes --query="task"` to find all task entities, then traverse relationships to understand dependencies
- **Decision Chain Analysis**: Use `search_nodes --query="decision"` and follow relationships to map decision impacts and architectural constraints
- **Pattern Utilization Review**: Search for pattern entities and analyze "implements" relationships to assess reuse and standardization
- **Progress Tracking**: Review completed tasks and their relationships to current active work
- **Blocker Dependencies**: Use relationships to trace blocked tasks back to their root causes and dependency chains
- **Research Utilization**: Review research decisions and their "builds_on" relationships to ensure findings are being applied

### 2. Technical Debt and Architecture Review

- **Code Quality Metrics**: Review test coverage, code quality indicators
- **Architecture Decisions**: Review recent architectural decisions and their impact
- **Dependency Updates**: Check for outdated dependencies and security issues
- **Performance Metrics**: Review application performance and resource utilization

### 3. Technology and Market Updates

- **Azure Service Updates**: Check for new Azure service capabilities that could benefit the project
- **Semantic Kernel Updates**: Review SK updates and new features
- **MCP Specification Updates**: Check for MCP protocol updates and improvements
- **Security Updates**: Review security advisories and compliance requirements

### 4. Strategic Planning

- **Goal Alignment**: Verify current work aligns with project objectives
- **Resource Planning**: Assess resource allocation and capacity planning
- **Risk Assessment**: Identify project risks and mitigation strategies
- **Milestone Planning**: Review upcoming milestones and delivery commitments

### 5. Stakeholder Communication

- **Status Summary**: Prepare comprehensive status summary
- **Next Steps**: Define clear next steps and priorities
- **Dependency Resolution**: Plan resolution of any external dependencies
- **Communication Plan**: Plan stakeholder updates and demo schedules

### 6. Action Items and Follow-up

- **Priority Queue**: Update task priorities based on analysis using `add_observations`
- **Resource Allocation**: Assign resources to highest-priority items
- **Memory MCP Strategic Updates**: Update Memory MCP with new priorities, patterns, and relationships:

```bash
# Create status review decision
create_entities --entities=[{
    "name": "project_status_${date}",
    "entityType": "decision",
    "observations": ["Priority updates: ${updates}", "Resource allocation: ${allocation}", "Next actions: ${actions}", "Relationship insights: ${relationship_findings}"]
}]

# Link status to critical decisions and patterns
create_relations --relations=[{
    "from": "project_status_${date}",
    "to": "${critical_decision_entity}",
    "relationType": "impacts"
}, {
    "from": "project_status_${date}",
    "to": "${priority_task_entity}",
    "relationType": "prioritizes"
}]
```

- **Documentation**: Update project documentation and roadmaps

## Status Report Structure

- **Executive Summary**: High-level project status and key metrics
- **Completed Work**: Major accomplishments since last review
- **Current Focus**: Active work streams and priorities
- **Upcoming Milestones**: Key deliverables and timelines
- **Risks and Issues**: Current risks and mitigation plans
- **Resource Needs**: Any resource or dependency requirements
- **Strategic Recommendations**: Suggestions for project direction
