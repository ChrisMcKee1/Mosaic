---
description: Complete project overview and strategic planning
mode: agent
---

# Project Status and Planning Workflow

Comprehensive project overview with strategic planning and priority assessment.

## Workflow Steps

### 1. Memory MCP Review and Analysis

- **Task Status Review**: Use `search_nodes --query="task"` to find all task entities and their current status
- **Priority Assessment**: Analyze task priorities and dependencies using `read_graph`
- **Progress Analysis**: Review completed tasks and velocity metrics from task observations
- **Blockers Identification**: Search for blocked or at-risk tasks using `search_nodes --query="blocked OR risk"`

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
- **Memory MCP Updates**: Update Memory MCP with new priorities and plans:

```
add_observations --observations=[{
    "entityName": "project_status_${date}",
    "contents": ["Priority updates: ${updates}", "Resource allocation: ${allocation}", "Next actions: ${actions}"]
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
