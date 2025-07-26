---
description: General development and testing prompt for the Mosaic MCP Tool project
mode: agent
---

# Development and Testing Workflow

General-purpose development and testing workflow for implementing features, fixing bugs, and validating functionality in the Mosaic MCP Tool project.

## Input

Task Description: ${input:task:Define the task to achieve, including specific requirements, constraints, and success criteria}

## Workflow Steps

### 1. Intelligent Task Discovery and Context Analysis

**Decision Tree for Task Discovery:**

- **If given specific task description**: Try `search_nodes --query="${task_description}"` to find existing related tasks
- **If given vague development request**:
  1. Use `search_nodes --query="${key_terms}"` to find relevant entities
  2. If search returns empty results: Use `read_graph` to explore available tasks, decisions, and patterns
  3. Identify the most relevant existing context or determine if this is completely new work
  4. Guide user to clarify scope and relationship to existing work

**Memory MCP Context Retrieval**: Use `search_nodes` to find task entity and analyze its relationships to understand:

- Related architectural decisions and constraints ("constrained_by" relations)
- Existing patterns that can be reused ("implements" and "uses" relations)
- Dependencies on other tasks or components ("depends_on" relations)
- Research decisions that inform this work ("has_research" relations)

**Requirement Analysis**: Break down task requirements, enriched by relationship context
**Constraint Mapping**: Identify technical, business, and timeline constraints from linked decisions
**Success Criteria**: Define clear, measurable success criteria based on task acceptance criteria
**Approach Planning**: Develop step-by-step implementation approach informed by proven patterns

### 2. Relationship-Enhanced Research and Validation

- **Context-Aware Research**: Based on relationships found in step 1, focus research on gaps not covered by existing decisions
- **Pattern Reuse**: Leverage "uses" and "implements" relationships to identify proven solutions and patterns
- **Integration Analysis**: Use relationships to understand integration requirements with existing systems
- **Constraint Validation**: Verify approach aligns with architectural constraints from linked decisions
- **Risk Assessment**: Evaluate risks considering lessons learned from related work (via relationships)

### 3. Implementation Planning

- **Architecture Design**: Design component architecture and interfaces
- **Test Strategy**: Plan unit, integration, and end-to-end testing approach
- **Implementation Steps**: Create detailed implementation checklist
- **Validation Plan**: Define validation and acceptance testing procedures

### 4. Development Execution

- **Code Implementation**: Implement features following project standards
- **Test Development**: Create comprehensive test coverage
- **Documentation**: Document APIs, configurations, and usage examples
- **Code Review**: Perform self-review and prepare for peer review

### 5. Testing and Validation

- **Unit Testing**: Verify individual component functionality
- **Integration Testing**: Test component interactions and data flow
- **End-to-End Testing**: Validate complete user scenarios
- **Performance Testing**: Verify performance meets requirements

### 6. Memory MCP Documentation and Tracking

- **Progress Logging**: Update task progress in Memory MCP:

```bash
add_observations --observations=[{
    "entityName": "${task_id}",
    "contents": ["Progress: Development completed", "Test coverage: ${coverage_percentage}", "Performance: ${performance_results}"]
}]
```

- **Pattern Documentation**: Create reusable pattern entities for novel solutions:

```bash
create_entities --entities=[{
    "name": "pattern_${pattern_name}",
    "entityType": "pattern",
    "observations": ["Description: ${pattern_description}", "Implementation: ${code_example}", "Use case: ${use_case}", "Performance: ${performance_notes}"]
}]

create_relations --relations=[{
    "from": "${task_id}",
    "to": "pattern_${pattern_name}",
    "relationType": "creates"
}]
```

- **Decision Logging**: Document significant technical decisions made during development:

```bash
create_entities --entities=[{
    "name": "decision_${decision_name}",
    "entityType": "decision",
    "observations": ["Context: ${decision_context}", "Options: ${alternatives_considered}", "Choice: ${selected_approach}", "Rationale: ${reasoning}"]
}]
```

- **Knowledge Sharing**: Link new patterns and decisions to relevant existing entities for future reference

## Quality Gates

- [ ] All requirements addressed
- [ ] Code follows project standards
- [ ] Comprehensive test coverage
- [ ] Documentation updated
- [ ] Performance validated
- [ ] Security considerations addressed
- [ ] ConPort tracking updated
