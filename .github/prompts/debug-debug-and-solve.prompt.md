---
description: Systematic debugging and problem-solving workflow
mode: agent
---

# Debug and Solve Workflow

Systematic problem-solving approach with full tracking and documentation.

## Input

Problem Description: ${input:problem:Describe the problem you're experiencing}

## Workflow Steps

### 1. Problem Analysis

- **Gather Information**: Collect error messages, logs, and reproduction steps
- **Context Review**: Examine recent changes, environment state, and related systems
- **Scope Definition**: Determine if this is a local, environment, or systemic issue

### 2. Investigation Phase

- **Log Analysis**: Review application logs and system logs for error patterns
- **Environment Check**: Verify environment variables, service states, and connectivity
- **Code Review**: Examine recent changes that might have introduced the issue
- **Dependency Check**: Verify all dependencies are properly installed and configured

### 3. Research and Validation

- **Known Issues**: Search for similar issues in documentation, GitHub issues, Stack Overflow
- **Azure Status**: Check Azure service health if cloud services are involved
- **Version Compatibility**: Verify all components are using compatible versions

### 4. Solution Development

- **Hypothesis Formation**: Develop theories about the root cause
- **Solution Planning**: Create step-by-step approach to resolve the issue
- **Risk Assessment**: Evaluate potential impact of proposed solutions
- **Backup Plan**: Prepare rollback strategy if solutions cause new issues

### 5. Implementation and Testing

- **Apply Solution**: Implement the proposed fix systematically
- **Validation Testing**: Verify the fix resolves the original problem
- **Regression Testing**: Ensure no new issues were introduced
- **Documentation**: Record the solution and prevention measures

### 6. Memory MCP Documentation

- **Log Decision**: Document the problem analysis and solution in Memory MCP:

```
create_entities --entities=[{
    "name": "debug_${problem_id}_${date}",
    "entityType": "decision",
    "observations": [
        "Problem: ${problem_description}",
        "Root cause: ${identified_cause}",
        "Solution: ${implemented_solution}",
        "Prevention: ${prevention_measures}"
    ]
}]
```

- **Link Related Items**: Connect to any related tasks or decisions using `create_relations`
- **Update Knowledge Base**: Add solution pattern entity for future reference
- **Post-Mortem**: If significant, create detailed post-mortem analysis entity

## Common Problem Categories

- **Azure Service Issues**: Authentication, networking, service availability
- **MCP Protocol Issues**: JSON-RPC errors, tool registration, communication
- **Python Environment**: Package conflicts, version mismatches, virtual environment
- **Git/GitHub Issues**: Branch conflicts, authentication, workflow failures
