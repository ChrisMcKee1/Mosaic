# Memory MCP Integration Guide

This guide explains how the Mosaic MCP Tool project uses Memory MCP for knowledge management, replacing the previous ConPort system.

## Memory MCP Overview

Memory MCP provides a knowledge graph-based memory system that allows persistent storage and retrieval of project information, tasks, decisions, and relationships.

## Core Concepts

### Entities

Entities are the primary nodes in the knowledge graph:

- **name**: Unique identifier
- **entityType**: Classification (task, decision, pattern, commit, etc.)
- **observations**: Array of facts/information about the entity

### Relations

Directed connections between entities in active voice:

- **from**: Source entity name
- **to**: Target entity name
- **relationType**: Relationship type (has_research, depends_on, implements, etc.)

### Observations

Discrete pieces of information attached to entities:

- Stored as strings
- Should be atomic (one fact per observation)
- Can be added, removed, or updated independently

## Entity Types Used in Mosaic

### Tasks (`entityType: "task"`)

Development tasks and work items:

```json
{
  "name": "FR-001",
  "entityType": "task",
  "observations": [
    "Title: Implement Azure OpenAI integration",
    "Description: Add Azure OpenAI service integration for chat functionality",
    "Priority: HIGH",
    "Status: IN_PROGRESS",
    "Acceptance Criteria: [...]",
    "Assigned to: developer_name",
    "Estimated effort: 3 days"
  ]
}
```

### Decisions (`entityType: "decision"`)

Architectural and technical decisions:

```json
{
  "name": "decision_azure_openai_2025_01",
  "entityType": "decision",
  "observations": [
    "Summary: Use Azure OpenAI Service for chat functionality",
    "Rationale: Better integration with existing Azure infrastructure",
    "Alternatives considered: OpenAI API, Anthropic Claude",
    "Implementation approach: Use Azure Identity SDK for authentication",
    "Impact: Improved security and cost management"
  ]
}
```

### Patterns (`entityType: "pattern"`)

Reusable code patterns and solutions:

```json
{
  "name": "pattern_azure_auth",
  "entityType": "pattern",
  "observations": [
    "Description: Standard Azure service authentication pattern",
    "Use case: Authenticating with Azure services using DefaultAzureCredential",
    "Code example: credential = DefaultAzureCredential()",
    "Dependencies: azure-identity package",
    "Best practices: Use managed identity in production"
  ]
}
```

### Commits (`entityType: "commit"`)

Git commit tracking:

```json
{
  "name": "commit_abc123",
  "entityType": "commit",
  "observations": [
    "Message: feat: add Azure OpenAI integration",
    "Hash: abc123def456",
    "Timestamp: 2025-01-24T10:30:00Z",
    "Files changed: src/azure_client.py, tests/test_azure.py",
    "Related task: FR-001"
  ]
}
```

## Common Operations

### Create a New Task

```
create_entities --entities=[{
    "name": "FR-005",
    "entityType": "task",
    "observations": [
        "Title: Implement vector search functionality",
        "Description: Add semantic search using Azure Cognitive Search",
        "Priority: HIGH",
        "Status: TODO",
        "Acceptance Criteria: Search returns relevant results with similarity scores"
    ]
}]
```

### Update Task Progress

```
add_observations --observations=[{
    "entityName": "FR-005",
    "contents": [
        "Status: IN_PROGRESS",
        "Progress: Research phase completed",
        "Next steps: Begin implementation"
    ]
}]
```

### Link Task to Decision

```
create_relations --relations=[{
    "from": "FR-005",
    "to": "decision_vector_search_2025_01",
    "relationType": "implements_decision"
}]
```

### Search for Tasks

```
search_nodes --query="task HIGH priority"
```

### Get Task Details

```
open_nodes --names=["FR-005"]
```

### Mark Task Complete

```
add_observations --observations=[{
    "entityName": "FR-005",
    "contents": [
        "Status: DONE",
        "Completed: 2025-01-24",
        "All acceptance criteria met"
    ]
}]
```

## Relationship Types

### Task Relationships

- `depends_on`: Task A depends on Task B
- `blocks`: Task A blocks Task B
- `implements_decision`: Task implements a decision
- `has_research`: Task has associated research
- `creates_pattern`: Task creates a reusable pattern

### Decision Relationships

- `informs`: Decision informs another decision
- `supersedes`: Decision replaces another decision
- `implements`: Decision implements a higher-level decision

### Pattern Relationships

- `extends`: Pattern extends another pattern
- `uses`: Pattern uses another pattern
- `replaces`: Pattern replaces another pattern

## Integration with Prompts

### Planning Start Day

- Uses `search_nodes` to find high-priority tasks
- Uses `add_observations` to mark task as started

### Implementation Workflow

- Uses `open_nodes` to get task context
- Uses `create_entities` to log research decisions
- Uses `create_relations` to link research to tasks
- Uses `add_observations` to track progress

### Project Status

- Uses `read_graph` to get complete project state
- Uses `search_nodes` to find blocked or at-risk tasks

## Best Practices

### Entity Naming

- Use consistent naming conventions
- Include timestamps for time-sensitive entities
- Use descriptive, searchable names

### Observations

- Keep observations atomic and specific
- Use consistent formatting for status updates
- Include timestamps for important events

### Relations

- Use active voice for relationship types
- Be specific about relationship meanings
- Create bidirectional relationships when needed

### Search Optimization

- Use descriptive terms in observations for better search
- Include relevant keywords in entity names
- Structure observations consistently

## Migration from ConPort

| ConPort Operation    | Memory MCP Equivalent                        |
| -------------------- | -------------------------------------------- |
| `get_tasks`          | `search_nodes --query="task"`                |
| `get_task_by_id`     | `open_nodes --names=["task_id"]`             |
| `start_task`         | `add_observations` with status               |
| `update_progress`    | `add_observations` with progress             |
| `log_decision`       | `create_entities` with entityType "decision" |
| `link_conport_items` | `create_relations`                           |
| `get_related_items`  | `read_graph` + filter relations              |
| `add_subtask`        | `create_entities` + `create_relations`       |
| `log_system_pattern` | `create_entities` with entityType "pattern"  |

## Setup and Configuration

To use Memory MCP with VS Code, add to your VS Code settings or `.vscode/mcp.json`:

```json
{
  "mcp": {
    "servers": {
      "memory": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"]
      }
    }
  }
}
```

## Querying Examples

### Find All Active Tasks

```
search_nodes --query="task IN_PROGRESS"
```

### Find High Priority Items

```
search_nodes --query="HIGH priority"
```

### Find Recent Decisions

```
search_nodes --query="decision 2025"
```

### Find Patterns Related to Azure

```
search_nodes --query="pattern azure"
```

### Get Complete Project Context

```
read_graph
```

This Memory MCP integration provides a more flexible and powerful knowledge management system compared to the previous ConPort approach, with better search capabilities and relationship modeling.
