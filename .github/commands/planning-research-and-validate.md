# Research and Validate Workflow with Memory MCP

Performs comprehensive research and analysis, producing a structured "Implementation Proposal" that is stored as interconnected entities in Memory MCP knowledge graph. Creates research, decision, and pattern entities with rich relationship mapping.

## Usage
`/research-and-validate <topic>`

## Arguments
- `$ARGUMENTS`: The research topic to investigate

## Chained Workflow

### 1. Create Research Session in Memory MCP

```typescript
// Create research entity for this investigation
await create_entities([
  {
    name: `research-${topic_slug}-${Date.now()}`,
    entityType: "research",
    observations: [
      `Research topic: ${$ARGUMENTS}`,
      `Started: ${new Date().toISOString()}`,
      "Status: IN_PROGRESS",
      "Type: technology_validation"
    ]
  }
]);

// Link to current development session
const activeSession = await search_nodes("entityType:session AND Status:ACTIVE");
if (activeSession.length > 0) {
  await create_relations([
    {
      from: `research-${topic_slug}-${Date.now()}`,
      to: activeSession[0].name,
      relationType: "part_of"
    }
  ]);
}
```

### 2. Research Planning with Memory MCP Context

Use `sequential-thinking` to break down the research topic from `$ARGUMENTS` into specific questions.

Query Memory MCP for related research and decisions:

```typescript
// Search for related research and decisions
await search_nodes(`entityType:research AND (${topic_keywords.join(' OR ')})`);
await search_nodes(`entityType:decision AND (${topic_keywords.join(' OR ')})`);

// Find related patterns and implementations
await search_nodes(`entityType:pattern AND (${topic_keywords.join(' OR ')})`);
```

### 3. Multi-Tool Information Gathering

**Primary Documentation:** Use `mcp_context72_get-library-docs` for core technology docs.

**Community Context:** Use `fetch_webpage` for real-world usage:
- Stack Overflow discussions
- Reddit community insights
- GitHub issue analysis
- Technical blog posts

**Store Research Findings:**

```typescript
// Update research entity with findings
await add_observations([
  {
    entityName: `research-${topic_slug}-${Date.now()}`,
    contents: [
      `Documentation analysis: ${doc_findings}`,
      `Community insights: ${community_findings}`,
      `Best practices identified: ${best_practices}`,
      `Potential issues: ${potential_issues}`
    ]
  }
]);
```

### 4. Synthesis & Knowledge Graph Enhancement

Use `sequential-thinking` to analyze all gathered information and create comprehensive proposal.

**Create Decision Entity:**

```typescript
// Create decision entity with structured proposal
await create_entities([
  {
    name: `decision-${topic_slug}-approach`,
    entityType: "decision",
    observations: [
      `Recommended approach: ${recommended_approach}`,
      `Technology choice rationale: ${rationale}`,
      `Implementation priority: ${priority}`,
      `Risk assessment: ${risk_assessment}`,
      `Success criteria: ${success_criteria}`
    ]
  }
]);

// Link decision to research
await create_relations([
  {
    from: `decision-${topic_slug}-approach`,
    to: `research-${topic_slug}-${Date.now()}`,
    relationType: "based_on"
  }
]);
```

### 5. Create Implementation Task Template

**Task Entity Creation:**

```typescript
// Create task entity based on research
await create_entities([
  {
    name: `task-${topic_slug}-implementation`,
    entityType: "task",
    observations: [
      `Title: ${proposed_task_title}`,
      `Description: ${detailed_description}`,
      `Acceptance criteria: ${acceptance_criteria}`,
      `Priority: ${recommended_priority}`,
      "Status: PLANNED",
      `Created from research: ${new Date().toISOString()}`
    ]
  }
]);

// Create comprehensive relationship network
await create_relations([
  { from: `task-${topic_slug}-implementation`, to: `decision-${topic_slug}-approach`, relationType: "guided_by" },
  { from: `task-${topic_slug}-implementation`, to: `research-${topic_slug}-${Date.now()}`, relationType: "validated_by" }
]);
```

### 6. Pattern and Best Practice Storage

**Extract Reusable Patterns:**

```typescript
// Create pattern entities for discovered approaches
await create_entities([
  {
    name: `pattern-${pattern_name}`,
    entityType: "pattern",
    observations: [
      `Pattern description: ${pattern_description}`,
      `Use cases: ${use_cases}`,
      `Implementation notes: ${implementation_notes}`,
      `Best practices: ${best_practices}`
    ]
  }
]);

// Link patterns to research and decisions
await create_relations([
  { from: `pattern-${pattern_name}`, to: `research-${topic_slug}-${Date.now()}`, relationType: "discovered_in" },
  { from: `pattern-${pattern_name}`, to: `decision-${topic_slug}-approach`, relationType: "supports" }
]);
```

### 7. Knowledge Discovery and Relationship Mapping

**Find Related Work:**

```typescript
// Search for similar implementations and decisions
await search_nodes("entityType:task AND Status:COMPLETED");

// Create relationships to related project areas
await create_relations([
  { from: `research-${topic_slug}-${Date.now()}`, to: "project-architecture", relationType: "informs" },
  { from: `decision-${topic_slug}-approach`, to: "technology-stack", relationType: "influences" }
]);
```

### 8. Complete Research Session

**Mark Research Complete:**

```typescript
// Complete research entity
await add_observations([
  {
    entityName: `research-${topic_slug}-${Date.now()}`,
    contents: [
      "Status: COMPLETED",
      `Completion time: ${new Date().toISOString()}`,
      `Decision created: decision-${topic_slug}-approach`,
      `Task template created: task-${topic_slug}-implementation`,
      `Patterns documented: ${pattern_count}`
    ]
  }
]);

// Update team knowledge base
await add_observations([
  {
    entityName: "team-knowledge",
    contents: [
      `Research completed: ${$ARGUMENTS}`,
      `Key insight: ${key_insight}`,
      `Technology recommendation: ${technology_choice}`,
      `Implementation approach: ${approach_summary}`
    ]
  }
]);
```

### 9. Handoff to User

Present the complete Implementation Proposal to the user with:

- Recommended approach summary
- Link to created Memory MCP entities
- Task template with acceptance criteria
- Related patterns and decisions
- Next steps for implementation

**Final Output:**
"Research complete and stored in Memory MCP knowledge graph. Created entities:
- Research: `research-${topic_slug}-${Date.now()}`
- Decision: `decision-${topic_slug}-approach`  
- Task Template: `task-${topic_slug}-implementation`

Use `/dev-implement task-${topic_slug}-implementation` to begin implementation based on this research."

This workflow creates a comprehensive knowledge network that captures not just the research findings, but how they connect to project goals, team decisions, and implementation strategies.
