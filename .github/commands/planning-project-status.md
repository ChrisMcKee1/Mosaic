# Project Status Overview with Memory MCP

Provides comprehensive project overview and strategic planning using Memory MCP knowledge graph analysis. Generates insights from development velocity, task completion patterns, team collaboration metrics, and technical debt tracking.

## Usage
`/project-status`

## Chained Workflow

### 1. Read Complete Memory MCP Knowledge Graph

```typescript
// Get complete project knowledge graph
const projectGraph = await read_graph();

// Analyze graph structure and relationships
const graphAnalysis = analyzeGraphStructure(projectGraph);
```

### 2. Query Key Project Entities

**Current Project State:**

```typescript
// Get active tasks and their progress
const activeTasks = await search_nodes("entityType:task AND (Status:IN_PROGRESS OR Status:TODO)");

// Get recent milestones and achievements  
const recentMilestones = await search_nodes("entityType:milestone AND Status:COMPLETED");

// Get current development sessions
const activeSessions = await search_nodes("entityType:session AND Status:ACTIVE");

// Get recent decisions and their impact
const recentDecisions = await search_nodes("entityType:decision");
```

**Technical Health Assessment:**

```typescript
// Search for patterns and technical debt
const codePatterns = await search_nodes("entityType:pattern AND type:code");
const technicalDebt = await search_nodes("entityType:issue AND type:technical_debt");

// Get deployment and infrastructure status
const deploymentStatus = await search_nodes("entityType:milestone AND type:deployment");
const infrastructureDecisions = await search_nodes("entityType:decision AND (infrastructure OR deployment)");
```

### 3. Analyze Development Velocity and Trends

Use `sequential-thinking` to analyze:
- Task completion rates and patterns
- Development velocity trends
- Blocking issues and dependencies
- Team collaboration effectiveness
- Technical decision impact

**Generate Velocity Metrics Entity:**

```typescript
await create_entities([
  {
    name: `velocity-analysis-${Date.now()}`,
    entityType: "decision",
    observations: [
      `Analysis date: ${new Date().toISOString()}`,
      `Tasks completed this period: ${completed_tasks_count}`,
      `Average task completion time: ${avg_completion_time}`,
      `Current velocity: ${velocity_metric}`,
      `Blocking issues: ${blocking_issues_count}`,
      `Team collaboration score: ${collaboration_score}`
    ]
  }
]);
```

### 4. Strategic Technology Assessment

**Query Technology Stack Decisions:**

```typescript
// Get technology choices and their outcomes
const techDecisions = await search_nodes("entityType:decision AND (technology OR architecture OR framework)");

// Get implementation patterns and their effectiveness
const implementationPatterns = await search_nodes("entityType:pattern AND (implementation OR architecture)");

// Search for research validation and outcomes
const researchFindings = await search_nodes("entityType:research");
```

**Create Technology Health Report:**

```typescript
await create_entities([
  {
    name: `tech-health-${Date.now()}`,
    entityType: "decision",
    observations: [
      `Technology stack maturity: ${tech_maturity_score}`,
      `Architecture decision effectiveness: ${arch_effectiveness}`,
      `Technical debt level: ${tech_debt_level}`,
      `Innovation opportunities: ${innovation_opportunities}`,
      `Risk factors: ${risk_factors}`
    ]
  }
]);
```

### 5. Team and Process Analysis

**Query Team Collaboration Patterns:**

```typescript
// Get team interaction patterns from sessions and PRs
const teamSessions = await search_nodes("entityType:session");
const prPatterns = await search_nodes("entityType:milestone AND type:pull_request");

// Analyze debugging and problem-solving patterns
const problemSolving = await search_nodes("entityType:issue AND Status:RESOLVED");
const debuggingPatterns = await search_nodes("entityType:pattern AND type:debugging");
```

**Generate Team Effectiveness Analysis:**

```typescript
await create_entities([
  {
    name: `team-analysis-${Date.now()}`,
    entityType: "pattern",
    observations: [
      `Team collaboration effectiveness: ${team_effectiveness}`,
      `Knowledge sharing patterns: ${knowledge_sharing}`,
      `Problem resolution efficiency: ${problem_resolution}`,
      `Code review quality: ${review_quality}`,
      `Learning and adaptation rate: ${learning_rate}`
    ]
  }
]);
```

### 6. Current Sprint/Milestone Progress

**Analyze Active Milestone Progress:**

```typescript
// Get current milestone and its task relationships
const currentMilestone = await search_nodes("entityType:milestone AND Status:ACTIVE");

// Get tasks contributing to current milestone
const milestoneTaskIds = [];
for (const milestone of currentMilestone) {
  const milestoneTasks = await search_nodes(`relates_to:${milestone.name} OR contributes_to:${milestone.name}`);
  milestoneTaskIds.push(...milestoneTasks.map(t => t.name));
}

// Analyze milestone completion progress
const milestoneProgress = calculateMilestoneProgress(milestoneTaskIds);
```

### 7. Risk and Opportunity Assessment

**Identify Project Risks:**

```typescript
// Search for blocking issues and dependencies
const blockingIssues = await search_nodes("entityType:issue AND Status:BLOCKING");
const dependencyRisks = await search_nodes("entityType:task AND depends_on");

// Create risk assessment
await create_entities([
  {
    name: `risk-assessment-${Date.now()}`,
    entityType: "decision",
    observations: [
      `High-risk areas: ${high_risk_areas}`,
      `Dependency bottlenecks: ${dependency_bottlenecks}`,
      `Technical risks: ${technical_risks}`,
      `Timeline risks: ${timeline_risks}`,
      `Mitigation strategies: ${mitigation_strategies}`
    ]
  }
]);
```

**Identify Growth Opportunities:**

```typescript
await create_entities([
  {
    name: `opportunities-${Date.now()}`,
    entityType: "decision",
    observations: [
      `Optimization opportunities: ${optimization_opportunities}`,
      `Innovation potential: ${innovation_potential}`,
      `Automation possibilities: ${automation_possibilities}`,
      `Knowledge leverage: ${knowledge_leverage}`,
      `Technology upgrades: ${technology_upgrades}`
    ]
  }
]);
```

### 8. Generate Strategic Recommendations

Use `sequential-thinking` to synthesize all analysis into strategic recommendations:

```typescript
await create_entities([
  {
    name: `strategic-recommendations-${Date.now()}`,
    entityType: "decision",
    observations: [
      `Priority focus areas: ${priority_focus}`,
      `Resource allocation recommendations: ${resource_allocation}`,
      `Technical investment priorities: ${tech_investment}`,
      `Process improvement suggestions: ${process_improvements}`,
      `Timeline adjustments: ${timeline_adjustments}`
    ]
  }
]);
```

### 9. Update Project Intelligence

**Create Project Status Summary:**

```typescript
await create_entities([
  {
    name: `project-status-${new Date().toISOString().split('T')[0]}`,
    entityType: "milestone",
    observations: [
      `Overall project health: ${project_health_score}`,
      `Completion percentage: ${completion_percentage}`,
      `Development velocity: ${current_velocity}`,
      `Team effectiveness: ${team_effectiveness_score}`,
      `Technical debt level: ${tech_debt_score}`,
      `Risk level: ${risk_level}`,
      "Type: status_report"
    ]
  }
]);

// Link status to all major project entities
await create_relations([
  { from: `project-status-${new Date().toISOString().split('T')[0]}`, to: "project-goals", relationType: "assesses" }
]);
```

### 10. Present Comprehensive Status Report

Generate and present detailed status report including:

#### Executive Summary
- Overall project health score
- Key achievements and milestones
- Critical path status
- Resource utilization

#### Development Metrics
- Task completion velocity
- Code quality trends
- Technical debt analysis
- Team collaboration effectiveness

#### Technical Assessment
- Architecture decision outcomes
- Technology stack maturity
- Infrastructure stability
- Security and compliance status

#### Strategic Recommendations
- Priority adjustments
- Resource reallocation needs
- Technology investment priorities
- Process optimization opportunities

#### Risk and Opportunity Analysis
- Current risk factors and mitigation
- Growth and innovation opportunities
- Timeline and scope considerations

#### Next Steps
- Immediate action items
- Short-term priorities (next sprint)
- Medium-term strategic initiatives
- Long-term vision alignment

### 11. Knowledge Graph Enhancement

```typescript
// Update project timeline
await add_observations([
  {
    entityName: "project-timeline",
    contents: [
      `Status review completed: ${new Date().toISOString()}`,
      `Next review scheduled: ${next_review_date}`,
      `Project phase: ${current_project_phase}`,
      `Milestone progress: ${milestone_progress}`
    ]
  }
]);

// Create trend analysis for future reference
await create_entities([
  {
    name: `trend-analysis-${Date.now()}`,
    entityType: "pattern",
    observations: [
      `Development trends identified: ${development_trends}`,
      `Team performance patterns: ${team_patterns}`,
      `Technology adoption patterns: ${tech_adoption}`,
      `Risk evolution patterns: ${risk_patterns}`
    ]
  }
]);
```

This workflow provides comprehensive project intelligence that combines quantitative metrics with qualitative insights from the Memory MCP knowledge graph, enabling data-driven strategic decision making.
