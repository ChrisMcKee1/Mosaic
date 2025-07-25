# GitHub Copilot Command Files Guide

This directory contains comprehensive workflow commands that leverage Memory MCP for advanced knowledge graph management during Mosaic MCP Tool development.

## How Arguments Work in GitHub Copilot Command Files

### Basic Syntax

- **With Arguments**: `/command-name <your argument here>`
- **Without Arguments**: `/command-name`

### The $ARGUMENTS Variable

In any custom command file, use `$ARGUMENTS` to reference what the user types after the command name.

**Example:**

- File: `research-and-validate.md`
- Contains: `Research and validate: $ARGUMENTS`
- Usage: `/research-and-validate Azure OpenAI Service authentication`
- Result: Copilot receives "Research and validate: Azure OpenAI Service authentication"

### Multiple Arguments

For commands needing multiple parameters, you can structure them as:

- **Space-separated**: `/command arg1 arg2 arg3`
- **Structured**: `/command --type=bug --priority=high Issue title here`
- **Natural language**: `/command Create user authentication with OAuth2 support`

## Memory MCP Integration Patterns

### Entity Types Used in Workflows

- **task**: Development tasks and user stories (replaces ConPort tasks)
- **decision**: Architectural and implementation decisions
- **milestone**: Project milestones and releases
- **issue**: Bug reports and problems
- **feature**: Feature implementations and enhancements
- **pattern**: Reusable code and architecture patterns
- **person**: Team members and contributors
- **research**: Research findings and validation results
- **session**: Development sessions and context

### Relationship Types

- **implements**: Feature implements requirement
- **depends_on**: Task depends on another task
- **relates_to**: General relationship between entities
- **blocks**: Entity blocks another entity
- **follows**: Sequential workflow relationships
- **extends**: Pattern extends another pattern
- **validates**: Research validates decision
- **completes**: Task completes milestone

### Advanced Memory MCP Operations

#### Multi-Entity Creation Patterns

```typescript
// Create coordinated entities for complex workflows
await create_entities([
  {
    name: "FR-5-hybrid-search",
    entityType: "task",
    observations: [
      "Implement hybrid vector+graph search",
      "Priority: HIGH",
      "Status: IN_PROGRESS",
    ],
  },
  {
    name: "hybrid-search-research",
    entityType: "research",
    observations: [
      "Azure Cognitive Search patterns",
      "Elasticsearch alternatives",
      "Performance benchmarks",
    ],
  },
]);

await create_relations([
  {
    from: "FR-5-hybrid-search",
    to: "hybrid-search-research",
    relationType: "validates",
  },
]);
```

#### Session-Aware Context Management

```typescript
// Track development session context
await create_entities([
  {
    name: "session-2025-01-15",
    entityType: "session",
    observations: [
      "Focus: Authentication implementation",
      "Team: chrismckee",
      "Started: 09:00",
    ],
  },
]);

// Link session to active work
await create_relations([
  {
    from: "session-2025-01-15",
    to: "FR-8-auth-system",
    relationType: "focuses_on",
  },
]);
```

## Available Workflow Commands

### Research & Planning Commands

#### `/research-and-validate <topic>`

**Purpose**: Comprehensive research with Memory MCP knowledge graph storage  
**Example**: `/research-and-validate FastAPI Server-Sent Events implementation`  
**Chains**: Sequential Thinking → Context7/Microsoft Docs → Web Search → Analysis → Memory MCP Entity Creation

**Memory MCP Operations**:

- Creates research entities with findings
- Creates decision entities with recommendations
- Links research to related tasks and patterns
- Tracks research validation relationships

#### `/project-status`

**Purpose**: Complete project overview using Memory MCP knowledge graph  
**Example**: `/project-status`  
**Chains**: Memory MCP Graph Read → Analysis → Technology Updates → Strategic Planning

**Memory MCP Operations**:

- Reads complete knowledge graph
- Searches for recent milestones and decisions
- Creates status summary entities
- Updates project timeline observations

### Implementation Commands

#### `/implement-with-tracking <feature description>`

**Purpose**: Full implementation with Memory MCP knowledge tracking  
**Example**: `/implement-with-tracking RetrievalPlugin with hybrid search for FR-5`  
**Chains**: Planning → Memory MCP Context → Implementation → Documentation → Relationship Mapping

**Memory MCP Operations**:

- Creates implementation task entities
- Links to research and decision entities
- Tracks progress through observations
- Creates pattern entities for reusable code

#### `/semantic-kernel-workflow <plugin name>`

**Purpose**: Specialized SK plugin implementation with Memory MCP integration  
**Example**: `/semantic-kernel-workflow MemoryPlugin for FR-9 and FR-10`  
**Chains**: Research → Planning → Implementation → Integration → Memory MCP Documentation

**Memory MCP Operations**:

- Creates plugin entities with specifications
- Links to Semantic Kernel patterns
- Tracks integration dependencies
- Documents plugin relationships

### Git & GitHub Commands

#### `/git-commit-workflow <commit message>`

**Purpose**: Comprehensive commit with Memory MCP validation and tracking  
**Example**: `/git-commit-workflow "Implement FastAPI MCP server with SSE support"`  
**Chains**: Pre-commit Checks → Validation → Commit → Memory MCP Updates

**Memory MCP Operations**:

- Creates commit entities with metadata
- Links commits to implemented features
- Updates task progress observations
- Tracks code change patterns

#### `/github-pr-workflow <PR title>`

**Purpose**: Complete PR creation with Memory MCP documentation  
**Example**: `/github-pr-workflow "Add semantic reranking functionality"`  
**Chains**: Planning → Testing → PR Creation → Documentation → Memory MCP Linking

**Memory MCP Operations**:

- Creates PR entities with context
- Links to completed tasks and features
- Documents review requirements
- Tracks deployment relationships

#### `/git-branch-workflow <branch name>`

**Purpose**: Branch creation with Memory MCP tracking  
**Example**: `/git-branch-workflow "feature/FR-8-semantic-reranking"`  
**Chains**: Planning → Branch Creation → Documentation → Memory MCP Environment Setup

**Memory MCP Operations**:

- Creates branch entities with purpose
- Links to feature requirements
- Tracks workflow dependencies
- Documents branch relationships

#### `/github-issue-workflow <issue title>`

**Purpose**: Comprehensive issue creation with Memory MCP context  
**Example**: `/github-issue-workflow "Azure Cosmos DB integration timeout"`  
**Chains**: Analysis → Context Research → Issue Creation → Planning → Memory MCP Linking

**Memory MCP Operations**:

- Creates issue entities with symptoms
- Links to related decisions and patterns
- Searches for similar historical issues
- Documents troubleshooting relationships

### Debugging & Deployment Commands

#### `/debug-and-solve <problem description>`

**Purpose**: Systematic problem-solving with Memory MCP knowledge tracking  
**Example**: `/debug-and-solve Azure authentication failing in development environment`  
**Chains**: Analysis → Investigation → Research → Solution → Memory MCP Documentation

**Memory MCP Operations**:

- Creates problem entities with symptoms
- Searches for related debugging patterns
- Documents solution approaches
- Links to prevention strategies

#### `/azure-deploy <deployment description>`

**Purpose**: Safe Azure deployment with Memory MCP validation  
**Example**: `/azure-deploy Production deployment of Mosaic MCP Tool v0.1.0`  
**Chains**: Planning → Validation → Deployment → Testing → Memory MCP Documentation

**Memory MCP Operations**:

- Creates deployment entities with details
- Links to tested features and patterns
- Tracks deployment dependencies
- Documents infrastructure relationships

## Command Argument Best Practices

### 1. Be Specific and Descriptive

**Good**: `/research-and-validate Azure OpenAI Service token management and rate limiting`  
**Poor**: `/research-and-validate auth`

### 2. Include Context When Helpful

**Good**: `/implement-with-tracking DiagramPlugin Mermaid generation for FR-12 with Azure OpenAI`  
**Poor**: `/implement-with-tracking diagrams`

### 3. Use Functional Requirement References

**Good**: `/semantic-kernel-workflow RefinementPlugin semantic reranking for FR-8`  
**Helpful**: Directly references our project requirements and creates proper Memory MCP links

### 4. Follow Project Naming Conventions

**Git branches**: `/git-branch-workflow "feature/FR-5-hybrid-search"`  
**Commits**: `/git-commit-workflow "Add RetrievalPlugin hybrid search functionality"`

## Advanced Memory MCP Usage Patterns

### Chaining Commands with Knowledge Context

1. `/research-and-validate Azure Container Apps deployment patterns`
   - Creates research and decision entities
2. `/implement-with-tracking Azure Container Apps Bicep template for FR-4`
   - Links implementation to research findings
3. `/git-commit-workflow "Add Azure Container Apps infrastructure template"`
   - Links commit to implementation and patterns
4. `/github-pr-workflow "Infrastructure: Add Azure Container Apps deployment"`
   - Links PR to commit, implementation, and validation

### Memory MCP Knowledge Discovery

Commands can now leverage historical knowledge through:

- **search_nodes**: Find related work across the entire project history
- **open_nodes**: Get detailed context for specific entities
- **read_graph**: Understand complete project relationships

### Session-Aware Development

Each command contributes to building a comprehensive knowledge graph that:

- Tracks decision rationale over time
- Links implementations to requirements
- Documents pattern evolution
- Preserves troubleshooting knowledge
- Maintains team learning context

## Using Commands for Different Development Phases

- **Planning Phase**: `/project-status`, `/research-and-validate`
  - Memory MCP: Creates research and decision entities
- **Implementation Phase**: `/implement-with-tracking`, `/semantic-kernel-workflow`
  - Memory MCP: Links implementation to requirements and patterns
- **Integration Phase**: `/git-commit-workflow`, `/github-pr-workflow`
  - Memory MCP: Tracks code changes and deployment readiness
- **Deployment Phase**: `/azure-deploy`, `/debug-and-solve`
  - Memory MCP: Documents infrastructure and solutions

## Tips for Maximum Memory MCP Effectiveness

1. **Always include relevant FR numbers** to create proper entity relationships
2. **Use descriptive arguments** that provide rich context for entity observations
3. **Chain commands logically** to build comprehensive knowledge graphs
4. **Reference existing Memory MCP context** when debugging or enhancing
5. **Be consistent with naming** to maintain entity relationship integrity
6. **Leverage search capabilities** to find related historical context
7. **Document patterns explicitly** for future reuse and learning

## Command Customization

You can modify any command file to:

- Add project-specific Memory MCP entity types
- Include additional relationship patterns
- Customize observation formats
- Add team-specific knowledge tracking
- Integrate specialized search patterns

Each command is designed to build and leverage the Memory MCP knowledge graph, creating a comprehensive development intelligence system that learns and improves over time.
