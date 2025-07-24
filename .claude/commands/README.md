# Claude Code Custom Commands Guide

This directory contains comprehensive workflow commands that chain multiple MCP tools for maximum effectiveness during Mosaic MCP Tool development.

## How Arguments Work in Claude Code Custom Commands

### Basic Syntax

- **With Arguments**: `/command-name <your argument here>`
- **Without Arguments**: `/command-name`

### The $ARGUMENTS Variable

In any custom command file, use `$ARGUMENTS` to reference what the user types after the command name.

**Example:**

- File: `research-and-validate.md`
- Contains: `Research and validate: $ARGUMENTS`
- Usage: `/research-and-validate Azure OpenAI Service authentication`
- Result: Claude receives "Research and validate: Azure OpenAI Service authentication"

### Multiple Arguments

For commands needing multiple parameters, you can structure them as:

- **Space-separated**: `/command arg1 arg2 arg3`
- **Structured**: `/command --type=bug --priority=high Issue title here`
- **Natural language**: `/command Create user authentication with OAuth2 support`

## Available Workflow Commands

### Research & Planning Commands

#### `/research-and-validate <topic>`

**Purpose**: Comprehensive research with validation before implementation  
**Example**: `/research-and-validate FastAPI Server-Sent Events implementation`  
**Chains**: Sequential Thinking → Context7 or Microsoft Docs → Web Search → Analysis → ConPort Storage

#### `/project-status`

**Purpose**: Complete project overview and planning  
**Example**: `/project-status`  
**Chains**: ConPort Review → Analysis → Technology Updates → Strategic Planning

### Implementation Commands

#### `/implement-with-tracking <feature description>`

**Purpose**: Full implementation with complete tracking  
**Example**: `/implement-with-tracking RetrievalPlugin with hybrid search for FR-5`  
**Chains**: Planning → Progress Tracking → Implementation → Documentation → Linking

#### `/semantic-kernel-workflow <plugin name>`

**Purpose**: Specialized SK plugin implementation  
**Example**: `/semantic-kernel-workflow MemoryPlugin for FR-9 and FR-10`  
**Chains**: Research → Planning → Implementation → Integration → Documentation

### Git & GitHub Commands

#### `/git-commit-workflow <commit message>`

**Purpose**: Comprehensive commit with validation and tracking  
**Example**: `/git-commit-workflow "Implement FastAPI MCP server with SSE support"`  
**Chains**: Pre-commit Checks → Validation → Commit → ConPort Updates

#### `/github-pr-workflow <PR title>`

**Purpose**: Complete PR creation with documentation  
**Example**: `/github-pr-workflow "Add semantic reranking functionality"`  
**Chains**: Planning → Testing → PR Creation → Documentation → Linking

#### `/git-branch-workflow <branch name>`

**Purpose**: Branch creation with proper tracking  
**Example**: `/git-branch-workflow "feature/FR-8-semantic-reranking"`  
**Chains**: Planning → Branch Creation → Documentation → Environment Setup

#### `/github-issue-workflow <issue title>`

**Purpose**: Comprehensive issue creation with context  
**Example**: `/github-issue-workflow "Azure Cosmos DB integration timeout"`  
**Chains**: Analysis → Context Research → Issue Creation → Planning → Linking

### Debugging & Deployment Commands

#### `/debug-and-solve <problem description>`

**Purpose**: Systematic problem-solving with full tracking  
**Example**: `/debug-and-solve Azure authentication failing in development environment`  
**Chains**: Analysis → Investigation → Research → Solution → Documentation

#### `/azure-deploy <deployment description>`

**Purpose**: Safe Azure deployment with validation  
**Example**: `/azure-deploy Production deployment of Mosaic MCP Tool v0.1.0`  
**Chains**: Planning → Validation → Deployment → Testing → Documentation

## Command Argument Best Practices

### 1. Be Specific and Descriptive

**Good**: `/research-and-validate Azure OpenAI Service token management and rate limiting`  
**Poor**: `/research-and-validate auth`

### 2. Include Context When Helpful

**Good**: `/implement-with-tracking DiagramPlugin Mermaid generation for FR-12 with Azure OpenAI`  
**Poor**: `/implement-with-tracking diagrams`

### 3. Use Functional Requirement References

**Good**: `/semantic-kernel-workflow RefinementPlugin semantic reranking for FR-8`  
**Helpful**: Directly references our project requirements

### 4. Follow Project Naming Conventions

**Git branches**: `/git-branch-workflow "feature/FR-5-hybrid-search"`  
**Commits**: `/git-commit-workflow "Add RetrievalPlugin hybrid search functionality"`

## Advanced Usage Patterns

### Chaining Commands in Sequence

1. `/research-and-validate Azure Container Apps deployment patterns`
2. `/implement-with-tracking Azure Container Apps Bicep template for FR-4`
3. `/git-commit-workflow "Add Azure Container Apps infrastructure template"`
4. `/github-pr-workflow "Infrastructure: Add Azure Container Apps deployment"`

### Using Commands for Different Development Phases

- **Planning Phase**: `/project-status`, `/research-and-validate`
- **Implementation Phase**: `/implement-with-tracking`, `/semantic-kernel-workflow`
- **Integration Phase**: `/git-commit-workflow`, `/github-pr-workflow`
- **Deployment Phase**: `/azure-deploy`, `/debug-and-solve`

## Tips for Maximum Effectiveness

1. **Always include relevant FR numbers** when implementing features
2. **Use descriptive arguments** that provide context for ConPort tracking
3. **Chain commands logically** for complete development workflows
4. **Reference existing ConPort context** when debugging or enhancing
5. **Be consistent with naming** to maintain project organization

## Command Customization

You can modify any command file to:

- Add project-specific validation steps
- Include additional MCP tool integrations
- Customize output formats
- Add team-specific processes

Each command is designed to be a starting point that can be adapted to your specific workflow needs while maintaining the core MCP tool chaining for comprehensive tracking and validation.
