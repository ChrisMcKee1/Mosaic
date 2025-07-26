# GitHub Copilot Configuration for Mosaic MCP Tool

This directory contains comprehensive GitHub Copilot customization for the Mosaic MCP Tool project, transitioning from Claude to VS Code Copilot with Memory MCP integration for task management.

## Directory Structure

```
.github/
├── copilot-instructions.md           # Global project guidelines
├── instructions/                     # Contextual instruction files
│   ├── mcp-development.instructions.md
│   ├── azure-integration.instructions.md
│   └── testing-quality.instructions.md
├── prompts/                         # Reusable workflow prompts
│   ├── planning-start-day.prompt.md
│   ├── planning-research-and-validate.prompt.md
│   ├── dev-implement.prompt.md
│   ├── dev-sk-plugin-implement.prompt.md
│   ├── git-workflow-commit.prompt.md
│   ├── git-workflow-branch.prompt.md
│   ├── debug-debug-and-solve.prompt.md
│   ├── ops-azure-deploy.prompt.md
│   ├── planning-project-status.prompt.md
│   ├── github-workflow-pr.prompt.md
│   ├── github-workflow-issue.prompt.md
│   ├── planning-create-task.prompt.md
│   ├── planning-sync.prompt.md
│   ├── planning-doc-check.prompt.md
│   └── dev-test.prompt.md
└── chatmodes/                       # Specialized chat modes
    ├── Architecture Solution.chatmode.md
    ├── Debug and Troubleshooting.chatmode.md
    ├── Azure Development.chatmode.md
    └── MCP and Semantic Kernel.chatmode.md
```

## How to Use

### 1. Global Instructions

The `copilot-instructions.md` file provides project-wide context and guidelines that are automatically included in every Copilot chat request.

### 2. Contextual Instructions

Instruction files in the `instructions/` directory are automatically applied based on file patterns:

- `mcp-development.instructions.md` applies to MCP and plugin-related files
- `azure-integration.instructions.md` applies to Azure and infrastructure files
- `testing-quality.instructions.md` applies to test files

### 3. Workflow Prompts

Use prompt files for common workflows:

#### In Chat View

Type `/` followed by the prompt name:

- `/planning-start-day` - Start daily development session
- `/dev-implement task_id description` - Implement a ConPort task
- `/git-workflow-commit "commit message"` - Commit with validation
- `/debug-debug-and-solve problem` - Systematic debugging

#### Command Palette

Use `Ctrl+Shift+P` → "Chat: Run Prompt" and select a prompt file.

#### Editor Integration

Open a prompt file and click the play button to run it.

### 4. Chat Modes

Select specialized chat modes for focused assistance:

- **Architecture Solution**: System design and architectural decisions
- **Debug and Troubleshooting**: Systematic problem-solving
- **Azure Development**: Cloud development and deployment
- **MCP and Semantic Kernel**: Protocol and AI integration

## Migration from Claude + Memory MCP Integration

This setup provides equivalent functionality to your Claude commands with Memory MCP replacing ConPort:

| Claude Command                       | VS Code Equivalent                       | Memory MCP Integration                                          |
| ------------------------------------ | ---------------------------------------- | --------------------------------------------------------------- |
| `/planning-start-day`                | `/planning-start-day` prompt             | `search_nodes` for tasks, `add_observations` for progress       |
| `/research-and-validate <topic>`     | `/planning-research-and-validate` prompt | `create_entities` for decisions, `create_relations` for linking |
| `/implement-with-tracking <task_id>` | `/dev-implement` prompt                  | `open_nodes` for task context, `add_observations` for progress  |
| `/semantic-kernel-workflow <plugin>` | `/dev-sk-plugin-implement` prompt        | Task and pattern entities with relations                        |
| `/git-commit-workflow <message>`     | `/git-workflow-commit` prompt            | `create_entities` for commit tracking                           |
| `/debug-and-solve <problem>`         | `/debug-debug-and-solve` prompt          | Decision and solution entities                                  |
| `/azure-deploy <description>`        | `/ops-azure-deploy` prompt               | Deployment and configuration entities                           |
| `/project-status`                    | `/planning-project-status` prompt        | `read_graph` for comprehensive analysis                         |

## VS Code Settings

The `.vscode/settings.json` file has been configured with:

- Automatic instruction file inclusion
- Custom commit message generation
- PR description generation guidelines
- Code review instructions
- File associations for prompt/instruction files

## Key Features

### 1. Automatic Context

- Project guidelines automatically included in every chat
- File-specific instructions based on what you're working on
- Environment variables and configuration context

### 2. Workflow Automation

- Structured prompts for common development tasks
- Input variables for parameterized workflows
- Integration with Memory MCP task management

### 3. Specialized Modes

- Context-aware chat modes for different types of work
- Focused assistance for architecture, debugging, and Azure development
- Consistent response styles and methodologies

### 4. Quality Assurance

- Built-in code review guidelines
- Testing and quality standards enforcement
- Documentation and tracking requirements

## Getting Started

1. **Enable Features**: Ensure VS Code settings are applied
2. **Verify Setup**: Check that instruction and prompt files are recognized
3. **Start Using**: Begin with `/planning-start-day` to set up your development session
4. **Explore Modes**: Try different chat modes for specialized assistance

## Customization

### Adding New Prompts

1. Create a new `.prompt.md` file in `.github/prompts/`
2. Add front matter with description and mode
3. Use `${input:variable:description}` for parameters
4. Document the workflow steps

### Adding New Instructions

1. Create a new `.instructions.md` file in `.github/instructions/`
2. Add `applyTo` pattern in front matter
3. Document specific guidelines for the context
4. Reference from other files if needed

### Creating Chat Modes

1. Create a new `.chatmode.md` file in `.github/chatmodes/`
2. Define the specialized context and focus areas
3. Specify response style and behavior
4. Document when to use the mode

## Best Practices

1. **Use Descriptive Names**: Make prompt and instruction names self-explanatory
2. **Include Context**: Always provide sufficient context in descriptions
3. **Test Workflows**: Validate that prompts work as expected
4. **Update Documentation**: Keep this README updated as you add new features
5. **Version Control**: Track changes to understand what works best
