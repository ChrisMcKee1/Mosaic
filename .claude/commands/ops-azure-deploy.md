# Azure Deploy Workflow

Comprehensive Azure deployment workflow using all MCP tools for safe, tracked, and validated deployments.

## Usage
Use `/azure-deploy` when you need to:
- Deploy Mosaic MCP Tool to Azure
- Update Azure infrastructure
- Validate deployment configuration
- Troubleshoot deployment issues

## Chained Workflow Steps

### 1. Pre-Deployment Planning
```
Use mcp__sequential-thinking__sequentialthinking to:
- Review deployment requirements
- Check all functional requirements (FR-1 through FR-13)
- Identify deployment dependencies
- Plan rollback strategy
```

### 2. ConPort Deployment Preparation
```
Use mcp__conport__get_progress to check implementation status
Use mcp__conport__get_decisions to review Azure architecture decisions
Use mcp__conport__search_custom_data_value_fts to find Azure configuration notes
Use mcp__conport__log_progress for deployment tracking:
- Status: "IN_PROGRESS"
- Description: "Azure deployment process initiated"
```

### 3. Azure Documentation Research
```
Use mcp__context7__resolve-library-id to find Azure services docs
Use mcp__context7__get-library-docs for:
- Azure Container Apps deployment
- Azure Developer CLI (azd) usage
- Bicep template validation
- Azure service configuration

Use WebSearch to validate:
- Current Azure service pricing
- Recent deployment best practices
- Known issues or limitations
```

### 4. Desktop Commander Pre-Deployment Checks
```
Use mcp__desktop-commander tools for validation:
- read_file: check all configuration files
- search_code: validate environment variables
- execute_command: run local tests
- list_directory: verify file structure
- get_file_info: check file permissions
```

### 5. Azure CLI Deployment
```
Use mcp__desktop-commander__execute_command for:
- az login (authenticate)
- azd auth login (azd authentication)
- azd provision (create infrastructure)
- azd deploy (deploy application)
- az resource list (verify resources)
```

### 6. Deployment Validation
```
Use mcp__sequential-thinking__sequentialthinking to:
- Analyze deployment output
- Verify all services are running
- Check health endpoints
- Validate MCP server functionality
```

### 7. ConPort Deployment Tracking
```
Use mcp__conport__log_decision for deployment choices:
- Summary: deployment configuration used
- Rationale: why specific settings chosen
- Implementation details: actual deployment steps

Use mcp__conport__log_custom_data for deployment info:
- Category: "Deployment"
- Key: deployment date/version
- Value: Azure resource details, URLs, configuration

Use mcp__conport__update_progress:
- Status: "DONE" (if successful) or document issues
- Description: deployment results and notes
```

### 8. Post-Deployment Validation
```
Use mcp__desktop-commander__execute_command for testing:
- curl health check endpoints
- Test MCP protocol compliance
- Validate all FR requirements

Use WebSearch to verify:
- Azure service status
- Any reported issues with our services
```

### 9. Documentation and Linking
```
Use mcp__conport__link_conport_items to connect:
- Deployment to functional requirements
- Azure resources to system patterns
- Deployment decisions to progress tracking

Use mcp__conport__log_system_pattern for deployment pattern:
- Name: "Azure Container Apps MCP Deployment"
- Description: successful deployment steps
- Tags: for future deployments
```

## Example Usage

"Deploy the Mosaic MCP Tool to Azure using azd, ensuring all services are properly configured and the MCP server is accessible with full progress tracking."

## Expected Outputs
- Successfully deployed Azure infrastructure
- Running MCP server with validated endpoints
- Complete deployment documentation in conport
- Deployment patterns captured for future use
- Post-deployment validation completed