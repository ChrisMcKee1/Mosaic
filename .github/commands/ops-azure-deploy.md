# Azure Deployment Workflow with Memory MCP

Safe Azure deployment with comprehensive Memory MCP validation, infrastructure tracking, and deployment knowledge graph management. Ensures deployment readiness while building institutional deployment intelligence.

## Usage

`/azure-deploy <deployment description>`

## Arguments

- `$ARGUMENTS`: Description of what is being deployed

## Chained Workflow

### 1. Create Deployment Entity in Memory MCP

```typescript
// Create deployment entity with metadata
await create_entities([
  {
    name: `deployment-${deployment_slug}-${Date.now()}`,
    entityType: "milestone",
    observations: [
      `Deployment description: ${$ARGUMENTS}`,
      `Initiated: ${new Date().toISOString()}`,
      "Status: PLANNING",
      "Type: azure_deployment",
      `Environment: ${target_environment}`,
      `Deployment strategy: ${deployment_strategy}`,
    ],
  },
]);

// Link to current development session
const activeSession = await search_nodes(
  "entityType:session AND Status:ACTIVE"
);
if (activeSession.length > 0) {
  await create_relations([
    {
      from: `deployment-${deployment_slug}-${Date.now()}`,
      to: activeSession[0].name,
      relationType: "initiated_in",
    },
  ]);
}
```

### 2. Query Memory MCP for Deployment Context

**Search for Deployment Dependencies:**

```typescript
// Find completed features ready for deployment
const readyFeatures = await search_nodes(
  "entityType:feature AND Status:COMPLETED"
);

// Search for infrastructure decisions and patterns
const infrastructureDecisions = await search_nodes(
  "entityType:decision AND (infrastructure OR azure OR deployment)"
);

// Get related deployment patterns
const deploymentPatterns = await search_nodes(
  "entityType:pattern AND (deployment OR infrastructure)"
);

// Find blocking issues
const blockingIssues = await search_nodes(
  "entityType:issue AND (Status:BLOCKING OR priority:CRITICAL)"
);
```

### 3. Pre-Deployment Validation

Use `sequential-thinking` to analyze deployment readiness:

- Feature completeness and testing status
- Infrastructure requirements and availability
- Security and compliance validation
- Rollback strategy and contingency planning

**Create Deployment Readiness Assessment:**

```typescript
await create_entities([
  {
    name: `readiness-assessment-${deployment_slug}`,
    entityType: "decision",
    observations: [
      `Features ready for deployment: ${ready_features_count}`,
      `Infrastructure requirements met: ${infrastructure_ready}`,
      `Security validation status: ${security_status}`,
      `Testing coverage: ${testing_coverage}`,
      `Rollback strategy: ${rollback_strategy}`,
      `Risk assessment: ${risk_level}`,
    ],
  },
]);

// Link assessment to deployment
await create_relations([
  {
    from: `readiness-assessment-${deployment_slug}`,
    to: `deployment-${deployment_slug}-${Date.now()}`,
    relationType: "validates",
  },
]);
```

### 4. Infrastructure Validation with Azure CLI

Use `run_in_terminal` for Azure infrastructure validation:

```bash
# Login and set subscription
az login
az account set --subscription "${AZURE_SUBSCRIPTION_ID}"

# Validate resource group and resources
az group show --name "${AZURE_RESOURCE_GROUP}"
az resource list --resource-group "${AZURE_RESOURCE_GROUP}"

# Check Azure Container Apps environment
az containerapp env show --name "${AZURE_CONTAINER_ENV}" --resource-group "${AZURE_RESOURCE_GROUP}"

# Validate Cosmos DB and other dependencies
az cosmosdb show --name "${AZURE_COSMOS_DB_ACCOUNT_NAME}" --resource-group "${AZURE_RESOURCE_GROUP}"
```

**Document Infrastructure Status:**

```typescript
await add_observations([
  {
    entityName: `deployment-${deployment_slug}-${Date.now()}`,
    contents: [
      `Azure subscription validated: ${subscription_status}`,
      `Resource group status: ${resource_group_status}`,
      `Container Apps environment: ${container_env_status}`,
      `Database connectivity: ${database_status}`,
      `Infrastructure health: ${overall_infrastructure_health}`,
    ],
  },
]);
```

### 5. Build and Test Validation

Use `run_in_terminal` for comprehensive pre-deployment testing:

```bash
# Run full test suite
pytest --cov=src --cov-report=term-missing

# Security and code quality checks
ruff check . --fix
ruff format .
bandit -r src/

# Build validation
docker build -t mosaic-test .
```

**Create Testing Results Entity:**

```typescript
await create_entities([
  {
    name: `testing-results-${deployment_slug}`,
    entityType: "decision",
    observations: [
      `Test suite results: ${test_results}`,
      `Code coverage: ${code_coverage}`,
      `Security scan results: ${security_scan_results}`,
      `Build validation: ${build_status}`,
      `Performance benchmarks: ${performance_results}`,
    ],
  },
]);

// Link testing to deployment
await create_relations([
  {
    from: `testing-results-${deployment_slug}`,
    to: `deployment-${deployment_slug}-${Date.now()}`,
    relationType: "validates",
  },
]);
```

### 6. Environment Configuration Validation

**Query Memory MCP for Configuration Patterns:**

```typescript
// Search for environment configuration decisions
const configDecisions = await search_nodes(
  "entityType:decision AND (configuration OR environment)"
);

// Get Azure service configuration patterns
const azurePatterns = await search_nodes(
  "entityType:pattern AND (azure OR configuration)"
);
```

Use `run_in_terminal` to validate environment variables and configuration:

```bash
# Validate environment variables
echo "Checking required environment variables..."
required_vars=("AZURE_RESOURCE_GROUP" "AZURE_LOCATION" "AZURE_ENV_NAME")
for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "ERROR: $var is not set"
    exit 1
  fi
done

# Validate Azure resource connectivity
az cognitiveservices account show --name "${AZURE_OPENAI_SERVICE_NAME}" --resource-group "${AZURE_RESOURCE_GROUP}"
```

### 7. Deployment Execution with Azure Developer CLI

**Execute Deployment:**

Use `run_in_terminal` for Azure deployment:

```bash
# Deploy using Azure Developer CLI
azd auth login
azd env set AZURE_ENV_NAME "${AZURE_ENV_NAME}"
azd up --environment "${AZURE_ENV_NAME}"
```

**Track Deployment Progress:**

```typescript
// Update deployment status
await add_observations([
  {
    entityName: `deployment-${deployment_slug}-${Date.now()}`,
    contents: [
      "Status: DEPLOYING",
      `Deployment started: ${new Date().toISOString()}`,
      `Azure environment: ${azure_environment}`,
      `Deployment method: Azure Developer CLI`,
    ],
  },
]);
```

### 8. Post-Deployment Validation

**Validate Deployed Services:**

Use `run_in_terminal` for service validation:

```bash
# Check deployed services
az containerapp list --resource-group "${AZURE_RESOURCE_GROUP}"
az containerapp show --name "mosaic-mcp-tool" --resource-group "${AZURE_RESOURCE_GROUP}"

# Validate service health endpoints
curl -f "${DEPLOYMENT_URL}/health" || echo "Health check failed"
curl -f "${DEPLOYMENT_URL}/api/status" || echo "API status check failed"
```

**Create Deployment Validation Results:**

```typescript
await create_entities([
  {
    name: `validation-results-${deployment_slug}`,
    entityType: "decision",
    observations: [
      `Service deployment status: ${service_status}`,
      `Health check results: ${health_check_results}`,
      `API functionality: ${api_functionality}`,
      `Performance validation: ${performance_validation}`,
      `Monitoring setup: ${monitoring_status}`,
    ],
  },
]);
```

### 9. Deployment Success and Documentation

**Mark Deployment Complete:**

```typescript
await add_observations([
  {
    entityName: `deployment-${deployment_slug}-${Date.now()}`,
    contents: [
      "Status: COMPLETED",
      `Completion time: ${new Date().toISOString()}`,
      `Deployment URL: ${deployment_url}`,
      `Total deployment time: ${deployment_duration}`,
      "Validation: PASSED",
    ],
  },
]);

// Create deployment success pattern
await create_entities([
  {
    name: `deployment-pattern-${pattern_type}`,
    entityType: "pattern",
    observations: [
      `Successful deployment approach: ${deployment_approach}`,
      `Key success factors: ${success_factors}`,
      `Validation strategy: ${validation_strategy}`,
      `Monitoring setup: ${monitoring_approach}`,
      `Rollback procedures: ${rollback_procedures}`,
    ],
  },
]);
```

### 10. Knowledge Graph Enhancement

**Link Deployment to Project Context:**

```typescript
// Link deployment to deployed features
for (const feature of deployedFeatures) {
  await create_relations([
    {
      from: `deployment-${deployment_slug}-${Date.now()}`,
      to: feature.name,
      relationType: "deploys",
    },
  ]);
}

// Update project infrastructure knowledge
await add_observations([
  {
    entityName: "azure-infrastructure",
    contents: [
      `Deployment completed: ${new Date().toISOString()}`,
      `Services deployed: ${deployed_services_count}`,
      `Infrastructure pattern: ${infrastructure_pattern}`,
      `Deployment strategy: ${deployment_strategy_used}`,
    ],
  },
]);

// Create deployment metrics
await add_observations([
  {
    entityName: "deployment-metrics",
    contents: [
      `Deployment frequency: ${deployment_frequency}`,
      `Success rate: ${deployment_success_rate}`,
      `Average deployment time: ${avg_deployment_time}`,
      `Infrastructure reliability: ${infrastructure_reliability}`,
    ],
  },
]);
```

### 11. Post-Deployment Monitoring Setup

**Create Monitoring and Alerting Configuration:**

```typescript
await create_entities([
  {
    name: `monitoring-${deployment_slug}`,
    entityType: "pattern",
    observations: [
      "Post-deployment monitoring setup",
      `Health monitoring endpoints: ${health_endpoints}`,
      `Performance metrics tracked: ${performance_metrics}`,
      `Alert configurations: ${alert_configs}`,
      `Incident response procedures: ${incident_response}`,
    ],
  },
]);

// Link monitoring to deployment
await create_relations([
  {
    from: `monitoring-${deployment_slug}`,
    to: `deployment-${deployment_slug}-${Date.now()}`,
    relationType: "monitors",
  },
]);
```

## Expected Outputs

- Azure deployment executed safely with full validation
- Comprehensive deployment knowledge stored in Memory MCP
- Infrastructure patterns and best practices documented
- Service health and monitoring established
- Deployment metrics and insights tracked
- Team deployment capabilities enhanced

This workflow builds institutional deployment knowledge while ensuring safe, reliable Azure deployments with comprehensive validation and monitoring.
