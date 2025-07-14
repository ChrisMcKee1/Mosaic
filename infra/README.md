# Mosaic MCP Tool - Infrastructure Deployment

## Overview

This infrastructure deployment implements Microsoft's **OmniRAG Pattern** from the CosmosAIGraph reference architecture, providing a unified, cost-effective backend for the Mosaic MCP Tool.

## Architecture Pattern

Based on [Microsoft's CosmosAIGraph](https://github.com/AzureCosmosDB/CosmosAIGraph), our deployment follows the OmniRAG pattern which consolidates:

- **Vector Search**: Azure Cosmos DB for NoSQL with DiskANN indexing
- **Graph Relationships**: Embedded JSON arrays (`dependency_ids`, `developers`, `used_by_lib`)
- **Memory Storage**: Unified containers for long-term and contextual memory
- **Configuration**: System entities and application settings

## Folder Structure

```
infra/
├── main.bicep                    # Entry point - resource group and outputs
├── resources.bicep               # Main orchestrator calling modules
├── omnirag/                      # OmniRAG-specific components
│   └── cosmos-omnirag.bicep      # Unified Cosmos DB backend
├── modules/                      # Reusable infrastructure modules
│   ├── container-apps.bicep      # Container Apps with OmniRAG config
│   └── fetch-container-image.bicep # Image reference helper
└── README.md                     # This file
```

## Key Design Principles

### 1. Modular Architecture
Following CosmosAIGraph patterns, each component is isolated into focused modules:
- **OmniRAG Module**: Self-contained Cosmos DB configuration
- **Container Apps Module**: Application hosting with proper environment configuration
- **Main Resources**: Orchestrates all modules with clear dependencies

### 2. Environment Variable Naming Convention
Adopting CAIG (CosmosAIGraph) patterns with `MOSAIC_` prefix:
```bash
# OmniRAG Configuration
MOSAIC_COSMOS_ENDPOINT=...
MOSAIC_DATABASE_NAME=mosaic-omnirag
MOSAIC_LIBRARIES_CONTAINER=libraries
MOSAIC_MEMORIES_CONTAINER=memories
MOSAIC_DOCUMENTS_CONTAINER=documents
MOSAIC_CONFIG_CONTAINER=config
```

### 3. Tagging Strategy
Consistent tagging for component identification:
```bicep
tags: union(tags, { 
  'omnirag-component': 'unified-backend'
  'pattern': 'microsoft-omnirag'
  'component': 'mcp-server'
})
```

## OmniRAG Implementation Details

### Cosmos DB Configuration
Our `cosmos-omnirag.bicep` module configures:

```bicep
// Primary OmniRAG containers
containers: [
  {
    name: 'libraries'           // Graph relationships via JSON arrays
    paths: ['/libtype']
    vectorIndexes: [{ path: '/embedding', type: 'diskANN' }]
    includedPaths: ['/dependency_ids/*', '/developers/*']
  }
  {
    name: 'memories'            // Agent memory with semantic search
    paths: ['/sessionId']
    vectorIndexes: [{ path: '/embedding', type: 'diskANN' }]
  }
  {
    name: 'documents'           // Document storage with vector search
    paths: ['/category']
    vectorIndexes: [{ path: '/embedding', type: 'diskANN' }]
  }
  {
    name: 'config'              // System configuration
    paths: ['/pk']
  }
]
```

### Graph Relationship Model
Following Microsoft's pattern, relationships are embedded in JSON:

```json
{
  "id": "pypi_flask",
  "libtype": "pypi",
  "libname": "flask",
  "developers": ["contact@palletsprojects.com"],
  "dependency_ids": ["pypi_werkzeug", "pypi_jinja2"],
  "used_by_lib": ["pypi_flask_sqlalchemy"],
  "embedding": [0.012, "...", -0.045]
}
```

## Deployment Commands

### Standard Deployment
```bash
# Deploy entire infrastructure
azd up

# Deploy only infrastructure changes
azd deploy
```

### Post-Deployment OAuth Setup
After successful deployment, configure OAuth 2.1:

```bash
# Create App Registration
az ad app create --display-name "Mosaic MCP Server" \
  --web-redirect-uris "https://YOUR_APP_URL/auth/callback"

# Update container app with OAuth Client ID
az containerapp update --name "mosaic" \
  --resource-group "rg-YOUR_ENV" \
  --set-env-vars "AZURE_OAUTH_CLIENT_ID=YOUR_APP_ID"
```

## Environment Variables

The deployment configures these key environment variables:

| Variable | Purpose | Source |
|----------|---------|---------|
| `MOSAIC_COSMOS_ENDPOINT` | OmniRAG backend endpoint | Cosmos DB module output |
| `MOSAIC_DATABASE_NAME` | Database name (`mosaic-omnirag`) | Fixed value |
| `MOSAIC_LIBRARIES_CONTAINER` | Graph relationships container | Fixed value (`libraries`) |
| `MOSAIC_MEMORIES_CONTAINER` | Memory storage container | Fixed value (`memories`) |
| `MOSAIC_DOCUMENTS_CONTAINER` | Document storage container | Fixed value (`documents`) |
| `MOSAIC_CONFIG_CONTAINER` | Configuration container | Fixed value (`config`) |
| `AZURE_USE_MANAGED_IDENTITY` | Enable managed identity auth | Always `true` |
| `MCP_OAUTH_ENABLED` | Enable OAuth 2.1 security | Always `true` |

## Benefits of This Approach

### 1. Cost Optimization
- **Single Cosmos DB account** replaces Azure AI Search + Cosmos DB
- **Serverless billing** for variable workloads
- **Consumption-based** Container Apps and Functions

### 2. Simplified Operations
- **Unified backend** reduces complexity
- **Managed identity** eliminates connection string management
- **Modular deployment** enables selective updates

### 3. Alignment with Microsoft Patterns
- **CosmosAIGraph compliance** ensures best practices
- **AVM modules** provide enterprise-grade reliability
- **azd integration** streamlines developer experience

## Troubleshooting

### Common Issues

1. **Missing fetch-container-image.bicep**:
   ```bash
   # Copy from existing AZD template or create minimal version
   cp /path/to/azd/template/infra/modules/fetch-container-image.bicep infra/modules/
   ```

2. **Role assignment delays**:
   ```bash
   # Wait 2-3 minutes after deployment for role propagation
   az role assignment list --assignee YOUR_IDENTITY_ID
   ```

3. **OAuth configuration**:
   - Ensure redirect URI matches deployed app URL
   - Verify App Registration has correct permissions
   - Check container app environment variables

For detailed troubleshooting, see the main project documentation.