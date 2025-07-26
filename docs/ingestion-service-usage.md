# Mosaic Ingestion Service - Usage Guide

## Overview

The Mosaic Ingestion Service is responsible for analyzing code repositories and populating the knowledge graph with extracted entities and relationships. It supports multiple deployment modes and entry points for different use cases.

## Entry Points

The ingestion service provides three main entry points, each optimized for different scenarios:

### 1. Production Entry Point: `main.py`

**Purpose**: Azure Container App Job deployment for production ingestion
**Framework**: Microsoft Semantic Kernel with Magentic AI orchestration
**Dependencies**: Full Azure stack (Cosmos DB, OpenAI, Key Vault)

```bash
python -m mosaic_ingestion.main --repository-url <URL> [OPTIONS]
```

**Features**:

- AI-powered agent orchestration (GitSleuth, CodeParser, GraphArchitect, DocuWriter, GraphAuditor)
- Production-grade Azure integration
- Comprehensive error handling and logging
- Scalable for large repositories

### 2. Local Development Entry Point: `local_main.py`

**Purpose**: Lightweight local development and testing
**Framework**: Simple Python-based processing
**Dependencies**: Minimal (GitPython, basic file processing)

```bash
python src/mosaic-ingestion/local_main.py --repository-url <URL> [OPTIONS]
```

**Features**:

- Fast iteration for development
- Basic entity extraction and statistics
- No Azure dependencies required
- CRUD-001 commit state tracking support

### 3. Enhanced Local Entry Point: `enhanced_local_main.py`

**Purpose**: Local development with real Cosmos DB persistence
**Framework**: Python processing with Cosmos DB storage
**Dependencies**: Cosmos DB SDK (local emulator or cloud)

```bash
python src/mosaic-ingestion/enhanced_local_main.py --repository-url <URL> [OPTIONS]
```

**Features**:

- Real Cosmos DB integration
- Golden Node entity creation
- Dual-mode support (local emulator or Azure cloud)
- Bridge between development and production

## Command Line Arguments

### Common Arguments (All Entry Points)

| Argument | Short | Type | Required | Default | Description |
|----------|-------|------|----------|---------|-------------|
| `--repository-url` | | string | Yes | | Git repository URL to ingest |
| `--branch` | | string | No | `main` | Git branch to process |
| `--force-override` | `-f` | flag | No | `false` | **🔥 Clear all memory/nodes for this repo+branch before ingestion** |

### Entry Point Specific Arguments

#### `main.py` (Production)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--config-file` | string | No | | Path to configuration file |

#### `local_main.py` (Local Development)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--debug` | flag | No | `false` | Enable debug logging |

#### `enhanced_local_main.py` (Enhanced Local)

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--mode` | string | No | `local` | Cosmos DB mode: `local` or `azure` |

## Force Override Flag: `--force-override` / `-f`

### ⚠️ Important: Data Deletion Warning

The `--force-override` flag is a **destructive operation** that permanently deletes all knowledge graph data for the specified repository and branch combination.

### Behavior

- **Scope**: Deletes ALL entities and relationships where:
  - `repository_url` matches exactly AND
  - `branch_name` matches exactly
- **Preservation**: Keeps data from:
  - Same repository, different branches
  - Different repositories, same branch name
- **Confirmation**: Prompts for user confirmation before deletion
- **Logging**: Comprehensive audit logging of all deletion operations

### Use Cases

1. **Repository Restructuring**: When a repository has been significantly refactored
2. **Branch Reset**: When a branch has been force-pushed or reset
3. **Data Corruption Recovery**: When ingestion data needs to be completely refreshed
4. **Testing**: When you need a clean slate for testing purposes

### Examples

```bash
# Force clear main branch of specific repo
python local_main.py --repository-url https://github.com/user/repo --force-override

# Force clear feature branch
python local_main.py --repository-url https://github.com/user/repo --branch feature/new-ui --force-override

# Using short flag
python enhanced_local_main.py --repository-url https://github.com/user/repo -f
```

## Usage Examples

### Local Development

```bash
# Basic local ingestion
python src/mosaic-ingestion/local_main.py \
  --repository-url https://github.com/microsoft/semantic-kernel \
  --branch main

# Debug mode
python src/mosaic-ingestion/local_main.py \
  --repository-url https://github.com/microsoft/semantic-kernel \
  --branch main \
  --debug

# Force override with confirmation
python src/mosaic-ingestion/local_main.py \
  --repository-url https://github.com/microsoft/semantic-kernel \
  --branch main \
  --force-override
```

### Enhanced Local Development

```bash
# Local Cosmos DB emulator
python src/mosaic-ingestion/enhanced_local_main.py \
  --repository-url https://github.com/microsoft/semantic-kernel \
  --mode local

# Azure Cosmos DB cloud
python src/mosaic-ingestion/enhanced_local_main.py \
  --repository-url https://github.com/microsoft/semantic-kernel \
  --mode azure \
  --force-override
```

### Production Deployment

```bash
# Production ingestion with Magentic orchestration
python -m mosaic_ingestion.main \
  --repository-url https://github.com/microsoft/semantic-kernel \
  --branch main \
  --config-file /app/config/production.yaml
```

## Environment Variables

### Required for All Modes

| Variable | Description | Example |
|----------|-------------|---------|
| `GITHUB_TOKEN` | GitHub personal access token | `ghp_xxxxxxxxxxxx` |
| `GIT_USERNAME` | Git username (alternative auth) | `your-username` |
| `GIT_PASSWORD` | Git password/token (alternative auth) | `your-password` |

### Azure Integration (Production & Enhanced Local with Azure mode)

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_COSMOS_ENDPOINT` | Cosmos DB endpoint URL | `https://account.documents.azure.com:443/` |
| `AZURE_COSMOS_KEY` | Cosmos DB access key | `primary-or-secondary-key` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | `https://resource.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | `your-api-key` |
| `AZURE_STORAGE_ACCOUNT` | Storage account name | `storageaccount` |
| `AZURE_STORAGE_KEY` | Storage account key | `storage-key` |

### Local Cosmos DB Emulator

| Variable | Description | Default |
|----------|-------------|---------|
| `COSMOS_EMULATOR_ENDPOINT` | Local emulator endpoint | `https://localhost:8081` |
| `COSMOS_EMULATOR_KEY` | Local emulator key | Well-known emulator key |

## Remote Triggering Patterns

### Azure Container Apps Job

The production ingestion service is deployed as an Azure Container App Job that can be triggered remotely:

```bash
# Trigger via Azure CLI
az containerapp job start \
  --name mosaic-ingestion-job \
  --resource-group rg-mosaic-prod \
  --args "--repository-url https://github.com/user/repo --branch main"

# Trigger with force override
az containerapp job start \
  --name mosaic-ingestion-job \
  --resource-group rg-mosaic-prod \
  --args "--repository-url https://github.com/user/repo --branch main --force-override"
```

### GitHub Actions Integration

Example GitHub Actions workflow for automatic ingestion:

```yaml
name: Trigger Mosaic Ingestion
on:
  push:
    branches: [main, develop]
  workflow_dispatch:
    inputs:
      force_override:
        description: 'Force override existing data'
        required: false
        default: false
        type: boolean

jobs:
  trigger-ingestion:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger Mosaic Ingestion
        run: |
          ARGS="--repository-url ${{ github.server_url }}/${{ github.repository }} --branch ${{ github.ref_name }}"
          if [[ "${{ github.event.inputs.force_override }}" == "true" ]]; then
            ARGS="$ARGS --force-override"
          fi
          
          az containerapp job start \
            --name mosaic-ingestion-job \
            --resource-group rg-mosaic-prod \
            --args "$ARGS"
        env:
          AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}
          AZURE_CLIENT_SECRET: ${{ secrets.AZURE_CLIENT_SECRET }}
          AZURE_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}
```

### REST API Triggering

For programmatic triggering, use Azure Container Apps REST API:

```bash
# Get access token
ACCESS_TOKEN=$(az account get-access-token --query accessToken -o tsv)

# Trigger job execution
curl -X POST \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "template": {
      "containers": [{
        "name": "mosaic-ingestion",
        "args": [
          "--repository-url", "https://github.com/user/repo",
          "--branch", "main",
          "--force-override"
        ]
      }]
    }
  }' \
  "https://management.azure.com/subscriptions/{subscription-id}/resourceGroups/{resource-group}/providers/Microsoft.App/containerApps/{app-name}/jobs/{job-name}/start?api-version=2022-10-01"
```

## Performance Considerations

### Repository Size Guidelines

| Repository Size | Recommended Entry Point | Expected Duration |
|----------------|------------------------|-------------------|
| Small (< 1k files) | Any | 30s - 2min |
| Medium (1k-10k files) | `enhanced_local_main.py` or `main.py` | 2-10min |
| Large (10k+ files) | `main.py` (production) | 10min+ |

### Optimization Tips

1. **Use appropriate entry point**: Production for large repos, local for quick iteration
2. **Branch targeting**: Process specific branches rather than all branches
3. **Force override sparingly**: Only use when necessary due to performance impact
4. **Monitor resources**: Large repositories may require increased Container App resources

## Error Handling and Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify GitHub token has repository access
   - Check Azure credentials are valid

2. **Cosmos DB Connection Issues**
   - Verify endpoint and key configuration
   - Check firewall rules for Azure Cosmos DB

3. **Memory/Resource Limits**
   - Increase Container App memory allocation
   - Consider processing smaller batches

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
python src/mosaic-ingestion/local_main.py \
  --repository-url https://github.com/user/repo \
  --debug
```

## Monitoring and Logging

### Log Locations

- **Local Development**: `mosaic_ingestion.log` in working directory
- **Production**: Azure Container Apps logs via Azure portal or CLI

### Key Metrics

- Files processed count
- Entities extracted count
- Processing time duration
- Error rates and types
- Force override operations (audit trail)

### Monitoring Commands

```bash
# View Container App logs
az containerapp logs show \
  --name mosaic-ingestion-job \
  --resource-group rg-mosaic-prod \
  --follow

# View job execution history
az containerapp job execution list \
  --name mosaic-ingestion-job \
  --resource-group rg-mosaic-prod
```

---

## Quick Reference

### Most Common Commands

```bash
# Local development (basic)
python src/mosaic-ingestion/local_main.py --repository-url <URL>

# Local development with Cosmos DB
python src/mosaic-ingestion/enhanced_local_main.py --repository-url <URL> --mode local

# Production deployment
python -m mosaic_ingestion.main --repository-url <URL>

# Force override (any entry point)
<command> --repository-url <URL> --force-override
```

### When to Use Force Override

✅ **Use when**:

- Repository structure changed significantly
- Previous ingestion failed or corrupted
- Testing requires clean state
- Branch was force-pushed/reset

❌ **Don't use when**:

- Regular updates to same repository/branch
- Minor code changes
- Uncertain about data impact
- In production without careful consideration

---

*For additional support, see the main Mosaic MCP Tool documentation or contact the development team.*
