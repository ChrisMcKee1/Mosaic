# Cosmos DB Cleanup for Fresh Testing

This directory contains comprehensive cleanup tools for Azure Cosmos DB to establish a clean testing baseline before implementing CRUD-complete, branch-aware ingestion functionality.

## 🎯 Task Overview

**Task ID:** CRUD-000  
**Priority:** HIGH  
**Status:** ✅ COMPLETED  
**Purpose:** Remove all existing data from Cosmos DB containers to establish clean testing baseline

## 📁 Files

| File | Purpose | Platform |
|------|---------|----------|
| `cosmos_cleanup.py` | Main cleanup script with comprehensive features | Cross-platform |
| `cleanup_cosmos.ps1` | PowerShell wrapper for Windows users | Windows |
| `tests/test_cosmos_cleanup.py` | Unit and integration tests | Cross-platform |

## 🚀 Quick Start

### Windows (PowerShell)
```powershell
# Safe interactive mode with backup
.\cleanup_cosmos.ps1 -Confirm -Backup

# Automated mode for CI/CD
.\cleanup_cosmos.ps1 -Force -NoBackup

# Run tests
.\cleanup_cosmos.ps1 -Test
```

### Cross-Platform (Python)
```bash
# Safe interactive mode with backup
python cosmos_cleanup.py --confirm --backup

# Automated mode for CI/CD  
python cosmos_cleanup.py --force --no-backup

# Run tests
python tests/test_cosmos_cleanup.py --unit
```

## 🔧 Features

### Safety Features
- ✅ **Backup Creation** - Automatic backup before deletion
- ✅ **Confirmation Prompts** - Interactive safety checks
- ✅ **Container Validation** - Verify cleanup success
- ✅ **Error Handling** - Graceful failure handling
- ✅ **Rollback Support** - Backup enables data recovery

### Container Coverage
- `knowledge` - Main knowledge graph data
- `memory` - Memory storage
- `golden_nodes` - Golden Node unified schema
- `diagrams` - Diagram storage
- `code_entities` - Code entities
- `code_relationships` - Code relationships  
- `repositories` - Repository metadata

### Operational Features
- ✅ **Progress Reporting** - Real-time progress updates
- ✅ **Statistics Tracking** - Detailed cleanup metrics
- ✅ **Comprehensive Logging** - File and console logging
- ✅ **Cross-Platform** - Works on Windows, macOS, Linux

## 📊 Usage Examples

### Development Workflow
```bash
# Before starting CRUD testing
python cosmos_cleanup.py --confirm --backup

# Quick cleanup between test runs
python cosmos_cleanup.py --force --no-backup
```

### CI/CD Pipeline
```bash
# Automated cleanup in build pipeline
python cosmos_cleanup.py --force --no-backup
```

### Testing the Cleanup
```bash
# Run all tests
python tests/test_cosmos_cleanup.py

# Unit tests only
python tests/test_cosmos_cleanup.py --unit

# Integration tests only  
python tests/test_cosmos_cleanup.py --integration

# With coverage report
python tests/test_cosmos_cleanup.py --coverage
```

## 🔍 Validation

The cleanup script automatically validates success by:

1. **Document Count Verification** - Ensures all containers are empty
2. **Container Schema Integrity** - Verifies containers still exist with proper structure
3. **Error Reporting** - Reports any cleanup failures
4. **Final Status** - Provides clear success/failure indication

## 📋 Prerequisites

### Environment Setup
1. **Python 3.12+** with required packages:
   ```bash
   pip install azure-cosmos azure-identity pydantic
   ```

2. **Azure Authentication** - One of:
   - Azure CLI logged in (`az login`)
   - Managed Identity (in Azure environments)
   - Service Principal credentials

3. **Environment Variables** - Configure in `.env`:
   ```bash
   AZURE_COSMOS_DB_ENDPOINT=https://your-cosmos.documents.azure.com:443/
   ```

### Configuration Validation
```bash
# Test connection before cleanup
python -c "from src.mosaic_mcp.config.settings import MosaicSettings; s=MosaicSettings(); print(f'Endpoint: {s.azure_cosmos_endpoint}')"
```

## 🔗 Integration with CRUD Tasks

This cleanup task (CRUD-000) is a prerequisite for:

- **CRUD-001** - Git Commit State Tracking
- **CRUD-002** - Branch-Aware Entity Model  
- **CRUD-006** - Comprehensive Integration Test Suite

## 🛡️ Security Considerations

- ✅ **Managed Identity** - Uses Azure DefaultAzureCredential
- ✅ **No Hardcoded Secrets** - All credentials from environment/Azure
- ✅ **Backup Before Delete** - Data recovery capability
- ✅ **Audit Logging** - All operations logged for compliance

## 📈 Next Steps

After successful cleanup:

1. **Verify Empty State** - Check Azure portal for zero documents
2. **Run CRUD Tests** - Execute CRUD-001 and CRUD-002 implementations  
3. **Monitor Performance** - Check ingestion performance with clean baseline
4. **Document Results** - Update task status in Memory MCP

## 🎉 Success Indicators

✅ **All containers empty** (0 documents)  
✅ **Container schemas intact**  
✅ **Backup created** (if requested)  
✅ **No error messages**  
✅ **Validation passed**  

The database is now ready for fresh CRUD testing with branch-aware functionality!
