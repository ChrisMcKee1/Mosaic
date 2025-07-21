# Mosaic MCP Tool - Architectural Decision Log

## Overview
This document captures critical architectural decisions, progress updates, and remaining tasks for the Mosaic MCP Tool project. This serves as the centralized project memory and context for development work.

---

## DECISION-2025-07-21-001: Critical Architectural Fix - Two-Service Separation

**Date**: 2025-07-21  
**Status**: IMPLEMENTED  
**Type**: ARCHITECTURAL_DECISION  
**Priority**: CRITICAL

### Description
Fixed fundamental architectural flaw of mixing heavy repository ingestion operations with real-time MCP query server. Separated system into two distinct services:

1. **Ingestion Service** (Azure Container App Job)
   - Heavy repository processing with GitPython
   - AST parsing using tree-sitter
   - Cosmos DB integration for OmniRAG pattern
   - Runs as scheduled/triggered job
   - Location: `src/ingestion_service/`

2. **Query Server** (Azure Container App)
   - Lightweight FastMCP service for real-time queries
   - Semantic Kernel plugin architecture
   - Real-time MCP protocol compliance
   - Location: `src/mosaic/`

### Impact
- **Query Server Performance**: Ensures Query Server remains responsive for real-time MCP requests
- **Scalability**: Enables independent scaling of ingestion vs query operations
- **Resource Optimization**: Heavy processing doesn't block lightweight queries
- **Architectural Clarity**: Clear separation of concerns between data ingestion and data retrieval

### Implementation Details
- Created separate Bicep templates: `infra/query-server.bicep` and `infra/ingestion-service.bicep`
- Updated `azure.yaml` for separate containerapp and containerjob services
- Implemented standalone ingestion service with complete repository analysis capabilities
- Removed heavy ingestion operations from query server

---

## PROGRESS-2025-07-21-001: Ingestion Service Architecture Complete

**Date**: 2025-07-21  
**Status**: COMPLETED  
**Type**: IMPLEMENTATION_PROGRESS

### Description
Successfully created standalone ingestion service architecture with the following capabilities:

- **Repository Analysis**: GitPython integration for complete repository traversal
- **AST Parsing**: tree-sitter integration for language-specific code analysis
- **Data Storage**: Direct Cosmos DB integration using OmniRAG pattern
- **Plugin Architecture**: Semantic Kernel IngestionPlugin implementation
- **Containerization**: Complete Dockerfile and deployment configuration

### Key Components Implemented
- `src/ingestion_service/main.py` - Service entry point
- `src/ingestion_service/plugins/ingestion.py` - Core ingestion logic
- `src/ingestion_service/Dockerfile` - Container configuration
- `infra/ingestion-service.bicep` - Azure infrastructure template

---

## PROGRESS-2025-07-21-002: Query Server Separation Complete

**Date**: 2025-07-21  
**Status**: COMPLETED  
**Type**: IMPLEMENTATION_PROGRESS

### Description
Successfully separated query server from heavy ingestion operations:

- **Lightweight FastMCP**: Removed all heavy processing operations
- **Plugin Focus**: RetrievalPlugin now purely for querying operations
- **Real-time Performance**: Optimized for immediate MCP protocol responses
- **Clean Architecture**: Clear separation between data ingestion and data retrieval

### Key Changes
- Removed GitPython dependencies from query server
- Eliminated AST parsing from real-time operations
- Focused RetrievalPlugin on Cosmos DB querying only
- Maintained full MCP protocol compliance

---

## PROGRESS-2025-07-21-003: Bicep Templates Created

**Date**: 2025-07-21  
**Status**: COMPLETED  
**Type**: INFRASTRUCTURE_PROGRESS

### Description
Created separate Azure infrastructure templates for two-service architecture:

- **Query Server Template**: `infra/query-server.bicep`
  - Azure Container App for real-time queries
  - FastMCP service configuration
  - OAuth 2.1 authentication
  - Cosmos DB connection for query operations

- **Ingestion Service Template**: `infra/ingestion-service.bicep`
  - Azure Container App Job for batch processing
  - Heavy computational resources
  - Repository analysis capabilities
  - Cosmos DB connection for data storage

### Integration
Both templates integrated into main deployment pipeline via `infra/main.bicep`

---

## PROGRESS-2025-07-21-004: Azure Developer CLI Updated

**Date**: 2025-07-21  
**Status**: COMPLETED  
**Type**: DEPLOYMENT_PROGRESS

### Description
Updated Azure Developer CLI configuration for two-service deployment:

- **azure.yaml**: Modified to support both containerapp and containerjob services
- **Deployment Pipeline**: Configured for independent service deployment
- **Environment Management**: Proper environment variable configuration for both services

### Key Configuration
- Query Server: Deployed as Container App for real-time availability
- Ingestion Service: Deployed as Container App Job for scheduled processing
- Shared resources: Cosmos DB, Redis, Azure OpenAI accessed by both services

---

## ACTIVE_CONTEXT: High-Priority Remaining Tasks

### TASK-001: Complete Ingestion Plugin Implementation
**Priority**: HIGH  
**Status**: IN_PROGRESS

#### Description
Need to copy remaining language-specific entity extraction methods from original implementation to new ingestion service.

#### Specific Actions Required
- Copy language-specific parsers from original RetrievalPlugin
- Implement complete AST parsing for Python, JavaScript, TypeScript, Go, etc.
- Ensure all entity extraction methods are functional in new service
- Validate language detection and parsing accuracy

#### Files to Update
- `src/ingestion_service/plugins/ingestion.py`
- Language-specific parser modules

---

### TASK-002: Refactor RetrievalPlugin 
**Priority**: HIGH  
**Status**: PENDING

#### Description
Remove any remaining ingestion concerns from RetrievalPlugin, focus purely on querying operations.

#### Specific Actions Required
- Audit RetrievalPlugin for any remaining heavy processing operations
- Ensure all methods are optimized for real-time query responses
- Implement efficient Cosmos DB query patterns
- Remove any file system or repository access code

#### Files to Update
- `src/mosaic/plugins/retrieval.py`
- Related query optimization utilities

---

### TASK-003: Update Documentation
**Priority**: MEDIUM  
**Status**: PENDING

#### Description
Consolidate fragmented architecture documentation into single TDD source of truth.

#### Specific Actions Required
- Review and consolidate `docs/architecture/` directory
- Update TDD document with two-service architecture
- Ensure documentation reflects current implementation
- Remove outdated architectural references

#### Files to Update
- `docs/Mosaic_MCP_Tool_TDD.md`
- `docs/architecture/` consolidation
- Architecture diagram updates

---

### TASK-004: Integration Testing
**Priority**: HIGH  
**Status**: PENDING

#### Description
Validate both services work correctly with OmniRAG pattern and complete end-to-end workflow.

#### Specific Actions Required
- Test ingestion service repository processing
- Validate query server MCP protocol compliance
- Test Cosmos DB integration for both services
- Validate OAuth 2.1 authentication
- Test complete `azd up` deployment workflow

#### Testing Scenarios
- Repository ingestion and data storage
- Real-time query operations
- MCP protocol message handling
- Authentication and authorization
- Service-to-service communication via Cosmos DB

---

## Architecture Status Summary

### Completed ✅
- [x] Two-service architectural separation
- [x] Ingestion service basic architecture
- [x] Query server separation and optimization
- [x] Bicep template creation
- [x] Azure Developer CLI configuration

### In Progress 🔄
- [ ] Complete ingestion plugin implementation
- [ ] Final RetrievalPlugin refactoring
- [ ] Integration testing

### Pending 📋
- [ ] Documentation consolidation
- [ ] End-to-end deployment validation
- [ ] Production readiness checklist

---

**Last Updated**: 2025-07-21  
**Next Review**: After ingestion plugin completion