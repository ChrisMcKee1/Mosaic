# AI-Powered Ingestion Service Refactor Plan

## Executive Summary

This document outlines the comprehensive refactoring of the Mosaic Ingestion Service from a monolithic plugin architecture to a specialized AI agent team powered by Magentic. This transformation will create a more robust, modular, and scalable ingestion pipeline that leverages AI to solve complex context engineering challenges.

## Architectural Vision

### Current State
- Single `IngestionPlugin` handles all repository processing
- Monolithic approach with limited scalability
- Basic AST parsing without AI enrichment

### Target State
- Team of 5 specialized AI agents orchestrated by Magentic
- Each agent is an expert in its domain
- AI-powered context enrichment and relationship mapping
- "Golden Node" schema for rich semantic representation

## AI Agent Team Architecture

### 1. GitSleuth Agent 🕵️ (Source Control Expert)
**Persona**: Meticulous version control specialist
**Core Responsibility**: Git repository interactions and state management

**Functions to Implement**:
- `clone_repository(url: str) -> Path`
- `get_diff_from_main(branch: str) -> list[str]`
- `get_file_content_at_commit(file_path: str, commit_hash: str) -> str`

### 2. CodeParser Agent 👩‍💻 (Polyglot Programmer)
**Persona**: Expert programmer fluent in multiple languages
**Core Responsibility**: Tree-sitter AST parsing and entity extraction

**Functions to Implement**:
- `parse_file(file_path: str) -> list[CodeEntity]`
- `extract_entities(file_path: str) -> list[CodeEntity]`

### 3. GraphArchitect Agent 🏛️ (Semantic Analyst)
**Persona**: Senior software architect with big-picture understanding
**Core Responsibility**: Building meaningful relationships in the knowledge graph

**Functions to Implement**:
- `map_entities_to_graph(entities: list[CodeEntity]) -> None`
- `find_relationships(entity: CodeEntity, full_entity_map: dict) -> list[Relationship]`
- `identify_inheritance(class_entity: CodeEntity, full_entity_map: dict) -> Relationship`

### 4. DocuWriter Agent ✍️ (AI Technical Writer)
**Persona**: Clear and concise technical writer
**Core Responsibility**: LLM-powered context enrichment

**Functions to Implement**:
- `summarize_code(code_snippet: str, entity_type: str) -> str`
- `estimate_complexity(code_snippet: str) -> int`
- `generate_keywords(code_snippet: str) -> list[str]`
- `enrich_entity(entity: CodeEntity) -> EnrichedCodeEntity`

### 5. GraphAuditor Agent 🩺 (Quality Assurance)
**Persona**: QA engineer obsessed with data integrity
**Core Responsibility**: Self-healing graph maintenance

**Functions to Implement**:
- `find_orphaned_nodes() -> list[CodeEntity]`
- `suggest_inferred_links(entity: CodeEntity) -> list[Relationship]`
- `find_dead_code_references() -> list[Relationship]`

## Golden Node Schema

```json
{
  "id": "src/mosaic/server/auth.py:OAuth2Handler.validate_token",
  "fqn": "mosaic.server.auth.OAuth2Handler.validate_token",
  "entity_type": "method",
  "name": "validate_token",
  "file_path": "src/mosaic/server/auth.py",
  "language": "python",
  "frameworks": ["fastmcp", "pyjwt"],
  "start_line": 55,
  "end_line": 97,
  "code_snippet": "async def validate_token(self, token: str) -> Dict[str, Any]:\\n...",
  "ai_analysis": {
    "summary": "Validates a JWT bearer token using Microsoft Entra ID's public keys",
    "tags": ["authentication", "jwt", "oauth2.1", "security"],
    "complexity_score": 8,
    "confidence": 0.95
  },
  "dependencies": [
    {
      "fqn": "mosaic.config.settings.MosaicSettings",
      "type": "parameter"
    }
  ],
  "embedding": [0.012, -0.045, 0.089],
  "last_modified": "2025-07-22T21:30:00Z",
  "git_commit_hash": "a1b2c3d4e5f6"
}
```

## Research Tasks & Dependencies

### 1. Magentic Framework Research
**Topic**: AI agent orchestration with Magentic
**Why**: Need to understand how to create and coordinate multiple AI agents
**Dependencies**: Python async/await patterns, LLM integration
**Research Sources**: 
- Microsoft Docs MCP for AI agent frameworks and orchestration patterns
- Context7 for Magentic documentation
- GitHub repository examples
- Official documentation

### 2. Microsoft Azure OpenAI Integration
**Topic**: Azure OpenAI Service for agent LLM capabilities
**Why**: DocuWriter and GraphAuditor agents need LLM capabilities
**Dependencies**: Azure SDK, authentication patterns
**Research Sources**:
- Microsoft Docs MCP for Azure OpenAI
- Microsoft Docs MCP for Semantic Kernel integration

### 3. Tree-sitter Multi-language Support
**Topic**: AST parsing for 11 programming languages
**Why**: CodeParser agent needs robust language detection and parsing
**Dependencies**: tree-sitter-python, language-specific grammars
**Research Sources**:
- Context7 for tree-sitter documentation
- Language-specific grammar repositories

### 4. Cosmos DB OmniRAG Pattern
**Topic**: Storing Golden Node entities in unified NoSQL backend
**Why**: Need efficient storage and querying of enriched code entities
**Dependencies**: Azure Cosmos DB SDK, embedding storage
**Research Sources**:
- Microsoft Docs MCP for Cosmos DB NoSQL API
- Microsoft Docs MCP for Cosmos DB data cleanup and container management
- Vector search capabilities documentation

### 5. Data Migration & Cleanup Strategy
**Topic**: Clearing existing Cosmos DB data before agent-based ingestion
**Why**: Prevent conflicts between old monolithic data and new Golden Node entities
**Dependencies**: Cosmos DB container operations, data versioning
**Research Sources**:
- Microsoft Docs MCP for Cosmos DB container deletion and recreation
- Microsoft Docs MCP for data migration best practices
- Local development vs. production cleanup strategies

## Implementation Checklist

### Phase 1: Foundation (Week 1)
- [ ] Research Magentic framework capabilities and patterns
- [ ] Define Python data models for Golden Node schema
- [ ] Create base Agent class with Semantic Kernel integration
- [ ] Set up Azure OpenAI client for agent LLM capabilities
- [ ] Update Cosmos DB schema for Golden Node storage
- [ ] **Data Cleanup**: Clear existing Cosmos DB data and invalidate previous ingestion results

### Phase 2: Agent Implementation (Week 2-3)
- [ ] Implement GitSleuth Agent with GitPython integration
- [ ] Implement CodeParser Agent with tree-sitter parsers
- [ ] Implement GraphArchitect Agent with relationship mapping
- [ ] Implement DocuWriter Agent with LLM enrichment
- [ ] Implement GraphAuditor Agent with quality checks

### Phase 3: Integration & Testing (Week 4)
- [ ] Create Magentic orchestrator for agent coordination
- [ ] Integrate agents with existing ingestion service entry point
- [ ] Update Azure Container Job configuration
- [ ] Add comprehensive logging and error handling
- [ ] Performance testing and optimization

### Phase 4: Documentation & Deployment (Week 5)
- [ ] Update TDD document with new architecture
- [ ] Update Azure infrastructure templates
- [ ] Create deployment scripts and CI/CD pipeline
- [ ] Update CLAUDE.md with new agent responsibilities

## Files to Create/Modify

### New Files
- `src/ingestion_service/agents/base_agent.py` - Base agent class
- `src/ingestion_service/agents/git_sleuth.py` - Git operations agent
- `src/ingestion_service/agents/code_parser.py` - AST parsing agent
- `src/ingestion_service/agents/graph_architect.py` - Relationship mapping agent
- `src/ingestion_service/agents/docu_writer.py` - AI enrichment agent
- `src/ingestion_service/agents/graph_auditor.py` - Quality assurance agent
- `src/ingestion_service/orchestrator.py` - Magentic agent orchestrator
- `src/ingestion_service/models/golden_node.py` - Data models for Golden Node

### Files to Modify
- `src/ingestion_service/main.py` - Update to use agent orchestrator
- `src/ingestion_service/plugins/ingestion.py` - Refactor into agent calls
- `requirements-ingestion.txt` - Add Magentic and dependencies
- `infra/ingestion-service.bicep` - Update resource requirements
- `docs/TDD_UNIFIED.md` - Document new architecture

## Dependencies & Environment

### New Python Dependencies
```
magentic>=0.28.0  # AI agent orchestration
tree-sitter>=0.20.0  # AST parsing
tree-sitter-python>=0.20.0  # Python grammar
tree-sitter-javascript>=0.20.0  # JavaScript grammar
# Additional language grammars as needed
```

### Azure Resources
- Increased memory allocation for ingestion job (4Gi -> 6Gi)
- Enhanced Azure OpenAI quota for agent LLM calls
- Additional Cosmos DB throughput for Golden Node storage

## Quality Assurance & Testing

### Unit Testing Strategy
- Mock external dependencies (Git, LLM, Cosmos DB)
- Test each agent independently
- Validate Golden Node schema compliance
- Test orchestrator coordination logic

### Integration Testing Strategy
- End-to-end repository ingestion with sample repos
- Performance benchmarking against current implementation
- Memory usage and resource consumption analysis
- Agent coordination and error handling validation

## Data Cleanup & Migration Strategy

### Production Environment Cleanup
1. **Container Deletion**: Delete existing Cosmos DB containers to remove all legacy data
2. **Fresh Schema**: Recreate containers with new Golden Node optimized schema
3. **Service Restart**: Restart ingestion service to clear any cached references
4. **Memory Invalidation**: Clear Redis cache and any persistent memory stores

### Local Development Cleanup
1. **Non-persistent Memory**: Local development already handles cleanup automatically
2. **Container Reset**: Simple container recreation for clean state
3. **Debug Mode**: Add flags to skip cleanup during development iterations

### Migration Commands
```bash
# Production cleanup (requires careful coordination)
az cosmosdb sql container delete --account-name mosaic-cosmos --database-name mosaic --name code_entities
az cosmosdb sql container delete --account-name mosaic-cosmos --database-name mosaic --name memory

# Local development cleanup  
docker-compose down -v  # Remove volumes
docker-compose up -d    # Fresh start
```

## Risk Mitigation

### Technical Risks
1. **Magentic Learning Curve**: Prototype with simple agents first
2. **LLM Cost Management**: Implement caching and rate limiting
3. **Increased Complexity**: Maintain clear agent boundaries and interfaces
4. **Performance Impact**: Monitor and optimize agent communication overhead
5. **Data Migration Risks**: Backup existing data before cleanup operations

### Mitigation Strategies
- Incremental rollout with fallback to current implementation
- Comprehensive logging and monitoring
- Load testing before production deployment
- Clear rollback procedures
- **Staged cleanup**: Test data deletion in dev environment first

## Success Criteria

### Functional Requirements
- [ ] All 5 agents implemented and integrated
- [ ] Golden Node schema stored in Cosmos DB
- [ ] Agent orchestration working with Magentic
- [ ] AI enrichment producing high-quality summaries and tags
- [ ] Relationship mapping creating accurate dependency graphs

### Performance Requirements
- [ ] Processing time within 150% of current implementation
- [ ] Memory usage within 200% of current implementation
- [ ] 95% accuracy in entity extraction and relationship mapping
- [ ] AI enrichment confidence scores >0.8 average

### Quality Requirements
- [ ] 90%+ unit test coverage for all agents
- [ ] Integration tests passing for sample repositories
- [ ] Documentation updated and comprehensive
- [ ] Code review approval from technical leadership

## Next Steps

1. **Immediate Actions**:
   - Begin Magentic framework research
   - Define Golden Node data models
   - Set up development environment with new dependencies

2. **This Week**:
   - Complete Phase 1 foundation work
   - Create base agent architecture
   - Prototype GitSleuth agent

3. **This Month**:
   - Complete all agent implementations
   - Full integration testing
   - Production deployment preparation

## ConPort Decision Log Requirements

The following architectural decisions need to be logged in ConPort:

1. **Agent Architecture Decision**: Transition from monolithic to agent-based architecture
2. **Magentic Framework Selection**: Choice of Magentic for AI agent orchestration  
3. **Golden Node Schema Design**: Comprehensive entity representation schema
4. **AI Enrichment Strategy**: LLM-powered context enhancement approach
5. **OmniRAG Integration**: Unified Cosmos DB storage pattern for enriched entities

---

*This document serves as the master plan for the AI-powered ingestion service refactor. All implementation work should reference and update this document as the project progresses.*