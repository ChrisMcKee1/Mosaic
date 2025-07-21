# Search and Retrieval Architecture

This document details the hybrid search architecture and OmniRAG pattern implementation for the Mosaic MCP Tool.

## Hybrid Search Architecture (FR-5)

```mermaid
flowchart TD
    START([Search Query]) --> EMBEDDING[Generate Query Embedding<br/>Azure OpenAI text-embedding-3-small]
    
    EMBEDDING --> PARALLEL{Parallel Search}
    
    PARALLEL --> VECTOR[Vector Similarity Search<br/>Cosmos DB Vector Index]
    PARALLEL --> KEYWORD[Keyword Search<br/>Cosmos DB SQL Query]
    
    VECTOR --> VRESULTS[Vector Results<br/>Semantic Relevance]
    KEYWORD --> KRESULTS[Keyword Results<br/>Lexical Relevance]
    
    VRESULTS --> AGGREGATE[Aggregate & Deduplicate<br/>Combine Results by ID]
    KRESULTS --> AGGREGATE
    
    AGGREGATE --> SCORE[Score & Rank<br/>Weighted Combination]
    SCORE --> FINAL[Final Result Set<br/>Document List]
    
    subgraph "OmniRAG Pattern"
        COSMOS[(Unified Cosmos DB<br/>NoSQL + Vector Search)]
        KEYWORD -.-> COSMOS
        VECTOR -.-> COSMOS
    end
    
    style PARALLEL fill:#e3f2fd
    style AGGREGATE fill:#f3e5f5
    style COSMOS fill:#fff3e0
```

## Sample Query Flow

1. **Input**: "authentication patterns in microservices"
2. **Embedding**: Convert to 1536-dimensional vector using Azure OpenAI
3. **Vector Search**: Find semantically similar documents in Cosmos DB
4. **Keyword Search**: SQL query for documents containing "authentication", "microservices"
5. **Aggregation**: Combine and deduplicate by document ID
6. **Scoring**: Weight vector results higher, merge with keyword matches
7. **Output**: Ranked list of relevant documents

## OmniRAG Pattern Implementation (FR-6)

```mermaid
graph TB
    subgraph "Traditional vs OmniRAG"
        TRADITIONAL[Traditional: Separate Services<br/>Vector DB + Graph DB + Search]
        OMNIRAG[OmniRAG: Unified Backend<br/>Single Cosmos DB Service]
    end
    
    subgraph "Document Structure"
        DOC["Document Example:<br/>{<br/>  id: 'pypi_flask',<br/>  libtype: 'pypi',<br/>  libname: 'flask',<br/>  developers: ['contact@palletsprojects.com'],<br/>  dependency_ids: ['pypi_werkzeug', 'pypi_jinja2'],<br/>  used_by_lib: ['pypi_flask_sqlalchemy'],<br/>  embedding: [0.012, ..., -0.045],<br/>  content: 'Flask is a micro web framework...'<br/>}"]
    end
    
    subgraph "Query Operations"
        Q1[Vector Search:<br/>VectorDistance function]
        Q2[Keyword Search:<br/>CONTAINS operations]
        Q3[Graph Traversal:<br/>dependency_ids arrays]
        Q4[Full-Text Search:<br/>Standard SQL queries]
    end
    
    OMNIRAG --> DOC
    DOC --> Q1
    DOC --> Q2
    DOC --> Q3
    DOC --> Q4
    
    style OMNIRAG fill:#fff3e0
    style DOC fill:#e8f5e8
```

## Benefits of OmniRAG Pattern

- **Simplified Architecture**: Single database for all data types
- **Reduced Complexity**: No need for separate vector, graph, and document stores
- **Unified Queries**: Single SQL API for all operations
- **Cost Efficiency**: One service instead of multiple managed services
- **Consistent Performance**: Single connection pool and optimization

## Sample Graph Query

```sql
-- Find all dependencies of Flask
SELECT c.libname, c.dependency_ids 
FROM c 
WHERE c.id = 'pypi_flask'

-- Find all libraries that depend on Flask
SELECT c.libname 
FROM c 
WHERE ARRAY_CONTAINS(c.dependency_ids, 'pypi_flask')
```

## RetrievalPlugin Functions

### hybrid_search(query: str, limit: int = 10)

**Purpose**: Performs parallel vector and keyword search

**Implementation**:

1. Generate query embedding using Azure OpenAI
2. Execute vector similarity search in Cosmos DB
3. Execute keyword search using SQL queries
4. Aggregate and deduplicate results
5. Apply relevance scoring and ranking

**Returns**: List of Document objects with relevance scores

### query_code_graph(library_id: str, relationship_type: str)

**Purpose**: Queries embedded graph relationships using OmniRAG pattern

**Implementation**:

1. Query NoSQL document by library ID
2. Extract relationship arrays (dependency_ids, used_by_lib)
3. Follow relationships to build graph context
4. Return connected library nodes with metadata

**Returns**: List of LibraryNode objects with relationship context

### aggregate_candidates(results: List[List[Document]])

**Purpose**: Combines and deduplicates results from multiple retrieval methods

**Implementation**:

1. Merge results from different retrieval sources
2. Deduplicate by document ID
3. Combine relevance scores using weighted average
4. Sort by final relevance score

**Returns**: Unified list of unique documents with combined scores

## Performance Considerations

### Vector Search Optimization

- **Index Configuration**: Optimized vector index for 1536-dimensional embeddings
- **Similarity Metrics**: Cosine similarity for semantic relevance
- **Batch Processing**: Efficient batch operations for multiple queries

### Keyword Search Optimization

- **Full-Text Indexing**: Optimized text indexes for keyword matching
- **Query Optimization**: Efficient SQL query patterns
- **Result Caching**: Redis caching for frequently accessed results

### Graph Query Optimization

- **Embedded Relationships**: Direct array queries instead of graph traversals
- **Denormalized Data**: Optimized for read performance
- **Batch Operations**: Efficient bulk relationship queries

## Related Documentation

- **[Semantic Reranking](semantic-reranking.md)** - Context refinement processes
- **[System Overview](system-overview.md)** - High-level architecture
- **[Azure Infrastructure](azure-infrastructure.md)** - Infrastructure deployment
