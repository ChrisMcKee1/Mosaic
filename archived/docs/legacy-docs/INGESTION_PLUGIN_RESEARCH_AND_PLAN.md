# IngestionPlugin Research Findings and Implementation Plan

## Research Summary

### Technology Stack Validation (2025)

#### 1. GitPython 2025 Best Practices
- **Library**: GitPython 3.1.43+ (actively maintained, Python 3.8+ support)
- **Authentication**: Token-based authentication for GitHub/GitLab/Azure DevOps
- **Cloning Patterns**:
  ```python
  import git
  from git import Repo
  
  # Secure cloning with token auth
  repo = Repo.clone_from(
      url=f"https://{token}@github.com/owner/repo.git",
      to_path=local_path,
      depth=1  # Shallow clone for performance
  )
  ```
- **Security**: Environment variable token storage, no hardcoded credentials
- **Performance**: Shallow cloning, selective branch/tag checkout

#### 2. Tree-sitter Python Bindings
- **Library**: py-tree-sitter 0.22.0+ with language parsers
- **Multi-language Support**: Validated parsers for Python, JavaScript, TypeScript, Go, Rust, Java, C#
- **AST Parsing Pattern**:
  ```python
  from tree_sitter import Language, Parser
  
  parser = Parser()
  parser.set_language(Language.build_library('build/my-languages.so', ['tree-sitter-python']))
  
  tree = parser.parse(bytes(source_code, "utf8"))
  ```
- **Performance**: Incremental parsing, memory efficient for large codebases
- **Integration**: Direct AST node traversal for dependency extraction

#### 3. Azure OpenAI text-embedding-3-small
- **Model**: text-embedding-3-small (1536 dimensions, optimized cost/performance)
- **API Pattern**:
  ```python
  from openai import AzureOpenAI
  
  client = AzureOpenAI(
      api_key=os.getenv("AZURE_OPENAI_API_KEY"),
      api_version="2024-02-01",
      azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
  )
  
  response = client.embeddings.create(
      input=text_chunks,
      model="text-embedding-3-small"
  )
  ```
- **Limits**: 8,191 input tokens, 3,000 requests/minute (TPM varies by tier)
- **Batching**: Process up to 2048 inputs per request for efficiency

#### 4. Azure Cosmos DB NoSQL Vector Search
- **Service**: Azure Cosmos DB for NoSQL with vector search (GA 2024)
- **Vector Indexing Pattern**:
  ```python
  # Container creation with vector policy
  vector_embedding_policy = {
      "vectorEmbeddings": [{
          "path": "/embedding",
          "dataType": "float32",
          "distanceFunction": "cosine",
          "dimensions": 1536
      }]
  }
  
  indexing_policy = {
      "vectorIndexes": [{
          "path": "/embedding",
          "type": "quantizedFlat"
      }]
  }
  ```
- **Query Pattern**: Hybrid vector + keyword search using NoSQL API
- **Performance**: Quantized indexes for cost optimization, up to 2000 dimensions

#### 5. OmniRAG Pattern Implementation
- **Architecture**: Unified Cosmos DB backend eliminating separate vector/graph databases
- **Document Schema**:
  ```json
  {
      "id": "file_abc123",
      "type": "code_file",
      "content": "file content",
      "embedding": [0.1, 0.2, ...],
      "dependencies": ["dep1", "dep2"],
      "metadata": {
          "language": "python",
          "path": "/src/main.py",
          "repo_id": "repo_xyz"
      }
  }
  ```

## Implementation Plan: IngestionPlugin Missing Methods

### Phase 1: Repository Management (`clone_repository`)

**Implementation Steps:**
1. **Setup GitPython with secure authentication**
   ```python
   def clone_repository(self, repo_url: str, local_path: str, auth_token: str = None) -> bool:
       try:
           # Prepare authenticated URL
           if auth_token:
               parsed_url = urlparse(repo_url)
               auth_url = f"{parsed_url.scheme}://{auth_token}@{parsed_url.netloc}{parsed_url.path}"
           else:
               auth_url = repo_url
           
           # Shallow clone for performance
           repo = Repo.clone_from(
               url=auth_url,
               to_path=local_path,
               depth=1,
               single_branch=True
           )
           return True
       except GitCommandError as e:
           self.logger.error(f"Git clone failed: {e}")
           return False
   ```

2. **Error handling patterns**:
   - Network timeout handling
   - Authentication failure recovery
   - Disk space validation
   - Invalid repository URL detection

3. **Security considerations**:
   - Token sanitization in logs
   - Temporary directory cleanup
   - Permission validation

### Phase 2: Multi-language AST Parsing (`analyze_code_structure`)

**Implementation Steps:**
1. **Tree-sitter integration**
   ```python
   def analyze_code_structure(self, file_path: str) -> Dict[str, Any]:
       language = self._detect_language(file_path)
       parser = self._get_parser(language)
       
       with open(file_path, 'rb') as f:
           source_code = f.read()
       
       tree = parser.parse(source_code)
       
       return {
           "functions": self._extract_functions(tree.root_node),
           "classes": self._extract_classes(tree.root_node),
           "imports": self._extract_imports(tree.root_node),
           "dependencies": self._extract_dependencies(tree.root_node)
       }
   ```

2. **Language detection**:
   - File extension mapping
   - Content-based detection for ambiguous cases
   - Fallback to plain text processing

3. **AST traversal patterns**:
   - Recursive node walking
   - Language-specific extraction rules
   - Error resilience for syntax errors

### Phase 3: Embedding Generation (`generate_embeddings`)

**Implementation Steps:**
1. **Azure OpenAI integration**
   ```python
   async def generate_embeddings(self, text_chunks: List[str]) -> List[List[float]]:
       client = AzureOpenAI(
           api_key=self.config.azure_openai_key,
           api_version="2024-02-01",
           azure_endpoint=self.config.azure_openai_endpoint
       )
       
       # Batch processing for efficiency
       embeddings = []
       for batch in self._batch_texts(text_chunks, batch_size=100):
           response = await client.embeddings.acreate(
               input=batch,
               model="text-embedding-3-small"
           )
           embeddings.extend([item.embedding for item in response.data])
       
       return embeddings
   ```

2. **Text preprocessing**:
   - Chunk size optimization (8000 tokens max)
   - Code comment extraction
   - Docstring preservation
   - Special token handling

3. **Rate limiting and batching**:
   - Exponential backoff for rate limits
   - Optimal batch sizing
   - Progress tracking for large datasets

### Phase 4: Cosmos DB Storage (`store_in_cosmos`)

**Implementation Steps:**
1. **OmniRAG document creation**
   ```python
   async def store_in_cosmos(self, documents: List[Dict[str, Any]]) -> bool:
       container = self.cosmos_client.get_container_client(
           database="mosaic", 
           container="code_repository"
       )
       
       for doc in documents:
           # Prepare OmniRAG document
           cosmos_doc = {
               "id": doc["id"],
               "type": "code_file",
               "content": doc["content"],
               "embedding": doc["embedding"],
               "dependencies": doc.get("dependencies", []),
               "metadata": {
                   "language": doc["language"],
                   "path": doc["path"],
                   "repo_id": doc["repo_id"],
                   "ingested_at": datetime.utcnow().isoformat()
               }
           }
           
           await container.upsert_item(cosmos_doc)
       
       return True
   ```

2. **Vector indexing configuration**:
   - Automatic vector index creation
   - Performance optimization settings
   - Consistency level configuration

3. **Batch operations**:
   - Bulk upsert for performance
   - Transaction handling
   - Conflict resolution

## Error Handling Strategy

### Repository Cloning Errors
- **Git authentication failures**: Retry with different auth methods
- **Network timeouts**: Exponential backoff with circuit breaker
- **Disk space**: Pre-validation and cleanup procedures
- **Invalid URLs**: URL validation and sanitization

### AST Parsing Errors  
- **Syntax errors**: Graceful degradation to regex-based parsing
- **Unsupported languages**: Fallback to basic text analysis
- **File encoding issues**: Multiple encoding detection attempts
- **Large files**: Streaming parser for memory efficiency

### Embedding Generation Errors
- **Rate limiting**: Intelligent backoff with jitter
- **Token limit exceeded**: Automatic text chunking
- **Service unavailable**: Retry with exponential backoff
- **Authentication issues**: Token refresh mechanisms

### Cosmos DB Storage Errors
- **Connection failures**: Connection pooling and retry logic  
- **Throttling**: Adaptive rate limiting based on RU consumption
- **Document conflicts**: Optimistic concurrency control
- **Index issues**: Automatic index policy validation

## Testing Strategy

### Unit Tests
1. **Repository cloning tests**:
   - Mock GitPython operations
   - Test various authentication scenarios
   - Validate error handling paths

2. **AST parsing tests**:
   - Sample files in each supported language
   - Syntax error handling validation
   - Performance benchmarks

3. **Embedding generation tests**:
   - Mock Azure OpenAI responses
   - Batch processing validation
   - Rate limiting simulation

4. **Cosmos DB storage tests**:
   - Mock Cosmos DB client
   - Document format validation
   - Vector index testing

### Integration Tests
1. **End-to-end workflow tests**:
   - Full repository ingestion pipeline
   - Performance with real repositories
   - Error recovery scenarios

2. **Azure service integration**:
   - Real Azure OpenAI calls (small scale)
   - Cosmos DB connectivity
   - Authentication validation

### Performance Tests
1. **Scalability validation**:
   - Large repository handling (1000+ files)
   - Memory usage profiling
   - Concurrent processing limits

2. **Cost optimization**:
   - Embedding API usage tracking
   - Cosmos DB RU consumption
   - Storage efficiency metrics

## Security Considerations

### Authentication and Authorization
- **Azure managed identity**: Primary authentication method
- **Token rotation**: Automatic refresh mechanisms  
- **Least privilege**: Minimal required permissions
- **Audit logging**: All access attempts logged

### Data Protection
- **Encryption in transit**: HTTPS for all API calls
- **Encryption at rest**: Cosmos DB automatic encryption
- **PII detection**: Scan for sensitive data before storage
- **Access controls**: Role-based access to repositories

### Input Validation
- **Repository URL validation**: Prevent malicious URLs
- **File type restrictions**: Allow-list approach
- **Size limits**: Prevent resource exhaustion
- **Content scanning**: Basic malware detection

## Next Steps

1. **Immediate Implementation** (Week 1):
   - Implement `clone_repository` with GitPython
   - Set up basic Tree-sitter parsing infrastructure
   - Create Azure OpenAI embedding client

2. **Core Functionality** (Week 2):
   - Complete `analyze_code_structure` implementation
   - Implement `generate_embeddings` with batching
   - Create Cosmos DB storage layer

3. **Integration and Testing** (Week 3):
   - End-to-end pipeline testing
   - Performance optimization
   - Error handling validation

4. **Production Readiness** (Week 4):
   - Security audit and hardening
   - Monitoring and alerting setup
   - Documentation and deployment

## Success Metrics

- **Functional**: All 4 missing methods implemented and tested
- **Performance**: Handle repositories with 1000+ files efficiently  
- **Reliability**: 99.9% success rate for supported file types
- **Security**: Pass security audit with no critical findings
- **Integration**: Seamless integration with existing Semantic Kernel plugins

---

*Research completed: July 21, 2025*  
*Implementation plan validated against FR-2 (Semantic Kernel) and FR-4 (Azure Native) requirements*