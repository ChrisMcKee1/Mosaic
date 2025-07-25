# Phase 2: SPARQL Integration and Natural Language Translation

## 📋 Phase Overview

**Duration**: 3-4 weeks  
**Team Size**: 2-3 developers  
**Complexity**: High  
**Dependencies**: Phase 1 (RDF Infrastructure) must be completed

This phase implements SPARQL query execution capabilities and natural language to SPARQL translation, enabling the query server to process graph-based queries from the MCP interface.

## 🎯 Phase Objectives

- [ ] Implement SPARQL query executor in Query Server
- [ ] Create natural language to SPARQL translation service
- [ ] Build graph plugin for MCP interface
- [ ] Establish RDF graph synchronization from Cosmos DB
- [ ] Implement query result formatting and serialization
- [ ] Create SPARQL query optimization and caching
- [ ] Integrate with existing retrieval plugin architecture

## 📚 Pre-Implementation Research (Complete Before Starting)

### Required Reading (3-4 days)

1. **SPARQL 1.1 Query Language** (8-10 hours)

   - Link: https://www.w3.org/TR/sparql11-query/
   - Focus: SELECT queries, graph patterns, property paths, aggregation
   - Practice: Write queries for code relationship scenarios

2. **Natural Language to SPARQL Translation** (6-8 hours)

   - Research Paper: https://arxiv.org/abs/2106.09675
   - Focus: Template-based approaches, semantic parsing patterns
   - Practice: Map common questions to SPARQL patterns

3. **Azure OpenAI for Code Understanding** (4-6 hours)

   - Link: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/use-your-data
   - Focus: Prompt engineering for structured output
   - Practice: Generate SPARQL from natural language prompts

4. **CosmosAIGraph SPARQL Implementation** (4-6 hours)

   - Study: https://github.com/AzureCosmosDB/CosmosAIGraph implementation patterns
   - Focus: Query execution, result formatting, performance optimization
   - Practice: Adapt patterns to your architecture

5. **FastMCP Plugin Architecture** (2-3 hours)
   - Review: Your existing plugin implementations
   - Focus: Async plugin patterns, error handling, result serialization
   - Practice: Create simple graph query plugin prototype

### Research Validation Checkpoints

- [ ] Can write complex SPARQL queries for code relationships
- [ ] Understand template-based NL2SPARQL approaches
- [ ] Familiar with Azure OpenAI structured output
- [ ] Can integrate new plugins with FastMCP architecture
- [ ] Have working prototype of NL2SPARQL translation

## 🛠️ Implementation Steps

### Step 1: Query Server SPARQL Infrastructure (Week 1, Days 1-3)

#### 1.1 Install Additional Dependencies

```bash
cd src/mosaic-mcp/
pip install SPARQLWrapper==2.0.0
pip install rdflib==7.1.1  # If not already installed
pip install networkx==3.2.1
pip install pydantic==2.5.0  # For structured outputs

# Update requirements.txt
echo "SPARQLWrapper==2.0.0" >> requirements.txt
echo "networkx==3.2.1" >> requirements.txt
```

#### 1.2 Create SPARQL Processing Structure

```bash
mkdir -p src/mosaic-mcp/sparql
mkdir -p src/mosaic-mcp/tests/sparql

touch src/mosaic-mcp/sparql/__init__.py
touch src/mosaic-mcp/sparql/query_executor.py
touch src/mosaic-mcp/sparql/nl2sparql.py
touch src/mosaic-mcp/sparql/result_formatter.py
touch src/mosaic-mcp/sparql/query_optimizer.py
```

#### 1.3 Environment Configuration Updates

```bash
# Add to .env files
echo "MOSAIC_SPARQL_TIMEOUT_MS=5000" >> .env
echo "MOSAIC_GRAPH_CACHE_SIZE=10000" >> .env
echo "MOSAIC_NL2SPARQL_MODEL=gpt-4" >> .env
echo "MOSAIC_SPARQL_DEBUG=false" >> .env
```

### Step 2: SPARQL Query Executor Implementation (Week 1, Days 4-5)

#### 2.1 Create `query_executor.py`

**Research Reference**: https://rdflib.readthedocs.io/en/stable/intro_to_sparql.html

```python
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import json
from rdflib import Graph, URIRef, Literal
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.results.jsonresults import JSONResultSerializer
import os

logger = logging.getLogger(__name__)

class SPARQLQueryExecutor:
    """
    Executes SPARQL queries against RDF graphs with caching and optimization
    """

    def __init__(self):
        self.graph = Graph()
        self.query_cache = {}
        self.cache_ttl = timedelta(minutes=10)
        self.timeout_ms = int(os.getenv("MOSAIC_SPARQL_TIMEOUT_MS", "5000"))
        self.debug_mode = os.getenv("MOSAIC_SPARQL_DEBUG", "false").lower() == "true"

    async def initialize(self):
        """
        Initialize SPARQL query executor
        """
        logger.info("Initializing SPARQL Query Executor...")

        # Bind common namespaces
        self.graph.bind("code", "http://mosaic.ai/ontology/code#")
        self.graph.bind("python", "http://mosaic.ai/ontology/python#")
        self.graph.bind("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        self.graph.bind("rdfs", "http://www.w3.org/2000/01/rdf-schema#")

        logger.info("SPARQL Query Executor initialized successfully")

    async def load_graph_from_cosmos(self, cosmos_documents: List[Dict]) -> None:
        """
        Load RDF triples from Cosmos DB documents into in-memory graph
        """
        start_time = datetime.now()
        triple_count = 0

        # Clear existing graph
        self.graph = Graph()

        # Re-bind namespaces after clearing
        self.graph.bind("code", "http://mosaic.ai/ontology/code#")
        self.graph.bind("python", "http://mosaic.ai/ontology/python#")
        self.graph.bind("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
        self.graph.bind("rdfs", "http://www.w3.org/2000/01/rdf-schema#")

        for document in cosmos_documents:
            triples = document.get("rdf_triples", [])
            for triple_dict in triples:
                try:
                    # Convert string URIs back to RDFLib objects
                    subject = URIRef(triple_dict["subject"])
                    predicate = URIRef(triple_dict["predicate"])

                    # Handle object type (URI or Literal)
                    obj_value = triple_dict["object"]
                    if isinstance(obj_value, str) and (obj_value.startswith("http://") or obj_value.startswith("file://")):
                        obj = URIRef(obj_value)
                    else:
                        obj = Literal(obj_value)

                    self.graph.add((subject, predicate, obj))
                    triple_count += 1

                except Exception as e:
                    logger.error(f"Failed to load triple: {triple_dict}, error: {e}")

        load_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Loaded {triple_count} RDF triples from {len(cosmos_documents)} documents in {load_time:.2f}s")

    async def execute_query(self, sparql_query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Execute SPARQL query with caching and timeout
        """
        # Check cache first
        cache_key = self._get_cache_key(sparql_query)
        if use_cache and cache_key in self.query_cache:
            cached_result, cached_time = self.query_cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                logger.debug(f"Returning cached result for query: {sparql_query[:50]}...")
                return cached_result

        start_time = datetime.now()

        try:
            # Execute query with timeout
            result = await asyncio.wait_for(
                self._execute_sparql_query(sparql_query),
                timeout=self.timeout_ms / 1000
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Format result
            formatted_result = {
                "status": "success",
                "results": result,
                "execution_time_ms": int(execution_time * 1000),
                "result_count": len(result) if isinstance(result, list) else 1,
                "query": sparql_query if self.debug_mode else None
            }

            # Cache successful results
            if use_cache:
                self.query_cache[cache_key] = (formatted_result, datetime.now())

            logger.info(f"SPARQL query executed successfully in {execution_time:.3f}s, {len(result) if isinstance(result, list) else 1} results")
            return formatted_result

        except asyncio.TimeoutError:
            logger.error(f"SPARQL query timeout after {self.timeout_ms}ms")
            return {
                "status": "timeout",
                "error": f"Query timeout after {self.timeout_ms}ms",
                "query": sparql_query if self.debug_mode else None
            }

        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query": sparql_query if self.debug_mode else None
            }

    async def _execute_sparql_query(self, sparql_query: str) -> List[Dict[str, Any]]:
        """
        Internal SPARQL query execution
        """
        results = []

        try:
            # Prepare and execute query
            query_result = self.graph.query(sparql_query)

            # Convert results to dictionaries
            for row in query_result:
                result_dict = {}
                for var_name in query_result.vars:
                    value = getattr(row, str(var_name), None)
                    if value is not None:
                        result_dict[str(var_name)] = self._convert_rdf_value(value)
                results.append(result_dict)

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

        return results

    def _convert_rdf_value(self, rdf_value) -> str:
        """
        Convert RDF values to serializable strings
        """
        if isinstance(rdf_value, URIRef):
            return str(rdf_value)
        elif isinstance(rdf_value, Literal):
            return str(rdf_value)
        else:
            return str(rdf_value)

    def _get_cache_key(self, query: str) -> str:
        """
        Generate cache key for query
        """
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded RDF graph
        """
        stats = {
            "total_triples": len(self.graph),
            "namespaces": dict(self.graph.namespaces()),
            "subjects": len(set(self.graph.subjects())),
            "predicates": len(set(self.graph.predicates())),
            "objects": len(set(self.graph.objects())),
            "cache_size": len(self.query_cache)
        }

        # Get entity type counts
        entity_counts = await self._get_entity_type_counts()
        stats["entity_counts"] = entity_counts

        return stats

    async def _get_entity_type_counts(self) -> Dict[str, int]:
        """
        Get counts of different entity types in the graph
        """
        count_query = """
        PREFIX code: <http://mosaic.ai/ontology/code#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

        SELECT ?type (COUNT(?entity) AS ?count)
        WHERE {
            ?entity rdf:type ?type .
            FILTER(STRSTARTS(STR(?type), "http://mosaic.ai/ontology/code#"))
        }
        GROUP BY ?type
        ORDER BY DESC(?count)
        """

        try:
            result = await self._execute_sparql_query(count_query)
            return {row["type"].split("#")[-1]: int(row["count"]) for row in result}
        except Exception as e:
            logger.error(f"Failed to get entity counts: {e}")
            return {}

    async def validate_query(self, sparql_query: str) -> Dict[str, Any]:
        """
        Validate SPARQL query syntax without execution
        """
        try:
            # Try to prepare the query
            prepareQuery(sparql_query)
            return {"valid": True, "message": "Query syntax is valid"}
        except Exception as e:
            return {"valid": False, "error": str(e)}

# Global instance
sparql_executor: Optional[SPARQLQueryExecutor] = None

async def get_sparql_executor() -> SPARQLQueryExecutor:
    """
    Get or create global SPARQL executor instance
    """
    global sparql_executor
    if sparql_executor is None:
        sparql_executor = SPARQLQueryExecutor()
        await sparql_executor.initialize()
    return sparql_executor
```

### Step 3: Natural Language to SPARQL Translation (Week 2, Days 1-3)

#### 3.1 Create `nl2sparql.py`

**Research Reference**: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs

```python
import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import re
from pydantic import BaseModel, Field
from openai import AsyncAzureOpenAI

logger = logging.getLogger(__name__)

class SPARQLQuery(BaseModel):
    """
    Structured output model for SPARQL query generation
    """
    sparql: str = Field(description="The generated SPARQL query")
    explanation: str = Field(description="Explanation of what the query does")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    query_type: str = Field(description="Type of query: select, construct, ask, describe")
    estimated_complexity: str = Field(description="Low, Medium, or High complexity")

class NL2SPARQLTranslator:
    """
    Translates natural language queries to SPARQL using Azure OpenAI
    """

    def __init__(self):
        self.client = None
        self.model_name = os.getenv("MOSAIC_NL2SPARQL_MODEL", "gpt-4")
        self.max_tokens = 1000
        self.temperature = 0.1  # Low temperature for consistent structured output

        # Common query patterns and templates
        self.query_patterns = self._load_query_patterns()

    async def initialize(self):
        """
        Initialize NL2SPARQL translator with Azure OpenAI
        """
        logger.info("Initializing NL2SPARQL Translator...")

        try:
            self.client = AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-15-preview",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            logger.info("NL2SPARQL Translator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NL2SPARQL translator: {e}")
            raise

    async def translate_query(self, natural_language_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Translate natural language query to SPARQL
        """
        start_time = datetime.now()

        try:
            # First, try template-based approach for common patterns
            template_result = self._try_template_matching(natural_language_query)
            if template_result:
                logger.info("Used template-based translation")
                return template_result

            # Fall back to AI-based translation
            ai_result = await self._ai_translate_query(natural_language_query, context)

            translation_time = (datetime.now() - start_time).total_seconds()
            ai_result["translated_in_ms"] = int(translation_time * 1000)
            ai_result["translation_method"] = "ai_based"

            return ai_result

        except Exception as e:
            logger.error(f"Query translation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "original_query": natural_language_query
            }

    def _try_template_matching(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Try to match query against predefined templates
        """
        query_lower = query.lower().strip()

        for pattern, template in self.query_patterns.items():
            if re.search(pattern, query_lower):
                # Extract entities from query for template substitution
                entities = self._extract_entities(query, pattern)
                sparql = template.format(**entities)

                return {
                    "status": "success",
                    "sparql": sparql,
                    "explanation": f"Template-based query for pattern: {pattern}",
                    "confidence": 0.9,
                    "query_type": "select",
                    "estimated_complexity": "Low",
                    "translation_method": "template_based"
                }

        return None

    def _extract_entities(self, query: str, pattern: str) -> Dict[str, str]:
        """
        Extract named entities from query based on pattern
        """
        entities = {}

        # Simple entity extraction - enhance based on your needs
        words = query.split()

        # Look for function names (often in quotes or after "function")
        for i, word in enumerate(words):
            if word.lower() in ["function", "method", "class", "module"]:
                if i + 1 < len(words):
                    entities["entity_name"] = words[i + 1].strip("\"'")

        # Look for file paths
        for word in words:
            if "/" in word or "\\" in word or word.endswith(".py"):
                entities["file_path"] = word.strip("\"'")

        # Default values
        entities.setdefault("entity_name", "ENTITY_NAME")
        entities.setdefault("file_path", "FILE_PATH")

        return entities

    async def _ai_translate_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Use Azure OpenAI to translate natural language to SPARQL
        """
        system_prompt = self._build_system_prompt(context)
        user_prompt = self._build_user_prompt(query)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            response_content = response.choices[0].message.content
            result = json.loads(response_content)

            # Validate the response structure
            sparql_query = SPARQLQuery(**result)

            return {
                "status": "success",
                "sparql": sparql_query.sparql,
                "explanation": sparql_query.explanation,
                "confidence": sparql_query.confidence,
                "query_type": sparql_query.query_type,
                "estimated_complexity": sparql_query.estimated_complexity
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            return {
                "status": "error",
                "error": "Invalid JSON response from AI model"
            }
        except Exception as e:
            logger.error(f"AI translation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _build_system_prompt(self, context: Optional[Dict] = None) -> str:
        """
        Build system prompt for SPARQL generation
        """
        base_prompt = """You are a SPARQL query generator for a code analysis system.

ONTOLOGY INFORMATION:
- Namespace: http://mosaic.ai/ontology/code#
- Main Classes: CodeEntity, Function, Class, Module, Library
- Main Properties: definedIn, calls, inheritsFrom, dependsOn, hasName, hasLineNumber

COMMON PREFIXES:
PREFIX code: <http://mosaic.ai/ontology/code#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

RESPONSE FORMAT:
Return a JSON object with these fields:
- sparql: The SPARQL query string
- explanation: What the query does
- confidence: Float between 0.0 and 1.0
- query_type: "select", "construct", "ask", or "describe"
- estimated_complexity: "Low", "Medium", or "High"

EXAMPLE QUERIES:
1. "What functions are defined in main.py?" -> Query for functions with definedIn relationship
2. "What does the authenticate function call?" -> Query for functions with calls relationship
3. "What classes inherit from BaseModel?" -> Query for classes with inheritsFrom relationship

Generate valid SPARQL that follows W3C standards."""

        if context:
            base_prompt += f"\n\nADDITIONAL CONTEXT:\n{json.dumps(context, indent=2)}"

        return base_prompt

    def _build_user_prompt(self, query: str) -> str:
        """
        Build user prompt for specific query
        """
        return f"""Convert this natural language query to SPARQL:

"{query}"

Remember to:
1. Use the correct prefixes
2. Generate valid SPARQL syntax
3. Focus on code relationships and entities
4. Provide a clear explanation
5. Estimate your confidence accurately"""

    def _load_query_patterns(self) -> Dict[str, str]:
        """
        Load common query patterns and their SPARQL templates
        """
        return {
            # Dependencies and relationships
            r"what.*depends.*on\s+(\w+)": """
                PREFIX code: <http://mosaic.ai/ontology/code#>
                SELECT ?dependency ?name
                WHERE {{
                    ?entity code:hasName "{entity_name}" .
                    ?entity code:dependsOn ?dependency .
                    ?dependency code:hasName ?name .
                }}
            """,

            r"what.*(\w+).*depends.*on": """
                PREFIX code: <http://mosaic.ai/ontology/code#>
                SELECT ?dependent ?name
                WHERE {{
                    ?dependent code:dependsOn ?entity .
                    ?entity code:hasName "{entity_name}" .
                    ?dependent code:hasName ?name .
                }}
            """,

            # Function calls
            r"what.*functions.*calls?\s+(\w+)": """
                PREFIX code: <http://mosaic.ai/ontology/code#>
                SELECT ?caller ?name
                WHERE {{
                    ?target code:hasName "{entity_name}" .
                    ?caller code:calls ?target .
                    ?caller code:hasName ?name .
                }}
            """,

            r"what.*does\s+(\w+).*call": """
                PREFIX code: <http://mosaic.ai/ontology/code#>
                SELECT ?called ?name
                WHERE {{
                    ?function code:hasName "{entity_name}" .
                    ?function code:calls ?called .
                    ?called code:hasName ?name .
                }}
            """,

            # Class inheritance
            r"what.*classes.*inherit.*from\s+(\w+)": """
                PREFIX code: <http://mosaic.ai/ontology/code#>
                SELECT ?subclass ?name
                WHERE {{
                    ?parent code:hasName "{entity_name}" .
                    ?subclass code:inheritsFrom ?parent .
                    ?subclass code:hasName ?name .
                }}
            """,

            # Functions in module
            r"what.*functions.*in\s+(.+\\.py)": """
                PREFIX code: <http://mosaic.ai/ontology/code#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                SELECT ?function ?name
                WHERE {{
                    ?function rdf:type code:Function .
                    ?function code:definedIn ?module .
                    ?function code:hasName ?name .
                    FILTER(CONTAINS(STR(?module), "{file_path}"))
                }}
            """,

            # General entity search
            r"show.*me.*(\w+)": """
                PREFIX code: <http://mosaic.ai/ontology/code#>
                SELECT ?entity ?name ?type
                WHERE {{
                    ?entity code:hasName ?name .
                    ?entity rdf:type ?type .
                    FILTER(CONTAINS(LCASE(?name), LCASE("{entity_name}")))
                }}
                LIMIT 10
            """
        }

# Global instance
nl2sparql_translator: Optional[NL2SPARQLTranslator] = None

async def get_nl2sparql_translator() -> NL2SPARQLTranslator:
    """
    Get or create global NL2SPARQL translator instance
    """
    global nl2sparql_translator
    if nl2sparql_translator is None:
        nl2sparql_translator = NL2SPARQLTranslator()
        await nl2sparql_translator.initialize()
    return nl2sparql_translator
```

### Step 4: Graph Plugin for MCP Interface (Week 2, Days 4-5)

#### 4.1 Create `graph_plugin.py`

**Research Reference**: Review your existing plugin architecture in `retrieval_plugin.py`

```python
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from ..sparql.query_executor import get_sparql_executor
from ..sparql.nl2sparql import get_nl2sparql_translator

logger = logging.getLogger(__name__)

class GraphPlugin:
    """
    MCP plugin for graph-based queries using SPARQL
    """

    def __init__(self):
        self.sparql_executor = None
        self.nl2sparql = None
        self.cosmos_client = None  # Will be injected

    async def initialize(self, cosmos_client):
        """
        Initialize graph plugin with dependencies
        """
        logger.info("Initializing Graph Plugin...")

        self.cosmos_client = cosmos_client
        self.sparql_executor = await get_sparql_executor()
        self.nl2sparql = await get_nl2sparql_translator()

        # Load initial graph data
        await self._refresh_graph_data()

        logger.info("Graph Plugin initialized successfully")

    async def query_code_graph(
        self,
        query: str,
        query_type: str = "natural_language",
        limit: int = 20,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Query the code graph using natural language or direct SPARQL

        Args:
            query: Natural language query or SPARQL query
            query_type: "natural_language" or "sparql"
            limit: Maximum number of results
            include_context: Whether to include additional context

        Returns:
            Structured query results
        """
        start_time = datetime.now()

        try:
            # Translate natural language to SPARQL if needed
            if query_type == "natural_language":
                translation_result = await self.nl2sparql.translate_query(query)
                if translation_result.get("status") != "success":
                    return translation_result

                sparql_query = translation_result["sparql"]
                explanation = translation_result["explanation"]
                confidence = translation_result.get("confidence", 0.0)
            else:
                sparql_query = query
                explanation = "Direct SPARQL query execution"
                confidence = 1.0

            # Add LIMIT clause if not present and limit is specified
            if limit and "LIMIT" not in sparql_query.upper():
                sparql_query += f" LIMIT {limit}"

            # Execute SPARQL query
            query_result = await self.sparql_executor.execute_query(sparql_query)

            if query_result.get("status") != "success":
                return query_result

            results = query_result["results"]

            # Enhance results with context if requested
            if include_context and results:
                enhanced_results = await self._enhance_results_with_context(results)
            else:
                enhanced_results = results

            total_time = (datetime.now() - start_time).total_seconds()

            return {
                "status": "success",
                "results": enhanced_results,
                "metadata": {
                    "query_type": query_type,
                    "original_query": query,
                    "sparql_query": sparql_query,
                    "explanation": explanation,
                    "confidence": confidence,
                    "result_count": len(enhanced_results),
                    "execution_time_ms": int(total_time * 1000),
                    "sparql_execution_time_ms": query_result.get("execution_time_ms", 0)
                }
            }

        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "original_query": query
            }

    async def get_entity_relationships(
        self,
        entity_name: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get all relationships for a specific entity
        """
        try:
            # Build relationship query
            if relationship_types:
                type_filter = " | ".join([f"code:{rt}" for rt in relationship_types])
                relationship_path = f"({type_filter})"
            else:
                relationship_path = "?relationship"

            sparql_query = f"""
            PREFIX code: <http://mosaic.ai/ontology/code#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?source ?relationship ?target ?sourceName ?targetName
            WHERE {{
                {{
                    ?source code:hasName "{entity_name}" .
                    ?source {relationship_path} ?target .
                    ?source code:hasName ?sourceName .
                    ?target code:hasName ?targetName .
                }}
                UNION
                {{
                    ?target code:hasName "{entity_name}" .
                    ?source {relationship_path} ?target .
                    ?source code:hasName ?sourceName .
                    ?target code:hasName ?targetName .
                }}
            }}
            LIMIT 50
            """

            return await self.query_code_graph(sparql_query, query_type="sparql")

        except Exception as e:
            logger.error(f"Failed to get entity relationships: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def get_dependency_chain(
        self,
        library_name: str,
        direction: str = "dependencies",
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Get dependency chain for a library (dependencies or dependents)
        """
        try:
            if direction == "dependencies":
                # What does this library depend on?
                sparql_query = f"""
                PREFIX code: <http://mosaic.ai/ontology/code#>

                SELECT ?dependency ?name ?level
                WHERE {{
                    ?library code:hasName "{library_name}" .
                    ?library code:dependsOn{{1,{max_depth}}} ?dependency .
                    ?dependency code:hasName ?name .
                }}
                ORDER BY ?name
                """
            else:
                # What depends on this library?
                sparql_query = f"""
                PREFIX code: <http://mosaic.ai/ontology/code#>

                SELECT ?dependent ?name ?level
                WHERE {{
                    ?library code:hasName "{library_name}" .
                    ?dependent code:dependsOn{{1,{max_depth}}} ?library .
                    ?dependent code:hasName ?name .
                }}
                ORDER BY ?name
                """

            return await self.query_code_graph(sparql_query, query_type="sparql")

        except Exception as e:
            logger.error(f"Failed to get dependency chain: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _enhance_results_with_context(self, results: List[Dict]) -> List[Dict]:
        """
        Enhance SPARQL results with additional context from Cosmos DB
        """
        enhanced_results = []

        for result in results:
            enhanced_result = dict(result)

            # Try to find additional context for entities in the result
            for key, value in result.items():
                if isinstance(value, str) and (value.startswith("file://") or "entity" in key.lower()):
                    # Look up additional context from Cosmos DB
                    context = await self._get_entity_context(value)
                    if context:
                        enhanced_result[f"{key}_context"] = context

            enhanced_results.append(enhanced_result)

        return enhanced_results

    async def _get_entity_context(self, entity_uri: str) -> Optional[Dict]:
        """
        Get additional context for an entity from Cosmos DB
        """
        try:
            # Query Cosmos DB for entity details
            # This is a simplified version - enhance based on your Cosmos DB schema
            query = f"SELECT * FROM c WHERE CONTAINS(c.id, '{entity_uri.split('#')[-1]}') OR CONTAINS(c.name, '{entity_uri.split('#')[-1]}')"

            async for item in self.cosmos_client.query_items(
                query=query,
                enable_cross_partition_query=True
            ):
                return {
                    "type": item.get("type"),
                    "entity_type": item.get("entity_type"),
                    "file_path": item.get("file_path"),
                    "content_preview": item.get("content", "")[:200] if item.get("content") else None
                }

        except Exception as e:
            logger.error(f"Failed to get entity context: {e}")

        return None

    async def _refresh_graph_data(self):
        """
        Refresh RDF graph with latest data from Cosmos DB
        """
        try:
            logger.info("Refreshing graph data from Cosmos DB...")

            # Query all documents with RDF triples
            query = "SELECT * FROM c WHERE c.rdf_triples != null AND ARRAY_LENGTH(c.rdf_triples) > 0"

            documents = []
            async for item in self.cosmos_client.query_items(
                query=query,
                enable_cross_partition_query=True
            ):
                documents.append(item)

            # Load into SPARQL executor
            await self.sparql_executor.load_graph_from_cosmos(documents)

            logger.info(f"Graph data refreshed with {len(documents)} documents")

        except Exception as e:
            logger.error(f"Failed to refresh graph data: {e}")

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded graph
        """
        return await self.sparql_executor.get_graph_statistics()

    async def validate_sparql_query(self, query: str) -> Dict[str, Any]:
        """
        Validate SPARQL query syntax
        """
        return await self.sparql_executor.validate_query(query)

# Plugin instance
graph_plugin: Optional[GraphPlugin] = None

async def get_graph_plugin() -> GraphPlugin:
    """
    Get or create global graph plugin instance
    """
    global graph_plugin
    if graph_plugin is None:
        graph_plugin = GraphPlugin()
    return graph_plugin
```

### Step 5: Integration with Existing MCP Architecture (Week 3, Days 1-2)

#### 5.1 Update Main Server to Include Graph Plugin

**Modify `src/mosaic-mcp/server/main.py`**:

```python
# Add to existing imports
from ..plugins.graph_plugin import get_graph_plugin

class MosaicMCPServer:
    def __init__(self):
        # Existing initialization
        self.graph_plugin = None

    async def initialize(self):
        """Enhanced initialization with graph plugin"""
        # Existing initialization code

        # Initialize graph plugin
        self.graph_plugin = await get_graph_plugin()
        await self.graph_plugin.initialize(self.cosmos_client)

    # Add new MCP tool for graph queries
    @self.mcp.tool("mosaic.graph.query")
    async def graph_query_tool(
        query: str,
        query_type: str = "natural_language",
        limit: int = 20
    ) -> str:
        """
        Query the code graph using natural language or SPARQL

        Args:
            query: Natural language query or SPARQL query
            query_type: "natural_language" or "sparql"
            limit: Maximum number of results
        """
        try:
            result = await self.graph_plugin.query_code_graph(
                query=query,
                query_type=query_type,
                limit=limit
            )

            if result.get("status") == "success":
                return self._format_graph_results(result)
            else:
                return f"Query failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"Graph query error: {str(e)}"

    @self.mcp.tool("mosaic.graph.relationships")
    async def graph_relationships_tool(
        entity_name: str,
        relationship_types: str = "",
        max_depth: int = 2
    ) -> str:
        """
        Get relationships for a specific entity

        Args:
            entity_name: Name of the entity to analyze
            relationship_types: Comma-separated list of relationship types
            max_depth: Maximum relationship traversal depth
        """
        try:
            rel_types = [rt.strip() for rt in relationship_types.split(",")] if relationship_types else None

            result = await self.graph_plugin.get_entity_relationships(
                entity_name=entity_name,
                relationship_types=rel_types,
                max_depth=max_depth
            )

            if result.get("status") == "success":
                return self._format_graph_results(result)
            else:
                return f"Relationship query failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"Relationship query error: {str(e)}"

    def _format_graph_results(self, result: Dict) -> str:
        """
        Format graph query results for MCP response
        """
        if not result.get("results"):
            return "No results found."

        results = result["results"]
        metadata = result.get("metadata", {})

        output = []

        # Add explanation if available
        if metadata.get("explanation"):
            output.append(f"Query: {metadata['explanation']}")
            output.append("")

        # Format results
        output.append(f"Found {len(results)} results:")
        output.append("")

        for i, row in enumerate(results, 1):
            output.append(f"{i}. {self._format_result_row(row)}")

        # Add metadata
        if metadata.get("execution_time_ms"):
            output.append("")
            output.append(f"Executed in {metadata['execution_time_ms']}ms")

        return "\n".join(output)

    def _format_result_row(self, row: Dict) -> str:
        """
        Format a single result row
        """
        # Extract meaningful values and format them nicely
        formatted_values = []
        for key, value in row.items():
            if not key.endswith("_context"):  # Skip context fields for main display
                if isinstance(value, str):
                    # Clean up URIs to show just the name part
                    if value.startswith("http://") or value.startswith("file://"):
                        clean_value = value.split("#")[-1].split("/")[-1]
                    else:
                        clean_value = value
                    formatted_values.append(f"{key}: {clean_value}")
                else:
                    formatted_values.append(f"{key}: {value}")

        return " | ".join(formatted_values)
```

### Step 6: Result Formatting and Optimization (Week 3, Days 3-4)

#### 6.1 Create `result_formatter.py`

```python
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SPARQLResultFormatter:
    """
    Formats SPARQL query results for different output formats
    """

    def __init__(self):
        self.supported_formats = ["json", "table", "graph", "mermaid"]

    def format_results(
        self,
        results: List[Dict],
        format_type: str = "json",
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Format SPARQL results in specified format
        """
        try:
            if format_type == "json":
                return self._format_as_json(results, metadata)
            elif format_type == "table":
                return self._format_as_table(results, metadata)
            elif format_type == "graph":
                return self._format_as_graph(results, metadata)
            elif format_type == "mermaid":
                return self._format_as_mermaid(results, metadata)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

        except Exception as e:
            logger.error(f"Result formatting failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "raw_results": results
            }

    def _format_as_json(self, results: List[Dict], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Format results as structured JSON
        """
        return {
            "status": "success",
            "format": "json",
            "results": results,
            "metadata": metadata or {},
            "result_count": len(results),
            "formatted_at": datetime.now().isoformat()
        }

    def _format_as_table(self, results: List[Dict], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Format results as ASCII table
        """
        if not results:
            return {
                "status": "success",
                "format": "table",
                "table": "No results found",
                "metadata": metadata or {}
            }

        # Get all unique column names
        columns = set()
        for result in results:
            columns.update(result.keys())
        columns = sorted(list(columns))

        # Calculate column widths
        col_widths = {}
        for col in columns:
            col_widths[col] = max(
                len(col),
                max(len(str(result.get(col, ""))) for result in results)
            )

        # Build table
        table_lines = []

        # Header
        header = " | ".join(col.ljust(col_widths[col]) for col in columns)
        table_lines.append(header)
        table_lines.append("-" * len(header))

        # Rows
        for result in results:
            row = " | ".join(
                str(result.get(col, "")).ljust(col_widths[col])
                for col in columns
            )
            table_lines.append(row)

        return {
            "status": "success",
            "format": "table",
            "table": "\n".join(table_lines),
            "columns": columns,
            "metadata": metadata or {}
        }

    def _format_as_graph(self, results: List[Dict], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Format results as graph structure (nodes and edges)
        """
        nodes = set()
        edges = []

        for result in results:
            # Identify nodes and relationships from result
            for key, value in result.items():
                if key in ["source", "target", "function", "class", "module"]:
                    nodes.add(value)
                elif key == "relationship" and "source" in result and "target" in result:
                    edges.append({
                        "source": result["source"],
                        "target": result["target"],
                        "relationship": value
                    })

        return {
            "status": "success",
            "format": "graph",
            "nodes": list(nodes),
            "edges": edges,
            "metadata": metadata or {}
        }

    def _format_as_mermaid(self, results: List[Dict], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Format results as Mermaid diagram syntax
        """
        if not results:
            return {
                "status": "success",
                "format": "mermaid",
                "diagram": "graph TD\n    A[No Results Found]",
                "metadata": metadata or {}
            }

        # Build Mermaid graph
        mermaid_lines = ["graph TD"]
        node_counter = 0
        node_map = {}

        def get_node_id(name: str) -> str:
            if name not in node_map:
                nonlocal node_counter
                node_counter += 1
                node_map[name] = f"N{node_counter}"
            return node_map[name]

        # Process results to create nodes and relationships
        for result in results:
            if "source" in result and "target" in result:
                source_id = get_node_id(result["source"])
                target_id = get_node_id(result["target"])
                relationship = result.get("relationship", "relates to")

                mermaid_lines.append(f"    {source_id}[{result['source']}] -->|{relationship}| {target_id}[{result['target']}]")
            else:
                # Single entity result
                for key, value in result.items():
                    if isinstance(value, str) and len(value) < 50:
                        node_id = get_node_id(value)
                        mermaid_lines.append(f"    {node_id}[{value}]")

        return {
            "status": "success",
            "format": "mermaid",
            "diagram": "\n".join(mermaid_lines),
            "metadata": metadata or {}
        }

# Global instance
result_formatter = SPARQLResultFormatter()

def get_result_formatter() -> SPARQLResultFormatter:
    """
    Get result formatter instance
    """
    return result_formatter
```

## 🧪 Testing Strategy (Week 3, Day 5 & Week 4)

### Test Cases to Implement

#### Unit Tests

```python
# tests/sparql/test_query_executor.py
import pytest
from src.mosaic.sparql.query_executor import SPARQLQueryExecutor

@pytest.mark.asyncio
async def test_sparql_execution():
    executor = SPARQLQueryExecutor()
    await executor.initialize()

    # Load test data
    test_documents = [
        {
            "rdf_triples": [
                {
                    "subject": "file:///test.py#func1",
                    "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                    "object": "http://mosaic.ai/ontology/code#Function"
                }
            ]
        }
    ]

    await executor.load_graph_from_cosmos(test_documents)

    # Test query execution
    query = """
    PREFIX code: <http://mosaic.ai/ontology/code#>
    SELECT ?function WHERE { ?function a code:Function }
    """

    result = await executor.execute_query(query)
    assert result["status"] == "success"
    assert len(result["results"]) > 0

# tests/sparql/test_nl2sparql.py
import pytest
from src.mosaic.sparql.nl2sparql import NL2SPARQLTranslator

@pytest.mark.asyncio
async def test_template_matching():
    translator = NL2SPARQLTranslator()
    await translator.initialize()

    result = translator._try_template_matching("what functions call authenticate")
    assert result is not None
    assert "SPARQL" in result["sparql"] or "sparql" in result["sparql"]
```

#### Integration Tests

```python
# tests/integration/test_graph_plugin.py
import pytest
from src.mosaic_mcp.plugins.graph_plugin import GraphPlugin

@pytest.mark.asyncio
async def test_natural_language_query():
    plugin = GraphPlugin()
    # Mock cosmos client for testing
    await plugin.initialize(mock_cosmos_client)

    result = await plugin.query_code_graph(
        "What functions are defined in main.py?",
        query_type="natural_language"
    )

    assert result["status"] == "success"
    assert "results" in result
    assert "metadata" in result
```

## ✅ Phase 2 Completion Checklist

- [ ] SPARQL query executor implemented and tested
- [ ] Natural language to SPARQL translation working
- [ ] Graph plugin integrated with MCP interface
- [ ] Result formatting supports multiple output types
- [ ] Graph data synchronization from Cosmos DB functional
- [ ] Query caching and optimization implemented
- [ ] All unit tests passing
- [ ] Integration tests demonstrate end-to-end functionality
- [ ] Performance benchmarks meet targets (<100ms for simple queries)
- [ ] Documentation updated with new capabilities

## 🚨 Common Pitfalls and Solutions

### Pitfall 1: SPARQL Query Performance

**Problem**: Complex queries timeout or consume too much memory
**Solution**: Implement query optimization, indexing strategies, and result limits

### Pitfall 2: Natural Language Translation Accuracy

**Problem**: AI generates invalid or incorrect SPARQL
**Solution**: Use template-based fallbacks, validation, and confidence scoring

### Pitfall 3: Graph Synchronization Issues

**Problem**: RDF graph becomes stale or inconsistent with Cosmos DB
**Solution**: Implement incremental updates and consistency checks

### Pitfall 4: Memory Usage with Large Graphs

**Problem**: In-memory RDF graphs consume too much memory
**Solution**: Implement graph partitioning and selective loading

## 📋 Post-Phase 2 Cleanup

1. **Performance Optimization**: Profile and optimize query execution
2. **Query Validation**: Add comprehensive SPARQL validation
3. **Monitoring**: Add metrics for query performance and accuracy
4. **Documentation**: Update API documentation with graph query capabilities
5. **Error Handling**: Improve error messages and fallback strategies

---

**Next Phase**: `phase-3-omnirag-orchestration.md` - Implementing the complete OmniRAG pattern with intelligent query routing
