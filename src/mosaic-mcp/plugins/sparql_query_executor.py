"""
SPARQL Query Executor for Mosaic MCP Tool

Implements comprehensive SPARQL query execution with caching, optimization,
and result formatting for the MCP query server.

Task: OMR-P2-001 - Implement SPARQL Query Executor in Query Server
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.sparql import SPARQLError
from rdflib.query import Result
import redis
from azure.cosmos import CosmosClient
from azure.core.exceptions import CosmosHttpResponseError

from ..config.settings import get_settings


class QueryType(Enum):
    """SPARQL query types supported by the executor."""

    SELECT = "SELECT"
    CONSTRUCT = "CONSTRUCT"
    ASK = "ASK"
    DESCRIBE = "DESCRIBE"


class ResultFormat(Enum):
    """Supported result serialization formats."""

    JSON = "json"
    XML = "xml"
    CSV = "csv"
    TSV = "tsv"
    TXT = "txt"


@dataclass
class QueryExecutionResult:
    """Result of SPARQL query execution with metadata."""

    query_type: QueryType
    results: Any
    execution_time: float
    cached: bool
    result_count: int
    format: ResultFormat
    timestamp: datetime


@dataclass
class QueryValidationResult:
    """Result of SPARQL query validation."""

    valid: bool
    query_type: Optional[QueryType]
    error_message: Optional[str]
    parsed_query: Optional[Any]


class SPARQLQueryError(Exception):
    """Custom exception for SPARQL query execution errors."""

    pass


class SPARQLQueryTimeoutError(SPARQLQueryError):
    """Exception raised when query execution times out."""

    pass


class SPARQLQueryExecutor:
    """
    Comprehensive SPARQL query execution system with caching and optimization.

    Supports all SPARQL query types with result caching, timeout handling,
    and multiple output formats for the Mosaic MCP Tool.
    """

    def __init__(
        self,
        cosmos_client: CosmosClient,
        redis_client: Optional[redis.Redis] = None,
        default_timeout: float = 5.0,
        default_cache_ttl: int = 3600,
        enable_caching: bool = True,
    ):
        """
        Initialize the SPARQL Query Executor.

        Args:
            cosmos_client: Azure Cosmos DB client for RDF data access
            redis_client: Redis client for result caching
            default_timeout: Default query timeout in seconds
            default_cache_ttl: Default cache TTL in seconds
            enable_caching: Whether to enable query result caching
        """
        self.cosmos_client = cosmos_client
        self.redis_client = redis_client
        self.default_timeout = default_timeout
        self.default_cache_ttl = default_cache_ttl
        self.enable_caching = enable_caching

        # Initialize logging
        self.logger = logging.getLogger(__name__)

        # RDF graph for query execution
        self.graph = Graph()

        # Query preparation cache
        self._prepared_queries: Dict[str, Any] = {}

        # Performance metrics
        self._execution_stats = {
            "total_queries": 0,
            "cached_hits": 0,
            "cache_misses": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
        }

        self.logger.info("SPARQLQueryExecutor initialized successfully")

    async def execute_query(
        self,
        query: str,
        format: ResultFormat = ResultFormat.JSON,
        timeout: Optional[float] = None,
        cache_ttl: Optional[int] = None,
        bindings: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> QueryExecutionResult:
        """
        Execute a SPARQL query with caching and result formatting.

        Args:
            query: SPARQL query string
            format: Desired result format
            timeout: Query timeout in seconds (uses default if None)
            cache_ttl: Cache TTL in seconds (uses default if None)
            bindings: Variable bindings for the query
            use_cache: Whether to use caching for this query

        Returns:
            QueryExecutionResult with formatted results and metadata

        Raises:
            SPARQLQueryError: If query execution fails
            SPARQLQueryTimeoutError: If query execution times out
        """
        start_time = time.time()
        timeout = timeout or self.default_timeout
        cache_ttl = cache_ttl or self.default_cache_ttl

        self.logger.info(
            f"Executing SPARQL query with timeout={timeout}s, format={format.value}"
        )

        try:
            # Generate cache key
            cache_key = self._generate_cache_key(query, bindings, format)

            # Check cache first
            if use_cache and self.enable_caching and self.redis_client:
                cached_result = await self._get_cached_result(cache_key)
                if cached_result:
                    self._execution_stats["cached_hits"] += 1
                    self._execution_stats["total_queries"] += 1

                    execution_time = time.time() - start_time
                    self.logger.info(
                        f"Cache hit for query (execution_time={execution_time:.3f}s)"
                    )

                    return QueryExecutionResult(
                        query_type=QueryType(cached_result["query_type"]),
                        results=cached_result["results"],
                        execution_time=execution_time,
                        cached=True,
                        result_count=cached_result["result_count"],
                        format=format,
                        timestamp=datetime.fromisoformat(cached_result["timestamp"]),
                    )

            # Validate query
            validation_result = self.validate_query(query)
            if not validation_result.valid:
                raise SPARQLQueryError(
                    f"Invalid query: {validation_result.error_message}"
                )

            # Load RDF data from Cosmos DB
            await self._load_rdf_data()

            # Prepare and execute query with timeout
            try:
                result = await asyncio.wait_for(
                    self._execute_sparql_query(
                        query, bindings, validation_result.query_type
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                raise SPARQLQueryTimeoutError(
                    f"Query execution timed out after {timeout} seconds"
                )

            # Format result
            formatted_result = self._format_result(
                result, format, validation_result.query_type
            )

            # Calculate metrics
            execution_time = time.time() - start_time
            result_count = self._get_result_count(result, validation_result.query_type)

            # Cache result
            if use_cache and self.enable_caching and self.redis_client:
                await self._cache_result(
                    cache_key,
                    validation_result.query_type,
                    formatted_result,
                    result_count,
                    cache_ttl,
                )
                self._execution_stats["cache_misses"] += 1

            # Update statistics
            self._execution_stats["total_queries"] += 1
            self._execution_stats["total_execution_time"] += execution_time
            self._execution_stats["average_execution_time"] = (
                self._execution_stats["total_execution_time"]
                / self._execution_stats["total_queries"]
            )

            self.logger.info(
                f"Query executed successfully (execution_time={execution_time:.3f}s, result_count={result_count})"
            )

            return QueryExecutionResult(
                query_type=validation_result.query_type,
                results=formatted_result,
                execution_time=execution_time,
                cached=False,
                result_count=result_count,
                format=format,
                timestamp=datetime.now(),
            )

        except SPARQLQueryTimeoutError:
            raise
        except SPARQLQueryError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during query execution: {str(e)}")
            raise SPARQLQueryError(f"Query execution failed: {str(e)}")

    def validate_query(self, query: str) -> QueryValidationResult:
        """
        Validate a SPARQL query and determine its type.

        Args:
            query: SPARQL query string to validate

        Returns:
            QueryValidationResult with validation status and metadata
        """
        try:
            # Parse query to validate syntax
            parsed_query = prepareQuery(query)

            # Determine query type
            query_upper = query.strip().upper()
            if query_upper.startswith("SELECT"):
                query_type = QueryType.SELECT
            elif query_upper.startswith("CONSTRUCT"):
                query_type = QueryType.CONSTRUCT
            elif query_upper.startswith("ASK"):
                query_type = QueryType.ASK
            elif query_upper.startswith("DESCRIBE"):
                query_type = QueryType.DESCRIBE
            else:
                return QueryValidationResult(
                    valid=False,
                    query_type=None,
                    error_message="Unsupported query type",
                    parsed_query=None,
                )

            self.logger.debug(f"Query validated successfully as {query_type.value}")

            return QueryValidationResult(
                valid=True,
                query_type=query_type,
                error_message=None,
                parsed_query=parsed_query,
            )

        except SPARQLError as e:
            self.logger.warning(f"SPARQL validation error: {str(e)}")
            return QueryValidationResult(
                valid=False, query_type=None, error_message=str(e), parsed_query=None
            )
        except Exception as e:
            self.logger.error(f"Unexpected validation error: {str(e)}")
            return QueryValidationResult(
                valid=False,
                query_type=None,
                error_message=f"Validation failed: {str(e)}",
                parsed_query=None,
            )

    def optimize_query(self, query: str) -> str:
        """
        Optimize a SPARQL query for better performance.

        Args:
            query: Original SPARQL query string

        Returns:
            Optimized SPARQL query string
        """
        try:
            # Basic query optimization strategies
            optimized = query.strip()

            # Remove unnecessary whitespace
            optimized = " ".join(optimized.split())

            # Cache prepared query for reuse
            cache_key = hashlib.md5(optimized.encode()).hexdigest()
            if cache_key not in self._prepared_queries:
                self._prepared_queries[cache_key] = prepareQuery(optimized)

            self.logger.debug(f"Query optimized and cached with key: {cache_key}")

            return optimized

        except Exception as e:
            self.logger.warning(f"Query optimization failed: {str(e)}")
            return query  # Return original query if optimization fails

    async def _execute_sparql_query(
        self, query: str, bindings: Optional[Dict[str, Any]], query_type: QueryType
    ) -> Result:
        """
        Execute SPARQL query against the RDF graph.

        Args:
            query: SPARQL query string
            bindings: Variable bindings for the query
            query_type: Type of SPARQL query

        Returns:
            Raw SPARQL query result
        """
        try:
            # Optimize query
            optimized_query = self.optimize_query(query)

            # Execute query with bindings
            if bindings:
                result = self.graph.query(optimized_query, initBindings=bindings)
            else:
                result = self.graph.query(optimized_query)

            return result

        except SPARQLError as e:
            raise SPARQLQueryError(f"SPARQL execution error: {str(e)}")
        except Exception as e:
            raise SPARQLQueryError(f"Query execution failed: {str(e)}")

    async def _load_rdf_data(self) -> None:
        """
        Load RDF data from Cosmos DB into the query graph.
        """
        try:
            # Get database and container references
            settings = get_settings()
            database = self.cosmos_client.get_database_client(
                settings.cosmos_database_name
            )
            container = database.get_container_client("rdf_triples")

            # Query all RDF triples
            query = "SELECT * FROM c WHERE c.entity_type = 'rdf_triple'"
            items = container.query_items(
                query=query, enable_cross_partition_query=True
            )

            # Clear existing graph data
            self.graph = Graph()

            # Load triples into graph
            triple_count = 0
            for item in items:
                if "subject" in item and "predicate" in item and "object" in item:
                    self.graph.add((item["subject"], item["predicate"], item["object"]))
                    triple_count += 1

            self.logger.info(f"Loaded {triple_count} RDF triples from Cosmos DB")

        except CosmosHttpResponseError as e:
            self.logger.error(f"Cosmos DB error loading RDF data: {str(e)}")
            raise SPARQLQueryError(f"Failed to load RDF data: {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading RDF data: {str(e)}")
            raise SPARQLQueryError(f"Data loading failed: {str(e)}")

    def _format_result(
        self, result: Result, format: ResultFormat, query_type: QueryType
    ) -> Any:
        """
        Format SPARQL query result according to specified format.

        Args:
            result: Raw SPARQL query result
            format: Desired output format
            query_type: Type of SPARQL query

        Returns:
            Formatted result data
        """
        try:
            if query_type == QueryType.ASK:
                # ASK queries return boolean
                bool_result = bool(result)
                if format == ResultFormat.JSON:
                    return {"boolean": bool_result}
                elif format == ResultFormat.XML:
                    return f'<?xml version="1.0"?><sparql xmlns="http://www.w3.org/2005/sparql-results#"><head></head><boolean>{str(bool_result).lower()}</boolean></sparql>'
                else:
                    return str(bool_result)

            elif query_type == QueryType.SELECT:
                # SELECT queries return variable bindings
                if format == ResultFormat.JSON:
                    return self._format_select_json(result)
                elif format == ResultFormat.XML:
                    return result.serialize(format="xml")
                elif format == ResultFormat.CSV:
                    return self._format_select_csv(result)
                elif format == ResultFormat.TSV:
                    return self._format_select_tsv(result)
                else:
                    return str(result.serialize())

            elif query_type in [QueryType.CONSTRUCT, QueryType.DESCRIBE]:
                # CONSTRUCT/DESCRIBE queries return RDF graph
                if format == ResultFormat.JSON:
                    return result.serialize(format="json-ld")
                elif format == ResultFormat.XML:
                    return result.serialize(format="xml")
                else:
                    return result.serialize(format="turtle")

            else:
                return str(result)

        except Exception as e:
            self.logger.warning(f"Result formatting error: {str(e)}")
            return str(result)  # Fallback to string representation

    def _format_select_json(self, result: Result) -> Dict[str, Any]:
        """Format SELECT query result as JSON."""
        variables = list(result.vars) if result.vars else []
        bindings = []

        for row in result:
            binding = {}
            for var in variables:
                if var in row and row[var] is not None:
                    binding[str(var)] = {
                        "type": (
                            "uri"
                            if hasattr(row[var], "n3") and row[var].n3().startswith("<")
                            else "literal"
                        ),
                        "value": str(row[var]),
                    }
            bindings.append(binding)

        return {
            "head": {"vars": [str(var) for var in variables]},
            "results": {"bindings": bindings},
        }

    def _format_select_csv(self, result: Result) -> str:
        """Format SELECT query result as CSV."""
        variables = list(result.vars) if result.vars else []
        lines = [",".join(str(var) for var in variables)]

        for row in result:
            values = []
            for var in variables:
                if var in row and row[var] is not None:
                    # Escape CSV values that contain commas or quotes
                    value = str(row[var])
                    if "," in value or '"' in value:
                        value = '"' + value.replace('"', '""') + '"'
                    values.append(value)
                else:
                    values.append("")
            lines.append(",".join(values))

        return "\n".join(lines)

    def _format_select_tsv(self, result: Result) -> str:
        """Format SELECT query result as TSV."""
        variables = list(result.vars) if result.vars else []
        lines = ["\t".join(str(var) for var in variables)]

        for row in result:
            values = []
            for var in variables:
                if var in row and row[var] is not None:
                    values.append(str(row[var]))
                else:
                    values.append("")
            lines.append("\t".join(values))

        return "\n".join(lines)

    def _get_result_count(self, result: Result, query_type: QueryType) -> int:
        """Get the count of results from a SPARQL query result."""
        try:
            if query_type == QueryType.ASK:
                return 1  # ASK queries always return exactly one boolean result
            elif query_type == QueryType.SELECT:
                return len(list(result))
            elif query_type in [QueryType.CONSTRUCT, QueryType.DESCRIBE]:
                return len(result)  # Graph length
            else:
                return 0
        except Exception:
            return 0

    def _generate_cache_key(
        self, query: str, bindings: Optional[Dict[str, Any]], format: ResultFormat
    ) -> str:
        """Generate a cache key for query results."""
        # Create a unique key based on query, bindings, and format
        key_data = {
            "query": query.strip(),
            "bindings": bindings or {},
            "format": format.value,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"sparql_query:{hashlib.sha256(key_string.encode()).hexdigest()}"

    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached query result."""
        if not self.redis_client:
            return None

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            self.logger.warning(f"Cache retrieval error: {str(e)}")

        return None

    async def _cache_result(
        self,
        cache_key: str,
        query_type: QueryType,
        result: Any,
        result_count: int,
        ttl: int,
    ) -> None:
        """Cache query result with TTL."""
        if not self.redis_client:
            return

        try:
            cache_data = {
                "query_type": query_type.value,
                "results": result,
                "result_count": result_count,
                "timestamp": datetime.now().isoformat(),
            }

            self.redis_client.setex(cache_key, ttl, json.dumps(cache_data, default=str))

        except Exception as e:
            self.logger.warning(f"Cache storage error: {str(e)}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get query execution statistics."""
        return {
            **self._execution_stats,
            "cache_hit_rate": (
                self._execution_stats["cached_hits"]
                / max(self._execution_stats["total_queries"], 1)
            )
            * 100,
            "prepared_queries_count": len(self._prepared_queries),
        }

    def clear_cache(self) -> None:
        """Clear the prepared query cache."""
        self._prepared_queries.clear()
        self.logger.info("Prepared query cache cleared")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the query executor."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "graph_size": len(self.graph),
            "prepared_queries": len(self._prepared_queries),
            "statistics": self.get_statistics(),
        }

        # Test basic functionality
        try:
            test_query = "ASK { ?s ?p ?o }"
            validation = self.validate_query(test_query)
            health_status["query_validation"] = validation.valid
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)

        # Test cache connectivity
        if self.redis_client:
            try:
                self.redis_client.ping()
                health_status["cache_status"] = "connected"
            except Exception as e:
                health_status["cache_status"] = "disconnected"
                health_status["cache_error"] = str(e)
        else:
            health_status["cache_status"] = "disabled"

        return health_status
