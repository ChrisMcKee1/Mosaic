"""
Query Server Integration for SPARQL Query Executor

Integrates the SPARQLQueryExecutor into the MCP query server,
providing SPARQL endpoints and handling query requests.

Task: OMR-P2-001 - Implement SPARQL Query Executor in Query Server
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import HTTPException, status
from pydantic import BaseModel, Field, validator
import redis
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

from .sparql_query_executor import (
    SPARQLQueryExecutor,
    ResultFormat,
    SPARQLQueryError,
    SPARQLQueryTimeoutError,
)
from ..config.settings import get_settings


class SPARQLQueryRequest(BaseModel):
    """Request model for SPARQL query execution."""

    query: str = Field(..., description="SPARQL query string", min_length=1)
    format: ResultFormat = Field(default=ResultFormat.JSON, description="Result format")
    timeout: Optional[float] = Field(
        default=None, description="Query timeout in seconds", gt=0, le=60
    )
    cache_ttl: Optional[int] = Field(
        default=None, description="Cache TTL in seconds", gt=0, le=86400
    )
    bindings: Optional[Dict[str, Any]] = Field(
        default=None, description="Variable bindings"
    )
    use_cache: bool = Field(default=True, description="Whether to use result caching")

    @validator("query")
    def validate_query_content(cls, v):
        """Validate query content is meaningful."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class SPARQLQueryResponse(BaseModel):
    """Response model for SPARQL query execution."""

    query_type: str = Field(..., description="Type of SPARQL query executed")
    results: Any = Field(..., description="Query results in requested format")
    execution_time: float = Field(..., description="Execution time in seconds")
    cached: bool = Field(..., description="Whether result was served from cache")
    result_count: int = Field(..., description="Number of results returned")
    format: str = Field(..., description="Result format")
    timestamp: datetime = Field(..., description="Execution timestamp")
    query_hash: str = Field(..., description="Query hash for debugging")


class SPARQLValidationResponse(BaseModel):
    """Response model for SPARQL query validation."""

    valid: bool = Field(..., description="Whether query is valid")
    query_type: Optional[str] = Field(None, description="Detected query type")
    error_message: Optional[str] = Field(None, description="Validation error message")


class SPARQLHealthResponse(BaseModel):
    """Response model for SPARQL executor health check."""

    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    graph_size: int = Field(..., description="Number of triples in graph")
    cache_status: str = Field(..., description="Cache connectivity status")
    statistics: Dict[str, Any] = Field(..., description="Execution statistics")


class SPARQLQueryService:
    """
    Service layer for SPARQL query operations in the MCP query server.

    Provides high-level interface for SPARQL query execution with
    authentication, rate limiting, and comprehensive error handling.
    """

    def __init__(self):
        """Initialize the SPARQL query service."""
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # Initialize clients
        self._cosmos_client: Optional[CosmosClient] = None
        self._redis_client: Optional[redis.Redis] = None
        self._query_executor: Optional[SPARQLQueryExecutor] = None

        # Service state
        self._initialized = False
        self._initialization_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize Azure clients and query executor."""
        async with self._initialization_lock:
            if self._initialized:
                return

            try:
                self.logger.info("Initializing SPARQL query service...")

                # Initialize Azure Cosmos DB client
                credential = DefaultAzureCredential()
                self._cosmos_client = CosmosClient(
                    url=f"https://{self.settings.cosmos_account_name}.documents.azure.com:443/",
                    credential=credential,
                )

                # Initialize Redis client
                if self.settings.redis_host:
                    self._redis_client = redis.Redis(
                        host=self.settings.redis_host,
                        port=self.settings.redis_port,
                        password=self.settings.redis_password,
                        ssl=self.settings.redis_ssl,
                        decode_responses=True,
                    )

                    # Test Redis connection
                    self._redis_client.ping()
                    self.logger.info("Redis connection established")
                else:
                    self.logger.warning("Redis not configured, caching disabled")

                # Initialize SPARQL query executor
                self._query_executor = SPARQLQueryExecutor(
                    cosmos_client=self._cosmos_client,
                    redis_client=self._redis_client,
                    default_timeout=self.settings.sparql_default_timeout,
                    default_cache_ttl=self.settings.sparql_cache_ttl,
                    enable_caching=self._redis_client is not None,
                )

                self._initialized = True
                self.logger.info("SPARQL query service initialized successfully")

            except Exception as e:
                self.logger.error(
                    f"Failed to initialize SPARQL query service: {str(e)}"
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Service initialization failed: {str(e)}",
                )

    async def execute_query(self, request: SPARQLQueryRequest) -> SPARQLQueryResponse:
        """
        Execute a SPARQL query with comprehensive error handling.

        Args:
            request: SPARQL query request

        Returns:
            Query execution response

        Raises:
            HTTPException: For various error conditions
        """
        await self._ensure_initialized()

        try:
            self.logger.info(f"Executing SPARQL query: {request.query[:100]}...")

            # Execute query
            result = await self._query_executor.execute_query(
                query=request.query,
                format=request.format,
                timeout=request.timeout,
                cache_ttl=request.cache_ttl,
                bindings=request.bindings,
                use_cache=request.use_cache,
            )

            # Generate query hash for debugging
            import hashlib

            query_hash = hashlib.md5(request.query.encode()).hexdigest()[:8]

            self.logger.info(
                f"Query executed successfully (hash={query_hash}, cached={result.cached})"
            )

            return SPARQLQueryResponse(
                query_type=result.query_type.value,
                results=result.results,
                execution_time=result.execution_time,
                cached=result.cached,
                result_count=result.result_count,
                format=result.format.value,
                timestamp=result.timestamp,
                query_hash=query_hash,
            )

        except SPARQLQueryTimeoutError as e:
            self.logger.warning(f"Query timeout: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_408_REQUEST_TIMEOUT, detail=str(e)
            )

        except SPARQLQueryError as e:
            self.logger.warning(f"Query error: {str(e)}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        except Exception as e:
            self.logger.error(f"Unexpected error executing query: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query execution failed: {str(e)}",
            )

    async def validate_query(self, query: str) -> SPARQLValidationResponse:
        """
        Validate a SPARQL query without executing it.

        Args:
            query: SPARQL query string to validate

        Returns:
            Validation response
        """
        await self._ensure_initialized()

        try:
            validation_result = self._query_executor.validate_query(query)

            return SPARQLValidationResponse(
                valid=validation_result.valid,
                query_type=(
                    validation_result.query_type.value
                    if validation_result.query_type
                    else None
                ),
                error_message=validation_result.error_message,
            )

        except Exception as e:
            self.logger.error(f"Query validation error: {str(e)}")
            return SPARQLValidationResponse(
                valid=False,
                query_type=None,
                error_message=f"Validation failed: {str(e)}",
            )

    async def get_health(self) -> SPARQLHealthResponse:
        """
        Get health status of the SPARQL query service.

        Returns:
            Health check response
        """
        try:
            await self._ensure_initialized()

            health_data = await self._query_executor.health_check()

            return SPARQLHealthResponse(
                status=health_data["status"],
                timestamp=datetime.fromisoformat(health_data["timestamp"]),
                graph_size=health_data["graph_size"],
                cache_status=health_data["cache_status"],
                statistics=health_data["statistics"],
            )

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return SPARQLHealthResponse(
                status="unhealthy",
                timestamp=datetime.now(),
                graph_size=0,
                cache_status="error",
                statistics={"error": str(e)},
            )

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics from the query executor.

        Returns:
            Statistics dictionary
        """
        await self._ensure_initialized()
        return self._query_executor.get_statistics()

    async def clear_cache(self) -> Dict[str, str]:
        """
        Clear the query cache.

        Returns:
            Operation result
        """
        await self._ensure_initialized()

        try:
            self._query_executor.clear_cache()

            # Also clear Redis cache if available
            if self._redis_client:
                # Clear all SPARQL query cache keys
                keys = self._redis_client.keys("sparql_query:*")
                if keys:
                    self._redis_client.delete(*keys)
                    self.logger.info(
                        f"Cleared {len(keys)} cached query results from Redis"
                    )

            return {"status": "success", "message": "Cache cleared successfully"}

        except Exception as e:
            self.logger.error(f"Cache clear failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def reload_data(self) -> Dict[str, str]:
        """
        Reload RDF data from Cosmos DB.

        Returns:
            Operation result
        """
        await self._ensure_initialized()

        try:
            # Force reload of RDF data
            await self._query_executor._load_rdf_data()

            # Clear cache since data has changed
            await self.clear_cache()

            graph_size = len(self._query_executor.graph)
            self.logger.info(f"RDF data reloaded successfully ({graph_size} triples)")

            return {
                "status": "success",
                "message": f"Data reloaded successfully ({graph_size} triples)",
            }

        except Exception as e:
            self.logger.error(f"Data reload failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def _ensure_initialized(self) -> None:
        """Ensure the service is initialized."""
        if not self._initialized:
            await self.initialize()

    async def shutdown(self) -> None:
        """Shutdown the service and clean up resources."""
        try:
            if self._redis_client:
                self._redis_client.close()

            # Note: CosmosClient doesn't have explicit close method
            self._cosmos_client = None
            self._query_executor = None
            self._initialized = False

            self.logger.info("SPARQL query service shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during service shutdown: {str(e)}")


# Global service instance
_sparql_service: Optional[SPARQLQueryService] = None


def get_sparql_service() -> SPARQLQueryService:
    """
    Get the global SPARQL query service instance.

    Returns:
        SPARQLQueryService instance
    """
    global _sparql_service
    if _sparql_service is None:
        _sparql_service = SPARQLQueryService()
    return _sparql_service


async def initialize_sparql_service() -> None:
    """Initialize the global SPARQL query service."""
    service = get_sparql_service()
    await service.initialize()


async def shutdown_sparql_service() -> None:
    """Shutdown the global SPARQL query service."""
    global _sparql_service
    if _sparql_service:
        await _sparql_service.shutdown()
        _sparql_service = None
