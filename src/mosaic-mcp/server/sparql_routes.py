"""
FastAPI Routes for SPARQL Query Executor

Provides HTTP endpoints for SPARQL query execution, validation,
and service management in the MCP query server.

Task: OMR-P2-001 - Implement SPARQL Query Executor in Query Server
"""

from typing import Dict, Any
import logging

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from .sparql_service import (
    get_sparql_service,
    SPARQLQueryService,
    SPARQLQueryRequest,
    SPARQLQueryResponse,
    SPARQLValidationResponse,
    SPARQLHealthResponse,
)
from ..plugins.sparql_query_executor import ResultFormat


# Initialize router
router = APIRouter(prefix="/sparql", tags=["SPARQL"])
logger = logging.getLogger(__name__)


@router.post(
    "/query",
    response_model=SPARQLQueryResponse,
    summary="Execute SPARQL Query",
    description="Execute a SPARQL query against the RDF knowledge graph with caching and optimization",
)
async def execute_sparql_query(
    request: SPARQLQueryRequest,
    service: SPARQLQueryService = Depends(get_sparql_service),
) -> SPARQLQueryResponse:
    """
    Execute a SPARQL query with comprehensive error handling and optimization.

    Supports all SPARQL query types (SELECT, CONSTRUCT, ASK, DESCRIBE) with
    result caching, timeout handling, and multiple output formats.

    Args:
        request: SPARQL query request with query string and options
        service: SPARQL query service dependency

    Returns:
        Query execution response with results and metadata

    Raises:
        HTTPException: For various error conditions
    """
    return await service.execute_query(request)


@router.post(
    "/validate",
    response_model=SPARQLValidationResponse,
    summary="Validate SPARQL Query",
    description="Validate SPARQL query syntax without executing it",
)
async def validate_sparql_query(
    query: str, service: SPARQLQueryService = Depends(get_sparql_service)
) -> SPARQLValidationResponse:
    """
    Validate a SPARQL query without executing it.

    Useful for syntax checking and query type detection before execution.

    Args:
        query: SPARQL query string to validate
        service: SPARQL query service dependency

    Returns:
        Validation response with status and error details
    """
    return await service.validate_query(query)


@router.get(
    "/health",
    response_model=SPARQLHealthResponse,
    summary="SPARQL Service Health Check",
    description="Get health status and statistics for the SPARQL query service",
)
async def get_sparql_health(
    service: SPARQLQueryService = Depends(get_sparql_service),
) -> SPARQLHealthResponse:
    """
    Get comprehensive health status of the SPARQL query service.

    Includes service status, graph size, cache connectivity,
    and execution statistics.

    Args:
        service: SPARQL query service dependency

    Returns:
        Health check response with detailed status information
    """
    return await service.get_health()


@router.get(
    "/statistics",
    response_model=Dict[str, Any],
    summary="Query Execution Statistics",
    description="Get detailed statistics about SPARQL query execution performance",
)
async def get_sparql_statistics(
    service: SPARQLQueryService = Depends(get_sparql_service),
) -> Dict[str, Any]:
    """
    Get detailed execution statistics for the SPARQL query service.

    Includes metrics on query counts, cache hit rates, execution times,
    and performance data.

    Args:
        service: SPARQL query service dependency

    Returns:
        Statistics dictionary with performance metrics
    """
    return await service.get_statistics()


@router.post(
    "/admin/clear-cache",
    response_model=Dict[str, str],
    summary="Clear Query Cache",
    description="Clear all cached SPARQL query results",
)
async def clear_sparql_cache(
    background_tasks: BackgroundTasks,
    service: SPARQLQueryService = Depends(get_sparql_service),
) -> Dict[str, str]:
    """
    Clear all cached SPARQL query results.

    This operation clears both the in-memory prepared query cache
    and Redis-based result cache.

    Args:
        background_tasks: FastAPI background tasks
        service: SPARQL query service dependency

    Returns:
        Operation result with status message
    """
    # Clear cache in background to avoid blocking
    background_tasks.add_task(service.clear_cache)

    return {"status": "success", "message": "Cache clearing initiated in background"}


@router.post(
    "/admin/reload-data",
    response_model=Dict[str, str],
    summary="Reload RDF Data",
    description="Reload RDF data from Cosmos DB and clear cache",
)
async def reload_rdf_data(
    background_tasks: BackgroundTasks,
    service: SPARQLQueryService = Depends(get_sparql_service),
) -> Dict[str, str]:
    """
    Reload RDF data from Cosmos DB and clear all caches.

    This operation refreshes the in-memory RDF graph with the latest
    data from Cosmos DB and clears all cached results.

    Args:
        background_tasks: FastAPI background tasks
        service: SPARQL query service dependency

    Returns:
        Operation result with status message
    """
    # Reload data in background to avoid blocking
    background_tasks.add_task(service.reload_data)

    return {"status": "success", "message": "Data reload initiated in background"}


@router.get(
    "/formats",
    response_model=Dict[str, Any],
    summary="Supported Result Formats",
    description="Get list of supported SPARQL result formats",
)
async def get_supported_formats() -> Dict[str, Any]:
    """
    Get information about supported SPARQL result formats.

    Returns:
        Dictionary with format information and usage examples
    """
    return {
        "formats": [format.value for format in ResultFormat],
        "descriptions": {
            "json": "JSON format with SPARQL Results JSON format for SELECT/ASK, JSON-LD for CONSTRUCT/DESCRIBE",
            "xml": "XML format following SPARQL Query Results XML Format specification",
            "csv": "Comma-separated values format (SELECT queries only)",
            "tsv": "Tab-separated values format (SELECT queries only)",
            "txt": "Plain text format for debugging and simple output",
        },
        "recommendations": {
            "SELECT": ["json", "csv", "tsv"],
            "ASK": ["json", "xml"],
            "CONSTRUCT": ["json", "xml"],
            "DESCRIBE": ["json", "xml"],
        },
    }


@router.get(
    "/examples",
    response_model=Dict[str, Any],
    summary="SPARQL Query Examples",
    description="Get example SPARQL queries for different use cases",
)
async def get_query_examples() -> Dict[str, Any]:
    """
    Get example SPARQL queries for common use cases.

    Returns:
        Dictionary with categorized query examples
    """
    return {
        "basic_queries": {
            "select_all": {
                "query": "SELECT ?subject ?predicate ?object WHERE { ?subject ?predicate ?object } LIMIT 10",
                "description": "Select first 10 triples from the graph",
            },
            "ask_existence": {
                "query": "ASK { ?s ?p ?o }",
                "description": "Check if the graph contains any triples",
            },
            "construct_subgraph": {
                "query": "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o } LIMIT 5",
                "description": "Construct a subgraph with first 5 triples",
            },
        },
        "entity_queries": {
            "find_entity_properties": {
                "query": "SELECT ?property ?value WHERE { <http://example.org/entity> ?property ?value }",
                "description": "Find all properties and values for a specific entity",
            },
            "find_entities_by_type": {
                "query": "SELECT ?entity WHERE { ?entity <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Type> }",
                "description": "Find all entities of a specific type",
            },
        },
        "code_analysis_queries": {
            "find_functions": {
                "query": "SELECT ?function ?name WHERE { ?function <http://mosaic.org/ontology/function_name> ?name }",
                "description": "Find all functions and their names",
            },
            "find_dependencies": {
                "query": "SELECT ?module ?dependency WHERE { ?module <http://mosaic.org/ontology/imports> ?dependency }",
                "description": "Find module dependencies",
            },
        },
        "metadata_queries": {
            "count_triples": {
                "query": "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }",
                "description": "Count total number of triples in the graph",
            },
            "list_predicates": {
                "query": "SELECT DISTINCT ?predicate WHERE { ?s ?predicate ?o }",
                "description": "List all distinct predicates in the graph",
            },
        },
    }


# Error handlers
@router.exception_handler(HTTPException)
async def sparql_http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler for SPARQL endpoints."""
    logger.warning(f"SPARQL HTTP error {exc.status_code}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "status_code": exc.status_code,
                "message": exc.detail,
                "endpoint": str(request.url),
            }
        },
    )


@router.exception_handler(Exception)
async def sparql_general_exception_handler(request, exc: Exception):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected SPARQL error: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "type": "internal_error",
                "status_code": 500,
                "message": "An unexpected error occurred",
                "endpoint": str(request.url),
            }
        },
    )


# Startup and shutdown events
@router.on_event("startup")
async def startup_sparql_service():
    """Initialize SPARQL service on startup."""
    try:
        logger.info("Initializing SPARQL query service...")
        service = get_sparql_service()
        await service.initialize()
        logger.info("SPARQL query service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SPARQL service: {str(e)}")
        raise


@router.on_event("shutdown")
async def shutdown_sparql_service():
    """Shutdown SPARQL service on application shutdown."""
    try:
        logger.info("Shutting down SPARQL query service...")
        service = get_sparql_service()
        await service.shutdown()
        logger.info("SPARQL query service shutdown completed")
    except Exception as e:
        logger.error(f"Error during SPARQL service shutdown: {str(e)}")
