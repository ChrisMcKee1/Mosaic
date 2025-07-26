"""
FastAPI routes for Natural Language to SPARQL Translation Service.
Provides HTTP API endpoints for NL2SPARQL functionality.
"""

from typing import Dict, List, Optional, Any
import logging

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator

from ..plugins.nl2sparql_service import NL2SPARQLService


logger = logging.getLogger(__name__)

# Global service instance
_nl2sparql_service: Optional[NL2SPARQLService] = None


async def get_nl2sparql_service() -> NL2SPARQLService:
    """Dependency to get NL2SPARQL service instance."""
    global _nl2sparql_service
    if _nl2sparql_service is None:
        _nl2sparql_service = NL2SPARQLService()
        await _nl2sparql_service.initialize()
    return _nl2sparql_service


# Request/Response models for API endpoints
class TranslateRequest(BaseModel):
    """Request model for translation endpoint."""

    query: str = Field(..., description="Natural language query to translate")
    context: Optional[str] = Field(
        None, description="Additional context for translation"
    )
    max_results: Optional[int] = Field(
        100, ge=1, le=10000, description="Maximum number of results"
    )

    @validator("query")
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 2000:
            raise ValueError("Query too long (max 2000 characters)")
        return v.strip()


class TranslateAndExecuteRequest(TranslateRequest):
    """Request model for translate and execute endpoint."""

    execute_query: bool = Field(
        True, description="Whether to execute the generated SPARQL query"
    )
    force_execution: bool = Field(
        False, description="Force execution even with low confidence"
    )


class BatchTranslateRequest(BaseModel):
    """Request model for batch translation endpoint."""

    queries: List[Dict[str, Any]] = Field(
        ..., description="List of queries to translate"
    )
    execute_queries: bool = Field(
        False, description="Whether to execute all generated queries"
    )
    max_batch_size: int = Field(50, ge=1, le=100, description="Maximum batch size")

    @validator("queries")
    def validate_queries(cls, v):
        if not v:
            raise ValueError("Queries list cannot be empty")
        if len(v) > 100:
            raise ValueError("Batch size too large (max 100)")

        for i, query_data in enumerate(v):
            if not isinstance(query_data, dict):
                raise ValueError(f"Query {i} must be a dictionary")
            if "query" not in query_data:
                raise ValueError(f"Query {i} missing 'query' field")
            if not query_data["query"] or not query_data["query"].strip():
                raise ValueError(f"Query {i} cannot be empty")

        return v


class ValidateQueryRequest(BaseModel):
    """Request model for query validation endpoint."""

    sparql_query: str = Field(..., description="SPARQL query to validate")

    @validator("sparql_query")
    def validate_sparql_query(cls, v):
        if not v or not v.strip():
            raise ValueError("SPARQL query cannot be empty")
        return v.strip()


class ClearCacheRequest(BaseModel):
    """Request model for cache clearing endpoint."""

    pattern: Optional[str] = Field(
        None, description="Cache key pattern to clear (default: all)"
    )


# Create router
router = APIRouter(prefix="/nl2sparql", tags=["Natural Language to SPARQL"])


@router.post("/translate", response_model=Dict[str, Any])
async def translate_query(
    request: TranslateRequest,
    service: NL2SPARQLService = Depends(get_nl2sparql_service),
) -> Dict[str, Any]:
    """
    Translate natural language query to SPARQL.

    Converts a natural language query into a structured SPARQL query
    using Azure OpenAI and predefined templates.
    """
    try:
        logger.info(f"Translating query: {request.query[:100]}...")

        response = await service.translate_only(
            natural_language_query=request.query,
            context=request.context,
            max_results=request.max_results,
        )

        return {
            "success": True,
            "natural_language_query": request.query,
            "sparql_query": response.sparql_query.to_sparql(),
            "confidence_score": response.confidence_score,
            "explanation": response.explanation,
            "detected_entities": [e.value for e in response.detected_entities],
            "detected_relations": [r.value for r in response.detected_relations],
            "suggested_alternatives": response.suggested_alternatives,
            "validation_errors": response.validation_errors,
            "metadata": {
                "context": request.context,
                "max_results": request.max_results,
                "query_length": len(request.query),
            },
        }

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@router.post("/translate-and-execute", response_model=Dict[str, Any])
async def translate_and_execute_query(
    request: TranslateAndExecuteRequest,
    service: NL2SPARQLService = Depends(get_nl2sparql_service),
) -> Dict[str, Any]:
    """
    Translate natural language query to SPARQL and execute it.

    Converts a natural language query to SPARQL and optionally executes
    it against the graph database, returning both the query and results.
    """
    try:
        logger.info(f"Translating and executing query: {request.query[:100]}...")

        result = await service.translate_and_execute(
            natural_language_query=request.query,
            context=request.context,
            execute_query=request.execute_query,
            max_results=request.max_results,
        )

        result["success"] = True
        return result

    except Exception as e:
        logger.error(f"Translation and execution failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Translation and execution failed: {str(e)}"
        )


@router.post("/batch-translate", response_model=Dict[str, Any])
async def batch_translate_queries(
    request: BatchTranslateRequest,
    background_tasks: BackgroundTasks,
    service: NL2SPARQLService = Depends(get_nl2sparql_service),
) -> Dict[str, Any]:
    """
    Batch translate multiple natural language queries.

    Processes multiple queries in parallel for efficient bulk translation.
    Optionally executes all generated SPARQL queries.
    """
    try:
        logger.info(f"Batch translating {len(request.queries)} queries...")

        # Validate batch size
        if len(request.queries) > request.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size {len(request.queries)} exceeds maximum {request.max_batch_size}",
            )

        results = await service.batch_translate(
            queries=request.queries, execute_queries=request.execute_queries
        )

        # Calculate summary statistics
        successful_count = sum(1 for r in results if r.get("success", False))
        failed_count = len(results) - successful_count

        if request.execute_queries:
            executed_count = sum(1 for r in results if r.get("executed", False))
        else:
            executed_count = 0

        return {
            "success": True,
            "total_queries": len(request.queries),
            "successful_translations": successful_count,
            "failed_translations": failed_count,
            "executed_queries": executed_count,
            "results": results,
            "summary": {
                "success_rate": (
                    successful_count / len(request.queries) if request.queries else 0
                ),
                "average_confidence": sum(
                    r.get("confidence_score", 0) for r in results if r.get("success")
                )
                / max(successful_count, 1),
            },
        }

    except Exception as e:
        logger.error(f"Batch translation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch translation failed: {str(e)}"
        )


@router.post("/validate", response_model=Dict[str, Any])
async def validate_sparql_query(
    request: ValidateQueryRequest,
    service: NL2SPARQLService = Depends(get_nl2sparql_service),
) -> Dict[str, Any]:
    """
    Validate a SPARQL query.

    Checks the syntax and structure of a SPARQL query for correctness.
    """
    try:
        logger.info("Validating SPARQL query...")

        result = await service.validate_sparql_query(request.sparql_query)
        result["success"] = True

        return result

    except Exception as e:
        logger.error(f"Query validation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Query validation failed: {str(e)}"
        )


@router.get("/templates", response_model=Dict[str, Any])
async def get_translation_templates(
    service: NL2SPARQLService = Depends(get_nl2sparql_service),
) -> Dict[str, Any]:
    """
    Get available translation templates.

    Returns information about available query templates including
    examples and pattern keywords.
    """
    try:
        logger.info("Retrieving translation templates...")

        templates = await service.get_translation_templates()
        templates["success"] = True

        return templates

    except Exception as e:
        logger.error(f"Failed to get templates: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get templates: {str(e)}"
        )


@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    service: NL2SPARQLService = Depends(get_nl2sparql_service),
) -> Dict[str, Any]:
    """
    Health check for NL2SPARQL service.

    Returns the health status of all service components including
    translator, query executor, and cache.
    """
    try:
        health_status = await service.health_check()

        # Set appropriate HTTP status code based on health
        if health_status["status"] == "unhealthy":
            raise HTTPException(status_code=503, detail=health_status)
        elif health_status["status"] == "degraded":
            # Return 200 but indicate degraded service
            health_status["warning"] = "Service is degraded but functional"

        return health_status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# Cache management endpoints
@router.get("/cache/stats", response_model=Dict[str, Any])
async def get_cache_stats(
    service: NL2SPARQLService = Depends(get_nl2sparql_service),
) -> Dict[str, Any]:
    """
    Get translation cache statistics.

    Returns cache performance metrics and usage statistics.
    """
    try:
        stats = await service.get_cache_stats()
        stats["success"] = True

        return stats

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get cache stats: {str(e)}"
        )


@router.post("/cache/clear", response_model=Dict[str, Any])
async def clear_cache(
    request: ClearCacheRequest,
    service: NL2SPARQLService = Depends(get_nl2sparql_service),
) -> Dict[str, Any]:
    """
    Clear translation cache.

    Removes cached translation results to force fresh translations.
    Optionally accepts a pattern to clear specific cache entries.
    """
    try:
        logger.info(f"Clearing cache with pattern: {request.pattern}")

        result = await service.clear_cache(request.pattern)
        result["success"] = True

        return result

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


# Utility endpoints
@router.get("/examples", response_model=Dict[str, Any])
async def get_query_examples() -> Dict[str, Any]:
    """
    Get example natural language queries.

    Returns a collection of example queries that demonstrate
    the capabilities of the NL2SPARQL service.
    """
    examples = {
        "function_calls": [
            "Which functions call the authenticate method?",
            "Show me all functions that invoke error handling routines",
            "Find functions that call database connection methods",
            "What methods are called by the UserController class?",
        ],
        "inheritance_hierarchy": [
            "What classes inherit from BaseModel?",
            "Show me the inheritance tree for the Vehicle class",
            "Which interfaces does the Logger class implement?",
            "Find all subclasses of the Exception class",
        ],
        "dependency_analysis": [
            "What modules does the authentication package import?",
            "Show me all dependencies of the core library",
            "Which packages use the numpy library?",
            "Find modules that depend on the database connection",
        ],
        "code_structure": [
            "What functions are defined in the utils module?",
            "Show me all classes in the models package",
            "Find all interfaces in the project",
            "What variables are used in the configuration module?",
        ],
        "complex_queries": [
            "Show me functions that call authentication methods and handle errors",
            "Find classes that inherit from BaseController and implement caching",
            "What modules import both logging and database libraries?",
            "Show the call chain from login to database operations",
        ],
    }

    return {
        "success": True,
        "examples": examples,
        "total_categories": len(examples),
        "total_examples": sum(
            len(category_examples) for category_examples in examples.values()
        ),
        "description": "Example natural language queries for different code analysis patterns",
    }


@router.get("/stats", response_model=Dict[str, Any])
async def get_service_stats(
    service: NL2SPARQLService = Depends(get_nl2sparql_service),
) -> Dict[str, Any]:
    """
    Get service usage statistics.

    Returns metrics about service usage and performance.
    """
    try:
        # Get cache stats
        cache_stats = await service.get_cache_stats()

        # Get template info
        templates = await service.get_translation_templates()

        return {
            "success": True,
            "service_info": {
                "name": "NL2SPARQL Translation Service",
                "version": "1.0.0",
                "description": "Natural Language to SPARQL Query Translation",
            },
            "available_templates": templates.get("total_templates", 0),
            "cache_stats": cache_stats,
            "supported_entities": [
                e.value for e in service.translator._detect_entities("test")
            ],
            "supported_relations": [
                r.value for r in service.translator._detect_relations("test")
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get service stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get service stats: {str(e)}"
        )


# Include router in main application
def get_router() -> APIRouter:
    """Get the NL2SPARQL router for inclusion in main app."""
    return router
