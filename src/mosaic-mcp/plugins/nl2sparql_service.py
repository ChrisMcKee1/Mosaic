"""
Natural Language to SPARQL Translation Service.
Integrates NL2SPARQLTranslator with SPARQL query execution and HTTP API.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import json

from fastapi import HTTPException
import redis.asyncio as redis

from ..config.settings import get_settings
from ..plugins.nl2sparql_translator import (
    NL2SPARQLTranslator,
    NL2SPARQLRequest,
    NL2SPARQLResponse,
)
from ..plugins.sparql_query_executor import SPARQLQueryExecutor


logger = logging.getLogger(__name__)


class NL2SPARQLService:
    """
    Service for natural language to SPARQL translation and execution.

    Provides high-level API for translating natural language queries to SPARQL
    and optionally executing them against the graph database.
    """

    def __init__(self):
        """Initialize the service with translator, executor, and cache."""
        self.settings = get_settings()
        self.translator = NL2SPARQLTranslator()
        self.query_executor = SPARQLQueryExecutor()
        self._redis_client: Optional[redis.Redis] = None
        self._cache_enabled = True
        self._cache_ttl = 3600  # 1 hour default TTL

    async def initialize(self):
        """Initialize async components (Redis cache)."""
        try:
            if self.settings.redis_connection_string:
                self._redis_client = redis.from_url(
                    self.settings.redis_connection_string, decode_responses=True
                )
                await self._redis_client.ping()
                logger.info("Redis cache initialized successfully")
            else:
                logger.warning("Redis connection string not provided, caching disabled")
                self._cache_enabled = False

        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            self._cache_enabled = False

    async def translate_and_execute(
        self,
        natural_language_query: str,
        context: Optional[str] = None,
        execute_query: bool = True,
        max_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Translate natural language to SPARQL and optionally execute.

        Args:
            natural_language_query: Natural language query string
            context: Additional context for translation
            execute_query: Whether to execute the generated SPARQL query
            max_results: Maximum number of results to return

        Returns:
            Dictionary containing translation results and optionally query results
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(
                natural_language_query, context, max_results
            )
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                logger.info("Returning cached translation result")
                return cached_result

            # Create translation request
            request = NL2SPARQLRequest(
                natural_language_query=natural_language_query,
                context=context,
                max_results=max_results or 100,
            )

            # Translate to SPARQL
            translation_response = await self.translator.translate_query(request)

            # Prepare response
            result = {
                "natural_language_query": natural_language_query,
                "sparql_query": translation_response.sparql_query.to_sparql(),
                "confidence_score": translation_response.confidence_score,
                "explanation": translation_response.explanation,
                "detected_entities": [
                    e.value for e in translation_response.detected_entities
                ],
                "detected_relations": [
                    r.value for r in translation_response.detected_relations
                ],
                "suggested_alternatives": translation_response.suggested_alternatives,
                "validation_errors": translation_response.validation_errors,
                "timestamp": datetime.utcnow().isoformat(),
                "executed": False,
                "results": None,
                "execution_time_ms": None,
                "result_count": None,
            }

            # Execute query if requested and valid
            if (
                execute_query
                and translation_response.confidence_score >= 0.7
                and not translation_response.validation_errors
            ):
                try:
                    start_time = datetime.utcnow()

                    # Execute the SPARQL query
                    query_results = await self.query_executor.execute_query(
                        translation_response.sparql_query.to_sparql()
                    )

                    end_time = datetime.utcnow()
                    execution_time = (end_time - start_time).total_seconds() * 1000

                    result.update(
                        {
                            "executed": True,
                            "results": query_results,
                            "execution_time_ms": round(execution_time, 2),
                            "result_count": (
                                len(query_results)
                                if isinstance(query_results, list)
                                else 1
                            ),
                        }
                    )

                    logger.info(
                        f"Query executed successfully in {execution_time:.2f}ms"
                    )

                except Exception as e:
                    logger.error(f"Query execution failed: {e}")
                    result["execution_error"] = str(e)

            elif execute_query:
                result["execution_skipped_reason"] = (
                    "Low confidence score or validation errors"
                    if translation_response.confidence_score < 0.7
                    else "Validation errors present"
                )

            # Cache the result
            await self._store_in_cache(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Translation and execution failed: {e}")
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

    async def translate_only(
        self,
        natural_language_query: str,
        context: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> NL2SPARQLResponse:
        """
        Translate natural language to SPARQL without execution.

        Args:
            natural_language_query: Natural language query string
            context: Additional context for translation
            max_results: Maximum number of results to return

        Returns:
            Translation response with SPARQL query and metadata
        """
        try:
            request = NL2SPARQLRequest(
                natural_language_query=natural_language_query,
                context=context,
                max_results=max_results or 100,
            )

            return await self.translator.translate_query(request)

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

    async def batch_translate(
        self, queries: List[Dict[str, Any]], execute_queries: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Batch translate multiple natural language queries.

        Args:
            queries: List of query dictionaries with 'query' and optional 'context'
            execute_queries: Whether to execute the generated SPARQL queries

        Returns:
            List of translation and execution results
        """
        try:
            tasks = []

            for query_data in queries:
                natural_language_query = query_data.get("query")
                context = query_data.get("context")
                max_results = query_data.get("max_results")

                if not natural_language_query:
                    continue

                task = self.translate_and_execute(
                    natural_language_query=natural_language_query,
                    context=context,
                    execute_query=execute_queries,
                    max_results=max_results,
                )
                tasks.append(task)

            # Execute all translations concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(
                        {
                            "query": queries[i].get("query"),
                            "error": str(result),
                            "success": False,
                        }
                    )
                else:
                    result["success"] = True
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Batch translation failed: {str(e)}"
            )

    async def validate_sparql_query(self, sparql_query: str) -> Dict[str, Any]:
        """
        Validate a SPARQL query string.

        Args:
            sparql_query: SPARQL query string to validate

        Returns:
            Validation result with errors and suggestions
        """
        try:
            # Use query executor's validation
            is_valid, errors = await self.query_executor.validate_query(sparql_query)

            return {
                "valid": is_valid,
                "errors": errors,
                "query": sparql_query,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Query validation failed: {e}")
            return {
                "valid": False,
                "errors": [str(e)],
                "query": sparql_query,
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def get_translation_templates(self) -> Dict[str, Any]:
        """
        Get available translation templates with examples.

        Returns:
            Dictionary of templates with descriptions and examples
        """
        templates_info = {}

        for name, template in self.translator.templates.items():
            templates_info[name] = {
                "name": template.name,
                "description": template.description,
                "pattern_keywords": template.pattern_keywords,
                "confidence_weight": template.confidence_weight,
                "example_sparql": template.base_query.to_sparql(),
                "example_queries": self._get_example_queries_for_template(name),
            }

        return {
            "templates": templates_info,
            "total_templates": len(templates_info),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _get_example_queries_for_template(self, template_name: str) -> List[str]:
        """Get example natural language queries for a template."""
        examples = {
            "function_calls": [
                "Which functions call the login method?",
                "Show me all function calls to error handlers",
                "Find functions that invoke database operations",
            ],
            "inheritance_hierarchy": [
                "What classes inherit from BaseController?",
                "Show me the inheritance hierarchy for User class",
                "Which classes extend the Animal interface?",
            ],
            "dependency_analysis": [
                "What modules does the auth package import?",
                "Show me all dependencies of the core module",
                "Which packages use the logging library?",
            ],
        }

        return examples.get(template_name, [])

    def _generate_cache_key(
        self,
        natural_language_query: str,
        context: Optional[str],
        max_results: Optional[int],
    ) -> str:
        """Generate cache key for query translation."""
        cache_data = {
            "query": natural_language_query,
            "context": context,
            "max_results": max_results,
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"nl2sparql:{hashlib.md5(cache_string.encode()).hexdigest()}"

    async def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get translation result from cache."""
        if not self._cache_enabled or not self._redis_client:
            return None

        try:
            cached_data = await self._redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

        return None

    async def _store_in_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Store translation result in cache."""
        if not self._cache_enabled or not self._redis_client:
            return

        try:
            cached_data = json.dumps(
                result, default=str
            )  # Handle datetime serialization
            await self._redis_client.setex(cache_key, self._cache_ttl, cached_data)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._cache_enabled or not self._redis_client:
            return {"cache_enabled": False}

        try:
            info = await self._redis_client.info()
            nl2sparql_keys = await self._redis_client.keys("nl2sparql:*")

            return {
                "cache_enabled": True,
                "total_keys": len(nl2sparql_keys),
                "memory_usage": info.get("used_memory_human", "unknown"),
                "hit_rate": info.get("keyspace_hits", 0)
                / max(
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0)), 1
                ),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"cache_enabled": True, "error": str(e)}

    async def clear_cache(self, pattern: Optional[str] = None) -> Dict[str, Any]:
        """Clear translation cache."""
        if not self._cache_enabled or not self._redis_client:
            return {"cache_enabled": False}

        try:
            pattern = pattern or "nl2sparql:*"
            keys = await self._redis_client.keys(pattern)

            if keys:
                deleted_count = await self._redis_client.delete(*keys)
            else:
                deleted_count = 0

            return {
                "cache_enabled": True,
                "deleted_keys": deleted_count,
                "pattern": pattern,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return {"cache_enabled": True, "error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the NL2SPARQL service."""
        health_status = {
            "service": "NL2SPARQL",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
        }

        # Check translator
        try:
            # Simple translation test
            NL2SPARQLRequest(natural_language_query="test query", max_results=1)
            # Don't actually translate, just check if translator is initialized
            if self.translator.client:
                health_status["components"]["translator"] = "healthy"
            else:
                health_status["components"]["translator"] = "unhealthy"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["components"]["translator"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"

        # Check query executor
        try:
            if self.query_executor:
                health_status["components"]["query_executor"] = "healthy"
            else:
                health_status["components"]["query_executor"] = "unhealthy"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["components"]["query_executor"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"

        # Check cache
        if self._cache_enabled and self._redis_client:
            try:
                await self._redis_client.ping()
                health_status["components"]["cache"] = "healthy"
            except Exception as e:
                health_status["components"]["cache"] = f"unhealthy: {str(e)}"
                # Cache failure is not critical
        else:
            health_status["components"]["cache"] = "disabled"

        return health_status
