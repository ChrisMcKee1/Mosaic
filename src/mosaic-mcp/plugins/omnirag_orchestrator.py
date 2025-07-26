"""
Multi-Source Query Orchestration Engine for OmniRAG Pattern

This module implements the core orchestration engine that coordinates queries across
graph, vector, and database sources based on intent detection from OMR-P3-001.

Key Features:
- Parallel execution of multiple retrieval strategies
- Dynamic strategy selection based on query complexity
- Error handling and graceful degradation
- Performance monitoring and strategy effectiveness tracking
- Integration with existing vector search and graph search capabilities
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from .query_intent_classifier import QueryIntentClassifier
from .retrieval import RetrievalPlugin
from .graph_plugin import GraphPlugin
from ..models.intent_models import (
    ClassificationResult,
    QueryIntentType,
    ClassificationRequest,
)

logger = logging.getLogger(__name__)


class RetrievalStrategy(ABC):
    """
    Base class for retrieval strategies in the OmniRAG orchestration.

    Each strategy handles a specific type of data source:
    - GraphRetrievalStrategy: SPARQL-based graph queries
    - VectorRetrievalStrategy: Semantic similarity searches
    - DatabaseRetrievalStrategy: Direct database entity lookups
    """

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.execution_time_ms = 0
        self.result_count = 0
        self.confidence = 0.0
        self.last_error = None

    @abstractmethod
    async def execute(
        self, query: str, intent: ClassificationResult, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the retrieval strategy.

        Args:
            query: User's natural language query
            intent: Classified query intent with confidence and strategy recommendations
            context: Additional context including limits, session info, etc.

        Returns:
            Dict containing:
            - status: "success" | "error" | "timeout"
            - strategy: Strategy name
            - results: List of retrieved items
            - metadata: Execution metadata
            - execution_time_ms: Time taken
            - error: Error message if status is "error"
        """
        pass

    async def initialize(self, **kwargs: Any):
        """Initialize strategy with dependencies. Override in subclasses."""
        pass


class GraphRetrievalStrategy(RetrievalStrategy):
    """
    Graph-based retrieval using SPARQL queries for relationship exploration.

    Handles queries that require understanding relationships between entities,
    inheritance hierarchies, dependency graphs, and structural code analysis.
    """

    def __init__(self):
        super().__init__("graph", weight=1.0)
        self.graph_plugin = None

    async def initialize(self, graph_plugin: GraphPlugin):
        """Initialize with GraphPlugin dependency."""
        self.graph_plugin = graph_plugin
        logger.debug("GraphRetrievalStrategy initialized")

    async def execute(
        self, query: str, intent: ClassificationResult, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        start_time = datetime.now()

        try:
            if not self.graph_plugin:
                raise RuntimeError(
                    "GraphRetrievalStrategy not initialized with GraphPlugin"
                )

            # Use graph plugin for relationship queries
            result = await self.graph_plugin.natural_language_query(
                query=query,
                include_visualization=False,
                max_results=context.get("limit", 20),
            )

            self.execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            if result.get("status") == "success":
                self.result_count = len(result.get("results", []))
                self.confidence = result.get("metadata", {}).get("confidence", 0.8)
                self.last_error = None

                return {
                    "status": "success",
                    "strategy": "graph",
                    "results": result["results"],
                    "metadata": result.get("metadata", {}),
                    "execution_time_ms": self.execution_time_ms,
                }
            else:
                self.last_error = result.get("error", "Graph query failed")
                return {
                    "status": "error",
                    "strategy": "graph",
                    "error": self.last_error,
                    "execution_time_ms": self.execution_time_ms,
                }

        except Exception as e:
            self.execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            self.last_error = str(e)
            logger.error(f"Graph retrieval failed: {e}")
            return {
                "status": "error",
                "strategy": "graph",
                "error": str(e),
                "execution_time_ms": self.execution_time_ms,
            }


class VectorRetrievalStrategy(RetrievalStrategy):
    """
    Vector-based retrieval using semantic similarity search.

    Handles queries that require finding semantically similar content,
    code examples, documentation, and conceptual relationships.
    """

    def __init__(self):
        super().__init__("vector", weight=1.0)
        self.retrieval_plugin = None

    async def initialize(self, retrieval_plugin: RetrievalPlugin):
        """Initialize with RetrievalPlugin dependency."""
        self.retrieval_plugin = retrieval_plugin
        logger.debug("VectorRetrievalStrategy initialized")

    async def execute(
        self, query: str, intent: ClassificationResult, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        start_time = datetime.now()

        try:
            if not self.retrieval_plugin:
                raise RuntimeError(
                    "VectorRetrievalStrategy not initialized with RetrievalPlugin"
                )

            # Use existing retrieval plugin for vector search
            documents = await self.retrieval_plugin.hybrid_search(query=query)

            self.execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )

            if documents:
                # Limit results based on context
                limited_documents = documents[: context.get("limit", 20)]
                self.result_count = len(limited_documents)
                self.confidence = 0.9  # High confidence for vector search
                self.last_error = None

                return {
                    "status": "success",
                    "strategy": "vector",
                    "results": [
                        doc.model_dump() if hasattr(doc, "model_dump") else doc.__dict__
                        for doc in limited_documents
                    ],
                    "metadata": {
                        "search_type": "hybrid_vector",
                        "confidence": self.confidence,
                    },
                    "execution_time_ms": self.execution_time_ms,
                }
            else:
                self.last_error = "No results found"
                return {
                    "status": "error",
                    "strategy": "vector",
                    "error": self.last_error,
                    "execution_time_ms": self.execution_time_ms,
                }

        except Exception as e:
            self.execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            self.last_error = str(e)
            logger.error(f"Vector retrieval failed: {e}")
            return {
                "status": "error",
                "strategy": "vector",
                "error": str(e),
                "execution_time_ms": self.execution_time_ms,
            }


class DatabaseRetrievalStrategy(RetrievalStrategy):
    """
    Direct database retrieval for specific entity lookup.

    Handles queries that require direct access to stored entities,
    metadata lookups, and structured data queries.
    """

    def __init__(self):
        super().__init__("database", weight=1.0)
        self.cosmos_client = None

    async def initialize(self, cosmos_client: Any):
        """Initialize with Cosmos DB client dependency."""
        self.cosmos_client = cosmos_client
        logger.debug("DatabaseRetrievalStrategy initialized")

    async def execute(
        self, query: str, intent: ClassificationResult, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        start_time = datetime.now()

        try:
            if not self.cosmos_client:
                raise RuntimeError(
                    "DatabaseRetrievalStrategy not initialized with Cosmos client"
                )

            # Extract entity names from query for direct lookup
            entity_names = self._extract_entity_names(query)

            if not entity_names:
                return {
                    "status": "error",
                    "strategy": "database",
                    "error": "No entities found in query for database lookup",
                }

            results = []
            for entity_name in entity_names:
                # Query Cosmos DB directly for entity information
                cosmos_query = f"""
                SELECT * FROM c
                WHERE CONTAINS(LOWER(c.name), LOWER('{entity_name}'))
                   OR CONTAINS(LOWER(c.content), LOWER('{entity_name}'))
                ORDER BY c.timestamp DESC
                """

                async for item in self.cosmos_client.query_items(
                    query=cosmos_query, enable_cross_partition_query=True
                ):
                    results.append(item)

                    # Limit results per entity
                    if len(results) >= context.get("limit", 10):
                        break

            self.execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            self.result_count = len(results)
            self.confidence = 0.85 if results else 0.0
            self.last_error = None

            return {
                "status": "success",
                "strategy": "database",
                "results": results,
                "metadata": {
                    "entities_searched": entity_names,
                    "confidence": self.confidence,
                },
                "execution_time_ms": self.execution_time_ms,
            }

        except Exception as e:
            self.execution_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            self.last_error = str(e)
            logger.error(f"Database retrieval failed: {e}")
            return {
                "status": "error",
                "strategy": "database",
                "error": str(e),
                "execution_time_ms": self.execution_time_ms,
            }

    def _extract_entity_names(self, query: str) -> List[str]:
        """
        Extract potential entity names from query.
        Simple heuristic-based extraction - can be enhanced with NER.
        """
        import re

        entities = []

        # Look for quoted strings
        quoted_entities = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_entities)

        # Look for capitalized words (potential class/function names)
        words = query.split()
        for word in words:
            # Skip common words and look for potential identifiers
            if (
                word[0].isupper()
                and len(word) > 2
                and word
                not in [
                    "What",
                    "Where",
                    "When",
                    "How",
                    "Why",
                    "Which",
                    "Who",
                    "The",
                    "This",
                    "That",
                ]
            ):
                entities.append(word)

        # Look for snake_case and camelCase identifiers
        identifier_pattern = (
            r"\b[a-z_][a-z0-9_]*[a-z0-9]\b|\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b"
        )
        identifiers = re.findall(identifier_pattern, query)
        entities.extend(identifiers)

        return list(set(entities))  # Remove duplicates


class OmniRAGOrchestrator:
    """
    Core orchestration engine that coordinates queries across multiple sources.

    Features:
    - Intent-based strategy selection
    - Parallel and sequential execution modes
    - Error handling and graceful degradation
    - Performance monitoring
    - Dynamic strategy weighting
    """

    def __init__(self):
        self.intent_classifier = None

        # Retrieval strategies
        self.strategies = {
            "graph": GraphRetrievalStrategy(),
            "vector": VectorRetrievalStrategy(),
            "database": DatabaseRetrievalStrategy(),
        }

        # Configuration
        self.parallel_enabled = (
            os.getenv("MOSAIC_PARALLEL_RETRIEVAL_ENABLED", "true").lower() == "true"
        )
        self.max_sources = int(os.getenv("MOSAIC_MAX_CONTEXT_SOURCES", "3"))
        self.timeout_seconds = int(
            os.getenv("MOSAIC_ORCHESTRATOR_TIMEOUT_SECONDS", "30")
        )

        logger.info(
            f"OmniRAG Orchestrator configured: parallel={self.parallel_enabled}, max_sources={self.max_sources}, timeout={self.timeout_seconds}s"
        )

    async def initialize(
        self,
        intent_classifier: QueryIntentClassifier,
        retrieval_plugin: RetrievalPlugin,
        graph_plugin: GraphPlugin,
        cosmos_client,
    ):
        """
        Initialize orchestrator with all dependencies.

        Args:
            intent_classifier: QueryIntentClassifier instance for intent detection
            retrieval_plugin: RetrievalPlugin for vector search
            graph_plugin: GraphPlugin for SPARQL queries
            cosmos_client: Cosmos DB client for direct database access
        """
        logger.info("Initializing OmniRAG Orchestrator...")

        try:
            # Store intent classifier
            self.intent_classifier = intent_classifier

            # Initialize strategies with their dependencies
            await self.strategies["graph"].initialize(graph_plugin)
            await self.strategies["vector"].initialize(retrieval_plugin)
            await self.strategies["database"].initialize(cosmos_client)

            logger.info("OmniRAG Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OmniRAG orchestrator: {e}")
            raise

    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process query through complete OmniRAG pipeline.

        Args:
            query: Natural language query from user
            context: Additional context for processing (limits, preferences, etc.)
            session_id: Optional session identifier for tracking

        Returns:
            Complete OmniRAG response with results, metadata, and performance stats
        """
        start_time = datetime.now()
        context = context or {}

        try:
            logger.info(f"Processing OmniRAG query: {query[:100]}...")

            # Step 1: Classify query intent
            classification_request = ClassificationRequest(query=query)
            intent = await self.intent_classifier.classify_intent(
                classification_request
            )
            logger.debug(
                f"Query classified as: {intent.intent} (confidence: {intent.confidence:.3f})"
            )

            # Step 2: Determine retrieval strategies to use
            strategies_to_use = self._select_strategies(intent, context)

            # Step 3: Execute retrieval strategies
            if self.parallel_enabled and len(strategies_to_use) > 1:
                strategy_results = await self._execute_parallel_retrieval(
                    query, intent, strategies_to_use, context
                )
            else:
                strategy_results = await self._execute_sequential_retrieval(
                    query, intent, strategies_to_use, context
                )

            # Step 4: Build final response
            total_time = int((datetime.now() - start_time).total_seconds() * 1000)

            # Collect successful results
            all_results = []
            successful_strategies = []
            failed_strategies = []

            for result in strategy_results:
                if result.get("status") == "success":
                    successful_strategies.append(result["strategy"])
                    all_results.extend(result.get("results", []))
                else:
                    failed_strategies.append(
                        {
                            "strategy": result.get("strategy", "unknown"),
                            "error": result.get("error", "Unknown error"),
                        }
                    )

            response = {
                "status": "success" if successful_strategies else "error",
                "query": query,
                "intent": {
                    "intent_type": intent.intent.value,
                    "confidence": intent.confidence,
                    "strategy": intent.strategy.value,
                },
                "strategies_used": successful_strategies,
                "strategies_failed": failed_strategies,
                "results": all_results,
                "metadata": {
                    "total_execution_time_ms": total_time,
                    "strategy_execution_times": {
                        s.name: s.execution_time_ms for s in strategies_to_use
                    },
                    "total_results": len(all_results),
                    "session_id": session_id,
                    "parallel_execution": self.parallel_enabled
                    and len(strategies_to_use) > 1,
                },
            }

            logger.info(
                f"OmniRAG query processed successfully in {total_time}ms using {len(successful_strategies)} strategies"
            )

            return response

        except Exception as e:
            total_time = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"OmniRAG query processing failed: {e}")

            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "metadata": {
                    "total_execution_time_ms": total_time,
                    "session_id": session_id,
                },
            }

    def _select_strategies(
        self, intent: ClassificationResult, context: Dict[str, Any]
    ) -> List[RetrievalStrategy]:
        """
        Select which retrieval strategies to use based on intent and context.

        Args:
            intent: Classified query intent
            context: Additional context and preferences

        Returns:
            List of retrieval strategies to execute
        """
        strategies = []

        # Primary strategy based on intent
        if intent.intent == QueryIntentType.GRAPH_RAG:
            strategies.append(self.strategies["graph"])
        elif intent.intent == QueryIntentType.VECTOR_RAG:
            strategies.append(self.strategies["vector"])
        elif intent.intent == QueryIntentType.DATABASE_RAG:
            strategies.append(self.strategies["database"])
        elif intent.intent == QueryIntentType.HYBRID:
            # Use multiple strategies for hybrid approach
            strategies.extend(
                [
                    self.strategies["graph"],
                    self.strategies["vector"],
                    self.strategies["database"],
                ]
            )

        # Add secondary strategies if confidence is low or explicitly requested
        if intent.confidence < 0.8 or context.get("use_multiple_sources", False):
            for strategy_name, strategy_obj in self.strategies.items():
                if (
                    strategy_obj not in strategies
                    and len(strategies) < self.max_sources
                ):
                    strategies.append(strategy_obj)

        # Ensure at least one strategy
        if not strategies:
            strategies.append(self.strategies["vector"])  # Default fallback

        return strategies

    async def _execute_parallel_retrieval(
        self,
        query: str,
        intent: ClassificationResult,
        strategies: List[RetrievalStrategy],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple retrieval strategies in parallel.
        """
        logger.debug(f"Executing {len(strategies)} strategies in parallel")

        # Create tasks for parallel execution
        tasks = []
        for strategy in strategies:
            task = asyncio.create_task(
                strategy.execute(query, intent, context),
                name=f"strategy_{strategy.name}",
            )
            tasks.append(task)

        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout_seconds,
            )

            # Process results and handle exceptions
            strategy_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Strategy {strategies[i].name} failed: {result}")
                    strategy_results.append(
                        {
                            "status": "error",
                            "strategy": strategies[i].name,
                            "error": str(result),
                        }
                    )
                else:
                    strategy_results.append(result)

            return strategy_results

        except asyncio.TimeoutError:
            logger.error(f"Parallel retrieval timeout after {self.timeout_seconds}s")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

            return [
                {
                    "status": "timeout",
                    "error": f"Retrieval timeout after {self.timeout_seconds}s",
                }
            ]

    async def _execute_sequential_retrieval(
        self,
        query: str,
        intent: ClassificationResult,
        strategies: List[RetrievalStrategy],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Execute retrieval strategies sequentially.
        """
        logger.debug(f"Executing {len(strategies)} strategies sequentially")

        results = []
        for strategy in strategies:
            try:
                result = await strategy.execute(query, intent, context)
                results.append(result)

                # Early termination if we have enough good results
                if (
                    result.get("status") == "success"
                    and len(result.get("results", [])) >= context.get("limit", 10)
                    and strategy.confidence > 0.8
                ):
                    logger.debug(
                        f"Early termination after {strategy.name} with high confidence"
                    )
                    break

            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed: {e}")
                results.append(
                    {"status": "error", "strategy": strategy.name, "error": str(e)}
                )

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status and performance metrics."""
        return {
            "parallel_enabled": self.parallel_enabled,
            "max_sources": self.max_sources,
            "timeout_seconds": self.timeout_seconds,
            "strategies_performance": {
                name: {
                    "last_execution_time_ms": strategy.execution_time_ms,
                    "last_result_count": strategy.result_count,
                    "last_confidence": strategy.confidence,
                    "last_error": strategy.last_error,
                }
                for name, strategy in self.strategies.items()
            },
        }


# Global instance
_omnirag_orchestrator: Optional[OmniRAGOrchestrator] = None


async def get_omnirag_orchestrator() -> OmniRAGOrchestrator:
    """
    Get or create global OmniRAG orchestrator instance.

    Returns:
        Singleton OmniRAGOrchestrator instance
    """
    global _omnirag_orchestrator
    if _omnirag_orchestrator is None:
        _omnirag_orchestrator = OmniRAGOrchestrator()
    return _omnirag_orchestrator
