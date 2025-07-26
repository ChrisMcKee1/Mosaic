"""
Test suite for OmniRAG Orchestrator

Tests the multi-source query orchestration engine with intent-based strategy selection.
Validates parallel and sequential execution, error handling, and strategy coordination.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.mosaic_mcp.plugins.omnirag_orchestrator import (
    OmniRAGOrchestrator,
    GraphRetrievalStrategy,
    VectorRetrievalStrategy,
    DatabaseRetrievalStrategy,
    get_omnirag_orchestrator,
)
from src.mosaic_mcp.models.intent_models import (
    ClassificationResult,
    QueryIntentType,
    RetrievalStrategy as IntentStrategy,
    ConfidenceLevel,
)


class TestOmniRAGOrchestrator:
    """Test OmniRAG orchestration engine functionality."""

    @pytest.fixture
    def mock_intent_classifier(self):
        """Mock intent classifier."""
        classifier = AsyncMock()
        return classifier

    @pytest.fixture
    def mock_retrieval_plugin(self):
        """Mock retrieval plugin."""
        plugin = AsyncMock()
        plugin.hybrid_search.return_value = [
            MagicMock(
                model_dump=lambda: {"id": "1", "content": "test content", "score": 0.9}
            )
        ]
        return plugin

    @pytest.fixture
    def mock_graph_plugin(self):
        """Mock graph plugin."""
        plugin = AsyncMock()
        plugin.natural_language_query.return_value = {
            "success": True,
            "results": [{"id": "graph_1", "content": "graph content"}],
            "metadata": {"confidence": 0.8},
        }
        return plugin

    @pytest.fixture
    def mock_cosmos_client(self):
        """Mock Cosmos DB client."""
        client = AsyncMock()
        client.query_items.return_value = [
            {"id": "db_1", "content": "database content"}
        ]
        return client

    @pytest.fixture
    async def orchestrator(
        self,
        mock_intent_classifier,
        mock_retrieval_plugin,
        mock_graph_plugin,
        mock_cosmos_client,
    ):
        """Initialized orchestrator fixture."""
        orchestrator = OmniRAGOrchestrator()
        await orchestrator.initialize(
            intent_classifier=mock_intent_classifier,
            retrieval_plugin=mock_retrieval_plugin,
            graph_plugin=mock_graph_plugin,
            cosmos_client=mock_cosmos_client,
        )
        return orchestrator

    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator.intent_classifier is not None
        assert len(orchestrator.strategies) == 3
        assert "graph" in orchestrator.strategies
        assert "vector" in orchestrator.strategies
        assert "database" in orchestrator.strategies

    async def test_graph_rag_strategy_selection(
        self, orchestrator, mock_intent_classifier
    ):
        """Test graph RAG strategy is selected for relationship queries."""
        # Mock intent classification for graph query
        mock_intent_classifier.classify_intent.return_value = ClassificationResult(
            intent=QueryIntentType.GRAPH_RAG,
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            strategy=IntentStrategy.GRAPH_TRAVERSAL,
            reasoning="Relationship query detected",
        )

        result = await orchestrator.process_query(
            "What are the dependencies of Flask?", context={"limit": 10}
        )

        assert result["status"] == "success"
        assert "graph" in result["strategies_used"]
        assert result["intent"]["intent_type"] == "GRAPH_RAG"
        assert result["metadata"]["total_execution_time_ms"] > 0

    async def test_vector_rag_strategy_selection(
        self, orchestrator, mock_intent_classifier
    ):
        """Test vector RAG strategy is selected for semantic queries."""
        mock_intent_classifier.classify_intent.return_value = ClassificationResult(
            intent=QueryIntentType.VECTOR_RAG,
            confidence=0.85,
            confidence_level=ConfidenceLevel.HIGH,
            strategy=IntentStrategy.SEMANTIC_SEARCH,
            reasoning="Semantic similarity query detected",
        )

        result = await orchestrator.process_query(
            "Find code similar to authentication patterns", context={"limit": 15}
        )

        assert result["status"] == "success"
        assert "vector" in result["strategies_used"]
        assert result["intent"]["intent_type"] == "VECTOR_RAG"

    async def test_hybrid_strategy_multi_source(
        self, orchestrator, mock_intent_classifier
    ):
        """Test hybrid strategy uses multiple sources."""
        mock_intent_classifier.classify_intent.return_value = ClassificationResult(
            intent=QueryIntentType.HYBRID,
            confidence=0.7,
            confidence_level=ConfidenceLevel.MEDIUM,
            strategy=IntentStrategy.MULTI_SOURCE,
            reasoning="Complex query requiring multiple sources",
        )

        result = await orchestrator.process_query(
            "Complex query requiring multiple approaches", context={"limit": 20}
        )

        assert result["status"] == "success"
        assert len(result["strategies_used"]) > 1
        assert result["intent"]["intent_type"] == "HYBRID"

    async def test_parallel_execution_enabled(
        self, orchestrator, mock_intent_classifier
    ):
        """Test parallel execution when multiple strategies are used."""
        orchestrator.parallel_enabled = True

        mock_intent_classifier.classify_intent.return_value = ClassificationResult(
            intent=QueryIntentType.HYBRID,
            confidence=0.8,
            confidence_level=ConfidenceLevel.HIGH,
            strategy=IntentStrategy.MULTI_SOURCE,
            reasoning="Multi-source query",
        )

        result = await orchestrator.process_query("Test parallel execution")

        assert result["status"] == "success"
        assert result["metadata"]["parallel_execution"]

    async def test_sequential_execution_disabled(
        self, orchestrator, mock_intent_classifier
    ):
        """Test sequential execution when parallel is disabled."""
        orchestrator.parallel_enabled = False

        mock_intent_classifier.classify_intent.return_value = ClassificationResult(
            intent=QueryIntentType.HYBRID,
            confidence=0.8,
            confidence_level=ConfidenceLevel.HIGH,
            strategy=IntentStrategy.MULTI_SOURCE,
            reasoning="Multi-source query",
        )

        result = await orchestrator.process_query("Test sequential execution")

        assert result["status"] == "success"
        assert not result["metadata"]["parallel_execution"]

    async def test_error_handling_strategy_failure(
        self, orchestrator, mock_intent_classifier, mock_graph_plugin
    ):
        """Test error handling when a strategy fails."""
        # Make graph plugin fail
        mock_graph_plugin.natural_language_query.side_effect = Exception(
            "Graph query failed"
        )

        mock_intent_classifier.classify_intent.return_value = ClassificationResult(
            intent=QueryIntentType.GRAPH_RAG,
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            strategy=IntentStrategy.GRAPH_TRAVERSAL,
            reasoning="Graph query",
        )

        result = await orchestrator.process_query("Test error handling")

        # Should still succeed if other strategies work
        assert len(result["strategies_failed"]) > 0
        assert result["strategies_failed"][0]["strategy"] == "graph"

    async def test_timeout_handling(self, orchestrator, mock_intent_classifier):
        """Test timeout handling for long-running queries."""
        orchestrator.timeout_seconds = 0.1  # Very short timeout

        # Mock a slow strategy
        async def slow_execute(*args, **kwargs):
            await asyncio.sleep(0.2)  # Longer than timeout
            return {"status": "success", "results": []}

        orchestrator.strategies["vector"].execute = slow_execute

        mock_intent_classifier.classify_intent.return_value = ClassificationResult(
            intent=QueryIntentType.VECTOR_RAG,
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            strategy=IntentStrategy.SEMANTIC_SEARCH,
            reasoning="Vector query",
        )

        result = await orchestrator.process_query("Test timeout")

        # Should handle timeout gracefully
        assert "timeout" in str(result).lower() or result["status"] == "error"

    async def test_low_confidence_fallback(self, orchestrator, mock_intent_classifier):
        """Test fallback behavior for low confidence classifications."""
        mock_intent_classifier.classify_intent.return_value = ClassificationResult(
            intent=QueryIntentType.VECTOR_RAG,
            confidence=0.5,  # Low confidence
            confidence_level=ConfidenceLevel.LOW,
            strategy=IntentStrategy.SEMANTIC_SEARCH,
            reasoning="Low confidence classification",
        )

        result = await orchestrator.process_query(
            "Ambiguous query", context={"use_multiple_sources": True}
        )

        # Should use multiple sources due to low confidence
        assert result["status"] == "success"
        assert len(result["strategies_used"]) >= 1

    async def test_orchestrator_status_metrics(self, orchestrator):
        """Test orchestrator status and performance metrics."""
        status = orchestrator.get_status()

        assert "parallel_enabled" in status
        assert "max_sources" in status
        assert "timeout_seconds" in status
        assert "strategies_performance" in status
        assert len(status["strategies_performance"]) == 3

    async def test_singleton_orchestrator_pattern(self):
        """Test singleton pattern for orchestrator instance."""
        orchestrator1 = await get_omnirag_orchestrator()
        orchestrator2 = await get_omnirag_orchestrator()

        assert orchestrator1 is orchestrator2

    async def test_database_strategy_entity_extraction(self):
        """Test database strategy entity extraction."""
        strategy = DatabaseRetrievalStrategy()

        entities = strategy._extract_entity_names(
            "Find dependencies of Flask and Django classes"
        )

        assert "Flask" in entities
        assert "Django" in entities

    async def test_result_aggregation_and_deduplication(
        self, orchestrator, mock_intent_classifier
    ):
        """Test result aggregation from multiple strategies."""
        mock_intent_classifier.classify_intent.return_value = ClassificationResult(
            intent=QueryIntentType.HYBRID,
            confidence=0.8,
            confidence_level=ConfidenceLevel.HIGH,
            strategy=IntentStrategy.MULTI_SOURCE,
            reasoning="Multi-source query",
        )

        result = await orchestrator.process_query("Test result aggregation")

        assert result["status"] == "success"
        assert "results" in result
        assert result["metadata"]["total_results"] >= 0


class TestRetrievalStrategies:
    """Test individual retrieval strategies."""

    async def test_graph_strategy_execution(self):
        """Test graph retrieval strategy execution."""
        strategy = GraphRetrievalStrategy()
        mock_plugin = AsyncMock()
        mock_plugin.natural_language_query.return_value = {
            "success": True,
            "results": [{"id": "1", "content": "test"}],
            "metadata": {"confidence": 0.9},
        }

        await strategy.initialize(mock_plugin)

        mock_intent = ClassificationResult(
            intent=QueryIntentType.GRAPH_RAG,
            confidence=0.9,
            confidence_level=ConfidenceLevel.HIGH,
            strategy=IntentStrategy.GRAPH_TRAVERSAL,
            reasoning="Graph query",
        )

        result = await strategy.execute("test query", mock_intent, {"limit": 10})

        assert result["status"] == "success"
        assert result["strategy"] == "graph"
        assert result["execution_time_ms"] > 0

    async def test_vector_strategy_execution(self):
        """Test vector retrieval strategy execution."""
        strategy = VectorRetrievalStrategy()
        mock_plugin = AsyncMock()
        mock_plugin.hybrid_search.return_value = [
            MagicMock(model_dump=lambda: {"id": "1", "content": "test"})
        ]

        await strategy.initialize(mock_plugin)

        mock_intent = ClassificationResult(
            intent=QueryIntentType.VECTOR_RAG,
            confidence=0.85,
            confidence_level=ConfidenceLevel.HIGH,
            strategy=IntentStrategy.SEMANTIC_SEARCH,
            reasoning="Vector query",
        )

        result = await strategy.execute("test query", mock_intent, {"limit": 10})

        assert result["status"] == "success"
        assert result["strategy"] == "vector"
        assert result["execution_time_ms"] > 0

    async def test_database_strategy_execution(self):
        """Test database retrieval strategy execution."""
        strategy = DatabaseRetrievalStrategy()
        mock_client = AsyncMock()
        mock_client.query_items.return_value = [{"id": "1", "content": "test"}]

        await strategy.initialize(mock_client)

        mock_intent = ClassificationResult(
            intent=QueryIntentType.DATABASE_RAG,
            confidence=0.8,
            confidence_level=ConfidenceLevel.HIGH,
            strategy=IntentStrategy.STRUCTURED_QUERY,
            reasoning="Database query",
        )

        result = await strategy.execute("Find Flask entity", mock_intent, {"limit": 10})

        assert result["status"] == "success"
        assert result["strategy"] == "database"
        assert result["execution_time_ms"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
