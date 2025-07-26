"""
Integration tests for context aggregation and fusion system.

This module tests the end-to-end integration of the OmniRAG context
aggregation system, including MCP tools and orchestrator integration.
"""

import asyncio
import logging
import pytest
from typing import List

from ..models.aggregation_models import (
    AggregationConfig,
    AggregationRequest,
    AggregationStrategy,
    ContextItem,
    MultiSourceResult,
    SourceResult,
    SourceType,
)
from ..plugins.context_aggregator import ContextAggregator
from ..plugins.aggregation_mcp_tools import ContextAggregationMCPTools

logger = logging.getLogger(__name__)


class TestContextAggregationIntegration:
    """Integration tests for context aggregation system."""

    @pytest.fixture
    def sample_source_results(self) -> List[SourceResult]:
        """Create sample source results for testing."""
        # Graph source results
        graph_items = [
            ContextItem(
                content="Azure OpenAI provides enterprise-grade AI capabilities with built-in security and compliance.",
                source_type=SourceType.GRAPH,
                source_id="graph_retriever_v1",
                confidence_score=0.9,
                metadata={"entity_type": "service", "category": "ai"},
            ),
            ContextItem(
                content="GPT-4 models in Azure OpenAI support function calling and structured outputs.",
                source_type=SourceType.GRAPH,
                source_id="graph_retriever_v1",
                confidence_score=0.85,
                metadata={"entity_type": "feature", "category": "ai"},
            ),
        ]

        # Vector source results
        vector_items = [
            ContextItem(
                content="Azure OpenAI Service offers powerful language models including GPT-4, GPT-3.5, and Embeddings.",
                source_type=SourceType.VECTOR,
                source_id="vector_search_v1",
                confidence_score=0.88,
                metadata={"document_id": "azure_ai_docs_001", "section": "overview"},
            ),
            ContextItem(
                content="Use Azure OpenAI for content generation, summarization, and semantic search applications.",
                source_type=SourceType.VECTOR,
                source_id="vector_search_v1",
                confidence_score=0.82,
                metadata={"document_id": "azure_ai_docs_002", "section": "use_cases"},
            ),
        ]

        # Database source results
        database_items = [
            ContextItem(
                content="Azure OpenAI pricing is based on token usage with different rates for input and output tokens.",
                source_type=SourceType.DATABASE,
                source_id="database_query_v1",
                confidence_score=0.78,
                metadata={"table": "pricing_info", "last_updated": "2025-01-01"},
            ),
        ]

        return [
            SourceResult(
                source_type=SourceType.GRAPH,
                source_id="graph_retriever_v1",
                source_reliability=0.95,
                items=graph_items,
                execution_time_ms=150.0,
            ),
            SourceResult(
                source_type=SourceType.VECTOR,
                source_id="vector_search_v1",
                source_reliability=0.90,
                items=vector_items,
                execution_time_ms=200.0,
            ),
            SourceResult(
                source_type=SourceType.DATABASE,
                source_id="database_query_v1",
                source_reliability=0.85,
                items=database_items,
                execution_time_ms=100.0,
            ),
        ]

    @pytest.fixture
    def context_aggregator(self) -> ContextAggregator:
        """Create a context aggregator instance for testing."""
        return ContextAggregator()

    @pytest.fixture
    def mcp_tools(
        self, context_aggregator: ContextAggregator
    ) -> ContextAggregationMCPTools:
        """Create MCP tools instance for testing."""
        return ContextAggregationMCPTools(context_aggregator)

    @pytest.mark.asyncio
    async def test_end_to_end_aggregation(
        self,
        context_aggregator: ContextAggregator,
        sample_source_results: List[SourceResult],
    ):
        """Test end-to-end context aggregation."""
        # Create multi-source result
        multi_source_result = MultiSourceResult(
            query="What are the capabilities of Azure OpenAI?",
            query_intent="HYBRID",
            source_results=sample_source_results,
        )

        # Create aggregation request
        request = AggregationRequest(
            multi_source_result=multi_source_result,
            strategy=AggregationStrategy.BALANCED,
            config=AggregationConfig(
                max_items=5,
                similarity_threshold=0.85,
                diversity_weight=0.3,
                relevance_weight=0.4,
                authority_weight=0.2,
                recency_weight=0.1,
            ),
        )

        # Perform aggregation
        result = await context_aggregator.aggregate_results(request)

        # Verify results
        assert result.query == "What are the capabilities of Azure OpenAI?"
        assert result.strategy_used == AggregationStrategy.BALANCED
        assert len(result.aggregated_items) <= 5
        assert result.total_items_processed == 5
        assert result.aggregation_time_ms > 0

        # Verify ranking
        scores = [item.relevance_score.final_score for item in result.aggregated_items]
        assert scores == sorted(scores, reverse=True), "Items should be ranked by score"

        # Verify diversity
        assert result.diversity_score >= 0.0

        # Verify at least one item from each source type
        source_types = {
            item.source_info["source_type"] for item in result.aggregated_items
        }
        assert len(source_types) >= 2, "Should include items from multiple source types"

    @pytest.mark.asyncio
    async def test_mcp_tool_aggregate_context(
        self,
        mcp_tools: ContextAggregationMCPTools,
        sample_source_results: List[SourceResult],
    ):
        """Test aggregate_context MCP tool."""
        # Prepare tool arguments
        arguments = {
            "query": "What are the capabilities of Azure OpenAI?",
            "query_intent": "HYBRID",
            "source_results": [
                {
                    "source_type": sr.source_type.value,
                    "source_id": sr.source_id,
                    "source_reliability": sr.source_reliability,
                    "items": [
                        {
                            "content": item.content,
                            "confidence_score": item.confidence_score,
                            "metadata": item.metadata,
                        }
                        for item in sr.items
                    ],
                    "execution_time_ms": sr.execution_time_ms,
                }
                for sr in sample_source_results
            ],
            "strategy": "balanced",
            "config": {
                "max_items": 5,
                "similarity_threshold": 0.85,
                "diversity_weight": 0.3,
                "relevance_weight": 0.4,
                "authority_weight": 0.2,
                "recency_weight": 0.1,
            },
        }

        # Call MCP tool
        result = await mcp_tools.handle_tool_call("aggregate_context", arguments)

        # Verify response
        assert result["success"] is True
        assert "result" in result

        tool_result = result["result"]
        assert tool_result["query"] == "What are the capabilities of Azure OpenAI?"
        assert tool_result["strategy_used"] == "balanced"
        assert len(tool_result["aggregated_items"]) <= 5
        assert tool_result["total_items_processed"] == 5
        assert tool_result["aggregation_time_ms"] > 0

        # Verify aggregated items structure
        for item in tool_result["aggregated_items"]:
            assert "content" in item
            assert "source_info" in item
            assert "rank" in item
            assert "relevance_score" in item
            assert "is_diverse" in item
            assert "metadata" in item

            # Verify relevance score structure
            relevance_score = item["relevance_score"]
            assert "query_similarity" in relevance_score
            assert "source_confidence" in relevance_score
            assert "authority_score" in relevance_score
            assert "recency_score" in relevance_score
            assert "diversity_penalty" in relevance_score
            assert "final_score" in relevance_score

    @pytest.mark.asyncio
    async def test_mcp_tool_rank_context(self, mcp_tools: ContextAggregationMCPTools):
        """Test rank_context MCP tool."""
        # Prepare tool arguments
        arguments = {
            "query": "Azure OpenAI pricing",
            "items": [
                {
                    "content": "Azure OpenAI pricing is based on token usage.",
                    "source_type": "database",
                    "source_id": "pricing_db",
                    "confidence_score": 0.9,
                    "metadata": {"table": "pricing"},
                },
                {
                    "content": "GPT-4 costs more than GPT-3.5 in Azure OpenAI.",
                    "source_type": "vector",
                    "source_id": "docs_vector",
                    "confidence_score": 0.85,
                    "metadata": {"document": "pricing_guide"},
                },
            ],
            "config": {
                "max_items": 10,
                "diversity_weight": 0.2,
                "relevance_weight": 0.5,
                "authority_weight": 0.2,
                "recency_weight": 0.1,
            },
        }

        # Call MCP tool
        result = await mcp_tools.handle_tool_call("rank_context", arguments)

        # Verify response
        assert result["success"] is True
        assert "result" in result

        tool_result = result["result"]
        assert tool_result["query"] == "Azure OpenAI pricing"
        assert len(tool_result["ranked_items"]) == 2

        # Verify ranking order
        scores = [
            item["relevance_score"]["final_score"]
            for item in tool_result["ranked_items"]
        ]
        assert scores == sorted(scores, reverse=True), "Items should be ranked by score"

    @pytest.mark.asyncio
    async def test_mcp_tool_get_stats(self, mcp_tools: ContextAggregationMCPTools):
        """Test get_aggregation_stats MCP tool."""
        # First perform some aggregations to generate stats
        context_aggregator = mcp_tools.context_aggregator

        # Create a simple multi-source result
        source_result = SourceResult(
            source_type=SourceType.VECTOR,
            source_id="test_source",
            source_reliability=0.9,
            items=[
                ContextItem(
                    content="Test content",
                    source_type=SourceType.VECTOR,
                    source_id="test_source",
                    confidence_score=0.8,
                )
            ],
            execution_time_ms=100.0,
        )

        multi_source_result = MultiSourceResult(
            query="test query",
            query_intent="VECTOR_RAG",
            source_results=[source_result],
        )

        request = AggregationRequest(
            multi_source_result=multi_source_result,
            strategy=AggregationStrategy.RELEVANCE_FIRST,
            config=AggregationConfig(),
        )

        # Perform aggregation to generate stats
        await context_aggregator.aggregate_results(request)

        # Call stats tool
        result = await mcp_tools.handle_tool_call("get_aggregation_stats", {})

        # Verify response
        assert result["success"] is True
        assert "result" in result

        stats = result["result"]
        assert stats["total_requests"] >= 1
        assert stats["average_processing_time_ms"] > 0
        assert stats["average_items_processed"] >= 1
        assert "strategy_usage_counts" in stats
        assert "last_updated" in stats

    @pytest.mark.asyncio
    async def test_mcp_tool_configure(self, mcp_tools: ContextAggregationMCPTools):
        """Test configure_aggregation MCP tool."""
        # Test configuration changes
        arguments = {
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "enable_gpu": False,
            "reset_stats": True,
        }

        # Call configuration tool
        result = await mcp_tools.handle_tool_call("configure_aggregation", arguments)

        # Verify response
        assert result["success"] is True
        assert "result" in result

        config_result = result["result"]
        assert "embedding_model_changed" in config_result
        assert "gpu_enabled" in config_result
        assert config_result["stats_reset"] is True

        # Verify changes were applied
        context_aggregator = mcp_tools.context_aggregator
        assert (
            context_aggregator.embedding_model_name
            == "sentence-transformers/all-MiniLM-L6-v2"
        )
        assert context_aggregator.enable_gpu is False

    @pytest.mark.asyncio
    async def test_mcp_tool_health_check(self, mcp_tools: ContextAggregationMCPTools):
        """Test aggregation_health_check MCP tool."""
        # Call health check tool
        result = await mcp_tools.handle_tool_call("aggregation_health_check", {})

        # Verify response
        assert result["success"] is True
        assert "result" in result

        health_status = result["result"]
        assert "status" in health_status
        assert "embedding_model_loaded" in health_status
        assert "memory_usage_mb" in health_status
        assert "last_aggregation_time" in health_status

    @pytest.mark.asyncio
    async def test_error_handling(self, mcp_tools: ContextAggregationMCPTools):
        """Test error handling in MCP tools."""
        # Test with invalid tool name
        result = await mcp_tools.handle_tool_call("invalid_tool", {})
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

        # Test with invalid arguments for aggregate_context
        invalid_arguments = {
            "query": "test",
            # Missing required fields
        }

        result = await mcp_tools.handle_tool_call(
            "aggregate_context", invalid_arguments
        )
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_deduplication_behavior(self, context_aggregator: ContextAggregator):
        """Test deduplication behavior with similar content."""
        # Create source results with similar content
        similar_items = [
            ContextItem(
                content="Azure OpenAI provides enterprise AI capabilities.",
                source_type=SourceType.GRAPH,
                source_id="source1",
                confidence_score=0.9,
            ),
            ContextItem(
                content="Azure OpenAI offers enterprise-grade AI capabilities.",
                source_type=SourceType.VECTOR,
                source_id="source2",
                confidence_score=0.85,
            ),
            ContextItem(
                content="Completely different content about database optimization.",
                source_type=SourceType.DATABASE,
                source_id="source3",
                confidence_score=0.8,
            ),
        ]

        # Create source results
        source_results = [
            SourceResult(
                source_type=SourceType.GRAPH,
                source_id="source1",
                source_reliability=0.9,
                items=[similar_items[0]],
                execution_time_ms=100.0,
            ),
            SourceResult(
                source_type=SourceType.VECTOR,
                source_id="source2",
                source_reliability=0.9,
                items=[similar_items[1]],
                execution_time_ms=100.0,
            ),
            SourceResult(
                source_type=SourceType.DATABASE,
                source_id="source3",
                source_reliability=0.9,
                items=[similar_items[2]],
                execution_time_ms=100.0,
            ),
        ]

        # Create aggregation request with high similarity threshold
        multi_source_result = MultiSourceResult(
            query="Azure OpenAI capabilities",
            query_intent="HYBRID",
            source_results=source_results,
        )

        request = AggregationRequest(
            multi_source_result=multi_source_result,
            strategy=AggregationStrategy.BALANCED,
            config=AggregationConfig(
                similarity_threshold=0.8,  # High threshold for deduplication
                max_items=10,
            ),
        )

        # Perform aggregation
        result = await context_aggregator.aggregate_results(request)

        # Verify deduplication occurred
        assert result.duplicates_removed > 0, (
            "Should have detected and removed duplicates"
        )
        assert len(result.aggregated_items) < len(similar_items), (
            "Should have fewer items after deduplication"
        )

        # Verify the highest confidence item was kept
        kept_contents = [item.content for item in result.aggregated_items]
        assert any(
            "enterprise AI capabilities" in content for content in kept_contents
        ), "Should keep high-confidence similar item"

    @pytest.mark.asyncio
    async def test_diversity_optimization(self, context_aggregator: ContextAggregator):
        """Test diversity optimization behavior."""
        # Create items with different topics but same source
        diverse_items = [
            ContextItem(
                content="Azure OpenAI pricing is token-based.",
                source_type=SourceType.VECTOR,
                source_id="docs",
                confidence_score=0.9,
                metadata={"topic": "pricing"},
            ),
            ContextItem(
                content="Azure OpenAI supports GPT-4 models.",
                source_type=SourceType.VECTOR,
                source_id="docs",
                confidence_score=0.88,
                metadata={"topic": "models"},
            ),
            ContextItem(
                content="Azure OpenAI has security features.",
                source_type=SourceType.VECTOR,
                source_id="docs",
                confidence_score=0.86,
                metadata={"topic": "security"},
            ),
            ContextItem(
                content="Another pricing detail about tokens.",
                source_type=SourceType.VECTOR,
                source_id="docs",
                confidence_score=0.84,
                metadata={"topic": "pricing"},
            ),
        ]

        source_result = SourceResult(
            source_type=SourceType.VECTOR,
            source_id="docs",
            source_reliability=0.9,
            items=diverse_items,
            execution_time_ms=100.0,
        )

        multi_source_result = MultiSourceResult(
            query="Azure OpenAI information",
            query_intent="VECTOR_RAG",
            source_results=[source_result],
        )

        # Test diversity-first strategy
        request = AggregationRequest(
            multi_source_result=multi_source_result,
            strategy=AggregationStrategy.DIVERSITY_FIRST,
            config=AggregationConfig(max_items=3, diversity_weight=0.5),
        )

        result = await context_aggregator.aggregate_results(request)

        # Verify diversity in results
        topics = [item.metadata.get("topic") for item in result.aggregated_items]
        unique_topics = set(topics)

        assert len(unique_topics) >= 2, "Should prioritize diverse topics"
        assert result.diversity_score > 0.0, "Should have positive diversity score"


if __name__ == "__main__":
    # Run integration tests
    asyncio.run(pytest.main([__file__, "-v"]))
