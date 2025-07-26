"""
MCP tools for context aggregation and fusion operations.

This module provides MCP tools for the OmniRAG context aggregation system,
enabling multi-source result fusion, deduplication, and relevance scoring.
"""

import logging
from typing import Any, Dict, List, Optional

from ..models.aggregation_models import (
    AggregationConfig,
    AggregationRequest,
    AggregationStrategy,
    ContextItem,
    MultiSourceResult,
    SourceResult,
    SourceType,
)
from .context_aggregator import ContextAggregator

logger = logging.getLogger(__name__)

# Global context aggregator instance
_context_aggregator: Optional[ContextAggregator] = None


def get_context_aggregator() -> ContextAggregator:
    """Get or create the global context aggregator instance."""
    global _context_aggregator
    if _context_aggregator is None:
        _context_aggregator = ContextAggregator()
    return _context_aggregator


def reset_context_aggregator() -> None:
    """Reset the global context aggregator instance."""
    global _context_aggregator
    _context_aggregator = None


class ContextAggregationMCPTools:
    """MCP tools for context aggregation operations."""

    def __init__(self, context_aggregator: Optional[ContextAggregator] = None):
        """
        Initialize MCP tools for context aggregation.

        Args:
            context_aggregator: Optional context aggregator instance
        """
        self.context_aggregator = context_aggregator or get_context_aggregator()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP-compatible tool definitions."""
        return [
            self._aggregate_context_tool_definition(),
            self._rank_context_tool_definition(),
            self._get_aggregation_stats_tool_definition(),
            self._configure_aggregation_tool_definition(),
            self._health_check_tool_definition(),
        ]

    def _aggregate_context_tool_definition(self) -> Dict[str, Any]:
        """Tool definition for aggregating multi-source context results."""
        return {
            "name": "aggregate_context",
            "description": "Aggregate and rank results from multiple retrieval sources with deduplication and diversity optimization",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The original query text",
                    },
                    "query_intent": {
                        "type": "string",
                        "description": "Detected query intent type (GRAPH_RAG, VECTOR_RAG, DATABASE_RAG, HYBRID)",
                    },
                    "source_results": {
                        "type": "array",
                        "description": "Results from multiple retrieval sources",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_type": {
                                    "type": "string",
                                    "enum": ["graph", "vector", "database", "hybrid"],
                                },
                                "source_id": {"type": "string"},
                                "source_reliability": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "content": {"type": "string"},
                                            "confidence_score": {
                                                "type": "number",
                                                "minimum": 0,
                                                "maximum": 1,
                                            },
                                            "metadata": {"type": "object"},
                                        },
                                        "required": ["content"],
                                    },
                                },
                                "execution_time_ms": {"type": "number"},
                                "error_message": {"type": "string"},
                            },
                            "required": ["source_type", "source_id", "items"],
                        },
                    },
                    "strategy": {
                        "type": "string",
                        "enum": [
                            "weighted_fusion",
                            "ranked_merge",
                            "diversity_first",
                            "relevance_first",
                            "balanced",
                        ],
                        "description": "Aggregation strategy to use",
                    },
                    "config": {
                        "type": "object",
                        "description": "Aggregation configuration",
                        "properties": {
                            "max_items": {
                                "type": "integer",
                                "minimum": 1,
                                "default": 10,
                            },
                            "similarity_threshold": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "default": 0.85,
                            },
                            "diversity_weight": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "default": 0.3,
                            },
                            "relevance_weight": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "default": 0.4,
                            },
                            "authority_weight": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "default": 0.2,
                            },
                            "recency_weight": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "default": 0.1,
                            },
                        },
                    },
                    "query_embedding": {
                        "type": "array",
                        "description": "Pre-computed query embedding (optional)",
                        "items": {"type": "number"},
                    },
                },
                "required": ["query", "query_intent", "source_results"],
            },
        }

    def _rank_context_tool_definition(self) -> Dict[str, Any]:
        """Tool definition for ranking context items by relevance and diversity."""
        return {
            "name": "rank_context",
            "description": "Rank context items by relevance and diversity without full aggregation",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The original query text",
                    },
                    "items": {
                        "type": "array",
                        "description": "Context items to rank",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "source_type": {
                                    "type": "string",
                                    "enum": ["graph", "vector", "database", "hybrid"],
                                },
                                "source_id": {"type": "string"},
                                "confidence_score": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "metadata": {"type": "object"},
                            },
                            "required": ["content", "source_type", "source_id"],
                        },
                    },
                    "config": {
                        "type": "object",
                        "description": "Ranking configuration",
                        "properties": {
                            "max_items": {
                                "type": "integer",
                                "minimum": 1,
                                "default": 10,
                            },
                            "diversity_weight": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "default": 0.3,
                            },
                            "relevance_weight": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "default": 0.4,
                            },
                            "authority_weight": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "default": 0.2,
                            },
                            "recency_weight": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "default": 0.1,
                            },
                        },
                    },
                    "query_embedding": {
                        "type": "array",
                        "description": "Pre-computed query embedding (optional)",
                        "items": {"type": "number"},
                    },
                },
                "required": ["query", "items"],
            },
        }

    def _get_aggregation_stats_tool_definition(self) -> Dict[str, Any]:
        """Tool definition for getting aggregation statistics."""
        return {
            "name": "get_aggregation_stats",
            "description": "Get current statistics about context aggregation performance",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        }

    def _configure_aggregation_tool_definition(self) -> Dict[str, Any]:
        """Tool definition for configuring aggregation behavior."""
        return {
            "name": "configure_aggregation",
            "description": "Configure context aggregation behavior and parameters",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "embedding_model_name": {
                        "type": "string",
                        "description": "Name of the sentence transformer model to use",
                    },
                    "enable_gpu": {
                        "type": "boolean",
                        "description": "Whether to enable GPU acceleration",
                    },
                    "reset_stats": {
                        "type": "boolean",
                        "description": "Whether to reset aggregation statistics",
                    },
                },
                "required": [],
            },
        }

    def _health_check_tool_definition(self) -> Dict[str, Any]:
        """Tool definition for checking aggregation system health."""
        return {
            "name": "aggregation_health_check",
            "description": "Check the health status of the context aggregation system",
            "inputSchema": {"type": "object", "properties": {}, "required": []},
        }

    async def handle_tool_call(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle MCP tool calls for context aggregation.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        try:
            if tool_name == "aggregate_context":
                return await self._handle_aggregate_context(arguments)
            elif tool_name == "rank_context":
                return await self._handle_rank_context(arguments)
            elif tool_name == "get_aggregation_stats":
                return await self._handle_get_stats(arguments)
            elif tool_name == "configure_aggregation":
                return await self._handle_configure(arguments)
            elif tool_name == "aggregation_health_check":
                return await self._handle_health_check(arguments)
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "tool_name": tool_name,
                }

        except Exception as e:
            logger.error(f"Error handling tool call {tool_name}: {e}")
            return {"success": False, "error": str(e), "tool_name": tool_name}

    async def _handle_aggregate_context(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle aggregate_context tool call."""
        try:
            # Parse arguments
            query = arguments["query"]
            query_intent = arguments["query_intent"]
            source_results_data = arguments["source_results"]
            strategy = arguments.get("strategy", "balanced")
            config_data = arguments.get("config", {})
            query_embedding = arguments.get("query_embedding")

            # Create source results
            source_results = []
            for sr_data in source_results_data:
                items = []
                for item_data in sr_data["items"]:
                    item = ContextItem(
                        content=item_data["content"],
                        source_type=SourceType(sr_data["source_type"]),
                        source_id=sr_data["source_id"],
                        confidence_score=item_data.get("confidence_score", 0.0),
                        metadata=item_data.get("metadata", {}),
                    )
                    items.append(item)

                source_result = SourceResult(
                    source_type=SourceType(sr_data["source_type"]),
                    source_id=sr_data["source_id"],
                    source_reliability=sr_data.get("source_reliability", 1.0),
                    items=items,
                    execution_time_ms=sr_data.get("execution_time_ms", 0.0),
                    error_message=sr_data.get("error_message"),
                )
                source_results.append(source_result)

            # Create multi-source result
            multi_source_result = MultiSourceResult(
                query=query, query_intent=query_intent, source_results=source_results
            )

            # Create configuration
            config = AggregationConfig(**config_data)

            # Create aggregation request
            request = AggregationRequest(
                multi_source_result=multi_source_result,
                strategy=AggregationStrategy(strategy),
                config=config,
                query_embedding=query_embedding,
            )

            # Perform aggregation
            result = await self.context_aggregator.aggregate_results(request)

            return {
                "success": True,
                "result": {
                    "query": result.query,
                    "strategy_used": result.strategy_used.value,
                    "aggregated_items": [
                        {
                            "content": item.content,
                            "source_info": item.source_info,
                            "rank": item.rank,
                            "relevance_score": {
                                "query_similarity": item.relevance_score.query_similarity,
                                "source_confidence": item.relevance_score.source_confidence,
                                "authority_score": item.relevance_score.authority_score,
                                "recency_score": item.relevance_score.recency_score,
                                "diversity_penalty": item.relevance_score.diversity_penalty,
                                "final_score": item.relevance_score.final_score,
                            },
                            "is_diverse": item.is_diverse,
                            "metadata": item.metadata,
                        }
                        for item in result.aggregated_items
                    ],
                    "total_items_processed": result.total_items_processed,
                    "duplicates_removed": result.duplicates_removed,
                    "diversity_score": result.diversity_score,
                    "aggregation_time_ms": result.aggregation_time_ms,
                    "metadata": result.metadata,
                },
            }

        except Exception as e:
            logger.error(f"Error in aggregate_context: {e}")
            return {"success": False, "error": str(e), "tool_name": "aggregate_context"}

    async def _handle_rank_context(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle rank_context tool call."""
        try:
            query = arguments["query"]
            items_data = arguments["items"]
            config_data = arguments.get("config", {})
            query_embedding = arguments.get("query_embedding")

            # Create context items
            items = []
            for item_data in items_data:
                item = ContextItem(
                    content=item_data["content"],
                    source_type=SourceType(item_data["source_type"]),
                    source_id=item_data["source_id"],
                    confidence_score=item_data.get("confidence_score", 0.0),
                    metadata=item_data.get("metadata", {}),
                )
                items.append(item)

            # Create configuration
            config = AggregationConfig(**config_data)

            # Rank items
            ranked_items = await self.context_aggregator.rank_context(
                items, query, config, query_embedding
            )

            return {
                "success": True,
                "result": {
                    "query": query,
                    "ranked_items": [
                        {
                            "content": item.content,
                            "source_info": item.source_info,
                            "rank": item.rank,
                            "relevance_score": {
                                "query_similarity": item.relevance_score.query_similarity,
                                "source_confidence": item.relevance_score.source_confidence,
                                "authority_score": item.relevance_score.authority_score,
                                "recency_score": item.relevance_score.recency_score,
                                "diversity_penalty": item.relevance_score.diversity_penalty,
                                "final_score": item.relevance_score.final_score,
                            },
                            "is_diverse": item.is_diverse,
                            "metadata": item.metadata,
                        }
                        for item in ranked_items
                    ],
                },
            }

        except Exception as e:
            logger.error(f"Error in rank_context: {e}")
            return {"success": False, "error": str(e), "tool_name": "rank_context"}

    async def _handle_get_stats(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_aggregation_stats tool call."""
        try:
            stats = self.context_aggregator.get_stats()

            return {
                "success": True,
                "result": {
                    "total_requests": stats.total_requests,
                    "average_processing_time_ms": stats.average_processing_time_ms,
                    "average_items_processed": stats.average_items_processed,
                    "average_duplicates_removed": stats.average_duplicates_removed,
                    "strategy_usage_counts": stats.strategy_usage_counts,
                    "last_updated": stats.last_updated.isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Error in get_aggregation_stats: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": "get_aggregation_stats",
            }

    async def _handle_configure(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle configure_aggregation tool call."""
        try:
            result_info = {}

            # Handle embedding model change
            if "embedding_model_name" in arguments:
                new_model = arguments["embedding_model_name"]
                old_model = self.context_aggregator.embedding_model_name
                self.context_aggregator.embedding_model_name = new_model
                self.context_aggregator._embedding_model = None  # Reset to reload
                result_info["embedding_model_changed"] = f"{old_model} -> {new_model}"

            # Handle GPU setting
            if "enable_gpu" in arguments:
                self.context_aggregator.enable_gpu = arguments["enable_gpu"]
                result_info["gpu_enabled"] = arguments["enable_gpu"]

            # Handle stats reset
            if arguments.get("reset_stats", False):
                from ..models.aggregation_models import AggregationStats

                self.context_aggregator._stats = AggregationStats()
                result_info["stats_reset"] = True

            return {"success": True, "result": result_info}

        except Exception as e:
            logger.error(f"Error in configure_aggregation: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": "configure_aggregation",
            }

    async def _handle_health_check(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle aggregation_health_check tool call."""
        try:
            health_status = await self.context_aggregator.health_check()

            return {"success": True, "result": health_status}

        except Exception as e:
            logger.error(f"Error in aggregation_health_check: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": "aggregation_health_check",
            }


def register_context_aggregation_tools() -> List[Dict[str, Any]]:
    """Register context aggregation MCP tools."""
    tools_instance = ContextAggregationMCPTools()
    return tools_instance.get_tool_definitions()


# Create global tools instance for easy access
context_aggregation_tools = ContextAggregationMCPTools()
