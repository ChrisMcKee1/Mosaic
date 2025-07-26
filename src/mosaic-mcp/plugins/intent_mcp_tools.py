"""
MCP tools for Query Intent Classification.
Exposes intent classification capabilities via Model Context Protocol.
"""

import asyncio
import logging
from typing import List
from datetime import datetime

from mcp.server import Server
from mcp.types import Tool, TextContent, JSONContent

from ..models.intent_models import (
    ClassificationRequest,
    RetrainingRequest,
    TrainingConfig,
    QueryIntentType,
    RetrievalStrategy,
)
from ..plugins.query_intent_classifier import (
    get_intent_classifier,
)
from ..training.intent_training_data import generate_training_data


logger = logging.getLogger(__name__)


class IntentClassificationMCPTools:
    """MCP tools for query intent classification."""

    def __init__(self, server: Server):
        """Initialize MCP tools."""
        self.server = server
        self.tools = self._create_tools()
        self._register_tools()

    def _create_tools(self) -> List[Tool]:
        """Create MCP tool definitions."""

        return [
            Tool(
                name="classify_query_intent",
                description=(
                    "Classify the intent of a user query to determine optimal retrieval strategy. "
                    "Returns intent type (GRAPH_RAG, VECTOR_RAG, DATABASE_RAG, HYBRID), confidence score, "
                    "and recommended retrieval strategy for OmniRAG orchestration."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user query to classify",
                            "minLength": 1,
                            "maxLength": 2000,
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional context or conversation history",
                            "default": "",
                        },
                        "user_id": {
                            "type": "string",
                            "description": "Optional user identifier for logging",
                            "default": "",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Optional session identifier for logging",
                            "default": "",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="predict_retrieval_strategy",
                description=(
                    "Predict the optimal retrieval strategy for a query without full classification details. "
                    "Returns simplified strategy recommendation (GRAPH_RAG, VECTOR_RAG, DATABASE_RAG, HYBRID)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user query to analyze",
                            "minLength": 1,
                            "maxLength": 2000,
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional context information",
                            "default": "",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="batch_classify_intents",
                description=(
                    "Classify multiple queries in batch for efficiency. "
                    "Returns list of classification results in same order as input queries."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "minLength": 1},
                                    "context": {"type": "string", "default": ""},
                                    "user_id": {"type": "string", "default": ""},
                                    "session_id": {"type": "string", "default": ""},
                                },
                                "required": ["query"],
                            },
                            "minItems": 1,
                            "maxItems": 50,
                            "description": "List of queries to classify (max 50)",
                        }
                    },
                    "required": ["queries"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="get_intent_classifier_status",
                description=(
                    "Get current status and health of the intent classification system. "
                    "Returns initialization status, model metrics, performance stats, and health check results."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_performance_stats": {
                            "type": "boolean",
                            "description": "Include detailed performance statistics",
                            "default": True,
                        },
                        "include_recent_history": {
                            "type": "boolean",
                            "description": "Include recent classification history",
                            "default": False,
                        },
                        "history_limit": {
                            "type": "integer",
                            "description": "Number of recent classifications to include",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100,
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="retrain_intent_classifier",
                description=(
                    "Retrain the intent classification model with new data or configuration. "
                    "Returns training metrics and updated model information."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Reason for retraining",
                            "minLength": 1,
                        },
                        "force_retrain": {
                            "type": "boolean",
                            "description": "Force retraining even if recently trained",
                            "default": False,
                        },
                        "training_config": {
                            "type": "object",
                            "description": "Optional training configuration",
                            "properties": {
                                "train_test_split": {
                                    "type": "number",
                                    "minimum": 0.1,
                                    "maximum": 0.9,
                                },
                                "random_state": {"type": "integer"},
                                "cv_folds": {
                                    "type": "integer",
                                    "minimum": 3,
                                    "maximum": 10,
                                },
                                "model_params": {"type": "object"},
                            },
                            "additionalProperties": False,
                        },
                        "samples_per_class": {
                            "type": "integer",
                            "description": "Number of training samples per intent class",
                            "default": 200,
                            "minimum": 50,
                            "maximum": 1000,
                        },
                    },
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="generate_intent_training_data",
                description=(
                    "Generate synthetic training data for intent classification. "
                    "Returns training dataset with pattern-based synthetic examples."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "samples_per_class": {
                            "type": "integer",
                            "description": "Number of samples to generate per intent class",
                            "default": 100,
                            "minimum": 10,
                            "maximum": 500,
                        },
                        "random_seed": {
                            "type": "integer",
                            "description": "Random seed for reproducibility",
                            "default": 42,
                        },
                        "export_format": {
                            "type": "string",
                            "enum": ["json", "csv", "parquet"],
                            "description": "Export format for training data",
                            "default": "json",
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="configure_intent_classifier",
                description=(
                    "Update configuration settings for the intent classifier. "
                    "Allows real-time adjustment of confidence thresholds, fallback strategies, and monitoring."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "confidence_threshold": {
                            "type": "number",
                            "description": "Minimum confidence for predictions",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "fallback_strategy": {
                            "type": "string",
                            "enum": [
                                "GRAPH_RAG",
                                "VECTOR_RAG",
                                "DATABASE_RAG",
                                "HYBRID",
                            ],
                            "description": "Strategy to use for low-confidence predictions",
                        },
                        "log_classifications": {
                            "type": "boolean",
                            "description": "Enable/disable classification logging",
                        },
                        "performance_monitoring": {
                            "type": "boolean",
                            "description": "Enable/disable performance monitoring",
                        },
                        "max_query_length": {
                            "type": "integer",
                            "description": "Maximum allowed query length",
                            "minimum": 100,
                            "maximum": 5000,
                        },
                    },
                    "additionalProperties": False,
                },
            ),
        ]

    def _register_tools(self) -> None:
        """Register tools with MCP server."""

        for tool in self.tools:
            self.server.register_tool(tool.name, tool.description, tool.inputSchema)(
                getattr(self, f"_handle_{tool.name}")
            )

    async def _handle_classify_query_intent(self, **kwargs) -> List[TextContent]:
        """Handle query intent classification."""

        try:
            # Get classifier
            classifier = await get_intent_classifier()

            # Create request
            request = ClassificationRequest(
                query=kwargs["query"],
                context=kwargs.get("context", ""),
                user_id=kwargs.get("user_id", ""),
                session_id=kwargs.get("session_id", ""),
            )

            # Classify intent
            result = await classifier.classify_intent(request)

            # Format response
            response = {
                "intent": result.intent.value,
                "confidence": round(result.confidence, 4),
                "confidence_level": result.confidence_level.value,
                "strategy": result.strategy.value,
                "reasoning": result.reasoning,
                "alternative_intents": [
                    {"intent": alt.intent.value, "confidence": round(alt.confidence, 4)}
                    for alt in result.alternative_intents
                ],
                "timestamp": datetime.utcnow().isoformat(),
            }

            return [JSONContent(content=response)]

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return [TextContent(content=f"Error: {str(e)}")]

    async def _handle_predict_retrieval_strategy(self, **kwargs) -> List[TextContent]:
        """Handle retrieval strategy prediction."""

        try:
            classifier = await get_intent_classifier()

            strategy = await classifier.predict_strategy(
                query=kwargs["query"], context=kwargs.get("context", "")
            )

            response = {
                "strategy": strategy.value,
                "timestamp": datetime.utcnow().isoformat(),
            }

            return [JSONContent(content=response)]

        except Exception as e:
            logger.error(f"Strategy prediction failed: {e}")
            return [TextContent(content=f"Error: {str(e)}")]

    async def _handle_batch_classify_intents(self, **kwargs) -> List[TextContent]:
        """Handle batch intent classification."""

        try:
            classifier = await get_intent_classifier()
            queries = kwargs["queries"]

            # Process queries concurrently
            tasks = []
            for query_data in queries:
                request = ClassificationRequest(
                    query=query_data["query"],
                    context=query_data.get("context", ""),
                    user_id=query_data.get("user_id", ""),
                    session_id=query_data.get("session_id", ""),
                )
                tasks.append(classifier.classify_intent(request))

            results = await asyncio.gather(*tasks)

            # Format response
            response = {
                "results": [
                    {
                        "query": queries[i]["query"],
                        "intent": result.intent.value,
                        "confidence": round(result.confidence, 4),
                        "strategy": result.strategy.value,
                        "reasoning": result.reasoning,
                    }
                    for i, result in enumerate(results)
                ],
                "batch_size": len(results),
                "timestamp": datetime.utcnow().isoformat(),
            }

            return [JSONContent(content=response)]

        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            return [TextContent(content=f"Error: {str(e)}")]

    async def _handle_get_intent_classifier_status(self, **kwargs) -> List[TextContent]:
        """Handle status and health check."""

        try:
            classifier = await get_intent_classifier()

            # Get health check
            health = await classifier.health_check()

            # Get performance stats if requested
            if kwargs.get("include_performance_stats", True):
                health["performance_stats"] = classifier.get_performance_stats()

            # Get recent history if requested
            if kwargs.get("include_recent_history", False):
                limit = kwargs.get("history_limit", 10)
                health["recent_classifications"] = (
                    classifier.get_classification_history(limit)
                )

            # Get model metrics
            try:
                model_metrics = await classifier.get_model_metrics()
                if model_metrics:
                    health["model_metrics"] = {
                        "accuracy": round(model_metrics.accuracy, 4),
                        "training_samples": model_metrics.training_samples,
                        "model_version": model_metrics.model_version,
                        "precision": {
                            k: round(v, 4) for k, v in model_metrics.precision.items()
                        },
                        "recall": {
                            k: round(v, 4) for k, v in model_metrics.recall.items()
                        },
                        "f1_score": {
                            k: round(v, 4) for k, v in model_metrics.f1_score.items()
                        },
                    }
            except Exception as e:
                health["model_metrics_error"] = str(e)

            health["timestamp"] = datetime.utcnow().isoformat()

            return [JSONContent(content=health)]

        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return [TextContent(content=f"Error: {str(e)}")]

    async def _handle_retrain_intent_classifier(self, **kwargs) -> List[TextContent]:
        """Handle model retraining."""

        try:
            classifier = await get_intent_classifier()

            # Create training config
            config_data = kwargs.get("training_config", {})
            config = TrainingConfig(**config_data)

            # Generate training dataset
            samples_per_class = kwargs.get("samples_per_class", 200)
            dataset = await asyncio.get_event_loop().run_in_executor(
                None, generate_training_data, samples_per_class
            )

            # Create retraining request
            retrain_request = RetrainingRequest(
                reason=kwargs["reason"],
                force_retrain=kwargs.get("force_retrain", False),
                config=config,
                dataset=dataset,
            )

            # Retrain model
            metrics = await classifier.retrain_model(retrain_request)

            response = {
                "status": "success",
                "reason": kwargs["reason"],
                "metrics": {
                    "accuracy": round(metrics.accuracy, 4),
                    "training_samples": metrics.training_samples,
                    "model_version": metrics.model_version,
                    "training_time": metrics.training_time,
                    "precision": {k: round(v, 4) for k, v in metrics.precision.items()},
                    "recall": {k: round(v, 4) for k, v in metrics.recall.items()},
                    "f1_score": {k: round(v, 4) for k, v in metrics.f1_score.items()},
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

            return [JSONContent(content=response)]

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return [TextContent(content=f"Error: {str(e)}")]

    async def _handle_generate_intent_training_data(
        self, **kwargs
    ) -> List[TextContent]:
        """Handle training data generation."""

        try:
            samples_per_class = kwargs.get("samples_per_class", 100)
            random_seed = kwargs.get("random_seed", 42)

            # Generate data in thread pool
            loop = asyncio.get_event_loop()
            dataset = await loop.run_in_executor(
                None, generate_training_data, samples_per_class, random_seed
            )

            # Format response based on export format
            export_format = kwargs.get("export_format", "json")

            if export_format == "json":
                # Convert to JSON serializable format
                data = []
                for sample in dataset.samples:
                    data.append(
                        {
                            "query": sample.query,
                            "intent": sample.intent.value,
                            "metadata": sample.metadata,
                        }
                    )

                response = {
                    "samples": data,
                    "total_samples": len(data),
                    "samples_per_class": samples_per_class,
                    "intent_distribution": {
                        intent.value: len(
                            [s for s in dataset.samples if s.intent == intent]
                        )
                        for intent in QueryIntentType
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return [JSONContent(content=response)]

            else:
                # For CSV/Parquet, return summary info
                response = {
                    "status": "generated",
                    "total_samples": len(dataset.samples),
                    "samples_per_class": samples_per_class,
                    "export_format": export_format,
                    "intent_distribution": {
                        intent.value: len(
                            [s for s in dataset.samples if s.intent == intent]
                        )
                        for intent in QueryIntentType
                    },
                    "note": f"Dataset generated in {export_format} format",
                    "timestamp": datetime.utcnow().isoformat(),
                }

                return [JSONContent(content=response)]

        except Exception as e:
            logger.error(f"Training data generation failed: {e}")
            return [TextContent(content=f"Error: {str(e)}")]

    async def _handle_configure_intent_classifier(self, **kwargs) -> List[TextContent]:
        """Handle classifier configuration updates."""

        try:
            classifier = await get_intent_classifier()

            # Update configuration
            updated_fields = []

            if "confidence_threshold" in kwargs:
                classifier.config.confidence_threshold = kwargs["confidence_threshold"]
                updated_fields.append("confidence_threshold")

            if "fallback_strategy" in kwargs:
                strategy_map = {
                    "GRAPH_RAG": RetrievalStrategy.GRAPH_RAG,
                    "VECTOR_RAG": RetrievalStrategy.VECTOR_RAG,
                    "DATABASE_RAG": RetrievalStrategy.DATABASE_RAG,
                    "HYBRID": RetrievalStrategy.HYBRID,
                }
                classifier.config.fallback_strategy = strategy_map[
                    kwargs["fallback_strategy"]
                ]
                updated_fields.append("fallback_strategy")

            if "log_classifications" in kwargs:
                classifier.config.log_classifications = kwargs["log_classifications"]
                updated_fields.append("log_classifications")

            if "performance_monitoring" in kwargs:
                classifier.config.performance_monitoring = kwargs[
                    "performance_monitoring"
                ]
                updated_fields.append("performance_monitoring")

            if "max_query_length" in kwargs:
                classifier.config.max_query_length = kwargs["max_query_length"]
                updated_fields.append("max_query_length")

            response = {
                "status": "success",
                "updated_fields": updated_fields,
                "current_config": {
                    "confidence_threshold": classifier.config.confidence_threshold,
                    "fallback_strategy": classifier.config.fallback_strategy.value,
                    "log_classifications": classifier.config.log_classifications,
                    "performance_monitoring": classifier.config.performance_monitoring,
                    "max_query_length": classifier.config.max_query_length,
                },
                "timestamp": datetime.utcnow().isoformat(),
            }

            return [JSONContent(content=response)]

        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return [TextContent(content=f"Error: {str(e)}")]


def register_intent_classification_tools(
    server: Server,
) -> IntentClassificationMCPTools:
    """Register intent classification tools with MCP server."""

    tools = IntentClassificationMCPTools(server)
    logger.info(f"Registered {len(tools.tools)} intent classification MCP tools")
    return tools
