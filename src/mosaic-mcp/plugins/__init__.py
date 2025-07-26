"""
Mosaic MCP Plugins.
Extensible plugin system for Model Context Protocol integrations.
"""

from .memory import MemoryPlugin
from .retrieval import RetrievalPlugin
from .refinement import RefinementPlugin
from .vector_search import VectorSearchPlugin
from .graph_data_service import GraphDataService
from .graph_visualization import GraphVisualizationPlugin
from .diagram import DiagramPlugin
from .query_intent_classifier import (
    QueryIntentClassifier,
    get_intent_classifier,
    reset_intent_classifier,
)
from .intent_mcp_tools import (
    IntentClassificationMCPTools,
    register_intent_classification_tools,
)
from .omnirag_orchestrator import (
    OmniRAGOrchestrator,
    get_omnirag_orchestrator,
    RetrievalStrategy,
    GraphRetrievalStrategy,
    VectorRetrievalStrategy,
    DatabaseRetrievalStrategy,
)
from .context_aggregator import ContextAggregator
from .aggregation_mcp_tools import (
    ContextAggregationMCPTools,
    context_aggregation_tools,
    register_context_aggregation_tools,
)

__all__ = [
    # Memory plugins
    "MemoryPlugin",
    # Retrieval plugins
    "RetrievalPlugin",
    # Refinement plugins
    "RefinementPlugin",
    # Vector search plugins
    "VectorSearchPlugin",
    # Graph data service plugins
    "GraphDataService",
    # Graph visualization plugins
    "GraphVisualizationPlugin",
    # Diagram plugins
    "DiagramPlugin",
    # Intent classification plugins
    "QueryIntentClassifier",
    "get_intent_classifier",
    "reset_intent_classifier",
    "IntentClassificationMCPTools",
    "register_intent_classification_tools",
    # OmniRAG orchestration plugins
    "OmniRAGOrchestrator",
    "get_omnirag_orchestrator",
    "RetrievalStrategy",
    "GraphRetrievalStrategy",
    "VectorRetrievalStrategy",
    "DatabaseRetrievalStrategy",
    # Context aggregation plugins
    "ContextAggregator",
    "ContextAggregationMCPTools",
    "context_aggregation_tools",
    "register_context_aggregation_tools",
]
