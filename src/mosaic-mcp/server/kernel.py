"""
Semantic Kernel Manager for Mosaic MCP Tool

Manages Semantic Kernel initialization, plugin loading, and service configuration.
Implements FR-2 requirement that ALL functionality must be implemented as
Semantic Kernel Plugins.
"""

import logging
from typing import Any, Dict, Optional

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
)
from semantic_kernel.core_plugins import TimePlugin, TextPlugin

from ..config.settings import MosaicSettings
from ..plugins.retrieval import RetrievalPlugin
from ..plugins.vector_search import VectorSearchPlugin
from ..plugins.refinement import RefinementPlugin
from ..plugins.memory import MemoryPlugin
from ..plugins.diagram import DiagramPlugin


logger = logging.getLogger(__name__)


class SemanticKernelManager:
    """
    Manages Semantic Kernel instance and plugin lifecycle.

    Provides centralized management for:
    - Kernel initialization with Azure services
    - Plugin registration and lifecycle
    - Service connector configuration
    - Plugin status monitoring
    """

    def __init__(self, settings: MosaicSettings):
        """Initialize the Semantic Kernel Manager."""
        self.settings = settings
        self.kernel: Optional[Kernel] = None
        self.plugins: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize Semantic Kernel and register all plugins."""
        try:
            # Validate required settings
            self.settings.validate_required_settings()

            # Create and configure kernel
            self.kernel = Kernel()

            # Configure Azure OpenAI service
            await self._configure_azure_openai()

            # Register core plugins
            await self._register_core_plugins()

            # Register Mosaic-specific plugins
            await self._register_mosaic_plugins()

            logger.info("Semantic Kernel initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Semantic Kernel: {e}")
            raise

    async def _configure_azure_openai(self) -> None:
        """Configure Azure OpenAI service connector with managed identity."""
        try:
            # Add Azure Chat Completion service (uses managed identity automatically)
            chat_service = AzureChatCompletion(
                deployment_name=self.settings.azure_openai_chat_deployment_name,
                endpoint=self.settings.azure_openai_endpoint,
                service_id="azure_openai_chat",
            )
            self.kernel.add_service(chat_service)

            # Add Azure Text Embedding service (uses managed identity automatically)
            embedding_service = AzureTextEmbedding(
                deployment_name=self.settings.azure_openai_text_embedding_deployment_name,
                endpoint=self.settings.azure_openai_endpoint,
                service_id="azure_openai_embedding",
            )
            self.kernel.add_service(embedding_service)

            logger.info("Azure OpenAI services configured with managed identity")

        except Exception as e:
            logger.error(f"Failed to configure Azure OpenAI: {e}")
            raise

    async def _register_core_plugins(self) -> None:
        """Register core Semantic Kernel plugins."""
        try:
            # Time plugin for temporal operations
            self.kernel.add_plugin(TimePlugin(), plugin_name="time")

            # Text plugin for text manipulation
            self.kernel.add_plugin(TextPlugin(), plugin_name="text")

            logger.info("Core plugins registered")

        except Exception as e:
            logger.error(f"Failed to register core plugins: {e}")
            raise

    async def _register_mosaic_plugins(self) -> None:
        """Register Mosaic Query Server plugins implementing FR-5 through FR-13."""
        try:
            # Vector Search Plugin - Native Azure Cosmos DB vector search with hierarchical relationships
            vector_search_plugin = VectorSearchPlugin(self.settings)
            await vector_search_plugin.initialize()
            self.kernel.add_plugin(vector_search_plugin, plugin_name="vector_search")
            self.plugins["vector_search"] = vector_search_plugin

            # Retrieval Plugin (FR-5, FR-6, FR-7) - QUERY ONLY with integrated VectorSearchPlugin
            retrieval_plugin = RetrievalPlugin(self.settings)
            await retrieval_plugin.initialize()
            self.kernel.add_plugin(retrieval_plugin, plugin_name="retrieval")
            self.plugins["retrieval"] = retrieval_plugin

            # Refinement Plugin (FR-8)
            refinement_plugin = RefinementPlugin(self.settings)
            await refinement_plugin.initialize()
            self.kernel.add_plugin(refinement_plugin, plugin_name="refinement")
            self.plugins["refinement"] = refinement_plugin

            # Memory Plugin (FR-9, FR-10, FR-11)
            memory_plugin = MemoryPlugin(self.settings)
            await memory_plugin.initialize()
            self.kernel.add_plugin(memory_plugin, plugin_name="memory")
            self.plugins["memory"] = memory_plugin

            # Diagram Plugin (FR-12, FR-13)
            diagram_plugin = DiagramPlugin(self.settings, self.kernel)
            await diagram_plugin.initialize()
            self.kernel.add_plugin(diagram_plugin, plugin_name="diagram")
            self.plugins["diagram"] = diagram_plugin

            logger.info("Mosaic Query Server plugins registered successfully")
            logger.info(
                "VectorSearchPlugin integrated with hierarchical relationships support"
            )

        except Exception as e:
            logger.error(f"Failed to register Mosaic plugins: {e}")
            raise

    async def get_plugin(self, plugin_name: str) -> Any:
        """Get a plugin instance by name."""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found")
        return self.plugins[plugin_name]

    async def get_plugin_status(self) -> Dict[str, Any]:
        """Get status information for all plugins."""
        status = {}

        for name, plugin in self.plugins.items():
            try:
                if hasattr(plugin, "get_status"):
                    status[name] = await plugin.get_status()
                else:
                    status[name] = {"status": "active", "type": type(plugin).__name__}
            except Exception as e:
                status[name] = {"status": "error", "error": str(e)}

        return status

    async def cleanup(self) -> None:
        """Cleanup resources and close connections."""
        try:
            # Cleanup plugins
            for name, plugin in self.plugins.items():
                if hasattr(plugin, "cleanup"):
                    await plugin.cleanup()

            self.plugins.clear()
            self.kernel = None

            logger.info("Semantic Kernel cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
