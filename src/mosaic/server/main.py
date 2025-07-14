"""
Mosaic MCP Tool - FastMCP Server Implementation

Main entry point for the Mosaic MCP server implementing FR-1 (MCP Server Implementation)
and FR-3 (Streamable HTTP Communication) using the FastMCP framework.

This server provides standardized MCP-compliant tools and resources for advanced
context engineering, including retrieval, refinement, memory, and diagram generation.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from pydantic import BaseModel

from ..config.settings import MosaicSettings
from ..models.base import Document, LibraryNode, MemoryEntry
from .kernel import SemanticKernelManager
from .auth import OAuth2Handler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MosaicMCPServer:
    """
    Main Mosaic MCP Server class implementing FastMCP framework.
    
    Provides MCP-compliant server with Streamable HTTP transport for:
    - Hybrid search and code graph analysis (RetrievalPlugin)
    - Semantic reranking (RefinementPlugin)  
    - Multi-layered memory management (MemoryPlugin)
    - Mermaid diagram generation (DiagramPlugin)
    """
    
    def __init__(self, settings: Optional[MosaicSettings] = None):
        """Initialize the Mosaic MCP Server."""
        self.settings = settings or MosaicSettings()
        self.kernel_manager = SemanticKernelManager(self.settings)
        self.oauth_handler = OAuth2Handler(self.settings)
        self.app = FastMCP("Mosaic")
        
        # Initialize MCP tools and resources
        self._register_tools()
        self._register_resources()
        
        logger.info("Mosaic MCP Server initialized")
    
    def _register_tools(self) -> None:
        """Register MCP tools implementing the required interface functions."""
        
        # Retrieval Plugin Tools (FR-5, FR-6, FR-7)
        @self.app.tool()
        async def hybrid_search(query: str) -> List[Document]:
            """
            Perform hybrid search using unified Cosmos DB backend (OmniRAG pattern).
            
            Args:
                query: Search query string
                
            Returns:
                List of relevant documents from vector and keyword search
            """
            retrieval_plugin = await self.kernel_manager.get_plugin("retrieval")
            return await retrieval_plugin.hybrid_search(query)
        
        @self.app.tool()
        async def query_code_graph(library_id: str, relationship_type: str) -> List[LibraryNode]:
            """
            Query code graph relationships using OmniRAG embedded JSON pattern.
            
            Args:
                library_id: Identifier for the code library
                relationship_type: Type of relationship to query
                
            Returns:
                List of related library nodes
            """
            retrieval_plugin = await self.kernel_manager.get_plugin("retrieval")
            return await retrieval_plugin.query_code_graph(library_id, relationship_type)
        
        # Refinement Plugin Tools (FR-8)
        @self.app.tool()
        async def rerank(query: str, documents: List[Document]) -> List[Document]:
            """
            Rerank documents using cross-encoder/ms-marco-MiniLM-L-12-v2 model.
            
            Args:
                query: Original search query
                documents: List of candidate documents to rerank
                
            Returns:
                Reranked list of documents
            """
            refinement_plugin = await self.kernel_manager.get_plugin("refinement")
            return await refinement_plugin.rerank(query, documents)
        
        # Memory Plugin Tools (FR-9, FR-10, FR-11)
        @self.app.tool()
        async def save_memory(session_id: str, content: str, memory_type: str) -> Dict[str, Any]:
            """
            Save memory using multi-layered storage (Redis + Cosmos DB).
            
            Args:
                session_id: Session identifier
                content: Memory content to save
                memory_type: Type of memory (episodic, semantic, procedural)
                
            Returns:
                Confirmation with memory ID
            """
            memory_plugin = await self.kernel_manager.get_plugin("memory")
            return await memory_plugin.save(session_id, content, memory_type)
        
        @self.app.tool()
        async def retrieve_memory(
            session_id: str, 
            query: str, 
            limit: int = 10
        ) -> List[MemoryEntry]:
            """
            Retrieve relevant memories using hybrid search.
            
            Args:
                session_id: Session identifier
                query: Memory retrieval query
                limit: Maximum number of memories to return
                
            Returns:
                List of relevant memory entries
            """
            memory_plugin = await self.kernel_manager.get_plugin("memory")
            return await memory_plugin.retrieve(session_id, query, limit)
        
        @self.app.tool()
        async def clear_memory(session_id: str) -> Dict[str, Any]:
            """
            Clear all memories for a session.
            
            Args:
                session_id: Session identifier
                
            Returns:
                Confirmation message
            """
            memory_plugin = await self.kernel_manager.get_plugin("memory")
            return await memory_plugin.clear(session_id)
        
        # Diagram Plugin Tools (FR-12, FR-13)
        @self.app.tool()
        async def generate_diagram(description: str) -> str:
            """
            Generate Mermaid diagram syntax from natural language description.
            
            Args:
                description: Natural language description of the diagram
                
            Returns:
                Mermaid diagram syntax
            """
            diagram_plugin = await self.kernel_manager.get_plugin("diagram")
            return await diagram_plugin.generate(description)
    
    def _register_resources(self) -> None:
        """Register MCP resources for stored diagrams and documentation."""
        
        @self.app.resource("mermaid://diagrams/{diagram_id}")
        async def get_diagram(diagram_id: str) -> str:
            """Retrieve stored Mermaid diagram by ID."""
            diagram_plugin = await self.kernel_manager.get_plugin("diagram")
            return await diagram_plugin.get_stored_diagram(diagram_id)
        
        @self.app.resource("mosaic://health")
        async def health_check() -> Dict[str, Any]:
            """Health check endpoint for monitoring."""
            return {
                "status": "healthy",
                "version": "0.1.0",
                "plugins": await self.kernel_manager.get_plugin_status()
            }
    
    async def start(self) -> None:
        """Start the Mosaic MCP Server."""
        try:
            # Initialize Semantic Kernel and plugins
            await self.kernel_manager.initialize()
            
            # Start OAuth2 handler if enabled
            if self.settings.oauth_enabled:
                await self.oauth_handler.initialize()
            
            logger.info(f"Starting Mosaic MCP Server on port {self.settings.server_port}")
            
            # Start the FastMCP server with Streamable HTTP transport
            await self.app.run(
                transport="streamable-http",
                host=self.settings.server_host,
                port=self.settings.server_port
            )
            
        except Exception as e:
            logger.error(f"Failed to start Mosaic MCP Server: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the Mosaic MCP Server and cleanup resources."""
        try:
            await self.kernel_manager.cleanup()
            if self.settings.oauth_enabled:
                await self.oauth_handler.cleanup()
            logger.info("Mosaic MCP Server stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping server: {e}")


async def main() -> None:
    """Main entry point for the Mosaic MCP Server."""
    try:
        # Load settings from environment
        settings = MosaicSettings()
        
        # Create and start server
        server = MosaicMCPServer(settings)
        await server.start()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        if 'server' in locals():
            await server.stop()


if __name__ == "__main__":
    asyncio.run(main())