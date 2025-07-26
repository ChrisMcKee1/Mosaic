"""
Example of invisible learning integration with existing MCP tools.

This shows how the learning system enhances existing tools transparently
without changing their interface or requiring user awareness.
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime, timezone

from mcp.server import Server
from mcp.types import Tool, TextContent

from ..plugins.invisible_learning import (
    invisible_learning_decorator,
    enable_invisible_learning,
    get_learning_diagnostics,
)
from ..plugins.context_aggregator import ContextAggregator


logger = logging.getLogger(__name__)


class EnhancedOmniRAGTools:
    """
    Enhanced OmniRAG tools with invisible learning capabilities.

    These tools work exactly the same as before from a user perspective,
    but they automatically get smarter over time through background learning.
    """

    def __init__(self):
        """Initialize enhanced tools with invisible learning."""
        # Enable invisible learning system
        self.learning_middleware = enable_invisible_learning()

        # Initialize existing tools
        self.context_aggregator = ContextAggregator()

        logger.info("Enhanced OmniRAG tools initialized with invisible learning")

    async def initialize(self) -> None:
        """Initialize all components."""
        await self.context_aggregator.initialize()
        logger.info("Enhanced OmniRAG tools ready")

    @invisible_learning_decorator(enable_invisible_learning())
    async def query_knowledge_base(
        self, query: str, context: Dict[str, Any], user_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Query the knowledge base (enhanced with invisible learning).

        From user perspective: Same as always - just ask questions
        Behind the scenes: Learning improves results over time
        """
        try:
            # Use context aggregator (which gets enhanced by learning)
            aggregated_context = await self.context_aggregator.aggregate_context(
                query=query, context=context
            )

            # Simulate knowledge base lookup
            knowledge_result = {
                "query": query,
                "answer": f"Knowledge base result for: {query}",
                "context": aggregated_context,
                "confidence": 0.8,
                "sources": ["knowledge_base"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # The learning decorator automatically:
            # 1. Tracks this interaction
            # 2. Infers feedback from user behavior
            # 3. Adapts strategies in background
            # 4. Enhances future responses

            return knowledge_result

        except Exception as e:
            logger.error(f"Error in knowledge base query: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @invisible_learning_decorator(enable_invisible_learning())
    async def semantic_search(
        self,
        query: str,
        search_context: Dict[str, Any],
        user_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Perform semantic search (enhanced with invisible learning).

        Users just search - system learns their preferences automatically.
        """
        try:
            # Simulate semantic search
            search_results = {
                "query": query,
                "results": [
                    {
                        "title": f"Result 1 for {query}",
                        "content": f"Semantic content related to {query}",
                        "relevance": 0.9,
                    },
                    {
                        "title": f"Result 2 for {query}",
                        "content": f"Additional information about {query}",
                        "relevance": 0.7,
                    },
                ],
                "search_context": search_context,
                "total_results": 2,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Learning happens automatically via decorator
            return search_results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @invisible_learning_decorator(enable_invisible_learning())
    async def generate_response(
        self,
        query: str,
        generation_context: Dict[str, Any],
        user_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate AI response (enhanced with invisible learning).

        Response quality improves over time based on user interaction patterns.
        """
        try:
            # Simulate AI response generation
            generated_response = {
                "query": query,
                "response": f"AI-generated response for: {query}",
                "context": generation_context,
                "model_confidence": 0.85,
                "response_type": "generated",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Learning decorator handles all the background improvement
            return generated_response

        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


def register_enhanced_omnirag_tools(server: Server) -> None:
    """
    Register enhanced OmniRAG tools with MCP server.

    These are the ONLY tools users see - all learning is invisible.
    """

    # Initialize enhanced tools
    omnirag_tools = EnhancedOmniRAGTools()

    @server.list_tools()
    async def list_omnirag_tools() -> List[Tool]:
        """List available OmniRAG tools (same as before - no learning tools exposed)."""
        return [
            Tool(
                name="query_knowledge_base",
                description="Query the knowledge base for information (automatically improving)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Your question or search query",
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context for the query",
                        },
                        "user_context": {
                            "type": "object",
                            "description": "User context (optional)",
                        },
                    },
                    "required": ["query", "context"],
                },
            ),
            Tool(
                name="semantic_search",
                description="Search for semantically related content (learns your preferences)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "search_context": {
                            "type": "object",
                            "description": "Search parameters",
                        },
                        "user_context": {
                            "type": "object",
                            "description": "User context (optional)",
                        },
                    },
                    "required": ["query", "search_context"],
                },
            ),
            Tool(
                name="generate_response",
                description="Generate AI responses (quality improves over time)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Input for AI generation",
                        },
                        "generation_context": {
                            "type": "object",
                            "description": "Generation parameters",
                        },
                        "user_context": {
                            "type": "object",
                            "description": "User context (optional)",
                        },
                    },
                    "required": ["query", "generation_context"],
                },
            ),
            Tool(
                name="system_diagnostics",
                description="Get system health and learning diagnostics (admin only)",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
        ]

    @server.call_tool()
    async def call_omnirag_tool(name: str, arguments: dict) -> List[TextContent]:
        """
        Handle tool calls with invisible learning enhancement.

        Users call tools normally - learning happens automatically.
        """
        try:
            if name == "query_knowledge_base":
                result = await omnirag_tools.query_knowledge_base(
                    query=arguments["query"],
                    context=arguments["context"],
                    user_context=arguments.get("user_context", {}),
                )

            elif name == "semantic_search":
                result = await omnirag_tools.semantic_search(
                    query=arguments["query"],
                    search_context=arguments["search_context"],
                    user_context=arguments.get("user_context", {}),
                )

            elif name == "generate_response":
                result = await omnirag_tools.generate_response(
                    query=arguments["query"],
                    generation_context=arguments["generation_context"],
                    user_context=arguments.get("user_context", {}),
                )

            elif name == "system_diagnostics":
                # ONLY admin tool - for monitoring invisible learning
                result = await get_learning_diagnostics()

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

            return [TextContent(type="text", text=str(result))]

        except Exception as e:
            logger.error(f"Error in {name}: {e}")
            return [TextContent(type="text", text=f"Error: {e}")]


# Example usage showing the invisible learning in action
async def demo_invisible_learning():
    """
    Demonstrate how invisible learning works without user awareness.
    """
    print("=== OmniRAG Invisible Learning Demo ===\n")

    # Initialize enhanced tools
    omnirag = EnhancedOmniRAGTools()
    await omnirag.initialize()

    # Simulate user interactions (users don't know learning is happening)
    user_context = {
        "user_id": "demo_user",
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0...",
        "client_id": "demo_client",
    }

    print("1. User makes first query (system starts learning silently)...")
    result1 = await omnirag.query_knowledge_base(
        query="What is machine learning?",
        context={"domain": "AI", "complexity": "beginner"},
        user_context=user_context,
    )
    print(f"Response: {result1['answer']}")

    print("\n2. User makes second query (system refines understanding)...")
    result2 = await omnirag.semantic_search(
        query="Deep learning algorithms",
        search_context={"domain": "AI", "detail_level": "advanced"},
        user_context=user_context,
    )
    print(f"Found {result2['total_results']} results")

    print("\n3. User continues interaction (system adapts in background)...")
    result3 = await omnirag.generate_response(
        query="Explain neural networks simply",
        generation_context={"style": "educational", "length": "medium"},
        user_context=user_context,
    )
    print(f"Generated: {result3['response']}")

    # Check learning diagnostics (admin only - users never see this)
    print("\n=== System Learning Diagnostics (Admin View) ===")
    diagnostics = await get_learning_diagnostics()
    print(f"Learning enabled: {diagnostics['learning_enabled']}")
    print(f"Interactions tracked: {diagnostics['tracked_interactions']}")
    print(f"System health: {diagnostics['system_health']}")

    print("\n✨ Users just experienced progressively better responses")
    print("✨ Learning happened completely invisibly")
    print("✨ No session management or adaptation tools needed")
    print("✨ OmniRAG became smarter automatically!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_invisible_learning())
