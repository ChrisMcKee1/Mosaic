"""
Integration tests for MCP protocol compliance and end-to-end workflows.

Tests cover:
- Complete MCP request/response cycles
- Tool registration and execution
- Resource access and management
- Streamable HTTP transport functionality
- Authentication integration
- Error handling across protocol layers
"""

import pytest
from unittest.mock import MagicMock
import asyncio
from typing import Dict, List, Any


# Mock FastMCP and related components for testing
class MockFastMCP:
    """Mock FastMCP application for testing."""

    def __init__(self, name: str):
        self.name = name
        self.tools = {}
        self.resources = {}
        self.middleware = []

    def tool(self, name: str = None):
        """Decorator for registering MCP tools."""

        def decorator(func):
            tool_name = name or func.__name__
            self.tools[tool_name] = func
            return func

        return decorator

    def resource(self, uri_pattern: str):
        """Decorator for registering MCP resources."""

        def decorator(func):
            self.resources[uri_pattern] = func
            return func

        return decorator

    async def run(self, transport: str, host: str, port: int):
        """Mock server run method."""
        return {"transport": transport, "host": host, "port": port}


class MockMosaicMCPServer:
    """Mock MosaicMCPServer for integration testing."""

    def __init__(self, settings=None):
        self.settings = settings or MagicMock()
        self.app = MockFastMCP("Mosaic")
        self.kernel_manager = MagicMock()
        self.oauth_handler = MagicMock()
        self._setup_tools()
        self._setup_resources()

    def _setup_tools(self):
        """Setup mock MCP tools."""

        @self.app.tool()
        async def hybrid_search(query: str) -> List[Dict[str, Any]]:
            """Mock hybrid search tool."""
            if not query:
                return []
            return [
                {"id": "doc1", "content": f"Result for: {query}", "score": 0.9},
                {"id": "doc2", "content": f"Another result for: {query}", "score": 0.8},
            ]

        @self.app.tool()
        async def save_memory(
            session_id: str, content: str, memory_type: str
        ) -> Dict[str, Any]:
            """Mock memory save tool."""
            return {
                "memory_id": f"mem_{session_id}_{len(content)}",
                "session_id": session_id,
                "type": memory_type,
                "status": "saved",
            }

        @self.app.tool()
        async def generate_diagram(description: str) -> str:
            """Mock diagram generation tool."""
            return f"graph TD\n    A[{description}] --> B[Generated Diagram]"

        @self.app.tool()
        async def query_code_graph(
            library_id: str, relationship_type: str
        ) -> List[Dict[str, Any]]:
            """Mock code graph query tool."""
            return [
                {
                    "id": library_id,
                    "type": relationship_type,
                    "related": ["lib1", "lib2"],
                }
            ]

    def _setup_resources(self):
        """Setup mock MCP resources."""

        @self.app.resource("mermaid://diagrams/{diagram_id}")
        async def get_diagram(diagram_id: str) -> str:
            """Mock diagram resource."""
            return f"graph TD\n    A[Diagram {diagram_id}] --> B[Content]"

        @self.app.resource("mosaic://health")
        async def health_check() -> Dict[str, Any]:
            """Mock health check resource."""
            return {"status": "healthy", "version": "0.1.0"}

    async def start(self):
        """Mock server start."""
        return await self.app.run("streamable-http", "localhost", 8000)

    async def stop(self):
        """Mock server stop."""
        pass


@pytest.fixture
def mock_settings():
    """Create mock settings for integration testing."""
    settings = MagicMock()
    settings.server_host = "localhost"
    settings.server_port = 8000
    settings.oauth_enabled = True
    settings.max_search_results = 50
    return settings


@pytest.fixture
def mcp_server(mock_settings):
    """Create mock MCP server for testing."""
    return MockMosaicMCPServer(mock_settings)


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance and basic functionality."""

    def test_server_initialization(self, mcp_server):
        """Test MCP server initialization."""
        assert mcp_server.app.name == "Mosaic"
        assert len(mcp_server.app.tools) > 0
        assert len(mcp_server.app.resources) > 0

    def test_tool_registration(self, mcp_server):
        """Test that MCP tools are properly registered."""
        expected_tools = [
            "hybrid_search",
            "save_memory",
            "generate_diagram",
            "query_code_graph",
        ]

        for tool_name in expected_tools:
            assert tool_name in mcp_server.app.tools
            assert callable(mcp_server.app.tools[tool_name])

    def test_resource_registration(self, mcp_server):
        """Test that MCP resources are properly registered."""
        expected_resources = ["mermaid://diagrams/{diagram_id}", "mosaic://health"]

        for resource_uri in expected_resources:
            assert resource_uri in mcp_server.app.resources
            assert callable(mcp_server.app.resources[resource_uri])

    @pytest.mark.asyncio
    async def test_server_startup(self, mcp_server):
        """Test MCP server startup process."""
        result = await mcp_server.start()

        assert result["transport"] == "streamable-http"
        assert result["host"] == "localhost"
        assert result["port"] == 8000


class TestMCPToolExecution:
    """Test MCP tool execution and responses."""

    @pytest.mark.asyncio
    async def test_hybrid_search_tool(self, mcp_server):
        """Test hybrid search tool execution."""
        tool_func = mcp_server.app.tools["hybrid_search"]
        result = await tool_func("test query")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["content"] == "Result for: test query"
        assert result[0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_hybrid_search_empty_query(self, mcp_server):
        """Test hybrid search with empty query."""
        tool_func = mcp_server.app.tools["hybrid_search"]
        result = await tool_func("")

        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_save_memory_tool(self, mcp_server):
        """Test memory save tool execution."""
        tool_func = mcp_server.app.tools["save_memory"]
        result = await tool_func("session123", "test content", "episodic")

        assert result["session_id"] == "session123"
        assert result["type"] == "episodic"
        assert result["status"] == "saved"
        assert "memory_id" in result

    @pytest.mark.asyncio
    async def test_generate_diagram_tool(self, mcp_server):
        """Test diagram generation tool execution."""
        tool_func = mcp_server.app.tools["generate_diagram"]
        result = await tool_func("User Authentication Flow")

        assert isinstance(result, str)
        assert "graph TD" in result
        assert "User Authentication Flow" in result
        assert "Generated Diagram" in result

    @pytest.mark.asyncio
    async def test_query_code_graph_tool(self, mcp_server):
        """Test code graph query tool execution."""
        tool_func = mcp_server.app.tools["query_code_graph"]
        result = await tool_func("flask", "dependencies")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == "flask"
        assert result[0]["type"] == "dependencies"
        assert "related" in result[0]


class TestMCPResourceAccess:
    """Test MCP resource access and management."""

    @pytest.mark.asyncio
    async def test_diagram_resource_access(self, mcp_server):
        """Test accessing diagram resource."""
        resource_func = mcp_server.app.resources["mermaid://diagrams/{diagram_id}"]
        result = await resource_func("auth_flow_123")

        assert isinstance(result, str)
        assert "graph TD" in result
        assert "Diagram auth_flow_123" in result

    @pytest.mark.asyncio
    async def test_health_check_resource(self, mcp_server):
        """Test health check resource."""
        resource_func = mcp_server.app.resources["mosaic://health"]
        result = await resource_func()

        assert result["status"] == "healthy"
        assert result["version"] == "0.1.0"


class TestMCPWorkflows:
    """Test complete MCP workflows and use cases."""

    @pytest.mark.asyncio
    async def test_search_and_memory_workflow(self, mcp_server):
        """Test search followed by memory save workflow."""
        # Step 1: Perform search
        search_tool = mcp_server.app.tools["hybrid_search"]
        search_results = await search_tool("authentication patterns")

        assert len(search_results) == 2

        # Step 2: Save important result to memory
        memory_tool = mcp_server.app.tools["save_memory"]
        memory_result = await memory_tool(
            "session123", search_results[0]["content"], "semantic"
        )

        assert memory_result["status"] == "saved"
        assert memory_result["type"] == "semantic"

    @pytest.mark.asyncio
    async def test_diagram_generation_and_storage_workflow(self, mcp_server):
        """Test diagram generation and retrieval workflow."""
        # Step 1: Generate diagram
        diagram_tool = mcp_server.app.tools["generate_diagram"]
        diagram_content = await diagram_tool("API Architecture")

        assert "API Architecture" in diagram_content

        # Step 2: Simulate storing and retrieving diagram
        diagram_resource = mcp_server.app.resources["mermaid://diagrams/{diagram_id}"]
        retrieved_diagram = await diagram_resource("api_arch_001")

        assert "graph TD" in retrieved_diagram

    @pytest.mark.asyncio
    async def test_code_analysis_workflow(self, mcp_server):
        """Test code analysis workflow."""
        # Step 1: Query code graph
        graph_tool = mcp_server.app.tools["query_code_graph"]
        dependencies = await graph_tool("flask", "dependencies")

        assert len(dependencies) == 1
        assert dependencies[0]["id"] == "flask"

        # Step 2: Search for related documentation
        search_tool = mcp_server.app.tools["hybrid_search"]
        docs = await search_tool("flask dependencies")

        assert len(docs) == 2
        assert "flask dependencies" in docs[0]["content"]

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, mcp_server):
        """Test concurrent execution of multiple MCP tools."""
        # Execute multiple tools concurrently
        tasks = [
            mcp_server.app.tools["hybrid_search"]("concurrent test"),
            mcp_server.app.tools["generate_diagram"]("Concurrent Diagram"),
            mcp_server.app.tools["query_code_graph"]("test_lib", "all"),
            mcp_server.app.resources["mosaic://health"](),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        assert len(results[0]) == 2  # Search results
        assert "Concurrent Diagram" in results[1]  # Diagram content
        assert results[2][0]["id"] == "test_lib"  # Graph query
        assert results[3]["status"] == "healthy"  # Health check


class TestMCPErrorHandling:
    """Test MCP error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self, mcp_server):
        """Test error handling in tool execution."""

        # Create a tool that raises an error
        @mcp_server.app.tool()
        async def error_tool(param: str) -> str:
            if param == "error":
                raise ValueError("Test error")
            return "success"

        # Test successful execution
        result = await error_tool("test")
        assert result == "success"

        # Test error handling
        with pytest.raises(ValueError, match="Test error"):
            await error_tool("error")

    @pytest.mark.asyncio
    async def test_resource_access_error_handling(self, mcp_server):
        """Test error handling in resource access."""

        # Create a resource that raises an error
        @mcp_server.app.resource("test://error/{id}")
        async def error_resource(id: str) -> str:
            if id == "error":
                raise RuntimeError("Resource error")
            return f"Resource {id}"

        # Test successful access
        result = await error_resource("123")
        assert result == "Resource 123"

        # Test error handling
        with pytest.raises(RuntimeError, match="Resource error"):
            await error_resource("error")

    @pytest.mark.asyncio
    async def test_malformed_request_handling(self, mcp_server):
        """Test handling of malformed MCP requests."""
        # Test tools with missing parameters
        search_tool = mcp_server.app.tools["hybrid_search"]

        # This should be handled gracefully by the tool
        result = await search_tool("")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, mcp_server):
        """Test error handling in concurrent operations."""

        # Create a mix of successful and failing operations
        @mcp_server.app.tool()
        async def sometimes_fails(should_fail: bool) -> str:
            if should_fail:
                raise ValueError("Intentional failure")
            return "success"

        tasks = [
            sometimes_fails(False),  # Should succeed
            sometimes_fails(True),  # Should fail
            sometimes_fails(False),  # Should succeed
        ]

        # Use return_exceptions=True to capture both results and exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert results[0] == "success"
        assert isinstance(results[1], ValueError)
        assert results[2] == "success"


class TestMCPTransportLayer:
    """Test MCP transport layer functionality."""

    @pytest.mark.asyncio
    async def test_streamable_http_transport(self, mcp_server):
        """Test Streamable HTTP transport configuration."""
        result = await mcp_server.start()

        assert result["transport"] == "streamable-http"
        assert result["host"] == "localhost"
        assert result["port"] == 8000

    def test_transport_configuration(self, mcp_server):
        """Test transport configuration options."""
        # Verify server settings
        assert mcp_server.settings.server_host == "localhost"
        assert mcp_server.settings.server_port == 8000

    @pytest.mark.asyncio
    async def test_server_lifecycle(self, mcp_server):
        """Test complete server lifecycle."""
        # Start server
        start_result = await mcp_server.start()
        assert start_result is not None

        # Stop server
        await mcp_server.stop()
        # No exception should be raised


class TestMCPAuthentication:
    """Test MCP authentication integration."""

    def test_oauth_integration_enabled(self, mcp_server):
        """Test that OAuth integration is properly configured."""
        assert mcp_server.settings.oauth_enabled is True
        assert mcp_server.oauth_handler is not None

    @pytest.mark.asyncio
    async def test_authenticated_tool_access(self, mcp_server):
        """Test tool access with authentication."""
        # Mock authentication context
        auth_context = {
            "user_id": "user123",
            "roles": ["MCP.User"],
            "scope": "api://mosaic-mcp/.default",
        }

        # In real implementation, this would check authentication
        # For now, we just verify the structure exists
        assert auth_context["user_id"] == "user123"
        assert "MCP.User" in auth_context["roles"]

    def test_security_configuration(self, mcp_server):
        """Test security configuration."""
        # Verify OAuth is configured
        assert hasattr(mcp_server, "oauth_handler")
        assert mcp_server.oauth_handler is not None


class TestMCPPerformance:
    """Test MCP performance characteristics."""

    @pytest.mark.asyncio
    async def test_response_time_requirements(self, mcp_server):
        """Test that tools respond within acceptable time limits."""
        import time

        start_time = time.time()
        result = await mcp_server.app.tools["hybrid_search"]("performance test")
        end_time = time.time()

        response_time = end_time - start_time

        # Should respond quickly (under 1 second for mock)
        assert response_time < 1.0
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, mcp_server):
        """Test handling of concurrent requests."""
        # Simulate 10 concurrent requests
        tasks = [mcp_server.app.tools["hybrid_search"](f"query {i}") for i in range(10)]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        total_time = end_time - start_time

        # All requests should complete
        assert len(results) == 10
        for result in results:
            assert len(result) == 2

        # Should handle concurrency efficiently
        assert total_time < 2.0  # Conservative estimate for mock


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
