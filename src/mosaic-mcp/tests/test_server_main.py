"""
Unit tests for MosaicMCPServer main server implementation.

Tests the FastMCP server setup, tool registration, and core MCP protocol
compliance for the Mosaic MCP Tool.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from server.main import MosaicMCPServer, main


class TestMosaicMCPServer:
    """Test cases for MosaicMCPServer class."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        settings.azure_cosmos_endpoint = "https://test.cosmos.azure.com/"
        settings.database_name = "test-database"
        settings.container_name = "test-container"
        settings.azure_redis_endpoint = "redis://test-redis:6379"
        settings.oauth_enabled = True
        settings.oauth_tenant_id = "test-tenant"
        settings.oauth_client_id = "test-client"
        return settings

    @pytest.fixture
    def mock_kernel_manager(self):
        """Create mock Semantic Kernel manager."""
        mock_manager = Mock()
        mock_manager.kernel = Mock()
        mock_manager.initialize = AsyncMock()
        mock_manager.get_plugin = Mock()
        return mock_manager

    @pytest.fixture
    def mock_oauth_handler(self):
        """Create mock OAuth handler."""
        mock_handler = Mock()
        mock_handler.authenticate = AsyncMock(return_value=True)
        mock_handler.get_user_info = AsyncMock(return_value={"sub": "test-user"})
        return mock_handler

    @pytest.fixture
    def mcp_server(self, mock_settings):
        """Create MosaicMCPServer instance with mocked dependencies."""
        with (
            patch("server.main.SemanticKernelManager") as mock_sk_manager,
            patch("server.main.OAuth2Handler") as mock_oauth,
            patch("server.main.FastMCP") as mock_fastmcp,
        ):
            mock_sk_manager.return_value = Mock()
            mock_oauth.return_value = Mock()
            mock_app = Mock()
            mock_fastmcp.return_value = mock_app

            server = MosaicMCPServer(mock_settings)
            server.app = mock_app
            return server

    def test_server_initialization(self, mcp_server, mock_settings):
        """Test MosaicMCPServer initialization."""
        assert mcp_server.settings == mock_settings
        assert mcp_server.kernel_manager is not None
        assert mcp_server.oauth_handler is not None
        assert mcp_server.app is not None

    def test_server_initialization_with_default_settings(self):
        """Test server initialization with default settings."""
        with (
            patch("server.main.MosaicSettings") as mock_settings_class,
            patch("server.main.SemanticKernelManager"),
            patch("server.main.OAuth2Handler"),
            patch("server.main.FastMCP"),
        ):
            mock_settings = Mock()
            mock_settings_class.return_value = mock_settings

            server = MosaicMCPServer()

            mock_settings_class.assert_called_once()
            assert server.settings == mock_settings

    @pytest.mark.asyncio
    async def test_register_tools(self, mcp_server):
        """Test MCP tool registration."""
        # Mock plugin instances
        mock_retrieval = Mock()
        mock_refinement = Mock()
        mock_memory = Mock()
        mock_diagram = Mock()

        with patch.object(
            mcp_server,
            "_initialize_plugins",
            return_value={
                "retrieval": mock_retrieval,
                "refinement": mock_refinement,
                "memory": mock_memory,
                "diagram": mock_diagram,
            },
        ):
            mcp_server._register_tools()

            # Verify app.tool decorator was called for each plugin method
            assert mcp_server.app.tool.call_count >= 4

    @pytest.mark.asyncio
    async def test_initialize_plugins(self, mcp_server):
        """Test plugin initialization."""
        with (
            patch("server.main.RetrievalPlugin") as mock_retrieval_class,
            patch("server.main.RefinementPlugin") as mock_refinement_class,
            patch("server.main.MemoryPlugin") as mock_memory_class,
            patch("server.main.DiagramPlugin") as mock_diagram_class,
        ):
            # Setup mock plugin instances
            mock_retrieval = Mock()
            mock_refinement = Mock()
            mock_memory = Mock()
            mock_diagram = Mock()

            mock_retrieval_class.return_value = mock_retrieval
            mock_refinement_class.return_value = mock_refinement
            mock_memory_class.return_value = mock_memory
            mock_diagram_class.return_value = mock_diagram

            plugins = mcp_server._initialize_plugins()

            # Verify all plugins were created with settings
            mock_retrieval_class.assert_called_once_with(mcp_server.settings)
            mock_refinement_class.assert_called_once_with(mcp_server.settings)
            mock_memory_class.assert_called_once_with(mcp_server.settings)
            mock_diagram_class.assert_called_once_with(mcp_server.settings)

            # Verify returned plugin dictionary
            assert plugins["retrieval"] == mock_retrieval
            assert plugins["refinement"] == mock_refinement
            assert plugins["memory"] == mock_memory
            assert plugins["diagram"] == mock_diagram

    @pytest.mark.asyncio
    async def test_hybrid_search_tool(self, mcp_server):
        """Test hybrid_search MCP tool."""
        mock_retrieval = Mock()
        mock_retrieval.hybrid_search = AsyncMock(
            return_value=[
                {"id": "doc1", "content": "Test document", "score": 0.95},
                {"id": "doc2", "content": "Another document", "score": 0.80},
            ]
        )

        mcp_server.plugins = {"retrieval": mock_retrieval}

        result = await mcp_server.hybrid_search("test query")

        mock_retrieval.hybrid_search.assert_called_once_with("test query")
        assert len(result) == 2
        assert result[0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_query_code_graph_tool(self, mcp_server):
        """Test query_code_graph MCP tool."""
        mock_retrieval = Mock()
        mock_retrieval.query_code_graph = AsyncMock(
            return_value=[
                {"id": "lib1", "name": "requests", "type": "library"},
                {"id": "lib2", "name": "flask", "type": "framework"},
            ]
        )

        mcp_server.plugins = {"retrieval": mock_retrieval}

        result = await mcp_server.query_code_graph("python", "dependencies")

        mock_retrieval.query_code_graph.assert_called_once_with(
            "python", "dependencies"
        )
        assert len(result) == 2
        assert result[0]["name"] == "requests"

    @pytest.mark.asyncio
    async def test_rerank_tool(self, mcp_server):
        """Test rerank MCP tool."""
        mock_refinement = Mock()
        mock_refinement.rerank = AsyncMock(
            return_value=[
                {"id": "doc1", "content": "Relevant doc", "score": 0.98},
                {"id": "doc2", "content": "Less relevant", "score": 0.75},
            ]
        )

        mcp_server.plugins = {"refinement": mock_refinement}

        documents = [
            {"id": "doc1", "content": "Relevant doc"},
            {"id": "doc2", "content": "Less relevant"},
        ]

        result = await mcp_server.rerank("search query", documents)

        mock_refinement.rerank.assert_called_once_with("search query", documents)
        assert result[0]["score"] == 0.98

    @pytest.mark.asyncio
    async def test_save_memory_tool(self, mcp_server):
        """Test save_memory MCP tool."""
        mock_memory = Mock()
        mock_memory.save = AsyncMock(
            return_value={"status": "saved", "id": "memory_123"}
        )

        mcp_server.plugins = {"memory": mock_memory}

        result = await mcp_server.save_memory(
            "session_1", "Important context", "episodic"
        )

        mock_memory.save.assert_called_once_with(
            "session_1", "Important context", "episodic"
        )
        assert result["status"] == "saved"

    @pytest.mark.asyncio
    async def test_retrieve_memory_tool(self, mcp_server):
        """Test retrieve_memory MCP tool."""
        mock_memory = Mock()
        mock_memory.retrieve = AsyncMock(
            return_value=[
                {"id": "mem1", "content": "Previous context", "score": 0.92},
                {"id": "mem2", "content": "Related memory", "score": 0.87},
            ]
        )

        mcp_server.plugins = {"memory": mock_memory}

        result = await mcp_server.retrieve_memory("session_1", "context query", 10)

        mock_memory.retrieve.assert_called_once_with("session_1", "context query", 10)
        assert len(result) == 2
        assert result[0]["score"] == 0.92

    @pytest.mark.asyncio
    async def test_generate_diagram_tool(self, mcp_server):
        """Test generate_diagram MCP tool."""
        mock_diagram = Mock()
        mock_diagram.generate = AsyncMock(
            return_value={
                "mermaid": "graph TD\n    A --> B\n    B --> C",
                "description": "Simple flow diagram",
            }
        )

        mcp_server.plugins = {"diagram": mock_diagram}

        result = await mcp_server.generate_diagram("Create a flow diagram")

        mock_diagram.generate.assert_called_once_with("Create a flow diagram")
        assert "mermaid" in result
        assert "graph TD" in result["mermaid"]

    @pytest.mark.asyncio
    async def test_start_server(self, mcp_server):
        """Test server startup."""
        with (
            patch.object(mcp_server.kernel_manager, "initialize") as mock_init,
            patch.object(mcp_server.app, "run") as mock_run,
        ):
            await mcp_server.start()

            mock_init.assert_called_once()
            mock_run.assert_called_once_with(
                host="0.0.0.0", port=8000, transport="stdio"
            )

    @pytest.mark.asyncio
    async def test_start_server_with_custom_config(self, mock_settings):
        """Test server startup with custom configuration."""
        mock_settings.server_host = "127.0.0.1"
        mock_settings.server_port = 8080
        mock_settings.mcp_transport = "http"

        with (
            patch("server.main.SemanticKernelManager"),
            patch("server.main.OAuth2Handler"),
            patch("server.main.FastMCP") as mock_fastmcp,
        ):
            mock_app = Mock()
            mock_fastmcp.return_value = mock_app

            server = MosaicMCPServer(mock_settings)

            with (
                patch.object(server.kernel_manager, "initialize"),
                patch.object(mock_app, "run") as mock_run,
            ):
                await server.start()

                mock_run.assert_called_once_with(
                    host="127.0.0.1", port=8080, transport="http"
                )

    @pytest.mark.asyncio
    async def test_shutdown_server(self, mcp_server):
        """Test server shutdown."""
        with patch.object(mcp_server.kernel_manager, "cleanup") as mock_cleanup:
            await mcp_server.shutdown()
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_authentication_middleware(self, mcp_server):
        """Test OAuth authentication middleware."""
        mock_request = Mock()
        mock_request.headers = {"Authorization": "Bearer test_token"}

        with patch.object(mcp_server.oauth_handler, "authenticate") as mock_auth:
            mock_auth.return_value = True

            # Test authentication decorator
            @mcp_server._require_auth
            async def protected_endpoint():
                return {"data": "protected"}

            result = await protected_endpoint()
            assert result["data"] == "protected"

    @pytest.mark.asyncio
    async def test_authentication_failure(self, mcp_server):
        """Test authentication failure handling."""
        with patch.object(mcp_server.oauth_handler, "authenticate") as mock_auth:
            mock_auth.return_value = False

            @mcp_server._require_auth
            async def protected_endpoint():
                return {"data": "protected"}

            with pytest.raises(Exception):  # Should raise authentication error
                await protected_endpoint()

    @pytest.mark.asyncio
    async def test_error_handling_in_tools(self, mcp_server):
        """Test error handling in MCP tools."""
        mock_retrieval = Mock()
        mock_retrieval.hybrid_search = AsyncMock(side_effect=Exception("Search failed"))

        mcp_server.plugins = {"retrieval": mock_retrieval}

        with pytest.raises(Exception, match="Search failed"):
            await mcp_server.hybrid_search("test query")

    def test_health_check_endpoint(self, mcp_server):
        """Test health check endpoint."""
        with patch.object(
            mcp_server,
            "_check_service_health",
            return_value={
                "status": "healthy",
                "services": {
                    "cosmos_db": "connected",
                    "redis": "connected",
                    "semantic_kernel": "initialized",
                },
            },
        ) as mock_health:
            result = mcp_server.health_check()

            mock_health.assert_called_once()
            assert result["status"] == "healthy"
            assert "services" in result

    @pytest.mark.asyncio
    async def test_mcp_resource_discovery(self, mcp_server):
        """Test MCP resource discovery."""
        with patch.object(
            mcp_server.app,
            "list_resources",
            return_value=[
                {"uri": "mosaic://documents", "type": "collection"},
                {"uri": "mosaic://memory", "type": "collection"},
                {"uri": "mosaic://diagrams", "type": "collection"},
            ],
        ) as mock_list:
            resources = mcp_server.list_resources()

            assert len(resources) == 3
            assert any(r["uri"] == "mosaic://documents" for r in resources)

    @pytest.mark.asyncio
    async def test_mcp_capability_negotiation(self, mcp_server):
        """Test MCP capability negotiation."""
        capabilities = mcp_server.get_capabilities()

        assert "tools" in capabilities
        assert "resources" in capabilities
        assert "experimental" in capabilities

        # Check specific tool capabilities
        tool_names = [tool["name"] for tool in capabilities["tools"]]
        assert "hybrid_search" in tool_names
        assert "query_code_graph" in tool_names
        assert "rerank" in tool_names
        assert "save_memory" in tool_names
        assert "retrieve_memory" in tool_names
        assert "generate_diagram" in tool_names


class TestMainFunction:
    """Test cases for main entry point function."""

    @pytest.mark.asyncio
    async def test_main_function_success(self):
        """Test successful main function execution."""
        with (
            patch("server.main.MosaicSettings") as mock_settings_class,
            patch("server.main.MosaicMCPServer") as mock_server_class,
        ):
            mock_settings = Mock()
            mock_settings_class.return_value = mock_settings

            mock_server = AsyncMock()
            mock_server_class.return_value = mock_server

            await main()

            mock_settings_class.assert_called_once()
            mock_server_class.assert_called_once_with(mock_settings)
            mock_server.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_function_with_keyboard_interrupt(self):
        """Test main function handling keyboard interrupt."""
        with (
            patch("server.main.MosaicSettings"),
            patch("server.main.MosaicMCPServer") as mock_server_class,
        ):
            mock_server = AsyncMock()
            mock_server.start.side_effect = KeyboardInterrupt()
            mock_server_class.return_value = mock_server

            # Should handle gracefully without raising
            await main()

            mock_server.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_function_with_exception(self):
        """Test main function handling unexpected exceptions."""
        with (
            patch("server.main.MosaicSettings"),
            patch("server.main.MosaicMCPServer") as mock_server_class,
            patch("server.main.logger") as mock_logger,
        ):
            mock_server = AsyncMock()
            mock_server.start.side_effect = Exception("Startup failed")
            mock_server_class.return_value = mock_server

            with pytest.raises(Exception, match="Startup failed"):
                await main()

            mock_logger.error.assert_called()
            mock_server.shutdown.assert_called_once()


class TestMosaicMCPServerIntegration:
    """Integration tests for MosaicMCPServer with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_complete_mcp_workflow(self):
        """Test complete MCP workflow from tool call to response."""
        mock_settings = Mock()
        mock_settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        mock_settings.azure_cosmos_endpoint = "https://test.cosmos.azure.com/"

        with (
            patch("server.main.SemanticKernelManager"),
            patch("server.main.OAuth2Handler"),
            patch("server.main.FastMCP"),
            patch("server.main.RetrievalPlugin") as mock_retrieval_class,
            patch("server.main.MemoryPlugin") as mock_memory_class,
        ):
            # Setup mock plugins
            mock_retrieval = Mock()
            mock_retrieval.hybrid_search = AsyncMock(
                return_value=[{"id": "doc1", "content": "Test result", "score": 0.95}]
            )
            mock_retrieval_class.return_value = mock_retrieval

            mock_memory = Mock()
            mock_memory.save = AsyncMock(return_value={"status": "saved"})
            mock_memory_class.return_value = mock_memory

            server = MosaicMCPServer(mock_settings)

            # Test search workflow
            search_result = await server.hybrid_search("AI context management")
            assert len(search_result) == 1
            assert search_result[0]["score"] == 0.95

            # Test memory workflow
            memory_result = await server.save_memory(
                "test_session", "Context saved", "episodic"
            )
            assert memory_result["status"] == "saved"

    @pytest.mark.asyncio
    async def test_mcp_error_recovery(self):
        """Test MCP server error recovery mechanisms."""
        mock_settings = Mock()

        with (
            patch("server.main.SemanticKernelManager"),
            patch("server.main.OAuth2Handler"),
            patch("server.main.FastMCP"),
            patch("server.main.RetrievalPlugin") as mock_retrieval_class,
        ):
            # Setup plugin that fails then recovers
            mock_retrieval = Mock()
            mock_retrieval.hybrid_search = AsyncMock(
                side_effect=[
                    Exception("Service temporarily unavailable"),
                    [{"id": "doc1", "content": "Recovered result"}],
                ]
            )
            mock_retrieval_class.return_value = mock_retrieval

            server = MosaicMCPServer(mock_settings)

            # First call should fail
            with pytest.raises(Exception):
                await server.hybrid_search("test query")

            # Second call should succeed (simulating service recovery)
            result = await server.hybrid_search("test query")
            assert len(result) == 1
            assert result[0]["content"] == "Recovered result"

    @pytest.mark.asyncio
    async def test_concurrent_mcp_requests(self):
        """Test handling concurrent MCP requests."""
        mock_settings = Mock()

        with (
            patch("server.main.SemanticKernelManager"),
            patch("server.main.OAuth2Handler"),
            patch("server.main.FastMCP"),
            patch("server.main.RetrievalPlugin") as mock_retrieval_class,
        ):
            # Setup plugin with realistic delays
            mock_retrieval = Mock()

            async def mock_search(query):
                await asyncio.sleep(0.1)  # Simulate processing time
                return [{"id": f"doc_{hash(query)}", "content": f"Result for {query}"}]

            mock_retrieval.hybrid_search = mock_search
            mock_retrieval_class.return_value = mock_retrieval

            server = MosaicMCPServer(mock_settings)

            # Execute concurrent requests
            tasks = [
                server.hybrid_search("query 1"),
                server.hybrid_search("query 2"),
                server.hybrid_search("query 3"),
            ]

            results = await asyncio.gather(*tasks)

            # Verify all requests completed successfully
            assert len(results) == 3
            for i, result in enumerate(results, 1):
                assert len(result) == 1
                assert f"query {i}" in result[0]["content"]
