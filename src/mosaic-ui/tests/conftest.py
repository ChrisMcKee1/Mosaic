"""
Pytest configuration and fixtures for Mosaic UI testing.

Provides comprehensive test fixtures for:
- Streamlit session state mocking
- Graph data simulation
- Azure service mocking
- UI component testing utilities
- OmniRAG plugin testing
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio


@pytest.fixture
def mock_streamlit_state():
    """Mock Streamlit session state for testing."""
    state = MagicMock()

    # Mock entities data
    state.entities = [
        {
            "id": "test_entity_1",
            "name": "TestComponent",
            "category": "server",
            "lines": 150,
            "complexity": 8,
            "description": "Test component for validation",
            "file_path": "src/test/component.py",
        },
        {
            "id": "test_entity_2",
            "name": "MockPlugin",
            "category": "plugin",
            "lines": 200,
            "complexity": 10,
            "description": "Mock plugin for testing",
            "file_path": "src/plugins/mock.py",
        },
        {
            "id": "test_entity_3",
            "name": "UIService",
            "category": "ui",
            "lines": 300,
            "complexity": 12,
            "description": "UI service component",
            "file_path": "src/ui/service.py",
        },
    ]

    # Mock relationships data
    state.relationships = [
        {
            "source": "test_entity_1",
            "target": "test_entity_2",
            "type": "imports",
            "description": "Component imports plugin",
        },
        {
            "source": "test_entity_2",
            "target": "test_entity_3",
            "type": "uses",
            "description": "Plugin uses UI service",
        },
    ]

    state.selected_node = None
    state.chat_history = []
    state.ingestion_status = "Not Started"
    state.mosaic_services = (None, None, None)

    return state


@pytest.fixture
def mock_mosaic_settings():
    """Mock Mosaic settings for Azure integration testing."""
    settings = MagicMock()
    settings.azure_cosmos_endpoint = "https://test.cosmos.azure.com"
    settings.azure_cosmos_database_name = "mosaic_test"
    settings.azure_openai_endpoint = "https://test.openai.azure.com"
    settings.azure_openai_text_embedding_deployment_name = "text-embedding-3-small"
    settings.azure_openai_api_version = "2023-12-01-preview"
    settings.oauth_enabled = True
    settings.debug = True
    return settings


@pytest.fixture
def sample_graph_data():
    """Sample graph data for visualization testing."""
    nodes = [
        {"id": "node1", "name": "Component A", "category": "server", "lines": 100},
        {"id": "node2", "name": "Component B", "category": "plugin", "lines": 200},
        {"id": "node3", "name": "Component C", "category": "ui", "lines": 150},
    ]

    links = [
        {"source": "node1", "target": "node2", "type": "imports"},
        {"source": "node2", "target": "node3", "type": "uses"},
    ]

    return {"nodes": nodes, "links": links}


@pytest.fixture
def mock_cosmos_client():
    """Mock Azure Cosmos DB client for testing."""
    client = MagicMock()
    database = MagicMock()
    container = MagicMock()

    # Mock query responses
    def mock_query_items(query, parameters=None, enable_cross_partition_query=False):
        # Return mock data based on query type
        if "code_entities" in query:
            return [
                {
                    "id": "entity1",
                    "name": "test_function",
                    "entity_type": "function",
                    "content": "def test_function(): pass",
                    "file_path": "test.py",
                    "embedding": [0.1] * 1536,  # Mock embedding
                }
            ]
        elif "code_relationships" in query:
            return [
                {
                    "id": "rel1",
                    "source_name": "component_a",
                    "target_name": "component_b",
                    "relationship_type": "imports",
                    "relationship_strength": 0.8,
                }
            ]
        return []

    container.query_items = mock_query_items
    database.get_container_client.return_value = container
    client.get_database_client.return_value = database

    return client


@pytest.fixture
def mock_embedding_service():
    """Mock Azure OpenAI embedding service for testing."""
    service = AsyncMock()

    # Mock embedding generation
    async def mock_generate_embeddings(texts):
        return [[0.1] * 1536 for _ in texts]  # Mock 1536-dim embeddings

    service.generate_embeddings = mock_generate_embeddings
    return service


@pytest.fixture
def sample_chat_history():
    """Sample chat history for testing chat interface."""
    return [
        ("user", "What AI agents are available?"),
        (
            "assistant",
            "The system includes 5 specialized AI agents: GitSleuth, CodeParser, GraphArchitect, DocuWriter, and GraphAuditor.",
        ),
        ("user", "Show me the ingestion status"),
        (
            "assistant",
            "Ingestion system status: 38 files processed, 696 entities extracted, system ready.",
        ),
    ]


@pytest.fixture
def mock_networkx_graph():
    """Mock NetworkX graph for graph analysis testing."""
    import networkx as nx

    G = nx.DiGraph()
    G.add_node("A", category="server", lines=100)
    G.add_node("B", category="plugin", lines=200)
    G.add_node("C", category="ui", lines=150)
    G.add_edge("A", "B", relationship="imports")
    G.add_edge("B", "C", relationship="uses")

    return G


@pytest.fixture
def mock_plotly_figure():
    """Mock Plotly figure for graph visualization testing."""
    fig = MagicMock()
    fig.to_html.return_value = "<div>Mock Plotly Graph</div>"
    return fig


@pytest.fixture
def mock_pyvis_network():
    """Mock Pyvis network for network visualization testing."""
    network = MagicMock()
    network.generate_html.return_value = "<div>Mock Pyvis Network</div>"
    return network


@pytest.fixture
def sample_omnirag_responses():
    """Sample OmniRAG responses for testing different strategies."""
    return {
        "database_rag": {
            "query": "what is Flask",
            "strategy": "database_rag",
            "results": {
                "method": "database_rag",
                "entities_found": 3,
                "entities": [
                    {"name": "Flask", "type": "class", "language": "python"},
                    {"name": "FlaskApp", "type": "function", "language": "python"},
                ],
            },
        },
        "vector_rag": {
            "query": "find similar components",
            "strategy": "vector_rag",
            "results": {
                "method": "vector_rag",
                "semantic_matches": 5,
                "top_matches": [{"name": "similar_component", "similarity": 0.85}],
            },
        },
        "graph_rag": {
            "query": "dependencies of FastAPI",
            "strategy": "graph_rag",
            "results": {
                "method": "graph_rag",
                "relationships_found": 4,
                "related_entities": ["Pydantic", "Uvicorn", "Starlette"],
            },
        },
    }


@pytest.fixture
def mock_d3_graph_html():
    """Mock D3.js graph HTML for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://d3js.org/d3.v7.min.js"></script>
        <style>
            .node { fill: #4ecdc4; }
            .link { stroke: #999; }
        </style>
    </head>
    <body>
        <svg id="graph" width="700" height="700"></svg>
        <script>
            const nodes = [{"id": "test", "name": "Test Node"}];
            const links = [];
            // Mock D3 visualization code
        </script>
    </body>
    </html>
    """


@pytest.fixture
def mock_ui_metrics():
    """Mock UI performance metrics for testing."""
    return {
        "total_components": 31,
        "total_relationships": 46,
        "categories": 7,
        "total_lines": 11937,
        "render_time_ms": 150,
        "memory_usage_mb": 45,
        "graph_nodes": 31,
        "graph_edges": 46,
    }


@pytest.fixture(autouse=True)
def mock_streamlit_imports():
    """Auto-mock Streamlit imports to prevent import errors during testing."""
    with patch.dict(
        "sys.modules",
        {
            "streamlit": MagicMock(),
            "streamlit.components.v1": MagicMock(),
            "pyvis": MagicMock(),
            "pyvis.network": MagicMock(),
            "plotly.graph_objects": MagicMock(),
            "networkx": MagicMock(),
        },
    ):
        yield


@pytest.fixture
def mock_asyncio_run():
    """Mock asyncio.run for testing async functions in Streamlit."""

    def mock_run(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with patch("asyncio.run", side_effect=mock_run):
        yield


@pytest.fixture
def sample_validation_data():
    """Sample validation data for testing system validation features."""
    return {
        "ingestion_stats": {
            "files_processed": 38,
            "entities_extracted": 696,
            "relationships_mapped": 234,
            "processing_time_seconds": 45.2,
            "success_rate": 0.98,
        },
        "system_health": {
            "database_connected": True,
            "ai_agents_active": 5,
            "memory_usage_mb": 256,
            "response_time_ms": 120,
        },
        "test_results": {
            "unit_tests_passed": 142,
            "integration_tests_passed": 28,
            "coverage_percentage": 85.3,
            "performance_benchmarks_met": True,
        },
    }


class StreamlitTestHelper:
    """Helper class for Streamlit testing utilities."""

    @staticmethod
    def mock_session_state(**kwargs):
        """Create a mock session state with specified attributes."""
        state = MagicMock()
        for key, value in kwargs.items():
            setattr(state, key, value)
        return state

    @staticmethod
    def mock_selectbox_response(value):
        """Mock selectbox return value."""
        return value

    @staticmethod
    def mock_button_click(clicked=True):
        """Mock button click response."""
        return clicked

    @staticmethod
    def mock_text_input(value=""):
        """Mock text input response."""
        return value

    @staticmethod
    def mock_file_uploader(files=None):
        """Mock file uploader response."""
        return files or []


@pytest.fixture
def streamlit_helper():
    """Provide Streamlit testing helper utilities."""
    return StreamlitTestHelper()


@pytest.fixture
def mock_azure_services():
    """Comprehensive mock of all Azure services for integration testing."""
    services = MagicMock()

    # Mock Cosmos DB
    services.cosmos_client = mock_cosmos_client()

    # Mock OpenAI embeddings
    services.embedding_service = mock_embedding_service()

    # Mock authentication
    services.auth_handler = MagicMock()
    services.auth_handler.validate_token.return_value = {"user_id": "test_user"}

    # Mock graph services
    services.graph_service = MagicMock()
    services.graph_service.get_entity_count.return_value = 696
    services.graph_service.get_relationship_count.return_value = 234

    return services


@pytest.fixture
def performance_test_data():
    """Data for performance testing of UI components."""
    # Generate larger dataset for performance testing
    large_entities = [
        {
            "id": f"entity_{i}",
            "name": f"Component_{i}",
            "category": ["server", "plugin", "ui", "model"][i % 4],
            "lines": 100 + (i * 10),
            "complexity": 5 + (i % 20),
            "description": f"Test component {i} for performance testing",
        }
        for i in range(100)
    ]

    large_relationships = [
        {
            "source": f"entity_{i}",
            "target": f"entity_{(i + 1) % 100}",
            "type": ["imports", "uses", "extends", "implements"][i % 4],
        }
        for i in range(150)
    ]

    return {
        "entities": large_entities,
        "relationships": large_relationships,
        "expected_render_time_ms": 500,  # Maximum acceptable render time
        "expected_memory_mb": 100,  # Maximum acceptable memory usage
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "ui: marks tests as UI component tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line(
        "markers", "visualization: marks tests as graph visualization tests"
    )
    config.addinivalue_line("markers", "omnirag: marks tests as OmniRAG plugin tests")


# Test data constants
SAMPLE_ENTITIES_COUNT = 31
SAMPLE_RELATIONSHIPS_COUNT = 46
EXPECTED_CATEGORIES = [
    "server",
    "plugin",
    "ingestion",
    "ai_agent",
    "config",
    "model",
    "ui",
]
SUPPORTED_VISUALIZATIONS = [
    "Enhanced D3.js (OmniRAG-style)",
    "Pyvis Network (Vis.js compatible)",
    "Plotly Graph (Advanced Analytics)",
    "Classic D3.js",
]
