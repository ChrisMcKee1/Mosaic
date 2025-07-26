"""Unit tests for SPARQL Query Executor.

Tests comprehensive SPARQL query execution functionality including
validation, caching, result formatting, and error handling.

Task: OMR-P2-001 - Implement SPARQL Query Executor in Query Server
"""

import asyncio
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import redis
from mosaic_mcp.plugins.sparql_query_executor import (
    QueryExecutionResult,
    QueryType,
    ResultFormat,
    SPARQLQueryError,
    SPARQLQueryExecutor,
    SPARQLQueryTimeoutError,
)


# Test fixtures
@pytest.fixture
def mock_cosmos_client():
    """Mock Azure Cosmos DB client."""
    client = MagicMock()
    database = MagicMock()
    container = MagicMock()

    client.get_database_client.return_value = database
    database.get_container_client.return_value = container

    return client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    client = MagicMock(spec=redis.Redis)
    client.ping.return_value = True
    return client


@pytest.fixture
def sample_rdf_data():
    """Sample RDF data for testing."""
    return [
        {
            "id": "1",
            "entity_type": "rdf_triple",
            "subject": "http://example.org/person/john",
            "predicate": "http://xmlns.com/foaf/0.1/name",
            "object": "John Doe",
        },
        {
            "id": "2",
            "entity_type": "rdf_triple",
            "subject": "http://example.org/person/john",
            "predicate": "http://xmlns.com/foaf/0.1/age",
            "object": "30",
        },
        {
            "id": "3",
            "entity_type": "rdf_triple",
            "subject": "http://example.org/person/jane",
            "predicate": "http://xmlns.com/foaf/0.1/name",
            "object": "Jane Smith",
        },
    ]


@pytest.fixture
async def query_executor(mock_cosmos_client, mock_redis_client):
    """Create SPARQL Query Executor instance for testing."""
    return SPARQLQueryExecutor(
        cosmos_client=mock_cosmos_client,
        redis_client=mock_redis_client,
        default_timeout=5.0,
        default_cache_ttl=3600,
        enable_caching=True,
    )


class TestSPARQLQueryExecutor:
    """Test suite for SPARQL Query Executor."""

    def test_initialization(self, mock_cosmos_client, mock_redis_client):
        """Test executor initialization with various configurations."""
        # Test with full configuration
        executor = SPARQLQueryExecutor(
            cosmos_client=mock_cosmos_client,
            redis_client=mock_redis_client,
            default_timeout=10.0,
            default_cache_ttl=1800,
            enable_caching=False,
        )

        assert executor.cosmos_client == mock_cosmos_client
        assert executor.redis_client == mock_redis_client
        assert executor.default_timeout == 10.0
        assert executor.default_cache_ttl == 1800
        assert executor.enable_caching is False

        # Test without Redis client
        executor_no_redis = SPARQLQueryExecutor(
            cosmos_client=mock_cosmos_client, redis_client=None
        )

        assert executor_no_redis.redis_client is None
        assert executor_no_redis.enable_caching is True

    def test_query_validation_select(self, query_executor):
        """Test validation of SELECT queries."""
        # Valid SELECT query
        query = "SELECT ?name WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name }"
        result = query_executor.validate_query(query)

        assert result.valid is True
        assert result.query_type == QueryType.SELECT
        assert result.error_message is None
        assert result.parsed_query is not None

    def test_query_validation_construct(self, query_executor):
        """Test validation of CONSTRUCT queries."""
        query = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"
        result = query_executor.validate_query(query)

        assert result.valid is True
        assert result.query_type == QueryType.CONSTRUCT
        assert result.error_message is None

    def test_query_validation_ask(self, query_executor):
        """Test validation of ASK queries."""
        query = "ASK { ?s ?p ?o }"
        result = query_executor.validate_query(query)

        assert result.valid is True
        assert result.query_type == QueryType.ASK
        assert result.error_message is None

    def test_query_validation_describe(self, query_executor):
        """Test validation of DESCRIBE queries."""
        query = "DESCRIBE <http://example.org/person/john>"
        result = query_executor.validate_query(query)

        assert result.valid is True
        assert result.query_type == QueryType.DESCRIBE
        assert result.error_message is None

    def test_query_validation_invalid_syntax(self, query_executor):
        """Test validation of syntactically invalid queries."""
        query = "INVALID QUERY SYNTAX"
        result = query_executor.validate_query(query)

        assert result.valid is False
        assert result.query_type is None
        assert result.error_message is not None

    def test_query_validation_unsupported_type(self, query_executor):
        """Test validation of unsupported query types."""
        query = "UPDATE { ?s ?p ?o } WHERE { ?s ?p ?old }"
        result = query_executor.validate_query(query)

        assert result.valid is False
        assert result.query_type is None
        assert "Unsupported query type" in result.error_message

    def test_query_optimization(self, query_executor):
        """Test query optimization functionality."""
        # Test basic optimization (whitespace normalization)
        query = "SELECT    ?name   WHERE   {   ?person   <http://xmlns.com/foaf/0.1/name>   ?name   }"
        optimized = query_executor.optimize_query(query)

        # Should normalize whitespace
        assert "    " not in optimized
        assert optimized.count(" ") < query.count(" ")

        # Test caching of prepared queries
        initial_cache_size = len(query_executor._prepared_queries)
        query_executor.optimize_query(query)
        query_executor.optimize_query(query)  # Second call should use cache

        # Should only add one entry to cache
        assert len(query_executor._prepared_queries) == initial_cache_size + 1

    @pytest.mark.asyncio
    async def test_load_rdf_data_success(self, query_executor, sample_rdf_data):
        """Test successful RDF data loading from Cosmos DB."""
        # Mock Cosmos DB responses
        query_executor.cosmos_client.get_database_client.return_value.get_container_client.return_value.query_items.return_value = sample_rdf_data

        with patch(
            "mosaic_mcp.plugins.sparql_query_executor.get_settings"
        ) as mock_settings:
            mock_settings.return_value.cosmos_database_name = "test_db"

            await query_executor._load_rdf_data()

            # Verify graph was populated
            assert len(query_executor.graph) == 3

    @pytest.mark.asyncio
    async def test_load_rdf_data_cosmos_error(self, query_executor):
        """Test RDF data loading with Cosmos DB error."""
        from azure.core.exceptions import CosmosHttpResponseError

        # Mock Cosmos DB error
        query_executor.cosmos_client.get_database_client.return_value.get_container_client.return_value.query_items.side_effect = CosmosHttpResponseError(
            status_code=500, message="Internal server error"
        )

        with patch(
            "mosaic_mcp.plugins.sparql_query_executor.get_settings"
        ) as mock_settings:
            mock_settings.return_value.cosmos_database_name = "test_db"

            with pytest.raises(SPARQLQueryError) as exc_info:
                await query_executor._load_rdf_data()

            assert "Failed to load RDF data" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_query_select_success(self, query_executor, sample_rdf_data):
        """Test successful execution of SELECT query."""
        # Setup RDF data
        query_executor.cosmos_client.get_database_client.return_value.get_container_client.return_value.query_items.return_value = sample_rdf_data

        with patch(
            "mosaic_mcp.plugins.sparql_query_executor.get_settings"
        ) as mock_settings:
            mock_settings.return_value.cosmos_database_name = "test_db"

            query = (
                "SELECT ?name WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name }"
            )

            result = await query_executor.execute_query(
                query=query, format=ResultFormat.JSON, use_cache=False
            )

            assert isinstance(result, QueryExecutionResult)
            assert result.query_type == QueryType.SELECT
            assert result.cached is False
            assert result.format == ResultFormat.JSON
            assert result.execution_time > 0
            assert isinstance(result.results, dict)

    @pytest.mark.asyncio
    async def test_execute_query_ask_success(self, query_executor, sample_rdf_data):
        """Test successful execution of ASK query."""
        # Setup RDF data
        query_executor.cosmos_client.get_database_client.return_value.get_container_client.return_value.query_items.return_value = sample_rdf_data

        with patch(
            "mosaic_mcp.plugins.sparql_query_executor.get_settings"
        ) as mock_settings:
            mock_settings.return_value.cosmos_database_name = "test_db"

            query = "ASK { ?s ?p ?o }"

            result = await query_executor.execute_query(
                query=query, format=ResultFormat.JSON, use_cache=False
            )

            assert result.query_type == QueryType.ASK
            assert isinstance(result.results, dict)
            assert "boolean" in result.results

    @pytest.mark.asyncio
    async def test_execute_query_with_cache_hit(self, query_executor, sample_rdf_data):
        """Test query execution with cache hit."""
        # Setup cache data
        cached_result = {
            "query_type": "SELECT",
            "results": {"head": {"vars": ["name"]}, "results": {"bindings": []}},
            "result_count": 0,
            "timestamp": datetime.now().isoformat(),
        }

        query_executor.redis_client.get.return_value = json.dumps(cached_result)

        query = "SELECT ?name WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name }"

        result = await query_executor.execute_query(
            query=query, format=ResultFormat.JSON, use_cache=True
        )

        assert result.cached is True
        assert result.query_type == QueryType.SELECT

    @pytest.mark.asyncio
    async def test_execute_query_timeout(self, query_executor, sample_rdf_data):
        """Test query execution timeout handling."""
        # Setup RDF data
        query_executor.cosmos_client.get_database_client.return_value.get_container_client.return_value.query_items.return_value = sample_rdf_data

        with patch(
            "mosaic_mcp.plugins.sparql_query_executor.get_settings"
        ) as mock_settings:
            mock_settings.return_value.cosmos_database_name = "test_db"

            # Mock slow query execution
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                query = "SELECT ?name WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name }"

                with pytest.raises(SPARQLQueryTimeoutError) as exc_info:
                    await query_executor.execute_query(
                        query=query, timeout=0.1, use_cache=False
                    )

                assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_query_invalid_query(self, query_executor):
        """Test execution of invalid query."""
        query = "INVALID QUERY SYNTAX"

        with pytest.raises(SPARQLQueryError) as exc_info:
            await query_executor.execute_query(query=query, use_cache=False)

        assert "Invalid query" in str(exc_info.value)

    def test_format_result_select_json(self, query_executor):
        """Test formatting SELECT results as JSON."""
        # Create mock result
        result = MagicMock()
        result.vars = ["name", "age"]

        # Mock row data
        row1 = MagicMock()
        row1.__contains__ = lambda self, key: key in ["name", "age"]
        row1.__getitem__ = lambda self, key: {
            "name": MagicMock(__str__=lambda x: "John Doe"),
            "age": MagicMock(__str__=lambda x: "30"),
        }[key]

        result.__iter__ = lambda self: iter([row1])

        formatted = query_executor._format_result(
            result, ResultFormat.JSON, QueryType.SELECT
        )

        assert isinstance(formatted, dict)
        assert "head" in formatted
        assert "results" in formatted
        assert formatted["head"]["vars"] == ["name", "age"]

    def test_format_result_ask_json(self, query_executor):
        """Test formatting ASK results as JSON."""
        # Mock ASK result (True)
        result = MagicMock()
        result.__bool__ = lambda self: True

        formatted = query_executor._format_result(
            result, ResultFormat.JSON, QueryType.ASK
        )

        assert formatted == {"boolean": True}

    def test_format_result_ask_xml(self, query_executor):
        """Test formatting ASK results as XML."""
        result = MagicMock()
        result.__bool__ = lambda self: False

        formatted = query_executor._format_result(
            result, ResultFormat.XML, QueryType.ASK
        )

        assert "sparql" in formatted
        assert "<boolean>false</boolean>" in formatted

    def test_generate_cache_key(self, query_executor):
        """Test cache key generation."""
        query = "SELECT ?name WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name }"
        bindings = {"person": "http://example.org/john"}
        format = ResultFormat.JSON

        key1 = query_executor._generate_cache_key(query, bindings, format)
        key2 = query_executor._generate_cache_key(query, bindings, format)

        # Same inputs should generate same key
        assert key1 == key2
        assert key1.startswith("sparql_query:")

        # Different inputs should generate different keys
        key3 = query_executor._generate_cache_key(query, None, format)
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_cache_result_success(self, query_executor):
        """Test successful result caching."""
        cache_key = "test_key"
        query_type = QueryType.SELECT
        result = {"test": "data"}
        result_count = 5
        ttl = 3600

        await query_executor._cache_result(
            cache_key, query_type, result, result_count, ttl
        )

        # Verify Redis client was called correctly
        query_executor.redis_client.setex.assert_called_once()
        args = query_executor.redis_client.setex.call_args[0]
        assert args[0] == cache_key
        assert args[1] == ttl

    @pytest.mark.asyncio
    async def test_get_cached_result_success(self, query_executor):
        """Test successful cache retrieval."""
        cache_key = "test_key"
        cached_data = {
            "query_type": "SELECT",
            "results": {"test": "data"},
            "result_count": 5,
            "timestamp": datetime.now().isoformat(),
        }

        query_executor.redis_client.get.return_value = json.dumps(cached_data)

        result = await query_executor._get_cached_result(cache_key)

        assert result == cached_data

    @pytest.mark.asyncio
    async def test_get_cached_result_miss(self, query_executor):
        """Test cache miss scenario."""
        cache_key = "nonexistent_key"
        query_executor.redis_client.get.return_value = None

        result = await query_executor._get_cached_result(cache_key)

        assert result is None

    def test_get_statistics(self, query_executor):
        """Test statistics retrieval."""
        # Update some statistics
        query_executor._execution_stats["total_queries"] = 10
        query_executor._execution_stats["cached_hits"] = 3
        query_executor._execution_stats["total_execution_time"] = 15.0

        stats = query_executor.get_statistics()

        assert stats["total_queries"] == 10
        assert stats["cached_hits"] == 3
        assert stats["cache_hit_rate"] == 30.0
        assert stats["average_execution_time"] == 1.5

    def test_clear_cache(self, query_executor):
        """Test cache clearing functionality."""
        # Add some prepared queries
        query_executor._prepared_queries["test1"] = "prepared_query_1"
        query_executor._prepared_queries["test2"] = "prepared_query_2"

        assert len(query_executor._prepared_queries) == 2

        query_executor.clear_cache()

        assert len(query_executor._prepared_queries) == 0

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, query_executor):
        """Test health check when everything is working."""
        health = await query_executor.health_check()

        assert health["status"] == "healthy"
        assert "timestamp" in health
        assert "graph_size" in health
        assert "query_validation" in health
        assert health["query_validation"] is True
        assert health["cache_status"] == "connected"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, query_executor):
        """Test health check when validation fails."""
        with patch.object(
            query_executor, "validate_query", side_effect=Exception("Validation error")
        ):
            health = await query_executor.health_check()

            assert health["status"] == "unhealthy"
            assert "error" in health

    @pytest.mark.asyncio
    async def test_health_check_no_redis(self, mock_cosmos_client):
        """Test health check without Redis client."""
        executor = SPARQLQueryExecutor(
            cosmos_client=mock_cosmos_client, redis_client=None
        )

        health = await executor.health_check()

        assert health["cache_status"] == "disabled"

    def test_format_select_csv(self, query_executor):
        """Test CSV formatting for SELECT results."""
        # Create mock result
        result = MagicMock()
        result.vars = ["name", "age"]

        row1 = MagicMock()
        row1.__contains__ = lambda self, key: key in ["name", "age"]
        row1.__getitem__ = lambda self, key: {
            "name": MagicMock(__str__=lambda x: "John Doe"),
            "age": MagicMock(__str__=lambda x: "30"),
        }[key]

        result.__iter__ = lambda self: iter([row1])

        csv_result = query_executor._format_select_csv(result)

        assert "name,age" in csv_result
        assert "John Doe,30" in csv_result

    def test_format_select_tsv(self, query_executor):
        """Test TSV formatting for SELECT results."""
        # Create mock result
        result = MagicMock()
        result.vars = ["name", "age"]

        row1 = MagicMock()
        row1.__contains__ = lambda self, key: key in ["name", "age"]
        row1.__getitem__ = lambda self, key: {
            "name": MagicMock(__str__=lambda x: "John Doe"),
            "age": MagicMock(__str__=lambda x: "30"),
        }[key]

        result.__iter__ = lambda self: iter([row1])

        tsv_result = query_executor._format_select_tsv(result)

        assert "name\tage" in tsv_result
        assert "John Doe\t30" in tsv_result


# Integration tests
class TestSPARQLQueryExecutorIntegration:
    """Integration tests for SPARQL Query Executor."""

    @pytest.mark.asyncio
    async def test_end_to_end_query_execution(
        self, mock_cosmos_client, mock_redis_client, sample_rdf_data
    ):
        """Test complete end-to-end query execution workflow."""
        # Setup
        executor = SPARQLQueryExecutor(
            cosmos_client=mock_cosmos_client,
            redis_client=mock_redis_client,
            enable_caching=True,
        )

        # Mock Cosmos DB data
        mock_cosmos_client.get_database_client.return_value.get_container_client.return_value.query_items.return_value = sample_rdf_data

        # Mock cache miss then hit
        mock_redis_client.get.side_effect = [
            None,
            json.dumps(
                {
                    "query_type": "SELECT",
                    "results": {
                        "head": {"vars": ["name"]},
                        "results": {"bindings": []},
                    },
                    "result_count": 2,
                    "timestamp": datetime.now().isoformat(),
                }
            ),
        ]

        with patch(
            "mosaic_mcp.plugins.sparql_query_executor.get_settings"
        ) as mock_settings:
            mock_settings.return_value.cosmos_database_name = "test_db"

            query = (
                "SELECT ?name WHERE { ?person <http://xmlns.com/foaf/0.1/name> ?name }"
            )

            # First execution (cache miss)
            result1 = await executor.execute_query(query, format=ResultFormat.JSON)
            assert result1.cached is False

            # Second execution (cache hit)
            result2 = await executor.execute_query(query, format=ResultFormat.JSON)
            assert result2.cached is True

            # Verify caching was used
            stats = executor.get_statistics()
            assert stats["total_queries"] == 2
            assert stats["cached_hits"] == 1
            assert stats["cache_misses"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
