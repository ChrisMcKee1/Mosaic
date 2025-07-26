"""
Unit tests for Natural Language to SPARQL Translation System.
Tests models, translator, service, and API endpoints.
"""

import json
import pytest
from unittest.mock import Mock, patch, AsyncMock

from pydantic import ValidationError

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.sparql_models import (
    SPARQLQuery,
    SPARQLPrefix,
    SPARQLVariable,
    SPARQLTriplePattern,
    SPARQLGraphPattern,
    QueryType,
    CodeEntityType,
    CodeRelationType,
    NL2SPARQLRequest,
    NL2SPARQLResponse,
)
from plugins.nl2sparql_translator import NL2SPARQLTranslator
from plugins.nl2sparql_service import NL2SPARQLService


class TestSPARQLModels:
    """Test SPARQL Pydantic models."""

    def test_sparql_prefix_creation(self):
        """Test SPARQLPrefix model creation and validation."""
        prefix = SPARQLPrefix(prefix="code", uri="http://example.com/code#")
        assert prefix.prefix == "code"
        assert prefix.uri == "http://example.com/code#"

    def test_sparql_prefix_validation(self):
        """Test SPARQLPrefix validation."""
        # Valid prefix
        SPARQLPrefix(prefix="code", uri="http://example.com#")

        # Invalid prefix with special characters
        with pytest.raises(ValidationError):
            SPARQLPrefix(prefix="code!", uri="http://example.com#")

    def test_sparql_variable_creation(self):
        """Test SPARQLVariable creation and string representation."""
        var = SPARQLVariable(name="function")
        assert var.name == "function"
        assert str(var) == "?function"

    def test_sparql_variable_validation(self):
        """Test SPARQLVariable validation."""
        # Valid variable name
        SPARQLVariable(name="valid_name")

        # Invalid variable name
        with pytest.raises(ValidationError):
            SPARQLVariable(name="invalid@name")

    def test_sparql_triple_pattern(self):
        """Test SPARQLTriplePattern creation."""
        pattern = SPARQLTriplePattern(
            subject="?function", predicate="code:calls", object="?target"
        )
        assert str(pattern) == "?function code:calls ?target"

    def test_sparql_query_creation(self):
        """Test complete SPARQLQuery creation."""
        query = SPARQLQuery(
            query_type=QueryType.SELECT,
            prefixes=[SPARQLPrefix(prefix="code", uri="http://example.com#")],
            select_variables=[SPARQLVariable(name="function")],
            where_pattern=SPARQLGraphPattern(
                triple_patterns=[
                    SPARQLTriplePattern(
                        subject="?function",
                        predicate="rdf:type",
                        object="code:Function",
                    )
                ]
            ),
        )

        assert query.query_type == QueryType.SELECT
        assert len(query.prefixes) == 1
        assert len(query.select_variables) == 1

    def test_sparql_query_to_sparql(self):
        """Test SPARQL query string generation."""
        query = SPARQLQuery(
            query_type=QueryType.SELECT,
            prefixes=[SPARQLPrefix(prefix="code", uri="http://example.com#")],
            select_variables=[SPARQLVariable(name="function")],
            where_pattern=SPARQLGraphPattern(
                triple_patterns=[
                    SPARQLTriplePattern(
                        subject="?function",
                        predicate="rdf:type",
                        object="code:Function",
                    )
                ]
            ),
            limit=10,
        )

        sparql_string = query.to_sparql()
        assert "PREFIX code: <http://example.com#>" in sparql_string
        assert "SELECT ?function" in sparql_string
        assert "WHERE {" in sparql_string
        assert "?function rdf:type code:Function ." in sparql_string
        assert "LIMIT 10" in sparql_string

    def test_nl2sparql_request_validation(self):
        """Test NL2SPARQLRequest validation."""
        # Valid request
        request = NL2SPARQLRequest(
            natural_language_query="Find all functions", max_results=50
        )
        assert request.natural_language_query == "Find all functions"
        assert request.max_results == 50

        # Invalid max_results
        with pytest.raises(ValidationError):
            NL2SPARQLRequest(
                natural_language_query="test",
                max_results=0,  # Should be >= 1
            )


class TestNL2SPARQLTranslator:
    """Test NL2SPARQLTranslator class."""

    @pytest.fixture
    def mock_translator(self):
        """Create translator with mocked Azure OpenAI."""
        with patch("openai.AzureOpenAI") as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            translator = NL2SPARQLTranslator()
            translator.client = mock_client
            return translator

    def test_translator_initialization(self, mock_translator):
        """Test translator initialization."""
        assert mock_translator.config.temperature == 0.1
        assert len(mock_translator.templates) > 0
        assert "function_calls" in mock_translator.templates

    def test_template_matching(self, mock_translator):
        """Test template matching logic."""
        # Test function calls template match
        match = mock_translator._match_templates(
            "Which functions call the login method?"
        )
        assert match is not None
        assert match[0] == "function_calls"
        assert match[2] > 0  # confidence score

        # Test inheritance template match
        match = mock_translator._match_templates("What classes inherit from BaseModel?")
        assert match is not None
        assert match[0] == "inheritance_hierarchy"

        # Test no match
        match = mock_translator._match_templates("random unrelated query")
        assert match is None

    def test_entity_detection(self, mock_translator):
        """Test code entity detection."""
        entities = mock_translator._detect_entities("Find all functions and classes")
        assert CodeEntityType.FUNCTION in entities
        assert CodeEntityType.CLASS in entities

        entities = mock_translator._detect_entities("Show me modules")
        assert CodeEntityType.MODULE in entities

    def test_relation_detection(self, mock_translator):
        """Test code relation detection."""
        relations = mock_translator._detect_relations(
            "Which functions call other methods?"
        )
        assert CodeRelationType.CALLS in relations

        relations = mock_translator._detect_relations(
            "Show classes that inherit from base"
        )
        assert CodeRelationType.INHERITS in relations

    @pytest.mark.asyncio
    async def test_translate_query_with_template_match(self, mock_translator):
        """Test query translation with template matching."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.function_call = Mock()
        mock_response.choices[0].message.function_call.name = "generate_sparql_query"
        mock_response.choices[0].message.function_call.arguments = json.dumps(
            {
                "query_type": "SELECT",
                "prefixes": [{"prefix": "code", "uri": "http://example.com#"}],
                "select_variables": [{"name": "function"}],
                "where_pattern": {
                    "triple_patterns": [
                        {
                            "subject": "?function",
                            "predicate": "rdf:type",
                            "object": "code:Function",
                        }
                    ]
                },
            }
        )

        mock_translator.client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = NL2SPARQLRequest(
            natural_language_query="Find all functions that call login", max_results=10
        )

        response = await mock_translator.translate_query(request)

        assert isinstance(response, NL2SPARQLResponse)
        assert response.confidence_score > 0
        assert response.sparql_query.query_type == QueryType.SELECT
        assert len(response.detected_entities) > 0
        assert len(response.detected_relations) > 0

    @pytest.mark.asyncio
    async def test_translate_query_fallback_to_template(self, mock_translator):
        """Test fallback to template when OpenAI fails."""
        # Mock OpenAI to raise exception
        mock_translator.client.chat.completions.create = AsyncMock(
            side_effect=Exception("OpenAI API error")
        )

        request = NL2SPARQLRequest(
            natural_language_query="Which functions call other methods?", max_results=10
        )

        response = await mock_translator.translate_query(request)

        # Should still return a response using template fallback
        assert isinstance(response, NL2SPARQLResponse)
        assert response.sparql_query.query_type == QueryType.SELECT


class TestNL2SPARQLService:
    """Test NL2SPARQLService class."""

    @pytest.fixture
    def mock_service(self):
        """Create service with mocked dependencies."""
        with patch("redis.asyncio.from_url") as mock_redis:
            mock_redis_client = AsyncMock()
            mock_redis.return_value = mock_redis_client

            service = NL2SPARQLService()
            service._redis_client = mock_redis_client

            # Mock translator
            service.translator = Mock()
            service.translator.translate_query = AsyncMock()

            # Mock query executor
            service.query_executor = Mock()
            service.query_executor.execute_query = AsyncMock()
            service.query_executor.validate_query = AsyncMock()

            return service

    @pytest.mark.asyncio
    async def test_translate_only(self, mock_service):
        """Test translation without execution."""
        # Setup mock response
        mock_response = NL2SPARQLResponse(
            sparql_query=SPARQLQuery(
                query_type=QueryType.SELECT,
                select_variables=[SPARQLVariable(name="function")],
                where_pattern=SPARQLGraphPattern(),
            ),
            confidence_score=0.8,
            explanation="Test query",
            detected_entities=[],
            detected_relations=[],
            suggested_alternatives=[],
            validation_errors=[],
        )
        mock_service.translator.translate_query.return_value = mock_response

        result = await mock_service.translate_only("Find all functions")

        assert isinstance(result, NL2SPARQLResponse)
        assert result.confidence_score == 0.8
        mock_service.translator.translate_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_translate_and_execute(self, mock_service):
        """Test translation with query execution."""
        # Setup mock response
        mock_response = NL2SPARQLResponse(
            sparql_query=SPARQLQuery(
                query_type=QueryType.SELECT,
                select_variables=[SPARQLVariable(name="function")],
                where_pattern=SPARQLGraphPattern(),
            ),
            confidence_score=0.8,
            explanation="Test query",
            detected_entities=[],
            detected_relations=[],
            suggested_alternatives=[],
            validation_errors=[],
        )
        mock_service.translator.translate_query.return_value = mock_response
        mock_service.query_executor.execute_query.return_value = [
            {"function": "test_func"}
        ]

        result = await mock_service.translate_and_execute(
            natural_language_query="Find all functions", execute_query=True
        )

        assert result["executed"] is True
        assert result["confidence_score"] == 0.8
        assert len(result["results"]) == 1
        assert result["results"][0]["function"] == "test_func"

    @pytest.mark.asyncio
    async def test_batch_translate(self, mock_service):
        """Test batch translation."""
        # Setup mock response
        mock_response = NL2SPARQLResponse(
            sparql_query=SPARQLQuery(
                query_type=QueryType.SELECT,
                select_variables=[SPARQLVariable(name="function")],
                where_pattern=SPARQLGraphPattern(),
            ),
            confidence_score=0.8,
            explanation="Test query",
            detected_entities=[],
            detected_relations=[],
            suggested_alternatives=[],
            validation_errors=[],
        )
        mock_service.translator.translate_query.return_value = mock_response

        queries = [{"query": "Find all functions"}, {"query": "Show classes"}]

        results = await mock_service.batch_translate(queries, execute_queries=False)

        assert len(results) == 2
        assert all(result["success"] for result in results)

    @pytest.mark.asyncio
    async def test_validate_sparql_query(self, mock_service):
        """Test SPARQL query validation."""
        mock_service.query_executor.validate_query.return_value = (True, [])

        result = await mock_service.validate_sparql_query(
            "SELECT ?s WHERE { ?s ?p ?o }"
        )

        assert result["valid"] is True
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_cache_operations(self, mock_service):
        """Test cache get/set operations."""
        # Test cache miss
        mock_service._redis_client.get.return_value = None
        cached_result = await mock_service._get_from_cache("test_key")
        assert cached_result is None

        # Test cache hit
        cached_data = {"test": "data"}
        mock_service._redis_client.get.return_value = json.dumps(cached_data)
        cached_result = await mock_service._get_from_cache("test_key")
        assert cached_result == cached_data

        # Test cache store
        await mock_service._store_in_cache("test_key", cached_data)
        mock_service._redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self, mock_service):
        """Test service health check."""
        health = await mock_service.health_check()

        assert "service" in health
        assert "status" in health
        assert "components" in health
        assert health["service"] == "NL2SPARQL"


class TestNL2SPARQLRoutes:
    """Test NL2SPARQL API routes."""

    @pytest.fixture
    def mock_service_dependency(self):
        """Mock service dependency for route testing."""
        mock_service = Mock()
        mock_service.translate_only = AsyncMock()
        mock_service.translate_and_execute = AsyncMock()
        mock_service.batch_translate = AsyncMock()
        mock_service.validate_sparql_query = AsyncMock()
        mock_service.get_translation_templates = AsyncMock()
        mock_service.health_check = AsyncMock()
        mock_service.get_cache_stats = AsyncMock()
        mock_service.clear_cache = AsyncMock()
        return mock_service

    def test_translate_request_validation(self):
        """Test translation request validation."""
        from plugins.nl2sparql_routes import TranslateRequest

        # Valid request
        request = TranslateRequest(query="Find all functions")
        assert request.query == "Find all functions"

        # Invalid empty query
        with pytest.raises(ValidationError):
            TranslateRequest(query="")

        # Invalid long query
        with pytest.raises(ValidationError):
            TranslateRequest(query="x" * 2001)

    def test_batch_request_validation(self):
        """Test batch translation request validation."""
        from plugins.nl2sparql_routes import BatchTranslateRequest

        # Valid request
        request = BatchTranslateRequest(
            queries=[{"query": "Find functions"}, {"query": "Show classes"}]
        )
        assert len(request.queries) == 2

        # Invalid empty queries
        with pytest.raises(ValidationError):
            BatchTranslateRequest(queries=[])

        # Invalid query without 'query' field
        with pytest.raises(ValidationError):
            BatchTranslateRequest(queries=[{"invalid": "field"}])

    @pytest.mark.asyncio
    async def test_translate_endpoint(self, mock_service_dependency):
        """Test translation endpoint."""
        from plugins.nl2sparql_routes import translate_query, TranslateRequest

        # Setup mock response
        mock_response = NL2SPARQLResponse(
            sparql_query=SPARQLQuery(
                query_type=QueryType.SELECT,
                select_variables=[SPARQLVariable(name="function")],
                where_pattern=SPARQLGraphPattern(),
            ),
            confidence_score=0.8,
            explanation="Test query",
            detected_entities=[],
            detected_relations=[],
            suggested_alternatives=[],
            validation_errors=[],
        )
        mock_service_dependency.translate_only.return_value = mock_response

        request = TranslateRequest(query="Find all functions")
        result = await translate_query(request, mock_service_dependency)

        assert result["success"] is True
        assert result["confidence_score"] == 0.8
        assert "sparql_query" in result

    @pytest.mark.asyncio
    async def test_translate_and_execute_endpoint(self, mock_service_dependency):
        """Test translate and execute endpoint."""
        from plugins.nl2sparql_routes import (
            translate_and_execute_query,
            TranslateAndExecuteRequest,
        )

        # Setup mock response
        mock_result = {
            "executed": True,
            "confidence_score": 0.8,
            "results": [{"function": "test_func"}],
        }
        mock_service_dependency.translate_and_execute.return_value = mock_result

        request = TranslateAndExecuteRequest(
            query="Find all functions", execute_query=True
        )
        result = await translate_and_execute_query(request, mock_service_dependency)

        assert result["success"] is True
        assert result["executed"] is True
        assert result["confidence_score"] == 0.8


@pytest.mark.integration
class TestNL2SPARQLIntegration:
    """Integration tests for the complete NL2SPARQL system."""

    @pytest.mark.asyncio
    async def test_end_to_end_translation(self):
        """Test end-to-end translation flow."""
        # This would require actual Azure OpenAI and graph database
        # For now, we'll test with mocked components
        with patch("openai.AzureOpenAI"):
            service = NL2SPARQLService()

            # Mock the translator
            mock_response = NL2SPARQLResponse(
                sparql_query=SPARQLQuery(
                    query_type=QueryType.SELECT,
                    select_variables=[SPARQLVariable(name="function")],
                    where_pattern=SPARQLGraphPattern(),
                ),
                confidence_score=0.8,
                explanation="Test query",
                detected_entities=[CodeEntityType.FUNCTION],
                detected_relations=[CodeRelationType.CALLS],
                suggested_alternatives=[],
                validation_errors=[],
            )

            service.translator.translate_query = AsyncMock(return_value=mock_response)
            service.query_executor.execute_query = AsyncMock(return_value=[])

            result = await service.translate_and_execute(
                natural_language_query="Find all functions that call login",
                execute_query=True,
            )

            assert "sparql_query" in result
            assert result["confidence_score"] == 0.8
            assert "Function" in result.get("detected_entities", [])
            assert "calls" in result.get("detected_relations", [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
