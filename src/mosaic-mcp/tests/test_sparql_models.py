"""
Simplified unit tests for NL2SPARQL models only.
Tests Pydantic models without external dependencies.
"""

import pytest

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

    def test_nl2sparql_response_creation(self):
        """Test NL2SPARQLResponse creation."""
        # Create a complete response
        query = SPARQLQuery(
            query_type=QueryType.SELECT,
            select_variables=[SPARQLVariable(name="function")],
            where_pattern=SPARQLGraphPattern(),
        )

        response = NL2SPARQLResponse(
            sparql_query=query,
            confidence_score=0.8,
            explanation="Test query for functions",
            detected_entities=[CodeEntityType.FUNCTION],
            detected_relations=[CodeRelationType.CALLS],
            suggested_alternatives=["Alternative query 1"],
            validation_errors=[],
        )

        assert response.confidence_score == 0.8
        assert response.explanation == "Test query for functions"
        assert CodeEntityType.FUNCTION in response.detected_entities
        assert CodeRelationType.CALLS in response.detected_relations
        assert len(response.suggested_alternatives) == 1
        assert len(response.validation_errors) == 0

    def test_code_entity_types(self):
        """Test CodeEntityType enum values."""
        assert CodeEntityType.FUNCTION == "Function"
        assert CodeEntityType.CLASS == "Class"
        assert CodeEntityType.MODULE == "Module"
        assert CodeEntityType.VARIABLE == "Variable"
        assert CodeEntityType.INTERFACE == "Interface"
        assert CodeEntityType.PACKAGE == "Package"

    def test_code_relation_types(self):
        """Test CodeRelationType enum values."""
        assert CodeRelationType.CALLS == "calls"
        assert CodeRelationType.INHERITS == "inherits"
        assert CodeRelationType.IMPLEMENTS == "implements"
        assert CodeRelationType.IMPORTS == "imports"
        assert CodeRelationType.EXTENDS == "extends"
        assert CodeRelationType.USES == "uses"
        assert CodeRelationType.REFERENCES == "references"
        assert CodeRelationType.DEPENDS_ON == "dependsOn"
        assert CodeRelationType.CONTAINS == "contains"

    def test_query_types(self):
        """Test QueryType enum values."""
        assert QueryType.SELECT == "SELECT"
        assert QueryType.CONSTRUCT == "CONSTRUCT"
        assert QueryType.ASK == "ASK"
        assert QueryType.DESCRIBE == "DESCRIBE"

    def test_complex_sparql_query(self):
        """Test complex SPARQL query generation."""
        query = SPARQLQuery(
            query_type=QueryType.SELECT,
            prefixes=[
                SPARQLPrefix(prefix="code", uri="http://example.com/code#"),
                SPARQLPrefix(
                    prefix="rdfs", uri="http://www.w3.org/2000/01/rdf-schema#"
                ),
            ],
            select_variables=[
                SPARQLVariable(name="function"),
                SPARQLVariable(name="target"),
                SPARQLVariable(name="label"),
            ],
            where_pattern=SPARQLGraphPattern(
                triple_patterns=[
                    SPARQLTriplePattern(
                        subject="?function",
                        predicate="rdf:type",
                        object="code:Function",
                    ),
                    SPARQLTriplePattern(
                        subject="?function", predicate="code:calls", object="?target"
                    ),
                    SPARQLTriplePattern(
                        subject="?function", predicate="rdfs:label", object="?label"
                    ),
                ],
                filters=['FILTER(REGEX(?label, "login", "i"))'],
            ),
            order_by=["?label"],
            limit=20,
        )

        sparql_string = query.to_sparql()

        # Check all components are present
        assert "PREFIX code: <http://example.com/code#>" in sparql_string
        assert "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>" in sparql_string
        assert "SELECT ?function ?target ?label" in sparql_string
        assert "?function rdf:type code:Function ." in sparql_string
        assert "?function code:calls ?target ." in sparql_string
        assert "?function rdfs:label ?label ." in sparql_string
        assert 'FILTER(REGEX(?label, "login", "i"))' in sparql_string
        assert "ORDER BY ?label" in sparql_string
        assert "LIMIT 20" in sparql_string


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
