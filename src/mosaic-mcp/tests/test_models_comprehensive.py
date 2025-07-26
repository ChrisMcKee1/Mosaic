"""
Unit tests for Mosaic MCP models and data structures.

Tests all Pydantic models used throughout the MCP server including
base models, specialized models, and validation logic.
"""

import pytest
from datetime import datetime, timezone

from pydantic import ValidationError

from models.base import Document, LibraryNode, MemoryEntry
from models.session_models import SessionData, ConversationTurn, UserContext
from models.intent_models import QueryIntent, IntentClassification, IntentConfidence
from models.sparql_models import SPARQLQuery, SPARQLResult, QueryPattern
from models.aggregation_models import (
    AggregationRule,
    AggregationResult,
    ContextAggregation,
)
from models.optimization_models import (
    PerformanceMetrics,
    CacheEntry,
    OptimizationSettings,
)


class TestBaseModels:
    """Test cases for base models."""

    def test_document_model_creation(self):
        """Test Document model creation and validation."""
        doc_data = {
            "id": "doc_123",
            "content": "This is a test document about machine learning algorithms",
            "metadata": {
                "source": "research_paper",
                "author": "Dr. Smith",
                "publication_date": "2024-01-15",
            },
            "score": 0.92,
            "embedding": [0.1, 0.2, -0.1, 0.5, 0.3],
        }

        document = Document(**doc_data)

        assert document.id == "doc_123"
        assert document.content == doc_data["content"]
        assert document.score == 0.92
        assert len(document.embedding) == 5
        assert document.metadata["author"] == "Dr. Smith"

    def test_document_model_validation(self):
        """Test Document model validation errors."""
        # Test invalid score (must be between 0 and 1)
        with pytest.raises(ValidationError):
            Document(
                id="doc_1",
                content="Test content",
                score=1.5,  # Invalid: > 1.0
            )

        # Test negative score
        with pytest.raises(ValidationError):
            Document(
                id="doc_2",
                content="Test content",
                score=-0.1,  # Invalid: < 0.0
            )

    def test_document_serialization(self):
        """Test Document JSON serialization."""
        document = Document(
            id="doc_456",
            content="Test document content",
            score=0.85,
            metadata={"type": "article"},
        )

        json_data = document.model_dump()

        assert json_data["id"] == "doc_456"
        assert json_data["score"] == 0.85
        assert "metadata" in json_data

    def test_library_node_model(self):
        """Test LibraryNode model creation."""
        node_data = {
            "id": "lib_requests",
            "name": "requests",
            "type": "library",
            "version": "2.31.0",
            "dependencies": ["urllib3>=1.21.1", "certifi>=2017.4.17"],
            "metadata": {
                "language": "python",
                "repository": "https://github.com/psf/requests",
                "description": "Python HTTP library",
            },
        }

        library_node = LibraryNode(**node_data)

        assert library_node.id == "lib_requests"
        assert library_node.name == "requests"
        assert library_node.type == "library"
        assert library_node.version == "2.31.0"
        assert len(library_node.dependencies) == 2
        assert library_node.metadata["language"] == "python"

    def test_memory_entry_model(self):
        """Test MemoryEntry model creation."""
        memory_data = {
            "id": "mem_789",
            "session_id": "session_abc123",
            "content": "User prefers TypeScript over JavaScript for large projects",
            "type": "preference",
            "importance_score": 0.85,
            "timestamp": datetime.now(timezone.utc),
            "metadata": {
                "category": "programming_language",
                "confidence": 0.9,
                "source": "conversation",
            },
        }

        memory_entry = MemoryEntry(**memory_data)

        assert memory_entry.id == "mem_789"
        assert memory_entry.session_id == "session_abc123"
        assert memory_entry.type == "preference"
        assert memory_entry.importance_score == 0.85
        assert isinstance(memory_entry.timestamp, datetime)

    def test_memory_entry_validation(self):
        """Test MemoryEntry validation."""
        # Test invalid importance score
        with pytest.raises(ValidationError):
            MemoryEntry(
                id="mem_1",
                session_id="session_1",
                content="Test memory",
                type="episodic",
                importance_score=2.0,  # Invalid: > 1.0
            )


class TestSessionModels:
    """Test cases for session management models."""

    def test_conversation_turn_model(self):
        """Test ConversationTurn model."""
        turn_data = {
            "id": "turn_123",
            "user_message": "How do I implement async functions in Python?",
            "assistant_response": "To implement async functions in Python, use the 'async def' keyword...",
            "timestamp": datetime.now(timezone.utc),
            "context_used": ["doc_1", "doc_2"],
            "tools_called": ["hybrid_search", "code_examples"],
            "metadata": {
                "response_time_ms": 1500,
                "tokens_used": 256,
                "satisfaction_score": 0.95,
            },
        }

        turn = ConversationTurn(**turn_data)

        assert turn.id == "turn_123"
        assert "async functions" in turn.user_message
        assert "async def" in turn.assistant_response
        assert len(turn.context_used) == 2
        assert len(turn.tools_called) == 2

    def test_user_context_model(self):
        """Test UserContext model."""
        context_data = {
            "user_id": "user_456",
            "session_id": "session_789",
            "preferences": {
                "programming_languages": ["Python", "TypeScript"],
                "frameworks": ["FastAPI", "React"],
                "coding_style": "functional",
            },
            "expertise_level": "intermediate",
            "current_project": {
                "name": "Web API Development",
                "technologies": ["Python", "FastAPI", "PostgreSQL"],
                "phase": "development",
            },
            "interaction_history": ["search", "code_generation", "explanation"],
        }

        user_context = UserContext(**context_data)

        assert user_context.user_id == "user_456"
        assert user_context.expertise_level == "intermediate"
        assert "Python" in user_context.preferences["programming_languages"]
        assert user_context.current_project["name"] == "Web API Development"

    def test_session_data_model(self):
        """Test SessionData model."""
        session_data = {
            "session_id": "session_abc123",
            "user_context": {
                "user_id": "user_456",
                "session_id": "session_abc123",
                "expertise_level": "advanced",
                "preferences": {},
            },
            "conversation_turns": [
                {
                    "id": "turn_1",
                    "user_message": "Hello",
                    "assistant_response": "Hi! How can I help?",
                    "timestamp": datetime.now(timezone.utc),
                }
            ],
            "active_tools": ["hybrid_search", "memory"],
            "session_metadata": {
                "start_time": datetime.now(timezone.utc),
                "total_turns": 1,
                "total_tokens": 50,
            },
        }

        session = SessionData(**session_data)

        assert session.session_id == "session_abc123"
        assert session.user_context.user_id == "user_456"
        assert len(session.conversation_turns) == 1
        assert len(session.active_tools) == 2


class TestIntentModels:
    """Test cases for intent classification models."""

    def test_query_intent_model(self):
        """Test QueryIntent model."""
        intent_data = {
            "intent_type": "code_search",
            "confidence": 0.92,
            "entities": {
                "programming_language": "Python",
                "topic": "async programming",
                "difficulty": "intermediate",
            },
            "suggested_tools": ["hybrid_search", "code_examples"],
            "metadata": {
                "classifier_model": "intent-classifier-v2",
                "processing_time_ms": 45,
            },
        }

        intent = QueryIntent(**intent_data)

        assert intent.intent_type == "code_search"
        assert intent.confidence == 0.92
        assert intent.entities["programming_language"] == "Python"
        assert "hybrid_search" in intent.suggested_tools

    def test_intent_classification_model(self):
        """Test IntentClassification model."""
        classification_data = {
            "query": "How to handle exceptions in async Python functions?",
            "primary_intent": {
                "intent_type": "how_to_guide",
                "confidence": 0.88,
                "entities": {"language": "Python", "topic": "exception_handling"},
            },
            "secondary_intents": [
                {
                    "intent_type": "code_search",
                    "confidence": 0.72,
                    "entities": {"language": "Python"},
                }
            ],
            "classification_metadata": {
                "model_version": "v1.2.0",
                "timestamp": datetime.now(timezone.utc),
            },
        }

        classification = IntentClassification(**classification_data)

        assert "async Python" in classification.query
        assert classification.primary_intent.intent_type == "how_to_guide"
        assert len(classification.secondary_intents) == 1
        assert classification.secondary_intents[0].confidence == 0.72

    def test_intent_confidence_model(self):
        """Test IntentConfidence model."""
        confidence_data = {
            "overall_confidence": 0.87,
            "intent_scores": {
                "code_search": 0.92,
                "how_to_guide": 0.88,
                "troubleshooting": 0.65,
                "explanation": 0.45,
            },
            "threshold_met": True,
            "confidence_factors": {
                "query_clarity": 0.9,
                "entity_extraction": 0.85,
                "context_match": 0.8,
            },
        }

        confidence = IntentConfidence(**confidence_data)

        assert confidence.overall_confidence == 0.87
        assert confidence.threshold_met is True
        assert confidence.intent_scores["code_search"] == 0.92
        assert len(confidence.confidence_factors) == 3


class TestSPARQLModels:
    """Test cases for SPARQL query models."""

    def test_sparql_query_model(self):
        """Test SPARQLQuery model."""
        query_data = {
            "query_id": "sparql_123",
            "sparql_text": """
                PREFIX code: <http://mosaic.local/code/>
                SELECT ?function ?name WHERE {
                    ?function rdf:type code:Function .
                    ?function code:name ?name .
                }
            """,
            "query_type": "SELECT",
            "parameters": {"module_name": "analytics", "language": "python"},
            "optimization_hints": ["use_index", "limit_results"],
            "metadata": {"created_by": "nl2sparql_translator", "complexity": "medium"},
        }

        sparql_query = SPARQLQuery(**query_data)

        assert sparql_query.query_id == "sparql_123"
        assert "SELECT ?function" in sparql_query.sparql_text
        assert sparql_query.query_type == "SELECT"
        assert sparql_query.parameters["language"] == "python"

    def test_sparql_result_model(self):
        """Test SPARQLResult model."""
        result_data = {
            "query_id": "sparql_123",
            "status": "success",
            "results": [
                {
                    "function": "http://mosaic.local/code/calculate_metrics",
                    "name": "calculate_metrics",
                },
                {
                    "function": "http://mosaic.local/code/process_data",
                    "name": "process_data",
                },
            ],
            "execution_time_ms": 125,
            "result_count": 2,
            "metadata": {"graph_size": 10000, "cache_hit": False},
        }

        sparql_result = SPARQLResult(**result_data)

        assert sparql_result.query_id == "sparql_123"
        assert sparql_result.status == "success"
        assert len(sparql_result.results) == 2
        assert sparql_result.execution_time_ms == 125
        assert sparql_result.result_count == 2

    def test_query_pattern_model(self):
        """Test QueryPattern model."""
        pattern_data = {
            "pattern_id": "pattern_func_search",
            "pattern_name": "Find Functions in Module",
            "sparql_template": """
                PREFIX code: <http://mosaic.local/code/>
                SELECT ?function ?name WHERE {
                    ?function rdf:type code:Function .
                    ?function code:name ?name .
                    ?function code:inModule ?module .
                    FILTER(?module = <{module_uri}>)
                }
            """,
            "parameters": ["module_uri"],
            "description": "Finds all functions in a specific module",
            "usage_count": 45,
            "success_rate": 0.96,
        }

        pattern = QueryPattern(**pattern_data)

        assert pattern.pattern_id == "pattern_func_search"
        assert "module_uri" in pattern.parameters
        assert pattern.usage_count == 45
        assert pattern.success_rate == 0.96


class TestAggregationModels:
    """Test cases for context aggregation models."""

    def test_aggregation_rule_model(self):
        """Test AggregationRule model."""
        rule_data = {
            "rule_id": "rule_code_context",
            "rule_name": "Code Context Aggregation",
            "conditions": {
                "intent_type": "code_search",
                "has_programming_language": True,
                "confidence_threshold": 0.8,
            },
            "aggregation_strategy": "weighted_merge",
            "weight_factors": {
                "recency": 0.3,
                "relevance": 0.5,
                "user_preference": 0.2,
            },
            "max_results": 10,
            "enabled": True,
        }

        rule = AggregationRule(**rule_data)

        assert rule.rule_id == "rule_code_context"
        assert rule.aggregation_strategy == "weighted_merge"
        assert rule.weight_factors["relevance"] == 0.5
        assert rule.max_results == 10

    def test_aggregation_result_model(self):
        """Test AggregationResult model."""
        result_data = {
            "aggregation_id": "agg_123",
            "rule_applied": "rule_code_context",
            "input_sources": ["retrieval", "memory", "code_graph"],
            "aggregated_items": [
                {
                    "id": "item_1",
                    "content": "Python async function example",
                    "source": "retrieval",
                    "weight": 0.9,
                },
                {
                    "id": "item_2",
                    "content": "User's previous async question",
                    "source": "memory",
                    "weight": 0.7,
                },
            ],
            "final_score": 0.85,
            "processing_time_ms": 89,
            "metadata": {
                "total_candidates": 25,
                "filtered_candidates": 15,
                "final_selection": 2,
            },
        }

        result = AggregationResult(**result_data)

        assert result.aggregation_id == "agg_123"
        assert len(result.input_sources) == 3
        assert len(result.aggregated_items) == 2
        assert result.final_score == 0.85

    def test_context_aggregation_model(self):
        """Test ContextAggregation model."""
        aggregation_data = {
            "session_id": "session_456",
            "query": "How to handle async exceptions in Python?",
            "aggregation_results": [
                {
                    "aggregation_id": "agg_1",
                    "rule_applied": "rule_code_context",
                    "input_sources": ["retrieval"],
                    "aggregated_items": [],
                    "final_score": 0.9,
                    "processing_time_ms": 45,
                }
            ],
            "final_context": {
                "documents": ["doc_1", "doc_2"],
                "memories": ["mem_3"],
                "code_examples": ["example_1"],
            },
            "aggregation_metadata": {
                "total_processing_time_ms": 156,
                "rules_evaluated": 3,
                "rules_applied": 1,
            },
        }

        context_agg = ContextAggregation(**aggregation_data)

        assert context_agg.session_id == "session_456"
        assert "async exceptions" in context_agg.query
        assert len(context_agg.aggregation_results) == 1
        assert len(context_agg.final_context["documents"]) == 2


class TestOptimizationModels:
    """Test cases for performance optimization models."""

    def test_performance_metrics_model(self):
        """Test PerformanceMetrics model."""
        metrics_data = {
            "operation_id": "op_search_123",
            "operation_type": "hybrid_search",
            "start_time": datetime.now(timezone.utc),
            "end_time": datetime.now(timezone.utc),
            "duration_ms": 234,
            "memory_usage_mb": 45.7,
            "cpu_usage_percent": 12.5,
            "cache_hit_rate": 0.67,
            "success": True,
            "error_message": None,
            "metadata": {
                "query_complexity": "medium",
                "result_count": 15,
                "user_id": "user_123",
            },
        }

        metrics = PerformanceMetrics(**metrics_data)

        assert metrics.operation_id == "op_search_123"
        assert metrics.operation_type == "hybrid_search"
        assert metrics.duration_ms == 234
        assert metrics.success is True
        assert metrics.cache_hit_rate == 0.67

    def test_cache_entry_model(self):
        """Test CacheEntry model."""
        cache_data = {
            "key": "search_ai_context_management",
            "value": {
                "results": [
                    {"id": "doc_1", "content": "AI context systems"},
                    {"id": "doc_2", "content": "Context management"},
                ]
            },
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc),
            "access_count": 7,
            "last_accessed": datetime.now(timezone.utc),
            "size_bytes": 2048,
            "metadata": {
                "cache_type": "search_results",
                "compression": "gzip",
                "hit_ratio": 0.85,
            },
        }

        cache_entry = CacheEntry(**cache_data)

        assert cache_entry.key == "search_ai_context_management"
        assert len(cache_entry.value["results"]) == 2
        assert cache_entry.access_count == 7
        assert cache_entry.size_bytes == 2048

    def test_optimization_settings_model(self):
        """Test OptimizationSettings model."""
        settings_data = {
            "cache_enabled": True,
            "cache_ttl_seconds": 3600,
            "max_cache_size_mb": 500,
            "enable_query_optimization": True,
            "max_concurrent_requests": 100,
            "request_timeout_seconds": 30,
            "memory_threshold_mb": 1000,
            "performance_monitoring": True,
            "optimization_rules": {
                "prefer_cache": True,
                "batch_requests": True,
                "parallel_processing": True,
            },
        }

        settings = OptimizationSettings(**settings_data)

        assert settings.cache_enabled is True
        assert settings.cache_ttl_seconds == 3600
        assert settings.max_concurrent_requests == 100
        assert settings.optimization_rules["prefer_cache"] is True


class TestModelValidation:
    """Test cases for model validation and error handling."""

    def test_model_field_validation(self):
        """Test comprehensive field validation across models."""
        # Test Document with invalid data
        with pytest.raises(ValidationError) as exc_info:
            Document(
                id="",  # Empty ID should fail
                content="Test content",
                score=0.5,
            )

        assert "ensure this value has at least 1 characters" in str(exc_info.value)

    def test_model_serialization_deserialization(self):
        """Test model serialization and deserialization."""
        # Create a complex memory entry
        original = MemoryEntry(
            id="mem_test",
            session_id="session_test",
            content="Test memory content",
            type="episodic",
            importance_score=0.75,
            timestamp=datetime.now(timezone.utc),
            metadata={"test": True},
        )

        # Serialize to JSON
        json_data = original.model_dump()

        # Deserialize back
        restored = MemoryEntry(**json_data)

        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.importance_score == original.importance_score

    def test_model_defaults(self):
        """Test model default values."""
        # Document with minimal data
        doc = Document(id="test_doc", content="Test content")

        # Should have default values
        assert doc.score is None  # Optional field
        assert doc.metadata == {}  # Default empty dict
        assert doc.embedding is None  # Optional field

    def test_nested_model_validation(self):
        """Test validation of nested models."""
        # SessionData with invalid user context
        with pytest.raises(ValidationError):
            SessionData(
                session_id="test_session",
                user_context={
                    "user_id": "",  # Invalid empty user_id
                    "session_id": "test_session",
                    "expertise_level": "beginner",
                },
                conversation_turns=[],
                active_tools=[],
            )

    def test_model_update_functionality(self):
        """Test model update and modification."""
        # Create initial document
        doc = Document(id="doc_update_test", content="Original content", score=0.8)

        # Update with new data
        updated_data = doc.model_dump()
        updated_data["score"] = 0.9
        updated_data["metadata"] = {"updated": True}

        updated_doc = Document(**updated_data)

        assert updated_doc.score == 0.9
        assert updated_doc.metadata["updated"] is True
        assert updated_doc.content == "Original content"  # Unchanged
