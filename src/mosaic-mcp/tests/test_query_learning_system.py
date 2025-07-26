"""
Comprehensive test suite for OMR-P3-004 Query Learning System.

Tests cover session management, incremental learning, Redis integration,
and MCP protocol compliance.
"""

import asyncio
import pytest
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.session_models import (
    QuerySession,
    QueryInteraction,
    FeedbackType,
    SessionStatus,
    ModelState,
)
from utils.redis_session import SessionManager, RedisSessionError
from plugins.incremental_learning import (
    QueryClassifierLearner,
    PreferenceRegressorLearner,
    LearningOrchestrator,
)
from plugins.query_learning_system import QueryLearningSystem


@pytest.fixture
def sample_session():
    """Create a sample session for testing."""
    return QuerySession(
        session_id=uuid4(), user_id="test_user_001", status=SessionStatus.ACTIVE
    )


@pytest.fixture
def sample_interaction():
    """Create a sample query interaction."""
    return QueryInteraction(
        query="What is the capital of France?",
        query_intent="factual_question",
        context={"domain": "geography", "complexity": "low"},
        response="The capital of France is Paris.",
        feedback_type=FeedbackType.POSITIVE,
    )


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.setex.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.smembers.return_value = set()
    mock_redis.sadd.return_value = 1
    mock_redis.srem.return_value = 1
    mock_redis.expire.return_value = True
    mock_redis.ttl.return_value = 3600
    mock_redis.scan_iter.return_value = []
    return mock_redis


class TestSessionModels:
    """Test session data models."""

    def test_query_interaction_creation(self):
        """Test creating query interaction."""
        interaction = QueryInteraction(query="Test query", context={"test": "value"})

        assert interaction.query == "Test query"
        assert interaction.context == {"test": "value"}
        assert interaction.interaction_id is not None
        assert isinstance(interaction.timestamp, datetime)

    def test_session_add_interaction(self, sample_session, sample_interaction):
        """Test adding interaction to session."""
        initial_count = sample_session.metrics.total_interactions

        sample_session.add_interaction(sample_interaction)

        assert len(sample_session.interactions) == 1
        assert sample_session.metrics.total_interactions == initial_count + 1
        assert sample_session.metrics.positive_feedback_count == 1

    def test_session_feedback_ratio(self, sample_session):
        """Test feedback ratio calculation."""
        # Add positive interaction
        positive_interaction = QueryInteraction(
            query="Good query", feedback_type=FeedbackType.POSITIVE
        )
        sample_session.add_interaction(positive_interaction)

        # Add negative interaction
        negative_interaction = QueryInteraction(
            query="Bad query", feedback_type=FeedbackType.NEGATIVE
        )
        sample_session.add_interaction(negative_interaction)

        assert sample_session.get_feedback_ratio() == 0.5

    def test_session_should_adapt(self, sample_session):
        """Test adaptation criteria."""
        # Not enough interactions
        assert not sample_session.should_adapt()

        # Add enough negative interactions
        for i in range(6):
            interaction = QueryInteraction(
                query=f"Query {i}", feedback_type=FeedbackType.NEGATIVE
            )
            sample_session.add_interaction(interaction)

        # Should trigger adaptation due to low feedback ratio
        assert sample_session.should_adapt()


class TestRedisSessionManager:
    """Test Redis session management."""

    @pytest.fixture
    def session_manager(self, mock_redis):
        """Create session manager with mocked Redis."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            manager = SessionManager(
                redis_url="redis://localhost:6379", session_ttl_hours=1
            )
            return manager

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager, mock_redis):
        """Test session creation."""
        mock_redis.setex.return_value = True
        mock_redis.sadd.return_value = 1
        mock_redis.expire.return_value = True

        session = await session_manager.create_session(user_id="test_user")

        assert session.user_id == "test_user"
        assert session.status == SessionStatus.ACTIVE
        assert session.expires_at is not None
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session(self, session_manager, mock_redis, sample_session):
        """Test session retrieval."""
        # Mock Redis response
        session_data = sample_session.model_dump(mode="json")
        mock_redis.get.return_value = json.dumps(session_data, default=str)

        retrieved_session = await session_manager.get_session(
            str(sample_session.session_id)
        )

        assert retrieved_session is not None
        assert retrieved_session.session_id == sample_session.session_id
        assert retrieved_session.user_id == sample_session.user_id

    @pytest.mark.asyncio
    async def test_save_session(self, session_manager, mock_redis, sample_session):
        """Test session saving."""
        mock_redis.setex.return_value = True

        result = await session_manager.save_session(sample_session)

        assert result is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_session(self, session_manager, mock_redis, sample_session):
        """Test session deletion."""
        # Mock session retrieval
        session_data = sample_session.model_dump(mode="json")
        mock_redis.get.return_value = json.dumps(session_data, default=str)
        mock_redis.delete.return_value = 1
        mock_redis.srem.return_value = 1

        result = await session_manager.delete_session(str(sample_session.session_id))

        assert result is True
        mock_redis.delete.assert_called()

    @pytest.mark.asyncio
    async def test_save_model_state(self, session_manager, mock_redis):
        """Test model state saving."""
        model_state = ModelState(
            model_type="TestModel",
            model_version="1.0",
            serialized_state=b"test_data",
            feature_names=["feature1", "feature2"],
        )

        mock_redis.setex.return_value = True

        result = await session_manager.save_model_state(
            session_id="test_session", model_name="test_model", model_state=model_state
        )

        assert result is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_connection_error(self, mock_redis):
        """Test Redis connection error handling."""
        mock_redis.ping.side_effect = Exception("Connection failed")

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            manager = SessionManager()

            with pytest.raises(RedisSessionError):
                await manager._connect()


class TestIncrementalLearning:
    """Test incremental learning components."""

    @pytest.fixture
    def query_classifier(self):
        """Create query classifier learner."""
        return QueryClassifierLearner(n_features=1000)

    @pytest.fixture
    def preference_regressor(self):
        """Create preference regressor learner."""
        return PreferenceRegressorLearner()

    @pytest.mark.asyncio
    async def test_query_classifier_partial_fit(self, query_classifier):
        """Test incremental training of query classifier."""
        queries = ["What is the weather?", "How to cook pasta?"]
        intents = ["weather", "cooking"]

        await query_classifier.partial_fit(queries, intents)

        assert query_classifier.is_fitted
        assert query_classifier.update_count == 1
        assert len(query_classifier.classes_) == 2

    @pytest.mark.asyncio
    async def test_query_classifier_predict(self, query_classifier):
        """Test query classifier prediction."""
        # Train first
        queries = ["What is the weather?", "How to cook pasta?", "Weather forecast"]
        intents = ["weather", "cooking", "weather"]

        await query_classifier.partial_fit(queries, intents)

        # Predict
        test_queries = ["Is it raining?", "Recipe for soup"]
        predictions = await query_classifier.predict(test_queries)

        assert len(predictions) == 2
        assert all(pred in ["weather", "cooking"] for pred in predictions)

    @pytest.mark.asyncio
    async def test_preference_regressor_partial_fit(self, preference_regressor):
        """Test incremental training of preference regressor."""
        queries = ["Good query", "Bad query"]
        scores = [1.0, 0.0]

        await preference_regressor.partial_fit(queries, scores)

        assert preference_regressor.is_fitted
        assert preference_regressor.update_count == 1

    @pytest.mark.asyncio
    async def test_preference_regressor_predict(self, preference_regressor):
        """Test preference regressor prediction."""
        # Train first
        queries = ["Excellent query", "Poor query", "Good query"]
        scores = [1.0, 0.0, 0.8]

        await preference_regressor.partial_fit(queries, scores)

        # Predict
        test_queries = ["Great question", "Terrible question"]
        predictions = await preference_regressor.predict(test_queries)

        assert len(predictions) == 2
        assert all(0.0 <= pred <= 1.0 for pred in predictions)

    def test_model_state_serialization(self, query_classifier):
        """Test model state serialization and loading."""
        # Create some training data
        queries = ["Test query 1", "Test query 2"]
        intents = ["intent1", "intent2"]

        # Train the model (sync for test)
        asyncio.run(query_classifier.partial_fit(queries, intents))

        # Serialize state
        model_state = query_classifier.serialize_state()

        assert model_state.model_type == "QueryClassifierLearner"
        assert model_state.model_version == "1.0"
        assert len(model_state.serialized_state) > 0
        assert len(model_state.feature_names) == 2

        # Create new classifier and load state
        new_classifier = QueryClassifierLearner()
        new_classifier.load_state(model_state)

        assert new_classifier.is_fitted
        assert new_classifier.update_count == 1
        assert len(new_classifier.classes_) == 2


class TestLearningOrchestrator:
    """Test learning orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        """Create learning orchestrator."""
        return LearningOrchestrator()

    @pytest.mark.asyncio
    async def test_learn_from_interaction(self, orchestrator, sample_interaction):
        """Test learning from single interaction."""
        result = await orchestrator.learn_from_interaction(sample_interaction)

        assert "timestamp" in result
        assert "intent_learning" in result or "preference_learning" in result

    @pytest.mark.asyncio
    async def test_learn_from_interactions_batch(self, orchestrator):
        """Test batch learning from multiple interactions."""
        interactions = [
            QueryInteraction(
                query="Weather query",
                query_intent="weather",
                feedback_type=FeedbackType.POSITIVE,
            ),
            QueryInteraction(
                query="Cooking query",
                query_intent="cooking",
                feedback_type=FeedbackType.NEGATIVE,
            ),
        ]

        result = await orchestrator.learn_from_interactions(interactions)

        assert result["interactions_processed"] == 2
        assert result["intent_samples"] == 2
        assert result["preference_samples"] == 2

    @pytest.mark.asyncio
    async def test_predict_intent(self, orchestrator):
        """Test intent prediction."""
        # Train with some data first
        interactions = [
            QueryInteraction(query="What's the weather?", query_intent="weather"),
            QueryInteraction(query="How to cook?", query_intent="cooking"),
        ]
        await orchestrator.learn_from_interactions(interactions)

        # Predict
        queries = ["Is it sunny?", "Recipe for pasta"]
        predictions = await orchestrator.predict_intent(queries)

        assert len(predictions) == 2
        assert all("query" in pred for pred in predictions)
        assert all("predicted_intent" in pred for pred in predictions)

    def test_feedback_to_score_conversion(self, orchestrator):
        """Test feedback type to score conversion."""
        assert orchestrator._feedback_to_score(FeedbackType.POSITIVE, None) == 1.0
        assert orchestrator._feedback_to_score(FeedbackType.NEGATIVE, None) == 0.0
        assert orchestrator._feedback_to_score(FeedbackType.NEUTRAL, None) == 0.5
        assert orchestrator._feedback_to_score(FeedbackType.EXPLICIT_RATING, 5) == 1.0
        assert orchestrator._feedback_to_score(FeedbackType.EXPLICIT_RATING, 1) == 0.0
        assert orchestrator._feedback_to_score(FeedbackType.EXPLICIT_RATING, 3) == 0.5


class TestQueryLearningSystem:
    """Test main query learning system."""

    @pytest.fixture
    def learning_system(self, mock_redis):
        """Create query learning system with mocked dependencies."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            session_manager = SessionManager(redis_url="redis://localhost:6379")
            system = QueryLearningSystem(session_manager=session_manager)
            return system

    @pytest.mark.asyncio
    async def test_track_interaction(self, learning_system, mock_redis):
        """Test interaction tracking."""
        # Mock session creation
        mock_redis.setex.return_value = True
        mock_redis.sadd.return_value = 1
        mock_redis.expire.return_value = True

        result = await learning_system.track_interaction(
            query="Test query",
            context={"domain": "test"},
            feedback_type=FeedbackType.POSITIVE,
        )

        assert "session_id" in result
        assert "interaction_id" in result
        assert "learning_results" in result
        assert "session_metrics" in result

    @pytest.mark.asyncio
    async def test_adapt_strategies(self, learning_system, mock_redis, sample_session):
        """Test strategy adaptation."""
        # Mock session retrieval and saving
        session_data = sample_session.model_dump(mode="json")
        mock_redis.get.return_value = json.dumps(session_data, default=str)
        mock_redis.setex.return_value = True

        # Add enough interactions to trigger adaptation
        for i in range(6):
            interaction = QueryInteraction(
                query=f"Query {i}", feedback_type=FeedbackType.NEGATIVE
            )
            sample_session.add_interaction(interaction)

        # Update mock data
        session_data = sample_session.model_dump(mode="json")
        mock_redis.get.return_value = json.dumps(session_data, default=str)

        result = await learning_system.adapt_strategies(
            session_id=str(sample_session.session_id), force_adaptation=True
        )

        assert result["adapted"] is True
        assert "old_strategy" in result
        assert "new_strategy" in result

    @pytest.mark.asyncio
    async def test_get_session_insights(
        self, learning_system, mock_redis, sample_session
    ):
        """Test session insights retrieval."""
        # Mock session retrieval
        session_data = sample_session.model_dump(mode="json")
        mock_redis.get.return_value = json.dumps(session_data, default=str)

        insights = await learning_system.get_session_insights(
            session_id=str(sample_session.session_id), include_interactions=True
        )

        assert "session_id" in insights
        assert "metrics" in insights
        assert "user_preferences" in insights
        assert "strategy_history" in insights

    @pytest.mark.asyncio
    async def test_predict_preferences(self, learning_system):
        """Test preference prediction."""
        queries = ["Test query 1", "Test query 2"]
        predictions = await learning_system.predict_preferences(queries)

        assert len(predictions) == 2
        assert all("query" in pred for pred in predictions)
        assert all("preference_prediction" in pred for pred in predictions)

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, learning_system, mock_redis):
        """Test expired session cleanup."""
        # Mock scan_iter to return some keys
        mock_redis.scan_iter.return_value = [
            b"mosaic:session:expired_session_1",
            b"mosaic:session:expired_session_2",
        ]
        mock_redis.ttl.return_value = -2  # Key doesn't exist (expired)

        result = await learning_system.cleanup_expired_sessions()

        assert "cleanup_count" in result
        assert "last_cleanup" in result

    @pytest.mark.asyncio
    async def test_get_system_stats(self, learning_system):
        """Test system statistics retrieval."""
        stats = await learning_system.get_system_stats()

        assert "total_interactions" in stats
        assert "total_adaptations" in stats
        assert "adaptation_rate" in stats
        assert "settings" in stats


class TestMCPIntegration:
    """Test MCP protocol integration."""

    @pytest.mark.asyncio
    async def test_tool_registration(self):
        """Test MCP tool registration."""
        from mcp.server import Server
        from ..plugins.query_learning_system import register_query_learning_tools

        # Create mock server and system
        server = Server("test_server")
        learning_system = MagicMock()

        # Register tools
        register_query_learning_tools(server, learning_system)

        # Test that tools are registered
        tools = await server.list_tools()
        tool_names = [tool.name for tool in tools]

        expected_tools = [
            "track_query_interaction",
            "adapt_query_strategies",
            "get_session_insights",
            "predict_query_preferences",
            "get_query_learning_stats",
        ]

        for expected_tool in expected_tools:
            assert expected_tool in tool_names


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_session_id(self, learning_system):
        """Test handling of invalid session ID."""
        with pytest.raises(Exception):  # Should raise QueryLearningError
            await learning_system.get_session_insights("invalid_session_id")

    @pytest.mark.asyncio
    async def test_empty_query_list(self, learning_system):
        """Test handling of empty query list."""
        predictions = await learning_system.predict_preferences([])
        assert predictions == []

    @pytest.mark.asyncio
    async def test_malformed_feedback(self, learning_system, mock_redis):
        """Test handling of malformed feedback data."""
        mock_redis.setex.return_value = True
        mock_redis.sadd.return_value = 1
        mock_redis.expire.return_value = True

        # This should not crash
        result = await learning_system.track_interaction(
            query="Test query",
            context={},
            feedback_type=None,  # No feedback
            feedback_value="invalid",  # Invalid value
        )

        assert "session_id" in result

    def test_model_serialization_error(self):
        """Test handling of model serialization errors."""
        classifier = QueryClassifierLearner()

        # Mock a serialization error
        with patch("pickle.dumps", side_effect=Exception("Serialization failed")):
            with pytest.raises(Exception):
                classifier.serialize_state()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
