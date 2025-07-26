"""
Comprehensive test suite for Query Intent Classification System.
Tests models, training, classification, and MCP integration.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path

from ..models.intent_models import (
    QueryIntentType,
    RetrievalStrategy,
    ConfidenceLevel,
    ClassificationRequest,
    ClassificationResult,
    TrainingConfig,
    IntentClassificationPlugin,
    RetrainingRequest,
)
from ..training.intent_training_data import (
    IntentTrainingDataGenerator,
    generate_training_data,
)
from ..training.model_trainer import (
    IntentModelTrainer,
    train_intent_classifier,
    load_intent_classifier,
)
from ..plugins.query_intent_classifier import (
    QueryIntentClassifier,
    get_intent_classifier,
    reset_intent_classifier,
)


class TestIntentModels:
    """Test intent classification models."""

    def test_query_intent_type_enum(self):
        """Test QueryIntentType enum values."""
        assert QueryIntentType.GRAPH_RAG.value == "GRAPH_RAG"
        assert QueryIntentType.VECTOR_RAG.value == "VECTOR_RAG"
        assert QueryIntentType.DATABASE_RAG.value == "DATABASE_RAG"
        assert QueryIntentType.HYBRID.value == "HYBRID"

    def test_retrieval_strategy_enum(self):
        """Test RetrievalStrategy enum values."""
        assert RetrievalStrategy.GRAPH_RAG.value == "GRAPH_RAG"
        assert RetrievalStrategy.VECTOR_RAG.value == "VECTOR_RAG"
        assert RetrievalStrategy.DATABASE_RAG.value == "DATABASE_RAG"
        assert RetrievalStrategy.HYBRID.value == "HYBRID"

    def test_confidence_level_enum(self):
        """Test ConfidenceLevel enum values."""
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"

    def test_classification_request_validation(self):
        """Test ClassificationRequest validation."""
        # Valid request
        request = ClassificationRequest(
            query="What functions call main?",
            context="Code analysis session",
            user_id="test_user",
            session_id="test_session",
        )
        assert request.query == "What functions call main?"
        assert request.context == "Code analysis session"

        # Empty query should be handled by the classifier, not the model
        request_empty = ClassificationRequest(query="")
        assert request_empty.query == ""

    def test_classification_result_creation(self):
        """Test ClassificationResult creation."""
        result = ClassificationResult(
            intent=QueryIntentType.GRAPH_RAG,
            confidence=0.85,
            confidence_level=ConfidenceLevel.HIGH,
            strategy=RetrievalStrategy.GRAPH_RAG,
            alternative_intents=[],
            reasoning="High confidence graph traversal query",
        )

        assert result.intent == QueryIntentType.GRAPH_RAG
        assert result.confidence == 0.85
        assert result.confidence_level == ConfidenceLevel.HIGH
        assert result.strategy == RetrievalStrategy.GRAPH_RAG

    def test_classification_result_from_probabilities(self):
        """Test ClassificationResult.from_probabilities method."""
        probabilities = {
            QueryIntentType.GRAPH_RAG: 0.8,
            QueryIntentType.VECTOR_RAG: 0.15,
            QueryIntentType.DATABASE_RAG: 0.03,
            QueryIntentType.HYBRID: 0.02,
        }

        result = ClassificationResult.from_probabilities(probabilities, "test query")

        assert result.intent == QueryIntentType.GRAPH_RAG
        assert result.confidence == 0.8
        assert result.confidence_level == ConfidenceLevel.HIGH
        assert len(result.alternative_intents) == 3
        assert result.alternative_intents[0].intent == QueryIntentType.VECTOR_RAG

    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()

        assert config.train_test_split == 0.2
        assert config.random_state == 42
        assert config.cv_folds == 5
        assert config.model_params["n_estimators"] == 100

    def test_intent_classification_plugin_defaults(self):
        """Test IntentClassificationPlugin default values."""
        plugin = IntentClassificationPlugin()

        assert plugin.confidence_threshold == 0.7
        assert plugin.fallback_strategy == RetrievalStrategy.HYBRID
        assert plugin.log_classifications is True
        assert plugin.performance_monitoring is True
        assert plugin.max_query_length == 2000


class TestTrainingDataGeneration:
    """Test training data generation."""

    def test_intent_training_data_generator_initialization(self):
        """Test IntentTrainingDataGenerator initialization."""
        generator = IntentTrainingDataGenerator(random_seed=42)

        assert generator.random_seed == 42
        assert len(generator.entities) > 0
        assert len(generator.technologies) > 0
        assert len(generator.patterns[QueryIntentType.GRAPH_RAG]) > 0

    def test_generate_training_data_structure(self):
        """Test generate_training_data function structure."""
        dataset = generate_training_data(samples_per_class=50, random_seed=42)

        assert len(dataset.samples) == 50 * 4  # 4 intent types
        assert dataset.metadata["samples_per_class"] == 50
        assert dataset.metadata["random_seed"] == 42

        # Check intent distribution
        intent_counts = {}
        for sample in dataset.samples:
            intent_counts[sample.intent] = intent_counts.get(sample.intent, 0) + 1

        for intent in QueryIntentType:
            assert intent_counts[intent] == 50

    def test_training_sample_quality(self):
        """Test quality of generated training samples."""
        dataset = generate_training_data(samples_per_class=10, random_seed=42)

        for sample in dataset.samples:
            # Each sample should have non-empty query
            assert len(sample.query.strip()) > 0

            # Each sample should have valid intent
            assert sample.intent in QueryIntentType

            # Each sample should have metadata
            assert "pattern" in sample.metadata
            assert "generated_at" in sample.metadata

    def test_pattern_distribution(self):
        """Test that patterns are distributed across intents."""
        generator = IntentTrainingDataGenerator(random_seed=42)

        # Test each intent type has patterns
        for intent in QueryIntentType:
            patterns = generator.patterns[intent]
            assert len(patterns) >= 5  # At least 5 patterns per intent

            # Test pattern structure
            for pattern in patterns[:3]:  # Test first 3 patterns
                assert "{" in pattern  # Should have placeholders
                assert "}" in pattern


class TestModelTraining:
    """Test model training pipeline."""

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_intent_model_trainer_initialization(self, temp_model_dir):
        """Test IntentModelTrainer initialization."""
        config = TrainingConfig()
        trainer = IntentModelTrainer(config)

        assert trainer.config == config
        assert trainer.vectorizer is not None
        assert trainer.classifier is not None
        assert trainer.scaler is not None

    def test_model_training_pipeline(self, temp_model_dir):
        """Test full model training pipeline."""
        config = TrainingConfig()

        # Generate small training dataset
        dataset = generate_training_data(samples_per_class=20, random_seed=42)

        # Train model
        trainer, metrics = train_intent_classifier(
            config=config, dataset=dataset, model_dir=temp_model_dir
        )

        # Verify trainer
        assert trainer is not None
        assert trainer.is_trained is True

        # Verify metrics
        assert metrics.accuracy > 0.5  # Should be reasonable
        assert metrics.training_samples == 80  # 20 * 4 intents
        assert len(metrics.precision) == 4  # 4 intents
        assert len(metrics.recall) == 4
        assert len(metrics.f1_score) == 4

        # Verify model files created
        model_dir = Path(temp_model_dir)
        assert (model_dir / "active").exists()
        assert (model_dir / "active" / "vectorizer.joblib").exists()
        assert (model_dir / "active" / "classifier.joblib").exists()
        assert (model_dir / "active" / "scaler.joblib").exists()
        assert (model_dir / "active" / "metadata.json").exists()

    def test_model_save_load_cycle(self, temp_model_dir):
        """Test model save and load cycle."""
        config = TrainingConfig()
        dataset = generate_training_data(samples_per_class=15, random_seed=42)

        # Train and save model
        trainer, metrics = train_intent_classifier(
            config=config, dataset=dataset, model_dir=temp_model_dir
        )

        # Load model
        loaded_trainer = load_intent_classifier(model_dir=temp_model_dir)

        # Verify loaded model
        assert loaded_trainer is not None
        assert loaded_trainer.is_trained is True

        # Test predictions match
        test_queries = [
            "What functions call main?",
            "Find similar documents",
            "Count total users",
            "Analyze relationships and find documents",
        ]

        original_predictions = trainer.predict(test_queries)
        loaded_predictions = loaded_trainer.predict(test_queries)

        assert original_predictions == loaded_predictions

    def test_model_prediction_consistency(self, temp_model_dir):
        """Test model prediction consistency."""
        config = TrainingConfig(random_state=42)
        dataset = generate_training_data(samples_per_class=25, random_seed=42)

        trainer, _ = train_intent_classifier(
            config=config, dataset=dataset, model_dir=temp_model_dir
        )

        # Test consistent predictions
        test_query = "What functions call main?"
        predictions_1 = trainer.predict([test_query])
        predictions_2 = trainer.predict([test_query])

        assert predictions_1 == predictions_2

    def test_model_probability_output(self, temp_model_dir):
        """Test model probability output."""
        config = TrainingConfig()
        dataset = generate_training_data(samples_per_class=20, random_seed=42)

        trainer, _ = train_intent_classifier(
            config=config, dataset=dataset, model_dir=temp_model_dir
        )

        # Test probability predictions
        test_queries = [
            "What functions call main?",  # Should be GRAPH_RAG
            "Find similar documents",  # Should be VECTOR_RAG
            "Count total users",  # Should be DATABASE_RAG
        ]

        probabilities = trainer.predict_probabilities(test_queries)

        assert len(probabilities) == 3

        for probs in probabilities:
            # Probabilities should sum to ~1.0
            total_prob = sum(probs.values())
            assert abs(total_prob - 1.0) < 0.01

            # Should have all 4 intent types
            assert len(probs) == 4
            assert all(intent in probs for intent in QueryIntentType)

            # All probabilities should be non-negative
            assert all(prob >= 0 for prob in probs.values())


@pytest.mark.asyncio
class TestQueryIntentClassifier:
    """Test QueryIntentClassifier plugin."""

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    async def classifier(self, temp_model_dir):
        """Create classifier instance."""
        config = IntentClassificationPlugin(
            confidence_threshold=0.6,
            log_classifications=False,
            performance_monitoring=True,
        )

        classifier = QueryIntentClassifier(config, model_dir=temp_model_dir)
        await classifier.initialize()
        return classifier

    async def test_classifier_initialization(self, temp_model_dir):
        """Test classifier initialization."""
        classifier = QueryIntentClassifier(model_dir=temp_model_dir)

        assert classifier.is_initialized is False
        assert classifier.trainer is None

        success = await classifier.initialize()

        assert success is True
        assert classifier.is_initialized is True
        assert classifier.trainer is not None

    async def test_classify_intent_basic(self, classifier):
        """Test basic intent classification."""
        request = ClassificationRequest(
            query="What functions call the main function?",
            context="Code analysis",
            user_id="test_user",
        )

        result = await classifier.classify_intent(request)

        assert isinstance(result, ClassificationResult)
        assert result.intent in QueryIntentType
        assert 0.0 <= result.confidence <= 1.0
        assert result.confidence_level in ConfidenceLevel
        assert result.strategy in RetrievalStrategy
        assert len(result.reasoning) > 0

    async def test_classify_intent_edge_cases(self, classifier):
        """Test edge cases for intent classification."""
        # Empty query
        request_empty = ClassificationRequest(query="")
        result_empty = await classifier.classify_intent(request_empty)

        # Should handle gracefully
        assert isinstance(result_empty, ClassificationResult)

        # Very long query
        long_query = "What functions call main? " * 200  # Very long
        request_long = ClassificationRequest(query=long_query)
        result_long = await classifier.classify_intent(request_long)

        assert isinstance(result_long, ClassificationResult)

    async def test_predict_strategy(self, classifier):
        """Test strategy prediction."""
        strategies = []

        test_queries = [
            "What functions call main?",  # GRAPH_RAG
            "Find similar documents",  # VECTOR_RAG
            "Count total users by region",  # DATABASE_RAG
            "Analyze code relationships and find similar patterns",  # HYBRID
        ]

        for query in test_queries:
            strategy = await classifier.predict_strategy(query)
            strategies.append(strategy)
            assert strategy in RetrievalStrategy

        # Should have some variety in strategies
        unique_strategies = set(strategies)
        assert len(unique_strategies) >= 2

    async def test_performance_monitoring(self, classifier):
        """Test performance monitoring features."""
        # Initial stats
        initial_stats = classifier.get_performance_stats()
        initial_count = initial_stats["total_classifications"]

        # Make some classifications
        test_queries = [
            "What functions call main?",
            "Find similar documents",
            "Count users",
        ]

        for query in test_queries:
            request = ClassificationRequest(query=query)
            await classifier.classify_intent(request)

        # Check updated stats
        updated_stats = classifier.get_performance_stats()

        assert updated_stats["total_classifications"] == initial_count + 3
        assert updated_stats["average_response_time"] > 0
        assert "confidence_distribution" in updated_stats
        assert "intent_distribution" in updated_stats

    async def test_health_check(self, classifier):
        """Test health check functionality."""
        health = await classifier.health_check()

        assert "status" in health
        assert health["status"] in ["healthy", "unhealthy"]
        assert "initialized" in health
        assert "trainer_loaded" in health
        assert "model_dir_exists" in health
        assert "performance_stats" in health
        assert "issues" in health

        # Should include test classification
        if health["status"] == "healthy":
            assert "test_classification" in health

    async def test_model_metrics(self, classifier):
        """Test model metrics retrieval."""
        metrics = await classifier.get_model_metrics()

        if metrics:  # Metrics available
            assert hasattr(metrics, "accuracy")
            assert hasattr(metrics, "training_samples")
            assert hasattr(metrics, "precision")
            assert hasattr(metrics, "recall")
            assert hasattr(metrics, "f1_score")

            assert 0.0 <= metrics.accuracy <= 1.0
            assert metrics.training_samples > 0

    async def test_classification_history(self, classifier):
        """Test classification history tracking."""
        # Enable logging
        classifier.config.log_classifications = True

        # Make classification
        request = ClassificationRequest(
            query="Test query for history",
            user_id="test_user",
            session_id="test_session",
        )
        await classifier.classify_intent(request)

        # Check history
        history = classifier.get_classification_history(limit=5)

        assert len(history) >= 1

        last_entry = history[-1]
        assert "event" in last_entry
        assert "timestamp" in last_entry
        assert "query" in last_entry
        assert "intent" in last_entry
        assert "confidence" in last_entry

    async def test_retrain_model(self, classifier):
        """Test model retraining."""
        # Generate new training data
        dataset = generate_training_data(samples_per_class=30, random_seed=123)

        retrain_request = RetrainingRequest(
            reason="Test retraining", force_retrain=True, dataset=dataset
        )

        # Retrain model
        metrics = await classifier.retrain_model(retrain_request)

        assert metrics is not None
        assert metrics.accuracy > 0
        assert metrics.training_samples == 120  # 30 * 4 intents
        assert classifier.last_retrain_time is not None

    async def test_global_classifier_instance(self, temp_model_dir):
        """Test global classifier instance management."""
        # Reset first
        reset_intent_classifier()

        # Get classifier
        config = IntentClassificationPlugin()
        classifier1 = await get_intent_classifier(config)
        classifier2 = await get_intent_classifier(config)

        # Should be the same instance
        assert classifier1 is classifier2
        assert classifier1.is_initialized is True

        # Reset and verify
        reset_intent_classifier()
        classifier3 = await get_intent_classifier(config)

        # Should be different instance
        assert classifier3 is not classifier1


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling and edge cases."""

    async def test_uninitialized_classifier_error(self):
        """Test error when using uninitialized classifier."""
        classifier = QueryIntentClassifier()

        request = ClassificationRequest(query="test")

        with pytest.raises(RuntimeError, match="Classifier not initialized"):
            await classifier.classify_intent(request)

    def test_invalid_training_config(self):
        """Test invalid training configuration."""
        with pytest.raises(ValueError):
            TrainingConfig(train_test_split=1.5)  # Invalid split

        with pytest.raises(ValueError):
            TrainingConfig(cv_folds=1)  # Too few folds

    def test_invalid_classification_request(self):
        """Test invalid classification requests."""
        # These should be handled by the classifier, not raise validation errors
        request_none = ClassificationRequest(query=None)
        assert request_none.query is None

        request_empty = ClassificationRequest(query="")
        assert request_empty.query == ""


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    async def test_end_to_end_workflow(self, temp_model_dir):
        """Test complete end-to-end workflow."""
        # 1. Create configuration
        config = IntentClassificationPlugin(
            confidence_threshold=0.7,
            fallback_strategy=RetrievalStrategy.HYBRID,
            log_classifications=True,
            performance_monitoring=True,
        )

        # 2. Initialize classifier
        classifier = QueryIntentClassifier(config, model_dir=temp_model_dir)
        success = await classifier.initialize()
        assert success is True

        # 3. Test various query types
        test_cases = [
            {
                "query": "What functions are called by the main function in the codebase?",
                "expected_intent_type": "graph",  # Should be graph-related
            },
            {
                "query": "Find documents similar to 'machine learning algorithms'",
                "expected_intent_type": "vector",  # Should be vector search
            },
            {
                "query": "How many users registered last month?",
                "expected_intent_type": "database",  # Should be database query
            },
            {
                "query": "Analyze the relationships between functions and find similar code patterns",
                "expected_intent_type": "hybrid",  # Should be hybrid approach
            },
        ]

        results = []
        for test_case in test_cases:
            request = ClassificationRequest(
                query=test_case["query"],
                user_id="integration_test",
                session_id="end_to_end_test",
            )

            result = await classifier.classify_intent(request)
            results.append(
                {
                    "query": test_case["query"],
                    "result": result,
                    "expected": test_case["expected_intent_type"],
                }
            )

            # Basic validation
            assert isinstance(result, ClassificationResult)
            assert result.confidence >= 0.0
            assert result.intent in QueryIntentType
            assert result.strategy in RetrievalStrategy

        # 4. Check performance stats
        stats = classifier.get_performance_stats()
        assert stats["total_classifications"] == 4
        assert stats["average_response_time"] > 0

        # 5. Check classification history
        history = classifier.get_classification_history()
        assert len(history) == 4

        # 6. Test health check
        health = await classifier.health_check()
        assert health["status"] == "healthy"
        assert health["initialized"] is True

        # 7. Test model metrics
        metrics = await classifier.get_model_metrics()
        if metrics:
            assert metrics.accuracy > 0.5  # Reasonable accuracy

        return results

    async def test_concurrent_classifications(self, temp_model_dir):
        """Test concurrent classification requests."""
        classifier = QueryIntentClassifier(model_dir=temp_model_dir)
        await classifier.initialize()

        # Create multiple concurrent requests
        queries = [f"What functions call main? Query {i}" for i in range(10)]

        requests = [
            ClassificationRequest(query=query, session_id=f"concurrent_{i}")
            for i, query in enumerate(queries)
        ]

        # Execute concurrently
        tasks = [classifier.classify_intent(req) for req in requests]
        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        assert len(results) == 10
        for result in results:
            assert isinstance(result, ClassificationResult)
            assert result.confidence >= 0.0

    async def test_model_persistence(self, temp_model_dir):
        """Test model persistence across instances."""
        # Create and train first instance
        config = IntentClassificationPlugin()
        classifier1 = QueryIntentClassifier(config, model_dir=temp_model_dir)
        await classifier1.initialize()

        # Make a classification
        request = ClassificationRequest(query="What functions call main?")
        result1 = await classifier1.classify_intent(request)

        # Create second instance (should load existing model)
        classifier2 = QueryIntentClassifier(config, model_dir=temp_model_dir)
        await classifier2.initialize()

        # Make same classification
        result2 = await classifier2.classify_intent(request)

        # Results should be identical
        assert result1.intent == result2.intent
        assert abs(result1.confidence - result2.confidence) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
