"""
Query Intent Classification Plugin for OmniRAG orchestration.
Intelligently classifies user queries to determine optimal retrieval strategies.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta

from ..models.intent_models import (
    QueryIntentType,
    RetrievalStrategy,
    ClassificationRequest,
    ClassificationResult,
    TrainingConfig,
    ModelMetrics,
    IntentClassificationPlugin,
    RetrainingRequest,
)
from ..training.model_trainer import (
    IntentModelTrainer,
    train_intent_classifier,
    load_intent_classifier,
)
from ..training.intent_training_data import generate_training_data


logger = logging.getLogger(__name__)


class QueryIntentClassifier:
    """
    Intelligent query intent classifier for OmniRAG orchestration.

    Determines whether queries should use:
    - GRAPH_RAG: Relationship and graph traversal queries
    - VECTOR_RAG: Semantic similarity and content queries
    - DATABASE_RAG: Structured data and aggregation queries
    - HYBRID: Complex queries requiring multiple strategies
    """

    def __init__(
        self,
        config: Optional[IntentClassificationPlugin] = None,
        model_dir: str = "models/intent_classifier",
    ):
        """Initialize the intent classifier."""

        self.config = config or IntentClassificationPlugin()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.trainer: Optional[IntentModelTrainer] = None
        self.is_initialized = False
        self.last_retrain_time: Optional[datetime] = None

        # Performance monitoring
        self.classification_history: List[Dict[str, Any]] = []
        self.performance_stats = {
            "total_classifications": 0,
            "accuracy_samples": [],
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "intent_distribution": {intent.value: 0 for intent in QueryIntentType},
            "average_response_time": 0.0,
        }

        logger.info(f"QueryIntentClassifier initialized with config: {self.config}")

    async def initialize(self) -> bool:
        """Initialize the classifier by loading or training a model."""

        try:
            # Try to load existing model
            if await self._load_existing_model():
                logger.info("Loaded existing intent classification model")
                self.is_initialized = True
                return True

            # Train new model if none exists
            logger.info("No existing model found, training new intent classifier...")
            await self._train_new_model()
            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize intent classifier: {e}")
            return False

    async def _load_existing_model(self) -> bool:
        """Try to load an existing trained model."""

        try:
            # Check for active model
            active_path = self.model_dir / "active"
            if not active_path.exists():
                return False

            # Load the model in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.trainer = await loop.run_in_executor(
                None, load_intent_classifier, None, str(self.model_dir)
            )

            return True

        except Exception as e:
            logger.warning(f"Failed to load existing model: {e}")
            return False

    async def _train_new_model(self) -> None:
        """Train a new intent classification model."""

        # Generate training data
        training_config = TrainingConfig()

        # Train model in thread pool
        loop = asyncio.get_event_loop()
        self.trainer, metrics = await loop.run_in_executor(
            None, train_intent_classifier, training_config, None, str(self.model_dir)
        )

        self.last_retrain_time = datetime.utcnow()

        logger.info(f"Trained new model with accuracy: {metrics.accuracy:.4f}")

        # Log training metrics
        if self.config.performance_monitoring:
            self._log_training_metrics(metrics)

    def _log_training_metrics(self, metrics: ModelMetrics) -> None:
        """Log training metrics for monitoring."""

        log_entry = {
            "event": "model_training",
            "timestamp": datetime.utcnow(),
            "accuracy": metrics.accuracy,
            "training_samples": metrics.training_samples,
            "model_version": metrics.model_version,
            "per_class_metrics": {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
            },
        }

        self.classification_history.append(log_entry)
        logger.info(f"Training completed: {log_entry}")

    async def classify_intent(
        self, request: ClassificationRequest
    ) -> ClassificationResult:
        """
        Classify the intent of a user query.

        Args:
            request: Classification request with query and context

        Returns:
            Classification result with intent, confidence, and strategy
        """

        if not self.is_initialized or not self.trainer:
            raise RuntimeError("Classifier not initialized. Call initialize() first.")

        start_time = datetime.utcnow()

        try:
            # Validate query
            if not request.query or len(request.query.strip()) == 0:
                raise ValueError("Query cannot be empty")

            if len(request.query) > self.config.max_query_length:
                raise ValueError(
                    f"Query too long (max {self.config.max_query_length} characters)"
                )

            # Get predictions in thread pool
            loop = asyncio.get_event_loop()
            probabilities_list = await loop.run_in_executor(
                None, self.trainer.predict_probabilities, [request.query]
            )

            probabilities = probabilities_list[0]

            # Create result
            result = ClassificationResult.from_probabilities(
                probabilities, request.query
            )

            # Apply confidence threshold and fallback logic
            if result.confidence < self.config.fallback_strategy:
                result = self._apply_fallback_strategy(result, request)

            # Update performance stats
            self._update_performance_stats(result, start_time)

            # Log classification if enabled
            if self.config.log_classifications:
                self._log_classification(request, result)

            return result

        except Exception as e:
            logger.error(f"Classification failed for query '{request.query}': {e}")

            # Return fallback result
            fallback_result = ClassificationResult(
                intent=QueryIntentType.HYBRID,
                confidence=0.0,
                confidence_level="low",
                strategy=self.config.fallback_strategy,
                alternative_intents=[],
                reasoning=f"Classification failed: {str(e)}. Using fallback strategy.",
            )

            return fallback_result

    def _apply_fallback_strategy(
        self, result: ClassificationResult, request: ClassificationRequest
    ) -> ClassificationResult:
        """Apply fallback strategy for low-confidence predictions."""

        # Use configured fallback strategy
        fallback_intent = QueryIntentType.HYBRID
        fallback_strategy = self.config.fallback_strategy

        # Update result
        result.intent = fallback_intent
        result.strategy = fallback_strategy
        result.reasoning += f" (Applied fallback: {fallback_strategy.value})"

        logger.debug(
            f"Applied fallback strategy for low confidence: {result.confidence:.3f}"
        )

        return result

    async def predict_strategy(
        self, query: str, context: Optional[str] = None
    ) -> RetrievalStrategy:
        """
        Predict the optimal retrieval strategy for a query.

        Args:
            query: Natural language query
            context: Optional context information

        Returns:
            Recommended retrieval strategy
        """

        request = ClassificationRequest(query=query, context=context)
        result = await self.classify_intent(request)
        return result.strategy

    def _update_performance_stats(
        self, result: ClassificationResult, start_time: datetime
    ) -> None:
        """Update performance monitoring statistics."""

        response_time = (datetime.utcnow() - start_time).total_seconds()

        self.performance_stats["total_classifications"] += 1
        self.performance_stats["confidence_distribution"][
            result.confidence_level.value
        ] += 1
        self.performance_stats["intent_distribution"][result.intent.value] += 1

        # Update average response time
        total = self.performance_stats["total_classifications"]
        current_avg = self.performance_stats["average_response_time"]
        self.performance_stats["average_response_time"] = (
            current_avg * (total - 1) + response_time
        ) / total

    def _log_classification(
        self, request: ClassificationRequest, result: ClassificationResult
    ) -> None:
        """Log classification decision for monitoring and debugging."""

        log_entry = {
            "event": "classification",
            "timestamp": datetime.utcnow(),
            "query": request.query[:100],  # Truncate for privacy
            "intent": result.intent.value,
            "confidence": result.confidence,
            "confidence_level": result.confidence_level.value,
            "strategy": result.strategy.value,
            "user_id": request.user_id,
            "session_id": request.session_id,
        }

        self.classification_history.append(log_entry)

        # Keep only recent history (last 1000 entries)
        if len(self.classification_history) > 1000:
            self.classification_history = self.classification_history[-1000:]

    async def retrain_model(self, request: RetrainingRequest) -> ModelMetrics:
        """
        Retrain the intent classification model.

        Args:
            request: Retraining request with configuration and dataset

        Returns:
            Training metrics for the new model
        """

        if not self.is_initialized:
            raise RuntimeError("Classifier not initialized")

        # Check if retraining is needed
        if not request.force_retrain and self.last_retrain_time:
            time_since_retrain = datetime.utcnow() - self.last_retrain_time
            if time_since_retrain < timedelta(hours=24):
                raise ValueError(
                    "Model was retrained recently. Use force_retrain=True to override."
                )

        logger.info(f"Retraining model: {request.reason}")

        # Use provided config or defaults
        config = request.config or TrainingConfig()
        dataset = request.dataset

        # Train new model in thread pool
        loop = asyncio.get_event_loop()
        trainer, metrics = await loop.run_in_executor(
            None, train_intent_classifier, config, dataset, str(self.model_dir)
        )

        # Update trainer and timestamp
        self.trainer = trainer
        self.last_retrain_time = datetime.utcnow()

        # Log retraining
        if self.config.performance_monitoring:
            self._log_training_metrics(metrics)

        logger.info(
            f"Model retrained successfully. New accuracy: {metrics.accuracy:.4f}"
        )

        return metrics

    async def get_model_metrics(self) -> Optional[ModelMetrics]:
        """Get current model performance metrics."""

        if not self.trainer:
            return None

        # Generate test dataset for evaluation
        loop = asyncio.get_event_loop()
        test_dataset = await loop.run_in_executor(
            None,
            generate_training_data,
            50,  # 50 samples per class for testing
            42,  # Fixed seed for consistent test set
        )

        # Evaluate model
        metrics = await loop.run_in_executor(
            None, self.trainer.evaluate_on_dataset, test_dataset
        )

        return metrics

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance monitoring statistics."""

        stats = self.performance_stats.copy()
        stats["last_retrain_time"] = self.last_retrain_time
        stats["is_initialized"] = self.is_initialized
        stats["model_dir"] = str(self.model_dir)
        stats["config"] = self.config.model_dump()

        return stats

    def get_classification_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent classification history."""

        return self.classification_history[-limit:]

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the classifier."""

        health_status = {
            "status": "healthy",
            "initialized": self.is_initialized,
            "trainer_loaded": self.trainer is not None,
            "model_dir_exists": self.model_dir.exists(),
            "last_retrain": self.last_retrain_time,
            "performance_stats": self.performance_stats,
            "issues": [],
        }

        # Check for issues
        if not self.is_initialized:
            health_status["status"] = "unhealthy"
            health_status["issues"].append("Classifier not initialized")

        if not self.trainer:
            health_status["status"] = "unhealthy"
            health_status["issues"].append("No trained model loaded")

        # Test with a simple query
        try:
            if self.is_initialized:
                test_request = ClassificationRequest(query="What functions call main?")
                test_result = await self.classify_intent(test_request)
                health_status["test_classification"] = {
                    "query": test_request.query,
                    "intent": test_result.intent.value,
                    "confidence": test_result.confidence,
                }
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["issues"].append(f"Test classification failed: {str(e)}")

        return health_status


# Global classifier instance
_classifier_instance: Optional[QueryIntentClassifier] = None


async def get_intent_classifier(
    config: Optional[IntentClassificationPlugin] = None,
) -> QueryIntentClassifier:
    """Get or create the global intent classifier instance."""

    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = QueryIntentClassifier(config)
        await _classifier_instance.initialize()

    return _classifier_instance


def reset_intent_classifier() -> None:
    """Reset the global classifier instance (for testing)."""

    global _classifier_instance
    _classifier_instance = None
