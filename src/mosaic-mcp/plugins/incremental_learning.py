"""
Incremental learning wrapper for online query adaptation.

This module provides scikit-learn based incremental learning capabilities
for the OMR-P3-004 query learning and adaptation system.
"""

import logging
import pickle
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveRegressor
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

from ..models.session_models import QueryInteraction, FeedbackType, ModelState


logger = logging.getLogger(__name__)


class IncrementalLearnerError(Exception):
    """Incremental learning operation errors."""

    pass


class BaseIncrementalLearner(ABC):
    """Abstract base class for incremental learning models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_fitted = False
        self.update_count = 0
        self.feature_names: List[str] = []
        self.last_updated = datetime.now(timezone.utc)

    @abstractmethod
    async def partial_fit(
        self,
        X: Union[List[str], np.ndarray],
        y: Union[List[Any], np.ndarray],
        classes: Optional[List[Any]] = None,
    ) -> None:
        """Incrementally train the model."""
        pass

    @abstractmethod
    async def predict(self, X: Union[List[str], np.ndarray]) -> np.ndarray:
        """Make predictions."""
        pass

    @abstractmethod
    def serialize_state(self) -> ModelState:
        """Serialize model state for persistence."""
        pass

    @abstractmethod
    def load_state(self, model_state: ModelState) -> None:
        """Load model state from persistence."""
        pass


class QueryClassifierLearner(BaseIncrementalLearner):
    """
    Incremental classifier for query intent and category learning.

    Uses SGDClassifier with hashing vectorizer for memory-efficient
    online learning of query patterns.
    """

    def __init__(
        self,
        model_name: str = "query_classifier",
        n_features: int = 10000,
        alpha: float = 0.0001,
        max_iter: int = 1000,
    ):
        super().__init__(model_name)

        self.n_features = n_features
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
        )

        self.classifier = SGDClassifier(
            loss="hinge",
            alpha=alpha,
            max_iter=max_iter,
            random_state=42,
            warm_start=True,
        )

        self.label_encoder = LabelEncoder()
        self.classes_: Optional[np.ndarray] = None

    async def partial_fit(
        self, X: List[str], y: List[str], classes: Optional[List[str]] = None
    ) -> None:
        """
        Incrementally train the classifier.

        Args:
            X: List of query texts
            y: List of intent labels
            classes: Optional list of all possible classes
        """
        try:
            if not X or not y:
                return

            # Vectorize text features
            X_vec = self.vectorizer.transform(X)

            # Handle label encoding
            if not self.is_fitted:
                if classes:
                    self.label_encoder.fit(classes)
                    self.classes_ = self.label_encoder.classes_
                else:
                    self.label_encoder.fit(y)
                    self.classes_ = self.label_encoder.classes_

            # Encode labels
            y_encoded = self.label_encoder.transform(y)

            # Partial fit
            if not self.is_fitted:
                self.classifier.partial_fit(
                    X_vec, y_encoded, classes=range(len(self.classes_))
                )
                self.is_fitted = True
            else:
                # Handle new classes
                new_labels = set(y) - set(self.classes_)
                if new_labels:
                    # Extend label encoder and retrain
                    all_labels = list(self.classes_) + list(new_labels)
                    self.label_encoder = LabelEncoder()
                    self.label_encoder.fit(all_labels)
                    self.classes_ = self.label_encoder.classes_
                    y_encoded = self.label_encoder.transform(y)

                self.classifier.partial_fit(X_vec, y_encoded)

            self.update_count += 1
            self.last_updated = datetime.now(timezone.utc)

            logger.debug(f"Updated {self.model_name} with {len(X)} samples")

        except Exception as e:
            logger.error(f"Error in partial_fit for {self.model_name}: {e}")
            raise IncrementalLearnerError(f"Partial fit failed: {e}")

    async def predict(self, X: List[str]) -> List[str]:
        """
        Predict intent labels for queries.

        Args:
            X: List of query texts

        Returns:
            List of predicted intent labels
        """
        if not self.is_fitted or not X:
            return []

        try:
            X_vec = self.vectorizer.transform(X)
            y_pred_encoded = self.classifier.predict(X_vec)
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
            return y_pred.tolist()

        except Exception as e:
            logger.error(f"Error in predict for {self.model_name}: {e}")
            return []

    async def predict_proba(self, X: List[str]) -> List[Dict[str, float]]:
        """
        Predict class probabilities for queries.

        Args:
            X: List of query texts

        Returns:
            List of class probability dictionaries
        """
        if not self.is_fitted or not X:
            return []

        try:
            X_vec = self.vectorizer.transform(X)
            probabilities = self.classifier.predict_proba(X_vec)

            results = []
            for prob_array in probabilities:
                prob_dict = {
                    label: float(prob) for label, prob in zip(self.classes_, prob_array)
                }
                results.append(prob_dict)

            return results

        except Exception as e:
            logger.error(f"Error in predict_proba for {self.model_name}: {e}")
            return []

    def serialize_state(self) -> ModelState:
        """Serialize classifier state."""
        state_dict = {
            "classifier": self.classifier,
            "label_encoder": self.label_encoder,
            "classes": self.classes_,
            "is_fitted": self.is_fitted,
            "n_features": self.n_features,
        }

        serialized_state = pickle.dumps(state_dict)

        return ModelState(
            model_type="QueryClassifierLearner",
            model_version="1.0",
            serialized_state=serialized_state,
            feature_names=list(self.classes_) if self.classes_ is not None else [],
            last_updated=self.last_updated,
            update_count=self.update_count,
        )

    def load_state(self, model_state: ModelState) -> None:
        """Load classifier state."""
        try:
            state_dict = pickle.loads(model_state.serialized_state)

            self.classifier = state_dict["classifier"]
            self.label_encoder = state_dict["label_encoder"]
            self.classes_ = state_dict["classes"]
            self.is_fitted = state_dict["is_fitted"]
            self.n_features = state_dict.get("n_features", self.n_features)

            self.last_updated = model_state.last_updated
            self.update_count = model_state.update_count

            logger.info(f"Loaded state for {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading state for {self.model_name}: {e}")
            raise IncrementalLearnerError(f"State loading failed: {e}")


class PreferenceRegressorLearner(BaseIncrementalLearner):
    """
    Incremental regressor for user preference learning.

    Uses PassiveAggressiveRegressor to learn user preferences
    based on feedback and interaction patterns.
    """

    def __init__(
        self,
        model_name: str = "preference_regressor",
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        super().__init__(model_name)

        self.regressor = PassiveAggressiveRegressor(
            C=C, max_iter=max_iter, random_state=42, warm_start=True
        )

        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.feature_dim = 384  # Dimension of all-MiniLM-L6-v2

    async def partial_fit(
        self, X: List[str], y: List[float], classes: Optional[List[Any]] = None
    ) -> None:
        """
        Incrementally train the preference regressor.

        Args:
            X: List of query texts or contexts
            y: List of preference scores (0.0 to 1.0)
            classes: Unused for regression
        """
        try:
            if not X or not y:
                return

            # Encode text to embeddings
            X_embeddings = self.sentence_transformer.encode(X)
            y_array = np.array(y)

            # Partial fit
            if not self.is_fitted:
                self.regressor.partial_fit(X_embeddings, y_array)
                self.is_fitted = True
            else:
                self.regressor.partial_fit(X_embeddings, y_array)

            self.update_count += 1
            self.last_updated = datetime.now(timezone.utc)

            logger.debug(f"Updated {self.model_name} with {len(X)} samples")

        except Exception as e:
            logger.error(f"Error in partial_fit for {self.model_name}: {e}")
            raise IncrementalLearnerError(f"Partial fit failed: {e}")

    async def predict(self, X: List[str]) -> np.ndarray:
        """
        Predict preference scores for queries.

        Args:
            X: List of query texts or contexts

        Returns:
            Array of predicted preference scores
        """
        if not self.is_fitted or not X:
            return np.array([])

        try:
            X_embeddings = self.sentence_transformer.encode(X)
            predictions = self.regressor.predict(X_embeddings)
            return predictions

        except Exception as e:
            logger.error(f"Error in predict for {self.model_name}: {e}")
            return np.array([])

    def serialize_state(self) -> ModelState:
        """Serialize regressor state."""
        state_dict = {
            "regressor": self.regressor,
            "is_fitted": self.is_fitted,
            "feature_dim": self.feature_dim,
        }

        serialized_state = pickle.dumps(state_dict)

        return ModelState(
            model_type="PreferenceRegressorLearner",
            model_version="1.0",
            serialized_state=serialized_state,
            feature_names=[f"embedding_{i}" for i in range(self.feature_dim)],
            last_updated=self.last_updated,
            update_count=self.update_count,
        )

    def load_state(self, model_state: ModelState) -> None:
        """Load regressor state."""
        try:
            state_dict = pickle.loads(model_state.serialized_state)

            self.regressor = state_dict["regressor"]
            self.is_fitted = state_dict["is_fitted"]
            self.feature_dim = state_dict.get("feature_dim", self.feature_dim)

            self.last_updated = model_state.last_updated
            self.update_count = model_state.update_count

            logger.info(f"Loaded state for {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading state for {self.model_name}: {e}")
            raise IncrementalLearnerError(f"State loading failed: {e}")


class LearningOrchestrator:
    """
    Orchestrates multiple incremental learning models for query adaptation.

    Manages classifier and regressor models, handles training data preparation,
    and provides unified learning interface.
    """

    def __init__(self):
        self.query_classifier = QueryClassifierLearner()
        self.preference_regressor = PreferenceRegressorLearner()
        self.models = {
            "query_classifier": self.query_classifier,
            "preference_regressor": self.preference_regressor,
        }

    async def learn_from_interaction(
        self, interaction: QueryInteraction
    ) -> Dict[str, Any]:
        """
        Learn from a single query interaction.

        Args:
            interaction: Query interaction data

        Returns:
            Learning results and metrics
        """
        results = {}

        try:
            # Prepare training data
            query_text = interaction.query

            # Learn query classification if intent is available
            if interaction.query_intent:
                await self.query_classifier.partial_fit(
                    [query_text], [interaction.query_intent]
                )
                results["intent_learning"] = "updated"

            # Learn preferences if feedback is available
            if interaction.feedback_type and interaction.feedback_value is not None:
                preference_score = self._feedback_to_score(
                    interaction.feedback_type, interaction.feedback_value
                )

                await self.preference_regressor.partial_fit(
                    [query_text], [preference_score]
                )
                results["preference_learning"] = "updated"

            results["timestamp"] = datetime.now(timezone.utc).isoformat()

        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")
            results["error"] = str(e)

        return results

    async def learn_from_interactions(
        self, interactions: List[QueryInteraction]
    ) -> Dict[str, Any]:
        """
        Learn from multiple query interactions in batch.

        Args:
            interactions: List of query interactions

        Returns:
            Batch learning results and metrics
        """
        if not interactions:
            return {"message": "No interactions to learn from"}

        # Separate data by learning type
        intent_queries, intent_labels = [], []
        preference_queries, preference_scores = [], []

        for interaction in interactions:
            query_text = interaction.query

            if interaction.query_intent:
                intent_queries.append(query_text)
                intent_labels.append(interaction.query_intent)

            if interaction.feedback_type and interaction.feedback_value is not None:
                preference_score = self._feedback_to_score(
                    interaction.feedback_type, interaction.feedback_value
                )
                preference_queries.append(query_text)
                preference_scores.append(preference_score)

        results = {
            "interactions_processed": len(interactions),
            "intent_samples": len(intent_queries),
            "preference_samples": len(preference_queries),
        }

        try:
            # Batch update intent classifier
            if intent_queries:
                await self.query_classifier.partial_fit(intent_queries, intent_labels)
                results["intent_learning"] = "updated"

            # Batch update preference regressor
            if preference_queries:
                await self.preference_regressor.partial_fit(
                    preference_queries, preference_scores
                )
                results["preference_learning"] = "updated"

            results["timestamp"] = datetime.now(timezone.utc).isoformat()

        except Exception as e:
            logger.error(f"Error in batch learning: {e}")
            results["error"] = str(e)

        return results

    async def predict_intent(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Predict intent for queries.

        Args:
            queries: List of query texts

        Returns:
            List of intent predictions with probabilities
        """
        if not queries:
            return []

        try:
            intents = await self.query_classifier.predict(queries)
            probabilities = await self.query_classifier.predict_proba(queries)

            results = []
            for i, query in enumerate(queries):
                result = {
                    "query": query,
                    "predicted_intent": intents[i] if i < len(intents) else None,
                    "probabilities": probabilities[i] if i < len(probabilities) else {},
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error predicting intent: {e}")
            return []

    async def predict_preferences(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Predict preference scores for queries.

        Args:
            queries: List of query texts

        Returns:
            List of preference predictions
        """
        if not queries:
            return []

        try:
            scores = await self.preference_regressor.predict(queries)

            results = []
            for i, query in enumerate(queries):
                result = {
                    "query": query,
                    "preference_score": float(scores[i]) if i < len(scores) else 0.5,
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error predicting preferences: {e}")
            return []

    def _feedback_to_score(
        self, feedback_type: FeedbackType, feedback_value: Any
    ) -> float:
        """Convert feedback to numerical preference score (0.0 to 1.0)."""
        if feedback_type == FeedbackType.POSITIVE:
            return 1.0
        elif feedback_type == FeedbackType.NEGATIVE:
            return 0.0
        elif feedback_type == FeedbackType.NEUTRAL:
            return 0.5
        elif feedback_type == FeedbackType.EXPLICIT_RATING:
            # Assume 1-5 scale, normalize to 0-1
            if isinstance(feedback_value, (int, float)):
                return max(0.0, min(1.0, (float(feedback_value) - 1) / 4))

        return 0.5  # Default neutral score

    def get_model_states(self) -> Dict[str, ModelState]:
        """Get serialized states of all models."""
        states = {}
        for name, model in self.models.items():
            try:
                states[name] = model.serialize_state()
            except Exception as e:
                logger.error(f"Error serializing model {name}: {e}")
        return states

    def load_model_states(self, states: Dict[str, ModelState]) -> Dict[str, bool]:
        """Load serialized states into models."""
        results = {}
        for name, state in states.items():
            if name in self.models:
                try:
                    self.models[name].load_state(state)
                    results[name] = True
                except Exception as e:
                    logger.error(f"Error loading model {name}: {e}")
                    results[name] = False
            else:
                logger.warning(f"Unknown model name: {name}")
                results[name] = False
        return results
