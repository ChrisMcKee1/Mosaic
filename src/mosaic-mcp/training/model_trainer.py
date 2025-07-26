"""
Model training pipeline for query intent classification.
Handles training, evaluation, and model management for the intent classifier.
"""

import json
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sentence_transformers import SentenceTransformer
import logging

from ..models.intent_models import (
    QueryIntentType,
    TrainingConfig,
    TrainingDataset,
    ModelMetrics,
    ModelInfo,
)
from .intent_training_data import generate_training_data


logger = logging.getLogger(__name__)


class IntentModelTrainer:
    """Trainer for query intent classification models."""

    def __init__(
        self, config: TrainingConfig, model_dir: str = "models/intent_classifier"
    ):
        """Initialize the model trainer."""
        self.config = config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sentence transformer
        logger.info(f"Loading sentence transformer: {config.model_name}")
        self.sentence_model = SentenceTransformer(config.model_name)

        # Initialize classifier
        self.classifier = RandomForestClassifier(
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            n_jobs=-1,  # Use all available cores
        )

        self.is_trained = False
        self.training_history: List[Dict[str, Any]] = []

    def prepare_features(self, queries: List[str]) -> np.ndarray:
        """Convert queries to feature vectors using sentence embeddings."""
        logger.info(f"Generating embeddings for {len(queries)} queries...")
        embeddings = self.sentence_model.encode(
            queries, batch_size=32, show_progress_bar=True, convert_to_numpy=True
        )
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def train(
        self, dataset: Optional[TrainingDataset] = None, save_model: bool = True
    ) -> ModelMetrics:
        """Train the intent classification model."""

        # Generate dataset if not provided
        if dataset is None:
            logger.info("Generating training dataset...")
            dataset = generate_training_data(
                samples_per_class=self.config.min_samples_per_class,
                random_seed=self.config.random_state,
            )

        logger.info(f"Training with {len(dataset)} samples")
        logger.info(f"Class distribution: {dataset.get_class_distribution()}")

        # Prepare features and labels
        X = self.prepare_features(dataset.queries)
        y = [intent.value for intent in dataset.intents]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")

        # Train the model
        logger.info("Training Random Forest classifier...")
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate the model
        metrics = self._evaluate_model(X_test, y_test, y)

        # Perform cross-validation
        cv_scores = self._cross_validate(X, y)
        metrics.accuracy = float(np.mean(cv_scores))

        logger.info("Model trained successfully!")
        logger.info(
            f"Cross-validation accuracy: {metrics.accuracy:.4f} (+/- {np.std(cv_scores) * 2:.4f})"
        )

        # Save model if requested
        if save_model:
            model_info = self._save_model(metrics, dataset)
            logger.info(f"Model saved to: {model_info.file_path}")

        # Update training history
        self.training_history.append(
            {
                "timestamp": datetime.utcnow(),
                "config": self.config.model_dump(),
                "dataset_size": len(dataset),
                "accuracy": metrics.accuracy,
                "metrics": metrics.model_dump(),
            }
        )

        return metrics

    def _evaluate_model(
        self, X_test: np.ndarray, y_test: List[str], y_all: List[str]
    ) -> ModelMetrics:
        """Evaluate the trained model."""

        # Predictions
        y_pred = self.classifier.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=list(QueryIntentType)
        )

        # Per-class metrics
        intent_classes = [intent.value for intent in QueryIntentType]
        precision_dict = dict(zip(intent_classes, precision))
        recall_dict = dict(zip(intent_classes, recall))
        f1_dict = dict(zip(intent_classes, f1))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=intent_classes)

        return ModelMetrics(
            accuracy=accuracy,
            precision=precision_dict,
            recall=recall_dict,
            f1_score=f1_dict,
            confusion_matrix=cm.tolist(),
            training_samples=len(y_all),
            model_version=self._generate_model_version(),
        )

    def _cross_validate(self, X: np.ndarray, y: List[str]) -> np.ndarray:
        """Perform cross-validation."""
        cv = StratifiedKFold(
            n_splits=self.config.cross_validation_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        scores = cross_val_score(
            self.classifier, X, y, cv=cv, scoring="accuracy", n_jobs=-1
        )

        return scores

    def _generate_model_version(self) -> str:
        """Generate a version string for the model."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"v1.0_{timestamp}"

    def _save_model(self, metrics: ModelMetrics, dataset: TrainingDataset) -> ModelInfo:
        """Save the trained model and metadata."""

        version = metrics.model_version
        model_path = self.model_dir / f"model_{version}.pkl"
        embedder_path = self.model_dir / f"embedder_{version}.pkl"
        config_path = self.model_dir / f"config_{version}.json"
        metrics_path = self.model_dir / f"metrics_{version}.json"

        # Save classifier
        joblib.dump(self.classifier, model_path)

        # Save sentence transformer model name (it's already cached by transformers)
        with open(embedder_path, "w") as f:
            json.dump({"model_name": self.config.model_name}, f)

        # Save config
        with open(config_path, "w") as f:
            json.dump(self.config.model_dump(), f, indent=2)

        # Save metrics
        with open(metrics_path, "w") as f:
            json.dump(metrics.model_dump(), f, indent=2, default=str)

        # Create model info
        model_info = ModelInfo(
            model_id=f"intent_classifier_{version}",
            version=version,
            config=self.config,
            metrics=metrics,
            file_path=str(model_path),
            is_active=True,
            description=f"Intent classifier trained on {dataset.metadata.get('total_samples', 'unknown')} samples",
        )

        # Save model info
        info_path = self.model_dir / f"info_{version}.json"
        with open(info_path, "w") as f:
            json.dump(model_info.model_dump(), f, indent=2, default=str)

        # Update active model symlink
        active_path = self.model_dir / "active"
        if active_path.is_symlink():
            active_path.unlink()

        # Create relative symlink to model file
        relative_model_path = model_path.relative_to(self.model_dir)
        active_path.symlink_to(relative_model_path)

        return model_info

    def load_model(self, model_path: Optional[str] = None) -> ModelInfo:
        """Load a trained model."""

        if model_path is None:
            # Load active model
            active_path = self.model_dir / "active"
            if not active_path.exists():
                raise FileNotFoundError("No active model found. Train a model first.")
            model_path = str(active_path.resolve())

        # Load classifier
        self.classifier = joblib.load(model_path)
        self.is_trained = True

        # Load model info
        model_file = Path(model_path)
        version = model_file.stem.split("_", 1)[1]  # Extract version from filename
        info_path = self.model_dir / f"info_{version}.json"

        if info_path.exists():
            with open(info_path, "r") as f:
                info_data = json.load(f)
                model_info = ModelInfo(**info_data)
        else:
            # Create minimal model info
            model_info = ModelInfo(
                model_id=f"intent_classifier_{version}",
                version=version,
                config=self.config,
                file_path=model_path,
                is_active=True,
                description="Loaded existing model",
            )

        logger.info(f"Loaded model: {model_info.model_id}")
        return model_info

    def predict_probabilities(self, queries: List[str]) -> List[Dict[str, float]]:
        """Predict intent probabilities for queries."""

        if not self.is_trained:
            raise ValueError("Model is not trained. Train or load a model first.")

        # Prepare features
        X = self.prepare_features(queries)

        # Get probabilities
        probabilities = self.classifier.predict_proba(X)
        classes = self.classifier.classes_

        # Convert to list of dictionaries
        results = []
        for prob_row in probabilities:
            prob_dict = dict(zip(classes, prob_row))
            results.append(prob_dict)

        return results

    def evaluate_on_dataset(self, dataset: TrainingDataset) -> ModelMetrics:
        """Evaluate the model on a given dataset."""

        if not self.is_trained:
            raise ValueError("Model is not trained. Train or load a model first.")

        # Prepare features
        X = self.prepare_features(dataset.queries)
        y = [intent.value for intent in dataset.intents]

        # Evaluate
        return self._evaluate_model(X, y, y)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""

        if not self.is_trained:
            raise ValueError("Model is not trained. Train or load a model first.")

        importance = self.classifier.feature_importances_

        # Since we use sentence embeddings, features are embedding dimensions
        feature_dict = {
            f"embedding_dim_{i}": float(imp) for i, imp in enumerate(importance)
        }

        # Get top features
        sorted_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)

        return dict(sorted_features[:20])  # Return top 20 features

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        return self.training_history

    def generate_classification_report(self, dataset: TrainingDataset) -> str:
        """Generate a detailed classification report."""

        if not self.is_trained:
            raise ValueError("Model is not trained. Train or load a model first.")

        # Prepare features
        X = self.prepare_features(dataset.queries)
        y = [intent.value for intent in dataset.intents]

        # Predictions
        y_pred = self.classifier.predict(X)

        # Generate report
        intent_classes = [intent.value for intent in QueryIntentType]
        report = classification_report(
            y, y_pred, labels=intent_classes, target_names=intent_classes, digits=4
        )

        return report


def train_intent_classifier(
    config: Optional[TrainingConfig] = None,
    dataset: Optional[TrainingDataset] = None,
    model_dir: str = "models/intent_classifier",
) -> Tuple[IntentModelTrainer, ModelMetrics]:
    """
    Convenience function to train an intent classifier.

    Args:
        config: Training configuration (uses defaults if None)
        dataset: Training dataset (generates if None)
        model_dir: Directory to save model files

    Returns:
        Tuple of (trainer, metrics)
    """

    if config is None:
        config = TrainingConfig()

    trainer = IntentModelTrainer(config, model_dir)
    metrics = trainer.train(dataset)

    return trainer, metrics


def load_intent_classifier(
    model_path: Optional[str] = None, model_dir: str = "models/intent_classifier"
) -> IntentModelTrainer:
    """
    Load a trained intent classifier.

    Args:
        model_path: Path to specific model file (uses active if None)
        model_dir: Directory containing model files

    Returns:
        Loaded trainer instance
    """

    # Use default config for loading (will be overridden by saved config)
    config = TrainingConfig()
    trainer = IntentModelTrainer(config, model_dir)
    trainer.load_model(model_path)

    return trainer
