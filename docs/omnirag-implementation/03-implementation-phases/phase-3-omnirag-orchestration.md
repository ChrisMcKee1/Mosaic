# Phase 3: OmniRAG Orchestration and Intent Detection

## 📋 Phase Overview

**Duration**: 4-6 weeks  
**Team Size**: 2-3 developers  
**Complexity**: Very High  
**Dependencies**: Phase 1 (RDF Infrastructure) and Phase 2 (SPARQL Integration) must be completed

This phase implements the complete OmniRAG pattern with intelligent intent detection, multi-source query orchestration, and advanced context aggregation. This is the culmination of the transformation from Basic RAG to OmniRAG.

## 🎯 Phase Objectives

- [ ] Implement intelligent query intent detection and classification
- [ ] Create multi-source query orchestration (Graph + Vector + Database)
- [ ] Build advanced context aggregation and fusion
- [ ] Develop query strategy routing and optimization
- [ ] Implement session-aware query learning and adaptation
- [ ] Create comprehensive result ranking and scoring
- [ ] Establish performance monitoring and optimization
- [ ] Complete end-to-end OmniRAG pipeline testing

## 📚 Pre-Implementation Research (Complete Before Starting)

### Required Reading (4-5 days)

1. **Intent Classification for Information Retrieval** (8-10 hours)

   - Research Paper: https://arxiv.org/abs/2010.12421
   - Focus: Multi-class intent classification, confidence scoring, query understanding
   - Practice: Build simple intent classifier with sample queries

2. **Multi-source Information Fusion** (6-8 hours)

   - Research Paper: https://dl.acm.org/doi/10.1145/3397271.3401075
   - Focus: Result fusion strategies, confidence weighting, redundancy elimination
   - Practice: Combine results from different sources with scoring

3. **Hybrid Graph-Vector Search Systems** (6-8 hours)

   - Research Paper: https://arxiv.org/abs/2106.06139
   - Focus: Parallel retrieval, result aggregation, performance optimization
   - Practice: Implement simple hybrid search with mock data

4. **CosmosAIGraph OmniRAG Implementation** (8-10 hours)

   - Study: Complete CosmosAIGraph codebase analysis
   - Links:
     - https://github.com/AzureCosmosDB/CosmosAIGraph/blob/main/impl/web_app/src/services/ai_service.py
     - https://github.com/AzureCosmosDB/CosmosAIGraph/blob/main/impl/web_app/src/services/graph_service.py
   - Focus: Strategy selection, orchestration patterns, result fusion
   - Practice: Adapt their orchestration patterns to your architecture

5. **Azure Machine Learning Reranking** (4-6 hours)
   - Link: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints
   - Focus: cross-encoder/ms-marco-MiniLM-L-12-v2 integration for semantic reranking
   - Practice: Set up reranking endpoint and test with sample results

### Research Validation Checkpoints

- [ ] Can classify query intents with high accuracy (>85%)
- [ ] Understand multi-source result fusion strategies
- [ ] Can implement parallel retrieval coordination
- [ ] Familiar with confidence scoring and ranking algorithms
- [ ] Have working prototype of intent-based query routing

## 🛠️ Implementation Steps

### Step 1: Intent Detection Infrastructure (Week 1, Days 1-3)

#### 1.1 Install Additional Dependencies

```bash
cd src/mosaic-mcp/
pip install scikit-learn==1.3.2
pip install transformers==4.36.2
pip install sentence-transformers==2.2.2
pip install torch==2.1.2
pip install numpy==1.24.3

# Update requirements.txt
echo "scikit-learn==1.3.2" >> requirements.txt
echo "transformers==4.36.2" >> requirements.txt
echo "sentence-transformers==2.2.2" >> requirements.txt
echo "torch==2.1.2" >> requirements.txt
echo "numpy==1.24.3" >> requirements.txt
```

#### 1.2 Create Intent Detection Structure

```bash
mkdir -p src/mosaic-mcp/intent
mkdir -p src/mosaic-mcp/orchestration
mkdir -p src/mosaic-mcp/models
mkdir -p src/mosaic-mcp/tests/intent

touch src/mosaic-mcp/intent/__init__.py
touch src/mosaic-mcp/intent/classifier.py
touch src/mosaic-mcp/intent/strategy_router.py
touch src/mosaic-mcp/intent/confidence_scorer.py
touch src/mosaic-mcp/intent/query_analyzer.py

touch src/mosaic-mcp/orchestration/__init__.py
touch src/mosaic-mcp/orchestration/omnirag_orchestrator.py
touch src/mosaic-mcp/orchestration/context_aggregator.py
touch src/mosaic-mcp/orchestration/result_ranker.py
```

#### 1.3 Environment Configuration Updates

```bash
# Add to .env files
echo "MOSAIC_INTENT_MODEL_PATH=./models/intent_classifier" >> .env
echo "MOSAIC_INTENT_CONFIDENCE_THRESHOLD=0.7" >> .env
echo "MOSAIC_RERANK_ENDPOINT_URL=https://your-ml-endpoint.azureml.net" >> .env
echo "MOSAIC_PARALLEL_RETRIEVAL_ENABLED=true" >> .env
echo "MOSAIC_CONTEXT_FUSION_STRATEGY=weighted_combination" >> .env
echo "MOSAIC_MAX_CONTEXT_SOURCES=3" >> .env
```

### Step 2: Query Intent Classification (Week 1, Days 4-5 & Week 2, Days 1-2)

#### 2.1 Create `classifier.py`

**Research Reference**: https://arxiv.org/abs/2010.12421

```python
import os
import json
import logging
import pickle
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class QueryIntent:
    """
    Represents a classified query intent with confidence scores
    """
    def __init__(
        self,
        primary_strategy: str,
        confidence: float,
        secondary_strategies: List[Tuple[str, float]] = None,
        query_features: Dict[str, Any] = None
    ):
        self.primary_strategy = primary_strategy
        self.confidence = confidence
        self.secondary_strategies = secondary_strategies or []
        self.query_features = query_features or {}

        # Strategy requirements
        self.requires_graph = primary_strategy in ["graph_traversal", "hybrid_multi_source"]
        self.requires_vector = primary_strategy in ["vector_similarity", "hybrid_multi_source"]
        self.requires_database = primary_strategy in ["database_lookup", "hybrid_multi_source"]

    def should_use_hybrid(self, threshold: float = 0.8) -> bool:
        """
        Determine if hybrid approach should be used based on confidence
        """
        return self.confidence < threshold or self.primary_strategy == "hybrid_multi_source"

class QueryIntentClassifier:
    """
    Classifies user queries into optimal retrieval strategies using ML
    """

    def __init__(self):
        self.model_path = Path(os.getenv("MOSAIC_INTENT_MODEL_PATH", "./models/intent_classifier"))
        self.confidence_threshold = float(os.getenv("MOSAIC_INTENT_CONFIDENCE_THRESHOLD", "0.7"))

        # Models
        self.tfidf_vectorizer = None
        self.classifier = None
        self.label_encoder = None
        self.sentence_transformer = None

        # Training data and features
        self.strategies = [
            "graph_traversal",      # Relationship queries, hierarchy, dependencies
            "vector_similarity",    # Similarity searches, "find similar", examples
            "database_lookup",      # Direct entity lookup, details, definitions
            "hybrid_multi_source"   # Complex queries requiring multiple sources
        ]

        self.is_trained = False

    async def initialize(self):
        """
        Initialize intent classifier with models
        """
        logger.info("Initializing Query Intent Classifier...")

        try:
            # Initialize sentence transformer for semantic features
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

            # Try to load existing model
            if await self._load_model():
                logger.info("Loaded existing intent classification model")
            else:
                # Train new model with default training data
                await self._train_default_model()
                logger.info("Trained new intent classification model")

            self.is_trained = True
            logger.info("Query Intent Classifier initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize intent classifier: {e}")
            raise

    async def classify_query(self, query: str) -> QueryIntent:
        """
        Classify a query and return intent with confidence scores
        """
        if not self.is_trained:
            await self.initialize()

        try:
            # Extract features from query
            features = self._extract_features(query)

            # Get predictions from all models
            ml_prediction = self._predict_with_ml(query)
            rule_prediction = self._predict_with_rules(query, features)

            # Combine predictions (weighted ensemble)
            final_prediction = self._combine_predictions(ml_prediction, rule_prediction)

            # Build query intent object
            intent = QueryIntent(
                primary_strategy=final_prediction["strategy"],
                confidence=final_prediction["confidence"],
                secondary_strategies=final_prediction.get("alternatives", []),
                query_features=features
            )

            logger.debug(f"Classified query '{query[:50]}...' as {intent.primary_strategy} (confidence: {intent.confidence:.3f})")
            return intent

        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            # Fallback to hybrid strategy
            return QueryIntent(
                primary_strategy="hybrid_multi_source",
                confidence=0.5,
                query_features={"error": str(e)}
            )

    def _extract_features(self, query: str) -> Dict[str, Any]:
        """
        Extract features from query for classification
        """
        query_lower = query.lower().strip()

        features = {
            "length": len(query),
            "word_count": len(query.split()),
            "has_question_words": any(word in query_lower for word in ["what", "how", "why", "where", "when", "which", "who"]),
            "has_relationship_terms": any(term in query_lower for term in ["depends", "calls", "inherits", "imports", "uses", "extends"]),
            "has_similarity_terms": any(term in query_lower for term in ["similar", "like", "related", "comparable", "examples"]),
            "has_lookup_terms": any(term in query_lower for term in ["what is", "tell me", "show me", "details", "information"]),
            "has_complex_terms": any(term in query_lower for term in ["analyze", "comprehensive", "complete", "full context", "everything"]),
            "has_code_terms": any(term in query_lower for term in ["function", "class", "method", "variable", "module", "library"]),
            "has_path_references": "/" in query or "\\" in query or ".py" in query_lower,
            "semantic_embedding": self.sentence_transformer.encode(query).tolist() if self.sentence_transformer else None
        }

        return features

    def _predict_with_ml(self, query: str) -> Dict[str, Any]:
        """
        Predict intent using ML model
        """
        if not self.classifier or not self.tfidf_vectorizer or not self.label_encoder:
            return {"strategy": "hybrid_multi_source", "confidence": 0.5}

        try:
            # Vectorize query
            query_vector = self.tfidf_vectorizer.transform([query])

            # Get prediction probabilities
            probabilities = self.classifier.predict_proba(query_vector)[0]
            predicted_class = self.classifier.predict(query_vector)[0]

            # Convert back to strategy names
            strategy = self.label_encoder.inverse_transform([predicted_class])[0]
            confidence = max(probabilities)

            # Get alternative strategies
            alternatives = []
            for i, prob in enumerate(probabilities):
                if i != predicted_class and prob > 0.1:  # Alternative threshold
                    alt_strategy = self.label_encoder.inverse_transform([i])[0]
                    alternatives.append((alt_strategy, prob))

            alternatives.sort(key=lambda x: x[1], reverse=True)

            return {
                "strategy": strategy,
                "confidence": confidence,
                "alternatives": alternatives[:2]  # Top 2 alternatives
            }

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return {"strategy": "hybrid_multi_source", "confidence": 0.5}

    def _predict_with_rules(self, query: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict intent using rule-based approach
        """
        query_lower = query.lower().strip()
        confidence = 0.9  # High confidence for rule-based matches

        # Graph traversal patterns
        if features["has_relationship_terms"]:
            if any(term in query_lower for term in ["depends on", "dependency", "dependencies"]):
                return {"strategy": "graph_traversal", "confidence": confidence}
            elif any(term in query_lower for term in ["calls", "calling", "called by"]):
                return {"strategy": "graph_traversal", "confidence": confidence}
            elif any(term in query_lower for term in ["inherits", "inheritance", "extends", "implements"]):
                return {"strategy": "graph_traversal", "confidence": confidence}

        # Vector similarity patterns
        if features["has_similarity_terms"]:
            if any(term in query_lower for term in ["similar to", "like", "related to", "comparable"]):
                return {"strategy": "vector_similarity", "confidence": confidence}
            elif any(term in query_lower for term in ["find functions", "search for", "examples of"]):
                return {"strategy": "vector_similarity", "confidence": confidence}

        # Database lookup patterns
        if features["has_lookup_terms"]:
            if any(term in query_lower for term in ["what is", "tell me about", "show me details"]):
                return {"strategy": "database_lookup", "confidence": confidence}
            elif query_lower.startswith("describe ") or query_lower.startswith("explain "):
                return {"strategy": "database_lookup", "confidence": confidence}

        # Complex/hybrid patterns
        if features["has_complex_terms"]:
            return {"strategy": "hybrid_multi_source", "confidence": confidence}

        # Default to vector similarity for general queries
        return {"strategy": "vector_similarity", "confidence": 0.6}

    def _combine_predictions(self, ml_pred: Dict, rule_pred: Dict) -> Dict[str, Any]:
        """
        Combine ML and rule-based predictions using weighted ensemble
        """
        ml_weight = 0.7  # Prefer ML for general cases
        rule_weight = 0.3

        # If rule-based prediction has high confidence, prefer it
        if rule_pred["confidence"] > 0.85:
            rule_weight = 0.8
            ml_weight = 0.2

        # Calculate weighted confidence
        if ml_pred["strategy"] == rule_pred["strategy"]:
            # Both agree - boost confidence
            final_confidence = min(1.0, ml_pred["confidence"] * ml_weight + rule_pred["confidence"] * rule_weight + 0.1)
            final_strategy = ml_pred["strategy"]
        else:
            # Disagreement - choose higher confidence or use rule-based for high-confidence rules
            if rule_pred["confidence"] > 0.85 and rule_pred["confidence"] > ml_pred["confidence"]:
                final_strategy = rule_pred["strategy"]
                final_confidence = rule_pred["confidence"] * 0.9  # Slight penalty for disagreement
            else:
                final_strategy = ml_pred["strategy"]
                final_confidence = ml_pred["confidence"] * 0.9  # Slight penalty for disagreement

        return {
            "strategy": final_strategy,
            "confidence": final_confidence,
            "alternatives": ml_pred.get("alternatives", [])
        }

    async def _train_default_model(self):
        """
        Train model with default training data
        """
        logger.info("Training default intent classification model...")

        # Load default training data
        training_data = self._get_default_training_data()

        # Prepare features
        queries = [item["query"] for item in training_data]
        labels = [item["strategy"] for item in training_data]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            queries, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Initialize and fit vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        X_train_vectorized = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_vectorized = self.tfidf_vectorizer.transform(X_test)

        # Initialize and fit label encoder
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        self.classifier.fit(X_train_vectorized, y_train_encoded)

        # Evaluate model
        y_pred = self.classifier.predict(X_test_vectorized)
        accuracy = accuracy_score(y_test_encoded, y_pred)

        logger.info(f"Model trained with accuracy: {accuracy:.3f}")

        # Save model
        await self._save_model()

    def _get_default_training_data(self) -> List[Dict[str, str]]:
        """
        Get default training data for intent classification
        """
        return [
            # Graph traversal queries
            {"query": "What are the dependencies of Flask?", "strategy": "graph_traversal"},
            {"query": "What functions call authenticate_user?", "strategy": "graph_traversal"},
            {"query": "Show me the inheritance hierarchy for Exception classes", "strategy": "graph_traversal"},
            {"query": "What libraries depend on requests?", "strategy": "graph_traversal"},
            {"query": "Which functions are called by main()?", "strategy": "graph_traversal"},
            {"query": "What does the UserService class inherit from?", "strategy": "graph_traversal"},
            {"query": "Map the dependency chain for numpy", "strategy": "graph_traversal"},
            {"query": "Show relationships between modules in this package", "strategy": "graph_traversal"},
            {"query": "What imports are used by the database module?", "strategy": "graph_traversal"},
            {"query": "Trace function calls from login to database", "strategy": "graph_traversal"},

            # Vector similarity queries
            {"query": "Find functions similar to authenticate_user", "strategy": "vector_similarity"},
            {"query": "Show me functions like process_payment", "strategy": "vector_similarity"},
            {"query": "Find similar authentication methods", "strategy": "vector_similarity"},
            {"query": "Search for functions related to user validation", "strategy": "vector_similarity"},
            {"query": "Find comparable error handling functions", "strategy": "vector_similarity"},
            {"query": "Show functions that are similar to this one", "strategy": "vector_similarity"},
            {"query": "Find related utility functions", "strategy": "vector_similarity"},
            {"query": "Search for equivalent methods in other classes", "strategy": "vector_similarity"},
            {"query": "Find functions with similar functionality", "strategy": "vector_similarity"},
            {"query": "Show me examples of async functions", "strategy": "vector_similarity"},

            # Database lookup queries
            {"query": "What is the Flask library?", "strategy": "database_lookup"},
            {"query": "Tell me about the requests module", "strategy": "database_lookup"},
            {"query": "Show me details of the User class", "strategy": "database_lookup"},
            {"query": "Describe the authentication function", "strategy": "database_lookup"},
            {"query": "What is numpy used for?", "strategy": "database_lookup"},
            {"query": "Explain the purpose of this module", "strategy": "database_lookup"},
            {"query": "Show information about the config file", "strategy": "database_lookup"},
            {"query": "What does this function do?", "strategy": "database_lookup"},
            {"query": "Give me details about the API endpoint", "strategy": "database_lookup"},
            {"query": "Describe the database schema", "strategy": "database_lookup"},

            # Hybrid multi-source queries
            {"query": "Analyze the complete authentication flow", "strategy": "hybrid_multi_source"},
            {"query": "Give me comprehensive information about Flask", "strategy": "hybrid_multi_source"},
            {"query": "Explain everything about the user management system", "strategy": "hybrid_multi_source"},
            {"query": "Analyze the full dependency tree and similar libraries", "strategy": "hybrid_multi_source"},
            {"query": "Provide complete context about the payment processing", "strategy": "hybrid_multi_source"},
            {"query": "Show me everything related to database operations", "strategy": "hybrid_multi_source"},
            {"query": "Comprehensively analyze the error handling patterns", "strategy": "hybrid_multi_source"},
            {"query": "Give me full context on security implementations", "strategy": "hybrid_multi_source"},
            {"query": "Analyze all aspects of the REST API design", "strategy": "hybrid_multi_source"},
            {"query": "Provide thorough analysis of the caching system", "strategy": "hybrid_multi_source"}
        ]

    async def _save_model(self):
        """
        Save trained model to disk
        """
        try:
            self.model_path.mkdir(parents=True, exist_ok=True)

            # Save components
            with open(self.model_path / "classifier.pkl", "wb") as f:
                pickle.dump(self.classifier, f)

            with open(self.model_path / "vectorizer.pkl", "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)

            with open(self.model_path / "label_encoder.pkl", "wb") as f:
                pickle.dump(self.label_encoder, f)

            # Save metadata
            metadata = {
                "strategies": self.strategies,
                "confidence_threshold": self.confidence_threshold,
                "model_version": "1.0"
            }

            with open(self.model_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved to {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    async def _load_model(self) -> bool:
        """
        Load existing model from disk
        """
        try:
            if not self.model_path.exists():
                return False

            # Check if all required files exist
            required_files = ["classifier.pkl", "vectorizer.pkl", "label_encoder.pkl", "metadata.json"]
            if not all((self.model_path / f).exists() for f in required_files):
                return False

            # Load components
            with open(self.model_path / "classifier.pkl", "rb") as f:
                self.classifier = pickle.load(f)

            with open(self.model_path / "vectorizer.pkl", "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)

            with open(self.model_path / "label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)

            # Load metadata
            with open(self.model_path / "metadata.json", "r") as f:
                metadata = json.load(f)
                self.strategies = metadata.get("strategies", self.strategies)

            logger.info(f"Model loaded from {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    async def retrain_with_feedback(self, query: str, correct_strategy: str):
        """
        Retrain model with user feedback (for future enhancement)
        """
        # TODO: Implement incremental learning with user feedback
        logger.info(f"Feedback received: '{query}' should be '{correct_strategy}'")

# Global instance
intent_classifier: Optional[QueryIntentClassifier] = None

async def get_intent_classifier() -> QueryIntentClassifier:
    """
    Get or create global intent classifier instance
    """
    global intent_classifier
    if intent_classifier is None:
        intent_classifier = QueryIntentClassifier()
        await intent_classifier.initialize()
    return intent_classifier
```

### Step 3: Multi-Source Orchestration Engine (Week 2, Days 3-5 & Week 3, Days 1-2)

#### 3.1 Create `omnirag_orchestrator.py`

**Research Reference**: https://dl.acm.org/doi/10.1145/3397271.3401075

```python
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from ..intent.classifier import QueryIntent, get_intent_classifier
from ..plugins.retrieval_plugin import get_retrieval_plugin
from ..plugins.graph_plugin import get_graph_plugin
from ..sparql.query_executor import get_sparql_executor
from .context_aggregator import get_context_aggregator
from .result_ranker import get_result_ranker

logger = logging.getLogger(__name__)

class RetrievalStrategy:
    """
    Base class for retrieval strategies
    """
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.execution_time_ms = 0
        self.result_count = 0
        self.confidence = 0.0

    async def execute(self, query: str, intent: QueryIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the retrieval strategy
        """
        raise NotImplementedError

class GraphRetrievalStrategy(RetrievalStrategy):
    """
    Graph-based retrieval using SPARQL queries
    """
    def __init__(self):
        super().__init__("graph", weight=1.0)
        self.graph_plugin = None

    async def initialize(self):
        self.graph_plugin = await get_graph_plugin()

    async def execute(self, query: str, intent: QueryIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now()

        try:
            # Use graph plugin for relationship queries
            result = await self.graph_plugin.query_code_graph(
                query=query,
                query_type="natural_language",
                limit=context.get("limit", 20),
                include_context=True
            )

            self.execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            if result.get("status") == "success":
                self.result_count = len(result.get("results", []))
                self.confidence = result.get("metadata", {}).get("confidence", 0.8)

                return {
                    "status": "success",
                    "strategy": "graph",
                    "results": result["results"],
                    "metadata": result.get("metadata", {}),
                    "execution_time_ms": self.execution_time_ms
                }
            else:
                return {
                    "status": "error",
                    "strategy": "graph",
                    "error": result.get("error", "Graph query failed"),
                    "execution_time_ms": self.execution_time_ms
                }

        except Exception as e:
            self.execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Graph retrieval failed: {e}")
            return {
                "status": "error",
                "strategy": "graph",
                "error": str(e),
                "execution_time_ms": self.execution_time_ms
            }

class VectorRetrievalStrategy(RetrievalStrategy):
    """
    Vector-based retrieval using similarity search
    """
    def __init__(self):
        super().__init__("vector", weight=1.0)
        self.retrieval_plugin = None

    async def initialize(self):
        self.retrieval_plugin = await get_retrieval_plugin()

    async def execute(self, query: str, intent: QueryIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now()

        try:
            # Use existing retrieval plugin for vector search
            result = await self.retrieval_plugin.hybrid_search(
                query=query,
                limit=context.get("limit", 20),
                include_embeddings=False
            )

            self.execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            if result.get("status") == "success":
                self.result_count = len(result.get("documents", []))
                self.confidence = 0.9  # High confidence for vector search

                return {
                    "status": "success",
                    "strategy": "vector",
                    "results": result["documents"],
                    "metadata": {
                        "search_type": "hybrid_vector",
                        "confidence": self.confidence
                    },
                    "execution_time_ms": self.execution_time_ms
                }
            else:
                return {
                    "status": "error",
                    "strategy": "vector",
                    "error": result.get("error", "Vector search failed"),
                    "execution_time_ms": self.execution_time_ms
                }

        except Exception as e:
            self.execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Vector retrieval failed: {e}")
            return {
                "status": "error",
                "strategy": "vector",
                "error": str(e),
                "execution_time_ms": self.execution_time_ms
            }

class DatabaseRetrievalStrategy(RetrievalStrategy):
    """
    Direct database retrieval for entity lookup
    """
    def __init__(self):
        super().__init__("database", weight=1.0)
        self.cosmos_client = None

    async def initialize(self, cosmos_client):
        self.cosmos_client = cosmos_client

    async def execute(self, query: str, intent: QueryIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now()

        try:
            # Extract entity names from query for direct lookup
            entity_names = self._extract_entity_names(query)

            if not entity_names:
                return {
                    "status": "error",
                    "strategy": "database",
                    "error": "No entities found in query for database lookup"
                }

            results = []
            for entity_name in entity_names:
                # Query Cosmos DB directly for entity information
                cosmos_query = f"""
                SELECT * FROM c
                WHERE CONTAINS(LOWER(c.name), LOWER('{entity_name}'))
                   OR CONTAINS(LOWER(c.content), LOWER('{entity_name}'))
                ORDER BY c.timestamp DESC
                """

                async for item in self.cosmos_client.query_items(
                    query=cosmos_query,
                    enable_cross_partition_query=True
                ):
                    results.append(item)
                    if len(results) >= context.get("limit", 10):
                        break

            self.execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.result_count = len(results)
            self.confidence = 0.85 if results else 0.0

            return {
                "status": "success",
                "strategy": "database",
                "results": results,
                "metadata": {
                    "entities_searched": entity_names,
                    "confidence": self.confidence
                },
                "execution_time_ms": self.execution_time_ms
            }

        except Exception as e:
            self.execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"Database retrieval failed: {e}")
            return {
                "status": "error",
                "strategy": "database",
                "error": str(e),
                "execution_time_ms": self.execution_time_ms
            }

    def _extract_entity_names(self, query: str) -> List[str]:
        """
        Extract potential entity names from query
        """
        # Simple entity extraction - enhance with NER if needed
        import re

        entities = []

        # Look for quoted strings
        quoted_entities = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_entities)

        # Look for capitalized words (potential class/function names)
        words = query.split()
        for word in words:
            # Skip common words and look for potential identifiers
            if (word[0].isupper() and
                len(word) > 2 and
                word not in ["What", "Where", "When", "How", "Why", "Which", "Who", "The", "This", "That"]):
                entities.append(word)

        # Look for snake_case and camelCase identifiers
        identifier_pattern = r'\b[a-z_][a-z0-9_]*[a-z0-9]\b|\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b'
        identifiers = re.findall(identifier_pattern, query)
        entities.extend(identifiers)

        return list(set(entities))  # Remove duplicates

class OmniRAGOrchestrator:
    """
    Orchestrates multiple retrieval strategies based on query intent
    """

    def __init__(self):
        self.intent_classifier = None
        self.context_aggregator = None
        self.result_ranker = None

        # Retrieval strategies
        self.strategies = {
            "graph": GraphRetrievalStrategy(),
            "vector": VectorRetrievalStrategy(),
            "database": DatabaseRetrievalStrategy()
        }

        # Configuration
        self.parallel_enabled = os.getenv("MOSAIC_PARALLEL_RETRIEVAL_ENABLED", "true").lower() == "true"
        self.max_sources = int(os.getenv("MOSAIC_MAX_CONTEXT_SOURCES", "3"))
        self.timeout_seconds = 30

    async def initialize(self, cosmos_client):
        """
        Initialize orchestrator with dependencies
        """
        logger.info("Initializing OmniRAG Orchestrator...")

        try:
            # Initialize core components
            self.intent_classifier = await get_intent_classifier()
            self.context_aggregator = await get_context_aggregator()
            self.result_ranker = await get_result_ranker()

            # Initialize strategies
            await self.strategies["graph"].initialize()
            await self.strategies["vector"].initialize()
            await self.strategies["database"].initialize(cosmos_client)

            logger.info("OmniRAG Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OmniRAG orchestrator: {e}")
            raise

    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process query through complete OmniRAG pipeline
        """
        start_time = datetime.now()
        context = context or {}

        try:
            logger.info(f"Processing OmniRAG query: {query[:100]}...")

            # Step 1: Classify query intent
            intent = await self.intent_classifier.classify_query(query)
            logger.debug(f"Query classified as: {intent.primary_strategy} (confidence: {intent.confidence:.3f})")

            # Step 2: Determine retrieval strategies to use
            strategies_to_use = self._select_strategies(intent, context)

            # Step 3: Execute retrieval strategies
            if self.parallel_enabled and len(strategies_to_use) > 1:
                strategy_results = await self._execute_parallel_retrieval(query, intent, strategies_to_use, context)
            else:
                strategy_results = await self._execute_sequential_retrieval(query, intent, strategies_to_use, context)

            # Step 4: Aggregate and rank results
            aggregated_context = await self.context_aggregator.aggregate_results(
                strategy_results, intent, context
            )

            ranked_results = await self.result_ranker.rank_results(
                aggregated_context, query, intent
            )

            # Step 5: Build final response
            total_time = int((datetime.now() - start_time).total_seconds() * 1000)

            response = {
                "status": "success",
                "query": query,
                "intent": {
                    "primary_strategy": intent.primary_strategy,
                    "confidence": intent.confidence,
                    "requires_graph": intent.requires_graph,
                    "requires_vector": intent.requires_vector,
                    "requires_database": intent.requires_database
                },
                "strategies_used": [s.name for s in strategies_to_use],
                "results": ranked_results,
                "metadata": {
                    "total_execution_time_ms": total_time,
                    "strategy_execution_times": {s.name: s.execution_time_ms for s in strategies_to_use},
                    "total_results": sum(s.result_count for s in strategies_to_use),
                    "session_id": session_id
                }
            }

            logger.info(f"OmniRAG query processed successfully in {total_time}ms")
            return response

        except Exception as e:
            total_time = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"OmniRAG query processing failed: {e}")

            return {
                "status": "error",
                "query": query,
                "error": str(e),
                "metadata": {
                    "total_execution_time_ms": total_time,
                    "session_id": session_id
                }
            }

    def _select_strategies(self, intent: QueryIntent, context: Dict[str, Any]) -> List[RetrievalStrategy]:
        """
        Select which retrieval strategies to use based on intent
        """
        strategies = []

        # Always include primary strategy
        primary_strategy = intent.primary_strategy

        if primary_strategy == "graph_traversal" and intent.requires_graph:
            strategies.append(self.strategies["graph"])
        elif primary_strategy == "vector_similarity" and intent.requires_vector:
            strategies.append(self.strategies["vector"])
        elif primary_strategy == "database_lookup" and intent.requires_database:
            strategies.append(self.strategies["database"])
        elif primary_strategy == "hybrid_multi_source":
            # Use all strategies for hybrid approach
            if intent.requires_graph:
                strategies.append(self.strategies["graph"])
            if intent.requires_vector:
                strategies.append(self.strategies["vector"])
            if intent.requires_database:
                strategies.append(self.strategies["database"])

        # Add secondary strategies if confidence is low or explicitly requested
        if intent.confidence < 0.8 or context.get("use_multiple_sources", False):
            for strategy_name, confidence in intent.secondary_strategies:
                if confidence > 0.3 and len(strategies) < self.max_sources:
                    strategy_obj = self.strategies.get(strategy_name.replace("_", ""))
                    if strategy_obj and strategy_obj not in strategies:
                        strategies.append(strategy_obj)

        # Ensure at least one strategy
        if not strategies:
            strategies.append(self.strategies["vector"])  # Default fallback

        return strategies

    async def _execute_parallel_retrieval(
        self,
        query: str,
        intent: QueryIntent,
        strategies: List[RetrievalStrategy],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple retrieval strategies in parallel
        """
        logger.debug(f"Executing {len(strategies)} strategies in parallel")

        # Create tasks for parallel execution
        tasks = []
        for strategy in strategies:
            task = asyncio.create_task(
                strategy.execute(query, intent, context),
                name=f"strategy_{strategy.name}"
            )
            tasks.append(task)

        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout_seconds
            )

            # Process results and handle exceptions
            strategy_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Strategy {strategies[i].name} failed: {result}")
                    strategy_results.append({
                        "status": "error",
                        "strategy": strategies[i].name,
                        "error": str(result)
                    })
                else:
                    strategy_results.append(result)

            return strategy_results

        except asyncio.TimeoutError:
            logger.error(f"Parallel retrieval timeout after {self.timeout_seconds}s")
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

            return [{
                "status": "timeout",
                "error": f"Retrieval timeout after {self.timeout_seconds}s"
            }]

    async def _execute_sequential_retrieval(
        self,
        query: str,
        intent: QueryIntent,
        strategies: List[RetrievalStrategy],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute retrieval strategies sequentially
        """
        logger.debug(f"Executing {len(strategies)} strategies sequentially")

        results = []
        for strategy in strategies:
            try:
                result = await strategy.execute(query, intent, context)
                results.append(result)

                # Early termination if we have enough good results
                if (result.get("status") == "success" and
                    len(result.get("results", [])) >= context.get("limit", 10) and
                    strategy.confidence > 0.8):
                    logger.debug(f"Early termination after {strategy.name} strategy")
                    break

            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed: {e}")
                results.append({
                    "status": "error",
                    "strategy": strategy.name,
                    "error": str(e)
                })

        return results

    async def get_orchestrator_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about orchestrator performance
        """
        return {
            "strategies_available": list(self.strategies.keys()),
            "parallel_enabled": self.parallel_enabled,
            "max_sources": self.max_sources,
            "timeout_seconds": self.timeout_seconds,
            "strategies_performance": {
                name: {
                    "last_execution_time_ms": strategy.execution_time_ms,
                    "last_result_count": strategy.result_count,
                    "last_confidence": strategy.confidence
                }
                for name, strategy in self.strategies.items()
            }
        }

# Global instance
omnirag_orchestrator: Optional[OmniRAGOrchestrator] = None

async def get_omnirag_orchestrator() -> OmniRAGOrchestrator:
    """
    Get or create global OmniRAG orchestrator instance
    """
    global omnirag_orchestrator
    if omnirag_orchestrator is None:
        omnirag_orchestrator = OmniRAGOrchestrator()
    return omnirag_orchestrator
```

### Step 4: Context Aggregation and Result Fusion (Week 3, Days 3-4)

#### 4.1 Create `context_aggregator.py`

**Research Reference**: https://arxiv.org/abs/2106.06139

```python
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import os
from ..intent.classifier import QueryIntent

logger = logging.getLogger(__name__)

class ContextAggregator:
    """
    Aggregates and fuses results from multiple retrieval strategies
    """

    def __init__(self):
        self.fusion_strategy = os.getenv("MOSAIC_CONTEXT_FUSION_STRATEGY", "weighted_combination")
        self.max_context_length = 4000  # Maximum context length for LLM
        self.deduplication_threshold = 0.8  # Similarity threshold for deduplication

    async def initialize(self):
        """
        Initialize context aggregator
        """
        logger.info("Initializing Context Aggregator...")

        # Initialize any required models or services
        # For now, we'll use simple rule-based aggregation

        logger.info("Context Aggregator initialized successfully")

    async def aggregate_results(
        self,
        strategy_results: List[Dict[str, Any]],
        intent: QueryIntent,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple strategies into unified context
        """
        try:
            logger.debug(f"Aggregating results from {len(strategy_results)} strategies")

            # Filter successful results
            successful_results = [r for r in strategy_results if r.get("status") == "success"]

            if not successful_results:
                return {
                    "status": "error",
                    "error": "No successful results from any strategy",
                    "raw_results": strategy_results
                }

            # Apply fusion strategy
            if self.fusion_strategy == "weighted_combination":
                aggregated = await self._weighted_combination_fusion(successful_results, intent)
            elif self.fusion_strategy == "ranked_merge":
                aggregated = await self._ranked_merge_fusion(successful_results, intent)
            elif self.fusion_strategy == "best_strategy_only":
                aggregated = await self._best_strategy_fusion(successful_results, intent)
            else:
                aggregated = await self._simple_concatenation_fusion(successful_results, intent)

            # Deduplicate results
            deduplicated = await self._deduplicate_results(aggregated)

            # Limit context length
            final_context = await self._limit_context_length(deduplicated)

            logger.debug(f"Aggregated {len(final_context.get('results', []))} final results")
            return final_context

        except Exception as e:
            logger.error(f"Context aggregation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "raw_results": strategy_results
            }

    async def _weighted_combination_fusion(
        self,
        results: List[Dict[str, Any]],
        intent: QueryIntent
    ) -> Dict[str, Any]:
        """
        Combine results using weighted strategy based on intent and confidence
        """
        # Define weights based on query intent
        strategy_weights = self._get_strategy_weights(intent)

        weighted_results = []

        for result in results:
            strategy_name = result.get("strategy", "unknown")
            weight = strategy_weights.get(strategy_name, 0.5)

            # Weight results from this strategy
            strategy_results = result.get("results", [])
            for item in strategy_results:
                weighted_item = dict(item)
                weighted_item["_strategy"] = strategy_name
                weighted_item["_weight"] = weight
                weighted_item["_original_score"] = item.get("score", 1.0)
                weighted_item["_weighted_score"] = item.get("score", 1.0) * weight
                weighted_results.append(weighted_item)

        # Sort by weighted score
        weighted_results.sort(key=lambda x: x.get("_weighted_score", 0), reverse=True)

        return {
            "status": "success",
            "fusion_strategy": "weighted_combination",
            "results": weighted_results,
            "metadata": {
                "strategy_weights": strategy_weights,
                "total_items": len(weighted_results)
            }
        }

    async def _ranked_merge_fusion(
        self,
        results: List[Dict[str, Any]],
        intent: QueryIntent
    ) -> Dict[str, Any]:
        """
        Merge results by interleaving top results from each strategy
        """
        # Prepare results from each strategy
        strategy_results = {}
        for result in results:
            strategy_name = result.get("strategy", "unknown")
            items = result.get("results", [])

            # Sort items by score if available
            sorted_items = sorted(items, key=lambda x: x.get("score", 0), reverse=True)
            strategy_results[strategy_name] = sorted_items

        # Round-robin merge
        merged_results = []
        max_items_per_strategy = 5  # Limit items per strategy in merge

        for position in range(max_items_per_strategy):
            for strategy_name, items in strategy_results.items():
                if position < len(items):
                    item = dict(items[position])
                    item["_strategy"] = strategy_name
                    item["_merge_position"] = len(merged_results)
                    merged_results.append(item)

        return {
            "status": "success",
            "fusion_strategy": "ranked_merge",
            "results": merged_results,
            "metadata": {
                "strategies_merged": list(strategy_results.keys()),
                "total_items": len(merged_results)
            }
        }

    async def _best_strategy_fusion(
        self,
        results: List[Dict[str, Any]],
        intent: QueryIntent
    ) -> Dict[str, Any]:
        """
        Use results from only the best-performing strategy
        """
        # Rank strategies by confidence and result count
        best_result = max(results, key=lambda r: (
            r.get("metadata", {}).get("confidence", 0),
            len(r.get("results", []))
        ))

        strategy_results = best_result.get("results", [])
        for item in strategy_results:
            item["_strategy"] = best_result.get("strategy", "unknown")
            item["_selected_as_best"] = True

        return {
            "status": "success",
            "fusion_strategy": "best_strategy_only",
            "results": strategy_results,
            "metadata": {
                "best_strategy": best_result.get("strategy"),
                "best_strategy_confidence": best_result.get("metadata", {}).get("confidence"),
                "total_items": len(strategy_results)
            }
        }

    async def _simple_concatenation_fusion(
        self,
        results: List[Dict[str, Any]],
        intent: QueryIntent
    ) -> Dict[str, Any]:
        """
        Simple concatenation of all results
        """
        all_results = []

        for result in results:
            strategy_name = result.get("strategy", "unknown")
            items = result.get("results", [])

            for item in items:
                item_copy = dict(item)
                item_copy["_strategy"] = strategy_name
                all_results.append(item_copy)

        return {
            "status": "success",
            "fusion_strategy": "simple_concatenation",
            "results": all_results,
            "metadata": {
                "total_items": len(all_results)
            }
        }

    def _get_strategy_weights(self, intent: QueryIntent) -> Dict[str, float]:
        """
        Get strategy weights based on query intent
        """
        weights = {
            "graph": 0.5,
            "vector": 0.5,
            "database": 0.5
        }

        # Adjust weights based on primary strategy
        if intent.primary_strategy == "graph_traversal":
            weights["graph"] = 1.0
            weights["vector"] = 0.3
            weights["database"] = 0.2
        elif intent.primary_strategy == "vector_similarity":
            weights["vector"] = 1.0
            weights["graph"] = 0.3
            weights["database"] = 0.2
        elif intent.primary_strategy == "database_lookup":
            weights["database"] = 1.0
            weights["vector"] = 0.3
            weights["graph"] = 0.2
        elif intent.primary_strategy == "hybrid_multi_source":
            weights["graph"] = 0.8
            weights["vector"] = 0.8
            weights["database"] = 0.6

        # Boost weights based on confidence
        if intent.confidence > 0.9:
            primary_strategy = intent.primary_strategy.replace("_", "")
            if primary_strategy in ["graph", "vector", "database"]:
                weights[primary_strategy] *= 1.2

        return weights

    async def _deduplicate_results(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove duplicate results based on similarity
        """
        results = aggregated.get("results", [])
        if len(results) <= 1:
            return aggregated

        deduplicated = []
        seen_content = set()

        for item in results:
            # Create a simple content signature for deduplication
            content_sig = self._get_content_signature(item)

            if content_sig not in seen_content:
                seen_content.add(content_sig)
                deduplicated.append(item)
            else:
                # Mark as duplicate but keep metadata about it
                for existing in deduplicated:
                    if self._get_content_signature(existing) == content_sig:
                        if "_duplicates" not in existing:
                            existing["_duplicates"] = []
                        existing["_duplicates"].append({
                            "strategy": item.get("_strategy"),
                            "score": item.get("_weighted_score", item.get("score", 0))
                        })
                        break

        # Update aggregated results
        deduplicated_result = dict(aggregated)
        deduplicated_result["results"] = deduplicated
        deduplicated_result["metadata"]["duplicates_removed"] = len(results) - len(deduplicated)

        return deduplicated_result

    def _get_content_signature(self, item: Dict[str, Any]) -> str:
        """
        Generate content signature for deduplication
        """
        # Use name, file path, or content for signature
        if "name" in item and "file_path" in item:
            return f"{item['name']}:{item['file_path']}"
        elif "id" in item:
            return str(item["id"])
        elif "content" in item:
            return item["content"][:100]  # First 100 chars
        else:
            return str(hash(json.dumps(item, sort_keys=True)))

    async def _limit_context_length(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """
        Limit total context length for LLM processing
        """
        results = aggregated.get("results", [])

        # Calculate current context length
        current_length = 0
        limited_results = []

        for item in results:
            # Estimate token count (rough approximation: 4 chars per token)
            item_content = item.get("content", "")
            item_tokens = len(item_content) // 4

            if current_length + item_tokens <= self.max_context_length:
                limited_results.append(item)
                current_length += item_tokens
            else:
                # Truncate item content to fit
                remaining_tokens = self.max_context_length - current_length
                if remaining_tokens > 50:  # Minimum meaningful content
                    truncated_item = dict(item)
                    truncated_item["content"] = item_content[:remaining_tokens * 4] + "..."
                    truncated_item["_truncated"] = True
                    limited_results.append(truncated_item)
                break

        # Update aggregated results
        limited_result = dict(aggregated)
        limited_result["results"] = limited_results
        limited_result["metadata"]["length_limited"] = len(results) - len(limited_results)
        limited_result["metadata"]["estimated_tokens"] = current_length

        return limited_result

# Global instance
context_aggregator: Optional[ContextAggregator] = None

async def get_context_aggregator() -> ContextAggregator:
    """
    Get or create global context aggregator instance
    """
    global context_aggregator
    if context_aggregator is None:
        context_aggregator = ContextAggregator()
        await context_aggregator.initialize()
    return context_aggregator
```

### Step 5: Result Ranking and Scoring (Week 3, Day 5 & Week 4, Days 1-2)

#### 4.2 Create `result_ranker.py`

**Research Reference**: Azure ML reranking models documentation

```python
import logging
import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import httpx

logger = logging.getLogger(__name__)

class ResultRanker:
    """
    Ranks and scores aggregated results using multiple ranking strategies
    """

    def __init__(self):
        self.sentence_transformer = None
        self.rerank_endpoint = os.getenv("MOSAIC_RERANK_ENDPOINT_URL")
        self.use_semantic_reranking = True
        self.max_rerank_candidates = 50

    async def initialize(self):
        """
        Initialize result ranker with models
        """
        logger.info("Initializing Result Ranker...")

        try:
            # Initialize sentence transformer for semantic similarity
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

            # Test rerank endpoint if available
            if self.rerank_endpoint:
                await self._test_rerank_endpoint()

            logger.info("Result Ranker initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize result ranker: {e}")
            self.use_semantic_reranking = False

    async def rank_results(
        self,
        aggregated_context: Dict[str, Any],
        query: str,
        intent: QueryIntent
    ) -> List[Dict[str, Any]]:
        """
        Rank and score aggregated results
        """
        try:
            results = aggregated_context.get("results", [])
            if not results:
                return []

            logger.debug(f"Ranking {len(results)} aggregated results")

            # Step 1: Initial scoring based on strategy weights and metadata
            scored_results = await self._initial_scoring(results, intent)

            # Step 2: Semantic similarity scoring
            if self.use_semantic_reranking and len(scored_results) > 1:
                scored_results = await self._semantic_similarity_scoring(scored_results, query)

            # Step 3: Cross-encoder reranking (if endpoint available)
            if self.rerank_endpoint and len(scored_results) > 1:
                top_candidates = scored_results[:self.max_rerank_candidates]
                reranked = await self._cross_encoder_reranking(top_candidates, query)
                if reranked:
                    scored_results = reranked + scored_results[self.max_rerank_candidates:]

            # Step 4: Final ranking combination
            final_ranked = await self._combine_ranking_signals(scored_results, query, intent)

            # Step 5: Apply diversity and quality filters
            filtered_results = await self._apply_quality_filters(final_ranked)

            logger.debug(f"Final ranking produced {len(filtered_results)} results")
            return filtered_results

        except Exception as e:
            logger.error(f"Result ranking failed: {e}")
            # Return original results as fallback
            return aggregated_context.get("results", [])

    async def _initial_scoring(
        self,
        results: List[Dict[str, Any]],
        intent: QueryIntent
    ) -> List[Dict[str, Any]]:
        """
        Apply initial scoring based on strategy weights and metadata
        """
        scored_results = []

        for result in results:
            score_components = {}

            # Strategy weight score
            strategy_weight = result.get("_weight", 0.5)
            score_components["strategy_weight"] = strategy_weight

            # Original score from strategy
            original_score = result.get("_original_score", result.get("score", 1.0))
            score_components["original_score"] = original_score

            # Metadata-based scoring
            metadata_score = self._calculate_metadata_score(result)
            score_components["metadata_score"] = metadata_score

            # Duplicate boost (if result was found by multiple strategies)
            duplicate_boost = len(result.get("_duplicates", [])) * 0.1
            score_components["duplicate_boost"] = duplicate_boost

            # Combine initial scores
            initial_score = (
                strategy_weight * 0.4 +
                original_score * 0.3 +
                metadata_score * 0.2 +
                duplicate_boost * 0.1
            )

            # Add scoring metadata to result
            scored_result = dict(result)
            scored_result["_initial_score"] = initial_score
            scored_result["_score_components"] = score_components
            scored_results.append(scored_result)

        # Sort by initial score
        scored_results.sort(key=lambda x: x.get("_initial_score", 0), reverse=True)
        return scored_results

    def _calculate_metadata_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate score based on result metadata quality
        """
        score = 0.5  # Base score

        # Boost for complete metadata
        if result.get("entity_type"):
            score += 0.1
        if result.get("file_path"):
            score += 0.1
        if result.get("content") and len(result["content"]) > 100:
            score += 0.1
        if result.get("name"):
            score += 0.1

        # Boost for code-specific metadata
        if result.get("rdf_triples"):
            score += 0.2
        if result.get("relationships"):
            score += 0.1

        return min(1.0, score)

    async def _semantic_similarity_scoring(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Score results based on semantic similarity to query
        """
        if not self.sentence_transformer:
            return results

        try:
            # Encode query
            query_embedding = self.sentence_transformer.encode(query, convert_to_tensor=True)

            # Encode result contents
            result_texts = []
            for result in results:
                # Create text representation of result
                text_parts = []
                if result.get("name"):
                    text_parts.append(f"Name: {result['name']}")
                if result.get("content"):
                    text_parts.append(f"Content: {result['content'][:200]}")
                if result.get("entity_type"):
                    text_parts.append(f"Type: {result['entity_type']}")

                result_text = " | ".join(text_parts) if text_parts else "No content available"
                result_texts.append(result_text)

            # Encode all result texts
            result_embeddings = self.sentence_transformer.encode(result_texts, convert_to_tensor=True)

            # Calculate similarities
            similarities = util.pytorch_cos_sim(query_embedding, result_embeddings)[0]

            # Add similarity scores to results
            for i, result in enumerate(results):
                similarity_score = float(similarities[i])
                result["_semantic_similarity"] = similarity_score

                # Update overall score
                current_score = result.get("_initial_score", 0.5)
                combined_score = current_score * 0.7 + similarity_score * 0.3
                result["_semantic_score"] = combined_score

            # Re-sort by semantic score
            results.sort(key=lambda x: x.get("_semantic_score", 0), reverse=True)
            return results

        except Exception as e:
            logger.error(f"Semantic similarity scoring failed: {e}")
            return results

    async def _cross_encoder_reranking(
        self,
        candidates: List[Dict[str, Any]],
        query: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Rerank top candidates using cross-encoder model via Azure ML endpoint
        """
        try:
            if not self.rerank_endpoint:
                return None

            # Prepare reranking request
            query_doc_pairs = []
            for candidate in candidates:
                doc_text = self._extract_document_text(candidate)
                query_doc_pairs.append({"query": query, "document": doc_text})

            # Call reranking endpoint
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.rerank_endpoint,
                    json={
                        "instances": query_doc_pairs,
                        "parameters": {
                            "return_scores": True
                        }
                    },
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    rerank_results = response.json()
                    scores = rerank_results.get("predictions", [])

                    # Apply rerank scores
                    for i, candidate in enumerate(candidates):
                        if i < len(scores):
                            rerank_score = scores[i]
                            candidate["_rerank_score"] = rerank_score

                            # Combine with existing scores
                            semantic_score = candidate.get("_semantic_score", 0.5)
                            combined_score = semantic_score * 0.6 + rerank_score * 0.4
                            candidate["_final_score"] = combined_score

                    # Re-sort by final score
                    candidates.sort(key=lambda x: x.get("_final_score", 0), reverse=True)
                    return candidates

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")

        return None

    def _extract_document_text(self, result: Dict[str, Any]) -> str:
        """
        Extract text representation for reranking
        """
        text_parts = []

        if result.get("name"):
            text_parts.append(result["name"])
        if result.get("entity_type"):
            text_parts.append(f"({result['entity_type']})")
        if result.get("content"):
            text_parts.append(result["content"][:300])  # Limit for reranking

        return " ".join(text_parts) if text_parts else "No content"

    async def _combine_ranking_signals(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent
    ) -> List[Dict[str, Any]]:
        """
        Combine all ranking signals into final score
        """
        for result in results:
            # Collect all available scores
            initial_score = result.get("_initial_score", 0.5)
            semantic_score = result.get("_semantic_similarity", 0.5)
            rerank_score = result.get("_rerank_score", None)

            # Calculate final score based on available signals
            if rerank_score is not None:
                # Use rerank score as primary signal
                final_score = rerank_score * 0.6 + semantic_score * 0.3 + initial_score * 0.1
            else:
                # Combine semantic and initial scores
                final_score = semantic_score * 0.7 + initial_score * 0.3

            # Apply intent-based boosts
            final_score = self._apply_intent_boosts(result, intent, final_score)

            result["_final_score"] = final_score

        # Final sort by combined score
        results.sort(key=lambda x: x.get("_final_score", 0), reverse=True)
        return results

    def _apply_intent_boosts(
        self,
        result: Dict[str, Any],
        intent: QueryIntent,
        current_score: float
    ) -> float:
        """
        Apply intent-specific boosts to scoring
        """
        boost = 1.0

        # Boost results that match the intent strategy
        result_strategy = result.get("_strategy", "")
        if intent.primary_strategy == "graph_traversal" and result_strategy == "graph":
            boost += 0.1
        elif intent.primary_strategy == "vector_similarity" and result_strategy == "vector":
            boost += 0.1
        elif intent.primary_strategy == "database_lookup" and result_strategy == "database":
            boost += 0.1

        # Boost results with relationship information for graph queries
        if intent.requires_graph and result.get("rdf_triples"):
            boost += 0.05

        # Boost exact matches for lookup queries
        if intent.primary_strategy == "database_lookup":
            query_terms = intent.query_features.get("query", "").lower().split()
            result_name = result.get("name", "").lower()
            if any(term in result_name for term in query_terms):
                boost += 0.15

        return min(1.0, current_score * boost)

    async def _apply_quality_filters(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply quality filters to final results
        """
        filtered = []

        for result in results:
            # Quality checks
            if self._passes_quality_check(result):
                # Clean up internal scoring metadata for final output
                clean_result = self._clean_result_metadata(result)
                filtered.append(clean_result)

        return filtered

    def _passes_quality_check(self, result: Dict[str, Any]) -> bool:
        """
        Check if result meets quality thresholds
        """
        # Minimum score threshold
        if result.get("_final_score", 0) < 0.1:
            return False

        # Must have meaningful content
        if not result.get("name") and not result.get("content"):
            return False

        # Must not be completely empty
        if all(not result.get(key) for key in ["name", "content", "entity_type", "file_path"]):
            return False

        return True

    def _clean_result_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean internal metadata from final result
        """
        clean_result = {}

        # Copy relevant fields
        for key, value in result.items():
            if not key.startswith("_"):
                clean_result[key] = value

        # Keep final score and strategy info
        clean_result["relevance_score"] = result.get("_final_score", 0.5)
        clean_result["source_strategy"] = result.get("_strategy", "unknown")

        # Keep duplicate info if available
        if result.get("_duplicates"):
            clean_result["found_by_multiple_strategies"] = True
            clean_result["duplicate_count"] = len(result["_duplicates"])

        return clean_result

    async def _test_rerank_endpoint(self):
        """
        Test rerank endpoint availability
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.rerank_endpoint}/health")
                if response.status_code == 200:
                    logger.info("Rerank endpoint is available")
                else:
                    logger.warning("Rerank endpoint health check failed")
                    self.rerank_endpoint = None
        except Exception as e:
            logger.warning(f"Rerank endpoint not available: {e}")
            self.rerank_endpoint = None

# Global instance
result_ranker: Optional[ResultRanker] = None

async def get_result_ranker() -> ResultRanker:
    """
    Get or create global result ranker instance
    """
    global result_ranker
    if result_ranker is None:
        result_ranker = ResultRanker()
        await result_ranker.initialize()
    return result_ranker
```

## Step 6: OmniRAG Plugin Integration (Week 4, Days 3-5)

#### 6.1 Create `omnirag_plugin.py`

```python
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from ..orchestration.omnirag_orchestrator import get_omnirag_orchestrator

logger = logging.getLogger(__name__)

class OmniRAGPlugin:
    """
    Main MCP plugin that provides the complete OmniRAG interface
    """

    def __init__(self):
        self.orchestrator = None
        self.cosmos_client = None

    async def initialize(self, cosmos_client):
        """
        Initialize OmniRAG plugin
        """
        logger.info("Initializing OmniRAG Plugin...")

        self.cosmos_client = cosmos_client
        self.orchestrator = await get_omnirag_orchestrator()
        await self.orchestrator.initialize(cosmos_client)

        logger.info("OmniRAG Plugin initialized successfully")

    async def omnirag_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        format_type: str = "structured"
    ) -> Dict[str, Any]:
        """
        Process query through complete OmniRAG pipeline

        Args:
            query: Natural language query
            session_id: Optional session identifier for tracking
            context: Additional context for query processing
            limit: Maximum number of results
            format_type: Output format (structured, narrative, json)

        Returns:
            Complete OmniRAG response
        """
        try:
            # Prepare context
            processing_context = context or {}
            processing_context["limit"] = limit
            processing_context["format_type"] = format_type

            # Process through orchestrator
            result = await self.orchestrator.process_query(
                query=query,
                context=processing_context,
                session_id=session_id
            )

            # Format results based on requested format
            if format_type == "narrative":
                result = await self._format_as_narrative(result)
            elif format_type == "json":
                result = await self._format_as_json(result)
            # structured format is already in correct format

            return result

        except Exception as e:
            logger.error(f"OmniRAG query failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "query": query,
                "session_id": session_id
            }

    async def _format_as_narrative(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format result as narrative text suitable for LLM context
        """
        if result.get("status") != "success":
            return result

        results = result.get("results", [])
        intent = result.get("intent", {})
        metadata = result.get("metadata", {})

        narrative_parts = []

        # Add intent explanation
        narrative_parts.append(f"Query Intent: {intent.get('primary_strategy', 'unknown')}")
        narrative_parts.append(f"Confidence: {intent.get('confidence', 0):.2f}")
        narrative_parts.append("")

        # Add strategies used
        strategies_used = result.get("strategies_used", [])
        if strategies_used:
            narrative_parts.append(f"Information Sources: {', '.join(strategies_used)}")
            narrative_parts.append("")

        # Format results
        if results:
            narrative_parts.append(f"Found {len(results)} relevant results:")
            narrative_parts.append("")

            for i, item in enumerate(results[:10], 1):  # Limit to top 10 for narrative
                item_text = []

                if item.get("name"):
                    item_text.append(f"**{item['name']}**")

                if item.get("entity_type"):
                    item_text.append(f"({item['entity_type']})")

                if item.get("file_path"):
                    item_text.append(f"in {item['file_path']}")

                if item.get("content"):
                    content = item["content"][:200] + "..." if len(item["content"]) > 200 else item["content"]
                    item_text.append(f"\n{content}")

                if item.get("relevance_score"):
                    item_text.append(f"\n(Relevance: {item['relevance_score']:.2f})")

                narrative_parts.append(f"{i}. {' '.join(item_text)}")
                narrative_parts.append("")
        else:
            narrative_parts.append("No relevant results found.")

        # Add performance info
        total_time = metadata.get("total_execution_time_ms", 0)
        narrative_parts.append(f"Query processed in {total_time}ms using {len(strategies_used)} sources.")

        # Update result with narrative
        narrative_result = dict(result)
        narrative_result["narrative"] = "\n".join(narrative_parts)
        narrative_result["format"] = "narrative"

        return narrative_result

    async def _format_as_json(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format result as clean JSON structure
        """
        if result.get("status") != "success":
            return result

        json_result = {
            "status": result["status"],
            "query": result["query"],
            "intent": result.get("intent", {}),
            "sources": result.get("strategies_used", []),
            "results": result.get("results", []),
            "performance": {
                "total_time_ms": result.get("metadata", {}).get("total_execution_time_ms", 0),
                "result_count": len(result.get("results", [])),
                "sources_used": len(result.get("strategies_used", []))
            },
            "format": "json"
        }

        return json_result

    async def get_omnirag_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive OmniRAG system statistics
        """
        try:
            orchestrator_stats = await self.orchestrator.get_orchestrator_statistics()

            # Add plugin-level statistics
            stats = {
                "omnirag_plugin": {
                    "status": "active",
                    "initialized": self.orchestrator is not None
                },
                "orchestrator": orchestrator_stats,
                "system_info": {
                    "parallel_processing": orchestrator_stats.get("parallel_enabled", False),
                    "max_sources": orchestrator_stats.get("max_sources", 3),
                    "timeout_seconds": orchestrator_stats.get("timeout_seconds", 30)
                }
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get OmniRAG statistics: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

# Global instance
omnirag_plugin: Optional[OmniRAGPlugin] = None

async def get_omnirag_plugin() -> OmniRAGPlugin:
    """
    Get or create global OmniRAG plugin instance
    """
    global omnirag_plugin
    if omnirag_plugin is None:
        omnirag_plugin = OmniRAGPlugin()
    return omnirag_plugin
```

### Step 7: MCP Interface Integration (Week 5, Days 1-2)

#### 7.1 Update Main Server with OmniRAG Plugin

**Modify `src/mosaic-mcp/server/main.py`**:

```python
# Add to existing imports
from ..plugins.omnirag_plugin import get_omnirag_plugin

class MosaicMCPServer:
    def __init__(self):
        # Existing initialization
        self.omnirag_plugin = None

    async def initialize(self):
        """Enhanced initialization with OmniRAG plugin"""
        # Existing initialization code

        # Initialize OmniRAG plugin
        self.omnirag_plugin = await get_omnirag_plugin()
        await self.omnirag_plugin.initialize(self.cosmos_client)

    # Replace or enhance existing query tools with OmniRAG
    @self.mcp.tool("mosaic.omnirag.query")
    async def omnirag_query_tool(
        query: str,
        session_id: str = "",
        limit: int = 20,
        format_type: str = "structured"
    ) -> str:
        """
        Query using the complete OmniRAG pipeline with intelligent intent detection

        Args:
            query: Natural language query
            session_id: Session identifier for tracking
            limit: Maximum number of results
            format_type: Output format (structured, narrative, json)
        """
        try:
            result = await self.omnirag_plugin.omnirag_query(
                query=query,
                session_id=session_id or None,
                limit=limit,
                format_type=format_type
            )

            if result.get("status") == "success":
                if format_type == "narrative":
                    return result.get("narrative", "No narrative available")
                else:
                    return self._format_omnirag_results(result)
            else:
                return f"OmniRAG query failed: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"OmniRAG query error: {str(e)}"

    @self.mcp.tool("mosaic.omnirag.statistics")
    async def omnirag_statistics_tool() -> str:
        """
        Get OmniRAG system statistics and performance metrics
        """
        try:
            stats = await self.omnirag_plugin.get_omnirag_statistics()
            return json.dumps(stats, indent=2)
        except Exception as e:
            return f"Statistics error: {str(e)}"

    def _format_omnirag_results(self, result: Dict) -> str:
        """
        Format OmniRAG results for MCP response
        """
        if not result.get("results"):
            return "No results found."

        results = result["results"]
        intent = result.get("intent", {})
        metadata = result.get("metadata", {})

        output = []

        # Add intent and strategy information
        output.append(f"🧠 Query Intent: {intent.get('primary_strategy', 'unknown').replace('_', ' ').title()}")
        output.append(f"🎯 Confidence: {intent.get('confidence', 0):.1%}")
        output.append(f"📊 Sources: {', '.join(result.get('strategies_used', []))}")
        output.append("")

        # Format results with enhanced metadata
        output.append(f"📋 Found {len(results)} results:")
        output.append("")

        for i, item in enumerate(results[:15], 1):  # Show top 15 results
            item_parts = []

            # Name and type
            if item.get("name"):
                item_parts.append(f"**{item['name']}**")
            if item.get("entity_type"):
                item_parts.append(f"({item['entity_type']})")

            # Location
            if item.get("file_path"):
                item_parts.append(f"📁 {item['file_path']}")

            # Content preview
            if item.get("content"):
                content = item["content"][:150] + "..." if len(item["content"]) > 150 else item["content"]
                item_parts.append(f"💾 {content}")

            # Relevance score and source
            score = item.get("relevance_score", 0)
            source = item.get("source_strategy", "unknown")
            item_parts.append(f"⚡ Score: {score:.2f} (from {source})")

            # Multi-strategy indicator
            if item.get("found_by_multiple_strategies"):
                item_parts.append(f"🔄 Found by {item.get('duplicate_count', 1) + 1} sources")

            output.append(f"{i}. {' | '.join(item_parts)}")
            output.append("")

        # Performance summary
        total_time = metadata.get("total_execution_time_ms", 0)
        sources_count = len(result.get("strategies_used", []))
        output.append(f"⏱️ Processed in {total_time}ms using {sources_count} sources")

        return "\n".join(output)
```

## 🧪 Complete Testing Strategy (Week 5, Days 3-5)

### End-to-End Integration Tests

```python
# tests/integration/test_omnirag_complete.py
import pytest
from src.mosaic_mcp.plugins.omnirag_plugin import get_omnirag_plugin

@pytest.mark.asyncio
async def test_complete_omnirag_pipeline():
    """Test complete OmniRAG pipeline with various query types"""

    plugin = await get_omnirag_plugin()
    await plugin.initialize(mock_cosmos_client)

    # Test different query types
    test_queries = [
        ("What are the dependencies of Flask?", "graph_traversal"),
        ("Find functions similar to authenticate_user", "vector_similarity"),
        ("What is the requests library?", "database_lookup"),
        ("Analyze the complete authentication system", "hybrid_multi_source")
    ]

    for query, expected_intent in test_queries:
        result = await plugin.omnirag_query(
            query=query,
            session_id="test_session",
            limit=10
        )

        assert result["status"] == "success"
        assert result["intent"]["primary_strategy"] == expected_intent
        assert len(result["results"]) > 0
        assert result["metadata"]["total_execution_time_ms"] > 0

@pytest.mark.asyncio
async def test_parallel_vs_sequential_performance():
    """Compare parallel vs sequential retrieval performance"""

    plugin = await get_omnirag_plugin()
    await plugin.initialize(mock_cosmos_client)

    query = "Comprehensive analysis of Flask framework"

    # Test parallel execution
    result_parallel = await plugin.omnirag_query(
        query=query,
        context={"parallel_enabled": True}
    )

    # Test sequential execution
    result_sequential = await plugin.omnirag_query(
        query=query,
        context={"parallel_enabled": False}
    )

    # Parallel should be faster for multi-source queries
    parallel_time = result_parallel["metadata"]["total_execution_time_ms"]
    sequential_time = result_sequential["metadata"]["total_execution_time_ms"]

    assert parallel_time < sequential_time
    assert result_parallel["status"] == "success"
    assert result_sequential["status"] == "success"
```

## ✅ Phase 3 Completion Checklist

- [ ] Intent classification system working with >85% accuracy
- [ ] Multi-source orchestration executing in parallel and sequential modes
- [ ] Context aggregation combining results from all sources effectively
- [ ] Result ranking using multiple signals (semantic, rerank, metadata)
- [ ] OmniRAG plugin providing complete MCP interface
- [ ] All query types (graph, vector, database, hybrid) functioning
- [ ] Performance targets met (<500ms for hybrid queries)
- [ ] Comprehensive integration tests passing
- [ ] Error handling and fallback strategies working
- [ ] Documentation and monitoring in place

## 🚨 Common Pitfalls and Solutions

### Pitfall 1: Intent Classification Accuracy

**Problem**: Queries misclassified leading to suboptimal retrieval strategies
**Solution**: Continuous training with user feedback, better feature engineering

### Pitfall 2: Parallel Execution Complexity

**Problem**: Race conditions, timeout handling, exception propagation
**Solution**: Robust async patterns, comprehensive error handling, timeout management

### Pitfall 3: Context Aggregation Quality

**Problem**: Redundant or conflicting information from multiple sources
**Solution**: Smart deduplication, confidence-based weighting, quality filters

### Pitfall 4: Performance Degradation

**Problem**: Multiple retrieval strategies increase latency
**Solution**: Intelligent caching, early termination, strategy optimization

## 📋 Post-Phase 3 Cleanup and Optimization

1. **Performance Optimization**

   - Profile end-to-end query execution
   - Optimize parallel execution coordination
   - Implement smart caching strategies
   - Fine-tune ranking algorithms

2. **Monitoring and Alerting**

   - Add comprehensive metrics collection
   - Set up performance monitoring dashboards
   - Implement query success rate tracking
   - Create alerting for system health

3. **User Experience Enhancement**

   - Add query suggestion capabilities
   - Implement query refinement feedback
   - Create interactive result exploration
   - Add export and sharing features

4. **System Maintenance**
   - Set up automated model retraining
   - Implement A/B testing for ranking algorithms
   - Create system health checks
   - Document operational procedures

---

**Next Documents**:

- `04-research-references.md` - Complete research links and materials
- `06-testing-validation.md` - Comprehensive testing strategies
- `07-deployment-migration.md` - Production deployment procedures

The OmniRAG transformation is now complete with intelligent intent detection, multi-source orchestration, and advanced result fusion capabilities!
