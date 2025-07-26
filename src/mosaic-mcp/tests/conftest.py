"""
Test configuration and fixtures for intent classification tests.
"""

import pytest
import tempfile
import shutil
import asyncio
from typing import Generator, AsyncGenerator, Dict, List

from ..models.intent_models import (
    IntentClassificationPlugin,
    TrainingConfig,
    RetrievalStrategy,
)
from ..plugins.query_intent_classifier import (
    QueryIntentClassifier,
    reset_intent_classifier,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_model_dir() -> Generator[str, None, None]:
    """Create temporary directory for model storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config() -> IntentClassificationPlugin:
    """Create test configuration."""
    return IntentClassificationPlugin(
        enabled=True,
        model_path="test_models/intent_classifier",
        fallback_strategy=RetrievalStrategy.MULTI_SOURCE,
        max_query_length=1000,
        cache_embeddings=True,
        log_classifications=False,  # Disable for tests
        performance_monitoring=True,
    )


@pytest.fixture
def test_training_config() -> TrainingConfig:
    """Create test training configuration."""
    return TrainingConfig(
        model_name="all-MiniLM-L6-v2",
        n_estimators=50,  # Reduced for faster training
        random_state=42,
        test_size=0.2,
        confidence_threshold=0.7,
        min_samples_per_class=100,  # Reduced for faster tests
        cross_validation_folds=3,  # Reduced for faster tests
    )


@pytest.fixture
async def classifier_instance(
    temp_model_dir: str, test_config: IntentClassificationPlugin
) -> AsyncGenerator[QueryIntentClassifier, None]:
    """Create classifier instance for testing."""
    reset_intent_classifier()  # Reset global state

    classifier = QueryIntentClassifier(test_config, model_dir=temp_model_dir)
    await classifier.initialize()

    yield classifier

    reset_intent_classifier()  # Clean up global state


# Test data samples for various scenarios
SAMPLE_QUERIES = {
    "graph_rag": [
        "What functions call the main function?",
        "Show me the relationship between classes A and B",
        "Which components depend on the authentication module?",
        "Find all functions that are called by processData",
        "What are the dependencies of the user service?",
        "Show me the call graph for the payment system",
        "Which classes inherit from BaseModel?",
        "Find all methods that override the validate function",
    ],
    "vector_rag": [
        "Find documents similar to machine learning",
        "Search for content about data processing",
        "What documents mention neural networks?",
        "Find articles similar to this one about AI",
        "Search for papers on distributed systems",
        "Find documentation about API best practices",
        "Locate content related to security vulnerabilities",
        "Search for code examples of async programming",
    ],
    "database_rag": [
        "How many users registered last month?",
        "What's the total revenue for Q3?",
        "Count the number of active sessions",
        "Show me the average response time",
        "List all users from California",
        "What are the top 10 products by sales?",
        "Count errors by type in the last week",
        "Show me user demographics data",
    ],
    "hybrid": [
        "Analyze code relationships and find similar patterns",
        "Find functions that call getUserData and show similar implementations",
        "Search for authentication flows and their dependencies",
        "Find all payment processing code and related documentation",
        "Analyze API usage patterns and find similar endpoints",
        "Show me user registration flow and related metrics",
        "Find error handling patterns and count recent errors",
        "Analyze data processing pipeline and performance metrics",
    ],
}


EDGE_CASE_QUERIES = [
    "",  # Empty query
    "   ",  # Whitespace only
    "a",  # Single character
    "What?",  # Single word question
    "Very " * 100 + "long query",  # Very long query
    "🤖 Find code with emojis 🔍",  # Special characters
    "SELECT * FROM users WHERE name = 'test'",  # SQL-like query
    "function main() { return 'hello'; }",  # Code snippet
    "https://example.com/api/docs",  # URL
    "2023-12-25 14:30:00",  # Timestamp
]


@pytest.fixture
def sample_queries() -> Dict[str, List[str]]:
    """Provide sample queries for testing."""
    return SAMPLE_QUERIES


@pytest.fixture
def edge_case_queries() -> List[str]:
    """Provide edge case queries for testing."""
    return EDGE_CASE_QUERIES
