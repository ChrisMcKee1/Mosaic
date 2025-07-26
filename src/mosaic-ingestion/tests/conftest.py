"""
Pytest configuration and shared fixtures for mosaic-ingestion tests.

Provides common test fixtures, mock setups, and configuration for all
mosaic-ingestion service tests.
"""

import pytest
from unittest.mock import Mock, AsyncMock
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_cosmos_client():
    """Create a mock Azure Cosmos DB client."""
    mock_client = Mock()
    mock_database = Mock()
    mock_container = Mock()

    # Setup method chaining
    mock_client.get_database_client.return_value = mock_database
    mock_database.get_container_client.return_value = mock_container

    # Mock successful operations
    mock_container.upsert_item.return_value = {"id": "success"}
    mock_container.query_items.return_value = []

    return mock_client


@pytest.fixture
def mock_semantic_kernel():
    """Create a mock Semantic Kernel instance."""
    mock_kernel = Mock()

    # Mock kernel functions
    mock_function = AsyncMock()
    mock_function.invoke.return_value = Mock(value="Mock AI response")
    mock_kernel.get_function.return_value = mock_function

    return mock_kernel


@pytest.fixture
def mock_azure_settings():
    """Create mock Azure settings."""
    settings = Mock()
    settings.azure_openai_endpoint = "https://test.openai.azure.com/"
    settings.azure_cosmos_endpoint = "https://test.cosmos.azure.com/"
    settings.database_name = "test-knowledge"
    settings.container_name = "test-golden-nodes"
    settings.openai_api_version = "2024-02-01"
    settings.embedding_model = "text-embedding-3-small"
    settings.chat_model = "gpt-4"
    return settings


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing AST parsing."""
    return '''
"""
Sample module for testing AST parsing.
"""

import os
import json
from typing import List, Dict, Optional


def calculate_metrics(data: List[float]) -> Dict[str, float]:
    """
    Calculate basic metrics for a list of numbers.
    
    Args:
        data: List of numeric values
        
    Returns:
        Dictionary containing calculated metrics
    """
    if not data:
        return {"error": "No data provided"}
    
    return {
        "count": len(data),
        "sum": sum(data),
        "average": sum(data) / len(data),
        "min": min(data),
        "max": max(data)
    }


class DataProcessor:
    """
    Processes and analyzes data from various sources.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the data processor."""
        self.config_path = config_path
        self.config = self._load_config() if config_path else {}
        
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load config: {e}")
            return {}
    
    async def process_batch(self, data_batch: List[Dict]) -> List[Dict]:
        """
        Process a batch of data asynchronously.
        
        Args:
            data_batch: List of data items to process
            
        Returns:
            List of processed data items
        """
        processed = []
        for item in data_batch:
            if self._validate_item(item):
                processed_item = await self._process_item(item)
                processed.append(processed_item)
        return processed
    
    def _validate_item(self, item: Dict) -> bool:
        """Validate a single data item."""
        required_fields = ['id', 'type', 'data']
        return all(field in item for field in required_fields)
    
    async def _process_item(self, item: Dict) -> Dict:
        """Process a single data item."""
        # Simulate async processing
        import asyncio
        await asyncio.sleep(0.001)
        
        return {
            **item,
            'processed': True,
            'timestamp': datetime.now().isoformat()
        }


# Module-level constants
DEFAULT_CONFIG = {
    'batch_size': 100,
    'timeout': 30,
    'retry_count': 3
}

SUPPORTED_TYPES = ['numeric', 'text', 'binary']
'''


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code for testing AST parsing."""
    return """
/**
 * User management utilities for the application.
 */

const crypto = require('crypto');
const bcrypt = require('bcrypt');

/**
 * Generate a secure random token.
 * @param {number} length - Length of the token
 * @returns {string} Generated token
 */
function generateToken(length = 32) {
    return crypto.randomBytes(length).toString('hex');
}

/**
 * Hash a password using bcrypt.
 * @param {string} password - Plain text password
 * @returns {Promise<string>} Hashed password
 */
async function hashPassword(password) {
    const saltRounds = 12;
    return await bcrypt.hash(password, saltRounds);
}

/**
 * User management class.
 */
class UserManager {
    constructor(database) {
        this.db = database;
        this.cache = new Map();
    }
    
    /**
     * Create a new user account.
     * @param {Object} userData - User data
     * @returns {Promise<Object>} Created user
     */
    async createUser(userData) {
        const { username, email, password } = userData;
        
        // Validate input
        if (!this.validateEmail(email)) {
            throw new Error('Invalid email format');
        }
        
        // Hash password
        const hashedPassword = await hashPassword(password);
        
        // Create user record
        const user = {
            id: generateToken(16),
            username,
            email,
            password: hashedPassword,
            createdAt: new Date().toISOString(),
            active: true
        };
        
        // Save to database
        await this.db.users.insert(user);
        
        // Cache user (without password)
        const { password: _, ...userForCache } = user;
        this.cache.set(user.id, userForCache);
        
        return userForCache;
    }
    
    /**
     * Validate email format.
     * @param {string} email - Email to validate
     * @returns {boolean} True if valid
     */
    validateEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
    
    /**
     * Get user by ID.
     * @param {string} userId - User ID
     * @returns {Promise<Object|null>} User object or null
     */
    async getUser(userId) {
        // Check cache first
        if (this.cache.has(userId)) {
            return this.cache.get(userId);
        }
        
        // Query database
        const user = await this.db.users.findById(userId);
        if (user) {
            const { password: _, ...userForCache } = user;
            this.cache.set(userId, userForCache);
            return userForCache;
        }
        
        return null;
    }
}

// Arrow function example
const utilities = {
    formatDate: (date) => date.toISOString().split('T')[0],
    generateSlug: (text) => text.toLowerCase().replace(/\s+/g, '-'),
    parseJson: (jsonString) => {
        try {
            return JSON.parse(jsonString);
        } catch (error) {
            return null;
        }
    }
};

module.exports = {
    UserManager,
    generateToken,
    hashPassword,
    utilities
};
"""


@pytest.fixture
def sample_file_info():
    """Sample file information for testing."""
    return {
        "path": "/tmp/repo/src/example.py",
        "relative_path": "src/example.py",
        "size": 2500,
        "language": "python",
    }


@pytest.fixture
def sample_git_context():
    """Sample git context information for testing."""
    return {
        "repository_url": "https://github.com/test/example-repo.git",
        "branch": "main",
        "commit_hash": "abc123def456789",
        "commit_message": "Add example functionality",
        "author_name": "Test Developer",
        "author_email": "dev@test.com",
        "commit_timestamp": datetime.now(timezone.utc),
    }


@pytest.fixture
def sample_golden_node_data():
    """Sample GoldenNode data for testing."""
    return {
        "id": str(uuid4()),
        "entity_type": "function",
        "name": "process_data",
        "content": "def process_data(input_data):\n    return [x * 2 for x in input_data]",
        "file_context": {
            "file_path": "/tmp/repo/src/processor.py",
            "relative_path": "src/processor.py",
            "file_size": 1500,
            "line_count": 45,
            "language": "python",
            "encoding": "utf-8",
            "file_hash": "def456abc789",
        },
        "git_context": {
            "repository_url": "https://github.com/test/processor-repo.git",
            "branch": "develop",
            "commit_hash": "xyz789abc123",
            "commit_message": "Implement data processing",
            "author_name": "Developer Two",
            "author_email": "dev2@test.com",
            "commit_timestamp": datetime.now(timezone.utc),
        },
        "start_line": 15,
        "end_line": 17,
        "start_column": 0,
        "end_column": 35,
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_repository_structure(temp_directory):
    """Create a sample repository structure in temp directory."""
    repo_path = Path(temp_directory)

    # Create directory structure
    (repo_path / "src").mkdir()
    (repo_path / "tests").mkdir()
    (repo_path / "docs").mkdir()

    # Create sample files
    (repo_path / "src" / "__init__.py").write_text("")
    (repo_path / "src" / "main.py").write_text("""
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")

    (repo_path / "src" / "utils.py").write_text("""
def helper_function(data):
    return data.upper()

class UtilityClass:
    def process(self, value):
        return value * 2
""")

    (repo_path / "tests" / "test_main.py").write_text("""
import unittest
from src.main import main

class TestMain(unittest.TestCase):
    def test_main(self):
        # Test would go here
        pass
""")

    (repo_path / "README.md").write_text(
        "# Test Repository\n\nThis is a test repository."
    )

    return str(repo_path)


@pytest.fixture
def mock_tree_sitter_parser():
    """Create a mock tree-sitter parser."""
    mock_parser = Mock()
    mock_tree = Mock()
    mock_root_node = Mock()

    # Mock AST structure
    mock_function_node = Mock()
    mock_function_node.type = "function_definition"
    mock_function_node.start_point = (0, 0)
    mock_function_node.end_point = (3, 0)
    mock_function_node.children = [Mock(type="identifier", text=b"test_function")]

    mock_class_node = Mock()
    mock_class_node.type = "class_definition"
    mock_class_node.start_point = (5, 0)
    mock_class_node.end_point = (10, 0)
    mock_class_node.children = [Mock(type="identifier", text=b"TestClass")]

    mock_root_node.children = [mock_function_node, mock_class_node]
    mock_tree.root_node = mock_root_node
    mock_parser.parse.return_value = mock_tree

    return mock_parser


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mocks after each test."""
    yield
    # Any cleanup code would go here


# Test markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.azure = pytest.mark.azure
pytest.mark.slow = pytest.mark.slow
