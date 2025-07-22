"""
Comprehensive test suite for AI-powered entity descriptions.

Tests the enhanced _enhance_entities_with_ai_descriptions implementation including
caching, batch processing, retry logic, and Semantic Kernel integration.
"""

import asyncio
import pytest
import hashlib
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Import the ingestion plugin
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestAIPoweredEntityDescriptions:
    """Test suite for AI-powered entity description generation."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.azure_openai_endpoint = "https://test.openai.azure.com/"
        settings.azure_openai_chat_deployment_name = "gpt-4o-mini"
        settings.azure_openai_api_key = "test-key"
        return settings
    
    @pytest.fixture
    def ingestion_plugin(self, mock_settings):
        """Create ingestion plugin instance with mocked dependencies."""
        from ingestion_service.plugins.ingestion import IngestionPlugin
        
        plugin = IngestionPlugin(mock_settings)
        
        # Mock the kernel and chat service
        plugin.kernel = Mock()
        plugin.chat_service = Mock()
        
        return plugin
    
    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        return [
            {
                "id": "test_func_1",
                "entity_type": "function_definition",
                "name": "calculate_fibonacci",
                "language": "python",
                "content": """def calculate_fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number using dynamic programming.\"\"\"
    if n <= 1:
        return n
    
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    
    return b""",
                "start_line": 1,
                "end_line": 10,
            },
            {
                "id": "test_class_1", 
                "entity_type": "class_definition",
                "name": "DatabaseManager",
                "language": "python",
                "content": """class DatabaseManager:
    \"\"\"Manages database connections and operations.\"\"\"
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    def connect(self):
        \"\"\"Establish database connection.\"\"\"
        self.connection = create_connection(self.connection_string)
    
    def execute_query(self, query, params=None):
        \"\"\"Execute a database query with optional parameters.\"\"\"
        if not self.connection:
            raise RuntimeError("Not connected to database")
        return self.connection.execute(query, params or [])""",
                "start_line": 15,
                "end_line": 30,
            },
            {
                "id": "test_small_func",
                "entity_type": "function_definition", 
                "name": "small_func",
                "language": "python",
                "content": """def small_func():
    return True""",
                "start_line": 1,
                "end_line": 2,
            },
            {
                "id": "test_existing_ai",
                "entity_type": "function_definition",
                "name": "existing_func", 
                "language": "python",
                "content": """def existing_func():
    \"\"\"Function with existing AI analysis.\"\"\"
    for i in range(10):
        print(i)
    return True""",
                "has_ai_analysis": True,
                "ai_description": "Existing description",
            }
        ]
    
    def test_filter_entities_for_ai_analysis(self, ingestion_plugin, sample_entities):
        """Test entity filtering logic."""
        candidates = ingestion_plugin._filter_entities_for_ai_analysis(sample_entities)
        
        # Should include function_definition and class_definition with >10 lines
        assert len(candidates) == 2
        
        # Should include the fibonacci function and database class
        candidate_names = [e.get("name") for e in candidates]
        assert "calculate_fibonacci" in candidate_names
        assert "DatabaseManager" in candidate_names
        
        # Should exclude small function (< 10 lines)
        assert "small_func" not in candidate_names
        
        # Should exclude entity with existing AI analysis
        assert "existing_func" not in candidate_names
    
    def test_entity_filtering_by_type(self, ingestion_plugin):
        """Test that only appropriate entity types are selected."""
        entities = [
            {
                "id": "func1",
                "entity_type": "function_definition",
                "content": "\n".join([f"line {i}" for i in range(15)]),  # 15 lines
            },
            {
                "id": "import1", 
                "entity_type": "import_statement",
                "content": "import os\nimport sys",  # Should be excluded
            },
            {
                "id": "class1",
                "entity_type": "class_declaration",
                "content": "\n".join([f"line {i}" for i in range(12)]),  # 12 lines
            }
        ]
        
        candidates = ingestion_plugin._filter_entities_for_ai_analysis(entities)
        
        # Should include function and class, exclude import
        assert len(candidates) == 2
        assert candidates[0]["id"] == "func1"
        assert candidates[1]["id"] == "class1"
    
    def test_content_size_filtering(self, ingestion_plugin):
        """Test filtering based on content size."""
        entities = [
            {
                "id": "too_short",
                "entity_type": "function_definition",
                "content": "def func(): pass",  # Too short
            },
            {
                "id": "just_right",
                "entity_type": "function_definition", 
                "content": "\n".join([f"    line {i}" for i in range(12)]),  # 12 lines
            },
            {
                "id": "too_long",
                "entity_type": "function_definition",
                "content": "x" * 5000,  # Too long (>4000 chars)
            }
        ]
        
        candidates = ingestion_plugin._filter_entities_for_ai_analysis(entities)
        
        # Should only include the "just right" entity
        assert len(candidates) == 1
        assert candidates[0]["id"] == "just_right"
    
    @pytest.mark.asyncio
    async def test_caching_mechanism(self, ingestion_plugin):
        """Test SHA-256 caching mechanism."""
        entity = {
            "id": "test_func",
            "entity_type": "function_definition",
            "content": "def test_func():\n    return True",
        }
        
        # Mock the Semantic Kernel call
        with patch.object(ingestion_plugin, '_call_semantic_kernel_for_description', 
                         new_callable=AsyncMock, return_value="Test description") as mock_sk:
            
            # First call should hit the API
            result1 = await ingestion_plugin._enhance_single_entity(entity)
            assert result1["ai_description"] == "Test description"
            assert result1["cache_hit"] is False
            assert mock_sk.call_count == 1
            
            # Second call should hit the cache
            result2 = await ingestion_plugin._enhance_single_entity(entity)
            assert result2["ai_description"] == "Test description"
            assert result2["cache_hit"] is True
            assert mock_sk.call_count == 1  # No additional API call
            
            # Verify cache metrics
            assert ingestion_plugin._cache_hits == 1
            assert ingestion_plugin._cache_misses == 1
    
    @pytest.mark.asyncio
    async def test_retry_logic_with_failures(self, ingestion_plugin):
        """Test exponential backoff retry logic."""
        entity = {
            "id": "test_func",
            "entity_type": "function_definition", 
            "content": "def test_func():\n    return True",
        }
        
        # Mock failures followed by success
        call_count = 0
        async def mock_call_with_failures(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary API failure")
            return "Success after retries"
        
        with patch.object(ingestion_plugin, '_call_semantic_kernel_for_description',
                         new_callable=AsyncMock, side_effect=mock_call_with_failures):
            
            result = await ingestion_plugin._enhance_single_entity(entity)
            
            # Should succeed after retries
            assert result["ai_description"] == "Success after retries"
            assert call_count == 3  # 2 failures + 1 success
    
    @pytest.mark.asyncio 
    async def test_retry_logic_max_retries_exceeded(self, ingestion_plugin):
        """Test behavior when max retries are exceeded."""
        entity = {
            "id": "test_func",
            "entity_type": "function_definition",
            "content": "def test_func():\n    return True",
        }
        
        # Mock continuous failures
        with patch.object(ingestion_plugin, '_call_semantic_kernel_for_description',
                         new_callable=AsyncMock, side_effect=Exception("Persistent failure")):
            
            result = await ingestion_plugin._enhance_single_entity(entity)
            
            # Should return None after max retries
            assert result is None
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, ingestion_plugin, sample_entities):
        """Test batch processing with concurrency control."""
        # Filter entities for processing
        candidates = ingestion_plugin._filter_entities_for_ai_analysis(sample_entities)
        
        # Mock successful Semantic Kernel calls
        descriptions = {
            "calculate_fibonacci": "Calculates Fibonacci numbers using dynamic programming",
            "DatabaseManager": "Manages database connections and query execution"
        }
        
        async def mock_call(entity):
            name = entity.get("name", "unknown")
            return descriptions.get(name, f"Description for {name}")
        
        with patch.object(ingestion_plugin, '_call_semantic_kernel_for_description',
                         new_callable=AsyncMock, side_effect=mock_call):
            
            enhanced_lookup = await ingestion_plugin._process_entities_in_batches(candidates)
            
            # Should process both candidates
            assert len(enhanced_lookup) == 2
            assert "test_func_1" in enhanced_lookup
            assert "test_class_1" in enhanced_lookup
            
            # Verify descriptions
            assert enhanced_lookup["test_func_1"]["ai_description"] == descriptions["calculate_fibonacci"]
            assert enhanced_lookup["test_class_1"]["ai_description"] == descriptions["DatabaseManager"]
    
    @pytest.mark.asyncio
    async def test_specialized_prompts(self, ingestion_plugin):
        """Test specialized prompts for different entity types."""
        test_cases = [
            ("function_definition", "Analyze this function and provide"),
            ("class_definition", "Analyze this class and provide"),
            ("method_declaration", "Analyze this method and provide"),
            ("interface_declaration", "Analyze this interface and describe"),
            ("struct_item", "Analyze this struct and describe"),
            ("constructor_declaration", "Analyze this constructor and describe"),
        ]
        
        for entity_type, expected_start in test_cases:
            entity = {"entity_type": entity_type}
            prompt = ingestion_plugin._get_entity_analysis_prompt(entity)
            assert prompt.startswith(expected_start), f"Unexpected prompt for {entity_type}: {prompt}"
    
    @pytest.mark.asyncio
    async def test_full_enhancement_pipeline(self, ingestion_plugin, sample_entities):
        """Test the complete enhancement pipeline."""
        # Mock Semantic Kernel initialization
        ingestion_plugin.kernel = Mock()
        
        # Mock successful API calls
        mock_descriptions = {
            "calculate_fibonacci": "Computes Fibonacci sequence values efficiently",
            "DatabaseManager": "Handles database connectivity and query operations"
        }
        
        async def mock_kernel_invoke(*args, **kwargs):
            entity_name = kwargs.get("arguments", {}).get("name", "unknown")
            return mock_descriptions.get(entity_name, f"AI description for {entity_name}")
        
        ingestion_plugin.kernel.invoke = AsyncMock(side_effect=mock_kernel_invoke)
        
        # Run the full enhancement pipeline
        enhanced_entities = await ingestion_plugin._enhance_entities_with_ai_descriptions(sample_entities)
        
        # Verify results
        assert len(enhanced_entities) == 4  # Same as input
        
        # Check enhanced entities
        enhanced_count = sum(1 for e in enhanced_entities if e.get("has_ai_analysis", False))
        assert enhanced_count == 3  # 2 new + 1 existing
        
        # Verify specific enhancements
        fibonacci_entity = next(e for e in enhanced_entities if e.get("name") == "calculate_fibonacci")
        assert fibonacci_entity["has_ai_analysis"] is True
        assert "ai_description" in fibonacci_entity
        assert "ai_analysis_timestamp" in fibonacci_entity
        assert fibonacci_entity["ai_model_used"] == "gpt-4o-mini"
    
    @pytest.mark.asyncio
    async def test_error_handling_graceful_degradation(self, ingestion_plugin, sample_entities):
        """Test graceful error handling when some entities fail."""
        ingestion_plugin.kernel = Mock()
        
        # Mock partial failures
        call_count = 0
        async def mock_failing_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First call fails")
            return "Successful description"
        
        ingestion_plugin.kernel.invoke = AsyncMock(side_effect=mock_failing_invoke)
        
        enhanced_entities = await ingestion_plugin._enhance_entities_with_ai_descriptions(sample_entities)
        
        # Should still return all entities
        assert len(enhanced_entities) == 4
        
        # Some should be enhanced despite failures
        enhanced_count = sum(1 for e in enhanced_entities if e.get("has_ai_analysis", False))
        assert enhanced_count >= 1  # At least the pre-existing one
    
    def test_description_length_limits(self, ingestion_plugin):
        """Test that descriptions are properly limited in length."""
        entity = {"entity_type": "function_definition"}
        
        # Test various entity types
        for entity_type in ["function_definition", "class_definition", "method_declaration"]:
            entity["entity_type"] = entity_type
            prompt = ingestion_plugin._get_entity_analysis_prompt(entity)
            
            # Prompts should be reasonable length
            assert len(prompt) < 500
            assert len(prompt) > 20
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_limits(self, ingestion_plugin):
        """Test that concurrency is properly limited."""
        # Verify semaphore is configured correctly
        assert ingestion_plugin._processing_semaphore._value == 5
        
        # Create many entities to test concurrent processing
        many_entities = []
        for i in range(30):
            entity = {
                "id": f"func_{i}",
                "entity_type": "function_definition",
                "content": "\n".join([f"line {j}" for j in range(15)]),  # 15 lines each
                "name": f"function_{i}"
            }
            many_entities.append(entity)
        
        # Mock slow API calls to test concurrency
        call_times = []
        async def slow_mock_call(*args, **kwargs):
            start_time = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)  # Simulate API delay
            call_times.append(start_time)
            return "Test description"
        
        with patch.object(ingestion_plugin, '_call_semantic_kernel_for_description',
                         new_callable=AsyncMock, side_effect=slow_mock_call):
            
            start_time = asyncio.get_event_loop().time()
            enhanced_lookup = await ingestion_plugin._process_entities_in_batches(many_entities)
            total_time = asyncio.get_event_loop().time() - start_time
            
            # Should process all entities
            assert len(enhanced_lookup) == 30
            
            # Should complete in reasonable time due to concurrency
            # Without concurrency: 30 * 0.1 = 3.0s
            # With concurrency (5 parallel): ~0.6s + overhead
            assert total_time < 2.0  # Should be much faster than sequential


if __name__ == "__main__":
    pytest.main([__file__, "-v"])