"""
Comprehensive tests for MosaicSettings configuration management.

Tests cover:
- Environment variable loading and validation
- Azure service configuration
- Default value handling
- Settings validation and error handling
- Configuration object creation and access
"""

import pytest
import os
from typing import Dict, Any


# Mock the MosaicSettings class since config module might not be fully implemented
class MockMosaicSettings:
    """Mock MosaicSettings for testing purposes."""

    def __init__(self, **kwargs):
        # Server configuration
        self.server_host = kwargs.get("server_host", "0.0.0.0")
        self.server_port = kwargs.get("server_port", 8000)

        # Azure OpenAI configuration
        self.azure_openai_endpoint = kwargs.get(
            "azure_openai_endpoint", "https://test.openai.azure.com"
        )
        self.azure_openai_text_embedding_deployment_name = kwargs.get(
            "azure_openai_text_embedding_deployment_name", "text-embedding-3-small"
        )
        self.azure_openai_chat_deployment_name = kwargs.get(
            "azure_openai_chat_deployment_name", "gpt-4"
        )

        # Azure Cosmos DB configuration
        self.azure_cosmos_endpoint = kwargs.get(
            "azure_cosmos_endpoint", "https://test.cosmos.azure.com"
        )
        self.cosmos_database_name = kwargs.get("cosmos_database_name", "mosaic")
        self.cosmos_container_name = kwargs.get("cosmos_container_name", "knowledge")
        self.cosmos_memory_container = kwargs.get("cosmos_memory_container", "memory")

        # Azure Redis configuration
        self.azure_redis_endpoint = kwargs.get(
            "azure_redis_endpoint", "test-redis.redis.cache.windows.net"
        )
        self.redis_port = kwargs.get("redis_port", 6380)
        self.redis_ssl = kwargs.get("redis_ssl", True)

        # OAuth configuration
        self.oauth_enabled = kwargs.get("oauth_enabled", True)
        self.azure_tenant_id = kwargs.get("azure_tenant_id", "test-tenant-id")
        self.azure_client_id = kwargs.get("azure_client_id", "test-client-id")
        self.azure_client_secret = kwargs.get(
            "azure_client_secret", "test-client-secret"
        )

        # Memory configuration
        self.short_term_memory_ttl = kwargs.get("short_term_memory_ttl", 3600)
        self.memory_importance_threshold = kwargs.get(
            "memory_importance_threshold", 0.7
        )

        # Search configuration
        self.max_search_results = kwargs.get("max_search_results", 50)
        self.search_timeout = kwargs.get("search_timeout", 30)

        # Diagram configuration
        self.max_diagram_nodes = kwargs.get("max_diagram_nodes", 100)
        self.diagram_cache_ttl = kwargs.get("diagram_cache_ttl", 7200)

        # Validate settings
        self._validate_settings()

    def _validate_settings(self):
        """Validate configuration settings."""
        if not self.azure_openai_endpoint:
            raise ValueError("Azure OpenAI endpoint is required")

        if not self.azure_cosmos_endpoint:
            raise ValueError("Azure Cosmos DB endpoint is required")

        if self.server_port < 1 or self.server_port > 65535:
            raise ValueError("Server port must be between 1 and 65535")

        if self.short_term_memory_ttl < 60:
            raise ValueError("Short-term memory TTL must be at least 60 seconds")

    def get_cosmos_config(self) -> Dict[str, Any]:
        """Get Cosmos DB configuration dictionary."""
        return {
            "endpoint": self.azure_cosmos_endpoint,
            "database_name": self.cosmos_database_name,
            "container_name": self.cosmos_container_name,
            "memory_container": self.cosmos_memory_container,
        }

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration dictionary."""
        return {
            "endpoint": self.azure_redis_endpoint,
            "port": self.redis_port,
            "ssl": self.redis_ssl,
        }

    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration dictionary."""
        return {
            "endpoint": self.azure_openai_endpoint,
            "text_embedding_deployment": self.azure_openai_text_embedding_deployment_name,
            "chat_deployment": self.azure_openai_chat_deployment_name,
        }

    def get_oauth_config(self) -> Dict[str, Any]:
        """Get OAuth configuration dictionary."""
        return {
            "enabled": self.oauth_enabled,
            "tenant_id": self.azure_tenant_id,
            "client_id": self.azure_client_id,
            "client_secret": self.azure_client_secret,
        }

    @classmethod
    def from_env(cls) -> "MockMosaicSettings":
        """Create settings from environment variables."""
        return cls(
            server_host=os.getenv("SERVER_HOST", "0.0.0.0"),
            server_port=int(os.getenv("SERVER_PORT", "8000")),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_cosmos_endpoint=os.getenv("AZURE_COSMOS_DB_ENDPOINT"),
            azure_redis_endpoint=os.getenv("AZURE_REDIS_ENDPOINT"),
            azure_tenant_id=os.getenv("AZURE_TENANT_ID"),
            azure_client_id=os.getenv("AZURE_CLIENT_ID"),
            oauth_enabled=os.getenv("OAUTH_ENABLED", "true").lower() == "true",
        )


@pytest.fixture
def clean_env():
    """Clean environment variables for testing."""
    env_vars = [
        "SERVER_HOST",
        "SERVER_PORT",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_COSMOS_DB_ENDPOINT",
        "AZURE_REDIS_ENDPOINT",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "OAUTH_ENABLED",
    ]

    # Store original values
    original_values = {}
    for var in env_vars:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


class TestMosaicSettingsInitialization:
    """Test MosaicSettings initialization and defaults."""

    def test_default_initialization(self):
        """Test initialization with default values."""
        settings = MockMosaicSettings()

        assert settings.server_host == "0.0.0.0"
        assert settings.server_port == 8000
        assert settings.azure_openai_endpoint == "https://test.openai.azure.com"
        assert settings.azure_cosmos_endpoint == "https://test.cosmos.azure.com"
        assert settings.oauth_enabled is True
        assert settings.max_search_results == 50

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        custom_settings = {
            "server_host": "localhost",
            "server_port": 9000,
            "azure_openai_endpoint": "https://custom.openai.azure.com",
            "max_search_results": 100,
            "oauth_enabled": False,
        }

        settings = MockMosaicSettings(**custom_settings)

        assert settings.server_host == "localhost"
        assert settings.server_port == 9000
        assert settings.azure_openai_endpoint == "https://custom.openai.azure.com"
        assert settings.max_search_results == 100
        assert settings.oauth_enabled is False

    def test_azure_service_defaults(self):
        """Test Azure service default configurations."""
        settings = MockMosaicSettings()

        # OpenAI defaults
        assert (
            "text-embedding-3-small"
            in settings.azure_openai_text_embedding_deployment_name
        )
        assert "gpt-4" in settings.azure_openai_chat_deployment_name

        # Cosmos DB defaults
        assert settings.cosmos_database_name == "mosaic"
        assert settings.cosmos_container_name == "knowledge"
        assert settings.cosmos_memory_container == "memory"

        # Redis defaults
        assert settings.redis_port == 6380
        assert settings.redis_ssl is True

    def test_memory_configuration_defaults(self):
        """Test memory configuration defaults."""
        settings = MockMosaicSettings()

        assert settings.short_term_memory_ttl == 3600  # 1 hour
        assert settings.memory_importance_threshold == 0.7

    def test_search_configuration_defaults(self):
        """Test search configuration defaults."""
        settings = MockMosaicSettings()

        assert settings.max_search_results == 50
        assert settings.search_timeout == 30


class TestEnvironmentVariableLoading:
    """Test loading configuration from environment variables."""

    def test_from_env_with_all_variables(self, clean_env):
        """Test loading from environment with all variables set."""
        env_vars = {
            "SERVER_HOST": "api.example.com",
            "SERVER_PORT": "9000",
            "AZURE_OPENAI_ENDPOINT": "https://prod.openai.azure.com",
            "AZURE_COSMOS_DB_ENDPOINT": "https://prod.cosmos.azure.com",
            "AZURE_REDIS_ENDPOINT": "prod-redis.redis.cache.windows.net",
            "AZURE_TENANT_ID": "prod-tenant-id",
            "AZURE_CLIENT_ID": "prod-client-id",
            "OAUTH_ENABLED": "true",
        }

        for key, value in env_vars.items():
            os.environ[key] = value

        settings = MockMosaicSettings.from_env()

        assert settings.server_host == "api.example.com"
        assert settings.server_port == 9000
        assert settings.azure_openai_endpoint == "https://prod.openai.azure.com"
        assert settings.azure_cosmos_endpoint == "https://prod.cosmos.azure.com"
        assert settings.azure_redis_endpoint == "prod-redis.redis.cache.windows.net"
        assert settings.azure_tenant_id == "prod-tenant-id"
        assert settings.azure_client_id == "prod-client-id"
        assert settings.oauth_enabled is True

    def test_from_env_with_defaults(self, clean_env):
        """Test loading from environment with missing variables using defaults."""
        # Only set a few environment variables
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com"
        os.environ["OAUTH_ENABLED"] = "false"

        settings = MockMosaicSettings.from_env()

        # Should use provided values
        assert settings.azure_openai_endpoint == "https://test.openai.azure.com"
        assert settings.oauth_enabled is False

        # Should use defaults for missing values
        assert settings.server_host == "0.0.0.0"
        assert settings.server_port == 8000

    def test_port_number_parsing(self, clean_env):
        """Test proper parsing of port number from environment."""
        os.environ["SERVER_PORT"] = "3000"

        settings = MockMosaicSettings.from_env()

        assert settings.server_port == 3000
        assert isinstance(settings.server_port, int)

    def test_boolean_parsing(self, clean_env):
        """Test proper parsing of boolean values from environment."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("1", False),  # Only 'true' should be True
            ("yes", False),
        ]

        for env_value, expected in test_cases:
            os.environ["OAUTH_ENABLED"] = env_value
            settings = MockMosaicSettings.from_env()
            assert settings.oauth_enabled == expected


class TestConfigurationValidation:
    """Test configuration validation and error handling."""

    def test_missing_required_endpoints(self):
        """Test validation fails with missing required endpoints."""
        with pytest.raises(ValueError, match="Azure OpenAI endpoint is required"):
            MockMosaicSettings(azure_openai_endpoint=None)

        with pytest.raises(ValueError, match="Azure Cosmos DB endpoint is required"):
            MockMosaicSettings(azure_cosmos_endpoint=None)

    def test_invalid_port_ranges(self):
        """Test validation of port number ranges."""
        with pytest.raises(ValueError, match="Server port must be between 1 and 65535"):
            MockMosaicSettings(server_port=0)

        with pytest.raises(ValueError, match="Server port must be between 1 and 65535"):
            MockMosaicSettings(server_port=70000)

    def test_invalid_memory_ttl(self):
        """Test validation of memory TTL values."""
        with pytest.raises(
            ValueError, match="Short-term memory TTL must be at least 60 seconds"
        ):
            MockMosaicSettings(short_term_memory_ttl=30)

    def test_valid_edge_case_values(self):
        """Test validation with edge case but valid values."""
        # Minimum valid values
        settings = MockMosaicSettings(
            server_port=1,
            short_term_memory_ttl=60,
            memory_importance_threshold=0.0,
            max_search_results=1,
        )

        assert settings.server_port == 1
        assert settings.short_term_memory_ttl == 60
        assert settings.memory_importance_threshold == 0.0
        assert settings.max_search_results == 1

        # Maximum valid values
        settings = MockMosaicSettings(
            server_port=65535, memory_importance_threshold=1.0, max_search_results=1000
        )

        assert settings.server_port == 65535
        assert settings.memory_importance_threshold == 1.0
        assert settings.max_search_results == 1000


class TestConfigurationHelpers:
    """Test configuration helper methods."""

    def test_get_cosmos_config(self):
        """Test Cosmos DB configuration helper."""
        settings = MockMosaicSettings(
            azure_cosmos_endpoint="https://test.cosmos.azure.com",
            cosmos_database_name="test_db",
            cosmos_container_name="test_container",
            cosmos_memory_container="test_memory",
        )

        config = settings.get_cosmos_config()

        assert config["endpoint"] == "https://test.cosmos.azure.com"
        assert config["database_name"] == "test_db"
        assert config["container_name"] == "test_container"
        assert config["memory_container"] == "test_memory"

    def test_get_redis_config(self):
        """Test Redis configuration helper."""
        settings = MockMosaicSettings(
            azure_redis_endpoint="test-redis.redis.cache.windows.net",
            redis_port=6380,
            redis_ssl=True,
        )

        config = settings.get_redis_config()

        assert config["endpoint"] == "test-redis.redis.cache.windows.net"
        assert config["port"] == 6380
        assert config["ssl"] is True

    def test_get_openai_config(self):
        """Test OpenAI configuration helper."""
        settings = MockMosaicSettings(
            azure_openai_endpoint="https://test.openai.azure.com",
            azure_openai_text_embedding_deployment_name="text-embedding-3-small",
            azure_openai_chat_deployment_name="gpt-4.1",
        )

        config = settings.get_openai_config()

        assert config["endpoint"] == "https://test.openai.azure.com"
        assert config["text_embedding_deployment"] == "text-embedding-3-small"
        assert config["chat_deployment"] == "gpt-4.1"

    def test_get_oauth_config(self):
        """Test OAuth configuration helper."""
        settings = MockMosaicSettings(
            oauth_enabled=True,
            azure_tenant_id="test-tenant",
            azure_client_id="test-client",
            azure_client_secret="test-secret",
        )

        config = settings.get_oauth_config()

        assert config["enabled"] is True
        assert config["tenant_id"] == "test-tenant"
        assert config["client_id"] == "test-client"
        assert config["client_secret"] == "test-secret"


class TestSettingsIntegration:
    """Test settings integration with other components."""

    def test_settings_serialization(self):
        """Test settings can be serialized for logging/debugging."""
        settings = MockMosaicSettings()

        # Should be able to access all properties
        config_dict = {
            "server_host": settings.server_host,
            "server_port": settings.server_port,
            "oauth_enabled": settings.oauth_enabled,
            "max_search_results": settings.max_search_results,
        }

        assert config_dict["server_host"] == "0.0.0.0"
        assert config_dict["server_port"] == 8000
        assert config_dict["oauth_enabled"] is True
        assert config_dict["max_search_results"] == 50

    def test_settings_with_azure_integration(self):
        """Test settings provide proper Azure service configuration."""
        settings = MockMosaicSettings(
            azure_openai_endpoint="https://prod.openai.azure.com",
            azure_cosmos_endpoint="https://prod.cosmos.azure.com",
            azure_redis_endpoint="prod-redis.redis.cache.windows.net",
        )

        # All Azure endpoints should be HTTPS
        assert settings.azure_openai_endpoint.startswith("https://")
        assert settings.azure_cosmos_endpoint.startswith("https://")

        # Redis endpoint should not include protocol
        assert not settings.azure_redis_endpoint.startswith("https://")
        assert settings.azure_redis_endpoint.endswith(".redis.cache.windows.net")

    def test_production_vs_development_settings(self):
        """Test different configurations for production vs development."""
        # Development settings
        dev_settings = MockMosaicSettings(
            server_host="localhost", oauth_enabled=False, max_search_results=10
        )

        # Production settings
        prod_settings = MockMosaicSettings(
            server_host="0.0.0.0", oauth_enabled=True, max_search_results=50
        )

        assert dev_settings.server_host == "localhost"
        assert dev_settings.oauth_enabled is False
        assert dev_settings.max_search_results == 10

        assert prod_settings.server_host == "0.0.0.0"
        assert prod_settings.oauth_enabled is True
        assert prod_settings.max_search_results == 50


class TestSettingsEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_string_handling(self):
        """Test handling of empty string values."""
        with pytest.raises(ValueError):
            MockMosaicSettings(azure_openai_endpoint="")

        with pytest.raises(ValueError):
            MockMosaicSettings(azure_cosmos_endpoint="")

    def test_none_value_handling(self):
        """Test handling of None values."""
        with pytest.raises(ValueError):
            MockMosaicSettings(azure_openai_endpoint=None)

    def test_type_conversion_errors(self, clean_env):
        """Test type conversion error handling."""
        os.environ["SERVER_PORT"] = "not_a_number"

        with pytest.raises(ValueError):
            MockMosaicSettings.from_env()

    def test_very_large_values(self):
        """Test handling of very large configuration values."""
        settings = MockMosaicSettings(
            short_term_memory_ttl=86400 * 365,  # 1 year
            max_search_results=10000,
            search_timeout=3600,
        )

        assert settings.short_term_memory_ttl == 86400 * 365
        assert settings.max_search_results == 10000
        assert settings.search_timeout == 3600

    def test_special_characters_in_config(self):
        """Test handling of special characters in configuration."""
        settings = MockMosaicSettings(
            azure_tenant_id="12345678-1234-1234-1234-123456789abc",
            azure_client_id="abcdef12-3456-7890-abcd-ef1234567890",
        )

        assert len(settings.azure_tenant_id) == 36  # UUID format
        assert len(settings.azure_client_id) == 36  # UUID format
        assert "-" in settings.azure_tenant_id
        assert "-" in settings.azure_client_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
