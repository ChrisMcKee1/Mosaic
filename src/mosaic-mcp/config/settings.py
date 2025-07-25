"""
Mosaic MCP Tool Configuration Settings

Centralized configuration management using Pydantic settings for all Azure services
and application components. Supports environment variables and .env files.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MosaicSettings(BaseSettings):
    """
    Main configuration class for Mosaic MCP Tool.

    Loads configuration from environment variables and .env files.
    All Azure services use managed identity authentication in production.
    """

    model_config = SettingsConfigDict(
        env_file="../../.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server Configuration
    server_host: str = Field(default="0.0.0.0", description="Server host address")
    server_port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")

    # OAuth 2.1 Authentication (FR-14)
    oauth_enabled: bool = Field(
        default=True, description="Enable OAuth 2.1 authentication"
    )
    azure_tenant_id: Optional[str] = Field(default=None, description="Azure Tenant ID")
    azure_client_id: Optional[str] = Field(default=None, description="Azure Client ID")
    azure_client_secret: Optional[str] = Field(
        default=None, description="Azure Client Secret"
    )

    # Azure OpenAI Service Configuration (Semantic Kernel standard env vars)
    azure_openai_endpoint: Optional[str] = Field(
        default=None,
        description="Azure OpenAI Service endpoint URL",
        alias="AZURE_OPENAI_ENDPOINT",
    )
    azure_openai_api_version: str = Field(
        default="2025-01-01-preview",
        description="Azure OpenAI API version",
        alias="AZURE_OPENAI_API_VERSION",
    )
    azure_openai_chat_deployment_name: str = Field(
        default="gpt-4.1",
        description="Azure OpenAI chat deployment name (main model for complex reasoning)",
        alias="AZURE_OPENAI_CHAT_DEPLOYMENT_NAME",
    )
    azure_openai_chat_deployment_name_mini: str = Field(
        default="gpt-4.1-mini",
        description="Azure OpenAI mini deployment name (for simple plugin tasks)",
        alias="AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_MINI",
    )
    azure_openai_text_embedding_deployment_name: str = Field(
        default="text-embedding-3-small",
        description="Azure OpenAI embedding deployment name",
        alias="AZURE_OPENAI_TEXT_EMBEDDING_DEPLOYMENT_NAME",
    )

    # Azure Cosmos DB Configuration (OmniRAG unified backend, managed identity)
    azure_cosmos_endpoint: Optional[str] = Field(
        default=None,
        description="Azure Cosmos DB endpoint URL",
        alias="AZURE_COSMOS_DB_ENDPOINT",
    )
    azure_cosmos_database_name: str = Field(
        default="mosaic", description="Cosmos DB database name"
    )
    azure_cosmos_container_name: str = Field(
        default="knowledge", description="Cosmos DB container name for vector search"
    )
    azure_cosmos_memory_container: str = Field(
        default="memory", description="Cosmos DB container name for memory storage"
    )

    # Golden Node Container - NEW for AI Agent architecture
    azure_cosmos_golden_nodes_container: str = Field(
        default="golden_nodes",
        description="Cosmos DB container name for Golden Node unified schema",
    )

    # Additional containers for complete plugin support
    azure_cosmos_diagrams_container: str = Field(
        default="diagrams", description="Cosmos DB container name for diagram storage"
    )
    azure_cosmos_code_entities_container: str = Field(
        default="code_entities",
        description="Cosmos DB container name for code entities",
    )
    azure_cosmos_code_relationships_container: str = Field(
        default="code_relationships",
        description="Cosmos DB container name for code relationships",
    )
    azure_cosmos_repositories_container: str = Field(
        default="repositories",
        description="Cosmos DB container name for repository metadata",
    )

    # Azure Cache for Redis Configuration (managed identity)
    azure_redis_endpoint: Optional[str] = Field(
        default=None,
        description="Azure Cache for Redis endpoint",
        alias="AZURE_REDIS_ENDPOINT",
    )
    azure_redis_port: int = Field(default=6380, description="Redis port (SSL)")
    azure_redis_ssl: bool = Field(
        default=True, description="Use SSL for Redis connection"
    )

    # Azure Machine Learning Configuration (Cross-encoder model, managed identity)
    azure_ml_endpoint_url: Optional[str] = Field(
        default=None,
        description="Azure ML endpoint URL for cross-encoder model",
        alias="AZURE_ML_ENDPOINT_URL",
    )

    # Semantic Kernel Configuration
    semantic_kernel_log_level: str = Field(
        default="INFO", description="Semantic Kernel logging level"
    )

    # Memory Configuration
    memory_consolidation_interval: int = Field(
        default=3600, description="Memory consolidation interval in seconds"
    )
    memory_importance_threshold: float = Field(
        default=0.7, description="Importance threshold for memory retention"
    )
    short_term_memory_ttl: int = Field(
        default=86400, description="Short-term memory TTL in seconds (24 hours)"
    )

    # Search and Retrieval Configuration
    max_search_results: int = Field(
        default=50, description="Maximum search results to return"
    )
    vector_search_dimensions: int = Field(
        default=1536, description="Vector embedding dimensions"
    )
    rerank_top_k: int = Field(
        default=10, description="Top K results to return after reranking"
    )

    # Performance Configuration
    max_concurrent_requests: int = Field(
        default=100, description="Maximum concurrent requests"
    )
    request_timeout: int = Field(default=30, description="Request timeout in seconds")

    def get_cosmos_config(self) -> dict:
        """Get Azure Cosmos DB configuration dictionary - UPDATED for Golden Node support."""
        return {
            "endpoint": self.azure_cosmos_endpoint,
            "database_name": self.azure_cosmos_database_name,
            # Query Server containers
            "container_name": self.azure_cosmos_container_name,  # knowledge
            "memory_container": self.azure_cosmos_memory_container,  # memory
            "diagrams_container": self.azure_cosmos_diagrams_container,  # diagrams
            # GraphDataService containers
            "code_entities_container": self.azure_cosmos_code_entities_container,
            "code_relationships_container": self.azure_cosmos_code_relationships_container,
            "repositories_container": self.azure_cosmos_repositories_container,
            # Golden Node container
            "golden_nodes_container": self.azure_cosmos_golden_nodes_container,
        }

    def get_redis_config(self) -> dict:
        """Get Azure Redis configuration dictionary."""
        return {
            "endpoint": self.azure_redis_endpoint,
            "port": self.azure_redis_port,
            "ssl": self.azure_redis_ssl,
        }

    def validate_required_settings(self) -> None:
        """Validate that required settings are provided."""
        required_settings = [
            ("AZURE_OPENAI_ENDPOINT", self.azure_openai_endpoint),
            ("AZURE_COSMOS_DB_ENDPOINT", self.azure_cosmos_endpoint),
            ("AZURE_REDIS_ENDPOINT", self.azure_redis_endpoint),
        ]

        missing_settings = [name for name, value in required_settings if not value]

        if missing_settings:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_settings)}. "
                "Please set these environment variables. Managed identity will be used for authentication."
            )
