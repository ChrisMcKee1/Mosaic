"""
Local development configuration for Mosaic MCP Tool
This module provides configuration for running with local emulators
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class LocalDevelopmentConfig(BaseSettings):
    """Configuration for local development with emulators"""

    # Environment
    environment: str = Field(default="local", env="ENVIRONMENT")

    # Cosmos DB Emulator Configuration
    cosmos_endpoint: str = Field(
        default="https://localhost:8081", env="MOSAIC_COSMOS_ENDPOINT"
    )
    cosmos_key: str = Field(
        env="MOSAIC_COSMOS_KEY",
        description="Cosmos DB key - must be set in environment",
    )
    cosmos_database: str = Field(default="MosaicLocal", env="MOSAIC_DATABASE_NAME")
    cosmos_disable_ssl: bool = Field(
        default=True, env="COSMOS_DB_DISABLE_SSL_VERIFICATION"
    )

    # Container names
    libraries_container: str = Field(
        default="libraries", env="MOSAIC_LIBRARIES_CONTAINER"
    )
    memories_container: str = Field(default="memories", env="MOSAIC_MEMORIES_CONTAINER")
    documents_container: str = Field(
        default="documents", env="MOSAIC_DOCUMENTS_CONTAINER"
    )
    config_container: str = Field(default="config", env="MOSAIC_CONFIG_CONTAINER")

    # Redis Configuration
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: str = Field(
        env="REDIS_PASSWORD", description="Redis password - must be set in environment"
    )
    redis_ssl: bool = Field(default=False, env="REDIS_SSL")

    # Azure Storage (Azurite)
    storage_connection_string: str = Field(
        env="AZURE_STORAGE_CONNECTION_STRING",
        description="Azure Storage connection string - must be set in environment",
    )

    # Azure OpenAI (still requires cloud service)
    openai_endpoint: Optional[str] = Field(default=None, env="AZURE_OPENAI_ENDPOINT")
    openai_api_version: str = Field(
        default="2025-01-01-preview", env="AZURE_OPENAI_API_VERSION"
    )
    openai_chat_deployment: str = Field(
        default="gpt-4.1", env="AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
    )
    openai_chat_deployment_mini: str = Field(
        default="gpt-4.1-mini", env="AZURE_OPENAI_CHAT_DEPLOYMENT_NAME_MINI"
    )
    openai_embedding_deployment: str = Field(
        default="text-embedding-3-small", env="AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    )

    # Authentication
    use_managed_identity: bool = Field(default=False, env="AZURE_USE_MANAGED_IDENTITY")
    tenant_id: Optional[str] = Field(default=None, env="AZURE_TENANT_ID")
    client_id: Optional[str] = Field(default=None, env="AZURE_CLIENT_ID")
    client_secret: Optional[str] = Field(default=None, env="AZURE_CLIENT_SECRET")

    # ML Configuration (local mode)
    ml_reranking_mode: str = Field(default="local", env="ML_RERANKING_MODE")
    ml_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-12-v2", env="ML_MODEL_NAME"
    )

    # MCP Server Configuration
    mcp_host: str = Field(default="127.0.0.1", env="MCP_SERVER_HOST")
    mcp_port: int = Field(default=8080, env="MCP_SERVER_PORT")

    # Logging
    log_level: str = Field(default="DEBUG", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = False

    def get_cosmos_client_kwargs(self) -> dict:
        """Get Cosmos DB client configuration for local emulator"""
        kwargs = {
            "url": self.cosmos_endpoint,
            "credential": self.cosmos_key,
        }

        # For local emulator, disable SSL verification
        if self.cosmos_disable_ssl and "localhost" in self.cosmos_endpoint:
            import ssl

            kwargs["connection_policy"] = {
                "ssl_configuration": ssl._create_unverified_context()
            }

        return kwargs

    def get_redis_connection_string(self) -> str:
        """Get Redis connection string for local development"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"

    def validate_required_settings(self) -> list[str]:
        """Validate that required settings are configured"""
        errors = []

        if not self.openai_endpoint:
            errors.append(
                "AZURE_OPENAI_ENDPOINT is required even for local development"
            )

        if not self.cosmos_key:
            errors.append("MOSAIC_COSMOS_KEY is required")

        if not self.redis_password:
            errors.append("REDIS_PASSWORD is required")

        if not self.storage_connection_string:
            errors.append("AZURE_STORAGE_CONNECTION_STRING is required")

        return errors


# Global instance
local_config = LocalDevelopmentConfig()


def get_local_config() -> LocalDevelopmentConfig:
    """Get the local development configuration"""
    return local_config


def is_local_development() -> bool:
    """Check if running in local development mode"""
    return local_config.environment.lower() in ("local", "development", "dev")
