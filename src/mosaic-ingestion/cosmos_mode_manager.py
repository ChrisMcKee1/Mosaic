#!/usr/bin/env python3
"""
Dual-Mode Cosmos DB Configuration Manager
Supports both local Docker emulator and Azure cloud modes
"""

import os
import logging
from typing import Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class CosmosMode(Enum):
    """Cosmos DB operation modes."""

    LOCAL = "local"
    AZURE = "azure"
    AUTO = "auto"


class CosmosModeManager:
    """Manages Cosmos DB configuration for local and Azure modes."""

    def __init__(self, mode: Optional[str] = None):
        """Initialize with specified mode or auto-detect."""
        self.mode = self._determine_mode(mode)
        self.config = self._load_config()

    def _determine_mode(self, mode: Optional[str] = None) -> CosmosMode:
        """Determine which mode to use."""
        if mode:
            try:
                return CosmosMode(mode.lower())
            except ValueError:
                logger.warning(f"Invalid mode '{mode}', falling back to auto-detection")

        # Check environment variable
        env_mode = os.getenv("COSMOS_MODE", "").lower()
        if env_mode in [m.value for m in CosmosMode]:
            return CosmosMode(env_mode)

        # Auto-detect based on available configuration
        if self._has_azure_config():
            logger.info("🔍 Auto-detected: Azure mode (cloud credentials found)")
            return CosmosMode.AZURE
        else:
            logger.info("🔍 Auto-detected: Local mode (no cloud credentials)")
            return CosmosMode.LOCAL

    def _has_azure_config(self) -> bool:
        """Check if Azure cloud configuration is available."""
        # Check for Azure credentials or managed identity
        azure_endpoint = os.getenv("AZURE_COSMOS_ENDPOINT") or os.getenv(
            "AZURE_COSMOS_DB_ENDPOINT"
        )

        if not azure_endpoint or "localhost" in azure_endpoint:
            return False

        # Check if we have authentication method
        has_key = bool(
            os.getenv("AZURE_COSMOS_KEY") or os.getenv("AZURE_COSMOS_DB_KEY")
        )
        has_managed_identity = (
            os.getenv("AZURE_USE_MANAGED_IDENTITY", "false").lower() == "true"
        )

        # Try to check if Azure CLI is logged in
        try:
            import subprocess

            result = subprocess.run(
                ["az", "account", "show"], capture_output=True, text=True, timeout=5
            )
            has_cli = result.returncode == 0
        except:
            has_cli = False

        return has_key or has_managed_identity or has_cli

    def _load_config(self) -> Dict[str, str]:
        """Load configuration based on current mode."""
        if self.mode == CosmosMode.LOCAL:
            return self._get_local_config()
        else:
            return self._get_azure_config()

    def _get_local_config(self) -> Dict[str, str]:
        """Get local Docker emulator configuration."""
        config = {
            "endpoint": "https://localhost:8081",
            "key": "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==",
            "database": os.getenv("MOSAIC_DATABASE_NAME", "mosaic"),
            "use_key_auth": True,
            "disable_ssl": True,  # Emulator uses self-signed certificates
        }

        logger.info("🐳 Using LOCAL mode - Docker Cosmos DB Emulator")
        logger.info(f"📍 Endpoint: {config['endpoint']}")
        logger.info(f"📊 Database: {config['database']}")

        return config

    def _get_azure_config(self) -> Dict[str, str]:
        """Get Azure cloud configuration."""
        endpoint = os.getenv("AZURE_COSMOS_ENDPOINT") or os.getenv(
            "AZURE_COSMOS_DB_ENDPOINT"
        )
        key = os.getenv("AZURE_COSMOS_KEY") or os.getenv("AZURE_COSMOS_DB_KEY")

        config = {
            "endpoint": endpoint,
            "key": key,
            "database": os.getenv("AZURE_COSMOS_DATABASE", "mosaic"),
            "use_key_auth": bool(key),
            "disable_ssl": False,
        }

        logger.info("☁️ Using AZURE mode - Cloud Cosmos DB")
        logger.info(f"📍 Endpoint: {config['endpoint']}")
        logger.info(f"📊 Database: {config['database']}")
        logger.info(
            f"🔑 Auth: {'Key-based' if config['use_key_auth'] else 'Managed Identity'}"
        )

        return config

    def get_cosmos_client(self):
        """Create and return a Cosmos DB client."""
        try:
            from azure.cosmos import CosmosClient
            from azure.identity import DefaultAzureCredential
        except ImportError as e:
            raise ImportError(f"Azure Cosmos SDK not available: {e}")

        if self.config["use_key_auth"]:
            # Use key-based authentication
            if self.config["disable_ssl"]:
                # For local emulator, disable SSL verification
                logger.info("🔓 SSL verification disabled for local emulator")
                import os
                import urllib3

                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

                # Set environment variables to disable SSL verification
                os.environ["PYTHONHTTPSVERIFY"] = "0"
                os.environ["CURL_CA_BUNDLE"] = ""

            client = CosmosClient(self.config["endpoint"], self.config["key"])
        else:
            # Use managed identity
            credential = DefaultAzureCredential()
            client = CosmosClient(self.config["endpoint"], credential)

        return client

    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to Cosmos DB."""
        try:
            client = self.get_cosmos_client()
            database = client.get_database_client(self.config["database"])
            database.read()

            message = f"✅ Successfully connected to {self.mode.value} Cosmos DB"
            logger.info(message)
            return True, message

        except Exception as e:
            message = f"❌ Failed to connect to {self.mode.value} Cosmos DB: {e}"
            logger.error(message)
            return False, message

    def get_current_mode(self) -> str:
        """Get the current mode as a string."""
        return self.mode.value

    def get_containers_config(self) -> Dict[str, str]:
        """Get container configuration."""
        return {
            "knowledge": "knowledge",
            "memory": "memory",
            "golden_nodes": "golden_nodes",
            "diagrams": "diagrams",
            "code_entities": "code_entities",
            "code_relationships": "code_relationships",
            "repositories": "repositories",
            "context": "context",
        }

    def create_containers_if_needed(self) -> bool:
        """Create containers if they don't exist (useful for local mode)."""
        try:
            client = self.get_cosmos_client()
            database = client.get_database_client(self.config["database"])

            # Try to create database if it doesn't exist (local mode)
            if self.mode == CosmosMode.LOCAL:
                try:
                    client.create_database(self.config["database"])
                    logger.info(f"Created database: {self.config['database']}")
                except:
                    pass  # Database might already exist

            containers = self.get_containers_config()
            created_count = 0

            for container_name in containers.values():
                try:
                    # Create container with default partition key
                    database.create_container(
                        id=container_name,
                        partition_key="/id",
                        offer_throughput=400,  # Minimum for local testing
                    )
                    logger.info(f"Created container: {container_name}")
                    created_count += 1
                except:
                    pass  # Container might already exist

            if created_count > 0:
                logger.info(f"Created {created_count} new containers")

            return True

        except Exception as e:
            logger.error(f"Failed to create containers: {e}")
            return False


def get_cosmos_manager(mode: Optional[str] = None) -> CosmosModeManager:
    """Factory function to get configured Cosmos manager."""
    return CosmosModeManager(mode)


# Convenience functions for backward compatibility
def get_cosmos_config(mode: Optional[str] = None) -> Dict[str, str]:
    """Get Cosmos configuration for specified mode."""
    manager = get_cosmos_manager(mode)
    return manager.config


def get_cosmos_client(mode: Optional[str] = None):
    """Get Cosmos client for specified mode."""
    manager = get_cosmos_manager(mode)
    return manager.get_cosmos_client()


def test_cosmos_connection(mode: Optional[str] = None) -> Tuple[bool, str]:
    """Test Cosmos connection for specified mode."""
    manager = get_cosmos_manager(mode)
    return manager.test_connection()


if __name__ == "__main__":
    # Test both modes
    print("Testing Cosmos DB Dual-Mode Configuration")
    print("=" * 50)

    for mode in ["local", "azure"]:
        print(f"\nTesting {mode.upper()} mode:")
        try:
            manager = CosmosModeManager(mode)
            success, message = manager.test_connection()
            print(message)
        except Exception as e:
            print(f"❌ Error testing {mode} mode: {e}")
