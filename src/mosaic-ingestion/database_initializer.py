"""
Database Initialization Module for Mosaic Services
Provides Entity Framework-style database initialization that checks for existence
and creates containers if they don't exist, regardless of local/Azure mode.
"""

import logging
from typing import Dict, List, Any, Optional
from azure.cosmos import ContainerProxy, DatabaseProxy
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from cosmos_mode_manager import CosmosModeManager

logger = logging.getLogger(__name__)


class MosaicDatabaseInitializer:
    """
    Entity Framework-style database initializer for Mosaic services.
    Ensures all required containers exist before service operations.
    """

    def __init__(self, service_name: str):
        """Initialize with service-specific container requirements."""
        self.service_name = service_name
        self.cosmos_manager = CosmosModeManager()
        self.cosmos_client = self.cosmos_manager.get_cosmos_client()
        self.database: Optional[DatabaseProxy] = None
        self.containers: Dict[str, ContainerProxy] = {}

        # Define container requirements per service
        self.container_definitions = self._get_container_definitions()

    def _get_container_definitions(self) -> List[Dict[str, Any]]:
        """Get container definitions based on service type."""

        # Common containers for all services
        common_containers = [
            {
                "id": "knowledge",
                "partition_key": "/id",
                "description": "Knowledge base documents",
            },
            {
                "id": "memory",
                "partition_key": "/id",
                "description": "Memory and context storage",
            },
            {
                "id": "repositories",
                "partition_key": "/id",
                "description": "Repository metadata",
            },
            {
                "id": "diagrams",
                "partition_key": "/id",
                "description": "Diagram and visualization data",
            },
        ]

        # Service-specific containers
        if self.service_name == "mosaic-ingestion":
            return common_containers + [
                {
                    "id": "entities",
                    "partition_key": "/id",
                    "description": "Code entities and components",
                },
                {
                    "id": "relationships",
                    "partition_key": "/id",
                    "description": "Entity relationships",
                },
                {
                    "id": "codefiles",
                    "partition_key": "/id",
                    "description": "Source code file metadata",
                },
                {
                    "id": "analysis",
                    "partition_key": "/id",
                    "description": "Code analysis results",
                },
            ]

        elif self.service_name == "mosaic-ui":
            return common_containers + [
                {
                    "id": "entities",
                    "partition_key": "/id",
                    "description": "UI-accessible entities",
                },
                {
                    "id": "relationships",
                    "partition_key": "/id",
                    "description": "UI-accessible relationships",
                },
                {
                    "id": "sessions",
                    "partition_key": "/id",
                    "description": "User session data",
                },
            ]

        elif self.service_name == "mosaic-mcp":
            return common_containers + [
                {
                    "id": "tools",
                    "partition_key": "/id",
                    "description": "MCP tool definitions",
                },
                {
                    "id": "requests",
                    "partition_key": "/id",
                    "description": "MCP request logs",
                },
                {
                    "id": "capabilities",
                    "partition_key": "/id",
                    "description": "MCP capabilities",
                },
            ]

        else:
            # Default to common containers
            return common_containers

    async def initialize_database(self) -> bool:
        """
        Initialize database and all required containers.
        Entity Framework pattern: Create if not exists.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info(f"🚀 Initializing database for {self.service_name}")
            logger.info(f"📍 Mode: {self.cosmos_manager.get_current_mode()}")
            logger.info(f"📍 Endpoint: {self.cosmos_manager.config['endpoint']}")
            logger.info(f"📊 Database: {self.cosmos_manager.config['database']}")

            # Step 1: Ensure database exists
            if not await self._ensure_database_exists():
                return False

            # Step 2: Ensure all required containers exist
            if not await self._ensure_containers_exist():
                return False

            logger.info(f"✅ Database initialization completed for {self.service_name}")
            logger.info(f"📦 Available containers: {list(self.containers.keys())}")

            return True

        except Exception as e:
            logger.error(f"❌ Critical error during database initialization: {e}")
            return False

    async def _ensure_database_exists(self) -> bool:
        """Ensure the database exists, create if it doesn't."""
        try:
            database_name = self.cosmos_manager.config["database"]

            # Try to get existing database
            try:
                self.database = self.cosmos_client.get_database_client(database_name)
                self.database.read()
                logger.info(f"✅ Database '{database_name}' exists")
                return True

            except CosmosResourceNotFoundError:
                # Database doesn't exist, create it
                logger.info(f"📦 Creating database '{database_name}'...")
                self.database = self.cosmos_client.create_database(id=database_name)
                logger.info(f"✅ Database '{database_name}' created successfully")
                return True

        except Exception as e:
            logger.error(f"❌ Failed to ensure database exists: {e}")
            return False

    async def _ensure_containers_exist(self) -> bool:
        """Ensure all required containers exist, create if they don't."""
        created_count = 0
        existing_count = 0
        failed_count = 0

        for container_def in self.container_definitions:
            container_id = container_def["id"]
            partition_key = container_def["partition_key"]
            description = container_def["description"]

            try:
                # Try to get existing container
                try:
                    container = self.database.get_container_client(container_id)
                    container.read()
                    self.containers[container_id] = container
                    existing_count += 1
                    logger.debug(f"✅ Container '{container_id}' exists")

                except CosmosResourceNotFoundError:
                    # Container doesn't exist, create it
                    logger.info(
                        f"📦 Creating container '{container_id}' - {description}"
                    )

                    container = self.database.create_container(
                        id=container_id, partition_key=partition_key
                    )

                    # Add initial test document to verify container works
                    test_doc = {
                        "id": f"_init_{container_id}",
                        "type": "initialization_marker",
                        "service": self.service_name,
                        "created": "2025-07-26T21:30:00Z",
                        "description": description,
                    }

                    container.upsert_item(body=test_doc)
                    self.containers[container_id] = container
                    created_count += 1
                    logger.info(f"✅ Container '{container_id}' created successfully")

            except Exception as e:
                logger.error(
                    f"❌ Failed to ensure container '{container_id}' exists: {e}"
                )
                failed_count += 1

        total_required = len(self.container_definitions)
        total_available = created_count + existing_count

        logger.info("📊 Container Summary:")
        logger.info(f"   • Existing: {existing_count}")
        logger.info(f"   • Created: {created_count}")
        logger.info(f"   • Failed: {failed_count}")
        logger.info(f"   • Available: {total_available}/{total_required}")

        # Consider initialization successful if we have at least core containers
        min_required = min(4, total_required)  # At least 4 core containers

        if total_available >= min_required:
            logger.info(f"💡 Sufficient containers available for {self.service_name}")
            return True
        else:
            logger.error(
                f"❌ Insufficient containers for {self.service_name} ({total_available}/{min_required} minimum)"
            )
            return False

    def get_container(self, container_name: str) -> Optional[ContainerProxy]:
        """Get a specific container client."""
        return self.containers.get(container_name)

    def get_all_containers(self) -> Dict[str, ContainerProxy]:
        """Get all available container clients."""
        return self.containers.copy()

    def is_container_available(self, container_name: str) -> bool:
        """Check if a specific container is available."""
        return container_name in self.containers


def create_database_initializer(service_name: str) -> MosaicDatabaseInitializer:
    """
    Factory function to create a database initializer for a specific service.

    Args:
        service_name: Name of the service ("mosaic-ingestion", "mosaic-ui", or "mosaic-mcp")

    Returns:
        Configured MosaicDatabaseInitializer instance
    """
    return MosaicDatabaseInitializer(service_name)
