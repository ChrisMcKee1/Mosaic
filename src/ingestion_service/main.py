"""
Mosaic Ingestion Service - Main Entry Point

Azure Container App Job for repository ingestion and knowledge graph population.
This service runs separately from the real-time Query Server to handle heavy operations.

Usage:
- Manual execution: python -m ingestion_service.main --repository-url https://github.com/user/repo
- Scheduled execution: Configured via Azure Container App Job triggers
- Batch processing: Support for multiple repositories via configuration
"""

import asyncio
import argparse
import logging
import sys
from typing import Optional
from pathlib import Path

# Add the parent directory to the path to import mosaic modules
sys.path.append(str(Path(__file__).parent.parent))

from mosaic.config.settings import MosaicSettings
from .plugins.ingestion import IngestionPlugin

# Configure logging for the ingestion service
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("ingestion_service.log")],
)
logger = logging.getLogger(__name__)


class IngestionService:
    """
    Main Ingestion Service class for Azure Container App Job deployment.

    Handles:
    - Repository cloning and analysis
    - Multi-language AST parsing
    - Knowledge graph population
    - Batch processing capabilities
    """

    def __init__(self, settings: Optional[MosaicSettings] = None):
        """Initialize the Ingestion Service."""
        self.settings = settings or MosaicSettings()
        self.ingestion_plugin: Optional[IngestionPlugin] = None

    async def initialize(self) -> None:
        """Initialize the ingestion plugin and Azure services."""
        try:
            self.ingestion_plugin = IngestionPlugin(self.settings)
            await self.ingestion_plugin.initialize()
            logger.info("Ingestion Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ingestion Service: {e}")
            raise

    async def ingest_repository(
        self, repository_url: str, branch: str = "main"
    ) -> dict:
        """
        Ingest a single repository.

        Args:
            repository_url: Git repository URL
            branch: Git branch to process

        Returns:
            Ingestion summary with statistics
        """
        try:
            logger.info(f"Starting ingestion for: {repository_url} (branch: {branch})")

            if not self.ingestion_plugin:
                raise RuntimeError("Ingestion plugin not initialized")

            result = await self.ingestion_plugin.ingest_repository(
                repository_url, branch
            )

            logger.info(f"Ingestion completed successfully: {result}")
            return result

        except Exception as e:
            logger.error(f"Repository ingestion failed: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.ingestion_plugin:
                await self.ingestion_plugin.cleanup()
            logger.info("Ingestion Service cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main() -> None:
    """Main entry point for the Ingestion Service."""
    parser = argparse.ArgumentParser(description="Mosaic Ingestion Service")
    parser.add_argument(
        "--repository-url", required=True, help="Git repository URL to ingest"
    )
    parser.add_argument(
        "--branch", default="main", help="Git branch to process (default: main)"
    )
    parser.add_argument("--config-file", help="Path to configuration file")

    args = parser.parse_args()

    try:
        # Load settings
        settings = MosaicSettings()
        if args.config_file:
            # TODO: Implement config file loading
            logger.info(f"Using config file: {args.config_file}")

        # Create and initialize service
        service = IngestionService(settings)
        await service.initialize()

        # Perform ingestion
        result = await service.ingest_repository(args.repository_url, args.branch)

        # Log final result
        logger.info("=== INGESTION SUMMARY ===")
        logger.info(f"Repository: {result['repository_url']}")
        logger.info(f"Branch: {result['branch']}")
        logger.info(f"Entities extracted: {result['entities_extracted']}")
        logger.info(f"Relationships found: {result['relationships_found']}")
        logger.info(f"Status: {result['status']}")
        logger.info("=== END SUMMARY ===")

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Ingestion Service error: {e}")
        sys.exit(1)
    finally:
        if "service" in locals():
            await service.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
