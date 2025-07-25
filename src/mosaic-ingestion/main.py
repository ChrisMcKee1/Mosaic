"""
Mosaic Ingestion Service - Main Entry Point

Azure Container App Job for repository ingestion and knowledge graph population.
This service runs separately from the real-time Query Server to handle heavy operations.

Usage:
- Manual execution: python -m mosaic_ingestion.main --repository-url https://github.com/user/repo
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

from mosaic_mcp.config.settings import MosaicSettings
from .orchestrator import MosaicMagenticOrchestrator

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

    Uses Microsoft Semantic Kernel Magentic orchestration to coordinate
    specialized AI agents for comprehensive repository ingestion:
    - GitSleuth: Repository cloning and analysis
    - CodeParser: Multi-language AST parsing and entity extraction
    - GraphArchitect: Relationship mapping and graph construction
    - DocuWriter: AI-powered enrichment and documentation
    - GraphAuditor: Quality assurance and validation
    """

    def __init__(self, settings: Optional[MosaicSettings] = None):
        """Initialize the Ingestion Service with Magentic orchestration."""
        self.settings = settings or MosaicSettings()
        self.orchestrator: Optional[MosaicMagenticOrchestrator] = None

    async def initialize(self) -> None:
        """Initialize the Magentic orchestration and Azure services."""
        try:
            logger.info(
                "🚀 Initializing Magentic orchestration for AI agent coordination..."
            )
            self.orchestrator = MosaicMagenticOrchestrator(self.settings)
            logger.info("✅ Ingestion Service initialized with Magentic orchestration")
        except Exception as e:
            logger.error(
                f"❌ Failed to initialize Ingestion Service with Magentic orchestration: {e}"
            )
            raise

    async def ingest_repository(
        self, repository_url: str, branch: str = "main"
    ) -> dict:
        """
        Ingest a repository using Magentic AI agent orchestration.

        The StandardMagenticManager will coordinate specialized agents to:
        1. Clone and analyze the repository structure
        2. Parse code and extract Golden Node entities
        3. Map relationships and build knowledge graph
        4. Enrich entities with AI-powered insights
        5. Validate and prepare for Cosmos DB storage

        Args:
            repository_url: Git repository URL
            branch: Git branch to process

        Returns:
            Comprehensive ingestion results with Golden Node entities
        """
        try:
            logger.info(
                f"🚀 Starting Magentic AI agent ingestion for: {repository_url} (branch: {branch})"
            )

            if not self.orchestrator:
                raise RuntimeError("Magentic orchestrator not initialized")

            # Execute the full AI agent orchestration
            result = await self.orchestrator.orchestrate_repository_ingestion(
                repository_url, branch
            )

            logger.info(f"✅ Magentic orchestration completed successfully: {result}")
            return result

        except Exception as e:
            logger.error(f"❌ Magentic AI agent ingestion failed: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup Magentic orchestration resources."""
        try:
            if self.orchestrator:
                await self.orchestrator.cleanup()
            logger.info("✅ Ingestion Service cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


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
        logger.info("🎯 === MAGENTIC ORCHESTRATION SUMMARY ===")
        logger.info(f"🔗 Repository: {result['repository_url']}")
        logger.info(f"🌿 Branch: {result['branch']}")
        logger.info(
            f"⚡ Framework: {result.get('framework', 'microsoft_semantic_kernel')}"
        )
        logger.info(f"🤖 Mode: {result.get('mode', 'magentic_orchestration')}")
        logger.info(f"🎯 Agents executed: {result.get('agents_executed', 0)}")
        logger.info(
            f"⏱️  Processing time: {result.get('processing_time_seconds', 0):.2f}s"
        )
        logger.info(f"✅ Status: {result['status']}")
        logger.info(
            f"📝 Result preview: {result.get('orchestration_result', 'N/A')[:200]}..."
        )
        logger.info("🎯 === END MAGENTIC ORCHESTRATION SUMMARY ===")

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
