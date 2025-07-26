"""
Container Configuration Utilities for CRUD-005

Provides utilities for configuring Cosmos DB containers with:
- Hierarchical partition keys for branch-aware data organization
- TTL policies for automatic branch cleanup
- Composite indexes for efficient branch-aware queries
- Performance monitoring and optimization

These utilities help set up optimal container configurations for
branch-aware multi-tenant scenarios following Microsoft best practices.

Author: Mosaic MCP Tool - CRUD-005 Implementation
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from azure.cosmos import CosmosClient
from azure.cosmos.database import DatabaseProxy
from azure.cosmos.container import ContainerProxy
from azure.cosmos.exceptions import CosmosResourceExistsError


logger = logging.getLogger(__name__)


class ContainerConfiguration:
    """
    Configuration class for branch-aware Cosmos DB containers.

    Provides optimal settings for hierarchical partitioning, TTL,
    and indexing based on CRUD-005 requirements.
    """

    @staticmethod
    def get_branch_aware_partition_key_definition() -> Dict[str, Any]:
        """
        Get partition key definition for branch-aware containers.

        Uses composite partition key: repository_url#branch_name
        Future-ready for hierarchical partition keys when supported.
        """
        return {"paths": ["/partition_key"], "kind": "Hash", "version": 2}

    @staticmethod
    def get_hierarchical_partition_key_definition() -> Dict[str, Any]:
        """
        Get hierarchical partition key definition (future enhancement).

        Three-level hierarchy:
        1. repository_url (tenant isolation)
        2. branch_name (branch isolation)
        3. entity_type (entity type grouping)
        """
        return {
            "paths": ["/repository_url", "/branch_name", "/entity_type"],
            "kind": "MultiHash",
            "version": 2,
        }

    @staticmethod
    def get_ttl_configuration(default_ttl: int = -1) -> Dict[str, Any]:
        """
        Get TTL configuration for branch cleanup.

        Args:
            default_ttl: Default TTL (-1 = disabled, per-item control)

        Returns:
            TTL configuration dict
        """
        return {"defaultTtl": default_ttl}

    @staticmethod
    def get_indexing_policy() -> Dict[str, Any]:
        """
        Get optimized indexing policy for branch-aware queries.

        Includes composite indexes for:
        - Branch-aware queries (repository_url + branch_name + entity_type)
        - TTL cleanup queries (ttl + cleanup_type)
        - Timestamp-based queries (repository_url + updated_at)
        - Relationship queries (entity relationships)
        """
        return {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [
                # Exclude large vectors from indexing to save RUs
                {"path": "/embedding/*"},
                {"path": "/_etag/?"},
            ],
            "compositeIndexes": [
                # Primary branch-aware query optimization
                [
                    {"path": "/repository_url", "order": "ascending"},
                    {"path": "/branch_name", "order": "ascending"},
                    {"path": "/entity_type", "order": "ascending"},
                ],
                # TTL cleanup optimization
                [
                    {"path": "/ttl", "order": "ascending"},
                    {"path": "/cleanup_type", "order": "ascending"},
                ],
                # Timestamp-based queries for incremental processing
                [
                    {"path": "/repository_url", "order": "ascending"},
                    {"path": "/updated_at", "order": "descending"},
                ],
                # File-based queries within branches
                [
                    {"path": "/repository_url", "order": "ascending"},
                    {"path": "/branch_name", "order": "ascending"},
                    {"path": "/file_context/file_path", "order": "ascending"},
                ],
                # Entity relationship queries
                [
                    {"path": "/repository_url", "order": "ascending"},
                    {"path": "/code_entity/name", "order": "ascending"},
                ],
                # Processing status queries
                [
                    {
                        "path": "/processing_metadata/processing_stage",
                        "order": "ascending",
                    },
                    {"path": "/updated_at", "order": "descending"},
                ],
            ],
            "spatialIndexes": [],
            "vectorIndexes": [],
        }

    @staticmethod
    def get_throughput_configuration(
        min_throughput: int = 400, max_throughput: int = 4000, auto_scale: bool = True
    ) -> Dict[str, Any]:
        """
        Get throughput configuration optimized for branch operations.

        Args:
            min_throughput: Minimum RU/s
            max_throughput: Maximum RU/s for autoscale
            auto_scale: Enable autoscale

        Returns:
            Throughput configuration
        """
        if auto_scale:
            return {"maxThroughput": max_throughput}
        else:
            return {"throughput": min_throughput}


class ContainerManager:
    """
    Manager for creating and configuring branch-aware Cosmos DB containers.
    """

    def __init__(self, cosmos_client: CosmosClient, database_name: str):
        """
        Initialize container manager.

        Args:
            cosmos_client: Cosmos DB client
            database_name: Database name
        """
        self.cosmos_client = cosmos_client
        self.database_name = database_name
        self.database: DatabaseProxy = cosmos_client.get_database_client(database_name)

        logger.info(f"ContainerManager initialized for database '{database_name}'")

    async def create_branch_aware_container(
        self,
        container_name: str,
        use_hierarchical_partitioning: bool = False,
        default_ttl: int = -1,
        throughput_config: Optional[Dict[str, Any]] = None,
    ) -> ContainerProxy:
        """
        Create a new container with branch-aware configuration.

        Args:
            container_name: Container name
            use_hierarchical_partitioning: Use hierarchical partition keys (future)
            default_ttl: Default TTL for container
            throughput_config: Throughput configuration

        Returns:
            Created container proxy
        """
        try:
            # Container properties
            container_properties = {
                "id": container_name,
                "partitionKey": (
                    ContainerConfiguration.get_hierarchical_partition_key_definition()
                    if use_hierarchical_partitioning
                    else ContainerConfiguration.get_branch_aware_partition_key_definition()
                ),
                "indexingPolicy": ContainerConfiguration.get_indexing_policy(),
                "defaultTtl": default_ttl,
            }

            # Throughput configuration
            if not throughput_config:
                throughput_config = (
                    ContainerConfiguration.get_throughput_configuration()
                )

            logger.info(
                f"Creating branch-aware container '{container_name}' with "
                f"hierarchical partitioning: {use_hierarchical_partitioning}, "
                f"TTL: {default_ttl}"
            )

            # Create container
            container = await self.database.create_container(
                **container_properties, **throughput_config
            )

            logger.info(f"Successfully created container '{container_name}'")
            return container

        except CosmosResourceExistsError:
            logger.info(f"Container '{container_name}' already exists")
            return self.database.get_container_client(container_name)
        except Exception as e:
            logger.error(f"Error creating container '{container_name}': {e}")
            raise

    async def update_container_indexing(
        self, container_name: str, new_indexing_policy: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update container indexing policy.

        Args:
            container_name: Container name
            new_indexing_policy: New indexing policy (optional)

        Returns:
            True if updated successfully
        """
        try:
            container = self.database.get_container_client(container_name)

            # Get current properties
            current_properties = await container.read()

            # Update indexing policy
            if not new_indexing_policy:
                new_indexing_policy = ContainerConfiguration.get_indexing_policy()

            current_properties["indexingPolicy"] = new_indexing_policy

            # Replace container properties
            await container.replace_container(current_properties)

            logger.info(f"Updated indexing policy for container '{container_name}'")
            return True

        except Exception as e:
            logger.error(f"Error updating indexing policy for '{container_name}': {e}")
            raise

    async def configure_container_ttl(
        self, container_name: str, default_ttl: int = -1
    ) -> bool:
        """
        Configure TTL for container.

        Args:
            container_name: Container name
            default_ttl: Default TTL (-1 = disabled, per-item control)

        Returns:
            True if configured successfully
        """
        try:
            container = self.database.get_container_client(container_name)

            # Get current properties
            current_properties = await container.read()

            # Update TTL
            current_properties["defaultTtl"] = default_ttl

            # Replace container properties
            await container.replace_container(current_properties)

            logger.info(
                f"Configured TTL for container '{container_name}': {default_ttl}"
            )
            return True

        except Exception as e:
            logger.error(f"Error configuring TTL for '{container_name}': {e}")
            raise

    async def get_container_metrics(self, container_name: str) -> Dict[str, Any]:
        """
        Get container performance and partition metrics.

        Args:
            container_name: Container name

        Returns:
            Container metrics
        """
        try:
            container = self.database.get_container_client(container_name)

            # Get container properties
            properties = await container.read()

            # Query for basic statistics
            stats_query = """
                SELECT 
                    COUNT(1) as total_documents,
                    MIN(c._ts) as oldest_timestamp,
                    MAX(c._ts) as newest_timestamp
                FROM c
            """

            stats_result = []
            async for item in container.query_items(
                query=stats_query, enable_cross_partition_query=True
            ):
                stats_result.append(item)

            # Query partition distribution
            partition_query = """
                SELECT c.partition_key, COUNT(1) as doc_count
                FROM c
                GROUP BY c.partition_key
            """

            partition_stats = []
            async for item in container.query_items(
                query=partition_query, enable_cross_partition_query=True
            ):
                partition_stats.append(item)

            # Compile metrics
            metrics = {
                "container_name": container_name,
                "properties": {
                    "partition_key_path": properties.get("partitionKey", {}).get(
                        "paths", []
                    ),
                    "default_ttl": properties.get("defaultTtl"),
                    "indexing_mode": properties.get("indexingPolicy", {}).get(
                        "indexingMode"
                    ),
                },
                "statistics": stats_result[0] if stats_result else {},
                "partition_distribution": partition_stats,
                "partition_count": len(partition_stats),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error getting metrics for container '{container_name}': {e}")
            raise

    async def optimize_all_containers(
        self,
        container_names: List[str],
        enable_ttl: bool = True,
        update_indexing: bool = True,
    ) -> Dict[str, bool]:
        """
        Optimize multiple containers for branch-aware operations.

        Args:
            container_names: List of container names to optimize
            enable_ttl: Enable TTL configuration
            update_indexing: Update indexing policies

        Returns:
            Dict mapping container names to success status
        """
        results = {}

        for container_name in container_names:
            try:
                logger.info(f"Optimizing container '{container_name}'")

                success = True

                # Update indexing policy
                if update_indexing:
                    await self.update_container_indexing(container_name)

                # Configure TTL
                if enable_ttl:
                    await self.configure_container_ttl(container_name, default_ttl=-1)

                results[container_name] = success

            except Exception as e:
                logger.error(f"Error optimizing container '{container_name}': {e}")
                results[container_name] = False

        logger.info(f"Container optimization completed: {results}")
        return results


class PerformanceMonitor:
    """
    Monitor for tracking branch-aware container performance.
    """

    def __init__(self, container_manager: ContainerManager):
        """
        Initialize performance monitor.

        Args:
            container_manager: Container manager instance
        """
        self.container_manager = container_manager
        self._monitoring_active = False
        self._metrics_history: List[Dict[str, Any]] = []

    async def start_monitoring(
        self, container_names: List[str], interval_seconds: int = 300
    ) -> None:
        """
        Start continuous performance monitoring.

        Args:
            container_names: Containers to monitor
            interval_seconds: Monitoring interval
        """
        self._monitoring_active = True

        logger.info(
            f"Starting performance monitoring for containers: {container_names}, "
            f"interval: {interval_seconds}s"
        )

        while self._monitoring_active:
            try:
                # Collect metrics for all containers
                for container_name in container_names:
                    metrics = await self.container_manager.get_container_metrics(
                        container_name
                    )
                    self._metrics_history.append(metrics)

                # Keep only recent metrics (last 24 hours worth)
                max_metrics = 24 * 60 * 60 // interval_seconds
                if len(self._metrics_history) > max_metrics:
                    self._metrics_history = self._metrics_history[-max_metrics:]

                # Wait for next interval
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(interval_seconds)

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self._monitoring_active = False
        logger.info("Performance monitoring stopped")

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance report.

        Returns:
            Performance report with trends and recommendations
        """
        if not self._metrics_history:
            return {"error": "No metrics collected yet"}

        # Analyze trends
        latest_metrics = self._metrics_history[-1]

        # Identify hot partitions
        partition_dist = latest_metrics.get("partition_distribution", [])
        if partition_dist:
            total_docs = sum(p.get("doc_count", 0) for p in partition_dist)
            avg_docs_per_partition = (
                total_docs / len(partition_dist) if partition_dist else 0
            )

            hot_partitions = [
                p["partition_key"]
                for p in partition_dist
                if p.get("doc_count", 0) > avg_docs_per_partition * 2
            ]

            cold_partitions = [
                p["partition_key"]
                for p in partition_dist
                if p.get("doc_count", 0) < avg_docs_per_partition * 0.5
            ]
        else:
            hot_partitions = []
            cold_partitions = []

        report = {
            "summary": {
                "total_containers_monitored": len(
                    set(m["container_name"] for m in self._metrics_history)
                ),
                "metrics_collected": len(self._metrics_history),
                "monitoring_duration_hours": len(self._metrics_history)
                * 5
                / 60,  # Assuming 5-min intervals
            },
            "latest_metrics": latest_metrics,
            "partition_analysis": {
                "hot_partitions": hot_partitions,
                "cold_partitions": cold_partitions,
                "partition_count": len(partition_dist),
                "partition_balance_score": 1.0
                - (len(hot_partitions) + len(cold_partitions))
                / max(len(partition_dist), 1),
            },
            "recommendations": self._generate_recommendations(
                latest_metrics, hot_partitions, cold_partitions
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return report

    def _generate_recommendations(
        self,
        latest_metrics: Dict[str, Any],
        hot_partitions: List[str],
        cold_partitions: List[str],
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if hot_partitions:
            recommendations.append(
                f"Consider redistributing data for hot partitions: {hot_partitions[:3]}. "
                "Use hierarchical partition keys or synthetic partition keys."
            )

        if cold_partitions:
            recommendations.append(
                f"Found {len(cold_partitions)} underutilized partitions. "
                "Consider data archival or TTL cleanup for better resource utilization."
            )

        partition_count = latest_metrics.get("partition_count", 0)
        if partition_count > 100:
            recommendations.append(
                "High partition count detected. Monitor RU consumption for cross-partition queries."
            )

        if not recommendations:
            recommendations.append(
                "Container performance is well-balanced. Continue monitoring."
            )

        return recommendations
