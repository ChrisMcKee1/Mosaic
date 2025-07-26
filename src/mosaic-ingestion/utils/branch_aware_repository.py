"""
Branch-Aware Repository Base Class for CRUD-005

Implements Azure Cosmos DB branch-aware partitioning strategy with:
- Hierarchical partition keys (/repository_url/branch_name/entity_type)
- Cross-partition query support for merge/conflict detection
- TTL configuration for automatic branch cleanup
- Composite indexing for efficient branch-aware queries
- Performance monitoring for partition distribution

Design follows Microsoft Cosmos DB best practices for multi-tenant applications
with branch isolation and optimal query performance.

Author: Mosaic MCP Tool - CRUD-005 Implementation
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from azure.cosmos import CosmosClient
from azure.cosmos.container import ContainerProxy
from azure.cosmos.database import DatabaseProxy
from azure.cosmos.exceptions import CosmosResourceNotFoundError


logger = logging.getLogger(__name__)


@dataclass
class BranchPartitionKey:
    """
    Hierarchical partition key for branch-aware data organization.

    Implements the pattern: /repository_url/branch_name/entity_type
    """

    repository_url: str
    branch_name: str
    entity_type: Optional[str] = None

    def to_composite_key(self) -> str:
        """Generate composite partition key string."""
        if self.entity_type:
            return f"{self.repository_url}#{self.branch_name}#{self.entity_type}"
        return f"{self.repository_url}#{self.branch_name}"

    def to_hierarchical_key(self) -> List[str]:
        """Generate hierarchical partition key array for Cosmos DB HPK."""
        if self.entity_type:
            return [self.repository_url, self.branch_name, self.entity_type]
        return [self.repository_url, self.branch_name]

    @classmethod
    def from_composite_key(cls, composite_key: str) -> "BranchPartitionKey":
        """Parse composite partition key string."""
        parts = composite_key.split("#")
        if len(parts) >= 3:
            return cls(parts[0], parts[1], parts[2])
        elif len(parts) == 2:
            return cls(parts[0], parts[1])
        else:
            raise ValueError(f"Invalid composite partition key: {composite_key}")


@dataclass
@dataclass
class TTLConfiguration:
    """TTL configuration for branch-aware cleanup."""

    # Default TTL for active branches (30 days)
    active_branch_ttl: int = 30 * 24 * 60 * 60

    # TTL for stale branches (7 days)
    stale_branch_ttl: int = 7 * 24 * 60 * 60

    # TTL for deleted branches (1 day)
    deleted_branch_ttl: int = 24 * 60 * 60

    # TTL for merged branches (30 days)
    merged_branch_ttl: int = 30 * 24 * 60 * 60

    # Container-level default TTL (-1 = off by default, per-item control)
    container_default_ttl: int = -1


@dataclass
class PartitionMetrics:
    """Partition distribution and performance metrics."""

    partition_count: int = 0
    total_documents: int = 0
    total_size_bytes: int = 0
    hot_partitions: List[str] = field(default_factory=list)
    cold_partitions: List[str] = field(default_factory=list)
    cross_partition_queries: int = 0
    average_query_latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BranchAwareRepository(ABC):
    """
    Abstract base repository for branch-aware Cosmos DB operations.

    Provides:
    - Hierarchical partition key management
    - Cross-partition query capabilities
    - TTL configuration and management
    - Performance monitoring
    - Branch isolation and conflict detection
    """

    def __init__(
        self,
        cosmos_client: CosmosClient,
        database_name: str,
        container_name: str,
        ttl_config: Optional[TTLConfiguration] = None,
    ):
        """
        Initialize branch-aware repository.

        Args:
            cosmos_client: Azure Cosmos DB client
            database_name: Database name
            container_name: Container name
            ttl_config: TTL configuration (optional)
        """
        self.cosmos_client = cosmos_client
        self.database_name = database_name
        self.container_name = container_name
        self.ttl_config = ttl_config or TTLConfiguration()

        # Get database and container references
        self.database: DatabaseProxy = self.cosmos_client.get_database_client(
            database_name
        )
        self.container: ContainerProxy = self.database.get_container_client(
            container_name
        )

        # Performance tracking
        self._metrics = PartitionMetrics()
        self._query_times: List[float] = []

        logger.info(
            f"BranchAwareRepository initialized for {database_name}.{container_name} "
            f"with TTL config: active={self.ttl_config.active_branch_ttl}s, "
            f"stale={self.ttl_config.stale_branch_ttl}s"
        )

    @abstractmethod
    def get_entity_type(self) -> str:
        """Get the entity type for this repository."""
        pass

    def generate_partition_key(
        self, repository_url: str, branch_name: str, use_entity_type: bool = True
    ) -> BranchPartitionKey:
        """
        Generate branch-aware partition key.

        Args:
            repository_url: Repository URL
            branch_name: Branch name
            use_entity_type: Whether to include entity type in partition key

        Returns:
            BranchPartitionKey instance
        """
        entity_type = self.get_entity_type() if use_entity_type else None
        return BranchPartitionKey(
            repository_url=repository_url,
            branch_name=branch_name,
            entity_type=entity_type,
        )

    async def upsert_item(
        self,
        item: Dict[str, Any],
        repository_url: str,
        branch_name: str,
        ttl_override: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Upsert item with branch-aware partitioning and TTL.

        Args:
            item: Item to upsert
            repository_url: Repository URL for partitioning
            branch_name: Branch name for partitioning
            ttl_override: Override TTL for this item

        Returns:
            Upserted item from Cosmos DB
        """
        start_time = datetime.now()

        try:
            # Generate partition key
            partition_key = self.generate_partition_key(repository_url, branch_name)

            # Add partition key and branch metadata to item
            item["partition_key"] = partition_key.to_composite_key()
            item["repository_url"] = repository_url
            item["branch_name"] = branch_name
            item["entity_type"] = self.get_entity_type()

            # Add TTL if configured
            if ttl_override is not None:
                item["ttl"] = ttl_override
            elif self.ttl_config.active_branch_ttl > 0:
                item["ttl"] = self.ttl_config.active_branch_ttl

            # Add timestamps for tracking
            now = datetime.now(timezone.utc).isoformat()
            item["updated_at"] = now
            if "created_at" not in item:
                item["created_at"] = now

            # Upsert to Cosmos DB
            result = await self.container.upsert_item(item)

            # Track performance
            self._track_query_performance(start_time)

            logger.debug(
                f"Upserted item {item.get('id', 'unknown')} to partition "
                f"{partition_key.to_composite_key()}"
            )

            return result

        except Exception as e:
            logger.error(f"Error upserting item to {self.container_name}: {e}")
            raise

    async def get_item(
        self, item_id: str, repository_url: str, branch_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get item by ID with branch-aware partitioning.

        Args:
            item_id: Item ID
            repository_url: Repository URL for partitioning
            branch_name: Branch name for partitioning

        Returns:
            Item if found, None otherwise
        """
        start_time = datetime.now()

        try:
            # Generate partition key
            partition_key = self.generate_partition_key(repository_url, branch_name)

            # Get item from specific partition
            item = await self.container.read_item(
                item=item_id, partition_key=partition_key.to_composite_key()
            )

            # Track performance
            self._track_query_performance(start_time)

            return item

        except CosmosResourceNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error getting item {item_id}: {e}")
            raise

    async def delete_item(
        self, item_id: str, repository_url: str, branch_name: str
    ) -> bool:
        """
        Delete item with branch-aware partitioning.

        Args:
            item_id: Item ID
            repository_url: Repository URL for partitioning
            branch_name: Branch name for partitioning

        Returns:
            True if deleted, False if not found
        """
        start_time = datetime.now()

        try:
            # Generate partition key
            partition_key = self.generate_partition_key(repository_url, branch_name)

            # Delete item from specific partition
            await self.container.delete_item(
                item=item_id, partition_key=partition_key.to_composite_key()
            )

            # Track performance
            self._track_query_performance(start_time)

            logger.debug(
                f"Deleted item {item_id} from partition "
                f"{partition_key.to_composite_key()}"
            )

            return True

        except CosmosResourceNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error deleting item {item_id}: {e}")
            raise

    async def query_branch_items(
        self,
        repository_url: str,
        branch_name: str,
        additional_filter: Optional[str] = None,
        parameters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query items within a specific branch (single partition).

        Args:
            repository_url: Repository URL for partitioning
            branch_name: Branch name for partitioning
            additional_filter: Additional WHERE clause
            parameters: Query parameters

        Returns:
            List of items in the branch
        """
        start_time = datetime.now()

        try:
            # Generate partition key
            partition_key = self.generate_partition_key(repository_url, branch_name)

            # Build query
            base_query = """
                SELECT * FROM c 
                WHERE c.partition_key = @partition_key
                AND c.entity_type = @entity_type
            """

            if additional_filter:
                base_query += f" AND ({additional_filter})"

            # Query parameters
            query_params = [
                {"name": "@partition_key", "value": partition_key.to_composite_key()},
                {"name": "@entity_type", "value": self.get_entity_type()},
            ]

            if parameters:
                query_params.extend(parameters)

            # Execute query (single partition, efficient)
            items = []
            async for item in self.container.query_items(
                query=base_query,
                parameters=query_params,
                enable_cross_partition_query=False,  # Single partition query
            ):
                items.append(item)

            # Track performance
            self._track_query_performance(start_time)

            logger.debug(
                f"Queried {len(items)} items from branch {repository_url}#{branch_name}"
            )

            return items

        except Exception as e:
            logger.error(f"Error querying branch items: {e}")
            raise

    async def query_cross_branch_items(
        self,
        repository_url: str,
        branch_names: Optional[List[str]] = None,
        additional_filter: Optional[str] = None,
        parameters: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query items across multiple branches (cross-partition).

        Used for merge conflict detection and repository-wide operations.

        Args:
            repository_url: Repository URL for filtering
            branch_names: Specific branch names (optional, all branches if None)
            additional_filter: Additional WHERE clause
            parameters: Query parameters

        Returns:
            List of items across branches
        """
        start_time = datetime.now()

        try:
            # Build cross-partition query
            base_query = """
                SELECT * FROM c
                WHERE c.repository_url = @repository_url
                AND c.entity_type = @entity_type
            """

            # Add branch filter if specified
            if branch_names:
                branch_placeholders = ", ".join(
                    [f"@branch_{i}" for i in range(len(branch_names))]
                )
                base_query += f" AND c.branch_name IN ({branch_placeholders})"

            if additional_filter:
                base_query += f" AND ({additional_filter})"

            # Query parameters
            query_params = [
                {"name": "@repository_url", "value": repository_url},
                {"name": "@entity_type", "value": self.get_entity_type()},
            ]

            # Add branch parameters
            if branch_names:
                for i, branch_name in enumerate(branch_names):
                    query_params.append({"name": f"@branch_{i}", "value": branch_name})

            if parameters:
                query_params.extend(parameters)

            # Execute cross-partition query
            items = []
            async for item in self.container.query_items(
                query=base_query,
                parameters=query_params,
                enable_cross_partition_query=True,  # Cross-partition query
            ):
                items.append(item)

            # Track cross-partition query
            self._metrics.cross_partition_queries += 1
            self._track_query_performance(start_time)

            logger.debug(
                f"Cross-partition query returned {len(items)} items from repository "
                f"{repository_url} across {len(branch_names) if branch_names else 'all'} branches"
            )

            return items

        except Exception as e:
            logger.error(f"Error in cross-branch query: {e}")
            raise

    async def mark_branch_for_cleanup(
        self, repository_url: str, branch_name: str, cleanup_type: str = "stale"
    ) -> int:
        """
        Mark all items in a branch for TTL cleanup.

        Args:
            repository_url: Repository URL
            branch_name: Branch name
            cleanup_type: Type of cleanup (stale, deleted, active)

        Returns:
            Number of items marked for cleanup
        """
        try:
            # Determine TTL based on cleanup type
            ttl_map = {
                "active": self.ttl_config.active_branch_ttl,
                "stale": self.ttl_config.stale_branch_ttl,
                "deleted": self.ttl_config.deleted_branch_ttl,
            }

            ttl_seconds = ttl_map.get(cleanup_type, self.ttl_config.stale_branch_ttl)

            # Get all items in the branch
            items = await self.query_branch_items(repository_url, branch_name)

            # Update TTL for each item
            update_count = 0
            for item in items:
                item["ttl"] = ttl_seconds
                item["cleanup_type"] = cleanup_type
                item["cleanup_timestamp"] = datetime.now(timezone.utc).isoformat()

                await self.container.upsert_item(item)
                update_count += 1

            logger.info(
                f"Marked {update_count} items in branch {repository_url}#{branch_name} "
                f"for {cleanup_type} cleanup (TTL: {ttl_seconds}s)"
            )

            return update_count

        except Exception as e:
            logger.error(f"Error marking branch for cleanup: {e}")
            raise

    async def get_partition_metrics(self) -> PartitionMetrics:
        """
        Get current partition distribution and performance metrics.

        Returns:
            PartitionMetrics with current statistics
        """
        try:
            # Query partition distribution
            partition_query = """
                SELECT c.partition_key, COUNT(1) as doc_count
                FROM c
                WHERE c.entity_type = @entity_type
                GROUP BY c.partition_key
            """

            partition_stats = {}
            async for result in self.container.query_items(
                query=partition_query,
                parameters=[{"name": "@entity_type", "value": self.get_entity_type()}],
                enable_cross_partition_query=True,
            ):
                partition_stats[result["partition_key"]] = result["doc_count"]

            # Update metrics
            self._metrics.partition_count = len(partition_stats)
            self._metrics.total_documents = sum(partition_stats.values())

            # Identify hot/cold partitions (simple heuristic)
            if partition_stats:
                avg_docs = self._metrics.total_documents / self._metrics.partition_count

                self._metrics.hot_partitions = [
                    pk for pk, count in partition_stats.items() if count > avg_docs * 2
                ]

                self._metrics.cold_partitions = [
                    pk
                    for pk, count in partition_stats.items()
                    if count < avg_docs * 0.5
                ]

            # Update average query latency
            if self._query_times:
                self._metrics.average_query_latency_ms = sum(self._query_times) / len(
                    self._query_times
                )
                # Keep only recent measurements
                self._query_times = self._query_times[-100:]

            self._metrics.timestamp = datetime.now(timezone.utc)

            return self._metrics

        except Exception as e:
            logger.error(f"Error getting partition metrics: {e}")
            raise

    def _track_query_performance(self, start_time: datetime) -> None:
        """Track query performance for monitoring."""
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        self._query_times.append(duration_ms)

    async def configure_container_optimizations(self) -> None:
        """
        Configure container-level optimizations for branch-aware operations.

        This includes:
        - TTL configuration
        - Index policy optimization
        - Throughput recommendations
        """
        try:
            logger.info(
                f"Configuring optimizations for container {self.container_name}"
            )

            # Note: Container schema changes require recreation in most cases
            # This method documents the recommended configuration

            recommended_config = {
                "ttl": {
                    "defaultTtl": self.ttl_config.container_default_ttl,
                    "description": "TTL disabled by default, controlled per-item",
                },
                "indexingPolicy": {
                    "indexingMode": "consistent",
                    "automatic": True,
                    "includedPaths": [{"path": "/*"}],
                    "excludedPaths": [
                        {"path": "/embedding/*"},  # Exclude large vectors
                        {"path": "/_etag/?"},
                    ],
                    "compositeIndexes": [
                        # Branch-aware query optimization
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
                        # Timestamp-based queries
                        [
                            {"path": "/repository_url", "order": "ascending"},
                            {"path": "/updated_at", "order": "descending"},
                        ],
                    ],
                },
                "hierarchicalPartitionKeys": [
                    "/repository_url",
                    "/branch_name",
                    "/entity_type",
                ],
            }

            logger.info(
                f"Recommended configuration for {self.container_name}: "
                f"{recommended_config}"
            )

            # For existing containers, log current configuration
            container_props = await self.container.read()
            logger.info(f"Current container configuration: {container_props}")

        except Exception as e:
            logger.error(f"Error configuring container optimizations: {e}")
            raise
