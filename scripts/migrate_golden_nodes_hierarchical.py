#!/usr/bin/env python3
"""
Golden Node Hierarchical Migration Script

Migrates existing GoldenNode documents to support new hierarchical relationships:
- Converts string-based parent_entity references to UUID-based parent_id
- Calculates hierarchy_level and hierarchy_path for all nodes
- Populates materialized paths for efficient traversal
- Updates partition keys to align with new schema

Based on research findings and implementation decisions logged in Conport:
- Uses Azure Cosmos DB native capabilities for optimal performance
- Maintains backward compatibility during transition
- Implements proper UUID-based parent-child relationships
- Follows Microsoft's best practices for document migration

Usage:
    python scripts/migrate_golden_nodes_hierarchical.py --dry-run
    python scripts/migrate_golden_nodes_hierarchical.py --execute
"""

import argparse
import asyncio
import logging
from typing import Dict, Any, List, Set
from collections import defaultdict, deque

from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError
from azure.identity import DefaultAzureCredential

# Assuming we can import from the project structure
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


class HierarchicalMigration:
    """
    Manages the migration of GoldenNode documents to hierarchical relationships.

    Migration Strategy:
    1. Read all existing GoldenNode documents
    2. Build parent-child mapping from parent_entity strings
    3. Create UUID-based relationships and materialized paths
    4. Calculate hierarchy levels using breadth-first traversal
    5. Update documents with new hierarchical fields
    6. Validate hierarchical consistency
    """

    def __init__(
        self, cosmos_endpoint: str, database_name: str = "mosaic", dry_run: bool = True
    ):
        self.cosmos_endpoint = cosmos_endpoint
        self.database_name = database_name
        self.dry_run = dry_run

        # Initialize Cosmos client
        credential = DefaultAzureCredential()
        self.cosmos_client = CosmosClient(cosmos_endpoint, credential)
        self.database = self.cosmos_client.get_database_client(database_name)
        self.container = self.database.get_container_client("golden_nodes")

        # Migration state
        self.nodes_by_id: Dict[str, Dict[str, Any]] = {}
        self.nodes_by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.parent_child_map: Dict[str, List[str]] = defaultdict(list)
        self.name_to_id_map: Dict[str, str] = {}

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def migrate(self) -> None:
        """Execute the complete hierarchical migration process."""
        try:
            self.logger.info("Starting Golden Node hierarchical migration")

            # Step 1: Load existing documents
            await self._load_existing_documents()

            # Step 2: Build parent-child relationships
            await self._build_parent_child_relationships()

            # Step 3: Calculate hierarchical fields
            await self._calculate_hierarchical_fields()

            # Step 4: Update documents (or simulate if dry run)
            await self._update_documents()

            # Step 5: Validate migration results
            await self._validate_migration()

            self.logger.info("Migration completed successfully")

        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise

    async def _load_existing_documents(self) -> None:
        """Load all existing GoldenNode documents from Cosmos DB."""
        self.logger.info("Loading existing GoldenNode documents...")

        try:
            query = "SELECT * FROM c WHERE c.document_type = 'golden_node'"

            items = list(
                self.container.query_items(
                    query=query, enable_cross_partition_query=True
                )
            )

            self.logger.info(f"Loaded {len(items)} existing documents")

            # Index documents by ID and name for relationship building
            for item in items:
                doc_id = item.get("id")
                entity_name = item.get("code_entity", {}).get("name", "")

                if doc_id:
                    self.nodes_by_id[doc_id] = item

                if entity_name:
                    self.nodes_by_name[entity_name].append(item)
                    # Simple name-to-id mapping (first occurrence wins)
                    if entity_name not in self.name_to_id_map:
                        self.name_to_id_map[entity_name] = doc_id

            self.logger.info(f"Indexed {len(self.nodes_by_id)} documents by ID")
            self.logger.info(f"Indexed {len(self.nodes_by_name)} unique entity names")

        except Exception as e:
            self.logger.error(f"Failed to load existing documents: {e}")
            raise

    async def _build_parent_child_relationships(self) -> None:
        """Build parent-child relationships from string-based parent_entity references."""
        self.logger.info("Building parent-child relationships...")

        orphaned_nodes = []
        relationship_count = 0

        for doc_id, document in self.nodes_by_id.items():
            parent_entity = document.get("code_entity", {}).get("parent_entity")

            if parent_entity:
                # Find parent by name
                parent_id = self.name_to_id_map.get(parent_entity)

                if parent_id and parent_id in self.nodes_by_id:
                    # Valid parent found
                    self.parent_child_map[parent_id].append(doc_id)
                    relationship_count += 1
                else:
                    # Parent not found - this node will be orphaned
                    orphaned_nodes.append(
                        {
                            "id": doc_id,
                            "name": document.get("code_entity", {}).get(
                                "name", "unknown"
                            ),
                            "missing_parent": parent_entity,
                        }
                    )

        self.logger.info(f"Built {relationship_count} parent-child relationships")

        if orphaned_nodes:
            self.logger.warning(
                f"Found {len(orphaned_nodes)} orphaned nodes with missing parents:"
            )
            for node in orphaned_nodes[:10]:  # Show first 10
                self.logger.warning(
                    f"  {node['name']} (ID: {node['id']}) -> missing parent: {node['missing_parent']}"
                )
            if len(orphaned_nodes) > 10:
                self.logger.warning(f"  ... and {len(orphaned_nodes) - 10} more")

    async def _calculate_hierarchical_fields(self) -> None:
        """Calculate hierarchy_level and hierarchy_path for all nodes using BFS."""
        self.logger.info("Calculating hierarchical fields...")

        # Find root nodes (nodes with no parent or orphaned nodes)
        root_nodes = []
        for doc_id, document in self.nodes_by_id.items():
            parent_entity = document.get("code_entity", {}).get("parent_entity")
            if (
                not parent_entity
                or self.name_to_id_map.get(parent_entity) not in self.nodes_by_id
            ):
                root_nodes.append(doc_id)

        self.logger.info(f"Found {len(root_nodes)} root nodes")

        # BFS traversal to calculate levels and paths
        visited: Set[str] = set()

        for root_id in root_nodes:
            if root_id in visited:
                continue

            # BFS from this root
            queue = deque([(root_id, 0, [])])  # (node_id, level, path_to_node)

            while queue:
                current_id, level, path = queue.popleft()

                if current_id in visited:
                    continue

                visited.add(current_id)

                # Update hierarchical fields in document
                if current_id in self.nodes_by_id:
                    document = self.nodes_by_id[current_id]
                    code_entity = document.setdefault("code_entity", {})

                    # Set parent_id (UUID-based relationship)
                    if path:
                        code_entity["parent_id"] = path[
                            -1
                        ]  # Last element in path is immediate parent
                    else:
                        code_entity["parent_id"] = None

                    # Set hierarchy level
                    code_entity["hierarchy_level"] = level

                    # Set hierarchy path (materialized path from root)
                    code_entity["hierarchy_path"] = list(path)

                    # Add children to queue
                    children = self.parent_child_map.get(current_id, [])
                    for child_id in children:
                        if child_id not in visited:
                            queue.append((child_id, level + 1, path + [current_id]))

        # Validate all nodes were processed
        unprocessed = set(self.nodes_by_id.keys()) - visited
        if unprocessed:
            self.logger.warning(
                f"Found {len(unprocessed)} unprocessed nodes (possible circular references):"
            )
            for node_id in list(unprocessed)[:5]:  # Show first 5
                node_name = (
                    self.nodes_by_id[node_id]
                    .get("code_entity", {})
                    .get("name", "unknown")
                )
                self.logger.warning(f"  {node_name} (ID: {node_id})")

        self.logger.info(f"Calculated hierarchical fields for {len(visited)} nodes")

    async def _update_documents(self) -> None:
        """Update documents with new hierarchical fields."""
        if self.dry_run:
            self.logger.info("DRY RUN: Simulating document updates...")
            await self._simulate_updates()
            return

        self.logger.info("Updating documents with hierarchical fields...")

        update_count = 0
        error_count = 0

        for doc_id, document in self.nodes_by_id.items():
            try:
                # Update partition_key to match new schema
                if "partition_key" not in document:
                    repository_url = document.get("git_context", {}).get(
                        "repository_url", "unknown"
                    )
                    document["partition_key"] = repository_url

                # Replace document in Cosmos DB
                self.container.replace_item(item=doc_id, body=document)
                update_count += 1

                if update_count % 100 == 0:
                    self.logger.info(f"Updated {update_count} documents...")

            except CosmosHttpResponseError as e:
                self.logger.error(f"Failed to update document {doc_id}: {e}")
                error_count += 1
            except Exception as e:
                self.logger.error(f"Unexpected error updating document {doc_id}: {e}")
                error_count += 1

        self.logger.info(f"Updated {update_count} documents successfully")
        if error_count > 0:
            self.logger.warning(f"Failed to update {error_count} documents")

    async def _simulate_updates(self) -> None:
        """Simulate document updates for dry run mode."""
        self.logger.info("Simulating updates (dry run mode):")

        # Show sample of changes
        sample_nodes = list(self.nodes_by_id.items())[:5]

        for doc_id, document in sample_nodes:
            code_entity = document.get("code_entity", {})
            name = code_entity.get("name", "unknown")
            parent_id = code_entity.get("parent_id")
            hierarchy_level = code_entity.get("hierarchy_level", 0)
            hierarchy_path = code_entity.get("hierarchy_path", [])

            self.logger.info(f"  {name} (ID: {doc_id}):")
            self.logger.info(f"    parent_id: {parent_id}")
            self.logger.info(f"    hierarchy_level: {hierarchy_level}")
            self.logger.info(f"    hierarchy_path: {hierarchy_path}")

        if len(self.nodes_by_id) > 5:
            self.logger.info(
                f"  ... and {len(self.nodes_by_id) - 5} more documents would be updated"
            )

    async def _validate_migration(self) -> None:
        """Validate the migration results."""
        self.logger.info("Validating migration results...")

        # Validation statistics
        root_count = 0
        max_depth = 0
        level_distribution = defaultdict(int)

        for document in self.nodes_by_id.values():
            code_entity = document.get("code_entity", {})
            hierarchy_level = code_entity.get("hierarchy_level", 0)
            parent_id = code_entity.get("parent_id")
            hierarchy_path = code_entity.get("hierarchy_path", [])

            # Count roots
            if parent_id is None and hierarchy_level == 0:
                root_count += 1

            # Track max depth
            max_depth = max(max_depth, hierarchy_level)

            # Level distribution
            level_distribution[hierarchy_level] += 1

            # Validate path consistency
            if len(hierarchy_path) != hierarchy_level:
                self.logger.warning(
                    f"Path inconsistency: {code_entity.get('name')} has level {hierarchy_level} "
                    f"but path length {len(hierarchy_path)}"
                )

        self.logger.info("Migration validation results:")
        self.logger.info(f"  Root nodes: {root_count}")
        self.logger.info(f"  Maximum hierarchy depth: {max_depth}")
        self.logger.info("  Level distribution:")
        for level in sorted(level_distribution.keys()):
            self.logger.info(f"    Level {level}: {level_distribution[level]} nodes")

        # Validate UUID formats in relationships
        uuid_validation_errors = 0
        for document in self.nodes_by_id.values():
            code_entity = document.get("code_entity", {})
            parent_id = code_entity.get("parent_id")
            hierarchy_path = code_entity.get("hierarchy_path", [])

            # Validate parent_id UUID format
            if parent_id:
                try:
                    from uuid import UUID

                    UUID(parent_id)
                except ValueError:
                    uuid_validation_errors += 1

            # Validate UUIDs in hierarchy_path
            for uuid_str in hierarchy_path:
                try:
                    from uuid import UUID

                    UUID(uuid_str)
                except ValueError:
                    uuid_validation_errors += 1

        if uuid_validation_errors > 0:
            self.logger.warning(
                f"Found {uuid_validation_errors} UUID format validation errors"
            )
        else:
            self.logger.info("All UUID formats are valid")


async def main():
    """Main migration execution function."""
    parser = argparse.ArgumentParser(
        description="Migrate GoldenNode documents to hierarchical relationships"
    )
    parser.add_argument(
        "--cosmos-endpoint", required=True, help="Azure Cosmos DB endpoint URL"
    )
    parser.add_argument(
        "--database-name",
        default="mosaic",
        help="Cosmos DB database name (default: mosaic)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate migration without making changes",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute actual migration (updates documents)",
    )

    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("ERROR: Must specify either --dry-run or --execute")
        parser.print_help()
        return

    if args.dry_run and args.execute:
        print("ERROR: Cannot specify both --dry-run and --execute")
        parser.print_help()
        return

    # Execute migration
    migration = HierarchicalMigration(
        cosmos_endpoint=args.cosmos_endpoint,
        database_name=args.database_name,
        dry_run=args.dry_run,
    )

    await migration.migrate()

    if args.dry_run:
        print("\nDry run completed. Use --execute to perform actual migration.")
    else:
        print("\nMigration completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
