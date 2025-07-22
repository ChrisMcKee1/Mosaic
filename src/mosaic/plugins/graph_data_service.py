"""
Graph Data Service for Mosaic
Handles data retrieval from Cosmos DB for graph visualization
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)


class GraphDataService:
    """
    Service for retrieving graph data from Cosmos DB for visualization.
    
    Connects to the unified OmniRAG Cosmos DB backend and transforms
    entity and relationship data for graph visualization libraries.
    """

    def __init__(self, settings):
        """Initialize with Mosaic settings."""
        self.settings = settings
        self.client = None
        self.database = None
        self.containers = {}
        
    async def initialize(self):
        """Initialize Cosmos DB connection."""
        try:
            # Use DefaultAzureCredential for authentication
            credential = DefaultAzureCredential()
            
            self.client = CosmosClient(
                url=self.settings.azure_cosmos_endpoint,
                credential=credential
            )
            
            # Connect to database
            self.database = self.client.get_database_client(
                self.settings.cosmos_database_name or "MosaicKnowledge"
            )
            
            # Get container references
            self.containers = {
                'entities': self.database.get_container_client("code_entities"),
                'relationships': self.database.get_container_client("code_relationships"),
                'repositories': self.database.get_container_client("repositories"),
                'libraries': self.database.get_container_client("libraries")
            }
            
            logger.info("GraphDataService initialized with Cosmos DB connection")
            
        except Exception as e:
            logger.error(f"Failed to initialize GraphDataService: {e}")
            raise

    async def get_repository_entities(
        self,
        repository_url: str,
        include_functions: bool = True,
        include_classes: bool = True,
        include_modules: bool = True,
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Retrieve code entities for a repository from Cosmos DB.
        
        Args:
            repository_url: Repository URL to filter by
            include_functions: Include function entities
            include_classes: Include class entities  
            include_modules: Include module entities
            limit: Maximum number of entities to retrieve
            
        Returns:
            DataFrame with entity data for visualization
        """
        try:
            # Build entity type filter
            entity_types = []
            if include_functions:
                entity_types.append("function")
            if include_classes:
                entity_types.append("class")
            if include_modules:
                entity_types.append("module")
            
            if not entity_types:
                return pd.DataFrame()
            
            # Query entities from Cosmos DB
            query = """
            SELECT 
                c.id,
                c.name as caption,
                c.entity_type,
                c.language,
                c.file_path,
                c.start_line,
                c.end_line,
                c.complexity_score,
                c.lines_of_code,
                c.repository_url
            FROM c 
            WHERE c.repository_url = @repository_url 
            AND c.entity_type IN ({})
            ORDER BY c.complexity_score DESC
            OFFSET 0 LIMIT @limit
            """.format(','.join([f"'{t}'" for t in entity_types]))
            
            parameters = [
                {"name": "@repository_url", "value": repository_url},
                {"name": "@limit", "value": limit}
            ]
            
            items = list(self.containers['entities'].query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            if not items:
                logger.warning(f"No entities found for repository: {repository_url}")
                return pd.DataFrame()
            
            # Convert to DataFrame and add visualization properties
            df = pd.DataFrame(items)
            
            # Add size property for visualization (based on complexity or LOC)
            if 'complexity_score' in df.columns and not df['complexity_score'].isna().all():
                df['size'] = df['complexity_score'].fillna(5) * 2 + 5
            elif 'lines_of_code' in df.columns and not df['lines_of_code'].isna().all():
                df['size'] = (df['lines_of_code'].fillna(10) / 10) + 5
            else:
                df['size'] = 10  # Default size
                
            # Ensure size is within reasonable bounds
            df['size'] = df['size'].clip(lower=5, upper=30)
            
            # Add complexity for sizing (alias)
            df['complexity'] = df.get('complexity_score', 5)
            
            logger.info(f"Retrieved {len(df)} entities for repository: {repository_url}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving repository entities: {e}")
            return pd.DataFrame()

    async def get_repository_relationships(
        self,
        repository_url: str,
        relationship_types: List[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Retrieve relationships between entities for a repository.
        
        Args:
            repository_url: Repository URL to filter by
            relationship_types: Types of relationships to include
            limit: Maximum number of relationships to retrieve
            
        Returns:
            DataFrame with relationship data for visualization
        """
        try:
            if relationship_types is None:
                relationship_types = ["imports", "calls", "inherits", "defines", "uses"]
            
            # Query relationships
            query = """
            SELECT 
                r.source_entity_id as source,
                r.target_entity_id as target,
                r.relationship_type as caption,
                r.relationship_type,
                r.strength,
                r.repository_url
            FROM r 
            WHERE r.repository_url = @repository_url
            AND r.relationship_type IN ({})
            ORDER BY r.strength DESC
            OFFSET 0 LIMIT @limit
            """.format(','.join([f"'{t}'" for t in relationship_types]))
            
            parameters = [
                {"name": "@repository_url", "value": repository_url},
                {"name": "@limit", "value": limit}
            ]
            
            items = list(self.containers['relationships'].query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            if not items:
                logger.warning(f"No relationships found for repository: {repository_url}")
                return pd.DataFrame()
            
            df = pd.DataFrame(items)
            
            # Ensure source and target are integers for neo4j-viz
            df['source'] = pd.to_numeric(df['source'], errors='coerce')
            df['target'] = pd.to_numeric(df['target'], errors='coerce')
            
            # Remove rows with invalid source/target
            df = df.dropna(subset=['source', 'target'])
            df['source'] = df['source'].astype(int)
            df['target'] = df['target'].astype(int)
            
            logger.info(f"Retrieved {len(df)} relationships for repository: {repository_url}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving repository relationships: {e}")
            return pd.DataFrame()

    async def get_dependency_data(
        self,
        repository_url: str,
        include_external: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get dependency-focused entities and relationships.
        
        Args:
            repository_url: Repository to analyze
            include_external: Include external library dependencies
            
        Returns:
            Tuple of (entities_df, relationships_df) for dependencies
        """
        try:
            # Query for modules and their dependencies
            entity_query = """
            SELECT 
                c.id,
                c.name as caption,
                c.entity_type,
                c.file_path,
                c.language,
                c.import_count as usage_count,
                c.is_external,
                c.repository_url
            FROM c 
            WHERE c.repository_url = @repository_url
            AND (c.entity_type = 'module' OR c.entity_type = 'import')
            """
            
            if not include_external:
                entity_query += " AND c.is_external = false"
            
            entity_params = [{"name": "@repository_url", "value": repository_url}]
            
            entities = list(self.containers['entities'].query_items(
                query=entity_query,
                parameters=entity_params,
                enable_cross_partition_query=True
            ))
            
            # Query for import/dependency relationships
            rel_query = """
            SELECT 
                r.source_entity_id as source,
                r.target_entity_id as target,
                r.relationship_type as caption,
                r.relationship_type as dependency_type
            FROM r 
            WHERE r.repository_url = @repository_url
            AND r.relationship_type IN ('imports', 'depends_on', 'requires')
            """
            
            rel_params = [{"name": "@repository_url", "value": repository_url}]
            
            relationships = list(self.containers['relationships'].query_items(
                query=rel_query,
                parameters=rel_params,
                enable_cross_partition_query=True
            ))
            
            # Convert to DataFrames
            entities_df = pd.DataFrame(entities)
            relationships_df = pd.DataFrame(relationships)
            
            # Add visualization properties
            if not entities_df.empty:
                entities_df['size'] = entities_df.get('usage_count', 1) * 2 + 5
                entities_df['size'] = entities_df['size'].clip(lower=5, upper=25)
                
                # Ensure numeric IDs
                entities_df['id'] = pd.to_numeric(entities_df['id'], errors='coerce')
                entities_df = entities_df.dropna(subset=['id'])
                entities_df['id'] = entities_df['id'].astype(int)
            
            if not relationships_df.empty:
                relationships_df['source'] = pd.to_numeric(relationships_df['source'], errors='coerce')
                relationships_df['target'] = pd.to_numeric(relationships_df['target'], errors='coerce')
                relationships_df = relationships_df.dropna(subset=['source', 'target'])
                relationships_df['source'] = relationships_df['source'].astype(int)
                relationships_df['target'] = relationships_df['target'].astype(int)
            
            logger.info(f"Retrieved dependency data: {len(entities_df)} entities, {len(relationships_df)} relationships")
            return entities_df, relationships_df
            
        except Exception as e:
            logger.error(f"Error retrieving dependency data: {e}")
            return pd.DataFrame(), pd.DataFrame()

    async def get_semantic_knowledge_graph(
        self,
        repository_url: str,
        max_nodes: int = 200,
        similarity_threshold: float = 0.7
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get knowledge graph with semantic relationships.
        
        Args:
            repository_url: Repository to analyze
            max_nodes: Maximum nodes to include
            similarity_threshold: Minimum similarity for semantic edges
            
        Returns:
            Tuple of (entities_df, relationships_df) with semantic data
        """
        try:
            # Query for entities with embeddings and semantic data
            entity_query = """
            SELECT TOP @max_nodes
                c.id,
                c.name as caption,
                c.entity_type,
                c.language,
                c.semantic_cluster as functional_cluster,
                c.importance_score,
                c.centrality_score,
                c.embedding
            FROM c 
            WHERE c.repository_url = @repository_url
            AND IS_DEFINED(c.embedding)
            ORDER BY c.importance_score DESC
            """
            
            entity_params = [
                {"name": "@repository_url", "value": repository_url},
                {"name": "@max_nodes", "value": max_nodes}
            ]
            
            entities = list(self.containers['entities'].query_items(
                query=entity_query,
                parameters=entity_params,
                enable_cross_partition_query=True
            ))
            
            # Query semantic similarity relationships
            rel_query = """
            SELECT 
                r.source_entity_id as source,
                r.target_entity_id as target,
                r.relationship_type as caption,
                r.similarity_score
            FROM r 
            WHERE r.repository_url = @repository_url
            AND r.relationship_type = 'semantic_similarity'
            AND r.similarity_score >= @threshold
            """
            
            rel_params = [
                {"name": "@repository_url", "value": repository_url},
                {"name": "@threshold", "value": similarity_threshold}
            ]
            
            relationships = list(self.containers['relationships'].query_items(
                query=rel_query,
                parameters=rel_params,
                enable_cross_partition_query=True
            ))
            
            # Convert to DataFrames
            entities_df = pd.DataFrame(entities)
            relationships_df = pd.DataFrame(relationships)
            
            # Add visualization properties
            if not entities_df.empty:
                # Size by importance score
                if 'importance_score' in entities_df.columns:
                    entities_df['size'] = (entities_df['importance_score'] * 20 + 8).clip(lower=8, upper=28)
                else:
                    entities_df['size'] = 12
                    
                # Ensure proper data types
                entities_df['id'] = pd.to_numeric(entities_df['id'], errors='coerce')
                entities_df = entities_df.dropna(subset=['id'])
                entities_df['id'] = entities_df['id'].astype(int)
            
            if not relationships_df.empty:
                relationships_df['source'] = pd.to_numeric(relationships_df['source'], errors='coerce')
                relationships_df['target'] = pd.to_numeric(relationships_df['target'], errors='coerce')
                relationships_df = relationships_df.dropna(subset=['source', 'target'])
                relationships_df['source'] = relationships_df['source'].astype(int)
                relationships_df['target'] = relationships_df['target'].astype(int)
            
            logger.info(f"Retrieved semantic knowledge graph: {len(entities_df)} entities, {len(relationships_df)} relationships")
            return entities_df, relationships_df
            
        except Exception as e:
            logger.error(f"Error retrieving semantic knowledge graph: {e}")
            return pd.DataFrame(), pd.DataFrame()

    async def get_repository_stats(self, repository_url: str) -> Dict[str, Any]:
        """Get statistics about a repository's entities and relationships."""
        try:
            # Entity count by type
            entity_stats_query = """
            SELECT 
                c.entity_type,
                COUNT(1) as count
            FROM c 
            WHERE c.repository_url = @repository_url
            GROUP BY c.entity_type
            """
            
            # Relationship count by type  
            rel_stats_query = """
            SELECT 
                r.relationship_type,
                COUNT(1) as count
            FROM r
            WHERE r.repository_url = @repository_url
            GROUP BY r.relationship_type
            """
            
            entity_params = [{"name": "@repository_url", "value": repository_url}]
            
            entity_stats = list(self.containers['entities'].query_items(
                query=entity_stats_query,
                parameters=entity_params,
                enable_cross_partition_query=True
            ))
            
            rel_stats = list(self.containers['relationships'].query_items(
                query=rel_stats_query,
                parameters=entity_params,
                enable_cross_partition_query=True
            ))
            
            return {
                "repository_url": repository_url,
                "entity_stats": {item['entity_type']: item['count'] for item in entity_stats},
                "relationship_stats": {item['relationship_type']: item['count'] for item in rel_stats},
                "total_entities": sum(item['count'] for item in entity_stats),
                "total_relationships": sum(item['count'] for item in rel_stats)
            }
            
        except Exception as e:
            logger.error(f"Error getting repository stats: {e}")
            return {"error": str(e)}