"""
Mosaic Graph Visualization Plugin
Interactive graph visualization of code relationships using Neo4j-viz
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import networkx as nx

from semantic_kernel.functions import kernel_function
from neo4j_viz import Node, Relationship, VisualizationGraph
from neo4j_viz.pandas import from_dfs
from .graph_data_service import GraphDataService

logger = logging.getLogger(__name__)


class GraphVisualizationPlugin:
    """
    Semantic Kernel plugin for interactive graph visualization of code relationships.
    
    Uses Neo4j-viz to create beautiful, interactive visualizations of:
    - Repository structures and dependencies
    - Code entities (functions, classes, modules)
    - Cross-file relationships and imports
    - Knowledge graph exploration
    """

    def __init__(self, settings=None):
        """Initialize the Graph Visualization Plugin."""
        self.settings = settings
        self.data_service = GraphDataService(settings)
        logger.info("GraphVisualizationPlugin initialized")
        
    async def initialize(self):
        """Initialize the plugin and data service."""
        await self.data_service.initialize()
        logger.info("GraphVisualizationPlugin initialized with data service")

    @kernel_function(
        name="visualize_repository_structure",
        description="Create interactive visualization of repository code structure and relationships"
    )
    async def visualize_repository_structure(
        self, 
        repository_url: str,
        include_functions: bool = True,
        include_classes: bool = True,
        color_by_language: bool = True,
        size_by_complexity: bool = True
    ) -> str:
        """
        Generate an interactive graph visualization of repository structure.
        
        Args:
            repository_url: Git repository URL to visualize
            include_functions: Include function nodes in visualization
            include_classes: Include class nodes in visualization  
            color_by_language: Color nodes by programming language
            size_by_complexity: Size nodes by code complexity/LOC
            
        Returns:
            HTML string containing interactive graph visualization
        """
        try:
            logger.info(f"Creating repository visualization for: {repository_url}")
            
            # Get entities and relationships from Cosmos DB via data service
            entities_df = await self.data_service.get_repository_entities(
                repository_url, 
                include_functions=include_functions, 
                include_classes=include_classes,
                include_modules=True
            )
            relationships_df = await self.data_service.get_repository_relationships(repository_url)
            
            if entities_df.empty:
                return self._create_empty_graph_message(repository_url)
            
            # Create visualization
            VG = from_dfs(entities_df, relationships_df)
            
            # Apply styling
            if color_by_language and 'language' in entities_df.columns:
                VG.color_nodes(property="language")
                
            if size_by_complexity and 'complexity' in entities_df.columns:
                VG.resize_nodes(property="complexity", node_radius_min_max=(8, 30))
            
            # Generate HTML
            html_output = VG.render()
            
            logger.info(f"Successfully created visualization with {len(entities_df)} nodes and {len(relationships_df)} relationships")
            return html_output
            
        except Exception as e:
            logger.error(f"Error creating repository visualization: {e}")
            return f"<div>Error creating visualization: {str(e)}</div>"

    @kernel_function(
        name="visualize_code_dependencies",
        description="Create interactive visualization focusing on code dependencies and imports"
    )
    async def visualize_code_dependencies(
        self,
        repository_url: str,
        dependency_types: List[str] = None,
        show_external_deps: bool = True,
        layout_algorithm: str = "force"
    ) -> str:
        """
        Create a dependency-focused graph visualization.
        
        Args:
            repository_url: Repository to analyze
            dependency_types: Types of dependencies to include (imports, calls, inherits)
            show_external_deps: Include external library dependencies
            layout_algorithm: Graph layout algorithm to use
            
        Returns:
            HTML string with dependency graph visualization
        """
        try:
            logger.info(f"Creating dependency visualization for: {repository_url}")
            
            if dependency_types is None:
                dependency_types = ["imports", "calls", "inherits"]
            
            # Get dependency data from data service
            entities_df, relationships_df = await self.data_service.get_dependency_data(
                repository_url, include_external=show_external_deps
            )
            
            if entities_df.empty:
                return self._create_empty_graph_message(repository_url, "dependencies")
            
            # Create focused dependency visualization
            VG = from_dfs(entities_df, relationships_df)
            
            # Style by dependency type
            if 'entity_type' in entities_df.columns:
                # Color external dependencies differently
                VG.color_nodes(property="entity_type")
            
            # Size by usage frequency
            if 'usage_count' in entities_df.columns:
                VG.resize_nodes(property="usage_count", node_radius_min_max=(5, 25))
            
            html_output = VG.render()
            logger.info(f"Created dependency visualization with {len(entities_df)} nodes")
            return html_output
            
        except Exception as e:
            logger.error(f"Error creating dependency visualization: {e}")
            return f"<div>Error creating dependency visualization: {str(e)}</div>"

    @kernel_function(
        name="visualize_knowledge_graph",
        description="Create comprehensive knowledge graph visualization with semantic relationships"
    )
    async def visualize_knowledge_graph(
        self,
        repository_url: str,
        include_semantic_similarity: bool = True,
        cluster_by_functionality: bool = True,
        max_nodes: int = 200
    ) -> str:
        """
        Create a comprehensive knowledge graph with semantic relationships.
        
        Args:
            repository_url: Repository to visualize
            include_semantic_similarity: Include semantic similarity edges
            cluster_by_functionality: Group nodes by functional similarity
            max_nodes: Maximum number of nodes to include
            
        Returns:
            HTML string with knowledge graph visualization
        """
        try:
            logger.info(f"Creating knowledge graph for: {repository_url}")
            
            # Get comprehensive knowledge graph data
            entities_df, relationships_df = await self.data_service.get_semantic_knowledge_graph(
                repository_url, max_nodes=max_nodes, similarity_threshold=0.7 if include_semantic_similarity else 1.0
            )
            
            if entities_df.empty:
                return self._create_empty_graph_message(repository_url, "knowledge graph")
            
            # Create rich visualization
            VG = from_dfs(entities_df, relationships_df)
            
            # Apply advanced styling
            if cluster_by_functionality and 'functional_cluster' in entities_df.columns:
                VG.color_nodes(property="functional_cluster")
            
            # Size by importance score
            if 'importance_score' in entities_df.columns:
                VG.resize_nodes(property="importance_score", node_radius_min_max=(6, 28))
            
            html_output = VG.render()
            logger.info(f"Created knowledge graph with {len(entities_df)} entities")
            return html_output
            
        except Exception as e:
            logger.error(f"Error creating knowledge graph: {e}")
            return f"<div>Error creating knowledge graph: {str(e)}</div>"

    @kernel_function(
        name="create_custom_graph",
        description="Create custom graph visualization from provided node and relationship data"
    )
    async def create_custom_graph(
        self,
        nodes_data: str,  # JSON string
        relationships_data: str,  # JSON string
        title: str = "Custom Graph Visualization",
        styling_options: str = "{}"  # JSON string
    ) -> str:
        """
        Create a custom graph from user-provided data.
        
        Args:
            nodes_data: JSON string with node data
            relationships_data: JSON string with relationship data
            title: Title for the visualization
            styling_options: JSON string with styling preferences
            
        Returns:
            HTML string with custom graph visualization
        """
        try:
            import json
            
            logger.info(f"Creating custom graph: {title}")
            
            # Parse data
            nodes = json.loads(nodes_data)
            relationships = json.loads(relationships_data)
            styling = json.loads(styling_options)
            
            # Convert to DataFrames
            entities_df = pd.DataFrame(nodes)
            relationships_df = pd.DataFrame(relationships)
            
            # Create visualization
            VG = from_dfs(entities_df, relationships_df)
            
            # Apply custom styling
            if styling.get("color_property") and styling["color_property"] in entities_df.columns:
                VG.color_nodes(property=styling["color_property"])
                
            if styling.get("size_property") and styling["size_property"] in entities_df.columns:
                min_size = styling.get("min_node_size", 5)
                max_size = styling.get("max_node_size", 25)
                VG.resize_nodes(property=styling["size_property"], node_radius_min_max=(min_size, max_size))
            
            html_output = VG.render()
            logger.info(f"Created custom graph with {len(entities_df)} nodes")
            return html_output
            
        except Exception as e:
            logger.error(f"Error creating custom graph: {e}")
            return f"<div>Error creating custom graph: {str(e)}</div>"

    @kernel_function(
        name="get_repository_graph_stats",
        description="Get statistics about repository entities and relationships for graph visualization"
    )
    async def get_repository_graph_stats(self, repository_url: str) -> str:
        """
        Get comprehensive statistics about a repository's graph structure.
        
        Args:
            repository_url: Repository URL to analyze
            
        Returns:
            JSON string with repository statistics
        """
        try:
            import json
            
            logger.info(f"Getting repository stats for: {repository_url}")
            
            stats = await self.data_service.get_repository_stats(repository_url)
            
            if "error" in stats:
                return json.dumps({"error": stats["error"]})
            
            # Add recommendations based on stats
            recommendations = []
            
            total_entities = stats.get("total_entities", 0)
            total_relationships = stats.get("total_relationships", 0)
            
            if total_entities == 0:
                recommendations.append("Repository not ingested yet - run ingestion service first")
            elif total_entities < 50:
                recommendations.append("Small repository - all visualization modes should work well")
            elif total_entities > 500:
                recommendations.append("Large repository - consider using filters or limiting node count")
            
            if total_relationships == 0:
                recommendations.append("No relationships found - dependency visualization may be limited")
            elif total_relationships / total_entities > 5:
                recommendations.append("High connectivity - graph may be dense")
            
            stats["recommendations"] = recommendations
            stats["timestamp"] = datetime.utcnow().isoformat()
            
            return json.dumps(stats, indent=2)
            
        except Exception as e:
            logger.error(f"Error getting repository stats: {e}")
            return json.dumps({"error": str(e)})

    def _create_empty_graph_message(self, repository_url: str, graph_type: str = "repository") -> str:
        """Create message for empty graph scenarios."""
        return f"""
        <div style="padding: 20px; text-align: center; font-family: Arial, sans-serif;">
            <h3>No {graph_type} data found</h3>
            <p>Repository: <code>{repository_url}</code></p>
            <p>Try ingesting the repository first using the ingestion service.</p>
            <p>Command: <code>python3 -m src.ingestion_service.local_main --repository-url {repository_url} --branch main</code></p>
        </div>
        """