"""
Mosaic Graph Plugin for MCP Interface
Provides natural language and SPARQL graph queries through MCP tools.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import json

from semantic_kernel.functions import kernel_function

from ..config.settings import get_settings
from .nl2sparql_service import NL2SPARQLService
from .graph_visualization import GraphVisualizationPlugin
from .graph_data_service import GraphDataService

logger = logging.getLogger(__name__)


class GraphPlugin:
    """
    Semantic Kernel plugin for graph-based queries and visualizations.

    Provides MCP-compatible tools for:
    - Natural language graph queries via NL2SPARQL translation
    - Direct SPARQL query execution
    - Graph result visualization and exploration
    - Graph schema discovery and exploration
    """

    def __init__(self, settings: Optional[Any] = None):
        """Initialize the Graph Plugin with required services."""
        self.settings = settings or get_settings()
        self.nl2sparql_service = NL2SPARQLService()
        self.visualization_plugin = GraphVisualizationPlugin(settings)
        self.data_service = GraphDataService(settings)
        logger.info("GraphPlugin initialized")

    async def initialize(self) -> None:
        """Initialize async components and services."""
        try:
            await self.nl2sparql_service.initialize()
            await self.visualization_plugin.initialize()
            await self.data_service.initialize()
            logger.info("GraphPlugin services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GraphPlugin services: {e}")
            raise

    @kernel_function(
        name="natural_language_query",
        description="Execute natural language queries against the graph database",
    )
    async def natural_language_query(
        self, query: str, include_visualization: bool = False, max_results: int = 100
    ) -> Dict[str, Any]:
        """
        Translate natural language to SPARQL and execute against graph database.

        Args:
            query: Natural language query string
            include_visualization: Whether to generate visualization of results
            max_results: Maximum number of results to return

        Returns:
            Dict containing query results, metadata, and optional visualization
        """
        try:
            start_time = datetime.now(timezone.utc)
            logger.info(f"Processing natural language graph query: {query}")

            # Translate and execute query
            response = await self.nl2sparql_service.translate_and_execute(
                natural_language_query=query,
                execute_query=True,
                max_results=max_results,
            )

            # Extract results and metadata
            results = response.get("results", [])
            sparql_query = response.get("sparql_query", "")
            execution_time = response.get("execution_time_ms", 0)

            # Prepare response
            graph_response: Dict[str, Any] = {
                "query": query,
                "sparql_query": sparql_query,
                "results": results,
                "result_count": len(results) if results else 0,
                "execution_time_ms": execution_time,
                "timestamp": start_time.isoformat(),
                "success": True,
            }

            # Add visualization if requested
            if include_visualization and results:
                try:
                    visualization_html = await self._create_results_visualization(
                        results, query
                    )
                    graph_response["visualization"] = visualization_html
                except Exception as viz_error:
                    logger.warning(f"Visualization generation failed: {viz_error}")
                    graph_response["visualization_error"] = str(viz_error)

            logger.info(
                f"Natural language query completed: {len(results) if results else 0} results in {execution_time}ms"
            )
            return graph_response

        except Exception as e:
            logger.error(f"Natural language query failed: {e}")
            return {
                "query": query,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @kernel_function(
        name="execute_sparql_query",
        description="Execute SPARQL queries directly against the graph database",
    )
    async def execute_sparql_query(
        self,
        sparql_query: str,
        include_visualization: bool = False,
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """
        Execute SPARQL query directly against the graph database.

        Args:
            sparql_query: SPARQL query string
            include_visualization: Whether to generate visualization of results
            max_results: Maximum number of results to return

        Returns:
            Dict containing query results, metadata, and optional visualization
        """
        try:
            start_time = datetime.now(timezone.utc)
            logger.info(f"Executing SPARQL query: {sparql_query[:100]}...")

            # For now, we'll use a natural language wrapper for the SPARQL query
            # This is a limitation of the current NL2SPARQL service design
            nl_wrapper = f"Execute this SPARQL query: {sparql_query}"

            response = await self.nl2sparql_service.translate_and_execute(
                natural_language_query=nl_wrapper,
                execute_query=True,
                max_results=max_results,
            )

            # Extract results and metadata
            results = response.get("results", [])
            execution_time = response.get("execution_time_ms", 0)

            # Prepare response
            graph_response: Dict[str, Any] = {
                "sparql_query": sparql_query,
                "results": results,
                "result_count": len(results) if results else 0,
                "execution_time_ms": execution_time,
                "timestamp": start_time.isoformat(),
                "success": True,
            }

            # Add visualization if requested
            if include_visualization and results:
                try:
                    visualization_html = await self._create_results_visualization(
                        results, f"SPARQL: {sparql_query[:50]}..."
                    )
                    graph_response["visualization"] = visualization_html
                except Exception as viz_error:
                    logger.warning(f"Visualization generation failed: {viz_error}")
                    graph_response["visualization_error"] = str(viz_error)

            logger.info(
                f"SPARQL query completed: {len(results) if results else 0} results in {execution_time}ms"
            )
            return graph_response

        except Exception as e:
            logger.error(f"SPARQL query execution failed: {e}")
            return {
                "sparql_query": sparql_query,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @kernel_function(
        name="visualize_graph_results",
        description="Create interactive visualizations of graph query results",
    )
    async def visualize_graph_results(
        self,
        results: List[Dict[str, Any]],
        title: str = "Graph Query Results",
        layout: str = "force",
    ) -> str:
        """
        Create interactive visualization of graph query results.

        Args:
            results: List of graph query results to visualize
            title: Title for the visualization
            layout: Layout algorithm (force, circular, hierarchical)

        Returns:
            HTML string containing interactive graph visualization
        """
        try:
            logger.info(f"Creating visualization for {len(results)} graph results")

            if not results:
                return self._create_empty_visualization(title)

            # Create visualization using existing plugin
            visualization_html = await self._create_results_visualization(
                results, title, layout
            )

            logger.info("Graph results visualization created successfully")
            return visualization_html

        except Exception as e:
            logger.error(f"Graph visualization failed: {e}")
            return self._create_error_visualization(str(e), title)

    @kernel_function(
        name="discover_graph_schema",
        description="Explore and discover graph database schema and relationships",
    )
    async def discover_graph_schema(
        self,
        entity_type: Optional[str] = None,
        include_counts: bool = True,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        """
        Discover available graph schema, entities, and relationships.

        Args:
            entity_type: Optional filter for specific entity type
            include_counts: Whether to include entity/relationship counts
            max_results: Maximum number of schema elements to return

        Returns:
            Dict containing schema information and metadata
        """
        try:
            start_time = datetime.now(timezone.utc)
            logger.info(f"Discovering graph schema (entity_type: {entity_type})")

            # Build schema discovery SPARQL query
            if entity_type:
                pass
            else:
                pass

            # Execute schema discovery query via NL2SPARQL service (using translate_and_execute with predefined SPARQL)
            # Since there's no direct execute_sparql method, we'll create a simple NL query that should translate to our SPARQL
            if entity_type:
                nl_query = f"Show properties and their types for {entity_type} entities"
            else:
                nl_query = "Show all entity types and their counts"

            response = await self.nl2sparql_service.translate_and_execute(
                natural_language_query=nl_query,
                execute_query=True,
                max_results=max_results,
            )

            results = response.get("results", [])
            execution_time = response.get("execution_time_ms", 0)

            # Process and format schema information
            schema_info: Dict[str, Any] = {
                "entity_type_filter": entity_type,
                "schema_elements": results,
                "element_count": len(results) if results else 0,
                "execution_time_ms": execution_time,
                "timestamp": start_time.isoformat(),
                "success": True,
            }

            # Add relationship discovery if no entity type filter
            if not entity_type:
                rel_nl_query = "Show all relationship types and their usage counts"
                rel_response = await self.nl2sparql_service.translate_and_execute(
                    natural_language_query=rel_nl_query,
                    execute_query=True,
                    max_results=20,
                )

                schema_info["relationships"] = rel_response.get("results", [])

            logger.info(
                f"Schema discovery completed: {len(results) if results else 0} elements in {execution_time}ms"
            )
            return schema_info

        except Exception as e:
            logger.error(f"Graph schema discovery failed: {e}")
            return {
                "entity_type_filter": entity_type,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def _create_results_visualization(
        self, results: List[Dict[str, Any]], title: str, layout: str = "force"
    ) -> str:
        """Create visualization HTML for graph query results."""
        try:
            # Convert results to visualization format
            nodes: List[Dict[str, Any]] = []
            edges: List[Dict[str, Any]] = []

            for result in results:
                # Extract entities and relationships from SPARQL results
                for key, value in result.items():
                    if key.endswith("_subject") or key.endswith("_entity"):
                        nodes.append(
                            {
                                "id": str(value),
                                "label": str(value).split("/")[
                                    -1
                                ],  # Use URI suffix as label
                                "type": "entity",
                                "data": result,
                            }
                        )
                    elif key.endswith("_relationship") or key.endswith("_predicate"):
                        # Add relationship as edge if we have subject/object
                        subject_key = key.replace("_relationship", "_subject").replace(
                            "_predicate", "_subject"
                        )
                        object_key = key.replace("_relationship", "_object").replace(
                            "_predicate", "_object"
                        )

                        if subject_key in result and object_key in result:
                            edges.append(
                                {
                                    "source": str(result[subject_key]),
                                    "target": str(result[object_key]),
                                    "label": str(value).split("/")[-1],
                                    "type": "relationship",
                                }
                            )

            # Remove duplicate nodes
            unique_nodes_dict = {node["id"]: node for node in nodes}
            unique_nodes = list(unique_nodes_dict.values())

            # Create simple HTML visualization if we have nodes/edges
            if unique_nodes or edges:
                return self._generate_interactive_graph_html(
                    unique_nodes, edges, title, layout
                )
            else:
                # Fallback to table visualization
                return self._generate_table_visualization(results, title)

        except Exception as e:
            logger.warning(f"Visualization creation failed: {e}")
            return self._generate_table_visualization(results, title)

    def _generate_interactive_graph_html(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        title: str,
        layout: str = "force",
    ) -> str:
        """Generate interactive graph HTML using D3.js."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                .node {{ fill: steelblue; stroke: #fff; stroke-width: 1.5px; }}
                .link {{ stroke: #999; stroke-opacity: 0.6; }}
                .node-label {{ font-family: Arial; font-size: 12px; }}
                .edge-label {{ font-family: Arial; font-size: 10px; fill: #666; }}
                svg {{ border: 1px solid #ccc; }}
            </style>
        </head>
        <body>
            <h2>{title}</h2>
            <p>Nodes: {len(nodes)}, Edges: {len(edges)}</p>
            <svg width="800" height="600"></svg>
            <script>
                const nodes = {json.dumps(nodes)};
                const links = {json.dumps(edges)};
                
                const svg = d3.select("svg");
                const width = +svg.attr("width");
                const height = +svg.attr("height");
                
                const simulation = d3.forceSimulation(nodes)
                    .force("link", d3.forceLink(links).id(d => d.id))
                    .force("charge", d3.forceManyBody().strength(-300))
                    .force("center", d3.forceCenter(width / 2, height / 2));
                
                const link = svg.append("g")
                    .selectAll("line")
                    .data(links)
                    .enter().append("line")
                    .attr("class", "link");
                
                const node = svg.append("g")
                    .selectAll("circle")
                    .data(nodes)
                    .enter().append("circle")
                    .attr("class", "node")
                    .attr("r", 5)
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));
                
                node.append("title").text(d => d.label);
                
                simulation.on("tick", () => {{
                    link.attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);
                    
                    node.attr("cx", d => d.x)
                        .attr("cy", d => d.y);
                }});
                
                function dragstarted(event, d) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }}
                
                function dragged(event, d) {{
                    d.fx = event.x;
                    d.fy = event.y;
                }}
                
                function dragended(event, d) {{
                    if (!event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }}
            </script>
        </body>
        </html>
        """

    def _generate_table_visualization(
        self, results: List[Dict[str, Any]], title: str
    ) -> str:
        """Generate table-based visualization for results."""
        if not results:
            return self._create_empty_visualization(title)

        # Create HTML table
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metadata {{ color: #666; font-size: 0.9em; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <h2>{title}</h2>
            <div class="metadata">Results: {len(results)} rows</div>
            <table>
                <thead>
                    <tr>
        """

        # Add headers
        if results:
            for key in results[0].keys():
                html += f"<th>{key}</th>"

        html += """
                    </tr>
                </thead>
                <tbody>
        """

        # Add data rows
        for result in results[:100]:  # Limit to 100 rows for display
            html += "<tr>"
            for value in result.values():
                # Truncate long values
                display_value = str(value)
                if len(display_value) > 100:
                    display_value = display_value[:100] + "..."
                html += f"<td>{display_value}</td>"
            html += "</tr>"

        html += """
                </tbody>
            </table>
        </body>
        </html>
        """

        return html

    def _create_empty_visualization(self, title: str) -> str:
        """Create empty visualization placeholder."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>{title}</title></head>
        <body>
            <h2>{title}</h2>
            <p>No results to visualize.</p>
        </body>
        </html>
        """

    def _create_error_visualization(self, error: str, title: str) -> str:
        """Create error visualization placeholder."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>{title} - Error</title></head>
        <body>
            <h2>{title}</h2>
            <p style="color: red;">Visualization Error: {error}</p>
        </body>
        </html>
        """
