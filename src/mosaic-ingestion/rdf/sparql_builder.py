"""
SPARQL Query Builder for Code Relationship Analysis.

This module provides SPARQL query capabilities for analyzing code relationships
stored in RDF format. It integrates with Cosmos DB to load triples and executes
SPARQL queries using RDFLib for code analysis scenarios.

Author: Mosaic MCP Tool
Task: OMR-P1-007 - Implement Basic SPARQL Query Capability
"""

import logging
from typing import List, Dict, Optional, Any
from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal
from rdflib.plugins.sparql import prepareQuery
from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions

logger = logging.getLogger(__name__)


class SparqlBuilder:
    """
    SPARQL query builder for code relationship analysis.

    Provides methods to load RDF triples from Cosmos DB and execute
    SPARQL queries for code analysis scenarios including function calls,
    inheritance hierarchies, and module dependencies.
    """

    def __init__(self, cosmos_client: CosmosClient):
        """
        Initialize the SPARQL builder with Cosmos DB client.

        Args:
            cosmos_client: Azure Cosmos DB client instance
        """
        self.cosmos_client = cosmos_client
        self.graph = Graph()

        # Define namespaces for code ontology
        self.namespaces = {
            "code": Namespace("http://mosaic.local/code/"),
            "rdf": RDF,
            "rdfs": RDFS,
            "mosaic": Namespace("http://mosaic.local/"),
        }

        # Bind namespaces to graph
        for prefix, namespace in self.namespaces.items():
            self.graph.bind(prefix, namespace)

        logger.info("SparqlBuilder initialized with RDFLib graph")

    def load_triples_from_documents(self, container_name: str) -> int:
        """
        Load RDF triples from Cosmos DB documents into in-memory graph.

        Args:
            container_name: Name of the Cosmos DB container containing RDF data

        Returns:
            Number of triples loaded into the graph

        Raises:
            Exception: If Cosmos DB access fails or RDF data is invalid
        """
        try:
            database = self.cosmos_client.get_database_client("mosaic-knowledge")
            container = database.get_container_client(container_name)

            triple_count = 0

            # Query all documents in the container
            query = "SELECT * FROM c"
            items = container.query_items(
                query=query, enable_cross_partition_query=True
            )

            for item in items:
                # Extract RDF triples from document
                if "rdf_triples" in item:
                    for triple_data in item["rdf_triples"]:
                        subject = URIRef(triple_data["subject"])
                        predicate = URIRef(triple_data["predicate"])

                        # Handle object type (URI or Literal)
                        if triple_data.get("object_type") == "literal":
                            obj = Literal(triple_data["object"])
                        else:
                            obj = URIRef(triple_data["object"])

                        # Add triple to graph
                        self.graph.add((subject, predicate, obj))
                        triple_count += 1

            logger.info(f"Loaded {triple_count} triples from {container_name}")
            return triple_count

        except cosmos_exceptions.CosmosResourceNotFoundError as e:
            logger.error(f"Container {container_name} not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load triples from {container_name}: {e}")
            raise

    def query_functions_in_module(self, module_name: str) -> List[Dict[str, Any]]:
        """
        Query all functions defined in a specific module.

        Args:
            module_name: Name of the module to query

        Returns:
            List of dictionaries containing function information
        """
        sparql_query = """
        PREFIX code: <http://mosaic.local/code/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?function ?functionName
        WHERE {
            ?function a code:Function ;
                     code:inModule ?module ;
                     code:name ?functionName .
            FILTER(regex(str(?module), ?moduleName, "i"))
        }
        """

        return self._execute_query(sparql_query, {"moduleName": Literal(module_name)})

    def query_function_calls(
        self, function_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query function call relationships.

        Args:
            function_name: Optional specific function to query calls for

        Returns:
            List of dictionaries containing caller/callee relationships
        """
        if function_name:
            sparql_query = """
            PREFIX code: <http://mosaic.local/code/>
            
            SELECT ?caller ?callee ?callerName ?calleeName
            WHERE {
                ?caller code:calls ?callee ;
                       code:name ?callerName .
                ?callee code:name ?calleeName .
                FILTER(regex(str(?callee), ?functionName, "i"))
            }
            """
            return self._execute_query(
                sparql_query, {"functionName": Literal(function_name)}
            )
        else:
            sparql_query = """
            PREFIX code: <http://mosaic.local/code/>
            
            SELECT ?caller ?callee ?callerName ?calleeName
            WHERE {
                ?caller code:calls ?callee ;
                       code:name ?callerName .
                ?callee code:name ?calleeName .
            }
            """
            return self._execute_query(sparql_query)

    def query_inheritance_hierarchy(
        self, class_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query class inheritance relationships.

        Args:
            class_name: Optional specific class to query inheritance for

        Returns:
            List of dictionaries containing child/parent class relationships
        """
        if class_name:
            sparql_query = """
            PREFIX code: <http://mosaic.local/code/>
            
            SELECT ?child ?parent ?childName ?parentName
            WHERE {
                ?child code:inherits ?parent ;
                      code:name ?childName .
                ?parent code:name ?parentName .
                FILTER(regex(str(?child), ?className, "i"))
            }
            """
            return self._execute_query(sparql_query, {"className": Literal(class_name)})
        else:
            sparql_query = """
            PREFIX code: <http://mosaic.local/code/>
            
            SELECT ?child ?parent ?childName ?parentName
            WHERE {
                ?child code:inherits ?parent ;
                      code:name ?childName .
                ?parent code:name ?parentName .
            }
            """
            return self._execute_query(sparql_query)

    def query_transitive_dependencies(self, module_name: str) -> List[Dict[str, Any]]:
        """
        Query transitive module dependencies using property paths.

        Args:
            module_name: Name of the module to find dependencies for

        Returns:
            List of dictionaries containing transitive dependencies
        """
        sparql_query = """
        PREFIX code: <http://mosaic.local/code/>
        
        SELECT ?module ?dependency ?dependencyName
        WHERE {
            ?module code:imports+ ?dependency ;
                   code:name ?moduleName .
            ?dependency code:name ?dependencyName .
            FILTER(regex(str(?module), ?moduleNameFilter, "i"))
        }
        """

        return self._execute_query(
            sparql_query, {"moduleNameFilter": Literal(module_name)}
        )

    def query_code_complexity_metrics(self) -> List[Dict[str, Any]]:
        """
        Query aggregated code complexity metrics.

        Returns:
            List of dictionaries containing complexity metrics
        """
        sparql_query = """
        PREFIX code: <http://mosaic.local/code/>
        
        SELECT ?function ?functionName (COUNT(?callee) as ?callCount)
        WHERE {
            ?function a code:Function ;
                     code:name ?functionName ;
                     code:calls ?callee .
        }
        GROUP BY ?function ?functionName
        ORDER BY DESC(?callCount)
        """

        return self._execute_query(sparql_query)

    def query_unused_functions(self) -> List[Dict[str, Any]]:
        """
        Query functions that are not called by any other function.

        Returns:
            List of dictionaries containing unused functions
        """
        sparql_query = """
        PREFIX code: <http://mosaic.local/code/>
        
        SELECT ?function ?functionName
        WHERE {
            ?function a code:Function ;
                     code:name ?functionName .
            FILTER NOT EXISTS { ?caller code:calls ?function }
        }
        """

        return self._execute_query(sparql_query)

    def _execute_query(
        self, sparql_query: str, bindings: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SPARQL query against the loaded graph.

        Args:
            sparql_query: SPARQL query string
            bindings: Optional variable bindings for the query

        Returns:
            List of dictionaries containing query results

        Raises:
            Exception: If query execution fails
        """
        try:
            # Prepare and execute query
            if bindings:
                prepared_query = prepareQuery(sparql_query)
                results = self.graph.query(prepared_query, initBindings=bindings)
            else:
                results = self.graph.query(sparql_query)

            # Convert results to list of dictionaries
            result_list = []
            for row in results:
                result_dict = {}
                for i, var in enumerate(results.vars):
                    value = row[i]
                    if value is not None:
                        result_dict[str(var)] = str(value)
                    else:
                        result_dict[str(var)] = None
                result_list.append(result_dict)

            logger.info(
                f"SPARQL query executed successfully, returned {len(result_list)} results"
            )
            return result_list

        except Exception as e:
            logger.error(f"SPARQL query execution failed: {e}")
            logger.error(f"Query: {sparql_query}")
            raise

    def get_graph_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the loaded RDF graph.

        Returns:
            Dictionary containing graph statistics
        """
        stats = {
            "total_triples": len(self.graph),
            "unique_subjects": len(set(self.graph.subjects())),
            "unique_predicates": len(set(self.graph.predicates())),
            "unique_objects": len(set(self.graph.objects())),
        }

        logger.info(f"Graph statistics: {stats}")
        return stats

    def clear_graph(self):
        """Clear all triples from the graph."""
        self.graph.remove((None, None, None))
        logger.info("RDF graph cleared")
