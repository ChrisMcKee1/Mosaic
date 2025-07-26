"""
Standalone SPARQL Builder for validation - no dependencies on existing RDF module.

This is a standalone implementation for testing OMR-P1-007 functionality
without the complex import dependencies of the existing system.

Task: OMR-P1-007 - Implement Basic SPARQL Query Capability
"""

import logging
from typing import List, Dict, Optional, Any
from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal
from rdflib.plugins.sparql import prepareQuery

logger = logging.getLogger(__name__)


class StandaloneSparqlBuilder:
    """
    Standalone SPARQL query builder for validation.

    This version is independent of the existing RDF module structure
    and provides the core SPARQL functionality for testing.
    """

    def __init__(self, cosmos_client=None):
        """
        Initialize the SPARQL builder.

        Args:
            cosmos_client: Azure Cosmos DB client instance (can be None for testing)
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

        logger.info("StandaloneSparqlBuilder initialized with RDFLib graph")

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


def validate_sparql_builder():
    """
    Validate SPARQL Builder functionality.
    """
    try:
        logger.info("=== Standalone SPARQL Builder Validation ===")

        # Create SPARQL builder
        sparql_builder = StandaloneSparqlBuilder()

        logger.info("✓ SparqlBuilder created successfully")

        # Test 1: Verify initialization
        assert hasattr(sparql_builder, "graph")
        assert hasattr(sparql_builder, "namespaces")
        assert isinstance(sparql_builder.graph, Graph)
        assert "code" in sparql_builder.namespaces
        logger.info("✓ Initialization test passed")

        # Test 2: Add sample data and test basic queries
        code_ns = sparql_builder.namespaces["code"]

        # Add sample function
        func_uri = URIRef(f"{code_ns}test_function")
        sparql_builder.graph.add((func_uri, RDF.type, URIRef(f"{code_ns}Function")))
        sparql_builder.graph.add(
            (func_uri, URIRef(f"{code_ns}name"), Literal("test_function"))
        )
        sparql_builder.graph.add(
            (func_uri, URIRef(f"{code_ns}inModule"), URIRef(f"{code_ns}test_module"))
        )

        # Add another function for call relationship
        func2_uri = URIRef(f"{code_ns}helper_function")
        sparql_builder.graph.add((func2_uri, RDF.type, URIRef(f"{code_ns}Function")))
        sparql_builder.graph.add(
            (func2_uri, URIRef(f"{code_ns}name"), Literal("helper_function"))
        )
        sparql_builder.graph.add((func_uri, URIRef(f"{code_ns}calls"), func2_uri))

        logger.info("✓ Sample data added successfully")

        # Test 3: Execute SPARQL query for functions in module
        try:
            functions = sparql_builder.query_functions_in_module("test_module")
            assert len(functions) >= 1
            assert functions[0]["functionName"] == "test_function"
            logger.info("✓ Functions in module query test passed")
        except Exception as e:
            logger.error(f"✗ Functions in module query failed: {e}")
            raise

        # Test 4: Execute SPARQL query for function calls
        try:
            calls = sparql_builder.query_function_calls()
            assert len(calls) >= 1
            assert calls[0]["callerName"] == "test_function"
            assert calls[0]["calleeName"] == "helper_function"
            logger.info("✓ Function calls query test passed")
        except Exception as e:
            logger.error(f"✗ Function calls query failed: {e}")
            raise

        # Test 5: Test graph statistics
        try:
            stats = sparql_builder.get_graph_statistics()
            assert stats["total_triples"] > 0
            assert stats["unique_subjects"] > 0
            logger.info(f"✓ Graph statistics test passed: {stats}")
        except Exception as e:
            logger.error(f"✗ Graph statistics failed: {e}")
            raise

        # Test 6: Test clear graph
        try:
            original_count = len(sparql_builder.graph)
            sparql_builder.clear_graph()
            assert len(sparql_builder.graph) == 0
            logger.info("✓ Clear graph test passed")
        except Exception as e:
            logger.error(f"✗ Clear graph failed: {e}")
            raise

        logger.info("=== All validation tests completed successfully ===")
        return True

    except Exception as e:
        logger.error(f"✗ Validation failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    success = validate_sparql_builder()
    if success:
        print("\n🎉 SPARQL Builder validation PASSED")
        sys.exit(0)
    else:
        print("\n❌ SPARQL Builder validation FAILED")
        sys.exit(1)
