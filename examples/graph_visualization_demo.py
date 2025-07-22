#!/usr/bin/env python3
"""
Mosaic Graph Visualization Demo
Demonstrates the graph visualization capabilities with sample data
"""

import asyncio
import logging
import pandas as pd
from pathlib import Path
import sys

# Add src to path to import Mosaic modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neo4j_viz import Node, Relationship, VisualizationGraph
from neo4j_viz.pandas import from_dfs

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_sample_code_graph():
    """Create a sample code repository graph visualization."""
    logger.info("Creating sample code repository graph...")
    
    # Sample entities (code components)
    entities_data = [
        {"id": 1, "caption": "main.py", "entity_type": "module", "language": "python", "size": 20, "complexity": 15},
        {"id": 2, "caption": "UserService", "entity_type": "class", "language": "python", "size": 18, "complexity": 25},
        {"id": 3, "caption": "authenticate_user", "entity_type": "function", "language": "python", "size": 12, "complexity": 8},
        {"id": 4, "caption": "database.py", "entity_type": "module", "language": "python", "size": 16, "complexity": 20},
        {"id": 5, "caption": "DatabaseConnection", "entity_type": "class", "language": "python", "size": 14, "complexity": 18},
        {"id": 6, "caption": "execute_query", "entity_type": "function", "language": "python", "size": 10, "complexity": 12},
        {"id": 7, "caption": "utils.py", "entity_type": "module", "language": "python", "size": 12, "complexity": 5},
        {"id": 8, "caption": "hash_password", "entity_type": "function", "language": "python", "size": 8, "complexity": 4},
        {"id": 9, "caption": "config.py", "entity_type": "module", "language": "python", "size": 10, "complexity": 3},
        {"id": 10, "caption": "Settings", "entity_type": "class", "language": "python", "size": 12, "complexity": 6},
    ]
    
    # Sample relationships (code dependencies)
    relationships_data = [
        {"source": 1, "target": 2, "caption": "IMPORTS", "relationship_type": "imports"},
        {"source": 1, "target": 4, "caption": "IMPORTS", "relationship_type": "imports"},
        {"source": 1, "target": 9, "caption": "IMPORTS", "relationship_type": "imports"},
        {"source": 2, "target": 3, "caption": "DEFINES", "relationship_type": "defines"},
        {"source": 2, "target": 5, "caption": "USES", "relationship_type": "uses"},
        {"source": 4, "target": 5, "caption": "DEFINES", "relationship_type": "defines"},
        {"source": 5, "target": 6, "caption": "DEFINES", "relationship_type": "defines"},
        {"source": 3, "target": 8, "caption": "CALLS", "relationship_type": "calls"},
        {"source": 7, "target": 8, "caption": "DEFINES", "relationship_type": "defines"},
        {"source": 9, "target": 10, "caption": "DEFINES", "relationship_type": "defines"},
    ]
    
    # Convert to DataFrames
    entities_df = pd.DataFrame(entities_data)
    relationships_df = pd.DataFrame(relationships_data)
    
    # Create visualization
    VG = from_dfs(entities_df, relationships_df)
    
    # Color by entity type
    VG.color_nodes(property="entity_type")
    
    # Size by complexity
    VG.resize_nodes(property="complexity", node_radius_min_max=(8, 25))
    
    return VG


def create_dependency_graph():
    """Create a dependency-focused visualization."""
    logger.info("Creating dependency graph...")
    
    # Dependencies and modules
    entities_data = [
        {"id": 1, "caption": "requests", "entity_type": "external_library", "usage_count": 15, "size": 25},
        {"id": 2, "caption": "pandas", "entity_type": "external_library", "usage_count": 8, "size": 20},
        {"id": 3, "caption": "flask", "entity_type": "external_library", "usage_count": 12, "size": 22},
        {"id": 4, "caption": "auth_module", "entity_type": "internal_module", "usage_count": 6, "size": 15},
        {"id": 5, "caption": "data_processing", "entity_type": "internal_module", "usage_count": 10, "size": 18},
        {"id": 6, "caption": "api_handlers", "entity_type": "internal_module", "usage_count": 8, "size": 16},
        {"id": 7, "caption": "models", "entity_type": "internal_module", "usage_count": 12, "size": 20},
    ]
    
    relationships_data = [
        {"source": 4, "target": 1, "caption": "IMPORTS", "dependency_type": "imports"},
        {"source": 5, "target": 2, "caption": "IMPORTS", "dependency_type": "imports"},
        {"source": 6, "target": 3, "caption": "IMPORTS", "dependency_type": "imports"},
        {"source": 6, "target": 4, "caption": "IMPORTS", "dependency_type": "imports"},
        {"source": 5, "target": 7, "caption": "IMPORTS", "dependency_type": "imports"},
        {"source": 6, "target": 7, "caption": "IMPORTS", "dependency_type": "imports"},
    ]
    
    entities_df = pd.DataFrame(entities_data)
    relationships_df = pd.DataFrame(relationships_data)
    
    VG = from_dfs(entities_df, relationships_df)
    
    # Color by internal vs external
    VG.color_nodes(property="entity_type")
    
    # Size by usage frequency
    VG.resize_nodes(property="usage_count", node_radius_min_max=(10, 30))
    
    return VG


def create_knowledge_graph():
    """Create a semantic knowledge graph."""
    logger.info("Creating semantic knowledge graph...")
    
    entities_data = [
        {"id": 1, "caption": "authentication", "functional_cluster": "security", "importance_score": 0.95, "size": 25},
        {"id": 2, "caption": "user_management", "functional_cluster": "user_ops", "importance_score": 0.88, "size": 22},
        {"id": 3, "caption": "data_validation", "functional_cluster": "data_quality", "importance_score": 0.75, "size": 18},
        {"id": 4, "caption": "error_handling", "functional_cluster": "reliability", "importance_score": 0.82, "size": 20},
        {"id": 5, "caption": "logging", "functional_cluster": "observability", "importance_score": 0.78, "size": 19},
        {"id": 6, "caption": "database_ops", "functional_cluster": "data_access", "importance_score": 0.90, "size": 24},
        {"id": 7, "caption": "api_endpoints", "functional_cluster": "web_interface", "importance_score": 0.85, "size": 21},
        {"id": 8, "caption": "business_logic", "functional_cluster": "core_logic", "importance_score": 0.92, "size": 23},
    ]
    
    # Semantic similarity relationships
    relationships_data = [
        {"source": 1, "target": 2, "caption": "RELATED", "similarity_score": 0.85},
        {"source": 1, "target": 4, "caption": "RELATED", "similarity_score": 0.72},
        {"source": 2, "target": 6, "caption": "RELATED", "similarity_score": 0.78},
        {"source": 3, "target": 4, "caption": "RELATED", "similarity_score": 0.80},
        {"source": 4, "target": 5, "caption": "RELATED", "similarity_score": 0.75},
        {"source": 6, "target": 8, "caption": "RELATED", "similarity_score": 0.88},
        {"source": 7, "target": 8, "caption": "RELATED", "similarity_score": 0.82},
    ]
    
    entities_df = pd.DataFrame(entities_data)
    relationships_df = pd.DataFrame(relationships_data)
    
    VG = from_dfs(entities_df, relationships_df)
    
    # Color by functional cluster
    VG.color_nodes(property="functional_cluster")
    
    # Size by importance
    VG.resize_nodes(property="importance_score", node_radius_min_max=(12, 28))
    
    return VG


def main():
    """Main demo function."""
    logger.info("🎨 Starting Mosaic Graph Visualization Demo")
    
    try:
        # Create output directory
        output_dir = Path("graph_visualizations")
        output_dir.mkdir(exist_ok=True)
        
        # Generate different types of visualizations
        logger.info("1. Creating code repository structure visualization...")
        code_graph = create_sample_code_graph()
        with open(output_dir / "code_structure.html", "w") as f:
            f.write(code_graph.render())
        logger.info(f"✅ Code structure saved to: {output_dir}/code_structure.html")
        
        logger.info("2. Creating dependency visualization...")
        dep_graph = create_dependency_graph()
        with open(output_dir / "dependencies.html", "w") as f:
            f.write(dep_graph.render())
        logger.info(f"✅ Dependencies saved to: {output_dir}/dependencies.html")
        
        logger.info("3. Creating knowledge graph visualization...")
        knowledge_graph = create_knowledge_graph()
        with open(output_dir / "knowledge_graph.html", "w") as f:
            f.write(knowledge_graph.render())
        logger.info(f"✅ Knowledge graph saved to: {output_dir}/knowledge_graph.html")
        
        logger.info("🎉 Demo completed successfully!")
        logger.info(f"📁 Open the HTML files in {output_dir} to view the interactive visualizations")
        
        # Show what features are available
        print("\n" + "="*60)
        print("🌟 INTERACTIVE FEATURES AVAILABLE:")
        print("="*60)
        print("• 🖱️  Click and drag nodes to reposition them")
        print("• 🔍 Zoom in/out with mouse wheel")
        print("• 📋 Hover over nodes and edges for details")
        print("• 🎨 Color-coded by entity type or functional cluster")
        print("• 📏 Node sizes represent complexity or importance")
        print("• 🔗 Edges show relationships and dependencies")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()