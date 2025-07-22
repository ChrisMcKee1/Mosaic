#!/usr/bin/env python3
"""
Simple test of neo4j-viz functionality without pandas dependency
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_neo4j_viz():
    """Test basic neo4j-viz functionality."""
    try:
        from neo4j_viz import Node, Relationship, VisualizationGraph
        
        print("✅ Successfully imported neo4j-viz")
        
        # Create sample nodes
        nodes = [
            Node(id=1, size=20, caption="main.py"),
            Node(id=2, size=15, caption="UserService"),
            Node(id=3, size=10, caption="authenticate_user"),
        ]
        
        # Create relationships
        relationships = [
            Relationship(source=1, target=2, caption="IMPORTS"),
            Relationship(source=2, target=3, caption="CALLS"),
        ]
        
        # Create visualization
        VG = VisualizationGraph(nodes=nodes, relationships=relationships)
        
        print("✅ Successfully created VisualizationGraph")
        
        # Generate HTML (this creates the interactive visualization)
        html_output = VG.render()
        
        print("✅ Successfully rendered graph to HTML")
        print(f"📊 HTML output length: {len(html_output)} characters")
        
        # Save to file
        with open("test_graph.html", "w") as f:
            f.write(html_output)
        
        print("✅ Graph saved to test_graph.html")
        print("🌟 Open test_graph.html in your browser to see the interactive visualization!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing neo4j-viz: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Neo4j-viz Graph Visualization")
    print("=" * 50)
    
    success = test_neo4j_viz()
    
    if success:
        print("=" * 50)
        print("🎉 All tests passed!")
        print("📁 Interactive visualization saved to test_graph.html")
    else:
        print("=" * 50)
        print("❌ Tests failed")
        sys.exit(1)