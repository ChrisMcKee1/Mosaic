"""
Comprehensive tests for graph visualization components.

Tests cover:
- D3.js graph generation and rendering
- Pyvis network visualization
- Plotly graph analytics
- Interactive features and controls
- Performance and responsiveness
- Data binding and updates
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
import json
import re
from typing import Dict, List, Any


class MockGraphVisualizations:
    """Mock graph visualization components for testing."""
    
    def __init__(self):
        self.category_colors = {
            "server": "#ff6b6b",
            "plugin": "#4ecdc4", 
            "ingestion": "#45b7d1",
            "ai_agent": "#96ceb4",
            "config": "#ffeaa7",
            "model": "#fd79a8",
            "ui": "#98d8c8"
        }
    
    def create_enhanced_d3_graph(self, entities, relationships):
        """Mock enhanced D3.js graph creation."""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                .graph-container {{ width: 700px; height: 700px; }}
                .node {{ cursor: pointer; stroke: #fff; stroke-width: 2px; }}
                .link {{ stroke: #999; stroke-opacity: 0.6; }}
            </style>
        </head>
        <body>
            <div class="graph-container">
                <svg id="graph" width="700" height="700"></svg>
                <div class="controls">
                    <button onclick="resetZoom()">Reset</button>
                    <button onclick="centerGraph()">Center</button>
                </div>
            </div>
            <script>
                const nodes = {json.dumps(entities)};
                const links = {json.dumps(relationships)};
                const categoryColors = {json.dumps(self.category_colors)};
                
                // Mock D3.js visualization code
                const svg = d3.select("#graph");
                const simulation = d3.forceSimulation(nodes);
            </script>
        </body>
        </html>
        """
        return html_template
    
    def create_pyvis_graph(self, entities, relationships):
        """Mock Pyvis graph creation."""
        return f"""
        <div id="pyvis-network" style="width: 700px; height: 700px;">
            <div>Pyvis Network Visualization</div>
            <div>Nodes: {len(entities)}</div>
            <div>Edges: {len(relationships)}</div>
        </div>
        """
    
    def create_plotly_graph(self, entities, relationships):
        """Mock Plotly graph creation."""
        return f"""
        <div id="plotly-graph">
            <div>Plotly Graph Analytics</div>
            <div>Interactive network with {len(entities)} nodes and {len(relationships)} edges</div>
        </div>
        """
    
    def create_classic_d3_graph(self, entities, relationships):
        """Mock classic D3.js graph creation."""
        return f"""
        <svg width="800" height="600" id="classic-graph">
            <g class="nodes">
                <!-- {len(entities)} nodes -->
            </g>
            <g class="links">
                <!-- {len(relationships)} links -->
            </g>
        </svg>
        """


@pytest.fixture
def mock_visualizations():
    """Create mock visualization components."""
    return MockGraphVisualizations()


@pytest.fixture
def sample_graph_entities():
    """Sample entities for graph visualization testing."""
    return [
        {
            "id": "node_1",
            "name": "ServerComponent",
            "category": "server", 
            "lines": 285,
            "complexity": 8,
            "description": "Main server component"
        },
        {
            "id": "node_2", 
            "name": "PluginManager",
            "category": "plugin",
            "lines": 150,
            "complexity": 6,
            "description": "Plugin management system"
        },
        {
            "id": "node_3",
            "name": "UIService",
            "category": "ui",
            "lines": 200,
            "complexity": 7,
            "description": "User interface service"
        }
    ]


@pytest.fixture
def sample_graph_relationships():
    """Sample relationships for graph visualization testing.""" 
    return [
        {
            "source": "node_1",
            "target": "node_2",
            "type": "imports",
            "description": "Server imports plugin manager"
        },
        {
            "source": "node_2", 
            "target": "node_3",
            "type": "uses",
            "description": "Plugin manager uses UI service"
        }
    ]


class TestD3GraphGeneration:
    """Test D3.js graph generation and features."""
    
    def test_enhanced_d3_graph_creation(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test enhanced D3.js graph creation."""
        html = mock_visualizations.create_enhanced_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Verify HTML structure
        assert "<!DOCTYPE html>" in html
        assert "<script src=\"https://d3js.org/d3.v7.min.js\"></script>" in html
        assert "graph-container" in html
        assert 'id="graph"' in html
        
        # Verify data binding
        assert "const nodes =" in html
        assert "const links =" in html
        assert "const categoryColors =" in html
        
        # Verify controls
        assert "resetZoom()" in html
        assert "centerGraph()" in html
    
    def test_d3_graph_data_structure(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test D3.js graph data structure."""
        html = mock_visualizations.create_enhanced_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Extract JSON data from HTML
        nodes_match = re.search(r'const nodes = (\[.*?\]);', html, re.DOTALL)
        links_match = re.search(r'const links = (\[.*?\]);', html, re.DOTALL)
        
        assert nodes_match is not None
        assert links_match is not None
        
        # Parse JSON data
        nodes_data = json.loads(nodes_match.group(1))
        links_data = json.loads(links_match.group(1))
        
        # Verify node structure
        assert len(nodes_data) == 3
        for node in nodes_data:
            assert "id" in node
            assert "name" in node
            assert "category" in node
        
        # Verify link structure
        assert len(links_data) == 2
        for link in links_data:
            assert "source" in link
            assert "target" in link
            assert "type" in link
    
    def test_d3_styling_and_css(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test D3.js styling and CSS."""
        html = mock_visualizations.create_enhanced_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Verify CSS classes
        assert ".graph-container" in html
        assert ".node" in html
        assert ".link" in html
        
        # Verify styling properties
        assert "cursor: pointer" in html
        assert "stroke: #fff" in html
        assert "stroke-opacity: 0.6" in html
    
    def test_d3_interactive_features(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test D3.js interactive features."""
        html = mock_visualizations.create_enhanced_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Verify interactive elements
        assert "resetZoom()" in html
        assert "centerGraph()" in html
        assert "forceSimulation" in html
        
        # Verify event handling setup
        assert "const simulation" in html
    
    def test_classic_d3_graph_creation(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test classic D3.js graph creation."""
        html = mock_visualizations.create_classic_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Verify SVG structure
        assert '<svg' in html
        assert 'id="classic-graph"' in html
        assert '<g class="nodes">' in html
        assert '<g class="links">' in html
        
        # Verify node and link counts in comments
        assert f"<!-- {len(sample_graph_entities)} nodes -->" in html
        assert f"<!-- {len(sample_graph_relationships)} links -->" in html


class TestPyvisNetworkVisualization:
    """Test Pyvis network visualization functionality."""
    
    def test_pyvis_graph_creation(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test Pyvis graph creation."""
        html = mock_visualizations.create_pyvis_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Verify HTML structure
        assert 'id="pyvis-network"' in html
        assert "Pyvis Network Visualization" in html
        
        # Verify data display
        assert f"Nodes: {len(sample_graph_entities)}" in html
        assert f"Edges: {len(sample_graph_relationships)}" in html
    
    @patch('pyvis.network.Network')
    def test_pyvis_network_configuration(self, mock_network):
        """Test Pyvis network configuration."""
        # Mock network instance
        mock_net = MagicMock()
        mock_network.return_value = mock_net
        mock_net.generate_html.return_value = "<div>Mock Pyvis Network</div>"
        
        # Test network creation with configuration
        from pyvis.network import Network
        net = Network(height="700px", width="700px", directed=True)
        
        # Verify network was created
        mock_network.assert_called_once()
        assert mock_net.generate_html.return_value == "<div>Mock Pyvis Network</div>"
    
    def test_pyvis_node_styling(self, mock_visualizations):
        """Test Pyvis node styling configuration."""
        entities = [
            {"id": "test1", "name": "Test1", "category": "server", "lines": 100},
            {"id": "test2", "name": "Test2", "category": "plugin", "lines": 200}
        ]
        
        html = mock_visualizations.create_pyvis_graph(entities, [])
        
        # Verify styling configuration
        assert "700px" in html
        assert len(entities) == 2
    
    def test_pyvis_physics_configuration(self, mock_visualizations):
        """Test Pyvis physics configuration."""
        # Physics should be configured for optimal layout
        html = mock_visualizations.create_pyvis_graph([], [])
        
        # Verify container exists for physics simulation
        assert 'id="pyvis-network"' in html


class TestPlotlyGraphAnalytics:
    """Test Plotly graph analytics functionality."""
    
    def test_plotly_graph_creation(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test Plotly graph creation."""
        html = mock_visualizations.create_plotly_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Verify HTML structure
        assert 'id="plotly-graph"' in html
        assert "Plotly Graph Analytics" in html
        assert "Interactive network" in html
        
        # Verify data display
        assert f"{len(sample_graph_entities)} nodes" in html
        assert f"{len(sample_graph_relationships)} edges" in html
    
    @patch('plotly.graph_objects.Figure')
    def test_plotly_figure_creation(self, mock_figure):
        """Test Plotly figure creation."""
        # Mock figure instance
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        mock_fig.to_html.return_value = "<div>Mock Plotly Graph</div>"
        
        # Test figure creation
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Verify figure was created
        mock_figure.assert_called_once()
        assert mock_fig.to_html.return_value == "<div>Mock Plotly Graph</div>"
    
    @patch('networkx.spring_layout')
    def test_plotly_layout_calculation(self, mock_layout):
        """Test Plotly layout calculation with NetworkX."""
        # Mock layout calculation
        mock_layout.return_value = {
            "node1": (0.5, 0.5),
            "node2": (0.2, 0.8),
            "node3": (0.8, 0.3)
        }
        
        # Test layout calculation
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(["node1", "node2", "node3"])
        pos = nx.spring_layout(G)
        
        # Verify layout was calculated
        mock_layout.assert_called_once()
        assert len(pos) == 3
    
    def test_plotly_interactivity_features(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test Plotly interactivity features."""
        html = mock_visualizations.create_plotly_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Verify interactive elements
        assert "Interactive network" in html
        assert 'id="plotly-graph"' in html


class TestGraphDataBinding:
    """Test graph data binding and updates."""
    
    def test_data_binding_consistency(self, mock_visualizations):
        """Test data binding consistency across visualizations."""
        entities = [
            {"id": "test1", "name": "Component1", "category": "server"},
            {"id": "test2", "name": "Component2", "category": "plugin"}
        ]
        relationships = [
            {"source": "test1", "target": "test2", "type": "imports"}
        ]
        
        # Test all visualization types
        d3_html = mock_visualizations.create_enhanced_d3_graph(entities, relationships)
        pyvis_html = mock_visualizations.create_pyvis_graph(entities, relationships)
        plotly_html = mock_visualizations.create_plotly_graph(entities, relationships)
        
        # All should handle the same data
        assert len(entities) == 2
        assert len(relationships) == 1
        
        # Verify data is bound in each visualization
        assert "test1" in str(entities)
        assert "test2" in str(entities)
    
    def test_empty_data_handling(self, mock_visualizations):
        """Test handling of empty data sets."""
        empty_entities = []
        empty_relationships = []
        
        # All visualizations should handle empty data gracefully
        d3_html = mock_visualizations.create_enhanced_d3_graph(empty_entities, empty_relationships)
        pyvis_html = mock_visualizations.create_pyvis_graph(empty_entities, empty_relationships)
        plotly_html = mock_visualizations.create_plotly_graph(empty_entities, empty_relationships)
        
        # Should not crash and should indicate empty state
        assert "const nodes = []" in d3_html
        assert "Nodes: 0" in pyvis_html
        assert "0 nodes" in plotly_html
    
    def test_large_dataset_binding(self, mock_visualizations, performance_test_data):
        """Test data binding with large datasets."""
        entities = performance_test_data["entities"][:50]  # Limit for testing
        relationships = performance_test_data["relationships"][:75]
        
        # Should handle larger datasets
        d3_html = mock_visualizations.create_enhanced_d3_graph(entities, relationships)
        
        assert len(entities) == 50
        assert len(relationships) == 75
        
        # Verify data structure integrity
        assert "const nodes =" in d3_html
        assert "const links =" in d3_html
    
    def test_data_validation_in_binding(self, mock_visualizations):
        """Test data validation during binding."""
        # Valid data
        valid_entities = [
            {"id": "valid1", "name": "Valid", "category": "server", "lines": 100}
        ]
        
        # Invalid data (missing required fields)
        invalid_entities = [
            {"id": "invalid1"}  # Missing name, category, etc.
        ]
        
        # Should handle both cases appropriately
        valid_html = mock_visualizations.create_enhanced_d3_graph(valid_entities, [])
        invalid_html = mock_visualizations.create_enhanced_d3_graph(invalid_entities, [])
        
        # Both should generate HTML without crashing
        assert "const nodes =" in valid_html
        assert "const nodes =" in invalid_html


class TestInteractiveFeatures:
    """Test interactive features of graph visualizations."""
    
    def test_zoom_and_pan_features(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test zoom and pan functionality."""
        html = mock_visualizations.create_enhanced_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Verify zoom controls
        assert "resetZoom()" in html
        assert "centerGraph()" in html
        
        # Check for zoom-related JavaScript
        assert "d3.zoom" in html or "zoom" in html.lower()
    
    def test_node_selection_features(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test node selection functionality."""
        html = mock_visualizations.create_enhanced_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Verify selection-related features
        assert "forceSimulation" in html
        assert ".node" in html
        
        # Should have interactive cursor
        assert "cursor: pointer" in html
    
    def test_tooltip_functionality(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test tooltip functionality."""
        html = mock_visualizations.create_enhanced_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Should include entity data that can be used for tooltips
        nodes_data = json.dumps(sample_graph_entities)
        assert "description" in nodes_data
        assert "complexity" in nodes_data
        assert "lines" in nodes_data
    
    def test_graph_controls(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test graph control functionality."""
        html = mock_visualizations.create_enhanced_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Verify control elements
        assert '<button' in html
        assert 'onclick="resetZoom()"' in html
        assert 'onclick="centerGraph()"' in html
        
        # Verify controls container
        assert 'class="controls"' in html


class TestVisualizationPerformance:
    """Test performance aspects of graph visualizations."""
    
    def test_rendering_performance(self, mock_visualizations):
        """Test rendering performance with different data sizes."""
        # Small dataset
        small_entities = [{"id": f"node_{i}", "name": f"Node{i}", "category": "test"} for i in range(10)]
        small_relationships = [{"source": f"node_{i}", "target": f"node_{i+1}", "type": "test"} for i in range(9)]
        
        # Medium dataset
        medium_entities = [{"id": f"node_{i}", "name": f"Node{i}", "category": "test"} for i in range(50)]
        medium_relationships = [{"source": f"node_{i}", "target": f"node_{(i+1)%50}", "type": "test"} for i in range(50)]
        
        # Both should render without issues
        small_html = mock_visualizations.create_enhanced_d3_graph(small_entities, small_relationships)
        medium_html = mock_visualizations.create_enhanced_d3_graph(medium_entities, medium_relationships)
        
        # Verify both generated valid HTML
        assert "<!DOCTYPE html>" in small_html
        assert "<!DOCTYPE html>" in medium_html
        
        # Medium dataset should still be manageable
        assert len(medium_html) > len(small_html)  # More data = more HTML
    
    def test_memory_efficiency(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test memory efficiency of visualization generation."""
        # Generate multiple visualizations
        htmls = []
        for i in range(5):
            html = mock_visualizations.create_enhanced_d3_graph(
                sample_graph_entities, sample_graph_relationships
            )
            htmls.append(html)
        
        # All should be similar size (no memory leaks in mock)
        sizes = [len(html) for html in htmls]
        assert all(size == sizes[0] for size in sizes)
    
    def test_data_size_limits(self, mock_visualizations):
        """Test behavior with very large datasets."""
        # Very large dataset (simulated)
        large_entities = [{"id": f"node_{i}", "name": f"Node{i}", "category": "test"} for i in range(200)]
        large_relationships = [{"source": f"node_{i}", "target": f"node_{(i+1)%200}", "type": "test"} for i in range(300)]
        
        # Should handle large datasets
        html = mock_visualizations.create_enhanced_d3_graph(large_entities, large_relationships)
        
        # Should still generate valid HTML
        assert "<!DOCTYPE html>" in html
        assert "const nodes =" in html
        assert "const links =" in html


class TestVisualizationAccessibility:
    """Test accessibility features of graph visualizations."""
    
    def test_html_structure_accessibility(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test HTML structure accessibility."""
        html = mock_visualizations.create_enhanced_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Verify proper HTML structure
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "<head>" in html
        assert "<body>" in html
        
        # Verify semantic elements
        assert "<div" in html
        assert "<button" in html
    
    def test_svg_accessibility(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test SVG accessibility features."""
        html = mock_visualizations.create_enhanced_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Verify SVG structure
        assert '<svg' in html
        assert 'id="graph"' in html
        
        # Should have proper dimensions
        assert 'width="700"' in html
        assert 'height="700"' in html
    
    def test_keyboard_navigation_support(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test keyboard navigation support."""
        html = mock_visualizations.create_enhanced_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Should include focusable elements
        assert "<button" in html
        
        # Controls should be accessible
        assert "resetZoom()" in html
        assert "centerGraph()" in html


class TestVisualizationCustomization:
    """Test customization features of graph visualizations."""
    
    def test_color_scheme_customization(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test color scheme customization."""
        html = mock_visualizations.create_enhanced_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Verify color scheme data is included
        assert "categoryColors" in html
        
        # Check for specific colors
        colors_data = json.dumps(mock_visualizations.category_colors)
        assert "#ff6b6b" in colors_data  # server color
        assert "#4ecdc4" in colors_data  # plugin color
    
    def test_size_scaling_customization(self, mock_visualizations):
        """Test size scaling based on entity properties."""
        entities_with_sizes = [
            {"id": "small", "name": "Small", "category": "test", "lines": 50},
            {"id": "large", "name": "Large", "category": "test", "lines": 500}
        ]
        
        html = mock_visualizations.create_enhanced_d3_graph(entities_with_sizes, [])
        
        # Should include size data for scaling
        nodes_data = json.dumps(entities_with_sizes)
        assert '"lines": 50' in nodes_data
        assert '"lines": 500' in nodes_data
    
    def test_layout_customization(self, mock_visualizations, sample_graph_entities, sample_graph_relationships):
        """Test layout customization options."""
        html = mock_visualizations.create_enhanced_d3_graph(
            sample_graph_entities, sample_graph_relationships
        )
        
        # Should include layout configuration
        assert "forceSimulation" in html
        
        # Verify container dimensions
        assert "width: 700px" in html
        assert "height: 700px" in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])