#!/usr/bin/env python3
"""
Create sample graph visualization for Mosaic using manual HTML/JS approach
This works without complex dependencies and shows the visualization concept
"""

import json
import sys
from pathlib import Path

def create_interactive_graph_html():
    """Create an interactive graph visualization using D3.js."""
    
    # Sample data from your Mosaic repository (based on what we've built)
    nodes = [
        {"id": 1, "name": "main.py", "type": "module", "language": "python", "size": 20, "complexity": 15},
        {"id": 2, "name": "GraphVisualizationPlugin", "type": "class", "language": "python", "size": 18, "complexity": 25},
        {"id": 3, "name": "visualize_repository_structure", "type": "function", "language": "python", "size": 12, "complexity": 18},
        {"id": 4, "name": "GraphDataService", "type": "class", "language": "python", "size": 16, "complexity": 22},
        {"id": 5, "name": "get_repository_entities", "type": "function", "language": "python", "size": 14, "complexity": 16},
        {"id": 6, "name": "MosaicMCPServer", "type": "class", "language": "python", "size": 20, "complexity": 28},
        {"id": 7, "name": "LocalIngestionService", "type": "class", "language": "python", "size": 15, "complexity": 20},
        {"id": 8, "name": "ingest_repository", "type": "function", "language": "python", "size": 10, "complexity": 12},
        {"id": 9, "name": "SemanticKernelManager", "type": "class", "language": "python", "size": 14, "complexity": 18},
        {"id": 10, "name": "CLAUDE.md", "type": "documentation", "language": "markdown", "size": 8, "complexity": 2},
    ]
    
    links = [
        {"source": 1, "target": 2, "type": "imports", "strength": 1.0},
        {"source": 1, "target": 6, "type": "imports", "strength": 1.0},
        {"source": 2, "target": 3, "type": "defines", "strength": 1.0},
        {"source": 2, "target": 4, "type": "uses", "strength": 0.8},
        {"source": 4, "target": 5, "type": "defines", "strength": 1.0},
        {"source": 6, "target": 2, "type": "uses", "strength": 0.9},
        {"source": 6, "target": 9, "type": "uses", "strength": 0.7},
        {"source": 7, "target": 8, "type": "defines", "strength": 1.0},
        {"source": 1, "target": 10, "type": "documented_by", "strength": 0.5},
    ]
    
    # Color scheme for different entity types
    colors = {
        "module": "#ff6b6b",
        "class": "#4ecdc4", 
        "function": "#45b7d1",
        "documentation": "#96ceb4"
    }
    
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mosaic Repository Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .subtitle {{
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        #graph {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .node {{
            cursor: pointer;
            stroke: #fff;
            stroke-width: 2px;
        }}
        
        .link {{
            stroke: rgba(255, 255, 255, 0.6);
            stroke-opacity: 0.8;
        }}
        
        .node-label {{
            font: 12px sans-serif;
            fill: white;
            text-anchor: middle;
            pointer-events: none;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }}
        
        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px;
            border-radius: 8px;
            pointer-events: none;
            font-size: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            max-width: 300px;
        }}
        
        .controls {{
            text-align: center;
            margin-bottom: 20px;
        }}
        
        .legend {{
            margin-top: 20px;
            text-align: center;
        }}
        
        .legend-item {{
            display: inline-block;
            margin: 0 15px;
            font-size: 14px;
        }}
        
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 8px;
            vertical-align: middle;
            border: 2px solid white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Mosaic Repository Graph</h1>
        <p class="subtitle">Interactive visualization of code relationships and dependencies</p>
        
        <div class="controls">
            <strong>🖱️ Click and drag nodes • 🔍 Zoom with mouse wheel • 📋 Hover for details</strong>
        </div>
        
        <svg id="graph" width="100%" height="600"></svg>
        
        <div class="legend">
            <div class="legend-item">
                <span class="legend-color" style="background-color: {colors['module']};"></span>
                Modules
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: {colors['class']};"></span>
                Classes
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: {colors['function']};"></span>
                Functions
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: {colors['documentation']};"></span>
                Documentation
            </div>
        </div>
    </div>

    <script>
        // Data
        const nodes = {json.dumps(nodes)};
        const links = {json.dumps(links)};
        const colors = {json.dumps(colors)};
        
        // Set up SVG
        const svg = d3.select("#graph");
        const width = 1160;
        const height = 600;
        svg.attr("width", width).attr("height", height);
        
        // Create tooltip
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);
        
        // Set up force simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => Math.sqrt(d.size) * 2 + 5));
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.3, 3])
            .on("zoom", (event) => {{
                container.attr("transform", event.transform);
            }});
        svg.call(zoom);
        
        // Container for zoomable content
        const container = svg.append("g");
        
        // Create links
        const link = container.append("g")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", d => Math.sqrt(d.strength) * 2);
        
        // Create nodes
        const node = container.append("g")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => Math.sqrt(d.complexity) * 2 + 5)
            .attr("fill", d => colors[d.type] || "#999")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("mouseover", function(event, d) {{
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(`
                    <strong>${{d.name}}</strong><br/>
                    Type: ${{d.type}}<br/>
                    Language: ${{d.language}}<br/>
                    Complexity: ${{d.complexity}}<br/>
                    Size: ${{d.size}} lines
                `)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }})
            .on("mouseout", function(d) {{
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            }});
        
        // Add labels
        const labels = container.append("g")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.name.length > 15 ? d.name.substring(0, 15) + "..." : d.name);
        
        // Update positions
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            labels
                .attr("x", d => d.x)
                .attr("y", d => d.y + 5);
        }});
        
        // Drag functions
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
    
    return html_template

def main():
    """Create and save the sample visualization."""
    print("🎨 Creating Mosaic Repository Graph Visualization...")
    
    try:
        # Generate HTML
        html_content = create_interactive_graph_html()
        
        # Save to file
        output_file = "mosaic_repository_graph.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"✅ Graph visualization saved to: {output_file}")
        print(f"🌐 Open {output_file} in your browser to view the interactive graph!")
        
        print("\n" + "="*60)
        print("🌟 INTERACTIVE FEATURES:")
        print("="*60)
        print("• 🖱️  Click and drag any node to reposition it")
        print("• 🔍 Use mouse wheel to zoom in and out") 
        print("• 📋 Hover over nodes to see detailed information")
        print("• 🎨 Nodes are color-coded by type (module, class, function)")
        print("• 📏 Node size represents code complexity")
        print("• 🔗 Lines show relationships (imports, defines, uses)")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)