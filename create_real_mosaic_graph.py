#!/usr/bin/env python3
"""
Create graph visualization using real data from the Mosaic repository
This demonstrates what the actual GraphVisualizationPlugin would generate
"""

import json
import sys
from pathlib import Path
import os
import re

def analyze_repository():
    """Analyze the actual Mosaic repository structure."""
    
    src_path = Path("src")
    nodes = []
    links = []
    node_id = 1
    file_to_id = {}
    
    print("🔍 Analyzing Mosaic repository structure...")
    
    # Analyze Python files
    for py_file in src_path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
            
        relative_path = py_file.relative_to(src_path)
        file_name = py_file.name
        
        # Count lines and estimate complexity
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            lines = len([line for line in content.split('\n') if line.strip()])
            
            # Simple complexity estimation
            complexity = len(re.findall(r'\bdef\s+\w+', content)) * 3
            complexity += len(re.findall(r'\bclass\s+\w+', content)) * 5
            complexity += len(re.findall(r'\bif\s+', content))
            complexity += len(re.findall(r'\bfor\s+', content))
            complexity += len(re.findall(r'\bwhile\s+', content))
            
            # Determine entity type
            if "main.py" in file_name:
                entity_type = "entry_point"
            elif "test_" in file_name or file_name.startswith("test"):
                entity_type = "test"
            elif "plugin" in str(relative_path).lower():
                entity_type = "plugin"
            elif "server" in str(relative_path).lower():
                entity_type = "server"
            elif "config" in str(relative_path).lower():
                entity_type = "config"
            else:
                entity_type = "module"
            
            nodes.append({
                "id": node_id,
                "name": file_name.replace(".py", ""),
                "full_path": str(relative_path),
                "type": entity_type,
                "language": "python",
                "size": min(lines, 100),  # Cap for visualization
                "complexity": min(complexity, 50),  # Cap for visualization
                "lines": lines
            })
            
            file_to_id[str(relative_path)] = node_id
            node_id += 1
            
        except Exception as e:
            print(f"⚠️  Warning: Could not analyze {py_file}: {e}")
            continue
    
    # Add documentation files
    for doc_file in ["CLAUDE.md", "README.md", "GRAPH_VISUALIZATION_README.md"]:
        if Path(doc_file).exists():
            try:
                content = Path(doc_file).read_text(encoding="utf-8", errors="ignore")
                lines = len(content.split('\n'))
                
                nodes.append({
                    "id": node_id,
                    "name": doc_file,
                    "full_path": doc_file,
                    "type": "documentation",
                    "language": "markdown",
                    "size": min(lines // 10, 30),  # Scale down for viz
                    "complexity": 1,
                    "lines": lines
                })
                
                file_to_id[doc_file] = node_id
                node_id += 1
                
            except Exception as e:
                print(f"⚠️  Warning: Could not read {doc_file}: {e}")
    
    # Analyze imports (simplified)
    link_id = 1
    for py_file in src_path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
            
        relative_path = py_file.relative_to(src_path)
        source_id = file_to_id.get(str(relative_path))
        
        if not source_id:
            continue
            
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            
            # Find local imports
            local_imports = re.findall(r'from\s+\.(\w+)\s+import', content)
            local_imports.extend(re.findall(r'from\s+\.\.(\w+)\s+import', content))
            
            for imported_module in local_imports:
                # Find corresponding file
                for target_path, target_id in file_to_id.items():
                    if imported_module in target_path and target_id != source_id:
                        links.append({
                            "source": source_id,
                            "target": target_id,
                            "type": "imports",
                            "strength": 0.8
                        })
                        break
                        
        except Exception as e:
            continue
    
    print(f"📊 Found {len(nodes)} files and {len(links)} relationships")
    return nodes, links

def create_enhanced_graph_html(nodes, links):
    """Create an enhanced interactive graph with real Mosaic data."""
    
    # Enhanced color scheme
    colors = {
        "entry_point": "#ff4757",     # Red for main entry points
        "server": "#2ed573",          # Green for server components  
        "plugin": "#ffa502",          # Orange for plugins
        "config": "#3742fa",          # Blue for configuration
        "test": "#a4b0be",           # Gray for tests
        "module": "#5f27cd",         # Purple for regular modules
        "documentation": "#00d2d3"    # Cyan for docs
    }
    
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mosaic Repository - Real Code Structure</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 50%, #9b59b6 100%);
            color: white;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            font-size: 3em;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
            background: linear-gradient(45deg, #f39c12, #e74c3c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .subtitle {{
            text-align: center;
            margin-bottom: 20px;
            font-size: 1.3em;
            opacity: 0.9;
            font-weight: 300;
        }}
        
        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 30px;
        }}
        
        .stat-item {{
            background: rgba(255, 255, 255, 0.1);
            padding: 15px 25px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            display: block;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        #graph {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3);
        }}
        
        .node {{
            cursor: pointer;
            stroke: #fff;
            stroke-width: 2px;
            transition: all 0.3s ease;
        }}
        
        .node:hover {{
            stroke-width: 4px;
            filter: brightness(1.3);
        }}
        
        .link {{
            stroke: rgba(255, 255, 255, 0.4);
            stroke-opacity: 0.6;
            transition: all 0.3s ease;
        }}
        
        .link:hover {{
            stroke-opacity: 1;
            stroke-width: 3px;
        }}
        
        .node-label {{
            font: 11px 'Segoe UI', sans-serif;
            fill: white;
            text-anchor: middle;
            pointer-events: none;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
            font-weight: 500;
        }}
        
        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.95);
            color: white;
            padding: 15px;
            border-radius: 10px;
            pointer-events: none;
            font-size: 13px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.5);
            max-width: 320px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        .controls {{
            text-align: center;
            margin-bottom: 25px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }}
        
        .legend {{
            margin-top: 25px;
            text-align: center;
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }}
        
        .legend-title {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
        }}
        
        .legend-items {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 20px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 13px;
            font-weight: 500;
        }}
        
        .legend-color {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            margin-right: 8px;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
        
        @media (max-width: 768px) {{
            .stats {{
                flex-direction: column;
                align-items: center;
            }}
            
            .legend-items {{
                flex-direction: column;
                align-items: center;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Mosaic Repository</h1>
        <p class="subtitle">Interactive visualization of real code structure and relationships</p>
        
        <div class="stats">
            <div class="stat-item">
                <span class="stat-number">{len(nodes)}</span>
                <span class="stat-label">Files Analyzed</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">{len(links)}</span>
                <span class="stat-label">Relationships</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">{sum(node.get('lines', 0) for node in nodes)}</span>
                <span class="stat-label">Total Lines</span>
            </div>
        </div>
        
        <div class="controls">
            <strong>🖱️ Click and drag nodes • 🔍 Zoom with mouse wheel • 📋 Hover for details • 🎯 Double-click to focus</strong>
        </div>
        
        <svg id="graph" width="100%" height="700"></svg>
        
        <div class="legend">
            <div class="legend-title">🎨 Component Types</div>
            <div class="legend-items">
                <div class="legend-item">
                    <span class="legend-color" style="background-color: {colors['entry_point']};"></span>
                    Entry Points
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: {colors['server']};"></span>
                    Server Components
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: {colors['plugin']};"></span>
                    Plugins
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: {colors['config']};"></span>
                    Configuration
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: {colors['test']};"></span>
                    Tests
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: {colors['module']};"></span>
                    Modules
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: {colors['documentation']};"></span>
                    Documentation
                </div>
            </div>
        </div>
    </div>

    <script>
        // Real repository data
        const nodes = {json.dumps(nodes)};
        const links = {json.dumps(links)};
        const colors = {json.dumps(colors)};
        
        console.log('Loaded repository data:', {{
            nodes: nodes.length,
            links: links.length
        }});
        
        // Set up SVG
        const svg = d3.select("#graph");
        const width = 1360;
        const height = 700;
        svg.attr("width", width).attr("height", height);
        
        // Create tooltip
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);
        
        // Set up force simulation with enhanced forces
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(80).strength(0.5))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => Math.sqrt(d.size) * 2 + 8))
            .force("x", d3.forceX(width / 2).strength(0.05))
            .force("y", d3.forceY(height / 2).strength(0.05));
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.2, 4])
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
            .attr("stroke-width", d => Math.sqrt(d.strength) * 2 + 1);
        
        // Create nodes
        const node = container.append("g")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => Math.sqrt(d.complexity + d.size/5) * 1.5 + 6)
            .attr("fill", d => colors[d.type] || "#95a5a6")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("mouseover", function(event, d) {{
                d3.select(this).transition().duration(200).attr("r", d => Math.sqrt(d.complexity + d.size/5) * 1.8 + 8);
                
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .95);
                tooltip.html(`
                    <div style="border-bottom: 1px solid #444; padding-bottom: 8px; margin-bottom: 8px;">
                        <strong style="color: ${{colors[d.type]}};">${{d.name}}</strong>
                    </div>
                    <div><strong>Path:</strong> ${{d.full_path}}</div>
                    <div><strong>Type:</strong> ${{d.type}}</div>
                    <div><strong>Language:</strong> ${{d.language}}</div>
                    <div><strong>Lines:</strong> ${{d.lines}}</div>
                    <div><strong>Complexity:</strong> ${{d.complexity}}</div>
                `)
                    .style("left", (event.pageX + 15) + "px")
                    .style("top", (event.pageY - 10) + "px");
            }})
            .on("mouseout", function(event, d) {{
                d3.select(this).transition().duration(200).attr("r", d => Math.sqrt(d.complexity + d.size/5) * 1.5 + 6);
                
                tooltip.transition()
                    .duration(300)
                    .style("opacity", 0);
            }})
            .on("dblclick", function(event, d) {{
                // Focus on this node
                const transform = d3.zoomTransform(svg.node());
                const x = -d.x * 2 + width / 2;
                const y = -d.y * 2 + height / 2;
                svg.transition().duration(750).call(zoom.transform, d3.zoomIdentity.translate(x, y).scale(2));
            }});
        
        // Add labels
        const labels = container.append("g")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.name.length > 20 ? d.name.substring(0, 17) + "..." : d.name);
        
        // Update positions on simulation tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => Math.max(20, Math.min(width - 20, d.x)))
                .attr("cy", d => Math.max(20, Math.min(height - 20, d.y)));
            
            labels
                .attr("x", d => Math.max(20, Math.min(width - 20, d.x)))
                .attr("y", d => Math.max(20, Math.min(height - 20, d.y)) + 4);
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
    """Create visualization with real Mosaic repository data."""
    print("🚀 Creating Real Mosaic Repository Graph Visualization...")
    print("=" * 60)
    
    try:
        # Analyze the actual repository
        nodes, links = analyze_repository()
        
        if not nodes:
            print("❌ No Python files found in src/ directory")
            return False
        
        # Generate enhanced HTML visualization
        html_content = create_enhanced_graph_html(nodes, links)
        
        # Save to file
        output_file = "mosaic_real_repository_graph.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"✅ Real repository graph saved to: {output_file}")
        print(f"🌐 Open {output_file} in your browser!")
        
        # Show analysis results
        print("\n📊 REPOSITORY ANALYSIS RESULTS:")
        print("=" * 60)
        
        # Count by type
        type_counts = {}
        total_lines = 0
        total_complexity = 0
        
        for node in nodes:
            node_type = node['type']
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
            total_lines += node['lines']
            total_complexity += node['complexity']
        
        for node_type, count in sorted(type_counts.items()):
            print(f"• {node_type.title()}: {count} files")
        
        print(f"• Total Lines of Code: {total_lines:,}")
        print(f"• Total Complexity Score: {total_complexity}")
        print(f"• Import Relationships: {len(links)}")
        
        print("\n🌟 ENHANCED INTERACTIVE FEATURES:")
        print("=" * 60)
        print("• 🖱️  Click and drag nodes to explore relationships")
        print("• 🔍 Zoom with mouse wheel (0.2x to 4x)")
        print("• 📋 Hover over nodes for detailed file information")  
        print("• 🎯 Double-click any node to focus and zoom in")
        print("• 🎨 Color-coded by component type (7 different types)")
        print("• 📏 Node size represents complexity + file size")
        print("• 🔗 Lines show import relationships between files")
        print("• 📊 Statistics panel shows repository metrics")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating real repository visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)