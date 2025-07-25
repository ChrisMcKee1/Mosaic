# 🎨 Mosaic Graph Visualization

Interactive graph visualization system for exploring code relationships and knowledge graphs in Mosaic.

## 🌟 Features

### **Interactive Graph Types**
- **📊 Repository Structure**: Visualize code entities (functions, classes, modules) with relationships
- **🔗 Dependency Graphs**: Explore import dependencies and external libraries
- **🧠 Knowledge Graphs**: Semantic relationships with similarity scoring
- **📈 Custom Graphs**: Create visualizations from custom data

### **Visualization Capabilities**
- **🎨 Color Coding**: Nodes colored by language, entity type, or functional cluster
- **📏 Dynamic Sizing**: Node sizes based on complexity, importance, or usage frequency
- **🖱️ Interactive Controls**: Click, drag, zoom, pan, hover for details
- **⚡ Real-time Rendering**: Fast WebGL-powered 3D visualization with Neo4j-viz
- **📱 Responsive Design**: Works on desktop and mobile browsers

## 🚀 Quick Start

### **Installation**

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install graph visualization dependencies
pip install neo4j-viz pandas networkx

# 3. Install additional dependencies for full functionality
pip install azure-cosmos azure-identity
```

### **Basic Usage**

```python
from mosaic.plugins.graph_visualization import GraphVisualizationPlugin

# Initialize plugin
settings = MosaicSettings()
viz_plugin = GraphVisualizationPlugin(settings)
await viz_plugin.initialize()

# Create repository visualization
html_output = await viz_plugin.visualize_repository_structure(
    repository_url="https://github.com/user/repo",
    include_functions=True,
    include_classes=True,
    color_by_language=True,
    size_by_complexity=True
)

# Save to file
with open("repository_graph.html", "w") as f:
    f.write(html_output)
```

## 🛠️ MCP Integration

The graph visualization system integrates seamlessly with Mosaic's MCP server:

### **Available MCP Tools**

```python
# 1. Repository Structure Visualization
await mosaic_server.visualize_repository_structure(
    repository_url="https://github.com/python/cpython",
    include_functions=True,
    color_by_language=True
)

# 2. Dependency Graph Visualization
await mosaic_server.visualize_code_dependencies(
    repository_url="https://github.com/python/cpython",
    show_external_deps=True,
    dependency_types=["imports", "calls"]
)

# 3. Knowledge Graph Visualization  
await mosaic_server.visualize_knowledge_graph(
    repository_url="https://github.com/python/cpython",
    include_semantic_similarity=True,
    max_nodes=200
)

# 4. Repository Statistics
stats = await mosaic_server.get_repository_graph_stats(
    repository_url="https://github.com/python/cpython"
)
```

### **MCP Protocol Usage**

```bash
# Call via MCP protocol
curl -X POST http://127.0.0.1:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "method": "visualize_repository_structure",
    "params": {
      "repository_url": "https://github.com/user/repo",
      "include_functions": true,
      "color_by_language": true
    }
  }'
```

## 📊 Visualization Examples

### **1. Code Repository Structure**
```python
# Creates interactive graph showing:
# - Modules, classes, functions as nodes
# - Import/call/inheritance relationships as edges
# - Color-coded by programming language
# - Sized by code complexity
```

### **2. Dependency Analysis**
```python
# Shows:
# - Internal and external dependencies
# - Usage frequency as node sizes
# - Import chains and circular dependencies
# - External library usage patterns
```

### **3. Semantic Knowledge Graph**
```python
# Displays:
# - Semantic similarity between code entities
# - Functional clustering
# - Importance scoring
# - Cross-repository relationships
```

## 🎯 Integration with Cosmos DB

The system connects to your existing Cosmos DB OmniRAG backend:

```python
# GraphDataService queries:
# - code_entities container for nodes
# - code_relationships container for edges
# - repositories container for metadata
# - libraries container for dependencies

# Supports complex queries with filters:
entities = await data_service.get_repository_entities(
    repository_url="https://github.com/user/repo",
    include_functions=True,
    include_classes=True,
    limit=500
)
```

## 🎨 Customization Options

### **Styling**
```python
# Color nodes by different properties
VG.color_nodes(property="language")  # By programming language
VG.color_nodes(property="entity_type")  # By code entity type
VG.color_nodes(property="functional_cluster")  # By semantic cluster

# Size nodes by metrics
VG.resize_nodes(property="complexity", node_radius_min_max=(5, 30))
VG.resize_nodes(property="importance_score", node_radius_min_max=(8, 25))
```

### **Layout Algorithms**
- **Force-directed**: Natural clustering of related nodes
- **Hierarchical**: Tree-like structures for dependencies
- **Circular**: Arranged in circles by categories
- **Grid**: Structured layout for large datasets

## 🔧 Advanced Features

### **Performance Optimization**
- **Node Filtering**: Limit by entity type, complexity, or importance
- **Relationship Pruning**: Show only high-confidence relationships
- **Progressive Loading**: Load additional nodes on demand
- **Clustering**: Group similar nodes to reduce visual complexity

### **Export Options**
```python
# Export formats
VG.export_json()  # For programmatic access
VG.export_png()   # High-resolution images
VG.export_svg()   # Scalable vector graphics
VG.render()       # Interactive HTML
```

## 📋 Troubleshooting

### **Common Issues**

1. **No data displayed**: Repository not ingested yet
   ```bash
   # Run ingestion first
   python3 -m src.ingestion_service.local_main --repository-url <url> --branch main
   ```

2. **Performance issues**: Too many nodes
   ```python
   # Reduce node count or use filtering
   max_nodes=100, include_functions=False
   ```

3. **Missing dependencies**: Install requirements
   ```bash
   pip install neo4j-viz pandas networkx azure-cosmos
   ```

### **Debugging**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check repository statistics
stats = await viz_plugin.get_repository_stats(repository_url)
print(stats)
```

## 🎯 Use Cases

### **1. Code Architecture Analysis**
- **Explore module dependencies** in large codebases
- **Identify tightly coupled components** needing refactoring
- **Visualize inheritance hierarchies** and class relationships
- **Track function call patterns** and hot paths

### **2. Knowledge Discovery**
- **Find semantically similar code** across repositories
- **Discover related functionality** through graph traversal
- **Identify expertise areas** based on contribution patterns
- **Explore cross-project dependencies** and shared libraries

### **3. Development Insights**
- **Onboard new developers** with visual code maps
- **Document architecture decisions** with interactive diagrams
- **Plan refactoring efforts** using dependency analysis
- **Monitor code evolution** through temporal graph changes

## 🌐 Browser Compatibility

- ✅ **Chrome/Chromium**: Full WebGL support
- ✅ **Firefox**: Full functionality
- ✅ **Safari**: Core features supported
- ✅ **Edge**: Full compatibility
- ⚠️ **Mobile**: Limited interaction on small screens

## 📈 Performance Guidelines

| Repository Size | Recommended Settings | Expected Performance |
|-----------------|---------------------|---------------------|
| Small (<100 files) | All features enabled | Instant rendering |
| Medium (100-1000 files) | Limit functions, max_nodes=300 | <2s load time |
| Large (1000+ files) | Modules only, max_nodes=150 | <5s load time |
| Enterprise | Custom filtering required | Varies |

## 🤝 Contributing

The graph visualization system is designed for extensibility:

```python
# Add custom visualization types
class CustomGraphPlugin(GraphVisualizationPlugin):
    @kernel_function(name="my_custom_visualization")
    async def create_custom_viz(self, data: str) -> str:
        # Custom visualization logic
        pass
```

---

**Next Steps**: Open any generated HTML file in your browser to explore the interactive visualizations! 🎉