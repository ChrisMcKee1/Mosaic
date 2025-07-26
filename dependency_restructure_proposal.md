# Mosaic Dependency Management Restructure Proposal

## Current Issues

- Multiple overlapping requirements files
- Inconsistent dependency versions across services
- Single pyproject.toml trying to handle all services
- Unclear service dependency boundaries

## Proposed Structure

### Option 1: Service-Specific pyproject.toml (Recommended)

```
Mosaic/
├── pyproject.toml                 # Root project metadata + shared tools config
├── src/
│   ├── mosaic-mcp/
│   │   ├── pyproject.toml        # MCP server dependencies
│   │   ├── requirements.txt      # Pinned versions for deployment
│   │   └── ...
│   ├── mosaic-ingestion/
│   │   ├── pyproject.toml        # Ingestion service dependencies
│   │   ├── requirements.txt      # Pinned versions for deployment
│   │   └── ...
│   └── mosaic-ui/
│       ├── pyproject.toml        # UI service dependencies
│       ├── requirements.txt      # Pinned versions for deployment
│       └── ...
├── shared/
│   └── mosaic-common/
│       └── pyproject.toml        # Shared utilities/models
└── requirements-dev.txt          # Development tools (shared)
```

### Option 2: Workspace with Shared Dependencies

```
Mosaic/
├── pyproject.toml                # Root workspace configuration
├── shared-deps/
│   ├── azure-core.txt            # Shared Azure dependencies
│   ├── ml-core.txt               # Shared ML dependencies
│   └── dev-tools.txt             # Development dependencies
├── src/
│   ├── mosaic-mcp/
│   │   └── requirements.txt      # Service-specific + includes
│   ├── mosaic-ingestion/
│   │   └── requirements.txt      # Service-specific + includes
│   └── mosaic-ui/
│       └── requirements.txt      # Service-specific + includes
└── docker/                       # Container-specific requirements
    ├── mcp-server.requirements.txt
    ├── ingestion.requirements.txt
    └── ui.requirements.txt
```

## Recommended Approach: Option 1

### Benefits:

1. **Clear Service Boundaries**: Each service owns its dependencies
2. **Independent Deployment**: Services can have different dependency versions
3. **Easier Maintenance**: Update dependencies per service
4. **Better CI/CD**: Build/test services independently
5. **Follows Python Standards**: Each service is a proper Python package

### Implementation Plan:

#### 1. Create Service-Specific pyproject.toml Files

**src/mosaic-mcp/pyproject.toml:**

```toml
[project]
name = "mosaic-mcp-server"
dependencies = [
    "fastmcp>=0.1.0",
    "semantic-kernel>=1.0.0",
    "azure-cosmos>=4.5.0",
    "azure-identity>=1.15.0",
    # MCP-specific deps only
]
```

**src/mosaic-ingestion/pyproject.toml:**

```toml
[project]
name = "mosaic-ingestion-service"
dependencies = [
    "GitPython>=3.1.40",
    "tree-sitter>=0.20.0",
    "semantic-kernel>=1.0.0",
    # Ingestion-specific deps only
]
```

**src/mosaic-ui/pyproject.toml:**

```toml
[project]
name = "mosaic-ui-app"
dependencies = [
    "streamlit>=1.28.0",
    "pyvis>=0.3.2",
    "plotly>=5.17.0",
    # UI-specific deps only
]
```

#### 2. Root pyproject.toml for Workspace Management

```toml
[project]
name = "mosaic-workspace"
# Development and tooling configuration only
# No service dependencies

[tool.black]
# Shared formatting rules

[tool.mypy]
# Shared type checking rules
```

#### 3. Shared Dependencies Package (Optional)

```toml
# shared/mosaic-common/pyproject.toml
[project]
name = "mosaic-common"
dependencies = [
    "pydantic>=2.5.0",
    "azure-identity>=1.15.0",
    # Only truly shared utilities
]
```

### Migration Steps:

1. **Audit Current Dependencies** - Map which deps belong to which service
2. **Create Service pyproject.toml** - Move relevant deps to each service
3. **Extract Shared Components** - Create mosaic-common for shared code
4. **Update CI/CD** - Build/test each service independently
5. **Update Documentation** - Service-specific setup instructions

Would you like me to implement this restructure?
