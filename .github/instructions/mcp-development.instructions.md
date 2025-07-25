---
description: Guidelines for implementing MCP tools and servers
applyTo: "**/mcp/**,**/plugins/**,**/*mcp*"
---

# MCP Development Instructions

## MCP Tool Implementation

- Use the latest MCP Python SDK (`mcp` package)
- Implement proper JSON-RPC 2.0 request/response handling
- Follow the MCP specification for tool discovery and metadata
- Use structured error responses with proper error codes
- Implement async patterns for all I/O operations

## Tool Registration Pattern

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("mosaic-mcp-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="tool_name",
            description="Clear description of what the tool does",
            inputSchema={
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Parameter description"}
                },
                "required": ["param"]
            }
        )
    ]
```

## Error Handling

- Use proper MCP error codes (InvalidRequest, MethodNotFound, etc.)
- Include contextual error messages
- Log errors with structured data for debugging
- Implement graceful degradation for optional features

## Plugin Architecture

- Use the Semantic Kernel plugin pattern for AI integrations
- Implement proper dependency injection for Azure services
- Create reusable base classes for common MCP operations
- Follow the Factory pattern for plugin instantiation
