"""
Mosaic Ingestion Service

Standalone service for heavy-duty repository ingestion and knowledge graph population.
Runs as Azure Container App Job (scheduled/manual execution) separate from real-time Query Server.

Architectural Separation:
- Ingestion Service: Offline, resource-intensive repository processing
- Query Server: Real-time, lightweight MCP requests
"""
