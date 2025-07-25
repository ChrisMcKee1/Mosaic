---
description: Azure integration patterns and best practices
applyTo: "**/azure/**,**/infra/**,**/*azure*"
---

# Azure Development Instructions

## Authentication Pattern

```python
from azure.identity import DefaultAzureCredential
from azure.cosmos import CosmosClient

credential = DefaultAzureCredential()
cosmos_client = CosmosClient(
    account_endpoint,
    credential=credential
)
```

## Resource Naming Conventions

- Use kebab-case for Azure resource names
- Include environment suffix: `-dev`, `-staging`, `-prod`
- Follow pattern: `{service}-{project}-{environment}`
- Example: `mosaic-cosmos-mosaic-dev`

## Bicep Templates

- Use modules for reusable components
- Parameterize all environment-specific values
- Include comprehensive outputs for dependent resources
- Use proper RBAC assignments with least privilege

## Error Handling

- Implement retry logic with exponential backoff
- Use Azure SDK's built-in retry policies
- Log Azure operation results with correlation IDs
- Handle throttling and rate limiting gracefully

## Cost Optimization

- Use appropriate service tiers for each environment
- Implement auto-scaling where applicable
- Monitor resource utilization with alerts
- Use reserved instances for production workloads
