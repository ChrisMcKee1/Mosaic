targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the environment that can be used as part of naming resource convention')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
param location string


param mosaicExists bool

@description('Id of the user or app to assign application roles')
param principalId string

@description('Principal type of user or app')
param principalType string

// Tags that should be applied to all resources.
// 
// Note that 'azd-service-name' tags should be applied separately to service host resources.
// Example usage:
//   tags: union(tags, { 'azd-service-name': <service name in azure.yaml> })
var tags = {
  'azd-env-name': environmentName
}

// Organize resources in a resource group
resource rg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: 'rg-${environmentName}'
  location: location
  tags: tags
}

module resources 'resources.bicep' = {
  scope: rg
  name: 'resources'
  params: {
    location: location
    tags: tags
    principalId: principalId
    principalType: principalType
    mosaicExists: mosaicExists
  }
}
// Core Infrastructure Outputs
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = resources.outputs.AZURE_CONTAINER_REGISTRY_ENDPOINT
output AZURE_RESOURCE_MOSAIC_ID string = resources.outputs.AZURE_RESOURCE_MOSAIC_ID
output AZURE_CLIENT_ID string = resources.outputs.AZURE_CLIENT_ID
output AZURE_USE_MANAGED_IDENTITY string = resources.outputs.AZURE_USE_MANAGED_IDENTITY

// OmniRAG Pattern Outputs (Microsoft CosmosAIGraph-based)
output MOSAIC_COSMOS_ENDPOINT string = resources.outputs.MOSAIC_COSMOS_ENDPOINT
output MOSAIC_DATABASE_NAME string = resources.outputs.MOSAIC_DATABASE_NAME
output MOSAIC_LIBRARIES_CONTAINER string = resources.outputs.MOSAIC_LIBRARIES_CONTAINER
output MOSAIC_MEMORIES_CONTAINER string = resources.outputs.MOSAIC_MEMORIES_CONTAINER
output MOSAIC_DOCUMENTS_CONTAINER string = resources.outputs.MOSAIC_DOCUMENTS_CONTAINER
output MOSAIC_CONFIG_CONTAINER string = resources.outputs.MOSAIC_CONFIG_CONTAINER

// Service Endpoints
output AZURE_REDIS_ENDPOINT string = resources.outputs.AZURE_REDIS_ENDPOINT
output AZURE_OPENAI_ENDPOINT string = resources.outputs.AZURE_OPENAI_ENDPOINT
output AZURE_ML_WORKSPACE_NAME string = resources.outputs.AZURE_ML_WORKSPACE_NAME
output AZURE_FUNCTIONS_ENDPOINT string = resources.outputs.AZURE_FUNCTIONS_ENDPOINT

// OAuth 2.1 Configuration Outputs (FR-14)
output AZURE_TENANT_ID string = resources.outputs.AZURE_TENANT_ID
output MOSAIC_APP_URL string = resources.outputs.MOSAIC_APP_URL
output OAUTH_SETUP_REQUIRED string = resources.outputs.OAUTH_SETUP_REQUIRED
