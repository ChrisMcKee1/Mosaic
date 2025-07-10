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
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = resources.outputs.AZURE_CONTAINER_REGISTRY_ENDPOINT
output AZURE_RESOURCE_MOSAIC_ID string = resources.outputs.AZURE_RESOURCE_MOSAIC_ID
output AZURE_AI_SEARCH_ENDPOINT string = resources.outputs.AZURE_AI_SEARCH_ENDPOINT
output AZURE_COSMOS_ENDPOINT string = resources.outputs.AZURE_COSMOS_ENDPOINT
output AZURE_REDIS_ENDPOINT string = resources.outputs.AZURE_REDIS_ENDPOINT
output AZURE_CLIENT_ID string = resources.outputs.AZURE_CLIENT_ID
output AZURE_USE_MANAGED_IDENTITY string = resources.outputs.AZURE_USE_MANAGED_IDENTITY
output AZURE_OPENAI_ENDPOINT string = resources.outputs.AZURE_OPENAI_ENDPOINT
output AZURE_ML_WORKSPACE_NAME string = resources.outputs.AZURE_ML_WORKSPACE_NAME
output AZURE_FUNCTIONS_ENDPOINT string = resources.outputs.AZURE_FUNCTIONS_ENDPOINT
