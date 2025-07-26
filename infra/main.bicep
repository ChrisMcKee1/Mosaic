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

// Two-Service Architecture: Query Server + Ingestion Service
module queryServer 'query-server.bicep' = {
  scope: rg
  name: 'query-server'
  params: {
    environmentName: environmentName
    location: location
    containerAppsEnvironmentId: resources.outputs.CONTAINER_APPS_ENVIRONMENT_ID
    azureOpenAIEndpoint: resources.outputs.AZURE_OPENAI_ENDPOINT
    cosmosDbEndpoint: resources.outputs.MOSAIC_COSMOS_ENDPOINT
    redisEndpoint: resources.outputs.AZURE_REDIS_ENDPOINT
    azureMLEndpointUrl: resources.outputs.AZURE_ML_WORKSPACE_NAME
    containerRegistryLoginServer: resources.outputs.AZURE_CONTAINER_REGISTRY_ENDPOINT
    managedIdentityPrincipalId: resources.outputs.AZURE_RESOURCE_MOSAIC_ID
  }
}

module ingestionService 'ingestion-service.bicep' = {
  scope: rg
  name: 'ingestion-service'
  params: {
    environmentName: environmentName
    location: location
    containerAppsEnvironmentId: resources.outputs.CONTAINER_APPS_ENVIRONMENT_ID
    azureOpenAIEndpoint: resources.outputs.AZURE_OPENAI_ENDPOINT
    cosmosDbEndpoint: resources.outputs.MOSAIC_COSMOS_ENDPOINT
    containerRegistryLoginServer: resources.outputs.AZURE_CONTAINER_REGISTRY_ENDPOINT
    managedIdentityPrincipalId: resources.outputs.AZURE_RESOURCE_MOSAIC_ID
  }
}

// Three-Service Architecture: Query Server + Ingestion Service + UI Service
module uiService 'ui-service.bicep' = {
  scope: rg
  name: 'ui-service'
  params: {
    environmentName: environmentName
    location: location
    containerAppsEnvironmentId: resources.outputs.CONTAINER_APPS_ENVIRONMENT_ID
    containerRegistryLoginServer: resources.outputs.AZURE_CONTAINER_REGISTRY_ENDPOINT
    managedIdentityPrincipalId: resources.outputs.AZURE_RESOURCE_MOSAIC_ID
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
output MOSAIC_APP_URL string = queryServer.outputs.queryServerUrl
output OAUTH_SETUP_REQUIRED string = resources.outputs.OAUTH_SETUP_REQUIRED

// Two-Service Architecture Outputs
output QUERY_SERVER_URL string = queryServer.outputs.queryServerUrl
output QUERY_SERVER_PRINCIPAL_ID string = queryServer.outputs.queryServerPrincipalId
output INGESTION_JOB_NAME string = ingestionService.outputs.ingestionJobName
output INGESTION_SCHEDULE_NAME string = ingestionService.outputs.ingestionScheduleName
output INGESTION_JOB_PRINCIPAL_ID string = ingestionService.outputs.ingestionJobPrincipalId

// Three-Service Architecture Outputs  
output UI_SERVICE_URL string = uiService.outputs.uiServiceUrl
output UI_SERVICE_PRINCIPAL_ID string = uiService.outputs.uiServicePrincipalId
