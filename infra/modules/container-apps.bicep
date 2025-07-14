// Mosaic MCP Server - Container Apps Module  
// Based on CosmosAIGraph deployment patterns

@description('The location used for all deployed resources')
param location string = resourceGroup().location

@description('Tags that will be applied to all resources')
param tags object = {}

@description('Unique resource token for naming')
param resourceToken string

@description('Abbreviations for resource naming')
param abbrs object

@description('Container registry login server')
param containerRegistryLoginServer string

@description('Managed identity resource ID')
param managedIdentityResourceId string

@description('Managed identity principal ID')
param managedIdentityPrincipalId string

@description('Managed identity client ID')
param managedIdentityClientId string

@description('Log Analytics workspace resource ID')
param logAnalyticsWorkspaceResourceId string

@description('Application Insights connection string')
param applicationInsightsConnectionString string

// OmniRAG Environment Variables
@description('Cosmos DB endpoint from OmniRAG module')
param cosmosEndpoint string

@description('Redis endpoint')
param redisEndpoint string

@description('OpenAI endpoint')
param openAIEndpoint string

@description('ML workspace name')
param mlWorkspaceName string

@description('Functions endpoint')
param functionsEndpoint string

@description('Azure tenant ID')
param tenantId string

@description('Whether container already exists')
param mosaicExists bool

// Container Apps Environment
module containerAppsEnvironment 'br/public:avm/res/app/managed-environment:0.4.5' = {
  name: 'container-apps-environment'
  params: {
    logAnalyticsWorkspaceResourceId: logAnalyticsWorkspaceResourceId
    name: '${abbrs.appManagedEnvironments}${resourceToken}'
    location: location
    zoneRedundant: false
    tags: union(tags, { 'component': 'container-environment' })
  }
}

// Fetch container image helper
module mosaicFetchLatestImage '../modules/fetch-container-image.bicep' = {
  name: 'mosaic-fetch-image'
  params: {
    exists: mosaicExists
    name: 'mosaic'
  }
}

// Mosaic MCP Server Container App
// Following CosmosAIGraph scaling and configuration patterns
module mosaic 'br/public:avm/res/app/container-app:0.8.0' = {
  name: 'mosaic-mcp-server'
  params: {
    name: 'mosaic'
    ingressTargetPort: 80
    // CosmosAIGraph scaling pattern: consistent min/max for reliability
    scaleMinReplicas: 1
    scaleMaxReplicas: 2
    secrets: {
      secureList: []
    }
    containers: [
      {
        image: mosaicFetchLatestImage.outputs.?containers[?0].?image ?? 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
        name: 'main'
        resources: {
          cpu: json('0.5')
          memory: '1.0Gi'
        }
        // Environment variables following CosmosAIGraph CAIG_ pattern
        env: [
          {
            name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
            value: applicationInsightsConnectionString
          }
          {
            name: 'AZURE_CLIENT_ID'
            value: managedIdentityClientId
          }
          // OmniRAG Configuration
          {
            name: 'MOSAIC_COSMOS_ENDPOINT'
            value: cosmosEndpoint
          }
          {
            name: 'MOSAIC_DATABASE_NAME'
            value: 'mosaic-omnirag'
          }
          {
            name: 'MOSAIC_LIBRARIES_CONTAINER'
            value: 'libraries'
          }
          {
            name: 'MOSAIC_MEMORIES_CONTAINER'
            value: 'memories'
          }
          {
            name: 'MOSAIC_DOCUMENTS_CONTAINER'
            value: 'documents'
          }
          {
            name: 'MOSAIC_CONFIG_CONTAINER'
            value: 'config'
          }
          // Additional service endpoints
          {
            name: 'AZURE_REDIS_ENDPOINT'
            value: redisEndpoint
          }
          {
            name: 'AZURE_OPENAI_ENDPOINT'
            value: openAIEndpoint
          }
          {
            name: 'AZURE_ML_WORKSPACE_NAME'
            value: mlWorkspaceName
          }
          {
            name: 'AZURE_FUNCTIONS_ENDPOINT'
            value: functionsEndpoint
          }
          // Managed Identity Configuration
          {
            name: 'AZURE_USE_MANAGED_IDENTITY'
            value: 'true'
          }
          {
            name: 'PORT'
            value: '80'
          }
          // OAuth 2.1 Configuration (FR-14)
          {
            name: 'AZURE_TENANT_ID'
            value: tenantId
          }
          {
            name: 'MCP_OAUTH_ENABLED'
            value: 'true'
          }
          {
            name: 'MCP_OAUTH_PROVIDER'
            value: 'entra_id'
          }
          // OmniRAG Pattern Flag
          {
            name: 'MOSAIC_OMNIRAG_ENABLED'
            value: 'true'
          }
        ]
      }
    ]
    managedIdentities: {
      systemAssigned: false
      userAssignedResourceIds: [managedIdentityResourceId]
    }
    registries: [
      {
        server: containerRegistryLoginServer
        identity: managedIdentityResourceId
      }
    ]
    environmentResourceId: containerAppsEnvironment.outputs.resourceId
    location: location
    tags: union(tags, { 
      'azd-service-name': 'mosaic'
      'omnirag-component': 'mcp-server'
    })
  }
}

// Outputs
output containerAppsEnvironmentResourceId string = containerAppsEnvironment.outputs.resourceId
output mosaicAppResourceId string = mosaic.outputs.resourceId
output mosaicAppFqdn string = mosaic.outputs.fqdn
output mosaicAppUrl string = 'https://${mosaic.outputs.fqdn}'