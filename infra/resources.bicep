@description('The location used for all deployed resources')
param location string = resourceGroup().location

@description('Tags that will be applied to all resources')
param tags object = {}


param mosaicExists bool

@description('Id of the user or app to assign application roles')
param principalId string

@description('Principal type of user or app')
param principalType string

var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = uniqueString(subscription().id, resourceGroup().id, location)

// Monitor application with Azure Monitor
module monitoring 'br/public:avm/ptn/azd/monitoring:0.1.0' = {
  name: 'monitoring'
  params: {
    logAnalyticsName: '${abbrs.operationalInsightsWorkspaces}${resourceToken}'
    applicationInsightsName: '${abbrs.insightsComponents}${resourceToken}'
    applicationInsightsDashboardName: '${abbrs.portalDashboards}${resourceToken}'
    location: location
    tags: tags
  }
}
// Container registry
module containerRegistry 'br/public:avm/res/container-registry/registry:0.1.1' = {
  name: 'registry'
  params: {
    name: '${abbrs.containerRegistryRegistries}${resourceToken}'
    location: location
    tags: tags
    publicNetworkAccess: 'Enabled'
    roleAssignments:[
      {
        principalId: mosaicIdentity.outputs.principalId
        principalType: 'ServicePrincipal'
        roleDefinitionIdOrName: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
      }
    ]
  }
}

// Container apps environment
module containerAppsEnvironment 'br/public:avm/res/app/managed-environment:0.4.5' = {
  name: 'container-apps-environment'
  params: {
    logAnalyticsWorkspaceResourceId: monitoring.outputs.logAnalyticsWorkspaceResourceId
    name: '${abbrs.appManagedEnvironments}${resourceToken}'
    location: location
    zoneRedundant: false
  }
}

module mosaicIdentity 'br/public:avm/res/managed-identity/user-assigned-identity:0.2.1' = {
  name: 'mosaicidentity'
  params: {
    name: '${abbrs.managedIdentityUserAssignedIdentities}mosaic-${resourceToken}'
    location: location
  }
}
module mosaicFetchLatestImage './modules/fetch-container-image.bicep' = {
  name: 'mosaic-fetch-image'
  params: {
    exists: mosaicExists
    name: 'mosaic'
  }
}

module mosaic 'br/public:avm/res/app/container-app:0.8.0' = {
  name: 'mosaic'
  params: {
    name: 'mosaic'
    ingressTargetPort: 80
    scaleMinReplicas: 1
    scaleMaxReplicas: 10
    secrets: {
      secureList:  [
      ]
    }
    containers: [
      {
        image: mosaicFetchLatestImage.outputs.?containers[?0].?image ?? 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
        name: 'main'
        resources: {
          cpu: json('0.5')
          memory: '1.0Gi'
        }
        env: [
          {
            name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
            value: monitoring.outputs.applicationInsightsConnectionString
          }
          {
            name: 'AZURE_CLIENT_ID'
            value: mosaicIdentity.outputs.clientId
          }
          {
            name: 'AZURE_AI_SEARCH_ENDPOINT'
            value: 'https://${aiSearch.outputs.name}.search.windows.net'
          }
          {
            name: 'AZURE_COSMOS_ENDPOINT'
            value: cosmosAccount.outputs.endpoint
          }
          {
            name: 'AZURE_REDIS_ENDPOINT'
            value: '${redisCache.outputs.hostName}:6380'
          }
          {
            name: 'AZURE_OPENAI_ENDPOINT'
            value: openAI.outputs.endpoint
          }
          {
            name: 'AZURE_USE_MANAGED_IDENTITY'
            value: 'true'
          }
          {
            name: 'PORT'
            value: '80'
          }
        ]
      }
    ]
    managedIdentities:{
      systemAssigned: false
      userAssignedResourceIds: [mosaicIdentity.outputs.resourceId]
    }
    registries:[
      {
        server: containerRegistry.outputs.loginServer
        identity: mosaicIdentity.outputs.resourceId
      }
    ]
    environmentResourceId: containerAppsEnvironment.outputs.resourceId
    location: location
    tags: union(tags, { 'azd-service-name': 'mosaic' })
  }
}
// Azure AI Search (Free F tier) - AVM with Managed Identity
module aiSearch 'br/public:avm/res/search/search-service:0.7.1' = {
  name: 'ai-search'
  params: {
    name: '${abbrs.searchSearchServices}${resourceToken}'
    location: location
    tags: tags
    sku: 'basic'
    publicNetworkAccess: 'Enabled'
    roleAssignments: [
      {
        principalId: mosaicIdentity.outputs.principalId
        principalType: 'ServicePrincipal'
        roleDefinitionIdOrName: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '8ebe5a00-799e-43f5-93ac-243d3dce84a7') // Search Service Contributor
      }
      {
        principalId: mosaicIdentity.outputs.principalId
        principalType: 'ServicePrincipal'
        roleDefinitionIdOrName: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '1407120a-92aa-4202-b7e9-c0e197c71c8f') // Search Index Data Contributor
      }
    ]
  }
}

// Azure Cosmos DB (Serverless, NoSQL + Gremlin API) - AVM with Managed Identity
module cosmosAccount 'br/public:avm/res/document-db/database-account:0.8.1' = {
  name: 'cosmos-account'
  params: {
    name: '${abbrs.documentDBDatabaseAccounts}${resourceToken}'
    location: location
    tags: tags
    capabilitiesToAdd: [
      'EnableGremlin'
      'EnableServerless'
    ]
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    sqlDatabases: [
      {
        name: 'mosaic-memory'
        containers: [
          {
            name: 'memories'
            paths: ['/sessionId']
            kind: 'Hash'
          }
        ]
      }
    ]
    gremlinDatabases: [
      {
        name: 'mosaic-graph'
        graphs: [
          {
            name: 'code-dependencies'
            // Note: No throughput specified for serverless containers
            indexingPolicy: {
              indexingMode: 'consistent'
              automatic: true
            }
          }
        ]
      }
    ]
    roleAssignments: [
      {
        principalId: mosaicIdentity.outputs.principalId
        principalType: 'ServicePrincipal'
        roleDefinitionIdOrName: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b24988ac-6180-42a0-ab88-20f7382dd24c') // Cosmos DB Contributor
      }
    ]
  }
}

// Azure Cache for Redis (Basic C0 tier) - AVM with Managed Identity  
module redisCache 'br/public:avm/res/cache/redis:0.9.0' = {
  name: 'redis-cache'
  params: {
    name: '${abbrs.cacheRedis}${resourceToken}'
    location: location
    tags: tags
    skuName: 'Basic'
    publicNetworkAccess: 'Enabled'
    redisConfiguration: {
      'aad-enabled': 'true'
    }
    roleAssignments: [
      {
        principalId: mosaicIdentity.outputs.principalId
        principalType: 'ServicePrincipal'
        roleDefinitionIdOrName: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3')
      }
    ]
  }
}

// Azure OpenAI Service
module openAI 'br/public:avm/res/cognitive-services/account:0.7.0' = {
  name: 'azure-openai'
  params: {
    name: '${abbrs.cognitiveServicesAccounts}${resourceToken}'
    location: location
    tags: tags
    kind: 'OpenAI'
    sku: 'S0'
    customSubDomainName: '${abbrs.cognitiveServicesAccounts}${resourceToken}'
    publicNetworkAccess: 'Enabled'
    roleAssignments: [
      {
        principalId: mosaicIdentity.outputs.principalId
        principalType: 'ServicePrincipal'
        roleDefinitionIdOrName: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
      }
    ]
  }
}

// Azure Machine Learning Workspace
module mlWorkspace 'br/public:avm/res/machine-learning-services/workspace:0.7.0' = {
  name: 'ml-workspace'
  params: {
    name: '${abbrs.machineLearningServicesWorkspaces}${resourceToken}'
    location: location
    tags: tags
    sku: 'Basic'
    associatedApplicationInsightsResourceId: monitoring.outputs.applicationInsightsResourceId
    associatedKeyVaultResourceId: keyVault.outputs.resourceId
    associatedStorageAccountResourceId: storageAccount.outputs.resourceId
    roleAssignments: [
      {
        principalId: mosaicIdentity.outputs.principalId
        principalType: 'ServicePrincipal'
        roleDefinitionIdOrName: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '8e3af657-a8ff-443c-a75c-2fe8c4bcb635')
      }
    ]
  }
}

// Storage Account for ML Workspace
module storageAccount 'br/public:avm/res/storage/storage-account:0.14.3' = {
  name: 'storage-account'
  params: {
    name: '${abbrs.storageStorageAccounts}${resourceToken}'
    location: location
    tags: tags
    kind: 'StorageV2'
    skuName: 'Standard_LRS'
    roleAssignments: [
      {
        principalId: mosaicIdentity.outputs.principalId
        principalType: 'ServicePrincipal'
        roleDefinitionIdOrName: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
      }
    ]
  }
}

// Key Vault for ML Workspace
module keyVault 'br/public:avm/res/key-vault/vault:0.9.0' = {
  name: 'key-vault'
  params: {
    name: '${abbrs.keyVaultVaults}${resourceToken}'
    location: location
    tags: tags
    enableSoftDelete: true
    roleAssignments: [
      {
        principalId: mosaicIdentity.outputs.principalId
        principalType: 'ServicePrincipal'
        roleDefinitionIdOrName: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '00482a5a-887f-4fb3-b363-3b7fe8e74483')
      }
    ]
  }
}

// Azure Functions for Memory Consolidation
module memoryConsolidatorFunction 'br/public:avm/res/web/site:0.11.1' = {
  name: 'memory-consolidator'
  params: {
    name: '${abbrs.webSitesFunctions}${resourceToken}'
    location: location
    tags: union(tags, { 'azd-service-name': 'memory-consolidator' })
    kind: 'functionapp'
    serverFarmResourceId: functionAppServicePlan.outputs.resourceId
    appInsightResourceId: monitoring.outputs.applicationInsightsResourceId
    storageAccountResourceId: storageAccount.outputs.resourceId
    managedIdentities: {
      systemAssigned: false
      userAssignedResourceIds: [mosaicIdentity.outputs.resourceId]
    }
    siteConfig: {
      pythonVersion: '3.11'
      functionAppScaleLimit: 200
      minimumElasticInstanceCount: 0
    }
    appSettingsKeyValuePairs: {
      FUNCTIONS_EXTENSION_VERSION: '~4'
      FUNCTIONS_WORKER_RUNTIME: 'python'
      AZURE_COSMOS_ENDPOINT: cosmosAccount.outputs.endpoint
      AZURE_USE_MANAGED_IDENTITY: 'true'
      AZURE_OPENAI_ENDPOINT: openAI.outputs.endpoint
      AZURE_CLIENT_ID: mosaicIdentity.outputs.clientId
    }
  }
}

// Function App Service Plan (Consumption)
module functionAppServicePlan 'br/public:avm/res/web/serverfarm:0.2.2' = {
  name: 'function-app-service-plan'
  params: {
    name: '${abbrs.webServerFarms}${resourceToken}'
    location: location
    tags: tags
    skuName: 'Y1'
    skuCapacity: 1
  }
}

// Outputs - Managed Identity Endpoints (No Connection Strings)
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerRegistry.outputs.loginServer
output AZURE_RESOURCE_MOSAIC_ID string = mosaic.outputs.resourceId
output AZURE_AI_SEARCH_ENDPOINT string = 'https://${aiSearch.outputs.name}.search.windows.net'
output AZURE_COSMOS_ENDPOINT string = cosmosAccount.outputs.endpoint
output AZURE_REDIS_ENDPOINT string = '${redisCache.outputs.hostName}:6380'
output AZURE_OPENAI_ENDPOINT string = openAI.outputs.endpoint
output AZURE_ML_WORKSPACE_NAME string = mlWorkspace.outputs.name
output AZURE_FUNCTIONS_ENDPOINT string = 'https://${memoryConsolidatorFunction.outputs.name}.azurewebsites.net'
output AZURE_CLIENT_ID string = mosaicIdentity.outputs.clientId
output AZURE_USE_MANAGED_IDENTITY string = 'true'
