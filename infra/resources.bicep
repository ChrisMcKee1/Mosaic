// Mosaic MCP Tool - Modular OmniRAG Infrastructure
// Based on Microsoft's CosmosAIGraph reference architecture

@description('The location used for all deployed resources')
param location string = resourceGroup().location

@description('Tags that will be applied to all resources')
param tags object = {}

@description('Whether mosaic container already exists')
param mosaicExists bool

@description('Id of the user or app to assign application roles')
param principalId string

@description('Principal type of user or app')
param principalType string

var abbrs = loadJsonContent('./abbreviations.json')
var resourceToken = uniqueString(subscription().id, resourceGroup().id, location)


// Simple Log Analytics workspace for monitoring
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${abbrs.operationalInsightsWorkspaces}${resourceToken}'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// Application Insights
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${abbrs.insightsComponents}${resourceToken}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
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
    roleAssignments: [
      {
        principalId: mosaicIdentity.outputs.principalId
        principalType: 'ServicePrincipal'
        roleDefinitionIdOrName: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
      }
    ]
  }
}

// Managed Identity for all services
module mosaicIdentity 'br/public:avm/res/managed-identity/user-assigned-identity:0.2.1' = {
  name: 'mosaicidentity'
  params: {
    name: '${abbrs.managedIdentityUserAssignedIdentities}mosaic-${resourceToken}'
    location: location
    tags: union(tags, { 'component': 'identity' })
  }
}

// OmniRAG Pattern: Unified Cosmos DB Backend
// Modular deployment following CosmosAIGraph architecture
module omniragCosmos './omnirag/cosmos-omnirag.bicep' = {
  name: 'omnirag-backend'
  params: {
    location: location
    tags: tags
    resourceToken: resourceToken
    abbrs: abbrs
    principalId: mosaicIdentity.outputs.principalId
  }
}

// Container Apps with OmniRAG configuration
module containerApps './modules/container-apps.bicep' = {
  name: 'container-apps'
  params: {
    location: location
    tags: tags
    resourceToken: resourceToken
    abbrs: abbrs
    containerRegistryLoginServer: containerRegistry.outputs.loginServer
    managedIdentityResourceId: mosaicIdentity.outputs.resourceId
    managedIdentityPrincipalId: mosaicIdentity.outputs.principalId
    managedIdentityClientId: mosaicIdentity.outputs.clientId
    logAnalyticsWorkspaceResourceId: logAnalytics.id
    applicationInsightsConnectionString: applicationInsights.properties.ConnectionString
    cosmosEndpoint: omniragCosmos.outputs.cosmosEndpoint
    redisEndpoint: '${redisCache.outputs.hostName}:6380'
    openAIEndpoint: aiServices.properties.endpoint
    mlWorkspaceName: 'ml-workspace-placeholder'
    functionsEndpoint: 'https://${memoryConsolidatorFunction.outputs.name}.azurewebsites.net'
    tenantId: tenant().tenantId
    mosaicExists: mosaicExists
  }
}
// Azure Cache for Redis (Basic C0 tier) - AVM with Managed Identity  
module redisCache 'br/public:avm/res/cache/redis:0.9.0' = {
  name: 'redis-cache'
  params: {
    name: '${abbrs.cacheRedis}${resourceToken}'
    location: location
    tags: union(tags, { 'component': 'short-term-memory' })
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

// Azure AI Foundry Hub
resource aiFoundryHub 'Microsoft.MachineLearningServices/workspaces@2024-04-01-preview' = {
  name: '${abbrs.machineLearningServicesWorkspaces}hub-${resourceToken}'
  location: location
  tags: union(tags, { 'component': 'ai-foundry-hub' })
  kind: 'Hub'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'Mosaic AI Foundry Hub'
    description: 'Azure AI Foundry Hub for Mosaic MCP Tool'
    storageAccount: storageAccount.outputs.resourceId
    keyVault: keyVault.outputs.resourceId
  }
}

// Azure AI Services for AI Foundry
resource aiServices 'Microsoft.CognitiveServices/accounts@2024-04-01-preview' = {
  name: '${abbrs.cognitiveServicesAccounts}${resourceToken}'
  location: location
  tags: union(tags, { 'component': 'ai-foundry-services' })
  kind: 'AIServices'
  sku: {
    name: 'S0'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: 'mosaic-ai-${resourceToken}'
  }
}

// GPT-4o Deployment (Latest GA model)
resource gpt4oDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: aiServices
  name: 'gpt-4o'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o'
      version: '2024-11-20'
    }
  }
  sku: {
    name: 'Standard'
    capacity: 10
  }
}

// GPT-4o-mini Deployment (Fast, cost-effective model)
resource gpt4oMiniDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: aiServices
  name: 'gpt-4o-mini'
  dependsOn: [gpt4oDeployment]
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o-mini'
      version: '2024-07-18'
    }
  }
  sku: {
    name: 'Standard'
    capacity: 10
  }
}

// Text Embedding Deployment (Latest model)
resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: aiServices
  name: 'text-embedding-3-small'
  dependsOn: [gpt4oMiniDeployment]
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-3-small'
      version: '1'
    }
  }
  sku: {
    name: 'Standard'
    capacity: 10
  }
}

// Storage Account for ML Workspace
module storageAccount 'br/public:avm/res/storage/storage-account:0.14.3' = {
  name: 'storage-account'
  params: {
    name: '${abbrs.storageStorageAccounts}${resourceToken}'
    location: location
    tags: union(tags, { 'component': 'ml-storage' })
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
    tags: union(tags, { 'component': 'ml-keyvault' })
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

// Azure Machine Learning Workspace - temporarily commented out due to soft delete conflict
// module mlWorkspace 'br/public:avm/res/machine-learning-services/workspace:0.7.0' = {
//   name: 'ml-workspace'
//   params: {
//     name: '${abbrs.machineLearningServicesWorkspaces}${resourceToken}'
//     location: location
//     tags: union(tags, { 'component': 'semantic-reranking' })
//     sku: 'Basic'
//     associatedApplicationInsightsResourceId: applicationInsights.id
//     associatedKeyVaultResourceId: keyVault.outputs.resourceId
//     associatedStorageAccountResourceId: storageAccount.outputs.resourceId
//     roleAssignments: [
//       {
//         principalId: mosaicIdentity.outputs.principalId
//         principalType: 'ServicePrincipal'
//         roleDefinitionIdOrName: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '8e3af657-a8ff-443c-a75c-2fe8c4bcb635')
//       }
//     ]
//   }
// }

// Function App Service Plan (Consumption)
module functionAppServicePlan 'br/public:avm/res/web/serverfarm:0.2.2' = {
  name: 'function-app-service-plan'
  params: {
    name: '${abbrs.webServerFarms}${resourceToken}'
    location: location
    tags: union(tags, { 'component': 'memory-consolidation' })
    skuName: 'Y1'
    skuCapacity: 1
  }
}

// Azure Functions for Memory Consolidation
module memoryConsolidatorFunction 'br/public:avm/res/web/site:0.11.1' = {
  name: 'memory-consolidator'
  params: {
    name: '${abbrs.webSitesFunctions}${resourceToken}'
    location: location
    tags: union(tags, { 
      'azd-service-name': 'memory-consolidator'
      'component': 'memory-consolidation'
    })
    kind: 'functionapp'
    serverFarmResourceId: functionAppServicePlan.outputs.resourceId
    appInsightResourceId: applicationInsights.id
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
      // OmniRAG Configuration for Functions
      MOSAIC_COSMOS_ENDPOINT: omniragCosmos.outputs.cosmosEndpoint
      MOSAIC_DATABASE_NAME: omniragCosmos.outputs.databaseName
      MOSAIC_MEMORIES_CONTAINER: omniragCosmos.outputs.memoriesContainerName
      AZURE_USE_MANAGED_IDENTITY: 'true'
      AZURE_OPENAI_ENDPOINT: aiServices.properties.endpoint
      AZURE_CLIENT_ID: mosaicIdentity.outputs.clientId
    }
  }
}

/* 
  OAuth 2.1 Setup Instructions (FR-14):
  
  After deployment, create an App Registration in Microsoft Entra ID:
  
  1. az ad app create --display-name "Mosaic MCP Server" \
     --web-redirect-uris "https://${containerApps.outputs.mosaicAppFqdn}/auth/callback"
     
  2. Configure the following application settings:
     - Authentication: Web platform with redirect URI
     - API permissions: User.Read (Microsoft Graph)
     - Token configuration: Add optional claims if needed
     
  3. Update container app with OAuth Client ID:
     az containerapp update --name "mosaic" \
       --resource-group "${resourceGroup().name}" \
       --set-env-vars "AZURE_OAUTH_CLIENT_ID=<app-id-from-step-1>"
       
  4. FastMCP will handle OAuth 2.1 flow automatically with these settings
*/

// Outputs - OmniRAG Configuration + Standard Infrastructure
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerRegistry.outputs.loginServer
output AZURE_RESOURCE_MOSAIC_ID string = containerApps.outputs.mosaicAppResourceId
output AZURE_CLIENT_ID string = mosaicIdentity.outputs.clientId
output AZURE_USE_MANAGED_IDENTITY string = 'true'

// OmniRAG Outputs
output MOSAIC_COSMOS_ENDPOINT string = omniragCosmos.outputs.cosmosEndpoint
output MOSAIC_DATABASE_NAME string = omniragCosmos.outputs.databaseName
output MOSAIC_LIBRARIES_CONTAINER string = omniragCosmos.outputs.librariesContainerName
output MOSAIC_MEMORIES_CONTAINER string = omniragCosmos.outputs.memoriesContainerName
output MOSAIC_DOCUMENTS_CONTAINER string = omniragCosmos.outputs.documentsContainerName
output MOSAIC_CONFIG_CONTAINER string = omniragCosmos.outputs.configContainerName

// Service Endpoints
output AZURE_REDIS_ENDPOINT string = '${redisCache.outputs.hostName}:6380'
output AZURE_OPENAI_ENDPOINT string = aiServices.properties.endpoint
output AZURE_ML_WORKSPACE_NAME string = 'ml-workspace-placeholder'
output AZURE_FUNCTIONS_ENDPOINT string = 'https://${memoryConsolidatorFunction.outputs.name}.azurewebsites.net'

// OAuth 2.1 Configuration Outputs (FR-14)
output AZURE_TENANT_ID string = tenant().tenantId
output MOSAIC_APP_URL string = containerApps.outputs.mosaicAppUrl
output OAUTH_SETUP_REQUIRED string = 'true'