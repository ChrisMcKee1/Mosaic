// Query Server - Azure Container App for real-time MCP requests
// Lightweight, always-on service for fast query operations

@description('Environment name')
param environmentName string = 'dev'

@description('Location for resources')
param location string = resourceGroup().location

@description('Container Apps Environment ID')
param containerAppsEnvironmentId string

@description('Azure OpenAI endpoint')
param azureOpenAIEndpoint string

@description('Azure Cosmos DB endpoint')  
param cosmosDbEndpoint string

@description('Azure Redis endpoint')
param redisEndpoint string

@description('Azure ML endpoint URL for reranking')
param azureMLEndpointUrl string

resource queryServer 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'mosaic-query-server-${environmentName}'
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnvironmentId
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
      secrets: []
    }
    template: {
      containers: [
        {
          name: 'query-server'
          image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest' // TODO: Replace with actual image
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
          env: [
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: azureOpenAIEndpoint
            }
            {
              name: 'AZURE_COSMOS_DB_ENDPOINT'
              value: cosmosDbEndpoint
            }
            {
              name: 'AZURE_REDIS_ENDPOINT'
              value: redisEndpoint
            }
            {
              name: 'AZURE_ML_ENDPOINT_URL'
              value: azureMLEndpointUrl
            }
            {
              name: 'SERVER_PORT'
              value: '8000'
            }
            {
              name: 'SERVER_HOST'
              value: '0.0.0.0'
            }
          ]
          command: ['python', '-m', 'mosaic.server.main']
        }
      ]
      scale: {
        minReplicas: 1  // Always-on for real-time queries
        maxReplicas: 3
        rules: [
          {
            name: 'http-scale'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
  identity: {
    type: 'SystemAssigned'
  }
}

output queryServerUrl string = 'https://${queryServer.properties.configuration.ingress.fqdn}'
output queryServerPrincipalId string = queryServer.identity.principalId