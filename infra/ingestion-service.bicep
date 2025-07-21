// Ingestion Service - Azure Container App Job for heavy repository processing
// Scheduled/manual execution for resource-intensive operations

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

resource ingestionJob 'Microsoft.App/jobs@2023-05-01' = {
  name: 'mosaic-ingestion-job-${environmentName}'
  location: location
  properties: {
    environmentId: containerAppsEnvironmentId
    configuration: {
      triggerType: 'Manual'  // Can be triggered manually or via schedule
      replicaTimeout: 3600   // 1 hour timeout for large repositories
      replicaRetryLimit: 2
      manualTriggerConfig: {
        replicaCompletionCount: 1
        parallelism: 1
      }
    }
    template: {
      containers: [
        {
          name: 'ingestion-service'
          image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest' // TODO: Replace with actual image
          resources: {
            cpu: json('2.0')      // Higher CPU for AST parsing
            memory: '4Gi'         // Higher memory for large repositories
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
              name: 'PYTHONPATH'
              value: '/app/src'
            }
          ]
          command: ['python', '-m', 'ingestion_service.main']
          args: ['--repository-url', '$(REPOSITORY_URL)', '--branch', '$(BRANCH)']
        }
      ]
    }
  }
  identity: {
    type: 'SystemAssigned'
  }
}

// Optional: Scheduled trigger for regular repository updates
resource ingestionSchedule 'Microsoft.App/jobs@2023-05-01' = {
  name: 'mosaic-ingestion-schedule-${environmentName}'
  location: location
  properties: {
    environmentId: containerAppsEnvironmentId
    configuration: {
      triggerType: 'Schedule'
      replicaTimeout: 3600
      replicaRetryLimit: 2
      scheduleTriggerConfig: {
        replicaCompletionCount: 1
        parallelism: 1
        cronExpression: '0 2 * * 0'  // Weekly on Sunday at 2 AM
      }
    }
    template: {
      containers: [
        {
          name: 'ingestion-service'
          image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest' // TODO: Replace with actual image
          resources: {
            cpu: json('2.0')
            memory: '4Gi'
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
              name: 'PYTHONPATH'
              value: '/app/src'
            }
            {
              name: 'REPOSITORY_URL'
              value: 'https://github.com/example/repo'  // Configure as needed
            }
            {
              name: 'BRANCH'
              value: 'main'
            }
          ]
          command: ['python', '-m', 'ingestion_service.main']
          args: ['--repository-url', '$(REPOSITORY_URL)', '--branch', '$(BRANCH)']
        }
      ]
    }
  }
  identity: {
    type: 'SystemAssigned'
  }
}

output ingestionJobName string = ingestionJob.name
output ingestionScheduleName string = ingestionSchedule.name
output ingestionJobPrincipalId string = ingestionJob.identity.principalId
output ingestionSchedulePrincipalId string = ingestionSchedule.identity.principalId