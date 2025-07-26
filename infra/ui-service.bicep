// UI Service - Azure Container App for Streamlit web interface
// Interactive graph visualization and chat interface

@description('Environment name')
param environmentName string = 'dev'

@description('Location for resources')
param location string = resourceGroup().location

@description('Container Apps Environment ID')
param containerAppsEnvironmentId string

@description('Container Registry Login Server')
param containerRegistryLoginServer string

@description('Managed Identity Principal ID for ACR access')
param managedIdentityPrincipalId string

resource uiService 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'mosaic-ui-${environmentName}'
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnvironmentId
    configuration: {
      ingress: {
        external: true
        targetPort: 8501
        transport: 'http'
        allowInsecure: false
      }
      secrets: []
    }
    template: {
      containers: [
        {
          name: 'ui-service'
          image: '${containerRegistryLoginServer}/mosaic-ui:latest'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'STREAMLIT_SERVER_PORT'
              value: '8501'
            }
            {
              name: 'STREAMLIT_SERVER_ADDRESS'
              value: '0.0.0.0'
            }
            {
              name: 'STREAMLIT_SERVER_HEADLESS'
              value: 'true'
            }
            {
              name: 'PYTHONPATH'
              value: '/app'
            }
          ]
          command: ['streamlit', 'run', 'app.py', '--server.port=8501', '--server.address=0.0.0.0']
        }
      ]
      scale: {
        minReplicas: 1 // Always-on for user access
        maxReplicas: 5
        rules: [
          {
            name: 'http-scale'
            http: {
              metadata: {
                concurrentRequests: '20'
              }
            }
          }
        ]
      }
    }
  }
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentityPrincipalId}': {}
    }
  }
}

output uiServiceUrl string = 'https://${uiService.properties.configuration.ingress.fqdn}'
output uiServicePrincipalId string = uiService.identity.principalId
