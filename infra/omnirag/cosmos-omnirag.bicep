// OmniRAG Pattern Implementation - Azure Cosmos DB for NoSQL
// Based on Microsoft's CosmosAIGraph reference architecture
// Provides unified vector search, embedded graph relationships, and memory storage

@description('The location used for all deployed resources')
param location string = resourceGroup().location

@description('Tags that will be applied to all resources')
param tags object = {}

@description('Unique resource token for naming')
param resourceToken string

@description('Abbreviations for resource naming')
param abbrs object

@description('Managed identity principal ID for role assignments')
param principalId string

// OmniRAG Pattern: Unified Azure Cosmos DB Configuration
// Follows Microsoft's CosmosAIGraph implementation for NoSQL-only approach
module cosmosAccount 'br/public:avm/res/document-db/database-account:0.8.1' = {
  name: 'omnirag-cosmos-account'
  params: {
    name: '${abbrs.documentDBDatabaseAccounts}${resourceToken}'
    location: location
    tags: union(tags, { 
      'omnirag-component': 'unified-backend'
      'pattern': 'microsoft-omnirag'
    })
    capabilitiesToAdd: [
      'EnableServerless' // Cost-optimized serverless billing for OmniRAG
    ]
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    // OmniRAG Database Configuration
    sqlDatabases: [
      {
        name: 'mosaic-omnirag'
        containers: [
          {
            name: 'libraries'
            paths: ['/libtype']
            kind: 'Hash'
            // Primary OmniRAG container with embedded graph relationships
            // Uses dependency_ids and developers arrays per CosmosAIGraph pattern
            indexingPolicy: {
              indexingMode: 'consistent'
              automatic: true
              // Additional indexes for graph relationship queries
              includedPaths: [
                {
                  path: '/*'
                }
                {
                  path: '/dependency_ids/*'
                }
                {
                  path: '/developers/*'
                }
                {
                  path: '/used_by_lib/*'
                }
              ]
            }
          }
          {
            name: 'memories'
            paths: ['/sessionId']
            kind: 'Hash'
            // Memory storage with vector search for semantic similarity
            indexingPolicy: {
              indexingMode: 'consistent'
              automatic: true
              includedPaths: [
                {
                  path: '/*'
                }
              ]
            }
          }
          {
            name: 'documents'
            paths: ['/category']
            kind: 'Hash'
            // Document storage with vector search for hybrid retrieval
            indexingPolicy: {
              indexingMode: 'consistent'
              automatic: true
              includedPaths: [
                {
                  path: '/*'
                }
              ]
            }
          }
          {
            name: 'config'
            paths: ['/pk']
            kind: 'Hash'
            // Configuration and system entities per CosmosAIGraph pattern
            indexingPolicy: {
              indexingMode: 'consistent'
              automatic: true
              includedPaths: [
                {
                  path: '/*'
                }
              ]
            }
          }
        ]
      }
    ]
    roleAssignments: [
      {
        principalId: principalId
        principalType: 'ServicePrincipal'
        roleDefinitionIdOrName: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b24988ac-6180-42a0-ab88-20f7382dd24c') // Cosmos DB Contributor
      }
    ]
  }
}

// OmniRAG Outputs following CosmosAIGraph conventions
output cosmosEndpoint string = cosmosAccount.outputs.endpoint
output cosmosAccountName string = cosmosAccount.outputs.name
output databaseName string = 'mosaic-omnirag'
output librariesContainerName string = 'libraries'
output memoriesContainerName string = 'memories'
output documentsContainerName string = 'documents'
output configContainerName string = 'config'