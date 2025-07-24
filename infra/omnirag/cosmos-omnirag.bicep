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
    // OmniRAG Database Configuration - FIXED: Aligned with application expectations
    sqlDatabases: [
      {
        name: 'mosaic' // ✅ FIXED: Changed from 'mosaic-omnirag' to match application
        containers: [
          // Golden Node Container - Enhanced with hierarchical relationships and vector search
          {
            name: 'golden_nodes'
            paths: ['/partition_key'] // Changed from nodeType to partition_key for better partitioning
            kind: 'Hash'
            // Optimized for Golden Node hierarchical schema with vector search
            indexingPolicy: {
              indexingMode: 'consistent'  
              automatic: true
              includedPaths: [
                {
                  path: '/*'
                }
                // Core entity paths
                {
                  path: '/code_entity/entity_type/?'
                }
                {
                  path: '/code_entity/language/?'
                }
                {
                  path: '/code_entity/name/?'
                }
                // Hierarchical relationship paths for efficient queries
                {
                  path: '/code_entity/parent_id/?'
                }
                {
                  path: '/code_entity/hierarchy_level/?'
                }
                {
                  path: '/code_entity/hierarchy_path/*'
                }
                // File and git context paths
                {
                  path: '/file_context/file_path/?'
                }
                {
                  path: '/git_context/repository_url/?'
                }
                {
                  path: '/git_context/branch_name/?'
                }
                // Relationship and metadata paths
                {
                  path: '/relationships/*/relationship_type/?'
                }
                {
                  path: '/relationships/*/source_entity_id/?'
                }
                {
                  path: '/relationships/*/target_entity_id/?'
                }
                {
                  path: '/tags/*'
                }
                {
                  path: '/document_type/?'
                }
                {
                  path: '/processing_metadata/processing_stage/?'
                }
              ]
              // Exclude embeddings from indexing for optimal insertion performance
              excludedPaths: [
                {
                  path: '/embedding/*'
                }
                {
                  path: '/ai_enrichment/embedding/*'
                }
              ]
              // Composite indexes for hierarchical queries
              compositeIndexes: [
                // Hierarchical traversal optimization
                {
                  path: '/code_entity/parent_id ASC,/code_entity/hierarchy_level ASC'
                }
                {
                  path: '/code_entity/hierarchy_level ASC,/code_entity/entity_type ASC'
                }
                // Repository and file-based queries
                {
                  path: '/partition_key ASC,/code_entity/entity_type ASC'
                }
                {
                  path: '/git_context/repository_url ASC,/file_context/file_path ASC'
                }
                // Processing and relationship queries
                {
                  path: '/processing_metadata/processing_stage ASC,/processing_metadata/created_at ASC'
                }
              ]
            }
            // Vector indexing policy for native VectorDistance queries
            vectorEmbeddingPolicy: {
              vectorEmbeddings: [
                {
                  path: '/embedding'
                  dataType: 'float32'
                  distanceFunction: 'cosine'
                  dimensions: 1536
                }
              ]
            }
            // Vector index configuration following Microsoft best practices
            vectorIndexingPolicy: {
              vectorIndexes: [
                {
                  path: '/embedding'
                  type: 'quantizedFlat' // Optimal for 1536-dimension Azure OpenAI embeddings
                }
              ]
            }
          }
          // Query Server Containers - FIXED names to match application
          {
            name: 'knowledge' // ✅ FIXED: Renamed from 'documents' 
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
              excludedPaths: [
                {
                  path: '/embedding/*' // Exclude embeddings from indexing
                }
              ]
            }
          }
          {
            name: 'memory' // ✅ FIXED: Renamed from 'memories'
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
              excludedPaths: [
                {
                  path: '/embedding/*' // Exclude embeddings from indexing
                }
              ]
            }
          }
          // Additional containers for complete plugin support
          {
            name: 'diagrams' // ✅ NEW: Required by DiagramPlugin
            paths: ['/diagramType']
            kind: 'Hash'
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
            name: 'code_entities' // ✅ NEW: Required by GraphDataService
            paths: ['/entityType']
            kind: 'Hash'
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
            name: 'code_relationships' // ✅ NEW: Required by GraphDataService
            paths: ['/relationshipType']
            kind: 'Hash'
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
            name: 'repositories' // ✅ NEW: Required by GraphDataService
            paths: ['/repositoryUrl']
            kind: 'Hash'
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
          // Legacy containers - keep for backward compatibility during migration
          {
            name: 'libraries'
            paths: ['/libtype']
            kind: 'Hash'
            // Primary OmniRAG container with embedded graph relationships
            indexingPolicy: {
              indexingMode: 'consistent'
              automatic: true
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

// OmniRAG Outputs - UPDATED with correct naming and Golden Node support
output cosmosEndpoint string = cosmosAccount.outputs.endpoint
output cosmosAccountName string = cosmosAccount.outputs.name
output databaseName string = 'mosaic' // ✅ FIXED: Correct database name
// Query Server containers
output knowledgeContainerName string = 'knowledge' // ✅ FIXED: Renamed from documentsContainerName
output memoryContainerName string = 'memory' // ✅ FIXED: Renamed from memoriesContainerName  
output diagramsContainerName string = 'diagrams' // ✅ NEW
// GraphDataService containers
output codeEntitiesContainerName string = 'code_entities' // ✅ NEW
output codeRelationshipsContainerName string = 'code_relationships' // ✅ NEW
output repositoriesContainerName string = 'repositories' // ✅ NEW
// Golden Node container
output goldenNodesContainerName string = 'golden_nodes' // ✅ NEW
// Legacy containers
output librariesContainerName string = 'libraries'
output configContainerName string = 'config'