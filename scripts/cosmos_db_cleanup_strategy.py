#!/usr/bin/env python3
"""
Cosmos DB Cleanup Strategy for Golden Node Schema Migration

This script provides a comprehensive strategy to clean up the existing Cosmos DB setup
and prepare for the new Golden Node schema while addressing configuration mismatches
between infrastructure and application layers.

Based on AI Agent Ingestion Plan research:
- Current infrastructure creates 'mosaic-omnirag' database with containers: libraries, memories, documents, config
- Application expects 'mosaic' database with containers: knowledge, memory
- New Golden Node schema needs unified container with proper partitioning

Strategy:
1. Fix configuration mismatches
2. Create new Golden Node container
3. Provide safe migration path without breaking query server
4. Clean up old containers after migration validation
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import Azure SDK components (these would be available in the actual environment)
try:
    from azure.cosmos import CosmosClient, ContainerProperties, PartitionKey
    from azure.identity import DefaultAzureCredential

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("⚠️  Azure SDK not available - running in simulation mode")

# Local imports
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from mosaic.config.settings import MosaicSettings
    from ingestion_service.models.golden_node import GoldenNode

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("⚠️  Mosaic models not available - running in simulation mode")


class CosmosDBCleanupStrategy:
    """
    Comprehensive Cosmos DB cleanup and migration strategy.

    Handles the transition from the current mismatched configuration to
    the new Golden Node schema with minimal disruption to the query server.
    """

    def __init__(self, dry_run: bool = True):
        """Initialize cleanup strategy."""
        self.dry_run = dry_run
        self.logger = self._setup_logging()

        # Configuration analysis
        self.config_issues = []
        self.migration_plan = []
        self.cleanup_actions = []

        # Azure resources (would be initialized with actual credentials)
        self.cosmos_client: Optional[CosmosClient] = None
        self.settings: Optional[MosaicSettings] = None

        self.logger.info(f"Cosmos DB Cleanup Strategy initialized (dry_run={dry_run})")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the cleanup strategy."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        return logging.getLogger("cosmos_cleanup")

    async def analyze_current_configuration(self) -> Dict[str, Any]:
        """
        Analyze current configuration mismatches and infrastructure state.

        Returns comprehensive analysis of what needs to be fixed.
        """
        self.logger.info("🔍 Analyzing current configuration...")

        analysis = {
            "configuration_mismatches": [],
            "infrastructure_state": {},
            "application_expectations": {},
            "golden_node_requirements": {},
            "migration_recommendations": [],
        }

        # 1. Identify configuration mismatches
        config_mismatches = [
            {
                "issue": "Database name mismatch",
                "infrastructure": "mosaic-omnirag",
                "application": "mosaic",
                "impact": "Application cannot connect to database",
                "severity": "CRITICAL",
            },
            {
                "issue": "Container name mismatch - knowledge",
                "infrastructure": "documents",
                "application": "knowledge",
                "impact": "RetrievalPlugin fails to find container",
                "severity": "CRITICAL",
            },
            {
                "issue": "Container name mismatch - memory",
                "infrastructure": "memories",
                "application": "memory",
                "impact": "MemoryPlugin fails to find container",
                "severity": "CRITICAL",
            },
            {
                "issue": "Missing containers",
                "infrastructure": "Not defined",
                "application": "diagrams, code_entities, code_relationships, repositories",
                "impact": "DiagramPlugin and GraphDataService fail",
                "severity": "HIGH",
            },
        ]

        analysis["configuration_mismatches"] = config_mismatches
        self.config_issues.extend(config_mismatches)

        # 2. Current infrastructure state (simulated)
        analysis["infrastructure_state"] = {
            "database_name": "mosaic-omnirag",
            "containers": {
                "libraries": {
                    "partition_key": "/libtype",
                    "purpose": "OmniRAG library dependencies",
                    "usage": "Active in query server",
                },
                "memories": {
                    "partition_key": "/sessionId",
                    "purpose": "Memory storage with vector search",
                    "usage": "Expected by MemoryPlugin but misnamed",
                },
                "documents": {
                    "partition_key": "/category",
                    "purpose": "Document storage with vector search",
                    "usage": "Expected by RetrievalPlugin but misnamed",
                },
                "config": {
                    "partition_key": "/pk",
                    "purpose": "Configuration and system entities",
                    "usage": "System configuration",
                },
            },
        }

        # 3. Application expectations
        analysis["application_expectations"] = {
            "database_name": "mosaic",
            "required_containers": {
                "knowledge": "RetrievalPlugin hybrid search",
                "memory": "MemoryPlugin multi-layered storage",
                "diagrams": "DiagramPlugin Mermaid storage",
                "code_entities": "GraphDataService AST entities",
                "code_relationships": "GraphDataService relationships",
                "repositories": "GraphDataService repository metadata",
            },
        }

        # 4. Golden Node requirements
        analysis["golden_node_requirements"] = {
            "new_container": "golden_nodes",
            "partition_strategy": "/nodeType or /entityType",
            "schema": "Unified Golden Node model",
            "benefits": [
                "Single source of truth for code entities",
                "Embedded relationships (OmniRAG pattern)",
                "AI enrichment data co-located",
                "Simplified query patterns",
                "Better performance through data locality",
            ],
        }

        # 5. Migration recommendations
        analysis["migration_recommendations"] = [
            {
                "priority": 1,
                "action": "Fix database name mismatch",
                "description": "Update infrastructure to use 'mosaic' database name",
                "impact": "Enables application connectivity",
            },
            {
                "priority": 2,
                "action": "Rename existing containers",
                "description": "Align container names with application expectations",
                "impact": "Existing plugins can function",
            },
            {
                "priority": 3,
                "action": "Create Golden Node container",
                "description": "Add new unified container for AI agent data",
                "impact": "Enables new ingestion service",
            },
            {
                "priority": 4,
                "action": "Migrate existing data",
                "description": "Transform and move data to Golden Node schema",
                "impact": "Preserves existing knowledge",
            },
            {
                "priority": 5,
                "action": "Remove old containers",
                "description": "Clean up after successful migration",
                "impact": "Simplified maintenance",
            },
        ]

        self.logger.info(
            f"✅ Analysis complete: {len(config_mismatches)} critical issues found"
        )
        return analysis

    async def create_migration_plan(self) -> List[Dict[str, Any]]:
        """
        Create detailed migration plan with specific steps.

        Returns step-by-step migration plan that won't break existing functionality.
        """
        self.logger.info("📋 Creating migration plan...")

        migration_steps = [
            {
                "step": 1,
                "phase": "Infrastructure Fix",
                "action": "Update Bicep template for database name",
                "description": "Change 'mosaic-omnirag' to 'mosaic' in cosmos-omnirag.bicep",
                "files_to_modify": ["infra/omnirag/cosmos-omnirag.bicep"],
                "risk": "LOW",
                "rollback_strategy": "Revert Bicep template change",
                "validation": "Deploy to dev environment and verify connectivity",
            },
            {
                "step": 2,
                "phase": "Infrastructure Fix",
                "action": "Rename containers to match application expectations",
                "description": "Update container names: documents→knowledge, memories→memory",
                "files_to_modify": ["infra/omnirag/cosmos-omnirag.bicep"],
                "risk": "MEDIUM",
                "rollback_strategy": "Revert container names in Bicep",
                "validation": "Verify query server plugins can connect",
            },
            {
                "step": 3,
                "phase": "Infrastructure Enhancement",
                "action": "Add missing containers",
                "description": "Add diagrams, code_entities, code_relationships, repositories containers",
                "files_to_modify": ["infra/omnirag/cosmos-omnirag.bicep"],
                "risk": "LOW",
                "rollback_strategy": "Remove new containers",
                "validation": "Verify DiagramPlugin and GraphDataService connectivity",
            },
            {
                "step": 4,
                "phase": "Golden Node Integration",
                "action": "Add Golden Node container",
                "description": "Create golden_nodes container with optimal partitioning",
                "container_spec": {
                    "name": "golden_nodes",
                    "partition_key": "/nodeType",
                    "indexing_policy": "Optimized for Golden Node queries",
                },
                "risk": "LOW",
                "rollback_strategy": "Remove golden_nodes container",
                "validation": "Test Golden Node model CRUD operations",
            },
            {
                "step": 5,
                "phase": "Application Configuration",
                "action": "Update application settings",
                "description": "Add Golden Node container configuration to MosaicSettings",
                "files_to_modify": ["src/mosaic/config/settings.py"],
                "risk": "LOW",
                "rollback_strategy": "Remove Golden Node settings",
                "validation": "Verify settings validation passes",
            },
            {
                "step": 6,
                "phase": "Data Migration",
                "action": "Create data migration script",
                "description": "Transform existing data to Golden Node schema",
                "implementation": "Gradual migration with dual-write pattern",
                "risk": "MEDIUM",
                "rollback_strategy": "Stop migration, revert to old containers",
                "validation": "Compare data integrity between old and new schemas",
            },
            {
                "step": 7,
                "phase": "Validation",
                "action": "Full system testing",
                "description": "Validate all components work with new configuration",
                "tests": [
                    "Query server functionality",
                    "AI agent ingestion pipeline",
                    "Memory and retrieval operations",
                    "Performance benchmarking",
                ],
                "risk": "LOW",
                "rollback_strategy": "Full rollback to previous configuration",
                "validation": "All functional and performance tests pass",
            },
            {
                "step": 8,
                "phase": "Cleanup",
                "action": "Remove old containers (optional)",
                "description": "Clean up old containers after successful migration",
                "timing": "Only after 30+ days of stable operation",
                "risk": "LOW",
                "rollback_strategy": "Recreate containers from backups",
                "validation": "Confirm no dependencies on old containers",
            },
        ]

        self.migration_plan = migration_steps
        self.logger.info(f"✅ Migration plan created: {len(migration_steps)} steps")
        return migration_steps

    async def generate_bicep_updates(self) -> Dict[str, str]:
        """
        Generate the updated Bicep template content to fix configuration issues.

        Returns the new Bicep template content that aligns infrastructure with application needs.
        """
        self.logger.info("🔧 Generating Bicep template updates...")

        # Read current Bicep template (simulated)
        current_bicep = """
        // Current problematic configuration
        sqlDatabases: [
          {
            name: 'mosaic-omnirag'  // ❌ MISMATCH: App expects 'mosaic'
            containers: [
              {
                name: 'libraries'     // ✅ Keep for OmniRAG
                paths: ['/libtype']
                kind: 'Hash'
              }
              {
                name: 'memories'      // ❌ MISMATCH: App expects 'memory'
                paths: ['/sessionId']
                kind: 'Hash'
              }
              {
                name: 'documents'     // ❌ MISMATCH: App expects 'knowledge'
                paths: ['/category']
                kind: 'Hash'
              }
              {
                name: 'config'        // ✅ Keep for system config
                paths: ['/pk']
                kind: 'Hash'
              }
            ]
          }
        ]
        """

        # Generate updated Bicep template
        updated_bicep = """// OmniRAG Pattern Implementation - Azure Cosmos DB for NoSQL
// UPDATED: Fixed configuration mismatches and added Golden Node support
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
// FIXED: Aligned database and container names with application expectations
module cosmosAccount 'br/public:avm/res/document-db/database-account:0.8.1' = {
  name: 'omnirag-cosmos-account'
  params: {
    name: '${abbrs.documentDBDatabaseAccounts}${resourceToken}'
    location: location
    tags: union(tags, { 
      'omnirag-component': 'unified-backend'
      'pattern': 'microsoft-omnirag'
      'ai-agent-ready': 'true'
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
    // FIXED: Database name aligned with application expectations
    sqlDatabases: [
      {
        name: 'mosaic' // ✅ FIXED: Changed from 'mosaic-omnirag' to match app
        containers: [
          // Golden Node Container - NEW for AI Agent architecture
          {
            name: 'golden_nodes'
            paths: ['/nodeType']
            kind: 'Hash'
            // Optimized for Golden Node unified schema
            indexingPolicy: {
              indexingMode: 'consistent'  
              automatic: true
              includedPaths: [
                {
                  path: '/*'
                }
                {
                  path: '/code_entity/entity_type/?'
                }
                {
                  path: '/code_entity/language/?'
                }
                {
                  path: '/file_context/file_path/?'
                }
                {
                  path: '/git_context/repository_url/?'
                }
                {
                  path: '/relationships/*/relationship_type/?'
                }
                {
                  path: '/tags/*'
                }
              ]
              // Exclude embeddings from indexing for performance
              excludedPaths: [
                {
                  path: '/embedding/*'
                }
                {
                  path: '/ai_enrichment/embedding/*'
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

// Updated outputs with correct naming
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
"""

        # Generate updated settings.py content
        updated_settings = '''
    # UPDATED: Added Golden Node container configuration
    azure_cosmos_golden_nodes_container: str = Field(
        default="golden_nodes", 
        description="Cosmos DB container name for Golden Node unified schema"
    )
    
    # UPDATED: Added missing container configurations for complete plugin support
    azure_cosmos_diagrams_container: str = Field(
        default="diagrams",
        description="Cosmos DB container name for diagram storage"
    )
    azure_cosmos_code_entities_container: str = Field(
        default="code_entities", 
        description="Cosmos DB container name for code entities"
    )
    azure_cosmos_code_relationships_container: str = Field(
        default="code_relationships",
        description="Cosmos DB container name for code relationships"
    )
    azure_cosmos_repositories_container: str = Field(
        default="repositories",
        description="Cosmos DB container name for repository metadata"
    )

    def get_cosmos_config(self) -> dict:
        """Get Azure Cosmos DB configuration dictionary - UPDATED for Golden Node support."""
        return {
            "endpoint": self.azure_cosmos_endpoint,
            "database_name": self.azure_cosmos_database_name,
            # Query Server containers
            "container_name": self.azure_cosmos_container_name,  # knowledge
            "memory_container": self.azure_cosmos_memory_container,  # memory
            "diagrams_container": self.azure_cosmos_diagrams_container,  # diagrams
            # GraphDataService containers  
            "code_entities_container": self.azure_cosmos_code_entities_container,
            "code_relationships_container": self.azure_cosmos_code_relationships_container,
            "repositories_container": self.azure_cosmos_repositories_container,
            # Golden Node container
            "golden_nodes_container": self.azure_cosmos_golden_nodes_container,
        }
'''

        updates = {
            "bicep_template": updated_bicep,
            "settings_additions": updated_settings,
            "summary": """
Configuration Updates Summary:
✅ Fixed database name: 'mosaic-omnirag' → 'mosaic'
✅ Fixed container names: 'documents' → 'knowledge', 'memories' → 'memory'  
✅ Added missing containers: diagrams, code_entities, code_relationships, repositories
✅ Added Golden Node container: golden_nodes with optimized indexing
✅ Updated settings.py with new container configurations
✅ Preserved legacy containers during migration period
            """.strip(),
        }

        self.logger.info("✅ Bicep template updates generated")
        return updates

    async def create_cleanup_script(self) -> str:
        """
        Generate executable cleanup script for development environments.

        Returns Azure CLI script to safely clean up containers.
        """
        self.logger.info("🧹 Creating cleanup script...")

        cleanup_script = """#!/bin/bash
# Cosmos DB Cleanup Script for Golden Node Migration
# DANGER: This script will delete containers and data
# Only run in development environments after proper backups

set -e  # Exit on any error

# Configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-rg-dev}"
COSMOS_ACCOUNT="${COSMOS_ACCOUNT_NAME:-}" 
DATABASE_NAME="mosaic"

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

echo -e "${YELLOW}🧹 Cosmos DB Cleanup Strategy for Golden Node Migration${NC}"
echo -e "${YELLOW}=================================================================${NC}"

# Safety checks
if [ -z "$COSMOS_ACCOUNT" ]; then
    echo -e "${RED}❌ COSMOS_ACCOUNT_NAME environment variable not set${NC}"
    echo "Please set COSMOS_ACCOUNT_NAME before running this script"
    exit 1
fi

if [ "$AZURE_ENVIRONMENT" = "production" ]; then
    echo -e "${RED}❌ DANGER: This script should not be run in production${NC}"
    echo "Current environment: $AZURE_ENVIRONMENT"
    exit 1
fi

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Cosmos Account: $COSMOS_ACCOUNT"
echo "  Database: $DATABASE_NAME"
echo ""

# Confirm before proceeding
read -p "⚠️  This will delete containers and data. Are you sure? (type 'DELETE' to confirm): " confirmation
if [ "$confirmation" != "DELETE" ]; then
    echo -e "${YELLOW}Operation cancelled by user${NC}"
    exit 0
fi

# Function to backup container data
backup_container() {
    local container_name=$1
    local backup_file="backup_${container_name}_$(date +%Y%m%d_%H%M%S).json"
    
    echo -e "${YELLOW}📦 Backing up container: $container_name${NC}"
    
    # Note: This is a placeholder - actual backup would require custom script
    # Azure CLI doesn't have direct export functionality for Cosmos DB
    echo "  Backup file: $backup_file"
    echo "  ⚠️  Manual backup required using Azure portal or custom script"
}

# Function to delete container safely
delete_container() {
    local container_name=$1
    
    echo -e "${YELLOW}🗑️  Deleting container: $container_name${NC}"
    
    if az cosmosdb sql container show \\
        --account-name "$COSMOS_ACCOUNT" \\
        --database-name "$DATABASE_NAME" \\
        --name "$container_name" \\
        --resource-group "$RESOURCE_GROUP" \\
        --output none 2>/dev/null; then
        
        az cosmosdb sql container delete \\
            --account-name "$COSMOS_ACCOUNT" \\
            --database-name "$DATABASE_NAME" \\
            --name "$container_name" \\
            --resource-group "$RESOURCE_GROUP" \\
            --yes
        
        echo -e "${GREEN}  ✅ Container deleted: $container_name${NC}"
    else
        echo -e "${YELLOW}  ⚠️  Container not found: $container_name${NC}"
    fi
}

# Main cleanup process
echo -e "${YELLOW}🔍 Starting cleanup process...${NC}"

# Phase 1: Backup existing containers (if they contain data)
echo -e "${YELLOW}📦 Phase 1: Backup Phase${NC}"
echo "Checking for existing containers to backup..."

# Check if containers exist and backup if needed
containers_to_backup=("knowledge" "memory" "documents" "memories" "libraries" "config")
for container in "${containers_to_backup[@]}"; do
    if az cosmosdb sql container show \\
        --account-name "$COSMOS_ACCOUNT" \\
        --database-name "$DATABASE_NAME" \\
        --name "$container" \\
        --resource-group "$RESOURCE_GROUP" \\
        --output none 2>/dev/null; then
        backup_container "$container"
    fi
done

# Phase 2: Delete old/misnamed containers
echo -e "${YELLOW}🗑️  Phase 2: Cleanup Phase${NC}"
echo "Deleting containers that need to be recreated with correct configuration..."

# Delete misnamed containers (these will be recreated with correct names)
misnamed_containers=("documents" "memories")
for container in "${misnamed_containers[@]}"; do
    delete_container "$container"
done

# Optionally delete old containers that will be replaced by Golden Node
echo -e "${YELLOW}🤔 Optional: Delete containers that will be replaced by Golden Node?${NC}"
echo "These containers might contain data that should be migrated to Golden Node:"
echo "  - code_entities"
echo "  - code_relationships" 
echo "  - repositories"
echo ""
read -p "Delete these containers? (y/N): " delete_optional
if [ "$delete_optional" = "y" ] || [ "$delete_optional" = "Y" ]; then
    optional_containers=("code_entities" "code_relationships" "repositories")
    for container in "${optional_containers[@]}"; do
        delete_container "$container"
    done
fi

# Phase 3: Verification
echo -e "${YELLOW}🔍 Phase 3: Verification${NC}"
echo "Listing remaining containers..."

if az cosmosdb sql container list \\
    --account-name "$COSMOS_ACCOUNT" \\
    --database-name "$DATABASE_NAME" \\
    --resource-group "$RESOURCE_GROUP" \\
    --output table; then
    echo -e "${GREEN}✅ Container listing successful${NC}"
else
    echo -e "${RED}❌ Failed to list containers${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Cleanup process completed!${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Deploy updated infrastructure with 'azd up'"
echo "2. Verify new containers are created correctly"
echo "3. Run data migration script (when available)"
echo "4. Test all application functionality"
echo ""
echo -e "${YELLOW}⚠️  Remember to update your application configuration to use the new container names${NC}"
"""

        self.cleanup_actions.append("cleanup_script_created")
        self.logger.info("✅ Cleanup script created")
        return cleanup_script

    async def execute_strategy(self) -> Dict[str, Any]:
        """
        Execute the complete cleanup strategy.

        Returns comprehensive execution results and next steps.
        """
        self.logger.info("🚀 Executing Cosmos DB cleanup strategy...")

        results = {
            "strategy_execution": {
                "started_at": datetime.now().isoformat(),
                "dry_run": self.dry_run,
                "status": "in_progress",
            },
            "analysis": {},
            "migration_plan": [],
            "generated_artifacts": {},
            "next_steps": [],
            "warnings": [],
            "errors": [],
        }

        try:
            # Step 1: Analyze current configuration
            self.logger.info("Step 1: Analyzing configuration...")
            results["analysis"] = await self.analyze_current_configuration()

            # Step 2: Create migration plan
            self.logger.info("Step 2: Creating migration plan...")
            results["migration_plan"] = await self.create_migration_plan()

            # Step 3: Generate artifacts
            self.logger.info("Step 3: Generating updated configurations...")
            bicep_updates = await self.generate_bicep_updates()
            cleanup_script = await self.create_cleanup_script()

            results["generated_artifacts"] = {
                "updated_bicep_template": bicep_updates["bicep_template"],
                "updated_settings": bicep_updates["settings_additions"],
                "cleanup_script": cleanup_script,
                "summary": bicep_updates["summary"],
            }

            # Step 4: Generate next steps
            results["next_steps"] = [
                {
                    "step": 1,
                    "action": "Review generated artifacts",
                    "description": "Review the updated Bicep template and settings.py changes",
                    "priority": "HIGH",
                },
                {
                    "step": 2,
                    "action": "Backup production data",
                    "description": "Create backups of any existing production data before changes",
                    "priority": "CRITICAL",
                },
                {
                    "step": 3,
                    "action": "Update Bicep template",
                    "description": "Apply the generated Bicep template changes to infra/omnirag/cosmos-omnirag.bicep",
                    "priority": "HIGH",
                },
                {
                    "step": 4,
                    "action": "Update application settings",
                    "description": "Add the new container configurations to src/mosaic/config/settings.py",
                    "priority": "HIGH",
                },
                {
                    "step": 5,
                    "action": "Deploy to development",
                    "description": "Run 'azd up' to deploy the updated infrastructure to development environment",
                    "priority": "MEDIUM",
                },
                {
                    "step": 6,
                    "action": "Test connectivity",
                    "description": "Verify all plugins can connect to their respective containers",
                    "priority": "HIGH",
                },
                {
                    "step": 7,
                    "action": "Test Golden Node operations",
                    "description": "Create, read, update, delete operations with Golden Node models",
                    "priority": "MEDIUM",
                },
                {
                    "step": 8,
                    "action": "Plan data migration",
                    "description": "If existing data needs to be preserved, plan migration to Golden Node schema",
                    "priority": "MEDIUM",
                },
            ]

            # Step 5: Add warnings
            results["warnings"] = [
                "This cleanup strategy will modify existing infrastructure",
                "Container name changes may break existing data queries",
                "Golden Node schema represents a significant architectural change",
                "Backup all production data before executing any changes",
                "Test thoroughly in development environment first",
            ]

            results["strategy_execution"]["status"] = "completed"
            results["strategy_execution"]["completed_at"] = datetime.now().isoformat()

            self.logger.info(
                "✅ Cosmos DB cleanup strategy execution completed successfully"
            )

        except Exception as e:
            results["errors"].append(f"Strategy execution failed: {str(e)}")
            results["strategy_execution"]["status"] = "failed"
            self.logger.error(f"❌ Strategy execution failed: {e}")

        return results


async def main():
    """Main execution function for the cleanup strategy."""
    print("🚀 Cosmos DB Cleanup Strategy for Golden Node Migration")
    print("=" * 60)

    # Initialize strategy (dry run mode for safety)
    strategy = CosmosDBCleanupStrategy(dry_run=True)

    # Execute the complete strategy
    results = await strategy.execute_strategy()

    # Display results
    print("\\n📊 STRATEGY EXECUTION RESULTS")
    print("=" * 40)

    print(f"Status: {results['strategy_execution']['status']}")
    print(
        f"Configuration Issues Found: {len(results['analysis']['configuration_mismatches'])}"
    )
    print(f"Migration Steps: {len(results['migration_plan'])}")
    print(f"Artifacts Generated: {len(results['generated_artifacts'])}")

    print("\\n⚠️  CRITICAL CONFIGURATION MISMATCHES:")
    for issue in results["analysis"]["configuration_mismatches"]:
        if issue["severity"] == "CRITICAL":
            print(f"  - {issue['issue']}: {issue['impact']}")

    print("\\n📋 NEXT STEPS:")
    for step in results["next_steps"][:5]:  # Show first 5 steps
        print(f"  {step['step']}. {step['action']} ({step['priority']})")

    print("\\n⚠️  WARNINGS:")
    for warning in results["warnings"]:
        print(f"  - {warning}")

    print("\\n✅ Strategy artifacts have been generated and are ready for review.")
    print("   Review the generated Bicep template and application settings updates.")
    print("   Ensure proper backups before applying any changes to production.")


# Script execution
if __name__ == "__main__":
    asyncio.run(main())
