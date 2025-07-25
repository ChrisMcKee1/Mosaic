"""
Mosaic AI Agent Orchestrator
Implements Microsoft Semantic Kernel Magentic orchestration for repository ingestion
"""

import logging
from typing import Dict, Any
from datetime import datetime

from semantic_kernel.agents import StandardMagenticManager, MagenticOrchestration
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel import Kernel

logger = logging.getLogger(__name__)


class MosaicMagenticOrchestrator:
    """
    Microsoft Semantic Kernel Magentic orchestration for repository ingestion.

    Uses the official StandardMagenticManager to coordinate specialized agents:
    1. GitSleuth - Repository cloning and analysis
    2. CodeParser - AST parsing and entity extraction
    3. GraphArchitect - Relationship mapping and graph construction
    4. DocuWriter - AI-powered enrichment and documentation
    5. GraphAuditor - Quality assurance and validation

    The manager dynamically selects which agent to invoke based on context.
    """

    def __init__(self, settings):
        """Initialize the Magentic orchestrator with all agents."""
        self.settings = settings
        self.runtime = None
        self.orchestration = None

        # Statistics
        self.stats = {
            "repositories_processed": 0,
            "golden_nodes_created": 0,
            "agents_executed": 0,
            "errors_handled": 0,
        }

        # Initialize orchestration
        self._initialize_magentic_orchestration()

    def _initialize_magentic_orchestration(self):
        """Initialize the official Semantic Kernel Magentic orchestration."""
        try:
            # Create Azure OpenAI chat completion service for the manager
            chat_service = AzureChatCompletion(
                service_id="azure_openai_chat",
                deployment_name=self.settings.azure_openai_chat_deployment_name,
                endpoint=self.settings.azure_openai_endpoint,
                api_version=self.settings.azure_openai_api_version,
            )

            # Create the Magentic Manager (the orchestrating brain)
            manager = StandardMagenticManager(chat_completion_service=chat_service)

            # Create specialized agent team
            agents = self._create_specialized_agents(chat_service)

            # Create the Magentic Orchestration
            self.orchestration = MagenticOrchestration(
                members=agents,
                manager=manager,
                agent_response_callback=self._agent_response_callback,
            )

            # Create and start runtime
            self.runtime = InProcessRuntime()
            self.runtime.start()

            logger.info(
                f"✅ Magentic orchestration initialized with {len(agents)} specialized agents"
            )

        except Exception as e:
            logger.error(f"❌ Failed to initialize Magentic orchestration: {e}")
            raise

    def _create_specialized_agents(self, chat_service) -> list:
        """Create the specialized agents for repository ingestion."""

        # Agent 1: GitSleuth - Repository cloning and analysis
        git_sleuth = ChatCompletionAgent(
            service_id="git_sleuth",
            kernel=Kernel(),
            name="GitSleuth",
            instructions="""
            You are GitSleuth, the repository analysis specialist.
            
            Your responsibilities:
            - Clone Git repositories using the provided URL and branch
            - Analyze repository structure and file organization
            - Extract git metadata (commits, branches, contributors)
            - Identify programming languages and project structure
            - Provide file listings for downstream processing
            
            Always respond with structured information about:
            - Repository clone status and path
            - Total files found and their types
            - Languages detected
            - Repository metadata (size, last commit, etc.)
            
            Be thorough but concise in your analysis.
            """,
            execution_settings=None,
        )

        # Agent 2: CodeParser - AST parsing and entity extraction
        code_parser = ChatCompletionAgent(
            service_id="code_parser",
            kernel=Kernel(),
            name="CodeParser",
            instructions="""
            You are CodeParser, the multi-language AST parsing specialist.
            
            Your responsibilities:
            - Parse source code files using tree-sitter for 11+ languages
            - Extract code entities: functions, classes, methods, modules, imports
            - Generate Golden Node entities with proper structure
            - Handle Python, JavaScript, TypeScript, Java, Go, Rust, C/C++, C#, HTML, CSS
            - Create fully qualified names (FQNs) for all entities
            
            For each file, extract:
            - Functions with signatures and docstrings
            - Classes with inheritance and methods
            - Imports and dependencies
            - Module-level entities
            
            Output Golden Node entities ready for graph storage.
            """,
            execution_settings=None,
        )

        # Agent 3: GraphArchitect - Relationship mapping
        graph_architect = ChatCompletionAgent(
            service_id="graph_architect",
            kernel=Kernel(),
            name="GraphArchitect",
            instructions="""
            You are GraphArchitect, the code relationship mapping specialist.
            
            Your responsibilities:
            - Analyze Golden Node entities and map their relationships
            - Identify imports, function calls, inheritance, composition
            - Create cross-file dependency relationships
            - Build the knowledge graph structure for OmniRAG
            - Map semantic relationships between code entities
            
            Generate relationship types:
            - imports, calls, inherits, contains, depends_on
            - file_to_module, function_to_class relationships
            - Cross-language bindings and interfaces
            
            Ensure all relationships have proper source/target Golden Node IDs.
            """,
            execution_settings=None,
        )

        # Agent 4: DocuWriter - AI-powered enrichment
        docu_writer = ChatCompletionAgent(
            service_id="docu_writer",
            kernel=Kernel(),
            name="DocuWriter",
            instructions="""
            You are DocuWriter, the AI-powered code enrichment specialist.
            
            Your responsibilities:
            - Generate intelligent summaries for code entities
            - Add complexity scores and quality metrics
            - Create semantic tags and categorizations
            - Generate embeddings-ready descriptions
            - Enhance Golden Nodes with AI insights
            
            For each entity, provide:
            - Concise but comprehensive summaries
            - Complexity and maintainability scores
            - Functional categorization tags
            - Semantic descriptions for embedding generation
            - Usage patterns and best practices
            
            Make code searchable and understandable through AI enrichment.
            """,
            execution_settings=None,
        )

        # Agent 5: GraphAuditor - Quality assurance
        graph_auditor = ChatCompletionAgent(
            service_id="graph_auditor",
            kernel=Kernel(),
            name="GraphAuditor",
            instructions="""
            You are GraphAuditor, the quality assurance and validation specialist.
            
            Your responsibilities:
            - Validate Golden Node entity completeness and accuracy
            - Check relationship integrity and consistency
            - Audit data quality for Cosmos DB storage
            - Ensure OmniRAG graph database compliance
            - Generate quality scores and recommendations
            
            Validation checks:
            - Golden Node schema compliance
            - Relationship referential integrity
            - Embedding vector readiness
            - Graph connectivity and structure
            - Data completeness and consistency
            
            Provide quality scores and actionable improvement recommendations.
            """,
            execution_settings=None,
        )

        return [git_sleuth, code_parser, graph_architect, docu_writer, graph_auditor]

    def _agent_response_callback(self, message: ChatMessageContent) -> None:
        """Callback to observe agent responses during orchestration."""
        self.stats["agents_executed"] += 1
        logger.info(
            f"🤖 **{message.name}**:\n{message.content[:200]}..."
        )  # Log first 200 chars

    async def orchestrate_repository_ingestion(
        self, repository_url: str, branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Orchestrate complete repository ingestion using Microsoft Semantic Kernel Magentic orchestration.

        The StandardMagenticManager will coordinate the specialized agents to:
        1. Clone and analyze the repository
        2. Parse code and extract entities
        3. Map relationships and build graph
        4. Enrich with AI-powered insights
        5. Validate and store in Cosmos DB

        Args:
            repository_url: Git repository URL
            branch: Git branch to process

        Returns:
            Comprehensive ingestion results with Golden Node entities
        """
        start_time = datetime.utcnow()
        logger.info(
            f"🚀 Starting Magentic orchestration for: {repository_url} (branch: {branch})"
        )

        try:
            # Create the task prompt for the Magentic Manager
            task_prompt = f"""
            Perform a comprehensive repository ingestion for the following Git repository:
            
            Repository: {repository_url}
            Branch: {branch}
            
            Complete the following workflow by coordinating the specialized agents:
            
            1. **GitSleuth**: Clone the repository and analyze its structure
               - Extract file listings and detect programming languages
               - Provide metadata about repository size and organization
               
            2. **CodeParser**: Parse all source code files using AST analysis
               - Extract functions, classes, modules, and other code entities
               - Generate Golden Node entities with proper structure
               - Handle multiple programming languages (Python, JS, TS, Java, Go, etc.)
               
            3. **GraphArchitect**: Map relationships between code entities
               - Identify imports, function calls, inheritance patterns
               - Build the knowledge graph structure for OmniRAG storage
               - Create cross-file dependency relationships
               
            4. **DocuWriter**: Enrich entities with AI-powered insights
               - Generate intelligent summaries and complexity scores
               - Add semantic tags and categorizations
               - Prepare descriptions for embedding generation
               
            5. **GraphAuditor**: Validate data quality and completeness
               - Ensure Golden Node schema compliance
               - Check relationship integrity
               - Generate quality scores and recommendations
               
            The final output should be Golden Node entities ready for Cosmos DB storage
            in the OmniRAG knowledge graph database.
            
            Repository URL: {repository_url}
            Target Branch: {branch}
            """

            # Invoke the Magentic orchestration
            logger.info("🎯 Invoking Magentic orchestration...")
            orchestration_result = await self.orchestration.invoke(task_prompt)

            # Wait for completion and get results
            final_result = await orchestration_result.get()

            # Compile orchestration statistics
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()

            # Extract relevant metrics from the orchestration result
            summary = {
                "repository_url": repository_url,
                "branch": branch,
                "status": "completed",
                "processing_time_seconds": processing_time,
                "orchestration_result": str(final_result)[:500],  # First 500 chars
                "agents_executed": self.stats["agents_executed"],
                "timestamp": end_time.isoformat(),
                "mode": "magentic_orchestration",
                "framework": "microsoft_semantic_kernel",
            }

            self.stats["repositories_processed"] += 1
            logger.info(
                f"✅ Magentic orchestration completed successfully in {processing_time:.2f}s"
            )

            return summary

        except Exception as e:
            logger.error(f"❌ Magentic orchestration failed: {e}")
            self.stats["errors_handled"] += 1
            raise

    async def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get comprehensive Magentic orchestration statistics."""
        return {
            "orchestrator_type": "microsoft_semantic_kernel_magentic",
            "orchestrator_stats": self.stats,
            "agents_available": [
                "GitSleuth",
                "CodeParser",
                "GraphArchitect",
                "DocuWriter",
                "GraphAuditor",
            ],
            "orchestration_healthy": self.orchestration is not None,
            "runtime_active": self.runtime is not None,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def cleanup(self) -> None:
        """Cleanup resources and stop the runtime."""
        try:
            if self.runtime:
                await self.runtime.stop_when_idle()
                logger.info("✅ Magentic orchestration runtime stopped")

            self.orchestration = None
            self.runtime = None

            logger.info("✅ Magentic orchestration cleanup completed")

        except Exception as e:
            logger.error(f"❌ Error during Magentic orchestration cleanup: {e}")
            raise
