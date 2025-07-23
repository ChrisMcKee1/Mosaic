"""
Base Agent Class with Semantic Kernel Integration

Provides the foundational agent architecture for the Mosaic Ingestion Service.
All specialized agents inherit from MosaicAgent to ensure consistency and
proper integration with Azure services and Semantic Kernel.

Key Features:
- Semantic Kernel integration for LLM capabilities
- Azure OpenAI service connectivity
- Golden Node model compatibility
- Structured error handling and retry logic
- Telemetry and performance monitoring
- Function calling support for complex operations

Based on Microsoft Docs MCP research:
- Semantic Kernel Python agents plugins integration patterns
- Azure OpenAI chat completion and embedding services
- Function calling and structured outputs implementation
- Agent orchestration and task coordination best practices
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Type, Union
from uuid import uuid4
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, ConfigDict
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.functions import KernelArguments
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.exceptions import ServiceInvalidExecutionSettingsError

# Import models and settings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from mosaic.config.settings import MosaicSettings
from ..models.golden_node import (
    GoldenNode,
    AgentTask,
    AgentResult,
    AgentType,
    ProcessingStatus,
)


class AgentError(Exception):
    """Base exception for agent errors."""

    def __init__(self, message: str, agent_type: str, task_id: Optional[str] = None):
        self.message = message
        self.agent_type = agent_type
        self.task_id = task_id
        super().__init__(message)


class AgentTimeout(AgentError):
    """Exception raised when agent processing times out."""

    pass


class AgentConfig(BaseModel):
    """Configuration for agent behavior and performance."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Agent identity
    agent_name: str = Field(
        description="Human-readable agent name", min_length=1, max_length=64
    )
    agent_type: AgentType = Field(description="Agent type classification")

    # Processing limits
    max_retry_attempts: int = Field(
        default=3,
        description="Maximum retry attempts for failed operations",
        ge=0,
        le=5,
    )
    default_timeout_seconds: int = Field(
        default=300, description="Default task timeout in seconds", gt=0, le=3600
    )
    batch_size: int = Field(
        default=10, description="Number of items to process in batch", gt=0, le=100
    )

    # LLM configuration
    temperature: float = Field(
        default=0.1, description="LLM temperature for consistency", ge=0.0, le=2.0
    )
    max_tokens: int = Field(
        default=2000, description="Maximum tokens per LLM response", gt=0, le=8000
    )

    # Performance settings
    enable_parallel_processing: bool = Field(
        default=True, description="Enable parallel processing when possible"
    )
    log_level: str = Field(default="INFO", description="Logging level for this agent")


class AgentExecutionContext(BaseModel):
    """Context information for agent execution."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Execution identity
    execution_id: str = Field(description="Unique execution identifier", min_length=1)
    task_id: Optional[str] = Field(
        default=None, description="Associated task ID if applicable"
    )

    # Timing
    started_at: datetime = Field(description="Execution start time")
    timeout_at: datetime = Field(description="Execution timeout deadline")

    # Data context
    golden_node_id: Optional[str] = Field(
        default=None, description="Golden Node being processed"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Execution parameters"
    )

    # State tracking
    retry_count: int = Field(default=0, description="Current retry attempt", ge=0)
    intermediate_results: Dict[str, Any] = Field(
        default_factory=dict, description="Intermediate processing results"
    )


class MosaicAgent(ABC):
    """
    Base agent class with Semantic Kernel integration.

    Provides common functionality for all specialized agents in the
    Mosaic Ingestion Service. Handles Azure service connectivity,
    error management, logging, and LLM interactions.
    """

    def __init__(self, config: AgentConfig, settings: Optional[MosaicSettings] = None):
        """Initialize the base agent."""
        self.config = config
        self.settings = settings or MosaicSettings()

        # Set up logging
        self.logger = logging.getLogger(f"mosaic.agent.{config.agent_name}")
        self.logger.setLevel(getattr(logging, config.log_level.upper()))

        # Initialize Semantic Kernel components
        self.kernel: Optional[Kernel] = None
        self.chat_service: Optional[AzureChatCompletion] = None
        self.embedding_service: Optional[AzureTextEmbedding] = None

        # Agent state
        self._initialized = False
        self._active_executions: Dict[str, AgentExecutionContext] = {}

        self.logger.info(f"Agent {self.config.agent_name} created")

    async def initialize(self) -> None:
        """Initialize Semantic Kernel and Azure services."""
        if self._initialized:
            return

        try:
            # Create and configure kernel
            self.kernel = Kernel()

            # Add Azure OpenAI chat completion service
            self.chat_service = AzureChatCompletion(
                service_id="azure_openai_chat",
                deployment_name=self.settings.azure_openai_chat_deployment_name,
                endpoint=self.settings.azure_openai_endpoint,
                api_version=self.settings.azure_openai_api_version,
            )
            self.kernel.add_service(self.chat_service)

            # Add Azure OpenAI embedding service
            self.embedding_service = AzureTextEmbedding(
                service_id="azure_openai_embedding",
                deployment_name=self.settings.azure_openai_text_embedding_deployment_name,
                endpoint=self.settings.azure_openai_endpoint,
                api_version=self.settings.azure_openai_api_version,
            )
            self.kernel.add_service(self.embedding_service)

            # Register agent-specific plugins
            await self._register_plugins()

            self._initialized = True
            self.logger.info(f"Agent {self.config.agent_name} initialized successfully")

        except Exception as e:
            self.logger.error(
                f"Failed to initialize agent {self.config.agent_name}: {e}"
            )
            raise AgentError(
                f"Agent initialization failed: {e}", self.config.agent_type.value
            )

    @abstractmethod
    async def _register_plugins(self) -> None:
        """Register agent-specific Semantic Kernel plugins."""
        pass

    @abstractmethod
    async def process_golden_node(
        self, golden_node: GoldenNode, context: AgentExecutionContext
    ) -> GoldenNode:
        """
        Process a Golden Node and return the modified version.

        This is the main processing method that each agent must implement.
        """
        pass

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """
        Execute an agent task with full error handling and monitoring.

        This method provides the standardized task execution framework
        that all agents use for consistency and reliability.
        """
        execution_id = str(uuid4())
        started_at = datetime.now(timezone.utc)
        timeout_at = started_at.replace(second=started_at.second + task.timeout_seconds)

        # Create execution context
        context = AgentExecutionContext(
            execution_id=execution_id,
            task_id=task.task_id,
            started_at=started_at,
            timeout_at=timeout_at,
            golden_node_id=task.golden_node_id,
            parameters=task.parameters,
        )

        self._active_executions[execution_id] = context

        try:
            self.logger.info(
                f"Starting task execution: {task.task_id} "
                f"for Golden Node: {task.golden_node_id}"
            )

            # Load the Golden Node (this would connect to Cosmos DB)
            golden_node = await self._load_golden_node(task.golden_node_id)
            if golden_node is None:
                raise AgentError(
                    f"Golden Node not found: {task.golden_node_id}",
                    self.config.agent_type.value,
                    task.task_id,
                )

            # Execute with timeout and retry logic
            result_node = await self._execute_with_retry(golden_node, context)

            # Save the updated Golden Node
            await self._save_golden_node(result_node)

            # Create successful result
            completed_at = datetime.now(timezone.utc)
            processing_time = (completed_at - started_at).total_seconds() * 1000

            result = AgentResult(
                task_id=task.task_id,
                agent_type=self.config.agent_type,
                status=ProcessingStatus.COMPLETED,
                output_data=context.intermediate_results,
                modified_fields=self._detect_modified_fields(golden_node, result_node),
                started_at=started_at,
                completed_at=completed_at,
                processing_time_ms=int(processing_time),
            )

            self.logger.info(
                f"Task completed successfully: {task.task_id} "
                f"in {processing_time:.1f}ms"
            )

            return result

        except AgentTimeout as e:
            self.logger.error(f"Task timed out: {task.task_id} - {e}")
            return self._create_error_result(
                task, started_at, str(e), retry_suggested=True
            )

        except AgentError as e:
            self.logger.error(f"Task failed: {task.task_id} - {e}")
            return self._create_error_result(
                task, started_at, str(e), retry_suggested=False
            )

        except Exception as e:
            self.logger.error(f"Unexpected error in task: {task.task_id} - {e}")
            return self._create_error_result(
                task, started_at, f"Unexpected error: {e}", retry_suggested=True
            )

        finally:
            # Clean up execution context
            self._active_executions.pop(execution_id, None)

    async def _execute_with_retry(
        self, golden_node: GoldenNode, context: AgentExecutionContext
    ) -> GoldenNode:
        """Execute the main processing with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retry_attempts + 1):
            try:
                context.retry_count = attempt

                # Check timeout
                if datetime.now(timezone.utc) >= context.timeout_at:
                    raise AgentTimeout(
                        f"Task timed out after {attempt} attempts",
                        self.config.agent_type.value,
                        context.task_id,
                    )

                # Execute the main processing
                result = await self.process_golden_node(golden_node, context)

                # Success - return result
                return result

            except AgentTimeout:
                # Don't retry timeouts
                raise

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retry_attempts:
                    wait_time = 2**attempt  # Exponential backoff
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(
                        f"All {self.config.max_retry_attempts + 1} attempts failed"
                    )

        # All retries exhausted
        raise AgentError(
            f"Max retries exceeded. Last error: {last_error}",
            self.config.agent_type.value,
            context.task_id,
        )

    async def chat_with_llm(
        self,
        messages: Union[str, List[Dict[str, str]]],
        context: AgentExecutionContext,
        system_prompt: Optional[str] = None,
        structured_output: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """
        Chat with Azure OpenAI using Semantic Kernel.

        Supports both simple string responses and structured Pydantic outputs.
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Create execution settings
            settings = AzureChatPromptExecutionSettings(
                service_id="azure_openai_chat",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            # Set up structured output if requested
            if structured_output:
                settings.response_format = structured_output

            # Create chat history
            chat_history = ChatHistory()

            # Add system message if provided
            if system_prompt:
                chat_history.add_system_message(system_prompt)

            # Add user messages
            if isinstance(messages, str):
                chat_history.add_user_message(messages)
            else:
                for msg in messages:
                    if msg.get("role") == "user":
                        chat_history.add_user_message(msg["content"])
                    elif msg.get("role") == "assistant":
                        chat_history.add_assistant_message(msg["content"])

            # Execute chat completion
            response = await self.chat_service.get_chat_message_contents(
                chat_history=chat_history,
                settings=settings,
                kernel=self.kernel,
                arguments=KernelArguments(),
            )

            if not response:
                raise AgentError(
                    "Empty response from Azure OpenAI",
                    self.config.agent_type.value,
                    context.task_id,
                )

            content = response[0].content

            # Parse structured output if requested
            if structured_output:
                try:
                    import json

                    parsed_data = json.loads(content)
                    return structured_output.model_validate(parsed_data)
                except (json.JSONDecodeError, ValueError) as e:
                    raise AgentError(
                        f"Failed to parse structured output: {e}",
                        self.config.agent_type.value,
                        context.task_id,
                    )

            return content

        except ServiceInvalidExecutionSettingsError as e:
            raise AgentError(
                f"Invalid LLM execution parameters: {e}",
                self.config.agent_type.value,
                context.task_id,
            )

        except Exception as e:
            raise AgentError(
                f"LLM chat failed: {e}", self.config.agent_type.value, context.task_id
            )

    async def generate_embedding(
        self, text: str, context: AgentExecutionContext
    ) -> List[float]:
        """Generate embedding vector for text using Azure OpenAI."""
        if not self._initialized:
            await self.initialize()

        try:
            embeddings = await self.embedding_service.generate_embeddings([text])
            if not embeddings or len(embeddings) == 0:
                raise AgentError(
                    "Empty embedding response",
                    self.config.agent_type.value,
                    context.task_id,
                )

            return embeddings[0]

        except Exception as e:
            raise AgentError(
                f"Embedding generation failed: {e}",
                self.config.agent_type.value,
                context.task_id,
            )

    async def _load_golden_node(self, golden_node_id: str) -> Optional[GoldenNode]:
        """Load Golden Node from Cosmos DB."""
        # TODO: Implement Cosmos DB integration
        # This is a placeholder that would be implemented with actual Cosmos DB queries
        self.logger.debug(f"Loading Golden Node: {golden_node_id}")
        return None

    async def _save_golden_node(self, golden_node: GoldenNode) -> None:
        """Save Golden Node to Cosmos DB."""
        # TODO: Implement Cosmos DB integration
        # This is a placeholder that would be implemented with actual Cosmos DB operations
        self.logger.debug(f"Saving Golden Node: {golden_node.id}")

    def _detect_modified_fields(
        self, original: GoldenNode, modified: GoldenNode
    ) -> List[str]:
        """Detect which fields were modified during processing."""
        modified_fields = []

        # Compare key fields that agents typically modify
        if original.ai_enrichment != modified.ai_enrichment:
            modified_fields.append("ai_enrichment")

        if original.embedding != modified.embedding:
            modified_fields.append("embedding")

        if original.relationships != modified.relationships:
            modified_fields.append("relationships")

        if original.dependency_graph != modified.dependency_graph:
            modified_fields.append("dependency_graph")

        if original.tags != modified.tags:
            modified_fields.append("tags")

        return modified_fields

    def _create_error_result(
        self,
        task: AgentTask,
        started_at: datetime,
        error_message: str,
        retry_suggested: bool = False,
    ) -> AgentResult:
        """Create an error result for failed task execution."""
        completed_at = datetime.now(timezone.utc)
        processing_time = (completed_at - started_at).total_seconds() * 1000

        return AgentResult(
            task_id=task.task_id,
            agent_type=self.config.agent_type,
            status=ProcessingStatus.FAILED,
            output_data={},
            modified_fields=[],
            started_at=started_at,
            completed_at=completed_at,
            processing_time_ms=int(processing_time),
            error_message=error_message,
            retry_suggested=retry_suggested,
        )

    @asynccontextmanager
    async def execution_context(self, task: AgentTask):
        """Context manager for safe agent execution."""
        execution_id = str(uuid4())
        started_at = datetime.now(timezone.utc)
        timeout_at = started_at.replace(second=started_at.second + task.timeout_seconds)

        context = AgentExecutionContext(
            execution_id=execution_id,
            task_id=task.task_id,
            started_at=started_at,
            timeout_at=timeout_at,
            golden_node_id=task.golden_node_id,
            parameters=task.parameters,
        )

        self._active_executions[execution_id] = context

        try:
            yield context
        finally:
            self._active_executions.pop(execution_id, None)

    async def cleanup(self) -> None:
        """Clean up agent resources."""
        self.logger.info(f"Cleaning up agent {self.config.agent_name}")

        # Cancel any active executions
        if self._active_executions:
            self.logger.warning(
                f"Cancelling {len(self._active_executions)} active executions"
            )
            self._active_executions.clear()

        self._initialized = False

    def get_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        return {
            "agent_name": self.config.agent_name,
            "agent_type": self.config.agent_type.value,
            "initialized": self._initialized,
            "active_executions": len(self._active_executions),
            "config": self.config.model_dump(),
        }
