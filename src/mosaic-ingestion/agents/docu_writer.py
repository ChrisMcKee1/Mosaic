"""
DocuWriter Agent - AI-Powered Documentation and Enrichment

Handles AI-powered analysis and enrichment of code entities:
- Generate human-readable summaries and descriptions
- Analyze code complexity and quality metrics
- Generate semantic tags and domain concepts
- Assess documentation quality
- Provide testing suggestions
- Generate embeddings for semantic search

This agent populates the ai_enrichment and embedding portions
of Golden Node models with AI-generated insights.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from .base_agent import MosaicAgent, AgentConfig, AgentExecutionContext, AgentError
from ..models.golden_node import GoldenNode, AgentType, AIEnrichment


class DocuWriterAgent(MosaicAgent):
    """
    Specialized agent for AI-powered documentation and enrichment.

    Uses Azure OpenAI to generate comprehensive analysis and documentation
    for code entities, enhancing searchability and understanding.
    """

    def __init__(self, settings=None):
        """Initialize DocuWriter agent with appropriate configuration."""
        config = AgentConfig(
            agent_name="DocuWriter",
            agent_type=AgentType.DOCU_WRITER,
            max_retry_attempts=3,  # AI operations can occasionally fail
            default_timeout_seconds=180,  # 3 minutes for AI processing
            batch_size=15,  # Process multiple entities efficiently
            temperature=0.3,  # Some creativity for better descriptions
            max_tokens=3000,  # Higher token limit for detailed analysis
            enable_parallel_processing=True,  # Process multiple entities in parallel
            log_level="INFO",
        )

        super().__init__(config, settings)
        self.logger = logging.getLogger("mosaic.agent.docu_writer")

    async def _register_plugins(self) -> None:
        """Register DocuWriter-specific Semantic Kernel plugins."""
        # TODO: Register plugins for code analysis, documentation generation
        # Could include complexity analysis, quality assessment plugins
        self.logger.info("DocuWriter agent plugins registered")

    async def process_golden_node(
        self, golden_node: GoldenNode, context: AgentExecutionContext
    ) -> GoldenNode:
        """
        Process Golden Node to generate AI-powered enrichment.

        Args:
            golden_node: The Golden Node to process
            context: Execution context with parameters

        Returns:
            Updated Golden Node with AI enrichment data
        """
        self.logger.info(f"Processing Golden Node {golden_node.id} for AI enrichment")

        try:
            # Generate AI enrichment
            ai_enrichment = await self._generate_ai_enrichment(golden_node, context)

            # Generate embedding for semantic search
            embedding = await self._generate_embedding(golden_node, context)

            # Update the Golden Node
            updated_node = golden_node.model_copy(deep=True)
            updated_node.ai_enrichment = ai_enrichment
            updated_node.embedding = embedding

            # Update processing metadata
            updated_node.processing_metadata.agent_history.append(
                {
                    "agent_type": self.config.agent_type.value,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "execution_id": context.execution_id,
                    "modifications": ["ai_enrichment", "embedding"],
                }
            )

            context.intermediate_results["ai_enrichment"] = {
                "summary_generated": bool(ai_enrichment.summary),
                "purpose_analyzed": bool(ai_enrichment.purpose),
                "complexity_assessed": bool(ai_enrichment.complexity_score),
                "semantic_tags_count": len(ai_enrichment.semantic_tags),
                "domain_concepts_count": len(ai_enrichment.domain_concepts),
                "embedding_dimensions": len(embedding) if embedding else 0,
            }

            self.logger.info(
                f"Successfully enriched Golden Node {golden_node.id} with AI analysis"
            )

            return updated_node

        except Exception as e:
            self.logger.error(f"Failed to process Golden Node {golden_node.id}: {e}")
            raise AgentError(
                f"AI enrichment failed: {e}",
                self.config.agent_type.value,
                context.task_id,
            )

    async def _generate_ai_enrichment(
        self, golden_node: GoldenNode, context: AgentExecutionContext
    ) -> AIEnrichment:
        """
        Generate comprehensive AI enrichment for the code entity.
        """
        self.logger.debug(
            f"Generating AI enrichment for: {golden_node.code_entity.name}"
        )

        try:
            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(golden_node)

            # Get AI analysis using structured output
            analysis_result = await self._get_structured_analysis(
                analysis_prompt, context
            )

            # Create AI enrichment object
            ai_enrichment = AIEnrichment(
                summary=analysis_result.get("summary"),
                purpose=analysis_result.get("purpose"),
                complexity_score=analysis_result.get("complexity_score"),
                semantic_tags=analysis_result.get("semantic_tags", []),
                domain_concepts=analysis_result.get("domain_concepts", []),
                documentation_quality=analysis_result.get("documentation_quality"),
                test_coverage_hints=analysis_result.get("test_coverage_hints", []),
                model_used=self.settings.azure_openai_deployment_name,
                processed_at=datetime.now(timezone.utc),
                confidence_score=analysis_result.get("confidence_score", 0.8),
            )

            return ai_enrichment

        except Exception as e:
            self.logger.error(f"AI enrichment generation failed: {e}")
            # Return minimal enrichment on failure
            return AIEnrichment(
                processed_at=datetime.now(timezone.utc),
                model_used=self.settings.azure_openai_deployment_name,
                confidence_score=0.0,
            )

    def _create_analysis_prompt(self, golden_node: GoldenNode) -> str:
        """Create a comprehensive analysis prompt for the code entity."""
        entity = golden_node.code_entity
        file_context = golden_node.file_context

        prompt = f"""
        Analyze this {entity.language.value} code entity and provide comprehensive insights:

        Entity Details:
        - Name: {entity.name}
        - Type: {entity.entity_type.value}
        - Language: {entity.language.value}
        - File: {file_context.file_path}
        - Lines: {file_context.start_line}-{file_context.end_line}

        Source Code:
        ```{entity.language.value}
        {entity.content}
        ```

        Please provide:
        1. A concise summary (1-2 sentences) of what this code does
        2. The primary purpose/functionality of this entity
        3. Complexity assessment (0.0-1.0 scale, where 1.0 is very complex)
        4. 3-5 semantic tags that describe this code
        5. 2-4 domain concepts this code relates to
        6. Documentation quality score (0.0-1.0, based on comments/docstrings)
        7. 2-3 test coverage suggestions
        8. Your confidence in this analysis (0.0-1.0)

        Focus on accuracy and usefulness for developers who might search for or work with this code.
        """

        return prompt

    async def _get_structured_analysis(
        self, prompt: str, context: AgentExecutionContext
    ) -> Dict[str, Any]:
        """
        Get structured analysis from Azure OpenAI.

        This would use the chat_with_llm method with structured output.
        """
        try:
            # TODO: Implement structured output with Pydantic model
            # This would define a response schema and use chat_with_llm

            # For now, simulate a structured response
            analysis_result = {
                "summary": f"Code entity analysis for {context.golden_node_id}",
                "purpose": "Handles specific functionality within the codebase",
                "complexity_score": 0.5,
                "semantic_tags": ["function", "utility", "helper"],
                "domain_concepts": ["data processing", "business logic"],
                "documentation_quality": 0.7,
                "test_coverage_hints": [
                    "Test edge cases and error conditions",
                ],
                "confidence_score": 0.8,
            }

            return analysis_result

        except Exception as e:
            self.logger.error(f"Structured analysis failed: {e}")
            return {}

    async def _generate_embedding(
        self, golden_node: GoldenNode, context: AgentExecutionContext
    ) -> Optional[List[float]]:
        """
        Generate embedding vector for semantic search.
        """
        try:
            # Create text for embedding
            embedding_text = self._create_embedding_text(golden_node)

            # Generate embedding using the base class method
            embedding = await self.generate_embedding(embedding_text, context)

            return embedding

        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return None

    def _create_embedding_text(self, golden_node: GoldenNode) -> str:
        """
        Create optimized text for embedding generation.

        Combines entity name, type, purpose, and key content.
        """
        entity = golden_node.code_entity

        # Start with basic entity information
        embedding_parts = [
            f"Name: {entity.name}",
            f"Type: {entity.entity_type.value}",
            f"Language: {entity.language.value}",
        ]

        # Add signature if available
        if entity.signature:
            embedding_parts.append(f"Signature: {entity.signature}")

        # Add AI-generated summary if available
        if golden_node.ai_enrichment and golden_node.ai_enrichment.summary:
            embedding_parts.append(f"Summary: {golden_node.ai_enrichment.summary}")

        # Add key parts of the content (truncated for optimal embedding)
        content_preview = entity.content[:500]  # First 500 characters
        embedding_parts.append(f"Content: {content_preview}")

        return " | ".join(embedding_parts)

    async def batch_enrich_entities(
        self, golden_nodes: List[GoldenNode], batch_parameters: Dict[str, Any]
    ) -> List[GoldenNode]:
        """
        Efficiently process multiple entities in batch for better performance.
        """
        self.logger.info(f"Batch enriching {len(golden_nodes)} entities")

        # TODO: Implement batch processing optimization
        # This would group similar entities and use batch API calls where possible

        enriched_nodes = []
        for node in golden_nodes:
            # Create context for each node
            context = AgentExecutionContext(
                execution_id=str(node.id),
                started_at=datetime.now(timezone.utc),
                timeout_at=datetime.now(timezone.utc).replace(
                    second=datetime.now(timezone.utc).second
                    + self.config.default_timeout_seconds
                ),
                golden_node_id=node.id,
                parameters=batch_parameters,
            )

            try:
                enriched_node = await self.process_golden_node(node, context)
                enriched_nodes.append(enriched_node)
            except Exception as e:
                self.logger.error(f"Failed to enrich node {node.id}: {e}")
                enriched_nodes.append(node)  # Return original on failure

        return enriched_nodes
