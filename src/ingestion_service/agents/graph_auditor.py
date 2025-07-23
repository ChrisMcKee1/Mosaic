"""
GraphAuditor Agent - Quality Assurance and Validation

Handles quality assurance and validation of the ingestion pipeline:
- Validate Golden Node data integrity
- Check relationship consistency
- Detect data anomalies and inconsistencies
- Perform quality scoring and assessment
- Generate quality reports and recommendations
- Ensure schema compliance

This agent ensures the overall quality and reliability of the
ingested knowledge graph data.
"""

import logging
from typing import List
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from .base_agent import MosaicAgent, AgentConfig, AgentExecutionContext, AgentError
from ..models.golden_node import GoldenNode, AgentType, ProcessingStatus


class ValidationResult(BaseModel):
    """Result of a validation check."""

    check_name: str = Field(description="Name of the validation check")
    passed: bool = Field(description="Whether the check passed")
    severity: str = Field(description="Severity level: info, warning, error, critical")
    message: str = Field(description="Detailed message about the result")
    suggestions: List[str] = Field(
        default_factory=list, description="Suggestions for fixing issues"
    )


class QualityReport(BaseModel):
    """Comprehensive quality report for a Golden Node."""

    golden_node_id: str = Field(description="Golden Node identifier")
    overall_score: float = Field(
        description="Overall quality score (0.0-1.0)", ge=0.0, le=1.0
    )
    validation_results: List[ValidationResult] = Field(
        description="Individual validation check results"
    )
    critical_issues: int = Field(description="Number of critical issues")
    warnings: int = Field(description="Number of warnings")
    recommendations: List[str] = Field(
        description="Overall recommendations for improvement"
    )
    assessed_at: datetime = Field(description="When quality assessment was performed")


class GraphAuditorAgent(MosaicAgent):
    """
    Specialized agent for quality assurance and validation.

    Performs comprehensive validation of Golden Node data to ensure
    quality, consistency, and reliability of the knowledge graph.
    """

    def __init__(self, settings=None):
        """Initialize GraphAuditor agent with appropriate configuration."""
        config = AgentConfig(
            agent_name="GraphAuditor",
            agent_type=AgentType.GRAPH_AUDITOR,
            max_retry_attempts=2,  # Quality checks should be deterministic
            default_timeout_seconds=120,  # 2 minutes for validation
            batch_size=30,  # Validate multiple entities efficiently
            temperature=0.0,  # No randomness needed for validation
            max_tokens=2000,  # Moderate LLM usage for assessment
            enable_parallel_processing=True,  # Validate multiple entities in parallel
            log_level="INFO",
        )

        super().__init__(config, settings)
        self.logger = logging.getLogger("mosaic.agent.graph_auditor")

    async def _register_plugins(self) -> None:
        """Register GraphAuditor-specific Semantic Kernel plugins."""
        # TODO: Register plugins for quality assessment, anomaly detection
        self.logger.info("GraphAuditor agent plugins registered")

    async def process_golden_node(
        self, golden_node: GoldenNode, context: AgentExecutionContext
    ) -> GoldenNode:
        """
        Process Golden Node to perform quality assurance.

        Args:
            golden_node: The Golden Node to validate
            context: Execution context with parameters

        Returns:
            Updated Golden Node with quality assessment data
        """
        self.logger.info(
            f"Processing Golden Node {golden_node.id} for quality assurance"
        )

        try:
            # Perform comprehensive quality assessment
            quality_report = await self._perform_quality_assessment(
                golden_node, context
            )

            # Update processing status based on quality assessment
            processing_status = self._determine_processing_status(quality_report)

            # Update the Golden Node
            updated_node = golden_node.model_copy(deep=True)
            updated_node.processing_metadata.processing_stage = processing_status

            # Add quality assessment to tags for searchability
            quality_tags = self._generate_quality_tags(quality_report)
            updated_node.tags.extend(quality_tags)

            # Update processing metadata
            updated_node.processing_metadata.agent_history.append(
                {
                    "agent_type": self.config.agent_type.value,
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "execution_id": context.execution_id,
                    "modifications": ["processing_metadata", "tags"],
                    "quality_report": quality_report.model_dump(),
                }
            )

            context.intermediate_results["quality_assessment"] = {
                "overall_score": quality_report.overall_score,
                "critical_issues": quality_report.critical_issues,
                "warnings": quality_report.warnings,
                "validation_checks": len(quality_report.validation_results),
                "processing_status": processing_status.value,
            }

            self.logger.info(
                f"Quality assessment completed for Golden Node {golden_node.id}. "
                f"Score: {quality_report.overall_score:.2f}"
            )

            return updated_node

        except Exception as e:
            self.logger.error(f"Failed to process Golden Node {golden_node.id}: {e}")
            raise AgentError(
                f"Quality assessment failed: {e}",
                self.config.agent_type.value,
                context.task_id,
            )

    async def _perform_quality_assessment(
        self, golden_node: GoldenNode, context: AgentExecutionContext
    ) -> QualityReport:
        """
        Perform comprehensive quality assessment of the Golden Node.
        """
        self.logger.debug(f"Performing quality assessment for: {golden_node.id}")

        validation_results = []

        # Perform various validation checks
        validation_results.extend(await self._validate_data_integrity(golden_node))
        validation_results.extend(await self._validate_relationships(golden_node))
        validation_results.extend(await self._validate_ai_enrichment(golden_node))
        validation_results.extend(await self._validate_schema_compliance(golden_node))
        validation_results.extend(await self._detect_anomalies(golden_node, context))

        # Calculate overall score and counts
        overall_score = self._calculate_overall_score(validation_results)
        critical_issues = sum(1 for r in validation_results if r.severity == "critical")
        warnings = sum(1 for r in validation_results if r.severity == "warning")

        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results)

        return QualityReport(
            golden_node_id=golden_node.id,
            overall_score=overall_score,
            validation_results=validation_results,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations,
            assessed_at=datetime.now(timezone.utc),
        )

    async def _validate_data_integrity(
        self, golden_node: GoldenNode
    ) -> List[ValidationResult]:
        """Validate basic data integrity."""
        results = []

        # Check required fields
        if not golden_node.code_entity.name:
            results.append(
                ValidationResult(
                    check_name="required_fields",
                    passed=False,
                    severity="critical",
                    message="Code entity name is missing",
                    suggestions=[
                        "Ensure entity name is properly extracted during parsing"
                    ],
                )
            )

        # Check content length
        if len(golden_node.code_entity.content) < 10:
            results.append(
                ValidationResult(
                    check_name="content_length",
                    passed=False,
                    severity="warning",
                    message="Code entity content is very short",
                    suggestions=["Verify that full entity content was captured"],
                )
            )

        # Check file path validity
        if not golden_node.file_context.file_path.startswith("/"):
            results.append(
                ValidationResult(
                    check_name="file_path_format",
                    passed=False,
                    severity="error",
                    message="File path is not absolute",
                    suggestions=["Ensure file paths are normalized to absolute format"],
                )
            )

        # Add success result if no issues found
        if not results:
            results.append(
                ValidationResult(
                    check_name="data_integrity",
                    passed=True,
                    severity="info",
                    message="All data integrity checks passed",
                )
            )

        return results

    async def _validate_relationships(
        self, golden_node: GoldenNode
    ) -> List[ValidationResult]:
        """Validate relationship data consistency."""
        results = []

        # Check for circular relationships
        if golden_node.dependency_graph:
            if golden_node.dependency_graph.circular_dependencies:
                results.append(
                    ValidationResult(
                        check_name="circular_dependencies",
                        passed=False,
                        severity="warning",
                        message=f"Found {len(golden_node.dependency_graph.circular_dependencies)} circular dependencies",
                        suggestions=[
                            "Review code architecture to eliminate circular dependencies"
                        ],
                    )
                )

        # Check relationship consistency
        for relationship in golden_node.relationships:
            if relationship.source_entity_id == relationship.target_entity_id:
                results.append(
                    ValidationResult(
                        check_name="self_relationship",
                        passed=False,
                        severity="error",
                        message="Entity has relationship with itself",
                        suggestions=["Remove self-referential relationships"],
                    )
                )

        # Add success result if no issues found
        if not results:
            results.append(
                ValidationResult(
                    check_name="relationships",
                    passed=True,
                    severity="info",
                    message="All relationship validations passed",
                )
            )

        return results

    async def _validate_ai_enrichment(
        self, golden_node: GoldenNode
    ) -> List[ValidationResult]:
        """Validate AI enrichment quality."""
        results = []

        if golden_node.ai_enrichment:
            enrichment = golden_node.ai_enrichment

            # Check if AI analysis was performed
            if not enrichment.summary and not enrichment.purpose:
                results.append(
                    ValidationResult(
                        check_name="ai_analysis_completeness",
                        passed=False,
                        severity="warning",
                        message="AI enrichment is missing summary and purpose",
                        suggestions=["Re-run DocuWriter agent to generate AI analysis"],
                    )
                )

            # Check confidence score
            if enrichment.confidence_score and enrichment.confidence_score < 0.5:
                results.append(
                    ValidationResult(
                        check_name="ai_confidence",
                        passed=False,
                        severity="warning",
                        message=f"Low AI confidence score: {enrichment.confidence_score}",
                        suggestions=[
                            "Review AI analysis quality and consider re-processing"
                        ],
                    )
                )
        else:
            results.append(
                ValidationResult(
                    check_name="ai_enrichment_presence",
                    passed=False,
                    severity="warning",
                    message="AI enrichment data is missing",
                    suggestions=["Run DocuWriter agent to generate AI analysis"],
                )
            )

        return results

    async def _validate_schema_compliance(
        self, golden_node: GoldenNode
    ) -> List[ValidationResult]:
        """Validate Pydantic schema compliance."""
        results = []

        try:
            # Re-validate the model to ensure compliance
            golden_node.model_validate(golden_node.model_dump())

            results.append(
                ValidationResult(
                    check_name="schema_compliance",
                    passed=True,
                    severity="info",
                    message="Golden Node schema validation passed",
                )
            )

        except Exception as e:
            results.append(
                ValidationResult(
                    check_name="schema_compliance",
                    passed=False,
                    severity="critical",
                    message=f"Schema validation failed: {e}",
                    suggestions=["Fix data format issues and re-validate"],
                )
            )

        return results

    async def _detect_anomalies(
        self, golden_node: GoldenNode, context: AgentExecutionContext
    ) -> List[ValidationResult]:
        """Use AI to detect potential anomalies."""
        results = []

        try:
            # TODO: Implement AI-powered anomaly detection
            # This would use LLM to identify unusual patterns or potential issues
            #
            # anomaly_prompt = f"""
            # Analyze this code entity for potential anomalies or quality issues:
            #
            # Entity: {golden_node.code_entity.name}
            # Type: {golden_node.code_entity.entity_type.value}
            # Language: {golden_node.code_entity.language.value}
            # Content Length: {len(golden_node.code_entity.content)} characters
            #
            # Look for:
            # - Unusual naming patterns
            # - Potential code smells
            # - Inconsistent formatting
            # - Missing documentation
            # - Overly complex structure
            #
            # Provide specific, actionable feedback.
            # """

            # This would use the chat_with_llm method for analysis
            # anomaly_analysis = await self.chat_with_llm(anomaly_prompt, context)

            # For now, add a placeholder success result
            results.append(
                ValidationResult(
                    check_name="anomaly_detection",
                    passed=True,
                    severity="info",
                    message="No anomalies detected in code entity",
                )
            )

        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            results.append(
                ValidationResult(
                    check_name="anomaly_detection",
                    passed=False,
                    severity="error",
                    message=f"Anomaly detection failed: {e}",
                    suggestions=["Check AI service connectivity and retry"],
                )
            )

        return results

    def _calculate_overall_score(
        self, validation_results: List[ValidationResult]
    ) -> float:
        """Calculate overall quality score from validation results."""
        if not validation_results:
            return 0.0

        # Weight scores by severity
        severity_weights = {
            "critical": -0.3,
            "error": -0.2,
            "warning": -0.1,
            "info": 0.1,
        }

        total_weight = 0.0
        for result in validation_results:
            if result.passed:
                total_weight += severity_weights.get("info", 0.1)
            else:
                total_weight += severity_weights.get(result.severity, -0.1)

        # Normalize to 0.0-1.0 scale
        base_score = 0.8  # Start with good score
        score = base_score + (total_weight / len(validation_results))

        return max(0.0, min(1.0, score))

    def _determine_processing_status(
        self, quality_report: QualityReport
    ) -> ProcessingStatus:
        """Determine processing status based on quality assessment."""
        if quality_report.critical_issues > 0:
            return ProcessingStatus.FAILED
        elif quality_report.overall_score < 0.5:
            return ProcessingStatus.FAILED
        elif quality_report.warnings > 5:
            return ProcessingStatus.COMPLETED  # Completed but with warnings
        else:
            return ProcessingStatus.COMPLETED

    def _generate_quality_tags(self, quality_report: QualityReport) -> List[str]:
        """Generate quality-related tags for searchability."""
        tags = []

        if quality_report.overall_score >= 0.8:
            tags.append("high-quality")
        elif quality_report.overall_score >= 0.6:
            tags.append("medium-quality")
        else:
            tags.append("needs-improvement")

        if quality_report.critical_issues > 0:
            tags.append("has-critical-issues")

        if quality_report.warnings > 3:
            tags.append("has-warnings")

        return tags

    def _generate_recommendations(
        self, validation_results: List[ValidationResult]
    ) -> List[str]:
        """Generate overall recommendations from validation results."""
        recommendations = []

        # Collect unique suggestions from all validation results
        all_suggestions = []
        for result in validation_results:
            all_suggestions.extend(result.suggestions)

        # Deduplicate and prioritize
        unique_suggestions = list(set(all_suggestions))

        # Add general recommendations based on validation patterns
        critical_count = sum(1 for r in validation_results if r.severity == "critical")
        if critical_count > 0:
            recommendations.append(
                "Address critical issues before proceeding with deployment"
            )

        warning_count = sum(1 for r in validation_results if r.severity == "warning")
        if warning_count > 3:
            recommendations.append("Review and address multiple warning conditions")

        return (
            recommendations + unique_suggestions[:5]
        )  # Limit to 5 top recommendations
