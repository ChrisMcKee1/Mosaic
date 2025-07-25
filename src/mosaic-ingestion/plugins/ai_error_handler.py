"""
AI Orchestration Error Handler with OpenTelemetry Integration

This module provides comprehensive error handling for AI operations using Semantic Kernel's
IFunctionFilter pattern with OpenTelemetry tracing aligned with Azure Application Insights.

Key Features:
- Semantic Kernel function filters for robust AI error handling
- OpenTelemetry integration with Azure Application Insights
- Automatic retry logic with exponential backoff
- Structured error logging and telemetry
- Circuit breaker pattern for resilience
"""

import asyncio
import logging
import time
from typing import Optional, Any, Dict
from opentelemetry import trace
from opentelemetry.exporter.azure.monitor import AzureMonitorTraceExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from semantic_kernel.filters import FunctionFilter, FunctionInvocationContext
from semantic_kernel.exceptions import KernelException


# Configure OpenTelemetry for Azure Application Insights
def configure_azure_telemetry(connection_string: Optional[str] = None) -> trace.Tracer:
    """
    Configure OpenTelemetry tracing for Azure Application Insights integration.

    Args:
        connection_string: Azure Application Insights connection string

    Returns:
        Configured tracer instance
    """
    # Create resource with service identification
    resource = Resource.create(
        {
            "service.name": "mosaic-ingestion-service",
            "service.version": "1.0.0",
            "service.namespace": "ai.orchestration",
        }
    )

    # Configure tracer provider
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer_provider = trace.get_tracer_provider()

    # Configure Azure Monitor exporter if connection string provided
    if connection_string:
        azure_exporter = AzureMonitorTraceExporter(connection_string=connection_string)
        span_processor = BatchSpanProcessor(azure_exporter)
        tracer_provider.add_span_processor(span_processor)

    return trace.get_tracer(__name__)


class CircuitBreaker:
    """Circuit breaker pattern for AI service resilience."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def can_execute(self) -> bool:
        """Check if operation can be executed based on circuit state."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class AIOrchestrationErrorHandler(FunctionFilter):
    """
    Semantic Kernel function filter for comprehensive AI error handling.

    This filter implements:
    - Automatic retry with exponential backoff
    - Circuit breaker pattern for resilience
    - OpenTelemetry tracing with Azure integration
    - Structured error logging and metrics
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_multiplier: float = 2.0,
    ):
        """
        Initialize error handler with Azure monitoring integration.

        Args:
            connection_string: Azure Application Insights connection string
            max_retries: Maximum number of retry attempts
            initial_delay: Initial retry delay in seconds
            max_delay: Maximum retry delay in seconds
            backoff_multiplier: Exponential backoff multiplier
        """
        self.tracer = configure_azure_telemetry(connection_string)
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.circuit_breaker = CircuitBreaker()

        # Configure structured logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    async def on_function_invocation(self, context: FunctionInvocationContext) -> None:
        """
        Handle function invocation with comprehensive error handling.

        This method implements the core error handling logic using Semantic Kernel's
        filter pattern with OpenTelemetry tracing and Azure monitoring integration.
        """
        function_name = context.function.name

        # Create OpenTelemetry span for function execution
        with self.tracer.start_as_current_span(f"ai_function_{function_name}") as span:
            # Set span attributes for Azure monitoring
            span.set_attributes(
                {
                    "ai.function.name": function_name,
                    "ai.function.plugin": context.function.plugin_name or "unknown",
                    "ai.service.type": "semantic_kernel",
                    "ai.model.provider": "azure_openai",
                }
            )

            try:
                # Check circuit breaker before execution
                if not self.circuit_breaker.can_execute():
                    error_msg = f"Circuit breaker OPEN for function {function_name}"
                    self.logger.warning(error_msg)
                    span.set_status(Status(StatusCode.ERROR, error_msg))
                    span.set_attribute("ai.error.type", "circuit_breaker_open")
                    raise KernelException(error_msg)

                # Execute with retry logic
                await self._execute_with_retry(context, span)

                # Record success for circuit breaker
                self.circuit_breaker.record_success()
                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                # Record failure for circuit breaker
                self.circuit_breaker.record_failure()

                # Enhanced error logging with context
                error_context = {
                    "function_name": function_name,
                    "plugin_name": context.function.plugin_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "circuit_breaker_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                }

                self.logger.error(
                    f"AI function {function_name} failed: {str(e)}",
                    extra={"error_context": error_context},
                )

                # Set span error status and attributes
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attributes(
                    {
                        "ai.error.type": type(e).__name__,
                        "ai.error.message": str(e),
                        "ai.circuit_breaker.state": self.circuit_breaker.state,
                        "ai.failure.count": self.circuit_breaker.failure_count,
                    }
                )

                # Re-raise the exception after logging
                raise

    async def _execute_with_retry(
        self, context: FunctionInvocationContext, span: trace.Span
    ) -> Any:
        """
        Execute function with exponential backoff retry logic.

        Args:
            context: Function invocation context
            span: OpenTelemetry span for tracing

        Returns:
            Function execution result

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        delay = self.initial_delay

        for attempt in range(self.max_retries + 1):
            try:
                # Set retry attempt attributes
                span.set_attributes(
                    {
                        "ai.retry.attempt": attempt,
                        "ai.retry.max_attempts": self.max_retries,
                    }
                )

                # Execute the original function
                result = await context.function.invoke(
                    context.kernel, context.arguments, context.result
                )

                if attempt > 0:
                    self.logger.info(
                        f"Function {context.function.name} succeeded on attempt {attempt + 1}"
                    )
                    span.set_attribute("ai.retry.success_attempt", attempt)

                return result

            except Exception as e:
                last_exception = e

                # Log retry attempt
                if attempt < self.max_retries:
                    self.logger.warning(
                        f"Function {context.function.name} failed on attempt {attempt + 1}: {str(e)}. "
                        f"Retrying in {delay} seconds..."
                    )

                    span.add_event(
                        name="ai.retry.attempt_failed",
                        attributes={
                            "ai.retry.attempt": attempt,
                            "ai.error.message": str(e),
                            "ai.retry.delay": delay,
                        },
                    )

                    # Wait with exponential backoff
                    await asyncio.sleep(delay)
                    delay = min(delay * self.backoff_multiplier, self.max_delay)
                else:
                    # Final attempt failed
                    span.set_attribute("ai.retry.final_failure", True)
                    break

        # All attempts failed
        raise last_exception


class AIFunctionMetrics:
    """Metrics collector for AI function performance monitoring."""

    def __init__(self, tracer: trace.Tracer):
        self.tracer = tracer
        self.function_call_counts = {}
        self.function_error_counts = {}
        self.function_latencies = {}

    def record_function_call(self, function_name: str, duration: float, success: bool):
        """Record function call metrics."""
        # Update call counts
        self.function_call_counts[function_name] = (
            self.function_call_counts.get(function_name, 0) + 1
        )

        # Update error counts
        if not success:
            self.function_error_counts[function_name] = (
                self.function_error_counts.get(function_name, 0) + 1
            )

        # Update latency tracking
        if function_name not in self.function_latencies:
            self.function_latencies[function_name] = []
        self.function_latencies[function_name].append(duration)

        # Create telemetry span for metrics
        with self.tracer.start_as_current_span("ai_function_metrics") as span:
            span.set_attributes(
                {
                    "ai.metrics.function_name": function_name,
                    "ai.metrics.duration_ms": duration * 1000,
                    "ai.metrics.success": success,
                    "ai.metrics.total_calls": self.function_call_counts[function_name],
                    "ai.metrics.error_rate": self._calculate_error_rate(function_name),
                }
            )

    def _calculate_error_rate(self, function_name: str) -> float:
        """Calculate error rate for a function."""
        total_calls = self.function_call_counts.get(function_name, 0)
        error_calls = self.function_error_counts.get(function_name, 0)

        if total_calls == 0:
            return 0.0

        return error_calls / total_calls

    def get_function_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive function statistics."""
        stats = {}

        for function_name in self.function_call_counts.keys():
            latencies = self.function_latencies.get(function_name, [])
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            stats[function_name] = {
                "total_calls": self.function_call_counts[function_name],
                "error_count": self.function_error_counts.get(function_name, 0),
                "error_rate": self._calculate_error_rate(function_name),
                "avg_latency_ms": avg_latency * 1000,
                "p95_latency_ms": self._calculate_percentile(latencies, 95) * 1000
                if latencies
                else 0,
            }

        return stats

    def _calculate_percentile(self, values: list, percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
