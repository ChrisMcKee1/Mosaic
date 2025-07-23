# AI Orchestration Error Handling Implementation

## Overview

This document describes the comprehensive error handling system implemented for AI orchestration in the Mosaic Ingestion Service. The implementation follows Semantic Kernel best practices and integrates with Azure Application Insights for production monitoring.

## Architecture Components

### 1. AIOrchestrationErrorHandler (Function Filter)

**Location**: `src/ingestion_service/plugins/ai_error_handler.py`

The core error handling component implementing Semantic Kernel's `IFunctionFilter` pattern:

- **Automatic Retry Logic**: Exponential backoff with configurable parameters
- **Circuit Breaker Pattern**: Prevents cascading failures in AI services
- **OpenTelemetry Integration**: Full distributed tracing with Azure monitoring
- **Structured Logging**: Comprehensive error context and telemetry

```python
class AIOrchestrationErrorHandler(FunctionFilter):
    async def on_function_invocation(self, context: FunctionInvocationContext):
        # Comprehensive error handling with telemetry
```

### 2. Circuit Breaker Implementation

Implements resilience patterns for AI service failures:

- **States**: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
- **Configurable Thresholds**: Failure count and recovery timeouts
- **Automatic Recovery**: Self-healing behavior after timeout periods

### 3. OpenTelemetry Integration

**Azure Application Insights Alignment**:
- Configured to use connection string from `APPLICATIONINSIGHTS_CONNECTION_STRING`
- Structured span attributes for AI function monitoring
- Resource identification for service tracking
- Batch export for production efficiency

```python
def configure_azure_telemetry(connection_string: Optional[str]) -> trace.Tracer:
    resource = Resource.create({
        "service.name": "mosaic-ingestion-service",
        "service.version": "1.0.0",
        "service.namespace": "ai.orchestration"
    })
```

### 4. Enhanced CodeAnalysisPlugin

**Location**: `src/ingestion_service/plugins/code_analysis_plugin.py`

Updated with comprehensive error handling:

- **Integrated Error Handler**: Automatic retry and circuit breaker protection
- **Azure Monitoring**: OpenTelemetry tracing for all AI function calls  
- **Metrics Collection**: Performance and reliability monitoring
- **Structured Outputs**: Pydantic models with error-safe parsing

## Integration Points

### 1. Kernel Configuration

The error handler is added as a filter to the Semantic Kernel:

```python
# In IngestionPlugin._initialize_semantic_kernel()
self.kernel.add_filter(self.error_handler)
```

### 2. Azure Infrastructure Alignment

The implementation aligns with existing Azure Bicep infrastructure:

**From `infra/resources.bicep`**:
- Log Analytics Workspace: `logAnalytics` (lines 24-34)
- Application Insights: `applicationInsights` (lines 37-46)
- Connection String: Available in container environment

### 3. Environment Variables

Required environment variables for full functionality:

```bash
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=your-key;...
```

## Error Handling Capabilities

### 1. Function-Level Error Handling

Every AI function call is wrapped with:
- **Exception Catching**: All AI service exceptions
- **Retry Logic**: Up to 3 attempts with exponential backoff
- **Circuit Breaker**: Automatic protection against failing services
- **Telemetry**: Full span tracing with error attributes

### 2. Telemetry Attributes

Comprehensive telemetry data for Azure monitoring:

```python
span.set_attributes({
    "ai.function.name": function_name,
    "ai.function.plugin": context.function.plugin_name,
    "ai.service.type": "semantic_kernel",
    "ai.model.provider": "azure_openai",
    "ai.error.type": type(e).__name__,
    "ai.circuit_breaker.state": self.circuit_breaker.state
})
```

### 3. Metrics Collection

The `AIFunctionMetrics` class tracks:
- **Call Counts**: Total function invocations
- **Error Rates**: Failure percentages by function
- **Latency Distribution**: Performance monitoring
- **P95 Latencies**: Performance percentiles

## Configuration Options

### Error Handler Configuration

```python
AIOrchestrationErrorHandler(
    connection_string=None,          # Azure Application Insights connection
    max_retries=3,                   # Maximum retry attempts
    initial_delay=1.0,               # Initial retry delay (seconds)
    max_delay=30.0,                  # Maximum retry delay (seconds)
    backoff_multiplier=2.0           # Exponential backoff multiplier
)
```

### Circuit Breaker Configuration

```python
CircuitBreaker(
    failure_threshold=5,             # Failures before opening circuit
    recovery_timeout=60              # Seconds before attempting recovery
)
```

## Monitoring and Observability

### 1. Azure Application Insights Integration

- **Distributed Tracing**: Full request tracing across AI function calls
- **Custom Metrics**: AI function performance and reliability metrics
- **Error Tracking**: Comprehensive exception monitoring
- **Performance Monitoring**: Latency and throughput metrics

### 2. Structured Logging

Enhanced logging with error context:

```python
error_context = {
    "function_name": function_name,
    "plugin_name": context.function.plugin_name,
    "error_type": type(e).__name__,
    "circuit_breaker_state": self.circuit_breaker.state,
    "failure_count": self.circuit_breaker.failure_count
}
```

### 3. Dashboard Queries

Example KQL queries for Azure Application Insights:

```kusto
// AI Function Error Rate
traces
| where customDimensions.["ai.service.type"] == "semantic_kernel"
| summarize 
    total_calls = count(),
    errors = countif(severityLevel >= 3)
    by tostring(customDimensions.["ai.function.name"])
| extend error_rate = errors * 100.0 / total_calls

// Circuit Breaker Status
traces  
| where customDimensions.["ai.circuit_breaker.state"] != "CLOSED"
| summarize by 
    tostring(customDimensions.["ai.function.name"]),
    tostring(customDimensions.["ai.circuit_breaker.state"])
```

## Dependencies

### Required Python Packages

```txt
# OpenTelemetry for Azure monitoring integration
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0
opentelemetry-exporter-azure-monitor==1.0.0b21
opentelemetry-instrumentation==0.42b0
```

### Azure Resources

- **Application Insights**: For telemetry collection
- **Log Analytics Workspace**: Backend for Application Insights
- **Container Apps Environment**: Provides connection string via environment

## Testing

### Unit Tests

**Location**: `test_error_handling_integration.py`

Comprehensive test coverage:
- Circuit breaker state transitions
- Retry logic with exponential backoff
- OpenTelemetry span creation and attributes
- Metrics collection and calculation
- Error handling integration

### Test Categories

1. **Circuit Breaker Tests**: State transitions and recovery
2. **Error Handler Tests**: Retry logic and failure handling
3. **Metrics Tests**: Performance monitoring and statistics
4. **Integration Tests**: Full pipeline with error simulation

## Production Readiness

### Performance Characteristics

- **Low Overhead**: Minimal performance impact on AI function calls
- **Async Operations**: Non-blocking error handling and telemetry
- **Memory Efficient**: Bounded circuit breaker state and metrics storage
- **Scalable**: Handles high-volume AI function invocations

### Reliability Features

- **Graceful Degradation**: Circuit breaker prevents cascading failures
- **Self-Healing**: Automatic recovery after service restoration
- **Comprehensive Monitoring**: Full observability into AI operations
- **Error Isolation**: Failed functions don't impact system stability

### Security Considerations

- **No Sensitive Data**: Error logs exclude function arguments and results
- **Azure Integration**: Uses managed identity for Application Insights access
- **Secure Defaults**: Conservative retry and timeout configurations

## Usage Examples

### Basic Integration

```python
# Initialize error handler
error_handler = AIOrchestrationErrorHandler(
    connection_string=os.getenv('APPLICATIONINSIGHTS_CONNECTION_STRING')
)

# Add to kernel
kernel.add_filter(error_handler)
```

### Custom Configuration

```python
# High-throughput configuration
error_handler = AIOrchestrationErrorHandler(
    max_retries=2,                   # Fewer retries for speed
    initial_delay=0.5,               # Faster initial retry
    max_delay=10.0,                  # Lower maximum delay
    backoff_multiplier=1.5           # Gentler backoff curve
)
```

### Monitoring Integration

```python
# Access metrics for monitoring
stats = error_handler.metrics.get_function_stats()
for function_name, metrics in stats.items():
    print(f"{function_name}: {metrics['error_rate']:.2%} error rate")
```

## Best Practices

1. **Monitor Circuit Breaker State**: Alert on OPEN circuits
2. **Tune Retry Parameters**: Based on AI service SLAs
3. **Review Error Patterns**: Use Application Insights for analysis
4. **Set Appropriate Timeouts**: Balance resilience vs. performance
5. **Test Failure Scenarios**: Validate error handling behavior

## Future Enhancements

1. **Adaptive Timeouts**: Dynamic timeout adjustment based on service performance
2. **Bulkhead Pattern**: Isolate different AI function types
3. **Health Checks**: Proactive service health monitoring
4. **Rate Limiting**: Prevent AI service quota exhaustion
5. **Custom Metrics**: Business-specific AI performance indicators

This comprehensive error handling system ensures the Mosaic Ingestion Service maintains high reliability and observability for AI orchestration operations in production environments.