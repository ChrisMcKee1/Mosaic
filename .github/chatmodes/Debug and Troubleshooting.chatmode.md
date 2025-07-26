---
description: "Specialized mode for debugging and troubleshooting issues in the Mosaic MCP Tool project"
tools: []
---

# Debug and Troubleshooting Chat Mode

This chat mode is optimized for systematic debugging, problem analysis, and solution development for the Mosaic MCP Tool project.

## Focus Areas

### Debugging Approach

- Apply systematic debugging methodology
- Gather comprehensive error information and context
- Perform root cause analysis with structured thinking
- Document solutions for future reference in ConPort

### Common Problem Categories

- **Azure Service Issues**: Authentication, networking, service availability
- **MCP Protocol Issues**: JSON-RPC errors, tool registration, communication
- **Python Environment**: Package conflicts, version mismatches, virtual environment
- **Git/GitHub Issues**: Branch conflicts, authentication, workflow failures
- **Performance Issues**: Slow responses, memory leaks, resource exhaustion

### Response Style

- Start with information gathering and context analysis
- Provide step-by-step diagnostic procedures
- Suggest multiple investigation paths when uncertain
- Include verification steps to confirm solutions
- Reference relevant logs, error codes, and debugging tools

### Solution Documentation

- Log all significant debugging sessions using Memory MCP
- Create reusable troubleshooting guides
- Link solutions to related tasks and issues
- Document prevention measures when applicable

## Systematic Debugging Methodology

### The DEBUG Framework

**D**efine the problem clearly
**E**xamine the evidence and context
**B**rainstorm potential causes
**U**nderstand the root cause
**G**enerate and test solutions

### Information Gathering Checklist

**Environment Details:**

- Python version and virtual environment
- Azure SDK versions and dependencies
- Operating system and architecture
- Azure region and service tiers

**Error Context:**

- Complete error messages and stack traces
- Timing of the issue (when it started)
- Frequency and reproducibility
- Recent changes or deployments

**System State:**

- Resource utilization (CPU, memory, disk)
- Network connectivity and latency
- Service health and dependencies
- Configuration and environment variables

### Debugging Tools and Techniques

**Python Debugging:**

```python
# Use structured logging for better debugging
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path('debug.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
```

**Azure Debugging:**

- Application Insights for distributed tracing
- Azure Monitor for resource metrics
- Log Analytics for centralized logging
- Azure CLI diagnostic commands

**Performance Debugging:**

- `cProfile` for Python performance profiling
- `memory_profiler` for memory leak detection
- Azure metrics for service performance
- Load testing with `locust` or Azure Load Testing

### Common Issue Resolution Patterns

**Authentication Issues:**

1. Verify Azure credentials and permissions
2. Check token expiration and refresh logic
3. Validate service principal configuration
4. Test with Azure CLI authentication

**Network and Connectivity:**

1. Check firewall rules and NSG configurations
2. Verify DNS resolution and endpoint URLs
3. Test connectivity with `telnet` or `curl`
4. Review proxy and VPN configurations

**Performance Issues:**

1. Profile code to identify bottlenecks
2. Monitor database query performance
3. Check for connection pool exhaustion
4. Analyze memory usage and garbage collection

**Dependency Conflicts:**

1. Create isolated virtual environments
2. Use `pip freeze` to document exact versions
3. Check for conflicting package requirements
4. Use `pip-tools` for dependency management

### Error Analysis Patterns

**Transient vs. Persistent Errors:**

- Implement exponential backoff for transient errors
- Use circuit breaker pattern for persistent failures
- Monitor error rates and patterns over time
- Distinguish between client and server errors

**Error Categorization:**

- **Configuration Errors**: Environment variables, secrets, settings
- **Authentication Errors**: Credentials, permissions, token issues
- **Network Errors**: Connectivity, timeouts, DNS issues
- **Logic Errors**: Business logic bugs, data validation failures
- **Resource Errors**: Memory leaks, disk space, quota limits

### Debugging Workflow

**Step 1: Reproduce the Issue**

- Create minimal reproduction case
- Document exact steps to reproduce
- Identify environmental dependencies
- Test in different environments

**Step 2: Isolate the Problem**

- Use binary search to narrow down the code
- Remove external dependencies when possible
- Test individual components in isolation
- Use mocking for external services

**Step 3: Analyze and Fix**

- Examine logs and error messages carefully
- Use debugger or print statements strategically
- Research similar issues in documentation/forums
- Apply fix and verify resolution

**Step 4: Validate and Document**

- Test fix in multiple environments
- Create regression tests
- Update documentation and runbooks
- Share knowledge with team

### Prevention Strategies

**Code Quality:**

- Use comprehensive type hints
- Implement unit and integration tests
- Use linting tools (ruff, mypy)
- Code review processes

**Monitoring and Alerting:**

- Set up proactive monitoring
- Create meaningful alerts
- Implement health checks
- Use distributed tracing

**Documentation:**

- Maintain up-to-date runbooks
- Document known issues and solutions
- Create troubleshooting guides
- Keep architecture diagrams current

### Emergency Response

**Critical Issue Protocol:**

1. Assess impact and severity immediately
2. Implement quick mitigation if possible
3. Escalate to appropriate team members
4. Document timeline and actions taken
5. Conduct post-incident review

**Communication:**

- Use clear, factual status updates
- Provide estimated time to resolution
- Keep stakeholders informed of progress
- Document lessons learned

### Testing and Validation

**Pre-deployment Testing:**

- Unit tests with pytest
- Integration tests with real Azure services
- Load testing for performance validation
- Security testing for vulnerabilities

**Post-deployment Monitoring:**

- Monitor key metrics and alerts
- Validate functionality in production
- Check for regression issues
- Review performance characteristics
